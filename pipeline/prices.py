"""Prices: Alpaca Market Data (free Basic plan → full-market SIP data delayed 15 min) with an unofficial
Yahoo Finance fallback. Daily history is cached in .cache/prices.json.gz.

Ticker conventions: we store class shares with a dot (BRK.B). Alpaca uses the dot as well; Yahoo uses a dash.
"""
from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

from .http import HttpClient, HttpError
from .util import ET, UTC, chunks, iso, log, now_et, now_utc, read_json, write_json

ALPACA = "https://data.alpaca.markets/v2/stocks"
MASSIVE = "https://api.massive.com"   # formerly Polygon.io; api.polygon.io serves the same API
BENCH = "SPY"


def _ord(d: dt.date) -> int:
    return d.toordinal()


class PriceStore:
    """{ticker: {"dates":[ordinal...], "c":[close...], "v":[volume...], "src": str, "upd": iso}}"""

    def __init__(self, cache_dir: Path):
        self.path = cache_dir / "prices.json.gz"
        self.data: dict[str, dict] = read_json(self.path, {}) or {}
        self.latest: dict[str, dict] = {}   # ticker -> {"price", "asof", "src"}
        self.dirty = False

    def series(self, t: str) -> dict | None:
        return self.data.get(t)

    def has_recent(self, t: str, max_age_days: int = 3) -> bool:
        s = self.data.get(t)
        if not s or not s.get("dates"):
            return False
        last = dt.date.fromordinal(s["dates"][-1])
        return (now_utc().date() - last).days <= max_age_days

    def put(self, t: str, rows: list[tuple[dt.date, float, float]], src: str) -> None:
        if not rows:
            return
        cur = self.data.get(t) or {"dates": [], "c": [], "v": []}
        new = {_ord(d): (round(float(c), 4), float(v or 0)) for d, c, v in rows if c is not None and c > 0}
        factor = _rebase_factor(cur, new)
        merged = {}
        first_new = min(new) if new else None
        for d, c, v in zip(cur["dates"], cur["c"], cur["v"]):
            if factor != 1.0 and first_new is not None and d < first_new:   # older rows the new download didn't cover
                c, v = round(c * factor, 4), (v / factor if v else v)
            merged[d] = (c, v)
        merged.update(new)
        keys = sorted(merged)[-800:]  # ~3 years of trading days
        self.data[t] = {"dates": keys, "c": [merged[k][0] for k in keys], "v": [merged[k][1] for k in keys],
                        "src": src, "upd": iso(now_utc())}
        self.dirty = True

    def close_on_or_after(self, t: str, d: dt.date) -> tuple[dt.date, float] | None:
        s = self.data.get(t)
        if not s:
            return None
        o = _ord(d)
        import bisect
        i = bisect.bisect_left(s["dates"], o)
        if i >= len(s["dates"]):
            return None
        return dt.date.fromordinal(s["dates"][i]), s["c"][i]

    def index_on_or_after(self, t: str, d: dt.date) -> int | None:
        s = self.data.get(t)
        if not s:
            return None
        import bisect
        i = bisect.bisect_left(s["dates"], _ord(d))
        return i if i < len(s["dates"]) else None

    def prune(self, keep: set[str]) -> None:
        for t in [t for t in self.data if t not in keep and t != BENCH]:
            upd = self.data[t].get("upd") or ""
            if upd < iso(now_utc() - dt.timedelta(days=45)):
                self.data.pop(t, None)
                self.dirty = True

    def save(self) -> None:
        if self.dirty:
            write_json(self.path, self.data)
            self.dirty = False


SPLITS_FILE = "splits.json.gz"


def load_splits(cache_dir: Path) -> dict[str, list]:
    """{ticker: [[execution date, share factor]]}: 2-for-1 → 2.0, 1-for-10 → 0.1. From Massive's splits feed
    or Yahoo's split events. Used to re-base share counts reported before a split."""
    d = read_json(cache_dir / SPLITS_FILE) or {}
    return d.get("by_ticker") or {}


def record_splits(cache_dir: Path, items: list[tuple[str, str, float]]) -> int:
    """Merge (ticker, date, share factor) items into the registry (kept ~3 years)."""
    if not items:
        return 0
    reg = load_splits(cache_dir)
    cut = (now_utc().date() - dt.timedelta(days=1100)).isoformat()
    n = 0
    for t, d, f in items:
        if not t or not d or not f or f <= 0 or not math.isfinite(f):
            continue
        lst = reg.setdefault(t, [])
        if not any(x[0] == d[:10] for x in lst):
            lst.append([d[:10], round(float(f), 8)])
            n += 1
    for t in list(reg):
        reg[t] = sorted(x for x in reg[t] if x[0] >= cut)
        if not reg[t]:
            reg.pop(t)
    write_json(cache_dir / SPLITS_FILE, {"by_ticker": reg, "updated": iso(now_utc())})
    return n


def _rebase_factor(cur: dict, new: dict) -> float:
    """Price providers re-adjust history after a split. Compare the oldest overlapping days: a steady ratio far
    from 1 means the stored rows predate the adjustment, so older stored rows must be scaled by it too."""
    if not cur.get("dates") or not new:
        return 1.0
    ratios = [new[d][0] / c for d, c in zip(cur["dates"], cur["c"]) if d in new and c][:5]
    if len(ratios) < 3:
        return 1.0
    lo, hi = min(ratios), max(ratios)
    med = sorted(ratios)[len(ratios) // 2]
    if abs(med - 1) > 0.02 and hi / lo < 1.02:
        return med
    return 1.0


class MarketMatrix:
    """Whole-market daily closes from Massive's free 'grouped daily' endpoint (one request = every US stock
    for one day). Stored compactly as numpy arrays in the heavy cache; ~2 years (the free plan's limit)."""

    def __init__(self, path: Path):
        import numpy as np
        self.np = np
        self.path = path
        self.dates: list[int] = []
        self.tickers: list[str] = []
        self.col: dict[str, int] = {}
        self.close = np.zeros((0, 0), dtype=np.float32)
        self.vol = np.zeros((0, 0), dtype=np.float32)
        self.checked: set[int] = set()   # days already requested (incl. holidays with no data)
        self.fetched: list[int] = []      # per row: the day we downloaded it (adjusted=true is as-of download)
        self.applied: set[str] = set()    # split ids already applied
        self.dirty = False
        if path.exists():
            try:
                z = np.load(path, allow_pickle=False)
                self.dates = [int(x) for x in z["dates"]]
                self.tickers = [str(x) for x in z["tickers"]]
                self.close, self.vol = z["close"], z["vol"]
                self.checked = {int(x) for x in z["checked"]}
                self.col = {t: i for i, t in enumerate(self.tickers)}
                today = now_et().date().toordinal()
                self.fetched = [int(x) for x in z["fetched"]] if "fetched" in z.files else [today] * len(self.dates)
                self.applied = {str(x) for x in z["applied"]} if "applied" in z.files else set()
            except Exception as e:  # noqa: BLE001 – a corrupt cache just means a rebuild
                log.warning("market matrix unreadable (%s); rebuilding", e)

    def add_day(self, day: dt.date, rows: list[tuple[str, float, float]]) -> None:
        """Invariant: close/vol always have shape (len(dates), len(tickers))."""
        import bisect
        np = self.np
        o = day.toordinal()
        self.checked.add(o)
        self.dirty = True
        if not rows:
            return
        new = [t for t in dict.fromkeys(t for t, _, _ in rows) if t not in self.col]
        if new:
            for t in new:
                self.col[t] = len(self.tickers)
                self.tickers.append(t)
            pad = np.full((len(self.dates), len(new)), np.nan, dtype=np.float32)
            self.close = np.hstack([self.close, pad])
            self.vol = np.hstack([self.vol, pad.copy()])
        today = now_et().date().toordinal()
        if o in self.dates:
            r = self.dates.index(o)
            self.fetched[r] = today
        else:
            r = bisect.bisect_left(self.dates, o)
            self.dates.insert(r, o)
            self.fetched.insert(r, today)
            blank = np.full((1, len(self.tickers)), np.nan, dtype=np.float32)
            self.close = np.vstack([self.close[:r], blank, self.close[r:]])
            self.vol = np.vstack([self.vol[:r], blank.copy(), self.vol[r:]])
        for t, c, v in rows:
            j = self.col[t]
            self.close[r, j] = c
            self.vol[r, j] = v or 0

    def apply_splits(self, splits: list[dict]) -> list[str]:
        """Bring rows downloaded before a split onto the post-split basis (Massive's adjusted=true is as of the
        download, so a day stored last month doesn't know about this week's split). Returns adjusted tickers."""
        np = self.np
        out = []
        dates = np.asarray(self.dates, dtype=np.int64)
        fetched = np.asarray(self.fetched, dtype=np.int64)
        for s in sorted(splits, key=lambda s: s.get("execution_date") or ""):
            sid = str(s.get("id") or f"{s.get('ticker')}:{s.get('execution_date')}")
            j = self.col.get(s.get("ticker") or "")
            frm, to = to_num(s.get("split_from")), to_num(s.get("split_to"))
            ex = s.get("execution_date")
            if sid in self.applied:
                continue
            if j is None or not frm or not to or not ex:
                self.applied.add(sid)
                continue
            e = dt.date.fromisoformat(ex[:10]).toordinal()
            factor = frm / to                       # 2-for-1 → old prices × 0.5
            have = ~np.isnan(self.close[:, j])
            pre = np.flatnonzero(have & (dates < e))
            post = np.flatnonzero(have & (dates >= e))
            if not len(pre):
                self.applied.add(sid)
                continue
            if not len(post):
                continue                            # wait until a post-split day is stored to verify against
            same_day_unadjusted = False
            if abs(np.log(factor)) >= 0.15 and fetched[pre[-1]] == e:
                rho = float(self.close[post[0], j] / self.close[pre[-1], j])
                same_day_unadjusted = abs(np.log(rho) - np.log(factor)) < abs(np.log(rho))
            rows = pre[(fetched[pre] < e) | ((fetched[pre] == e) & same_day_unadjusted)]
            if len(rows):
                self.close[rows, j] = self.close[rows, j] * factor
                self.vol[rows, j] = self.vol[rows, j] / factor
                out.append(s["ticker"])
            self.applied.add(sid)
            self.dirty = True
        return out

    def missing_days(self, max_days: int = 500) -> list[dt.date]:
        today = now_et().date()
        out, d = [], today - dt.timedelta(days=1)   # the free plan serves past days only
        while len(out) < max_days and d > today - dt.timedelta(days=int(max_days * 1.5)):
            if d.weekday() < 5 and d.toordinal() not in self.checked:
                out.append(d)
            d -= dt.timedelta(days=1)
        return out

    def series(self, t: str) -> dict | None:
        j = self.col.get(t)
        if j is None:
            return None
        np = self.np
        c = self.close[:, j]
        mask = ~np.isnan(c)
        if not mask.any():
            return None
        return {"dates": [d for d, m in zip(self.dates, mask) if m], "c": [round(float(x), 4) for x in c[mask]],
                "v": [float(x) for x in self.vol[:, j][mask]], "src": "massive", "upd": iso(now_utc())}

    def export(self, tickers: list[str], store: "PriceStore") -> int:
        n = 0
        for t in tickers:
            s = self.series(t)
            if s:
                store.data[t] = s
                store.dirty = True
                n += 1
        return n

    def save(self) -> None:
        if not self.dirty:
            return
        np = self.np
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, dates=np.asarray(self.dates, dtype=np.int32), tickers=np.asarray(self.tickers),
                            close=self.close, vol=self.vol, checked=np.asarray(sorted(self.checked), dtype=np.int32),
                            fetched=np.asarray(self.fetched, dtype=np.int32),
                            applied=np.asarray(sorted(self.applied), dtype=str))
        import os
        os.replace(tmp, self.path)
        self.dirty = False


def massive_grouped(http: HttpClient, key: str, day: dt.date) -> list[tuple[str, float, float]]:
    d = http.get_json(f"{MASSIVE}/v2/aggs/grouped/locale/us/market/stocks/{day.isoformat()}?adjusted=true&apiKey={key}")
    out = []
    for r in d.get("results") or []:
        t, c = r.get("T"), r.get("c")
        if t and c and _plausible(t):
            out.append((t, float(c), float(r.get("v") or 0)))
    return out


def massive_splits(http: HttpClient, key: str, since: dt.date, max_pages: int = 5) -> list[dict]:
    """Stock splits (and stock dividends) executed on/after `since`. Free plan: 2 years of history."""
    url = (f"{MASSIVE}/stocks/v1/splits?execution_date.gte={since.isoformat()}"
           f"&sort=execution_date.asc&limit=5000&apiKey={key}")
    out: list[dict] = []
    for _ in range(max_pages):
        d = http.get_json(url)
        out.extend(d.get("results") or [])
        nxt = d.get("next_url")
        if not nxt:
            break
        url = nxt + ("&" if "?" in nxt else "?") + f"apiKey={key}"
    return out


def backfill_market(http: HttpClient, key: str, matrix: MarketMatrix, budget_seconds: float) -> int:
    """Fetch missing days newest-first within the time budget (free plan: 5 requests/minute)."""
    import time
    t0 = time.monotonic()
    n = 0
    for day in matrix.missing_days():
        if time.monotonic() - t0 > budget_seconds:
            break
        try:
            matrix.add_day(day, massive_grouped(http, key, day))
            n += 1
        except HttpError as e:
            old = (now_et().date() - day).days > 700
            if e.status in (403, 404) and old:   # outside the free plan's ~2-year window → stop going further back
                matrix.checked.add(day.toordinal())
                matrix.dirty = True
                if e.status == 403:
                    break
                continue
            if e.status in (401, 403):           # a recent day refused: the key is wrong or lapsed. Say so loudly.
                raise RuntimeError(f"Massive refused {day} (HTTP {e.status}): check the MASSIVE_API_KEY secret "
                                   "and plan") from None
            log.warning("massive %s: %s", day, e)
            break
    return n


class PriceClient:
    def __init__(self, http: HttpClient, store: PriceStore, key_id: str | None, secret: str | None,
                 matrix: MarketMatrix | None = None):
        self.http = http
        self.store = store
        self.alpaca = bool(key_id and secret)
        self.headers = {"APCA-API-KEY-ID": key_id or "", "APCA-API-SECRET-KEY": secret or ""}
        self.errors: list[str] = []
        self.split_events: list[tuple[str, str, float]] = []    # (ticker, date, share factor) seen on Yahoo
        self.matrix = matrix if (matrix is not None and matrix.dates) else None
        hist = "Massive EOD (whole market)" if self.matrix else "Yahoo (unofficial)"
        live = "Alpaca IEX real-time (indicative)" if self.alpaca else "Yahoo (unofficial, delayed)"
        self.source = f"history: {hist}; latest: {live}"

    # ------------------------------------------------------------ daily history
    def refresh_daily(self, tickers: list[str], days: int = 400, max_tickers: int = 4000) -> int:
        """Ensure ~`days` calendar days of daily closes for each ticker (incremental if cached)."""
        tickers = [t for t in dict.fromkeys(tickers) if t and _plausible(t)][:max_tickers]
        if BENCH not in tickers:
            tickers.insert(0, BENCH)
        if self.matrix is not None:
            return self.matrix.export(tickers, self.store)
        tickers = tickers[:150]  # unofficial per-ticker fallback: keep it polite
        end = now_utc() - dt.timedelta(minutes=16)
        need_full, need_inc = [], []
        for t in tickers:
            s = self.store.series(t)
            if not s or not s["dates"] or dt.date.fromordinal(s["dates"][0]) > (end.date() - dt.timedelta(days=days - 10)):
                need_full.append(t)
            elif not self.store.has_recent(t, 0):
                need_inc.append(t)
        n = 0
        for t in need_full + need_inc:
            n += self._yahoo_daily(t, "2y" if days > 370 else "1y")
        return n

    def _alpaca_daily(self, tickers: list[str], start: dt.datetime, end: dt.datetime) -> int:
        got = 0
        for batch in chunks(tickers, 100):
            token = None
            rows: dict[str, list] = {}
            for _ in range(60):
                url = (f"{ALPACA}/bars?symbols={','.join(batch)}&timeframe=1Day&start={iso(start)}&end={iso(end)}"
                       f"&adjustment=split&feed=sip&limit=10000")
                if token:
                    url += f"&page_token={token}"
                try:
                    d = self.http.get_json(url, headers=self.headers)
                except HttpError as e:
                    self.errors.append(f"alpaca bars: {e}")
                    log.warning("alpaca bars failed: %s", e)
                    break
                for sym, bars in (d.get("bars") or {}).items():
                    for b in bars:
                        day = dt.datetime.fromisoformat(b["t"].replace("Z", "+00:00")).astimezone(UTC).date()
                        rows.setdefault(sym, []).append((day, b.get("c"), b.get("v")))
                token = d.get("next_page_token")
                if not token:
                    break
            for sym, r in rows.items():
                self.store.put(sym, r, "alpaca")
                got += 1
        return got

    def _yahoo_daily(self, t: str, rng: str) -> int:
        sym = t.replace(".", "-")
        try:
            d = self.http.get_json(f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?range={rng}&interval=1d"
                                   f"&includePrePost=false&events=div%2Csplit")
        except HttpError as e:
            self.errors.append(f"yahoo {t}: {e.status}")
            return 0
        res = ((d.get("chart") or {}).get("result") or [None])[0]
        if not res:
            return 0
        ts = res.get("timestamp") or []
        q = ((res.get("indicators") or {}).get("quote") or [{}])[0]
        # "close" is split-adjusted only, the same basis as Massive's history (adjclose also removes dividends,
        # which would mix two bases in one stored series)
        closes = q.get("close") or []
        vols = q.get("volume") or []
        from .forecast import last_complete_session
        done = last_complete_session(now_et())
        rows = []
        for i, t0 in enumerate(ts):
            c = closes[i] if i < len(closes) else None
            day = dt.datetime.fromtimestamp(t0, ET).date()
            if c is None or day > done:                # today's bar is still moving until the close
                continue
            rows.append((day, c, vols[i] if i < len(vols) else 0))
        for ev in ((res.get("events") or {}).get("splits") or {}).values():
            try:
                num, den = float(ev.get("numerator") or 0), float(ev.get("denominator") or 0)
                if num > 0 and den > 0:
                    self.split_events.append((t, dt.datetime.fromtimestamp(int(ev["date"]), ET).date().isoformat(),
                                              num / den))
            except (TypeError, ValueError, KeyError):
                continue
        self.store.put(t, rows, "yahoo")
        meta = res.get("meta") or {}
        if meta.get("regularMarketPrice"):
            self.store.latest[t] = {"price": meta["regularMarketPrice"],
                                    "asof": iso(dt.datetime.fromtimestamp(meta.get("regularMarketTime") or 0, UTC)),
                                    "src": "yahoo"}
        return 1

    # ------------------------------------------------------------ latest (delayed) price
    def refresh_latest(self, tickers: list[str], max_yahoo: int = 40) -> int:
        """Latest price. Alpaca's free feed is IEX-only (~2-3% of volume): fine as an indicative real-time
        price, labelled as such. Without Alpaca, a rotating slice of tickers is refreshed via Yahoo."""
        tickers = [t for t in dict.fromkeys(tickers) if t and _plausible(t)]
        if not tickers:
            return 0
        n = 0
        if self.alpaca:
            for batch in chunks(tickers, 100):
                try:
                    d = self.http.get_json(f"{ALPACA}/snapshots?symbols={','.join(batch)}&feed=iex", headers=self.headers)
                except HttpError as e:
                    self.errors.append(f"alpaca snapshots: {e.status}")
                    break
                snaps = d.get("snapshots", d) if isinstance(d, dict) else {}
                for sym, sn in (snaps or {}).items():
                    if not isinstance(sn, dict):
                        continue
                    lt, db = sn.get("latestTrade") or {}, sn.get("dailyBar") or {}
                    price, asof = lt.get("p"), lt.get("t")
                    if not price and db.get("c"):
                        price, asof = db.get("c"), db.get("t")
                    if price:
                        self.store.latest[sym] = {"price": float(price), "asof": asof, "src": "alpaca-iex"}
                        n += 1
            return n
        stalest = sorted(tickers, key=lambda t: (self.store.data.get(t) or {}).get("upd") or "")
        for t in stalest[:max_yahoo]:
            n += self._yahoo_daily(t, "5d")
        return n


def to_num(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def _plausible(t: str) -> bool:
    return 1 <= len(t) <= 6 and all(ch.isalnum() or ch == "." for ch in t) and not t[0].isdigit()
