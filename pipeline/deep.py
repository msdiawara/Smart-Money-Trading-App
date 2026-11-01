"""Deep price history: whole-market daily bars back to 2016 from Alpaca's free plan, so the walk-forward backtest
(and through it the learners) can test on about ten years of months instead of the two the free Massive plan keeps.

Alpaca's free Basic plan serves historical bars from all US exchanges (the consolidated "SIP" feed) back to 2016,
except the latest 15 minutes, at 200 requests a minute. It has no one-request-per-day endpoint like Massive's, so
the history is built once, in resumable stages of a couple of minutes per run:

1. the universe: every US stock and ETF Alpaca lists (listed ones on the main exchanges, inactive ones anywhere).
   Alpaca has no list of companies delisted before it knew them, so with a Massive key the stocks, ADRs, ETFs and
   ETNs Massive lists as delisted since the start are added (leaving out those whose company still trades on a main
   exchange under a new ticker: its bars come with the new one). Without them the backtest would see mostly
   survivors and look better than reality;
2. a monthly sweep of all of them (one bar per month) to find every one that was ever liquid enough for the
   backtest: at least `min_month_dv` dollars traded in some month. That is a generous net, not a ranking: the
   backtest picks its stocks on each date by what was known then (20-day dollar volume of $10M or more), and any
   stock that passes on any date traded $100M+ in one of the two months its 20 days fall in. Ranking by later
   trading instead would keep the stocks that went on to do well and drop the ones that faded (a look-ahead);
3. daily bars for those, split-adjusted like Massive's, into a matrix the backtest reads like the Massive one.

Alpaca resolves a ticker to a company by date ("asof"): a listed company's bars include its history under earlier
tickers, and a delisted one's are asked for as of the month before it was delisted, so a ticker later reused by
another company isn't mixed with it.

After that the Massive matrix (which keeps every day it has fetched) carries the recent years. Nothing from here is
published: the backtest only uses it to grade forecasts made on past dates. Alpaca's data is for personal use.
"""
from __future__ import annotations

import datetime as dt
import re
import secrets
import time
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import numpy as np

from .http import HttpClient, HttpError
from .prices import ALPACA, BENCH, MASSIVE
from .util import UTC, iso, log, now_et, now_utc, read_json, redact

TRADING = ("https://paper-api.alpaca.markets/v2", "https://api.alpaca.markets/v2")   # paper keys, then live ones
EXCHANGES = {"NYSE", "NASDAQ", "AMEX", "ARCA", "BATS", "NYSEARCA", "NYSEMKT"}
SYMBOL = re.compile(r"^[A-Z]{1,5}(\.[A-Z])?$")
START = "2016-01-01"
META_V = 2
MIN_STOCKS = 200          # a finished build with fewer stocks holding a year of prices is treated as a failed one
# Dollars traded in a month that keep a stock: the backtest's floor (quant.liquid_universe: $10M a day over 20 days)
# times 20 days, halved because 20 sessions can straddle two months. Any stock the backtest could pick on any date
# clears it in some month, so the matrix holds every one of them. (Raise the backtest's floor freely; lowering it
# needs this lowered too.)
MIN_MONTH_DV = 10e6 * 20 / 2
DELISTED_TYPES = ("CS", "ADRC", "ETF", "ETN")      # Massive's codes: common stock, ADR, ETF, ETN
RENAMED_TYPES = {"CS", "ADRC"}                     # (an ETF trust's CIK covers many funds: no rename test for those)
WAIT_FOR_SEC_DAYS = 3     # how long the universe waits for the SEC ticker list before leaving delisted stocks out


def listed_ciks(tick: dict | None) -> set[str]:
    """The SEC company IDs whose ticker trades on a main exchange now. The SEC's ticker file also lists
    over-the-counter tickers, where a bankrupt company's stock often ends up: that company is gone, not renamed."""
    return {c for c, v in ((tick or {}).get("by_cik") or {}).items()
            if str((v or {}).get("exchange") or "").strip().upper() not in ("", "OTC")}


def _without_key(url: str) -> str:
    """A Massive next-page link without the API key (the key is added to each request, never stored)."""
    p = urlsplit(url)
    return urlunsplit(p._replace(query=urlencode([(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
                                                  if k.lower() != "apikey"])))


def _asof_before(day: str) -> str:
    """The last day of the month before `day`: when a company delisted on `day` still had its ticker."""
    return (dt.date.fromisoformat(day[:10]).replace(day=1) - dt.timedelta(days=1)).isoformat()


class DeepHistory:
    """The deep matrix (dates × tickers, closes and volumes) and the progress of building it. Exposes `dates`,
    `tickers`, `col`, `close` and `vol` like MarketMatrix, so quant.Panel can wrap it. The arrays (one to two hundred
    MB) are read from disk only when first used."""

    def __init__(self, heavy_dir: Path, start: str = START, min_month_dv: float = MIN_MONTH_DV,
                 min_stocks: int = MIN_STOCKS):
        self.npz = heavy_dir / "deep.npz"
        self.meta_path = heavy_dir / "deep.json.gz"
        self.min_dv = float(min_month_dv)
        self.min_stocks = min_stocks
        self.meta: dict = read_json(self.meta_path) or {}
        self._sb: str | None = None
        if self.meta.get("v") != META_V or self.meta.get("start") != start or self.meta.get("min_dv") != self.min_dv:
            self._reset(start)
        self._loaded = False
        self._dates: list[int] = []
        self._tickers: list[str] = []
        self._col: dict[str, int] = {}
        self._close = np.zeros((0, 0), dtype=np.float32)
        self._vol = np.zeros((0, 0), dtype=np.float32)
        self.dirty = False
        if (self.meta.get("stage") in ("daily", "done") and int(self.meta.get("daily_i") or 0) > 0
                and self._stored_build() != self.meta.get("build")):
            log.warning("deep history: the daily matrix is missing or from another build; fetching it again")
            self.meta.update(stage="daily", daily_i=0)

    def _reset(self, start: str) -> None:
        try:
            self.npz.unlink()                         # the old matrix belongs to another build
        except FileNotFoundError:
            pass
        self._sb = None
        self.meta = {"v": META_V, "start": start, "min_dv": self.min_dv, "stage": "universe",
                     "build": secrets.token_hex(4)}

    def _stored_build(self) -> str | None:
        """The build the matrix on disk belongs to (read without loading the big arrays)."""
        if self._sb is None:
            try:
                with np.load(self.npz, allow_pickle=False) as z:
                    self._sb = str(z["build"])
            except Exception:  # noqa: BLE001 – missing, an older format or unreadable: not this build's
                self._sb = ""
        return self._sb or None

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if self.meta.get("stage") not in ("daily", "done") or self._stored_build() != self.meta.get("build"):
            return
        try:
            with np.load(self.npz, allow_pickle=False) as z:
                self._dates = [int(x) for x in z["dates"]]
                self._tickers = [str(x) for x in z["tickers"]]
                self._close, self._vol = z["close"], z["vol"]
            self._col = {t: j for j, t in enumerate(self._tickers)}
        except Exception as e:  # noqa: BLE001 – a corrupt file just means the daily stage starts again
            log.warning("deep history unreadable (%s); refetching the daily bars", e)
            self.meta.update(stage="daily", daily_i=0)
            self._dates, self._tickers, self._col = [], [], {}
            from .util import write_json
            write_json(self.meta_path, self.meta)     # saved now: no later step may run this time

    @property
    def dates(self) -> list[int]:
        self._load()
        return self._dates

    @property
    def tickers(self) -> list[str]:
        self._load()
        return self._tickers

    @property
    def col(self) -> dict[str, int]:
        self._load()
        return self._col

    @property
    def close(self) -> np.ndarray:
        self._load()
        return self._close

    @property
    def vol(self) -> np.ndarray:
        self._load()
        return self._vol

    @property
    def done(self) -> bool:
        """Built and on disk (checked without reading the arrays)."""
        return (self.meta.get("stage") == "done" and int(self.meta.get("days") or 0) > 260
                and self._stored_build() == self.meta.get("build"))

    # ------------------------------------------------------------ requests
    def _bars(self, http: HttpClient, headers: dict, symbols: list[str], tf: str, start: str, end: str,
              asof: str = "") -> dict[str, list[tuple[dt.date, float, float]]]:
        """Split-adjusted bars from the consolidated feed for a batch of symbols (as of `asof`: which company each
        ticker meant then; blank for today). A symbol Alpaca rejects is found by splitting the batch and left out,
        rather than costing the whole batch; any other error stops the run (it resumes from here next time)."""
        out: dict[str, list] = {}
        token = None
        for _ in range(1000):
            url = (f"{ALPACA}/bars?symbols={','.join(symbols)}&timeframe={tf}&start={start}T00:00:00Z"
                   f"&end={end}T23:59:59Z&adjustment=split&feed=sip&limit=10000")
            if asof:
                url += f"&asof={asof}"
            if token:
                url += f"&page_token={token}"
            try:
                d = http.get_json(url, headers=headers)
            except HttpError as e:
                bad_symbol = e.status in (400, 404, 422) and "symbol" in (e.body or "").lower()
                if bad_symbol and len(symbols) > 1 and not token:
                    half = len(symbols) // 2
                    out.update(self._bars(http, headers, symbols[:half], tf, start, end, asof))
                    out.update(self._bars(http, headers, symbols[half:], tf, start, end, asof))
                    return out
                if bad_symbol and len(symbols) == 1:
                    return out
                raise
            for sym, bars in (d.get("bars") or {}).items():
                for b in bars or []:
                    try:
                        day = dt.datetime.fromisoformat(str(b["t"]).replace("Z", "+00:00")).astimezone(UTC).date()
                        c, v = float(b.get("c")), float(b.get("v") or 0.0)
                    except (KeyError, TypeError, ValueError):
                        continue
                    if c > 0:
                        out.setdefault(sym, []).append((day, c, v))
            token = d.get("next_page_token")
            if not token:
                break
        return out

    def _alpaca_assets(self, http: HttpClient, headers: dict) -> dict[str, str]:
        syms: dict[str, str] = {}
        for k, base in enumerate(TRADING):
            try:
                for status in ("active", "inactive"):
                    for a in http.get_json(f"{base}/assets?status={status}&asset_class=us_equity", headers=headers) or []:
                        s = str(a.get("symbol") or "").upper()
                        # listed ones on the main exchanges; inactive ones wherever they ended up (often OTC)
                        if SYMBOL.match(s) and (status == "inactive" or a.get("exchange") in EXCHANGES):
                            if status == "active" or s not in syms:
                                syms[s] = status
                break
            except HttpError as e:
                if e.status in (401, 403) and k + 1 < len(TRADING):
                    continue                    # live-account keys don't work on the paper host
                raise
        return syms

    def _massive_delisted(self, http: HttpClient, key: str, over) -> None:
        """Stocks, ADRs, ETFs and ETNs Massive lists as delisted since the start, newest first (one page of 1,000 a
        request, five requests a minute). Resumable: an error stops the run and the next one continues from the same
        page; only a refused key, or a request Massive won't accept even from the first page, ends the list early
        (and the note says so). A ticker delisted more than once keeps its latest delisting."""
        m = self.meta
        q = m.setdefault("delisted_q", {"ti": 0, "next": None, "restarts": 0})
        found = m.setdefault("delisted", {})
        types = DELISTED_TYPES
        n = 0
        while q["ti"] < len(types) and (n == 0 or not over()):
            url = q.get("next") or (f"{MASSIVE}/v3/reference/tickers?market=stocks&type={types[q['ti']]}"
                                    f"&active=false&sort=delisted_utc&order=desc&limit=1000")
            try:
                d = http.get_json(url + ("&" if "?" in url else "?") + f"apiKey={key}")
            except HttpError as e:
                if e.status in (400, 404, 422) and q.get("next") and int(q.get("restarts") or 0) < 3:
                    q.update(next=None, restarts=int(q.get("restarts") or 0) + 1)   # a stale page link: start the
                    continue                                                        # type again (found ones stay)
                if e.status in (401, 403, 400, 404, 422):
                    log.warning("deep history: Massive's delisted list: %s", e)
                    m["delisted_note"] = (f"Massive's list of delisted stocks was cut short (HTTP {e.status}, at "
                                          f"{types[q['ti']]} of {', '.join(types)}): the older years may lack some "
                                          "delisted stocks")
                    q["ti"] = len(types)
                    break
                raise                          # a passing failure (rate limit, server error): resume next run
            n += 1
            res = d.get("results") or []
            days = []
            for r in res:
                day = str(r.get("delisted_utc") or "")[:10]
                s = str(r.get("ticker") or "").upper()
                if not day:
                    continue
                days.append(day)
                if day >= m["start"] and SYMBOL.match(s) and day > (found.get(s) or [""])[0]:
                    found[s] = [day, types[q["ti"]], str(r.get("cik") or "").zfill(10) if r.get("cik") else ""]
            nxt = d.get("next_url")
            # sorted newest first: a page that is all older than the start ends this type's list
            if not nxt or not res or (days and max(days) < m["start"]):
                q.update(ti=q["ti"] + 1, next=None, restarts=0)
            else:
                q["next"] = _without_key(nxt)
        if q["ti"] >= len(types):
            m["delisted_done"] = True

    def _universe(self, http: HttpClient, headers: dict, over, massive_key: str | None,
                  listed_ciks: set[str] | None) -> int:
        m = self.meta
        n = 0
        if "alpaca" not in m:
            m["alpaca"] = self._alpaca_assets(http, headers)
            n += len(m["alpaca"])
        if massive_key and not m.get("delisted_done"):
            if n and over():
                return n
            self._massive_delisted(http, massive_key, over)
            if not m.get("delisted_done"):
                return max(n, 1)
        delisted = m.get("delisted") or {}
        if any(k in RENAMED_TYPES for _, k, _ in delisted.values()) and not listed_ciks:
            # without the SEC's list a renamed company would be counted twice (old ticker and new): wait for it a
            # few days, then leave delisted stocks out rather than double count
            since = m.setdefault("sec_wait_since", now_utc().isoformat())
            if (now_utc() - dt.datetime.fromisoformat(since)).days < WAIT_FOR_SEC_DAYS:
                m["sec_note"] = "waiting for the SEC ticker list (it tells renamed companies from delisted ones)"
                return max(n, 1)
            delisted = {s: x for s, x in delisted.items() if x[1] not in RENAMED_TYPES}
            m["delisted_note"] = ("delisted stocks left out: the SEC ticker list (it needs the SEC_CONTACT_EMAIL "
                                  "secret) wasn't there to tell renamed companies from gone ones")
        m.pop("sec_note", None)
        uni = {s: "" for s in m["alpaca"]}
        listed = {s for s, st in m["alpaca"].items() if st == "active"}
        for s, (day, kind, cik) in delisted.items():
            if s in listed:
                continue                      # the ticker belongs to a listed company now
            if kind in RENAMED_TYPES and cik and listed_ciks and cik in listed_ciks:
                continue                      # renamed, not gone: its bars come with its current ticker
            uni[s] = _asof_before(day)
        uni[BENCH] = ""
        end = (now_et().date() - dt.timedelta(days=1)).isoformat()
        m.update(stage="rank", universe=sorted(([s, a] for s, a in uni.items()), key=lambda x: (x[1], x[0])),
                 rank_i=0, dv={}, end=end, n_delisted=sum(1 for a in uni.values() if a))
        for k in ("alpaca", "delisted", "delisted_q", "sec_wait_since"):
            m.pop(k, None)
        return max(n, len(uni))

    @staticmethod
    def _batch(items: list[list[str]], i: int, size: int) -> list[list[str]]:
        """Up to `size` items from position i that share one "as of" date (one request asks for one)."""
        out = []
        for s, a in items[i:i + size]:
            if out and a != out[0][1]:
                break
            out.append([s, a])
        return out

    def _rank(self, http: HttpClient, headers: dict, over) -> int:
        uni, dv = self.meta["universe"], self.meta["dv"]
        i, n = int(self.meta.get("rank_i") or 0), 0
        while i < len(uni) and (n == 0 or not over()):           # at least one batch per run
            batch = self._batch(uni, i, 100)
            got = self._bars(http, headers, [s for s, _ in batch], "1Month", self.meta["start"], self.meta["end"],
                             batch[0][1])
            for sym, bars in got.items():
                dv[sym] = round(max([dv.get(sym, 0.0)] + [c * v for _, c, v in bars]), 0)    # its busiest month
            i += len(batch)
            n += len(batch)
            self.meta["rank_i"] = i
        if i >= len(uni):
            if len(dv) < min(self.min_stocks, max(3, len(uni) // 4)):       # (a small universe: a test's, say)
                self._backoff()
                self.meta.update(rank_i=0, dv={})
                raise RuntimeError(f"the monthly sweep found prices for only {len(dv)} tickers; it starts again later")
            asof = dict((s, a) for s, a in uni)
            chosen = {BENCH} | {s for s, x in dv.items() if x >= self.min_dv}
            self.meta.update(stage="daily", chosen=sorted(([s, asof.get(s, "")] for s in chosen),
                                                          key=lambda x: (x[1], x[0])), daily_i=0)
            self.meta.pop("dv", None)                 # the ranking is done: it isn't needed again
            self.meta.pop("universe", None)
        return n

    def _daily(self, http: HttpClient, headers: dict, over) -> int:
        self._load()
        chosen = self.meta["chosen"]
        start, end = self.meta["start"], self.meta["end"]
        if not self._dates or int(self.meta.get("daily_i") or 0) == 0:     # the trading days: the S&P 500 ETF's
            spy = sorted(self._bars(http, headers, [BENCH], "1Day", start, end).get(BENCH) or [])
            if len(spy) < 260:
                raise RuntimeError(f"only {len(spy)} daily bars for {BENCH} since {start}")
            self._dates = [d.toordinal() for d, _, _ in spy]
            self._tickers = [s for s, _ in chosen]
            self._col = {t: j for j, t in enumerate(self._tickers)}
            self._close = np.full((len(self._dates), len(self._tickers)), np.nan, dtype=np.float32)
            self._vol = np.full((len(self._dates), len(self._tickers)), np.nan, dtype=np.float32)
            self.meta["daily_i"] = 0
        pos = {d: k for k, d in enumerate(self._dates)}
        i, n = int(self.meta.get("daily_i") or 0), 0
        while i < len(chosen) and (n == 0 or not over()):        # at least one batch per run
            batch = self._batch(chosen, i, 50)
            got = self._bars(http, headers, [s for s, _ in batch], "1Day", start, end, batch[0][1])
            for sym, bars in got.items():
                j = self._col.get(sym)
                if j is None:
                    continue
                for day, c, v in bars:
                    k = pos.get(day.toordinal())
                    if k is not None:
                        self._close[k, j], self._vol[k, j] = c, v
            i += len(batch)
            n += len(batch)
            self.meta["daily_i"] = i
            self.dirty = True
        if i >= len(chosen):
            have = int(np.sum(np.sum(np.isfinite(self._close), axis=0) >= 260))
            need = min(self.min_stocks, max(3, len(chosen) // 2))
            if have < need:
                self._backoff()
                self.meta.update(daily_i=0)
                self._dates, self.dirty = [], False
                raise RuntimeError(f"only {have} stocks came back with a year of daily prices "
                                   f"(at least {need} expected); the daily stage starts again later")
            self.meta.update(stage="done", built=now_utc().date().isoformat(), days=len(self._dates), stocks=have)
            self.meta.pop("fails", None)
            self.meta.pop("retry_after", None)
        return n

    def _backoff(self) -> None:
        """A build that came back (nearly) empty is tried again a day later, then two, … up to a week."""
        fails = int(self.meta.get("fails") or 0) + 1
        self.meta.update(fails=fails, retry_after=(now_utc() + dt.timedelta(days=min(7, fails))).isoformat())

    # ------------------------------------------------------------ the job
    def step(self, http: HttpClient, key_id: str, secret: str, seconds: float = 150.0, massive_key: str | None = None,
             listed_ciks: set[str] | None = None) -> tuple[int, str]:
        """Advance the build by up to `seconds` (saving the matrix counts: the time it took last is kept back for
        it). Returns (items done, a note for the Data tab). Progress is saved whatever happens; an error is raised
        after saving, so the Data tab shows it."""
        t0 = time.monotonic()
        keep_back = float(self.meta.get("save_s") or 0.0) + 5.0

        def over() -> bool:
            return time.monotonic() - t0 > max(0.0, seconds - keep_back)
        ra = self.meta.get("retry_after")
        if ra and now_utc().isoformat() < ra:
            return 0, self.note()
        headers = {"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret}
        n = 0
        failed: Exception | None = None
        try:
            if self.meta["stage"] == "universe":
                n += self._universe(http, headers, over, massive_key, listed_ciks)
            if self.meta["stage"] == "rank" and (n == 0 or not over()):
                n += self._rank(http, headers, over)
            if self.meta["stage"] == "daily" and (n == 0 or not over()):
                n += self._daily(http, headers, over)
            self.meta.pop("error", None)
        except HttpError as e:
            src = "Massive" if "massive" in (e.url or "") or "polygon" in (e.url or "") else "Alpaca"
            self.meta["error"] = ("Alpaca rejected the keys (check ALPACA_KEY_ID and ALPACA_SECRET_KEY)"
                                  if e.status in (401, 403) and src == "Alpaca" else f"{src} answered HTTP {e.status}")
            failed = e
        except Exception as e:  # noqa: BLE001 – keep the progress; say what stopped it
            self.meta["error"] = redact(str(e))[:200]          # (a network error's message has the request's URL)
            failed = e
        finally:
            self.meta["updated"] = iso(now_utc())
            self.save()
        if failed is not None:
            log.warning("deep history: %s", redact(str(failed)))
            raise RuntimeError(f"deep history paused: {self.meta['error']} (progress kept; it resumes next run)")
        return n, self.note()

    def note(self) -> str:
        m = self.meta
        if m.get("error"):
            return f"paused: {m['error']}"
        st = m.get("stage")
        extra = f" ({m['delisted_note']})" if m.get("delisted_note") else ""
        if st == "universe":
            got = len(m.get("delisted") or {})
            return ("listing every US stock and ETF, listed and delisted" + (f": {got} delisted since "
                    f"{m['start'][:4]} so far" if got else "") + (f"; {m['sec_note']}" if m.get("sec_note") else "")
                    + extra)
        if st == "rank":
            return (f"finding every stock liquid enough for the backtest since {m['start'][:4]}: {m.get('rank_i', 0)} of "
                    f"{len(m.get('universe') or [])} checked ({m.get('n_delisted', 0)} delisted ones included)" + extra)
        if st == "daily":
            return f"daily prices since {m['start'][:4]}: {m.get('daily_i', 0)} of {len(m.get('chosen') or [])} stocks"
        if not self.done:
            return "complete"
        d0, d1 = (dt.date.fromordinal(int(x)).isoformat() for x in (self.dates[0], self.dates[-1]))
        return f"complete: {len(self.tickers)} stocks × {len(self.dates)} days ({d0} to {d1})" + extra

    def save(self) -> None:
        from .util import write_json
        if self.dirty and self._dates:
            t0 = time.monotonic()
            tmp = self.npz.with_name("deep.tmp.npz")
            np.savez_compressed(tmp, dates=np.asarray(self._dates, dtype=np.int64), tickers=np.asarray(self._tickers),
                                close=self._close, vol=self._vol, build=np.asarray(self.meta["build"]))
            tmp.replace(self.npz)
            self._sb = self.meta["build"]
            self.dirty = False
            self.meta["save_s"] = round(time.monotonic() - t0, 1)
        write_json(self.meta_path, self.meta)
