"""Win rates computed by the app from official filings + prices ("would copying them have worked?").

Method (same for insiders and Congress): enter at the close of the first trading day AFTER the filing /
disclosure date (what a copier could actually do), exit 63 trading days (~3 months) later.
  win  = stock went up;   beat = stock beat SPY over the same window.
Funds: buy the fund's top 13F positions (value-weighted) after each filing, hold to the next filing.
Ranking uses the Wilson lower bound of the beat rate so a 3-for-3 streak doesn't outrank 30-for-45."""
from __future__ import annotations

import bisect
import csv
import datetime as dt
import io
import math
import zipfile
from collections import defaultdict
from pathlib import Path

from . import sec
from .http import HttpClient, HttpError
from .prices import BENCH, PriceClient, PriceStore
from .util import date_str, log, now_utc, parse_date, read_json, title_name, to_float, write_json

HOLD = 63
DERA_URLS = ["https://www.sec.gov/files/datastandardsinnovation/data/insider-transactions-data-sets/{y}q{q}_form345.zip",
             "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{y}q{q}_form345.zip"]


def wilson_low(k: int, n: int, z: float = 1.28) -> float:
    if n == 0:
        return 0.0
    p = k / n
    den = 1 + z * z / n
    c = p + z * z / (2 * n)
    adj = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - adj) / den


def forward_return(store: PriceStore, ticker: str, after: dt.date, hold: int = HOLD,
                   info: dict | None = None) -> tuple[float, float, str] | None:
    """(stock return, SPY return, entry date): enter at the close of the first trading day after `after`, exit
    `hold` trading days later. Both legs use the same calendar (SPY's), so a stock with gaps in its own
    history is still measured over exactly the same window as the benchmark.

    A stock that stopped trading inside the window (its history was refreshed after the exit date but ends
    before it) is measured to its last close against SPY over the same days, rather than dropped: dropping
    delisted stocks would hide many of the worst outcomes."""
    s, b = store.series(ticker), store.series(BENCH)
    if not s or not b:
        return None
    j = store.index_on_or_after(BENCH, after + dt.timedelta(days=1))
    if j is None or j + hold >= len(b["c"]):
        return None
    d_in, d_out = b["dates"][j], b["dates"][j + hold]
    i = bisect.bisect_left(s["dates"], d_in)
    if i >= len(s["dates"]) or s["dates"][i] - d_in > 5:      # no price near entry → delisted/renamed/halted
        return None
    k = bisect.bisect_right(s["dates"], d_out) - 1
    if k <= i:
        return None
    jb = j + hold
    if d_out - s["dates"][k] > 5:                              # stopped trading before the exit date
        upd = parse_date((s.get("upd") or "")[:10])
        if k != len(s["dates"]) - 1 or not upd or upd.toordinal() < d_out + 3:
            return None                                        # a gap, or history not refreshed since: can't tell
        jb = bisect.bisect_right(b["dates"], s["dates"][k]) - 1
        if jb <= j:
            return None
        if info is not None:
            info["delisted"] = True
    r = s["c"][k] / s["c"][i] - 1
    rb = b["c"][jb] / b["c"][j] - 1
    return r, rb, dt.date.fromordinal(s["dates"][i]).isoformat()


def summarize(trades: list[dict]) -> dict:
    done = [t for t in trades if t.get("ret") is not None]
    n = len(done)
    if not n:
        return {"n": 0, "pending": len(trades)}
    wins = sum(1 for t in done if t["ret"] > 0)
    beats = sum(1 for t in done if t["ret"] > t["spy"])
    return {"n": n, "pending": len(trades) - n, "win_rate": wins / n, "beat_rate": beats / n,
            "avg_ret": sum(t["ret"] for t in done) / n, "avg_excess": sum(t["ret"] - t["spy"] for t in done) / n,
            "rank_score": wilson_low(beats, n)}


# ---------------------------------------------------------------- insiders (SEC DERA data sets)

def _quarters_back(n: int) -> list[tuple[int, int]]:
    d = now_utc().date()
    y, q = d.year, (d.month - 1) // 3 + 1
    out = []
    for _ in range(n):
        q -= 1
        if q == 0:
            y, q = y - 1, 4
        out.append((y, q))
    return out


def _read_tsv(z: zipfile.ZipFile, name: str, cols: list[str]) -> list[dict]:
    member = next((m for m in z.namelist() if m.upper().endswith(name.upper())), None)
    if not member:
        return []
    with z.open(member) as fh:
        rd = csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8", errors="replace"), delimiter="\t")
        want = [c for c in cols if c in (rd.fieldnames or [])]
        return [{c: row.get(c) for c in want} for row in rd]


def load_dera_quarter(http: HttpClient, cache_dir: Path, y: int, q: int) -> list[dict] | None:
    """Compact list of Form 4 open-market P/S trades for one quarter (cached forever)."""
    path = cache_dir / "dera" / f"{y}q{q}.json.gz"
    cached = read_json(path)
    if cached is not None:
        return cached
    raw = None
    for pattern in DERA_URLS:
        try:
            raw = http.get(pattern.format(y=y, q=q), timeout=180).content
            break
        except HttpError as e:
            if e.status not in (403, 404):
                raise
    if raw is None:
        return None
    z = zipfile.ZipFile(io.BytesIO(raw))
    subs = {r["ACCESSION_NUMBER"]: r for r in _read_tsv(z, "SUBMISSION.tsv", [
        "ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERCIK", "ISSUERNAME", "ISSUERTRADINGSYMBOL", "AFF10B5ONE"])}
    owners: dict[str, dict] = {}
    for r in _read_tsv(z, "REPORTINGOWNER.tsv", ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNERNAME",
                                                 "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE"]):
        owners.setdefault(r["ACCESSION_NUMBER"], r)
    agg: dict[tuple, dict] = {}
    for r in _read_tsv(z, "NONDERIV_TRANS.tsv", ["ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE", "TRANS_SHARES",
                                                 "TRANS_PRICEPERSHARE"]):
        code = (r.get("TRANS_CODE") or "").strip().upper()
        if code not in ("P", "S"):
            continue
        acc = r["ACCESSION_NUMBER"]
        s = subs.get(acc)
        if not s or (s.get("DOCUMENT_TYPE") or "").strip() not in ("4", "4/A"):
            continue
        o = owners.get(acc) or {}
        key = (acc, code)
        a = agg.setdefault(key, {"acc": acc, "code": code, "owner": (o.get("RPTOWNERCIK") or "").zfill(10),
                                 "owner_name": o.get("RPTOWNERNAME"), "role": o.get("RPTOWNER_RELATIONSHIP"),
                                 "title": o.get("RPTOWNER_TITLE"), "issuer": (s.get("ISSUERCIK") or "").zfill(10),
                                 "ticker": (s.get("ISSUERTRADINGSYMBOL") or "").upper().strip(),
                                 "filed": date_str(parse_date(s.get("FILING_DATE"))),
                                 "traded": date_str(parse_date(r.get("TRANS_DATE"))), "value": 0.0,
                                 "plan": (s.get("AFF10B5ONE") or "").strip() in ("1", "true", "True")})
        sh, px = to_float(r.get("TRANS_SHARES")) or 0, to_float(r.get("TRANS_PRICEPERSHARE")) or 0
        a["value"] += sh * px
    rows = list(agg.values())
    write_json(path, rows)
    (cache_dir / ".dirty").write_text("1")
    log.info("DERA %dQ%d: %d open-market trades", y, q, len(rows))
    return rows


def build_insider_records(http: HttpClient, settings, hist: PriceStore, prices: PriceClient, budget) -> dict:
    cache = settings.heavy_dir
    allrows: list[dict] = []
    got = []
    for y, q in _quarters_back(16):
        if not budget.ok(120):
            break
        rows = load_dera_quarter(http, cache, y, q)
        if rows is None:
            continue
        got.append(f"{y}Q{q}")
        allrows.extend(rows)
    # calendar of trade months per insider (for routine vs opportunistic)
    months: dict[str, set] = defaultdict(set)
    for r in allrows:
        d = parse_date(r.get("traded"))
        if d:
            months[r["owner"]].add((d.year, d.month))
    cutoff = now_utc().date() - dt.timedelta(days=int(2.5 * 365))
    buys = [r for r in allrows if r["code"] == "P" and r["ticker"] and not r["plan"]
            and (parse_date(r["filed"]) or cutoff) >= cutoff and r["value"] >= 1000]
    tickers = sorted({b["ticker"].replace(" ", "") for b in buys})
    if budget.ok(180):
        prices.store = hist
        prices.refresh_daily(tickers, days=int(2.7 * 365), max_tickers=6000)
    per: dict[str, list] = defaultdict(list)
    meta: dict[str, dict] = {}
    for b in buys:
        f = parse_date(b["filed"])
        fr = forward_return(hist, b["ticker"], f) if f else None
        per[b["owner"]].append({"ret": fr[0] if fr else None, "spy": fr[1] if fr else None, "ticker": b["ticker"],
                                "filed": b["filed"], "value": b["value"]})
        meta[b["owner"]] = {"name": title_name(b["owner_name"] or ""), "title": b.get("title") or b.get("role"),
                            "ticker": b["ticker"]}
    out = {}
    for owner, trades in per.items():
        s = summarize(trades)
        if s.get("n", 0) >= 1:
            s.update(meta[owner])
            s["recent"] = sorted(trades, key=lambda t: t["filed"], reverse=True)[:5]
            out[f"cik:{owner}"] = s
    priced = sum(1 for t in [x for v in per.values() for x in v] if t["ret"] is not None)
    first_q = min((tuple(int(x) for x in g.split("Q")) for g in got), default=None)
    mon = {f"cik:{k}": sorted(f"{y}-{m:02d}" for y, m in v) for k, v in months.items()}
    if first_q:
        mon["_window_start"] = f"{first_q[0]}-{3 * first_q[1] - 2:02d}"
    return {"records": out, "months": mon,
            "quarters": got, "buys": len(buys), "priced": priced, "built": now_utc().isoformat()}


def is_routine(months: dict, actor_id: str, trade_date: str | None) -> bool | None:
    """Cohen-Malloy-Pomorski: routine = traded in the same calendar month in each of the 3 prior years."""
    d = parse_date(trade_date)
    hist = months.get(actor_id)
    if not d or not hist:
        return None
    ws = months.get("_window_start")
    if not isinstance(ws, str):
        ws = min((v[0] for k, v in months.items() if not k.startswith("_") and isinstance(v, list) and v), default=None)
    if not ws or f"{d.year - 3}-{d.month:02d}" < ws:     # the month 3 years back is before the data starts
        return None
    ym = set(hist)
    years_with_trades = {int(x[:4]) for x in ym if int(x[:4]) < d.year}
    if not {d.year - 1, d.year - 2, d.year - 3} <= years_with_trades:
        return None
    return all(f"{d.year - k}-{d.month:02d}" in ym for k in (1, 2, 3))


# ---------------------------------------------------------------- Congress

def build_congress_records(events: list[dict], hist_rows: list[dict], hist: PriceStore, prices: PriceClient,
                           budget) -> dict:
    rows = [e for e in events if e.get("kind") == "congress" and e.get("action") == "buy" and e.get("ticker")]
    rows += [r for r in hist_rows if r.get("action") == "buy" and r.get("ticker")]
    seen, uniq = set(), []
    for r in rows:
        k = (r.get("actor", {}).get("id") or r.get("actor_id"), r.get("ticker"), r.get("trade_date"))
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    if budget.ok(90):
        prices.store = hist
        prices.refresh_daily(sorted({r["ticker"] for r in uniq}), days=int(2.7 * 365))
    per: dict[str, list] = defaultdict(list)
    names = {}
    for r in uniq:
        aid = r.get("actor", {}).get("id") or r.get("actor_id")
        f = parse_date(r.get("filed_at") or r.get("filed"))
        fr = forward_return(hist, r["ticker"], f) if f else None
        per[aid].append({"ret": fr[0] if fr else None, "spy": fr[1] if fr else None, "ticker": r["ticker"],
                         "filed": date_str(f)})
        names[aid] = {"name": r.get("actor", {}).get("name") or r.get("actor_name"), "role": r.get("actor", {}).get("role")}
    out = {}
    for aid, trades in per.items():
        s = summarize(trades)
        s.update(names.get(aid, {}))
        s["recent"] = sorted(trades, key=lambda t: t["filed"] or "", reverse=True)[:5]
        out[aid] = s
    return {"records": out, "built": now_utc().isoformat(), "buys": len(uniq)}


# ---------------------------------------------------------------- Funds: copy-the-13F backtest

def build_fund_records(http: HttpClient, settings, figi, hist: PriceStore, prices: PriceClient, budget,
                       load_13f) -> dict:
    top_n = settings.c("funds", "top_positions_for_backtest", default=50)
    out = {}
    for f in settings.c("funds", "list", default=[]):
        if not budget.ok(120):
            break
        cik = f["cik"]
        try:
            sub = sec.submissions(http, cik)
        except HttpError:
            continue
        if f.get("expect") and f["expect"].upper() not in (sub.get("name") or "").upper():
            continue
        filings = [r for r in sec.recent_filings(sub, {"13F-HR"})][:9]
        books = []
        for r in filings:
            if not budget.ok(90):
                break
            try:
                b = load_13f(http, settings, cik, r)
            except (HttpError, ValueError):
                b = None
            if b:
                books.append(b)
        books.sort(key=lambda b: b["filed"])
        cusips = [p["cusip"] for b in books for p in list(b["positions"].values())[:top_n] if not p.get("put_call")]
        figi.map(cusips, max_requests=40)
        tickers = set()
        legs = []
        for i, b in enumerate(books):
            start = parse_date(b["filed"])
            end = parse_date(books[i + 1]["filed"]) if i + 1 < len(books) else now_utc().date()
            pos = [p for p in list(b["positions"].values()) if not p.get("put_call")][:top_n]
            mapped = [(figi.lookup(p["cusip"]) or {}).get("ticker") for p in pos]
            legs.append((start, end, pos, mapped))
            tickers.update(t for t in mapped if t)
        if budget.ok(60):
            prices.store = hist
            prices.refresh_daily(sorted(tickers), days=int(2.7 * 365))
        quarters = []
        for start, end, pos, mapped in legs:
            if not start or (end - start).days < 20:
                continue
            num = den = covered = 0.0
            total = sum(p["value"] for p in pos) or 1
            for p, t in zip(pos, mapped):
                if not t:
                    continue
                a = hist.close_on_or_after(t, start + dt.timedelta(days=1))
                z = hist.close_on_or_after(t, end) if end < now_utc().date() else None
                s = hist.series(t)
                if a and not z and s:
                    z = (dt.date.fromordinal(s["dates"][-1]), s["c"][-1])
                if not a or not z or z[0] <= a[0]:
                    continue
                w = p["value"]
                num += w * (z[1] / a[1] - 1)
                den += w
                covered += p["value"]
            ba = hist.close_on_or_after(BENCH, start + dt.timedelta(days=1))
            bs = hist.series(BENCH)
            bz = hist.close_on_or_after(BENCH, end) if end < now_utc().date() else ((dt.date.fromordinal(bs["dates"][-1]), bs["c"][-1]) if bs else None)
            if den <= 0 or not ba or not bz:
                continue
            quarters.append({"start": start.isoformat(), "end": end.isoformat(), "ret": num / den,
                             "spy": bz[1] / ba[1] - 1, "coverage": covered / total, "open": end >= now_utc().date()})
        closed = [q for q in quarters if not q["open"]]
        if not quarters:
            continue
        cum = 1.0
        cum_spy = 1.0
        for q in closed:
            cum *= 1 + q["ret"]
            cum_spy *= 1 + q["spy"]
        beats = sum(1 for q in closed if q["ret"] > q["spy"])
        out[f"cik:{cik}"] = {"name": f["name"], "manager": f.get("manager"), "style": f.get("style"),
                             "quarters": len(closed), "beat_rate": (beats / len(closed)) if closed else None,
                             "cum_ret": cum - 1, "cum_spy": cum_spy - 1,
                             "avg_coverage": sum(q["coverage"] for q in quarters) / len(quarters),
                             "legs": quarters[-8:], "rank_score": wilson_low(beats, len(closed)) if closed else 0}
    figi.save()
    return {"records": out, "built": now_utc().isoformat()}
