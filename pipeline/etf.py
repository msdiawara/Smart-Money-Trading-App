"""Daily trades of transparent active ETFs, from the holdings files the funds publish themselves.

ETFs that rely on SEC Rule 6c-11 must post their full portfolio on their website every trading day. For an
actively managed fund that's a daily, official record of what a professional manager bought and sold yesterday:
the closest thing to a live trade feed that exists for a fund. ARK's funds are the best-known example (one CSV
per fund; the date in the file is the *next* trading day, i.e. positions at the start of that day).

The app checks each file every couple of hours (a conditional request: nothing is re-downloaded while the file is
unchanged), keeps the previous day's snapshot, and turns the difference into trades:
* new position / exit: unambiguous;
* add / trim: a share change that differs from the fund's own flow. Creations and redemptions change every
  holding by roughly the same percentage, so the fund-wide median share change is treated as flow and only the
  deviation from it counts as a trade (the method ARK trackers use; ARK's own trade e-mails are the only exact
  record, and they aren't public files).
Only the changes are published, with a link to the fund's file. Known limits: a ticker change shows up as an exit plus
a new position on the same day, and a holiday before the file's date shifts the trade date by a day.
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import statistics
from pathlib import Path

from .http import HttpClient, HttpError
from .util import iso, log, now_et, now_utc, parse_date, parse_iso, read_json, write_json

# Verified file (ARKK) plus the other ARK active funds; fund names changed over the years, so each has
# candidate file names (newest first). The job remembers which one works.
_CDN = "https://assets.ark-funds.com/fund-documents/funds-etf-csv/"
DEFAULT_FUNDS = [
    {"ticker": "ARKK", "name": "ARK Innovation ETF", "manager": "ARK Invest (Cathie Wood)",
     "urls": [_CDN + "ARK_INNOVATION_ETF_ARKK_HOLDINGS.csv"]},
    {"ticker": "ARKW", "name": "ARK Next Generation Technology ETF", "manager": "ARK Invest (Cathie Wood)",
     "urls": [_CDN + "ARK_NEXT_GENERATION_TECHNOLOGY_ETF_ARKW_HOLDINGS.csv",
              _CDN + "ARK_NEXT_GENERATION_INTERNET_ETF_ARKW_HOLDINGS.csv"]},
    {"ticker": "ARKG", "name": "ARK Genomic Revolution ETF", "manager": "ARK Invest (Cathie Wood)",
     "urls": [_CDN + "ARK_GENOMIC_REVOLUTION_ETF_ARKG_HOLDINGS.csv",
              _CDN + "ARK_GENOMIC_REVOLUTION_MULTISECTOR_ETF_ARKG_HOLDINGS.csv"]},
    {"ticker": "ARKQ", "name": "ARK Autonomous Technology & Robotics ETF", "manager": "ARK Invest (Cathie Wood)",
     "urls": [_CDN + "ARK_AUTONOMOUS_TECH._&_ROBOTICS_ETF_ARKQ_HOLDINGS.csv",
              _CDN + "ARK_AUTONOMOUS_TECHNOLOGY_&_ROBOTICS_ETF_ARKQ_HOLDINGS.csv"]},
    {"ticker": "ARKF", "name": "ARK Blockchain & Fintech Innovation ETF", "manager": "ARK Invest (Cathie Wood)",
     "urls": [_CDN + "ARK_BLOCKCHAIN_&_FINTECH_INNOVATION_ETF_ARKF_HOLDINGS.csv",
              _CDN + "ARK_FINTECH_INNOVATION_ETF_ARKF_HOLDINGS.csv"]},
    {"ticker": "ARKX", "name": "ARK Space & Defense Innovation ETF", "manager": "ARK Invest (Cathie Wood)",
     "urls": [_CDN + "ARK_SPACE_&_DEFENSE_INNOVATION_ETF_ARKX_HOLDINGS.csv",
              _CDN + "ARK_SPACE_EXPLORATION_&_INNOVATION_ETF_ARKX_HOLDINGS.csv"]},
]
MIN_TRADE_USD = 250_000       # smaller day-to-day wobbles are flow and rounding, not decisions
MIN_DEVIATION = 0.015         # share change must differ from the fund's flow by 1.5 points


def parse_holdings(text: str) -> tuple[str | None, dict[str, dict]]:
    """(file date ISO, {ticker: {shares, value, weight, company, cusip}}) from a fund holdings CSV.
    Tolerates the disclaimer paragraphs ARK appends after the table, blank tickers (cash, notes) and
    numbers with thousands separators."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    hdr_i = next((i for i, ln in enumerate(lines) if "ticker" in ln.lower() and "shares" in ln.lower()), None)
    if hdr_i is None:
        raise ValueError("no holdings table (header with 'ticker' and 'shares') in the file")
    rdr = csv.reader(io.StringIO("\n".join(lines[hdr_i:])))
    header = [h.strip().lower() for h in next(rdr)]

    def col(*names):
        for n in names:
            for i, h in enumerate(header):
                if h.startswith(n):
                    return i
        return None

    ci = {"date": col("date"), "fund": col("fund"), "company": col("company"), "ticker": col("ticker"),
          "cusip": col("cusip"), "shares": col("shares"), "value": col("market value", "value"), "weight": col("weight")}
    if ci["ticker"] is None or ci["shares"] is None:
        raise ValueError("holdings table lacks ticker/shares columns")

    def num(row, k):
        i = ci[k]
        if i is None or i >= len(row):
            return None
        s = row[i].strip().replace(",", "").replace("$", "").replace("%", "")
        try:
            return float(s) if s else None
        except ValueError:
            return None

    out, file_date, wsum, any_w = {}, None, 0.0, False
    for row in rdr:
        if len(row) <= ci["ticker"]:
            continue
        d = parse_date(row[ci["date"]]) if ci["date"] is not None and ci["date"] < len(row) else None
        if d is None:
            continue                                  # disclaimer lines and blanks
        file_date = file_date or d.isoformat()
        w_all = num(row, "weight")
        if w_all is not None:
            wsum, any_w = wsum + w_all, True
        t = row[ci["ticker"]].strip().upper().replace(" ", "")
        sh = num(row, "shares")
        if not t or sh is None or sh <= 0:
            continue
        out[t] = {"shares": sh, "value": num(row, "value"), "weight": num(row, "weight"),
                  "company": row[ci["company"]].strip() if ci["company"] is not None and ci["company"] < len(row) else None,
                  "cusip": row[ci["cusip"]].strip() if ci["cusip"] is not None and ci["cusip"] < len(row) else None}
    if not out:
        raise ValueError("holdings table has no rows")
    out["__weight_total__"] = {"shares": 0, "value": None, "weight": round(wsum, 3) if any_w else None, "company": None, "cusip": None}
    return file_date, out


def weight_total(holdings: dict) -> float | None:
    meta = holdings.get("__weight_total__") or {}
    return meta.get("weight")


def positions(holdings: dict) -> dict:
    return {t: h for t, h in holdings.items() if not t.startswith("__")}


def prev_business_day(d: dt.date) -> dt.date:
    d -= dt.timedelta(days=1)
    while d.weekday() >= 5:
        d -= dt.timedelta(days=1)
    return d


def _easter(y: int) -> dt.date:
    a, b, c = y % 19, y // 100, y % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    m = (32 + 2 * e + 2 * i - h - k) % 7
    n = (a + 11 * h + 22 * m) // 451
    month = (h + m - 7 * n + 114) // 31
    day = (h + m - 7 * n + 114) % 31 + 1
    return dt.date(y, month, day)


def nyse_holidays(y: int) -> set[dt.date]:
    """NYSE full-day closures by rule (observed dates), for days the stored market calendar doesn't cover yet."""
    def observed(d: dt.date) -> dt.date | None:
        if d.weekday() == 5:
            return d - dt.timedelta(days=1) if d.month != 1 else None    # Sat Jan 1: no Friday closure
        return d + dt.timedelta(days=1) if d.weekday() == 6 else d

    def nth_weekday(month: int, weekday: int, n: int) -> dt.date:
        d = dt.date(y, month, 1)
        d += dt.timedelta(days=(weekday - d.weekday()) % 7)
        return d + dt.timedelta(days=7 * (n - 1))

    last_mon_may = dt.date(y, 5, 31)
    last_mon_may -= dt.timedelta(days=(last_mon_may.weekday() - 0) % 7)
    days = {observed(dt.date(y, 1, 1)), nth_weekday(1, 0, 3), nth_weekday(2, 0, 3), _easter(y) - dt.timedelta(days=2),
            last_mon_may, observed(dt.date(y, 6, 19)), observed(dt.date(y, 7, 4)), nth_weekday(9, 0, 1),
            nth_weekday(11, 3, 4), observed(dt.date(y, 12, 25))}
    return {d for d in days if d}


def prev_trading_day(d: dt.date, calendar: list[int] | None) -> dt.date:
    """The trading day before `d`. The stored market calendar (SPY's trading days as ordinals) decides for the days it
    covers; it always lags a day or two behind today, so beyond its last entry weekends and NYSE holidays by rule are
    skipped instead."""
    cal = set(calendar or [])
    lo, hi = (min(cal), max(cal)) if cal else (None, None)
    x = d - dt.timedelta(days=1)
    for _ in range(14):
        o = x.toordinal()
        if lo is not None and lo <= o <= hi:
            if o in cal:
                return x
        elif x.weekday() < 5 and x not in nyse_holidays(x.year):
            return x
        x -= dt.timedelta(days=1)
    return prev_business_day(d)


def diff_holdings(old: dict[str, dict], new: dict[str, dict]) -> list[dict]:
    """Trades implied by two consecutive snapshots (see the module docstring for the flow adjustment)."""
    old, new = positions(old), positions(new)
    both = [t for t in new if t in old and old[t]["shares"] > 0]
    changes = {t: new[t]["shares"] / old[t]["shares"] - 1 for t in both}
    flow = statistics.median(changes.values()) if len(changes) >= 5 else 0.0
    out = []
    for t, h in new.items():
        px = (h["value"] / h["shares"]) if h.get("value") and h["shares"] else None
        if t not in old:
            if px and h["shares"] * px < MIN_TRADE_USD:
                continue
            out.append({"ticker": t, "action": "new", "shares": h["shares"], "delta": h["shares"], "price": px,
                        "value": h["shares"] * px if px else None, "weight": h.get("weight"), "prev_weight": None,
                        "pct_change": None, "company": h.get("company"), "cusip": h.get("cusip"), "flow": flow})
            continue
        o = old[t]
        dev = changes[t] - flow
        expected = o["shares"] * (1 + flow)
        delta = h["shares"] - expected                 # shares traded beyond what flow explains
        if abs(dev) < MIN_DEVIATION or (px and abs(delta) * px < MIN_TRADE_USD):
            continue
        out.append({"ticker": t, "action": "add" if delta > 0 else "trim", "shares": h["shares"], "delta": delta,
                    "price": px, "value": abs(delta) * px if px else None, "weight": h.get("weight"),
                    "prev_weight": o.get("weight"), "pct_change": dev * 100, "company": h.get("company"),
                    "cusip": h.get("cusip"), "flow": flow})
    for t, o in old.items():
        if t in new:
            continue
        px = (o["value"] / o["shares"]) if o.get("value") and o["shares"] else None
        if px and o["shares"] * px < MIN_TRADE_USD:
            continue
        out.append({"ticker": t, "action": "exit", "shares": 0, "delta": -o["shares"], "price": px,
                    "value": o["shares"] * px if px else None, "weight": 0.0, "prev_weight": o.get("weight"),
                    "pct_change": -100.0, "company": o.get("company"), "cusip": o.get("cusip"), "flow": flow})
    out.sort(key=lambda d: -(d["value"] or 0))
    return out


def build_events(fund: dict, trades: list[dict], file_date: str, url: str, calendar: list[int] | None = None) -> list[dict]:
    """Event per trade, in the app's common shape (kind 'etf')."""
    fd = parse_date(file_date)
    trade_date = prev_trading_day(fd, calendar).isoformat() if fd else file_date
    seen = iso(now_utc())
    labels = {"new": "New position", "add": "Added", "trim": "Trimmed", "exit": "Sold out"}
    out = []
    for tr in trades:
        out.append({
            "id": f"etf:{fund['ticker']}:{file_date}:{tr['ticker']}:{tr['action']}",
            "kind": "etf",
            "ticker": tr["ticker"],
            "company": tr.get("company"),
            "actor": {"name": f"{fund['name']} ({fund['ticker']})", "id": f"etf:{fund['ticker']}",
                      "role": fund.get("manager") or "Active ETF", "type": "etf"},
            "action": tr["action"],
            "action_label": labels[tr["action"]],
            "trade_date": trade_date,
            "filed_at": seen,
            "lag_days": 1,
            "shares": tr["shares"],
            "price": round(tr["price"], 4) if tr.get("price") else None,
            "value": round(tr["value"], 2) if tr.get("value") else None,
            "details": {"fund": fund["ticker"], "file_date": file_date, "delta_shares": round(tr["delta"], 2),
                        "weight": tr.get("weight"), "prev_weight": tr.get("prev_weight"),
                        "pct_change": round(tr["pct_change"], 2) if tr.get("pct_change") is not None else None,
                        "flow_pct": round(tr["flow"] * 100, 3), "cusip": tr.get("cusip"),
                        "method": "day-over-day change in the fund's published holdings, net of fund flow"},
            "source": {"name": f"{fund['ticker']} daily holdings file", "url": url},
        })
    return out


def snapshot_path(cache_dir: Path) -> Path:
    return cache_dir / "etf_holdings.json.gz"


MIN_FILE_SHARE = 0.7          # a file with far fewer rows than yesterday's is incomplete, not a sell-off
MIN_WEIGHT_TOTAL = 97.0       # a complete file's weights add up to ~100% (cash included); a cut-off table doesn't
PENDING_HOURS = 2             # a suspicious file is accepted once the same one has been served for this long


def refresh(http: HttpClient, cache_dir: Path, funds: list[dict], state: dict,
            calendar: list[int] | None = None) -> tuple[list[dict], str]:
    """Download each fund's file; when its date moved on, diff against the stored snapshot. Returns (events, note).
    A file that is older than the stored snapshot, or has far fewer holdings, is ignored and the snapshot kept."""
    snaps = read_json(snapshot_path(cache_dir)) or {}
    st = state.setdefault("etf", {})
    urls_ok = st.setdefault("urls", {})
    events, notes, missing = [], [], []
    for fund in funds:
        tk = fund["ticker"]
        cands = ([urls_ok[tk]] if urls_ok.get(tk) else []) + [u for u in fund.get("urls") or [] if u != urls_ok.get(tk)]
        if fund.get("url"):
            cands = [fund["url"]] + [c for c in cands if c != fund["url"]]
        text = used = None
        last_err = None
        etags = st.setdefault("etags", {})
        for u in cands:
            try:
                hdrs = {"If-None-Match": etags[u]} if etags.get(u) else {}
                r = http.get(u, timeout=45, headers=hdrs, ok_statuses=(304,))
                if r.status == 304:                           # unchanged since last time: nothing to download
                    last_err, used = "unchanged", u
                    break
                if r.status != 200 or len(r.content) < 200:
                    last_err = f"HTTP {r.status}"
                    continue
                text, used = r.text(), u
                et = r.headers.get("ETag") or r.headers.get("etag")
                if et:
                    etags[u] = et
                break
            except HttpError as e:
                last_err = f"HTTP {e.status}"
            except Exception as e:  # noqa: BLE001
                last_err = f"{type(e).__name__}"
        if text is None and last_err == "unchanged":
            notes.append(f"{tk}: file unchanged")
            continue
        if text is None:
            missing.append(f"{tk} ({last_err})")
            st.setdefault("errors", {})[tk] = last_err
            continue
        try:
            file_date, holdings = parse_holdings(text)
        except ValueError as e:
            notes.append(f"{tk}: {e}")
            st.setdefault("errors", {})[tk] = str(e)[:120]
            continue
        urls_ok[tk] = used
        st.setdefault("errors", {}).pop(tk, None)
        prev = snaps.get(tk)
        if prev and prev.get("date") == file_date:
            notes.append(f"{tk}: no new file yet ({file_date})")
            continue
        if prev and prev.get("date") and file_date and file_date < prev["date"]:
            notes.append(f"{tk}: served an older file ({file_date} < {prev['date']}); ignored")
            continue
        wt = weight_total(holdings)
        prev_wt = weight_total(prev["holdings"]) if prev and prev.get("holdings") else None
        n_new, n_old = len(positions(holdings)), len(positions(prev["holdings"])) if prev and prev.get("holdings") else 0
        # the weight test only applies when yesterday's file added up (a provider whose weights never reach 100%
        # isn't blocked for ever)
        suspicious = ((n_old and n_new < MIN_FILE_SHARE * n_old)
                      or (wt is not None and prev_wt is not None and prev_wt >= MIN_WEIGHT_TOTAL and wt < MIN_WEIGHT_TOTAL))
        if suspicious:
            # accept it only once the very same file has been served for a while (a real shrink, not a cut-off download)
            pend = st.setdefault("pending", {}).get(tk) or {}
            same = pend.get("date") == file_date and pend.get("n") == n_new and pend.get("w") == wt
            first = parse_iso(pend.get("since")) if same else None
            if not (first and (now_utc() - first).total_seconds() >= PENDING_HOURS * 3600):
                if not same:
                    st["pending"][tk] = {"date": file_date, "n": n_new, "w": wt, "since": iso(now_utc())}
                why = (f"only {n_new} holdings vs {n_old} in the stored snapshot ({prev.get('date')})" if n_old and n_new < MIN_FILE_SHARE * n_old
                       else f"weights add up to {wt:.0f}%")
                notes.append(f"{tk}: file dated {file_date} looks incomplete ({why}); ignored until the same file "
                             f"has been served for {PENDING_HOURS} hours")
                st.setdefault("errors", {})[tk] = "incomplete holdings file"
                continue
        st.setdefault("pending", {}).pop(tk, None)
        if prev and prev.get("date") and file_date and prev["date"] < file_date:
            trades = diff_holdings(positions(prev["holdings"]), positions(holdings))
            events += build_events(fund, trades, file_date, used, calendar)
            notes.append(f"{tk}: {len(trades)} trades on {file_date}")
        else:
            notes.append(f"{tk}: first snapshot ({file_date}), trades start tomorrow")
        snaps[tk] = {"date": file_date, "holdings": holdings, "fetched": iso(now_utc()), "n": n_new}
    write_json(snapshot_path(cache_dir), snaps)
    st["last"] = {tk: {"date": v.get("date"), "n": v.get("n")} for tk, v in snaps.items()}
    st["checked"] = now_et().isoformat()
    if missing:
        notes.append(f"no holdings file for {', '.join(missing)}: set each fund's 'Full Holdings CSV' link in config.yaml")
    return events, "; ".join(notes) if notes else "no funds configured"


def fund_list(settings) -> list[dict]:
    cfg = settings.c("etf", "funds", default=None)
    funds = cfg if isinstance(cfg, list) and cfg else DEFAULT_FUNDS
    out = []
    for f in funds:
        if not isinstance(f, dict) or not f.get("ticker"):
            continue
        tk = str(f["ticker"]).upper()
        base = next((d for d in DEFAULT_FUNDS if d["ticker"] == tk), {})
        out.append({**base, **f, "ticker": tk, "name": f.get("name") or base.get("name") or tk})
    return out


__all__ = ["parse_holdings", "diff_holdings", "build_events", "refresh", "fund_list", "DEFAULT_FUNDS", "log"]
