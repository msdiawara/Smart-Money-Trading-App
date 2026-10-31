"""Cheap, high-value context feeds for every stock in the app.

* Short interest (FINRA, free, no key): shares sold short, days to cover and the change since the last report.
  Published twice a month (positions as of mid-month and month-end, released ~7 business days later) as one file
  for every listed stock: https://cdn.finra.org/equity/otcmarket/biweekly/shrtYYYYMMDD.csv
* Daily short-sale volume (FINRA Reg SHO daily files, free, no key): for every stock, how much of the day's
  off-exchange volume was sold short, posted by 6pm ET the same day:
  https://cdn.finra.org/equity/regsho/daily/CNMSshvolYYYYMMDD.txt  (Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market)
  FINRA's own caveat: off-exchange trades only (not consolidated with the exchanges), so the ratio is a daily
  pressure gauge, comparable with its own history, not a share of all trading.
* Next earnings date (Alpha Vantage EARNINGS_CALENDAR, free key, one request a day for every company): earnings
  are the most common reason a stock gaps past a stop.

All optional; failures only remove the extra line from the app.
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import re
from pathlib import Path

from .http import HttpClient, HttpError
from .util import log, now_et, parse_date, read_json, to_float, write_json

FINRA = "https://cdn.finra.org/equity/otcmarket/biweekly/shrt{d:%Y%m%d}.csv"
FINRA_DAILY = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{d:%Y%m%d}.txt"
SHORT_VOL_DAYS = 25           # history kept per ticker (20-day average + a few spares)
AV = "https://www.alphavantage.co/query?function=EARNINGS_CALENDAR&horizon=3month&apikey={key}"


def _norm(h: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (h or "").lower())


def _business_days_after(d: dt.date, n: int) -> dt.date:
    while n > 0:
        d += dt.timedelta(days=1)
        if d.weekday() < 5:
            n -= 1
    return d


def _settlement_candidates(today: dt.date, months: int = 3) -> list[dt.date]:
    """Mid-month (15th, or the business day before) and month-end business days, newest first."""
    out = []
    y, m = today.year, today.month
    for _ in range(months):
        mid = dt.date(y, m, 15)
        while mid.weekday() >= 5:
            mid -= dt.timedelta(days=1)
        end = (dt.date(y + (m == 12), m % 12 + 1, 1) - dt.timedelta(days=1))
        while end.weekday() >= 5:
            end -= dt.timedelta(days=1)
        out += [end, mid]
        y, m = (y, m - 1) if m > 1 else (y - 1, 12)
    return sorted({d for d in out if _business_days_after(d, 7) <= today}, reverse=True)


def parse_finra(text: str) -> tuple[str | None, dict]:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None, {}
    delim = "|" if lines[0].count("|") > lines[0].count(",") else ","
    rd = csv.reader(lines, delimiter=delim)
    header = [_norm(h) for h in next(rd)]

    def col(*needles: str) -> int | None:
        for n in needles:
            for i, h in enumerate(header):
                if n in h:
                    return i
        return None
    c_sym = col("symbolcode", "symbol")
    c_cur = col("currentshortpositionquantity", "currentshort", "shortinterest")
    c_prev = col("previousshortpositionquantity", "previousshort")
    c_adv = col("averagedailyvolumequantity", "averagedailyvolume", "avgdailyvolume")
    c_dtc = col("daystocoverquantity", "daystocover")
    c_date = col("settlementdate", "settlement")
    if c_sym is None or c_cur is None:
        raise ValueError(f"unexpected FINRA header: {header[:8]}")
    rows, settle = {}, None
    for r in rd:
        if len(r) <= max(c_sym, c_cur):
            continue
        sym = r[c_sym].strip().upper().replace("/", ".")
        cur = to_float(r[c_cur])
        if not sym or cur is None:
            continue
        if settle is None and c_date is not None and len(r) > c_date:
            settle = str(parse_date(r[c_date].strip()) or r[c_date].strip())
        rows[sym] = [cur, to_float(r[c_prev]) if c_prev is not None and len(r) > c_prev else None,
                     to_float(r[c_adv]) if c_adv is not None and len(r) > c_adv else None,
                     to_float(r[c_dtc]) if c_dtc is not None and len(r) > c_dtc else None]
    return settle, rows


def refresh_short_interest(http: HttpClient, cache_dir: Path) -> tuple[int, str]:
    path = cache_dir / "short_interest.json.gz"
    have = read_json(path) or {}
    for d in _settlement_candidates(now_et().date()):
        if have.get("file") == d.isoformat():
            return 0, f"up to date (positions as of {d})"
        try:
            resp = http.get(FINRA.format(d=d), ok_statuses=(403, 404), timeout=90)
        except HttpError as e:
            log.warning("FINRA short interest %s: %s", d, e)
            continue
        text = resp.text()
        if resp.status != 200 or not text or text.lstrip().startswith("<"):   # not published yet
            continue
        settle, rows = parse_finra(text)
        if len(rows) < 500:
            continue
        write_json(path, {"file": d.isoformat(), "settlement": settle or d.isoformat(), "rows": rows})
        return len(rows), f"{len(rows)} stocks, positions as of {settle or d}"
    return 0, "no new FINRA file yet"


def parse_short_volume(text: str) -> tuple[str | None, dict[str, tuple[float, float]]]:
    """(trade date ISO, {symbol: (short volume, total volume)}) from a Reg SHO daily file."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines or "|" not in lines[0]:
        raise ValueError("not a FINRA daily short-volume file")
    header = [_norm(h) for h in lines[0].split("|")]
    try:
        c_date, c_sym, c_short, c_tot = (header.index("date"), header.index("symbol"), header.index("shortvolume"),
                                         header.index("totalvolume"))
    except ValueError:
        raise ValueError(f"unexpected daily short-volume header: {header}") from None
    day, out = None, {}
    for ln in lines[1:]:
        parts = ln.split("|")
        if len(parts) <= max(c_sym, c_short, c_tot) or not parts[c_sym].strip():
            continue
        sv, tv = to_float(parts[c_short]), to_float(parts[c_tot])
        if sv is None or not tv:
            continue
        day = day or (str(parse_date(parts[c_date].strip()) or "") or None)
        out[parts[c_sym].strip().upper().replace("/", ".")] = (sv, tv)
    return day, out


def _recent_trading_days(now: dt.datetime, n: int = 5) -> list[dt.date]:
    """Trading days whose file could be out already (today's appears by ~6pm ET), newest first."""
    d = now.date() if now.hour >= 18 else now.date() - dt.timedelta(days=1)
    out = []
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d -= dt.timedelta(days=1)
    return out


def refresh_short_volume(http: HttpClient, cache_dir: Path, tickers: set[str]) -> tuple[int, str]:
    """Keep ~25 days of (short, total) volume for the tickers the app shows. One ~500 KB file per trading day.
    A day with no file is marked missing (a holiday) only once it is two days old, so a late posting or a
    transient error is retried; stocks that enter the feed later get the last ten known days backfilled."""
    path = cache_dir / "short_volume.json.gz"
    have = read_json(path) or {"days": {}, "by_ticker": {}}
    days = have.setdefault("days", {})
    bt = have.setdefault("by_ticker", {})
    known = set(have.get("tickers") or [])
    new_tickers = {t for t in tickers if t not in known}        # on the first run nothing is known and nothing to re-read
    got, refetch = [], []
    if new_tickers:
        refetch = sorted(k for k, v in days.items() if v != "missing")[-10:]
    today = now_et().date()

    def take(rows: dict, key: str, only: set[str]) -> None:
        for t in only:
            r = rows.get(t)
            if r:
                bt.setdefault(t, {})[key] = [r[0], r[1]]

    recent = _recent_trading_days(now_et(), 12 if not days else 5)        # first run: two weeks of history
    older = [parse_date(k) for k in refetch if k not in {x.isoformat() for x in recent}]
    for d in recent + older:
        key = d.isoformat()
        if key in days and key not in refetch:
            continue
        try:
            resp = http.get(FINRA_DAILY.format(d=d), ok_statuses=(403, 404), timeout=90)
        except HttpError as e:
            log.warning("FINRA daily short volume %s: %s", d, e)
            continue
        text = resp.text()
        ok = resp.status == 200 and text and not text.lstrip().startswith("<")
        rows = {}
        if ok:
            try:
                day, rows = parse_short_volume(text)
            except ValueError as e:
                log.warning("FINRA daily short volume %s: %s", d, e)
                ok = False
        if not ok or len(rows) < 1000:
            if key not in days and (today - d).days >= 2:
                days[key] = "missing"                   # two days with no file: a holiday; stop asking
            continue                                    # not posted yet or a blip: try again next run
        if key in refetch:
            take(rows, key, new_tickers)
            continue
        days[key] = day or key
        take(rows, key, tickers)
        got.append(key)
    have["tickers"] = sorted(tickers)
    # prune: last SHORT_VOL_DAYS trading days, tickers no longer shown
    keep_days = sorted(k for k in days)[-SHORT_VOL_DAYS - 10:]
    have["days"] = {k: days[k] for k in keep_days}
    for t in list(bt):
        if t not in tickers:
            bt.pop(t, None)
            continue
        bt[t] = {k: v for k, v in bt[t].items() if k in have["days"]}
    write_json(path, have)
    n_days = sum(1 for v in have["days"].values() if v != "missing")
    return len(got), f"{n_days} trading days on file for {len(bt)} stocks" + (f"; added {', '.join(got)}" if got else "")


def parse_av_calendar(text: str) -> dict:
    rd = csv.reader(io.StringIO(text))
    try:
        header = [_norm(h) for h in next(rd)]
    except StopIteration:
        return {}
    if "symbol" not in header or "reportdate" not in header:
        raise ValueError(f"unexpected earnings calendar header: {header[:6]}")
    i_sym, i_date = header.index("symbol"), header.index("reportdate")
    i_est = header.index("estimate") if "estimate" in header else None
    i_tod = next((k for k, h in enumerate(header) if h.startswith("timeoftheday")), None)
    today = now_et().date().isoformat()
    out: dict = {}
    for r in rd:
        if len(r) <= max(i_sym, i_date):
            continue
        sym, d = r[i_sym].strip().upper().replace("-", "."), r[i_date].strip()
        if not sym or not d or d < today:
            continue
        if sym not in out or d < out[sym]["date"]:
            out[sym] = {"date": d, "estimate": to_float(r[i_est]) if i_est is not None and len(r) > i_est else None,
                        "time": r[i_tod].strip() if i_tod is not None and len(r) > i_tod else None}
    return out


def refresh_earnings(http: HttpClient, cache_dir: Path, key: str) -> tuple[int, str]:
    text = http.get(AV.format(key=key), timeout=90).text()
    if text.lstrip().startswith("{"):                       # JSON = an error/limit message, not data
        raise ValueError("Alpha Vantage returned a message instead of data (check the key or daily limit)")
    cal = parse_av_calendar(text)
    write_json(cache_dir / "earnings.json.gz", {"asof": now_et().date().isoformat(), "next": cal})
    return len(cal), f"{len(cal)} companies with a scheduled report in the next 3 months"


def short_volume_view(hist: dict | None) -> dict | None:
    """Latest day's short share of off-exchange volume vs its 20-day average."""
    if not hist:
        return None
    days = sorted(hist)
    last = days[-1]
    sv, tv = hist[last]
    if not tv:
        return None
    pcts = [h[0] / h[1] * 100 for k in days[-21:-1] for h in [hist[k]] if h[1]]
    return {"pct": round(sv / tv * 100, 1), "avg20_pct": round(sum(pcts) / len(pcts), 1) if len(pcts) >= 5 else None,
            "days": len(pcts) + 1, "total": tv, "asof": last}


def attach(quotes: dict, cache_dir: Path, panel=None) -> None:
    si = read_json(cache_dir / "short_interest.json.gz") or {}
    rows = si.get("rows") or {}
    er = (read_json(cache_dir / "earnings.json.gz") or {}).get("next") or {}
    svh = (read_json(cache_dir / "short_volume.json.gz") or {}).get("by_ticker") or {}
    today = now_et().date()
    for t, q in quotes.items():
        sv = short_volume_view(svh.get(t))
        if sv:
            q["short_volume"] = sv
        else:
            q.pop("short_volume", None)
        rv = relative_volume(panel, t) if panel is not None else None
        if rv:
            q["rel_volume"] = rv
        else:
            q.pop("rel_volume", None)
        r = rows.get(t)
        if r:
            cur, prev, adv, dtc = (r + [None] * 4)[:4]
            shares_out = (q.get("mcap") / q["price"]) if (q.get("mcap") and q.get("price")) else None
            q["short"] = {"shares": cur, "days_to_cover": dtc,
                          "pct_shares": (cur / shares_out * 100) if (cur and shares_out) else None,
                          "change_pct": ((cur / prev - 1) * 100) if (cur and prev) else None,
                          "asof": si.get("settlement")}
        else:
            q.pop("short", None)
        e = er.get(t)
        if e:
            d = parse_date(e["date"])
            q["earnings"] = {**e, "days": (d - today).days if d else None}
        else:
            q.pop("earnings", None)


def relative_volume(panel, t: str) -> dict | None:
    """Yesterday's consolidated volume versus its 20-day average, from the stored whole-market history
    (Massive, end of day): an 'unusual volume' flag with a one-day lag."""
    try:
        v = panel.v(t)
    except Exception:  # noqa: BLE001 – no panel column
        return None
    if v is None or panel.T < 25:
        return None
    import numpy as np
    last = panel.T - 1
    vv = v[last - 20:last + 1]                           # the latest day plus the 20 before it
    if np.isnan(vv[-1]) or vv[-1] <= 0:
        return None
    base = vv[:-1]
    base = base[~np.isnan(base) & (base > 0)]
    if len(base) < 10:
        return None
    ratio = float(vv[-1] / np.mean(base))
    return {"ratio": round(ratio, 2), "date": panel.date(last), "volume": float(vv[-1]), "avg20": float(np.mean(base))}
