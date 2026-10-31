"""Company size & profitability from the SEC XBRL 'frames' API: one request returns a concept for every
reporting company, so the whole market costs a handful of requests per day."""
from __future__ import annotations

import datetime as dt

from .http import HttpClient
from .sec import frames
from .util import log, now_utc


def _recent_quarters(n: int = 4) -> list[str]:
    today = now_utc().date()
    y, q = today.year, (today.month - 1) // 3 + 1
    out = []
    for _ in range(n):
        q -= 1
        if q == 0:
            y, q = y - 1, 4
        out.append(f"CY{y}Q{q}I")
    return out


def load_fundamentals(http: HttpClient) -> dict[str, dict]:
    """{cik10: {"shares": float, "shares_end": date, "net_income": float, "ni_year": int}}"""
    out: dict[str, dict] = {}
    # Shares outstanding (cover-page dei tag, instantaneous). Newest quarter first; older fills gaps.
    for period in _recent_quarters(4):
        rows = frames(http, "dei", "EntityCommonStockSharesOutstanding", "shares", period)
        log.info("frames shares %s: %d companies", period, len(rows))
        for r in rows:
            c = str(r.get("cik")).zfill(10)
            if c not in out or not out[c].get("shares"):
                out.setdefault(c, {})["shares"] = r.get("val")
                out[c]["shares_end"] = r.get("end")
    # Net income, latest full calendar year available
    year = now_utc().year
    for y in (year - 1, year - 2):
        rows = frames(http, "us-gaap", "NetIncomeLoss", "USD", f"CY{y}")
        log.info("frames net income CY%d: %d companies", y, len(rows))
        for r in rows:
            c = str(r.get("cik")).zfill(10)
            d = out.setdefault(c, {})
            if "net_income" not in d:
                d["net_income"] = r.get("val")
                d["ni_year"] = y
        if len(rows) > 2000:
            break
    return out


def is_profitable(f: dict | None) -> bool | None:
    if not f or f.get("net_income") is None:
        return None
    return f["net_income"] > 0


def market_cap(f: dict | None, price: float | None) -> float | None:
    if not f or not f.get("shares") or not price:
        return None
    return float(f["shares"]) * float(price)


def stale_days(f: dict | None) -> int | None:
    if not f or not f.get("shares_end"):
        return None
    try:
        return (now_utc().date() - dt.date.fromisoformat(f["shares_end"])).days
    except ValueError:
        return None
