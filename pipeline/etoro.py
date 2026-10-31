"""eToro Popular Investors via eToro's official Public API (needs your own keys).

Leaderboard: GET /api/v2/portfolios/rankings  (winRatio, gain, riskScore, copiers, drawdowns …)
Trades:      GET /api/v1/user-info/people/{username}/portfolio/live, diffed between runs to detect newly
             opened / closed positions.
Symbols:     GET /api/v2/market-data/instruments?instrumentsIds=…"""
from __future__ import annotations

import datetime as dt

from .http import HttpClient, HttpError
from .util import chunks, log, now_utc, parse_iso, to_float

BASE = "https://public-api.etoro.com"


class Etoro:
    def __init__(self, http: HttpClient, api_key: str, user_key: str):
        self.http = http
        self.h = {"x-api-key": api_key, "x-user-key": user_key, "Accept": "application/json"}

    def rankings(self, period: str, max_risk: int, min_copiers: int, pages: int = 3) -> list[dict]:
        out = []
        for p in range(1, pages + 1):
            url = (f"{BASE}/api/v2/portfolios/rankings?period={period}&popularInvestor=true&riskScoreMax={max_risk}"
                   f"&copiersMin={min_copiers}&sort=-copiers&page={p}&pageSize=100")
            d = self.http.get_json(url, headers=self.h)
            out.extend(d.get("results") or [])
            if not (d.get("pagination") or {}).get("hasNext"):
                break
        return out

    def live_portfolio(self, username: str) -> dict:
        return self.http.get_json(f"{BASE}/api/v1/user-info/people/{username}/portfolio/live", headers=self.h)

    def instruments(self, ids: list[int]) -> dict[int, dict]:
        out: dict[int, dict] = {}
        for batch in chunks(sorted(set(ids)), 50):
            q = "&".join(f"instrumentsIds={i}" for i in batch)
            try:
                d = self.http.get_json(f"{BASE}/api/v2/market-data/instruments?{q}&pageSize=100", headers=self.h)
            except HttpError as e:
                log.warning("eToro instruments lookup failed: %s", e)
                continue
            for r in d.get("results") or []:
                out[int(r["instrumentId"])] = {"symbol": r.get("symbol"), "name": r.get("displayName"),
                                               "type": r.get("type")}
        return out


def quality(r: dict) -> float:
    """Composite used to pick whom to follow: consistency matters more than one lucky year."""
    wr = (to_float(r.get("winRatio")) or 0) / 100
    pm = to_float(r.get("profitableMonthsPct")) or 0
    pm = pm / 100 if pm > 1 else pm
    gain = to_float(r.get("gain")) or 0
    dd = abs(to_float(r.get("peakToValley")) or 0)
    dd = dd / 100 if dd > 1 else dd
    risk = to_float(r.get("riskScore")) or 5
    return round(40 * wr + 30 * pm + 20 * max(-1, min(1, gain)) - 25 * dd - 2 * risk, 3)


def pick_leaders(rows: list[dict], top_n: int, min_trades: int) -> list[dict]:
    good = [r for r in rows if (r.get("trades") or 0) >= min_trades and r.get("type", "trader") == "trader"]
    for r in good:
        r["_quality"] = quality(r)
    good.sort(key=lambda r: r["_quality"], reverse=True)
    return good[:top_n]


def leader_card(r: dict) -> dict:
    def pct(v):
        f = to_float(v)
        if f is None:
            return None
        return f * 100 if abs(f) <= 5 and not isinstance(v, int) else f
    return {
        "username": r.get("username"), "name": r.get("fullName") or r.get("username"),
        "gain_pct": round((to_float(r.get("gain")) or 0) * 100, 1),
        "win_ratio": to_float(r.get("winRatio")), "risk_score": r.get("riskScore"),
        "copiers": r.get("copiers"), "trades": r.get("trades"),
        "profitable_months_pct": pct(r.get("profitableMonthsPct")),
        "peak_to_valley_pct": pct(r.get("peakToValley")),
        "annualized_return_pct": pct(r.get("annualizedReturn")),
        "country": r.get("country"), "tier": r.get("subType"), "quality": r.get("_quality"),
        "url": f"https://www.etoro.com/people/{r.get('username')}",
    }


def diff_positions(prev: dict | None, cur_positions: list[dict], first_run_window_h: int = 48) -> tuple[list, list, dict]:
    """Return (opened, closed, new_snapshot). On the first snapshot only very recent opens are reported."""
    snap = {}
    for p in cur_positions:
        pid = str(p.get("positionId"))
        snap[pid] = {"instrumentId": p.get("instrumentId"), "isBuy": p.get("isBuy", True),
                     "openRate": p.get("openRate"), "openTimestamp": p.get("openTimestamp"),
                     "investmentPct": p.get("investmentPct"), "leverage": p.get("leverage"),
                     "netProfit": p.get("netProfit")}
    opened, closed = [], []
    if prev is None:
        cutoff = now_utc() - dt.timedelta(hours=first_run_window_h)
        for pid, p in snap.items():
            t = parse_iso(p.get("openTimestamp"))
            if t and t >= cutoff:
                opened.append((pid, p))
    else:
        for pid, p in snap.items():
            if pid not in prev:
                opened.append((pid, p))
        for pid, p in prev.items():
            if pid not in snap:
                closed.append((pid, p))
    return opened, closed, snap
