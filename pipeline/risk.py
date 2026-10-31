"""Risk metrics per ticker → a 1–5 risk level with plain-English factors.

Inputs are facts (volatility, drawdown, beta vs. S&P 500, company size, profitability, liquidity), not
opinions. Level 1 = 'Lower risk', 5 = 'Speculative'. No stock is 'safe'."""
from __future__ import annotations

import math

import numpy as np

from .prices import BENCH, PriceStore

LABELS = {1: "Lower risk", 2: "Moderate", 3: "Elevated", 4: "High", 5: "Speculative"}


def _bucket(x: float | None, edges: list[float]) -> int | None:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    for i, e in enumerate(edges, 1):
        if x < e:
            return i
    return len(edges) + 1


def metrics(store: PriceStore, t: str) -> dict | None:
    s = store.series(t)
    if not s or len(s["c"]) < 30:
        return None
    c = np.asarray(s["c"][-253:], dtype=float)
    v = np.asarray(s["v"][-253:], dtype=float)
    dates = s["dates"][-253:]
    r = np.diff(np.log(c))
    vol = float(np.std(r, ddof=1) * math.sqrt(252)) if len(r) > 20 else None
    peak = np.maximum.accumulate(c)
    mdd = float(np.max(1 - c / peak)) if len(c) else None
    beta = None
    b = store.series(BENCH)
    if b and t != BENCH:
        bmap = dict(zip(b["dates"], b["c"]))
        common = [(d, cc) for d, cc in zip(dates, c) if d in bmap]
        if len(common) > 60:
            x = np.asarray([bmap[d] for d, _ in common], dtype=float)
            y = np.asarray([cc for _, cc in common], dtype=float)
            rx, ry = np.diff(np.log(x)), np.diff(np.log(y))
            var = float(np.var(rx, ddof=1))
            if var > 0:
                beta = float(np.cov(rx, ry, ddof=1)[0, 1] / var)
    last = float(c[-1])

    def ret(n: int) -> float | None:
        return float(c[-1] / c[-1 - n] - 1) * 100 if len(c) > n else None

    adv = float(np.mean((c * v)[-20:])) if len(c) >= 20 else None
    return {
        "last_close": last,
        "last_close_date": dates[-1],
        "vol_1y": vol, "mdd_1y": mdd, "beta": beta,
        "hi_52w": float(np.max(c)), "lo_52w": float(np.min(c)),
        "ret_1m": ret(21), "ret_3m": ret(63), "ret_1y": ret(252) if len(c) > 252 else ret(len(c) - 1),
        "adv_usd": adv,
        "days": len(c),
    }


def risk_level(m: dict | None, mcap: float | None, profitable: bool | None, is_fund: bool = False) -> dict:
    factors: list[str] = []
    if not m:
        return {"level": None, "label": "Not enough price history", "factors": []}
    vb = _bucket((m.get("vol_1y") or 0) * 100, [25, 40, 60, 90])
    db = _bucket((m.get("mdd_1y") or 0) * 100, [20, 35, 50, 70])
    bb = _bucket(m.get("beta"), [0.8, 1.2, 1.6, 2.2])
    cb = None if mcap is None else _bucket(mcap / 1e9, [0.3, 2, 10, 200])
    if cb is not None:
        cb = 6 - cb  # bigger company → lower risk bucket
    parts = [(vb, 0.35), (db, 0.25), (cb, 0.25), (bb, 0.15)]
    num = sum(b * w for b, w in parts if b is not None)
    den = sum(w for b, w in parts if b is not None)
    score = num / den if den else 3.0
    if profitable is False:
        score += 0.6
        factors.append("Lost money last fiscal year")
    elif profitable is True:
        factors.append("Profitable last fiscal year")
    if m.get("adv_usd") is not None and m["adv_usd"] < 5e6:
        score += 0.6
        factors.append("Thinly traded (under $5M a day)")
    level = int(min(5, max(1, round(score))))
    vol = m.get("vol_1y")
    if vol is not None:
        factors.insert(0, f"Moves about {vol*100:.0f}% a year (S&P 500 is usually ~15–20%)")
    if m.get("mdd_1y") is not None:
        factors.append(f"Worst drop in the past year: {m['mdd_1y']*100:.0f}% from its high")
    if m.get("beta") is not None:
        factors.append(f"Beta {m['beta']:.2f} vs. S&P 500")
    if mcap is not None:
        factors.append(f"Company size: {_cap_label(mcap)}")
    elif not is_fund:
        factors.append("Company size unknown")
    return {"level": level, "label": LABELS[level], "score": round(score, 2), "factors": factors}


def _cap_label(mcap: float) -> str:
    b = mcap / 1e9
    if b >= 1000:
        return f"mega-cap (${b/1000:,.2f}T)"
    if b >= 200:
        return f"mega-cap (${b:,.0f}B)"
    if b >= 10:
        return f"large-cap (${b:,.0f}B)"
    if b >= 2:
        return f"mid-cap (${b:,.1f}B)"
    if b >= 0.3:
        return f"small-cap (${b*1000:,.0f}M)"
    return f"micro-cap (${b*1000:,.0f}M)"


def cap_tier(mcap: float | None) -> str | None:
    if mcap is None:
        return None
    b = mcap / 1e9
    return "mega" if b >= 200 else "large" if b >= 10 else "mid" if b >= 2 else "small" if b >= 0.3 else "micro"
