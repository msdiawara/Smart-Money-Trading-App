"""Market context: trend, breadth, sectors and macro, plus a plain-English regime read.

Computed from the whole-market daily history (Massive) and, if a free FRED key is set, official macro series
from the St. Louis Fed. The regime rules are simple and published (trend-following: Faber 2007; Moskowitz, Ooi &
Pedersen 2012; credit spreads and volatility as stress gauges). They describe the backdrop; they don't time it.
"""
from __future__ import annotations

import datetime as dt
import math

import numpy as np

from .http import HttpClient, HttpError
from .quant import BENCH, Panel, liquid_universe
from .util import log, now_utc, to_float

SECTORS = {"XLK": "Technology", "XLF": "Financials", "XLE": "Energy", "XLV": "Health care",
           "XLY": "Consumer discretionary", "XLP": "Consumer staples", "XLI": "Industrials", "XLB": "Materials",
           "XLU": "Utilities", "XLRE": "Real estate", "XLC": "Communication services"}
ASSETS = {"SPY": "S&P 500", "QQQ": "Nasdaq 100", "IWM": "Small caps (Russell 2000)", "TLT": "Long-term Treasuries",
          "GLD": "Gold", "HYG": "High-yield bonds"}
# Public-domain Federal Reserve series plus two series FRED marks "citation required" (Cboe's VIX; FRED's
# 10y-2y spread). ICE's high-yield spread is deliberately NOT used: its licence forbids reproduction, and this
# app's page is public. Credit stress is read from the high-yield bond ETF (HYG) price instead.
FRED_SERIES = {
    "VIXCLS": "VIX (expected S&P 500 volatility; source: Cboe via FRED)",
    "DGS10": "10-year Treasury yield (Federal Reserve H.15 via FRED)",
    "DGS2": "2-year Treasury yield (Federal Reserve H.15 via FRED)",
    "T10Y2Y": "10-year minus 2-year Treasury yield (FRED)",
    "DFF": "Effective fed funds rate (Federal Reserve H.15 via FRED)",
}
FRED = "https://api.stlouisfed.org/fred/series/observations"
FRED_NOTICE = ("This product uses the FRED® API but is not endorsed or certified by the Federal Reserve Bank of "
               "St. Louis. Data: Federal Reserve Bank of St. Louis (FRED), https://fred.stlouisfed.org; "
               "terms: https://fred.stlouisfed.org/docs/api/terms_of_use.html")


def _ret(c: np.ndarray, i: int, n: int) -> float | None:
    if i - n < 0 or np.isnan(c[i]) or np.isnan(c[i - n]) or c[i - n] <= 0:
        return None
    return float(c[i] / c[i - n] - 1)


def _sma(c: np.ndarray, i: int, n: int) -> float | None:
    w = c[max(0, i - n + 1):i + 1]
    w = w[~np.isnan(w)]
    return float(w.mean()) if len(w) >= int(n * 0.9) else None


def fetch_fred(http: HttpClient, key: str) -> dict:
    out = {}
    for sid, label in FRED_SERIES.items():
        try:
            d = http.get_json(f"{FRED}?series_id={sid}&api_key={key}&file_type=json&sort_order=desc&limit=40")
        except (HttpError, ValueError) as e:
            log.warning("FRED %s: %s", sid, e)
            continue
        obs = [(o.get("date"), to_float(o.get("value"))) for o in d.get("observations") or []]
        obs = [(a, b) for a, b in obs if b is not None]
        if not obs:
            continue
        latest = obs[0]
        month_ago = next((b for a, b in obs if a <= (dt.date.fromisoformat(latest[0]) - dt.timedelta(days=28)).isoformat()),
                         None)
        out[sid] = {"label": label, "value": latest[1], "date": latest[0],
                    "change_1m": (latest[1] - month_ago) if month_ago is not None else None}
    return out


def context(panel: Panel | None, allowed: set[str] | None = None, fred: dict | None = None) -> dict:
    out: dict = {"asof": None, "built": now_utc().isoformat(), "fred": fred or {},
                 "fred_notice": FRED_NOTICE if fred else None}
    reasons_pos, reasons_neg = [], []
    score = 0
    if panel is not None and panel.ok():
        i = panel.T - 1
        out["asof"] = panel.date(i)
        s = panel.c(BENCH)
        spy = float(s[i])
        sma50, sma200 = _sma(s, i, 50), _sma(s, i, 200)
        hi = float(np.nanmax(s[max(0, i - 251):i + 1]))
        lr = np.diff(np.log(s[max(0, i - 21):i + 1][~np.isnan(s[max(0, i - 21):i + 1])]))
        vol20 = float(np.std(lr, ddof=1) * math.sqrt(252)) if len(lr) > 10 else None
        out["spy"] = {"price": spy, "sma50": sma50, "sma200": sma200, "from_high": spy / hi - 1, "vol20": vol20,
                      "r1m": _ret(s, i, 21), "r3m": _ret(s, i, 63), "r6m": _ret(s, i, 126)}
        if sma200:
            if spy > sma200:
                score += 1
                reasons_pos.append(f"S&P 500 is {spy / sma200 - 1:+.1%} vs its 200-day average (uptrend)")
            else:
                score -= 1
                reasons_neg.append(f"S&P 500 is {spy / sma200 - 1:+.1%} vs its 200-day average (downtrend)")
        if sma50 and sma200:
            if sma50 > sma200:
                score += 1
                reasons_pos.append("50-day average above the 200-day")
            else:
                score -= 1
                reasons_neg.append("50-day average below the 200-day")
        # breadth across liquid US companies
        uni = liquid_universe(panel, i, allowed, min_price=5, min_adv=5e6)
        above = highs = lows = 0
        counted = 0
        for t in uni:
            c = panel.c(t)
            m = _sma(c, i, 200)
            if m is None:
                continue
            counted += 1
            above += c[i] > m
            w = c[max(0, i - 251):i]
            w = w[~np.isnan(w)]
            if len(w) > 200:
                highs += c[i] >= w.max()
                lows += c[i] <= w.min()
        if counted >= 200:
            pct = above / counted
            out["breadth"] = {"above_200d": pct, "new_highs": int(highs), "new_lows": int(lows), "universe": counted}
            if pct >= 0.55:
                score += 1
                reasons_pos.append(f"{pct:.0%} of stocks are above their 200-day average (broad participation)")
            elif pct <= 0.40:
                score -= 1
                reasons_neg.append(f"Only {pct:.0%} of stocks are above their 200-day average (narrow market)")
            if highs > 2 * max(lows, 1):
                score += 1
                reasons_pos.append(f"New 52-week highs outnumber lows ({highs} vs {lows})")
            elif lows > 2 * max(highs, 1):
                score -= 1
                reasons_neg.append(f"New 52-week lows outnumber highs ({lows} vs {highs})")
        rows = []
        for t, name in {**SECTORS, **{k: v for k, v in ASSETS.items() if k != BENCH}}.items():
            c = panel.c(t)
            if c is None:
                continue
            r3 = _ret(c, i, 63)
            if r3 is None:
                continue
            rows.append({"ticker": t, "name": name, "r1m": _ret(c, i, 21), "r3m": r3, "r6m": _ret(c, i, 126),
                         "vs_spy_3m": r3 - (out["spy"]["r3m"] or 0), "sector": t in SECTORS})
        rows.sort(key=lambda r: -(r["r3m"] or 0))
        out["sectors"] = rows
    f = fred or {}
    vix = (f.get("VIXCLS") or {}).get("value")
    if vix is not None:
        if vix >= 30:
            score -= 2
            reasons_neg.append(f"VIX at {vix:.0f}: markets pricing in large swings (stress)")
        elif vix >= 22:
            score -= 1
            reasons_neg.append(f"VIX at {vix:.0f}: above-normal expected volatility")
        elif vix <= 15:
            reasons_pos.append(f"VIX at {vix:.0f}: calm")
    hyg = next((r for r in out.get("sectors") or [] if r["ticker"] == "HYG"), None)
    if hyg and hyg.get("r1m") is not None:
        if hyg["r1m"] <= -0.03:
            score -= 1
            reasons_neg.append(f"High-yield bonds (HYG) {hyg['r1m']:+.1%} in a month: credit markets under stress")
        elif hyg["r1m"] >= 0.01 and (hyg.get("r3m") or 0) > 0:
            reasons_pos.append(f"High-yield bonds (HYG) {hyg['r1m']:+.1%} in a month: credit markets calm")
    curve = (f.get("T10Y2Y") or {}).get("value")
    if curve is not None and curve < 0:
        reasons_neg.append(f"Yield curve inverted ({curve:+.2f}%): historically a recession warning with long, variable lags")
    label = "Risk-on" if score >= 3 else "Risk-off" if score <= -2 else "Mixed"
    out["regime"] = {"label": label, "score": score, "positives": reasons_pos, "negatives": reasons_neg,
                     "method": "Trend (200/50-day averages), breadth, new highs vs lows, VIX and high-yield bond "
                               "prices. Describes the backdrop; it is not a timing signal."}
    return out
