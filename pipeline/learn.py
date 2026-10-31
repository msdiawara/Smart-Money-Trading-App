"""How the forecasts learn from their own mistakes, using only forecasts that have been graded.

1. Calibration (per horizon, daily). If the model's 80% ranges contained only 70% of outcomes, they were too
   narrow: the app finds the stretch that would have made 80% fit, and the median nudge that removes a
   consistent bias, then applies them to new forecasts. Shrunk toward "no change" until there are hundreds of
   graded forecasts from many different days (outcomes on the same day move together), and capped.
2. Signal tilts (weekly). Do the app's signals (insider buying, momentum, valuation, Claude's stance…) predict
   where the price ends up relative to the model's median? A heavily shrunk (ridge) regression is fitted on
   older forecasts and must improve the error on newer ones it never saw, in both halves of that test period.
   Only then does it nudge the drift; otherwise the weights stay at zero. Most signals are weak (see
   docs/METHODS.md), so "rejected" is the expected, honest outcome for a long time.
3. Model vs Claude blend: weights follow each one's measured skill versus the naive forecast.
4. Claude's post-mortems of its biggest misses (analyst workflow) become lessons it reads next time.
"""
from __future__ import annotations

import datetime as dt
import math

import numpy as np

from .forecast import HORIZONS, trading_days_after

FEATS = ["sig", "mom", "rev", "val", "claude", "ins", "bub"]
FEAT_LABELS = {"sig": "Smart-money signal score", "mom": "12-month momentum", "rev": "Last month's move",
               "val": "Valuation (rich vs cheap)", "claude": "Claude's stance", "ins": "Insider net buying",
               "bub": "Bubble-risk score"}


def _rows_window(L: dict, src: str, h: int, days: int) -> list[dict]:
    rows = [r for r in (L.get("graded", {}).get(src, {}).get(str(h), [])) if "w10" in r and "er" in r]
    if not rows:
        return []
    last = max(r["made"] for r in rows)
    cut = (dt.date.fromisoformat(last) - dt.timedelta(days=days)).isoformat()
    return [r for r in rows if r["made"] >= cut]


def _fit_scale(rows: list[dict], w: float, lo: float, hi: float) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    er = np.asarray([r["er"] for r in rows], dtype=float)
    sd = 1.4826 * float(np.median(np.abs(er - np.median(er)))) or float(np.std(er)) or 1e-4
    shift = float(np.clip(np.median(er) * w, -0.25 * sd, 0.25 * sd))
    w10 = np.maximum(np.asarray([r["w10"] for r in rows], dtype=float), 1e-6)
    w90 = np.maximum(np.asarray([r["w90"] for r in rows], dtype=float), 1e-6)
    e2 = er - shift
    need = np.where(e2 >= 0, e2 / w90, -e2 / w10)
    s_hat = float(np.quantile(need, 0.8))
    scale = float(np.clip(1 + (s_hat - 1) * w, lo, hi))
    return scale, shift, s_hat, er, np.stack([w10, w90])


def calibration(L: dict, src: str = "quant", min_n: int = 200, min_dates: int = 20, d0: int = 20,
                window_days: int = 200, lo: float = 0.6, hi: float = 2.0, fallback: str | None = "quant_bt") -> dict:
    """Per horizon. Learns from live graded forecasts of the last `window_days`; until there are enough, from the
    walk-forward backtest (the same model run on past dates with only the data available then), and says which
    it used. Outcomes on the same day move together, so the evidence is counted in days, not rows: the correction
    is shrunk toward "no change" by days/(days + d0). The coverage it expects afterwards is measured on the most
    recent 30% of days, fitted only on the older 70% (a hold-out check, not the fit grading itself)."""
    out = {}
    for name, h in HORIZONS:
        used = src
        rows = _rows_window(L, src, h, window_days)
        if fallback and (len(rows) < min_n or len({r["made"] for r in rows}) < min_dates):
            bt = _rows_window(L, fallback, h, window_days * 3)
            if len(bt) >= min_n and len({r["made"] for r in bt}) >= min_dates:
                rows, used = bt, fallback
        days = sorted({r["made"] for r in rows})
        n, nd = len(rows), len(days)
        base = {"n": n, "dates": nd, "scale": 1.0, "shift": 0.0, "from": "live" if used == src else "backtest"}
        if n:
            base["cov80_raw"] = float(np.mean([r.get("in80r", 0) for r in rows]))
        if n < min_n or nd < min_dates:
            out[name] = {**base, "status": "learning", "need": {"n": min_n, "dates": min_dates}}
            continue
        w = nd / (nd + d0)
        scale, shift, s_hat, _, _ = _fit_scale(rows, w, lo, hi)
        res = {**base, "status": "applied", "scale": round(scale, 4), "shift": round(shift, 5), "s_hat": round(s_hat, 4),
               "weight": round(w, 3)}
        cut = days[int(nd * 0.7)]
        cut_d, ends = dt.date.fromisoformat(cut), {}
        for d in days:                            # each forecast's outcome date: h trading sessions after it was made
            if d < cut:
                ends[d] = trading_days_after(dt.date.fromisoformat(d), h)[-1]
        old = [r for r in rows if r["made"] < cut and ends[r["made"]] <= cut_d]   # outcome known before the test days
        new = [r for r in rows if r["made"] >= cut]
        if len(old) >= min_n // 2 and len(new) >= 30:
            nd_old = len({r["made"] for r in old})
            sc, shf, _, _, _ = _fit_scale(old, nd_old / (nd_old + d0), lo, hi)
            er = np.asarray([r["er"] for r in new], dtype=float) - shf
            w10 = np.maximum(np.asarray([r["w10"] for r in new], dtype=float), 1e-6)
            w90 = np.maximum(np.asarray([r["w90"] for r in new], dtype=float), 1e-6)
            need = np.where(er >= 0, er / w90, -er / w10)
            res.update(holdout_n=len(new), holdout_days=nd - int(nd * 0.7),
                       cov80_holdout_raw=float(np.mean([r.get("in80r", 0) for r in new])),
                       cov80_holdout=float(np.mean(need <= sc)))
        out[name] = res
    return out


def _ridge(Z: np.ndarray, y: np.ndarray, kappa: float) -> tuple[float, np.ndarray]:
    n, k = Z.shape
    A = np.hstack([np.ones((n, 1)), Z])
    P = np.eye(k + 1) * kappa * n
    P[0, 0] = 0.0
    b = np.linalg.solve(A.T @ A + P, A.T @ y)
    return float(b[0]), b[1:]


def tilt_weights(L: dict, h: int = 21, min_n: int = 600, min_months: int = 6, kappa: float = 1.0,
                 min_gain: float = 0.002, min_t: float = 2.5) -> dict:
    """Walk-forward test: for each month, fit on everything made before it (minus a gap the length of the
    horizon, so no outcome overlaps), predict that month, and record the relative error reduction. Accept only
    if the average monthly gain is positive with t ≥ 2.5 (standard error inflated by √2 for neighbouring months
    whose outcome windows overlap; 2.5 rather than 2 because the test is repeated every week) and the last three
    test months are positive on average. The caller also requires two passes in a row before switching it on."""
    rows = [r for r in L["graded"].get("quant", {}).get(str(h), []) if r.get("f") and "er" in r]
    for r in rows:                       # the error against the median *before* any tilt, so the target doesn't
        r.setdefault("er0", r["er"])      # move once a tilt is applied (no feedback loop)
    rows.sort(key=lambda r: r["made"])
    months = sorted({r["made"][:7] for r in rows})
    res = {"h": h, "n": len(rows), "months": len(months), "accepted": False, "weights": {}, "feats": FEATS}
    if len(rows) < min_n or len(months) < min_months + 3:
        res["status"] = f"learning: needs {min_n}+ graded {h}-day forecasts over {min_months + 3}+ months"
        return res
    X = np.asarray([[float(r["f"].get(k, 0.0)) for k in FEATS] for r in rows])
    y = np.asarray([r["er0"] for r in rows], dtype=float)
    made = np.asarray([r["made"] for r in rows])
    gap = int(math.ceil(h * 1.4))
    gains = []
    for m in months[min_months:]:
        start = dt.date.fromisoformat(m + "-01")
        cut = (start - dt.timedelta(days=gap)).isoformat()
        tr = made < cut
        te = np.asarray([x[:7] == m for x in made])
        if tr.sum() < min_n * 0.5 or te.sum() < 20:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd == 0] = 1.0
        a, w = _ridge((X[tr] - mu) / sd, y[tr], kappa)
        pred = a + ((X[te] - mu) / sd) @ w
        base = float(np.mean(y[tr]))
        m0 = float(np.mean((y[te] - base) ** 2))
        m1 = float(np.mean((y[te] - pred) ** 2))
        if m0 > 0:
            gains.append(1 - m1 / m0)
    res["test_months"] = len(gains)
    if len(gains) < 3:
        res["status"] = "learning: not enough months to test on yet"
        return res
    g = np.asarray(gains)
    mean = float(g.mean())
    se = float(g.std(ddof=1) / math.sqrt(len(g))) * math.sqrt(2) if len(g) > 1 else float("inf")
    t = mean / se if se > 0 else 0.0
    recent = float(g[-3:].mean())
    res.update(test_gain=round(mean, 5), t=round(t, 2), recent_gain=round(recent, 5),
               monthly_gains=[round(float(x), 4) for x in g[-12:]])
    if mean > min_gain and t >= min_t and recent > 0:
        mu, sd = X.mean(0), X.std(0)
        sd[sd == 0] = 1.0
        a, w = _ridge((X - mu) / sd, y, kappa)
        res.update(accepted=True, status="applied: improved errors on later forecasts it never saw",
                   weights={k: round(float(v), 6) for k, v in zip(FEATS, w)}, mu=mu.round(5).tolist(),
                   sd=sd.round(5).tolist(), cap=round(0.25 * float(np.std(y)), 5))
    else:
        res["status"] = "rejected: didn't reliably improve forecasts it never saw (weights stay at zero)"
    return res


def tilt_for(tw: dict | None, feats: dict) -> tuple[float, list[dict], int]:
    """(log-return tilt over the tested horizon, parts, that horizon in sessions) for one stock's features."""
    if not tw or not tw.get("accepted") or not tw.get("on", True):
        return 0.0, [], 21
    h = int(tw.get("h") or 21)
    parts, tot = [], 0.0
    for k, m, sd in zip(FEATS, tw["mu"], tw["sd"]):
        x = feats.get(k)
        if x is None:
            continue
        c = tw["weights"][k] * (float(x) - m) / sd
        tot += c
        parts.append({"k": k, "label": FEAT_LABELS[k], "h_ret": round(c, 5)})
    cap = tw.get("cap") or 0.02
    tot = float(np.clip(tot, -cap, cap))
    return tot, sorted(parts, key=lambda p: -abs(p["h_ret"]))[:4], h


def tilt_gate(prev: dict | None, new: dict) -> dict:
    """Hysteresis for the weekly re-test: on only after two passes in a row; once on, it stays on while the test
    keeps t ≥ 1.5 with the last three months positive, and switches off otherwise."""
    prev = prev or {}
    passes = (prev.get("passes", 0) + 1) if new.get("accepted") else 0
    if prev.get("on"):
        if (new.get("t") or 0) >= 1.5 and (new.get("recent_gain") or 0) > 0:
            if not new.get("accepted"):
                new = {**new, "weights": prev.get("weights"), "mu": prev.get("mu"), "sd": prev.get("sd"),
                       "cap": prev.get("cap"), "accepted": True,
                       "status": "applied: still holding up on later forecasts (re-tested weekly)"}
            return {**new, "passes": passes, "on": True}
        return {**new, "accepted": False, "weights": {}, "passes": passes, "on": False,
                "status": "switched off: stopped improving forecasts it never saw"}
    on = passes >= 2
    if new.get("accepted") and not on:
        new = {**new, "status": "passed once: needs a second weekly pass in a row before it's applied"}
    return {**new, "passes": passes, "on": on}


def blend_weights(L: dict, n0: int = 40) -> dict:
    """Weight on Claude in the blend, per horizon, from Claude's and the model's skill on the *same* forecasts:
    each graded Claude forecast also scores the model's forecast from the same close on the same outcome."""
    out = {}
    for name, h in HORIZONS:
        rows = [r for r in L.get("graded", {}).get("claude", {}).get(str(h), []) if "qs_m" in r and "qsn" in r]
        if not rows:
            out[name] = {"w_claude": 0.5, "status": "equal weights until both are graded on the same stocks", "n": 0}
            continue
        qsn = sum(r["qsn"] for r in rows)
        sc = 1 - sum(r["qs"] for r in rows) / qsn if qsn > 0 else 0.0
        sm = 1 - sum(r["qs_m"] for r in rows) / qsn if qsn > 0 else 0.0
        nd = len({r["made"] for r in rows})
        shrink = nd / (nd + n0)
        w = 0.5 + float(np.clip((sc - sm) * 2, -0.3, 0.3)) * shrink
        out[name] = {"w_claude": round(w, 3), "n": len(rows), "days": nd, "skill_claude": round(sc, 4),
                     "skill_model": round(sm, 4), "status": "weighted by skill on the same forecasts"}
    return out


def features(e_scores: list[dict], quote: dict | None, val: dict | None, bub: dict | None, claude_stance: str | None,
             closes: list[float] | None) -> dict:
    """Signal features recorded with each forecast (for learning which ones matter)."""
    f: dict = {}
    s = sum(((x.get("score") or 0) / 100.0) * (1 if x.get("direction") == "bullish" else -1 if x.get("direction") == "bearish" else 0)
            for x in e_scores)
    f["sig"] = max(-2.0, min(2.0, s))
    if closes and len(closes) > 253:
        c = closes
        if c[-253] > 0 and c[-22] > 0 and c[-1] > 0:
            f["mom"] = math.log(c[-22] / c[-253])
            f["rev"] = math.log(c[-1] / c[-22])
    if val and val.get("rich_pctl") is not None:
        f["val"] = val["rich_pctl"] / 100 - 0.5
    f["claude"] = {"favorable": 1.0, "watch": 0.5, "neutral": 0.0, "unfavorable": -1.0}.get(claude_stance or "", 0.0)
    ins_b = sum(1 for x in e_scores if x.get("kind") == "insider" and x.get("action") == "buy" and x.get("signal") != "noise")
    ins_s = sum(1 for x in e_scores if x.get("kind") == "insider" and x.get("action") in ("sell", "sale_notice"))
    f["ins"] = max(-3, min(3, ins_b - ins_s)) / 3
    if bub and bub.get("score") is not None:
        f["bub"] = min(1.0, bub["score"] / 10)
    return f
