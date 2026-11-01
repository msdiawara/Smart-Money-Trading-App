"""How the forecasts learn from their own mistakes, using only forecasts that have been graded.

1. Calibration (per horizon, daily). If the model's 80% ranges contained only 70% of outcomes, they were too
   narrow: the app finds the stretch that would have made 80% fit, and the median nudge that removes a
   consistent bias. Forecasts whose outcome windows overlap (a 1-month forecast made today and one made
   tomorrow) mostly share the same outcome, and every stock rides the same market, so evidence is counted in
   independent periods of one horizon, not in days or rows: the correction is shrunk toward "no change" by
   periods/(periods + 8) (the median nudge more strongly), and capped. It must also pass a hold-out test: a
   correction fitted only on older forecasts has to score at least as well on the newest ones as no correction,
   or it isn't applied ("do no harm").
2. Signal tilts (weekly) and the range learner (daily): the machine learning, in ml.py. Ridge regression,
   gradient-boosted trees, a neural network and their average compete on months they never saw; a conditional
   calibration learns which stocks' ranges need more or less stretch. Both must beat simpler choices on unseen
   data before they're used; the signal tilt also needs two weekly passes in a row.
3. Model vs Claude blend: weights follow each one's measured skill on the same forecasts, shrunk by the number
   of independent periods graded.
4. Claude's post-mortems of its biggest misses (analyst workflow) become lessons it reads next time.
"""
from __future__ import annotations

import datetime as dt
import math

import numpy as np

from . import ml
from .forecast import HORIZONS, trading_days_after

FEATS = ml.FEATS
FEAT_V = ml.FEAT_V       # feature definitions (3: price, volume and signal features; see ml.py)
price_features = ml.price_features
tilt_rows = ml.rows_for
tilt_weights = ml.compete


def _rows_window(L: dict, src: str, h: int, days: int) -> list[dict]:
    rows = [r for r in (L.get("graded", {}).get(src, {}).get(str(h), [])) if "w10" in r and "er" in r]
    if not rows:
        return []
    last = max(r["made"] for r in rows)
    cut = (dt.date.fromisoformat(last) - dt.timedelta(days=days)).isoformat()
    return [r for r in rows if r["made"] >= cut]


def _fit_scale(rows: list[dict], w: float, lo: float, hi: float, w_shift: float | None = None
               ) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    er = np.asarray([r["er"] for r in rows], dtype=float)
    sd = 1.4826 * float(np.median(np.abs(er - np.median(er)))) or float(np.std(er)) or 1e-4
    ws = w if w_shift is None else w_shift
    shift = float(np.clip(np.median(er) * ws, -0.25 * sd, 0.25 * sd))
    w10 = np.maximum(np.asarray([r["w10"] for r in rows], dtype=float), 1e-6)
    w90 = np.maximum(np.asarray([r["w90"] for r in rows], dtype=float), 1e-6)
    e2 = er - shift
    need = np.where(e2 >= 0, e2 / w90, -e2 / w10)
    s_hat = float(np.quantile(need, 0.8))
    scale = float(np.clip(1 + (s_hat - 1) * w, lo, hi))
    return scale, shift, s_hat, er, np.stack([w10, w90])


def _pinball3(er: np.ndarray, w10: np.ndarray, w90: np.ndarray, scale: float, shift: float) -> float:
    """Mean pinball loss of the 10th, 50th and 90th percentiles after a stretch and nudge (a proper score: lower is
    better, and it rewards ranges that are right, not just wide)."""
    tot = 0.0
    for tau, q in ((0.1, shift - scale * w10), (0.5, np.full_like(er, shift)), (0.9, shift + scale * w90)):
        u = er - q
        tot += float(np.mean(np.maximum(tau * u, (tau - 1) * u)))
    return tot / 3


def calibration(L: dict, src: str = "quant", min_n: int = 200, min_dates: int = 20, b0: float = 6.0,
                window_days: int = 460, lo: float = 0.6, hi: float = 2.0, fallback: str | None = "quant_bt",
                b_shift: float = 60.0, prev: dict | None = None, switch_edge: float = 0.002) -> dict:
    """Per horizon. Learns from live graded forecasts (the last `window_days`, about the 15 months kept); until there
    are enough, from the walk-forward backtest (the same model run on past dates with only the data available
    then), and says which it used. It corrects lasting misfit (tails fatter than the model allows, say), not this
    month's volatility, which the model already tracks. Evidence is counted in independent periods (a month, or one
    horizon if longer): the stretch is shrunk by periods/(periods + b0), the median nudge, for which a past market
    move is weak evidence, by periods/(periods + b_shift). A hold-out check fits on older forecasts only (whose
    outcomes were known before the newest ones were made) and scores the newest 30% of days: the correction is
    applied only if it scored at least as well there as no correction. So the ranges don't flip from one day to the
    next as the hold-out days roll forward, a correction held back the day before (`prev`) must score at least
    `switch_edge` (0.2%) better to be applied again; one applied the day before stays applied while it does no
    worse than no correction."""
    def enough(rr: list[dict], h: int) -> bool:
        days = {r["made"] for r in rr}
        return len(rr) >= min_n and len(days) >= min_dates and ml.periods(days, h) >= 3

    out = {}
    for name, h in HORIZONS:
        used = src
        win = max(window_days, int(math.ceil(h * 7 / 5)) * 10)
        rows = _rows_window(L, src, h, win)
        # live grades once there are enough of them to use (independent periods included); until then the backtest,
        # so the correction doesn't switch off while the first live months come in
        if fallback and not enough(rows, h):
            bt = _rows_window(L, fallback, h, max(win, window_days * 3))
            if enough(bt, h):
                rows, used = bt, fallback
        days = sorted({r["made"] for r in rows})
        n, nd = len(rows), len(days)
        periods = ml.periods(days, h)
        base = {"n": n, "dates": nd, "periods": periods, "scale": 1.0, "shift": 0.0,
                "from": "live" if used == src else "backtest"}
        if n:
            base["cov80_raw"] = float(np.mean([r.get("in80r", 0) for r in rows]))
        if n < min_n or nd < min_dates or periods < 3:
            out[name] = {**base, "status": "learning", "need": {"n": min_n, "dates": min_dates, "periods": 3}}
            continue
        w = periods / (periods + b0)
        scale, shift, s_hat, _, _ = _fit_scale(rows, w, lo, hi, periods / (periods + b_shift))
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
            p_old = ml.periods([r["made"] for r in old], h)
            sc, shf, _, _, _ = _fit_scale(old, p_old / (p_old + b0), lo, hi, p_old / (p_old + b_shift))
            er = np.asarray([r["er"] for r in new], dtype=float)
            w10 = np.maximum(np.asarray([r["w10"] for r in new], dtype=float), 1e-6)
            w90 = np.maximum(np.asarray([r["w90"] for r in new], dtype=float), 1e-6)
            e2 = er - shf
            need = np.where(e2 >= 0, e2 / w90, -e2 / w10)
            loss0, loss1 = _pinball3(er, w10, w90, 1.0, 0.0), _pinball3(er, w10, w90, sc, shf)
            res.update(holdout_n=len(new), holdout_days=nd - int(nd * 0.7),
                       cov80_holdout_raw=float(np.mean([r.get("in80r", 0) for r in new])),
                       cov80_holdout=float(np.mean(need <= sc)),
                       holdout_gain=round(1 - loss1 / loss0, 5) if loss0 > 0 else None)
            was = ((prev or {}).get(name) or {}).get("status")
            gain = 1 - loss1 / loss0 if loss0 > 0 else 0.0
            # it would have made the newest forecasts worse (or, held back yesterday, isn't clearly better yet)
            if loss0 > 0 and (loss1 > loss0 or (was == "held" and gain < switch_edge)):
                res.update(status="held", scale=1.0, shift=0.0, fitted={"scale": round(scale, 4), "shift": round(shift, 5)})
        out[name] = res
    return out


def tilt_gate(prev: dict | None, new: dict) -> dict:
    """Hysteresis for the weekly re-test: on only after two passes in a row, the second with a newly completed test
    month in it (re-running the same months the next week is not new evidence); once on, it stays on while the
    test keeps t ≥ 1.5 with the last six test months positive on average, and the algorithm it would put in use keeps
    t ≥ 1.5 on its own record; it switches off otherwise. While on, the algorithm in use is that week's best."""
    prev = prev or {}
    if new.get("accepted"):
        fresh = not prev.get("passes") or new.get("last_month") != prev.get("last_month")
        passes = prev.get("passes", 0) + (1 if fresh else 0)
    else:
        passes = 0
    if prev.get("on"):
        hold = new.get("accepted") or ((new.get("best_t") or 0) >= 1.5 and new.get("best") not in (None, "none"))
        if (new.get("t") or 0) >= 1.5 and (new.get("recent_gain") or 0) > 0 and hold:
            if not new.get("accepted"):
                # a weaker but still positive week: the algorithm in use is this week's best (with this week's
                # trio, if it's the fused model), so what runs is always what the latest test measured
                new = {**new, "model": new["best"], "accepted": True,
                       "status": "applied: still holding up on later forecasts (re-tested weekly)"}
            return {**new, "passes": passes, "on": True}
        return {**new, "accepted": False, "model": "none", "passes": passes, "on": False,
                "status": "switched off: stopped improving forecasts it never saw"}
    on = passes >= 2
    if new.get("accepted") and not on:
        new = {**new, "status": "passed once: needs a second pass, with a newly completed month, before it's applied"}
    return {**new, "passes": passes, "on": on}


def tilt_in_force(lr: dict, now: dt.datetime, max_days: int = 21) -> dict | None:
    """The signal learner's last decision, except that one switched on is paused when its weekly re-test is two
    weeks overdue (a nudge nobody has re-checked for three weeks isn't applied), and switched off when it was made
    by an older version of the test (other candidates or settings). It needs two passes again after either."""
    tilt = lr.get("tilt")
    if not tilt or not tilt.get("on"):
        return tilt
    if not ml.current(tilt):
        return {**tilt, "on": False, "accepted": False, "model": "none", "passes": 0,
                "status": "switched off: the test has changed since it passed (it's applied again once it passes the "
                          "new one)"}
    try:
        at = dt.datetime.fromisoformat(str(lr.get("tilt_at")).replace("Z", "+00:00"))
        age = (now - at).days
    except ValueError:
        age = None
    if age is None or age > max_days:
        return {**tilt, "on": False, "accepted": False, "model": "none", "passes": 0,
                "status": "paused: the weekly re-test hasn't run for three weeks (it's applied again once it passes)"}
    return tilt


def blend_weights(L: dict, b0: float = 10.0) -> dict:
    """Weight on Claude in the blend, per horizon, from Claude's and the model's skill on the *same* forecasts:
    each graded Claude forecast also scores the model's forecast from the same close on the same outcome. The
    difference is trusted in proportion to the independent periods graded: periods/(periods + b0)."""
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
        periods = ml.periods([r["made"] for r in rows], h)
        shrink = periods / (periods + b0)
        w = 0.5 + float(np.clip((sc - sm) * 2, -0.3, 0.3)) * shrink
        out[name] = {"w_claude": round(w, 3), "n": len(rows), "days": nd, "periods": periods, "skill_claude": round(sc, 4),
                     "skill_model": round(sm, 4), "status": "weighted by skill on the same forecasts"}
    return out


def features(e_scores: list[dict], quote: dict | None, val: dict | None, bub: dict | None, claude_stance: str | None,
             closes: list[float] | None, bench: list[float] | None = None, volumes: list[float] | None = None,
             context: dict | None = None) -> dict:
    """Signal features recorded with each forecast (for learning which ones matter): see ml.py. `context`: the news,
    filing and financial-statement signals known at the anchor close (ml.context_features)."""
    return {**ml.price_features(closes, bench, volumes), **ml.signal_features(e_scores, val, bub, claude_stance),
            **(context or {})}
