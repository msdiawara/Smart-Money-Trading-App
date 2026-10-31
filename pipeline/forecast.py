"""Price forecasts for every stock: a full probability distribution of where the price could be after a day, a
week, a month, three months and a year, how high and low it is likely to trade on the way, and when the riskiest
days are.

Method (filtered historical simulation, Barone-Adesi, Giannopoulos & Vosper 1999; volatility as in RiskMetrics):
  1. Daily log returns over the last ~3 years. Volatility is tracked with an exponentially weighted average
     (λ = 0.94), skipping earnings-reaction days, which are modelled separately.
  2. Each past return divided by the volatility at the time gives a "standardized shock". Resampling those keeps
     the stock's own fat tails and skew instead of assuming a bell curve.
  3. Volatility then drifts back toward its long-run level (half-life ~40 trading days), as it does in practice.
  4. Scheduled events get extra variance: earnings (scaled by how hard this stock has moved on its past
     earnings days, from its 8-K item 2.02 filings), Fed decisions and CPI releases.
  5. Drift: the risk-free rate plus beta × an equity premium (expected-return assumptions, not a prediction),
     plus any learned "tilt" that has passed an out-of-sample test (see learn.py). The median path sits below the
     average path for volatile stocks: that is the arithmetic of compounding, not pessimism.
  6. 2,000 simulated paths. Everything shown comes from those paths: ranges, the chance of finishing higher, the
     chance of touching +/-10% along the way, the typical high and low, drawdowns.
  7. Calibration learned from the app's own graded forecasts widens or narrows the ranges (and nudges the
     median) per horizon until "80% ranges" really contain about 80% of outcomes.

What this can't do: tell you the day of a peak or a dip. For a random-walk-like price the timing of the maximum
is close to uniform-to-U-shaped (the arcsine law), so the app shows *risk windows* (event days that carry a big
share of the risk) and the probabilities of touching levels instead of fake dates.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import math
from functools import lru_cache
from statistics import NormalDist

import numpy as np

HORIZONS = [("1d", 1), ("1w", 5), ("1m", 21), ("3m", 63), ("1y", 252)]
H_MAX = 252
TAUS = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
QK = ["05", "10", "25", "50", "75", "90", "95"]
_N = NormalDist()
Z_TAUS = np.array([_N.inv_cdf(float(t)) for t in TAUS])
MODEL = "fhs-ewma-1"

# Scheduled US macro events (decision/release days), including past years so the backtest sees the same events
# the live model does. Sources: federalreserve.gov FOMC calendars (2027 as published, tentative); bls.gov CPI
# release schedules, with the 2025 shutdown changes (September 2025 CPI moved to Oct 24, October's cancelled,
# November's on Dec 18) and January 2026's move to Feb 13 (checked 31 Oct 2026). Config can add more.
FOMC = ["2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07",
        "2024-12-18", "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17",
        "2025-10-29", "2025-12-10", "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16", "2026-10-28",
        "2026-12-09", "2027-01-27", "2027-03-17", "2027-04-28", "2027-06-09", "2027-07-28", "2027-09-15",
        "2027-10-27", "2027-12-08", "2028-01-26"]
CPI = ["2024-01-11", "2024-02-13", "2024-03-12", "2024-04-10", "2024-05-15", "2024-06-12", "2024-07-11",
       "2024-08-14", "2024-09-11", "2024-10-10", "2024-11-13", "2024-12-11", "2025-01-15", "2025-02-12",
       "2025-03-12", "2025-04-10", "2025-05-13", "2025-06-11", "2025-07-15", "2025-08-12", "2025-09-11",
       "2025-10-24", "2025-12-18", "2026-01-13", "2026-02-13", "2026-03-11", "2026-04-10", "2026-05-12",
       "2026-06-10", "2026-07-14", "2026-08-12", "2026-09-11", "2026-10-14", "2026-11-10", "2026-12-10", "2027-01-13", "2027-02-11", "2027-03-10", "2027-04-13", "2027-05-12",
       "2027-06-10", "2027-07-14", "2027-08-11", "2027-09-14", "2027-10-14", "2027-11-10", "2027-12-10"]


@lru_cache(maxsize=64)
def _holidays(y: int) -> frozenset:
    from .etf import nyse_holidays
    return frozenset(nyse_holidays(y))


def is_trading_day(d: dt.date) -> bool:
    return d.weekday() < 5 and d not in _holidays(d.year)


def trading_days_after(start: dt.date, n: int) -> list[dt.date]:
    out, d = [], start
    while len(out) < n:
        d += dt.timedelta(days=1)
        if is_trading_day(d):
            out.append(d)
    return out


def next_trading_day_on_or_after(d: dt.date) -> dt.date:
    while not is_trading_day(d):
        d += dt.timedelta(days=1)
    return d


def early_close(d: dt.date) -> bool:
    """NYSE closes at 1pm on July 3 (when it trades), the day after Thanksgiving and Christmas Eve."""
    if not is_trading_day(d):
        return False
    if (d.month, d.day) in ((7, 3), (12, 24)):
        return True
    if d.month == 11 and d.weekday() == 4:
        thanksgiving = dt.date(d.year, 11, 1) + dt.timedelta(days=(3 - dt.date(d.year, 11, 1).weekday()) % 7 + 21)
        return d == thanksgiving + dt.timedelta(days=1)
    return False


def close_hm(d: dt.date) -> tuple[int, int]:
    return (13, 0) if early_close(d) else (16, 0)


def last_complete_session(now_et: dt.datetime) -> dt.date:
    """The latest trading day whose closing price is final (15 minutes after the close)."""
    d = now_et.date()
    ch, cm = close_hm(d)
    if is_trading_day(d) and (now_et.hour, now_et.minute) >= (ch, cm + 15):
        return d
    d -= dt.timedelta(days=1)
    while not is_trading_day(d):
        d -= dt.timedelta(days=1)
    return d


def reference_session(made_at: dt.datetime) -> dt.date:
    """For a forecast written at `made_at` (ET): the first session whose close comes after it. Grading Claude
    from that close means news it read after the last close (an after-hours earnings release, say) can't make
    its short-term calls look better than they are."""
    d = made_at.date()
    if is_trading_day(d) and (made_at.hour, made_at.minute) < close_hm(d):
        return d
    return trading_days_after(d, 1)[0]


# Ratios no ordinary trading day produces by itself. 5-for-4, 4-for-3 and 3-for-2 splits look like −20%, −25%
# and −33% days, which earnings reactions produce all the time, so those count only when the split registry
# confirms them (split_near).
SPLIT_RATIOS = (2, 2.5, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50, 100)


def split_like(r: float, tol: float = 0.02) -> bool:
    """Is this one-day log return what an un-adjusted stock split would look like (−log 2 for 2-for-1, +log 10
    for a 1-for-10 reverse split…)? Used to hold off forecasting or grading until the history is re-based."""
    if not math.isfinite(r) or abs(r) < 0.2:
        return False
    return any(abs(abs(r) - math.log(k)) < tol for k in SPLIT_RATIOS)


def split_near(events: list | None, day: dt.date, r: float, tol: float = 0.05) -> bool:
    """Does the split registry ({ticker: [[execution date, share factor]]}) have a split within a few days of
    `day` whose factor explains the one-day log return `r` (a 2-for-1 split, factor 2, halves the price)?"""
    if not events or not math.isfinite(r):
        return False
    for ed, k in events:
        try:
            d = dt.date.fromisoformat(str(ed)[:10])
        except ValueError:
            continue
        if k and k > 0 and -4 <= (day - d).days <= 4 and abs(r + math.log(float(k))) < tol:
            return True
    return False


def seed_for(t: str, made: str) -> int:
    return int(hashlib.sha256(f"{t}:{made}".encode()).hexdigest()[:12], 16)


# ---------------------------------------------------------------- volatility & shocks

def ewma_path(r: np.ndarray, skip: set[int] | None = None, lam: float = 0.94) -> tuple[np.ndarray, float]:
    """(variance known before each day, variance for the next day). Days in `skip` don't update it."""
    skip = skip or set()
    base = [x for i, x in enumerate(r[:40]) if i not in skip]
    v = float(np.var(base)) if len(base) >= 10 else float(np.var(r))
    v = max(v, 1e-8)
    pred = np.empty(len(r))
    for i, x in enumerate(r):
        pred[i] = v
        if i not in skip:
            v = lam * v + (1 - lam) * float(x) * float(x)
            v = max(v, 1e-8)
    return pred, v


def earnings_reaction_idx(r_dates: list[int], past: list[str], r: np.ndarray) -> set[int]:
    """Index of the reaction day for each past earnings release (that day or the next, whichever moved more)."""
    import bisect
    out = set()
    for s in past:
        try:
            o = dt.date.fromisoformat(s[:10]).toordinal()
        except ValueError:
            continue
        i = bisect.bisect_left(r_dates, o)
        cands = [k for k in (i, i + 1) if 0 <= k < len(r)]
        if cands and r_dates[cands[0]] - o <= 4:
            out.add(max(cands, key=lambda k: abs(r[k])))
    return out


def _interp_calib(calib: dict | None, k: int, key: str, default: float) -> float:
    """Calibration learned per horizon, interpolated in log-horizon for the days in between."""
    if not calib:
        return default
    pts = sorted((h, float(c.get(key, default))) for name, h in HORIZONS for c in [calib.get(name) or {}]
                 if c.get(key) is not None)
    if not pts:
        return default
    if k <= pts[0][0]:
        return pts[0][1] if pts[0][0] == 1 or key != "shift" else pts[0][1] * k / pts[0][0]
    if k >= pts[-1][0]:                    # beyond what grading has measured: fade the correction out
        h_last, v_last = pts[-1]
        w = max(0.0, 1.0 - math.log2(k / h_last) / 2)
        return default + (v_last - default) * w
    for (h0, v0), (h1, v1) in zip(pts, pts[1:]):
        if h0 <= k <= h1:
            w = (math.log(k) - math.log(h0)) / (math.log(h1) - math.log(h0))
            return v0 + (v1 - v0) * w
    return default


def _sig(x: float, n: int = 4) -> float:
    if x == 0 or not math.isfinite(x):
        return 0.0
    return round(x, max(0, n - 1 - int(math.floor(math.log10(abs(x))))))


# ---------------------------------------------------------------- the forecast

def build(t: str, dates: list[int], closes: list[float], *, beta: float | None = None, rf: float = 0.04,
          erp: float = 0.05, earnings_next: dict | None = None, past_earnings: list[str] | None = None,
          macro: list[tuple[str, str]] | None = None, calib: dict | None = None, tilt_ret: float = 0.0,
          tilt_days: int = 21, tilt_parts: list | None = None, div_yield: float | None = None, paths: int = 2000,
          half_life: float = 40.0, mults: dict | None = None, keep_paths: bool = False) -> dict | None:
    """One stock's forecast from its daily closes (oldest first, last = the anchor session's close).

    `tilt_ret`: a learned log-return nudge over the next `tilt_days` sessions (only what was tested; nothing is
    extrapolated beyond it). `div_yield`: trailing dividend yield; prices drop by the dividend on ex-dates and the
    history is not dividend-adjusted, so price drift = cash rate + beta × premium − yield."""
    c = np.asarray(closes, dtype=float)
    ok = np.isfinite(c) & (c > 0)
    if ok.sum() < 130:
        return None
    dd = [d for d, k in zip(dates, ok) if k][-760:]
    c = c[ok][-760:]
    r = np.diff(np.log(c))
    r_dates = dd[1:]
    anchor = dt.date.fromordinal(int(dd[-1]))
    mults = {"fomc": 1.25, "cpi": 1.15, **(mults or {})}
    eidx = earnings_reaction_idx(r_dates, past_earnings or [], r)
    pred, v_next = ewma_path(r, eidx)
    normal = np.asarray([i for i in range(len(r)) if i not in eidx])
    z = r[normal] / np.sqrt(pred[normal])
    z = np.clip(z, -8, 8)
    z = z - z.mean()                     # keep its natural scale: it carries the volatility-estimate noise
    ez = [abs(r[i]) / math.sqrt(pred[i]) for i in sorted(eidx)]
    m_e = float(np.clip(math.sqrt(np.mean(np.square(ez))), 1.5, 6.0)) if len(ez) >= 3 else 3.0
    rr = r[normal][-504:]
    cap = np.quantile(np.abs(rr), 0.995) if len(rr) > 50 else np.max(np.abs(rr))
    v_lr = float(np.mean(np.square(np.clip(rr, -cap, cap))))
    v_lr = max(v_lr, 1e-8)
    phi = 0.5 ** (1.0 / half_life)
    k = np.arange(H_MAX)
    var = v_lr + (phi ** k) * (v_next - v_lr)                     # day 1..H variance before events
    fut = trading_days_after(anchor, H_MAX)
    fut_ix = {d: i for i, d in enumerate(fut)}
    mult = np.ones(H_MAX)
    events: list[dict] = []
    # earnings: the announced date if we have it, else every ~91 days after the last release (estimated)
    e_days: list[tuple[dt.date, bool, str]] = []
    if earnings_next and earnings_next.get("date"):
        try:
            ed = dt.date.fromisoformat(earnings_next["date"][:10])
            pre = "pre" in str(earnings_next.get("time") or "").lower()
            react = next_trading_day_on_or_after(ed) if pre else trading_days_after(ed, 1)[0]
            if react > anchor:                # a report after the anchor day's close moves the next session
                e_days.append((react, False, ed.isoformat()))
        except (ValueError, IndexError):
            pass
    if past_earnings:
        try:
            last = max(dt.date.fromisoformat(s[:10]) for s in past_earnings)
        except ValueError:
            last = None
        if last is not None and (anchor - last).days <= 150:
            for j in range(1, 6):
                ed = last + dt.timedelta(days=91 * j)
                react = trading_days_after(ed, 1)[0]
                if react <= anchor or (e_days and abs((ed - dt.date.fromisoformat(e_days[0][2])).days) < 45):
                    continue
                if (ed - anchor).days > 370:
                    break
                e_days.append((react, True, ed.isoformat()))
    for react, est, ed in sorted(e_days):
        i = fut_ix.get(react)
        if i is None:
            continue
        mult[i] = max(mult[i], m_e ** 2)
        events.append({"date": react.isoformat(), "kind": "earnings", "estimated": est, "announce": ed,
                       "label": ("Earnings (estimated date)" if est else "Earnings"), "i": i})
    for d, kind in (macro or []):
        try:
            dd_ = dt.date.fromisoformat(d)
        except ValueError:
            continue
        i = fut_ix.get(dd_)
        if i is None or kind not in mults:
            continue
        mult[i] = max(mult[i], mults[kind])
        events.append({"date": d, "kind": kind, "label": {"fomc": "Fed rate decision", "cpi": "CPI inflation report"}[kind],
                       "i": i})
    b = 1.0 if beta is None or not math.isfinite(beta) else float(np.clip(beta, 0.3, 2.5))
    dy = 0.0 if div_yield is None or not math.isfinite(div_yield) else float(np.clip(div_yield, 0.0, 0.15))
    mu_ann = rf + b * erp - dy
    v_tot = var * mult
    drift = mu_ann / 252.0 - 0.5 * v_tot
    td = max(1, min(int(tilt_days or 21), H_MAX))
    tr_ = float(tilt_ret) if math.isfinite(float(tilt_ret or 0.0)) else 0.0
    if tr_:
        drift = drift.copy()
        drift[:td] += tr_ / td
    rng = np.random.default_rng(seed_for(t, anchor.isoformat()))
    Z = z[rng.integers(0, len(z), size=(paths, H_MAX))]
    X_raw = np.cumsum(drift[None, :] + Z * np.sqrt(v_tot)[None, :], axis=1)
    med_raw = np.median(X_raw, axis=0)
    scale = np.asarray([_interp_calib(calib, kk + 1, "scale", 1.0) for kk in range(H_MAX)])
    shift = np.asarray([_interp_calib(calib, kk + 1, "shift", 0.0) for kk in range(H_MAX)])
    X = med_raw[None, :] + shift[None, :] + scale[None, :] * (X_raw - med_raw[None, :])
    p0 = float(c[-1])
    # fan for the chart: daily for a month, then thinner
    times = list(range(1, 22)) + list(range(25, 64, 4)) + list(range(73, 253, 10))
    if times[-1] != H_MAX:
        times.append(H_MAX)
    qs = np.quantile(X[:, [x - 1 for x in times]], TAUS, axis=0)        # (7, len(times))
    fan = {"dates": [anchor.isoformat()] + [fut[x - 1].isoformat() for x in times],
           "p": {QK[j]: [_sig(p0)] + [_sig(p0 * math.exp(v)) for v in qs[j]] for j in range(len(QK))}}
    run_max = np.maximum.accumulate(np.maximum(X, 0.0), axis=1)
    run_min = np.minimum.accumulate(np.minimum(X, 0.0), axis=1)
    peak = np.maximum.accumulate(np.concatenate([np.zeros((paths, 1)), X], axis=1), axis=1)[:, 1:]
    dd_path = np.maximum.accumulate(peak - X, axis=1)
    hz = {}
    for name, h in HORIZONS:
        xh, xr = X[:, h - 1], X_raw[:, h - 1]
        q = np.quantile(xh, TAUS)
        qr = np.quantile(xr, TAUS)
        hi, lo, mdd = run_max[:, h - 1], run_min[:, h - 1], dd_path[:, h - 1]
        touch = {}
        for x in (0.05, 0.10, 0.20, 0.30):
            touch[f"+{int(x*100)}"] = float(np.mean(hi >= math.log(1 + x)))
            touch[f"-{int(x*100)}"] = float(np.mean(lo <= math.log(1 - x)))
        hz[name] = {
            "h": h, "date": fut[h - 1].isoformat(),
            "q": {QK[j]: _sig(p0 * math.exp(q[j])) for j in range(len(QK))},
            "lq": [round(float(v), 5) for v in q], "lq_raw": [round(float(v), 5) for v in qr],
            "p_up": round(float(np.mean(xh > 0)), 4), "exp_ret": round(float(np.mean(np.exp(xh)) - 1), 5),
            "hi_med": round(float(math.expm1(np.median(hi))), 4), "lo_med": round(float(math.expm1(np.median(lo))), 4),
            "hi_p90": round(float(math.expm1(np.quantile(hi, 0.9))), 4), "lo_p10": round(float(math.expm1(np.quantile(lo, 0.1))), 4),
            "touch": {k_: round(v_, 4) for k_, v_ in touch.items()},
            "mdd_med": round(float(-math.expm1(-np.median(mdd))), 4), "mdd_p90": round(float(-math.expm1(-np.quantile(mdd, 0.9))), 4),
            "p_dd20": round(float(np.mean(mdd >= -math.log(0.8))), 4),
            "sigma": round(float(np.std(xh)), 5),
            "tilt": round(tr_ * min(h, td) / td, 5),
        }
    # where the risk sits over the next 3 months, and whether the low tends to come right after an event
    w3 = v_tot[:63].sum()
    argmin3 = np.argmin(np.concatenate([np.zeros((paths, 1)), X[:, :63]], axis=1), axis=1) - 1   # -1 = today
    for ev in events:
        i = ev.pop("i")
        if i < 63:
            ev["risk_share_3m"] = round(float(v_tot[i] / w3), 4)
            if ev["kind"] == "earnings":
                ev["p_low_near"] = round(float(np.mean((argmin3 >= i - 1) & (argmin3 <= i + 2))), 4)
    events.sort(key=lambda e: e["date"])
    naive = float(np.std(r[-252:], ddof=1))
    out = {
        "t": t, "model": MODEL, "made": anchor.isoformat(), "anchor": _sig(p0, 6), "paths": paths,
        "vol_now": round(math.sqrt(v_next * 252), 4), "vol_lr": round(math.sqrt(v_lr * 252), 4),
        "earn_mult": round(m_e, 2), "earn_samples": len(ez), "beta": round(b, 3), "drift_ann": round(mu_ann, 4),
        "div_yield": round(dy, 4),
        "tilt": {"ret": round(tr_, 5), "days": td, "parts": tilt_parts or []},
        "calib": {name: {"scale": round(_interp_calib(calib, h, "scale", 1.0), 3),
                         "shift": round(_interp_calib(calib, h, "shift", 0.0), 4)} for name, h in HORIZONS},
        "naive_sigma_d": round(naive, 6), "fan": fan, "horizons": hz, "events": events,
        "crash": {"p_dd40_1y": round(float(np.mean(run_min[:, 251] <= math.log(0.6))), 4),
                  "p_dd20_3m": round(float(np.mean(run_min[:, 62] <= math.log(0.8))), 4)},
        "low_timing_3m": _timing(argmin3, 63),
    }
    if keep_paths:
        out["_X"] = X
    return out


def _timing(argmin: np.ndarray, h: int) -> dict:
    """Share of paths whose lowest point in the window falls in each third of it (plus 'right away')."""
    a = np.asarray(argmin)
    thirds = [int(h / 3), int(2 * h / 3)]
    return {"now": round(float(np.mean(a < 0)), 3), "early": round(float(np.mean((a >= 0) & (a < thirds[0]))), 3),
            "middle": round(float(np.mean((a >= thirds[0]) & (a < thirds[1]))), 3),
            "late": round(float(np.mean(a >= thirds[1])), 3)}


# ---------------------------------------------------------------- other forecasters in the same format

def naive_lq(sigma_d: float, h: int) -> list[float]:
    """The benchmark every forecast is scored against: a random walk with no drift and last year's plain
    volatility (a normal distribution)."""
    s = sigma_d * math.sqrt(h)
    return [round(float(z * s), 5) for z in Z_TAUS]


def from_p10_p50_p90(lo: float, mid: float, hi: float, ref: float) -> list[float] | None:
    """Price quantiles (10th, 50th, 90th) → the 7 log-return quantiles, with a split-normal in log space.
    Returns None for an unusable answer (non-positive prices or out-of-order bands)."""
    if not all(isinstance(x, (int, float)) and x > 0 for x in (lo, mid, hi, ref)) or not (lo < mid < hi):
        return None
    l10, l50, l90 = math.log(lo / ref), math.log(mid / ref), math.log(hi / ref)
    z90 = _N.inv_cdf(0.9)
    s_lo, s_hi = max((l50 - l10) / z90, 1e-4), max((l90 - l50) / z90, 1e-4)
    return [round(l50 + (s_lo if z < 0 else s_hi) * z, 5) for z in Z_TAUS]


def blend(lq_a: list[float], lq_b: list[float], w_a: float) -> list[float]:
    """Quantile averaging (Vincentization): keeps the result a valid, ordered distribution."""
    return [round(w_a * a + (1 - w_a) * b, 5) for a, b in zip(lq_a, lq_b)]


def macro_events(extra: list | None = None) -> list[tuple[str, str]]:
    ev = [(d, "fomc") for d in FOMC] + [(d, "cpi") for d in CPI]
    for x in extra or []:
        if isinstance(x, dict) and x.get("date") and x.get("kind") in ("fomc", "cpi"):
            ev.append((str(x["date"])[:10], x["kind"]))
    return sorted(set(ev))
