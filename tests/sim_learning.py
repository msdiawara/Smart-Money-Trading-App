"""Audit harness: does the app's forecast learning loop make forecasts more accurate as graded data builds up?

Runs the app's own code (forecast.build, fledger.record/grade/add_backtest, learn.calibration/tilt_weights/
tilt_gate/tilt_for/features) day by day over a simulated market, exactly in the order the pipeline uses them:
backtest rows from the history first, then each day: grade, learn (calibration daily, signal tilt weekly with the
two-pass gate), forecast every stock with what has been learned, record to the ledger.

Every forecast is also made without any learning (no calibration, no tilt) and both are scored on the same
outcomes, so the difference is what learning added. The market index is recorded too (as the pipeline records the
S&P 500) but not scored, and the ledger is compacted daily with each version's own retention rules.

The simulated market has the problems real markets have: a market factor, volatility clustering, fat tails,
earnings jumps the model isn't told about, and a step up in volatility after the history period. A planted
cross-sectional 1-month reversal effect of strength K (0 = none) tests whether the signal learner finds a real
effect and stays off when there isn't one.

Usage: python3 tests/sim_learning.py <repo_root> <K> <out.json> [stocks] [days] [paths] [seed] [jump_x] [vol_step] [nonlinear]
       python3 tests/sim_learning_report.py <out.json> ...   (5-30 minutes per run with the defaults)
"""
import datetime as dt
import json
import math
import sys
import time

ROOT, K, OUT = sys.argv[1], float(sys.argv[2]), sys.argv[3]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 36
DAYS = int(sys.argv[5]) if len(sys.argv) > 5 else 400
PATHS = int(sys.argv[6]) if len(sys.argv) > 6 else 500
SEED = int(sys.argv[7]) if len(sys.argv) > 7 else 7
JUMPX = float(sys.argv[8]) if len(sys.argv) > 8 else 1.0     # earnings jumps this much bigger once the app runs
REGIME = float(sys.argv[9]) if len(sys.argv) > 9 else 1.25   # stock volatility this much higher once the app runs
NONLIN = int(sys.argv[10]) if len(sys.argv) > 10 else 0     # 1: the planted reversal works only in volatile stocks
sys.path.insert(0, ROOT)

import inspect  # noqa: E402

import numpy as np  # noqa: E402
from pipeline import fledger, forecast as F, learn  # noqa: E402

try:                                                                 # version 3: the algorithm competition
    from pipeline import ml  # noqa: E402
except ImportError:
    ml = None
V3 = ml is not None
NEW = V3 or "bench" in inspect.signature(learn.features).parameters  # version 2 or later

WARM = 760                       # history before the app starts (about three years)
EVAL_H = (5, 21, 63)
T = WARM + DAYS + 64

# ---------------------------------------------------------------- the simulated market
rng = np.random.default_rng(SEED)
days = []
d = dt.date(2022, 1, 3)
while len(days) < T:
    if F.is_trading_day(d):
        days.append(d)
    d += dt.timedelta(days=1)
ords = [x.toordinal() for x in days]

beta = rng.uniform(0.6, 1.6, N)
s_id = rng.uniform(0.012, 0.03, N)              # idiosyncratic daily vol (long run)
earn_off = rng.integers(0, 63, N)
t5 = lambda size: rng.standard_t(5, size) / math.sqrt(5 / 3)
t4 = lambda size: rng.standard_t(4, size) / math.sqrt(2)

logm = np.zeros(T)                              # the market index (the "S&P 500")
logp = np.zeros((T, N))
logp[0] = np.log(rng.uniform(20, 300, N))
vm = 0.010 ** 2
vi = s_id ** 2
idio_hist = np.zeros((T, N))                    # idiosyncratic log return per day (for the planted effect)
for t in range(1, T):
    regime = 1.0 if t < WARM else REGIME        # volatility steps up once the app is running
    lm, li = (0.010 * (1.0 if t < WARM else 1.15)) ** 2, (s_id * regime) ** 2
    em = math.sqrt(vm) * t5(1)[0]
    vm = 0.05 * lm + 0.90 * vm + 0.05 * em ** 2
    ei = np.sqrt(vi) * t4(N)
    vi = 0.06 * li + 0.89 * vi + 0.05 * ei ** 2
    jump = np.where((t + earn_off) % 63 == 0, rng.normal(0, 3 * s_id * regime * (1.0 if t < WARM else JUMPX)), 0.0)
    r21 = idio_hist[max(0, t - 21):t].sum(0)    # last month's idiosyncratic move
    k_i = K * (2.0 * (s_id > np.median(s_id)) if NONLIN else 1.0)   # nonlinear: only the more volatile half
    mu = 0.0003 - k_i * r21 / 21.0              # planted cross-sectional reversal (K = 0: none)
    idio_hist[t] = ei + jump
    logp[t] = logp[t - 1] + mu + beta * em + ei + jump - 0.5 * (beta ** 2 * vm + vi)
    logm[t] = logm[t - 1] + 0.0003 + em - 0.5 * vm

P = np.exp(logp)
M = 100 * np.exp(logm)
liq_level = rng.uniform(13, 18, N)                                  # log dollar volume per stock
VOL = np.exp(liq_level[None, :] + rng.normal(0, 0.3, (T, N)) + 3 * np.abs(np.diff(logp, axis=0, prepend=logp[:1]))) / P


def feats_at(i, j):
    hist = P[: i + 1, j].tolist()
    if V3:
        return learn.features([], None, None, None, None, hist, M[: i + 1].tolist(), VOL[: i + 1, j].tolist())
    if NEW:
        return learn.features([], None, None, None, None, hist, M[: i + 1].tolist())
    return learn.features([], None, None, None, None, hist)


def bt_feats_at(i, j):
    if V3:
        f = ml.price_features(P[: i + 1, j].tolist(), M[: i + 1].tolist(), VOL[: i + 1, j].tolist())
        return f if "rev" in f else None
    if NEW:
        f = learn.price_features(P[: i + 1, j].tolist(), M[: i + 1].tolist())
        return f if "mom" in f else None
    return None
names = [f"S{i:02d}" for i in range(N)]


BENCH = "SPY"                    # the market index, recorded like the pipeline records the S&P 500 (not scored)


def series_upto(i_now):
    def get(tk):
        if tk == BENCH:
            return ords[: i_now + 1], M[: i_now + 1]
        j = names.index(tk)
        return ords[: i_now + 1], P[: i_now + 1, j]
    return get


def bench_feats_at(i):
    if V3:
        return learn.features([], None, None, None, None, M[: i + 1].tolist(), M[: i + 1].tolist(), None)
    if NEW:
        return learn.features([], None, None, None, None, M[: i + 1].tolist(), M[: i + 1].tolist())
    return learn.features([], None, None, None, None, M[: i + 1].tolist())


def scores(lq, r, nv):
    return {"in80": int(lq[1] <= r <= lq[5]), "qs": fledger.qscore(lq, r), "qsn": fledger.qscore(nv, r)}


# ---------------------------------------------------------------- the app, as the pipeline runs it
L = {"v": 1, "pending": {}, "graded": {}, "agg": {}, "last": {}}
t0 = time.time()

# walk-forward backtest over the history (like job_backtest: weekly from version 3, every 10 sessions before;
# outcomes already known)
for i in range(260, WARM - 1, 5 if V3 else 10):
    for j, tk in enumerate(names + [BENCH]):
        bench = tk == BENCH
        hist_i = (M[: i + 1] if bench else P[: i + 1, j]).tolist()
        fc = F.build(tk, ords[: i + 1], hist_i, beta=1.0 if bench else float(beta[j]), paths=PATHS)
        if not fc:
            continue
        feats = None if bench else bt_feats_at(i, j)
        for name, h in F.HORIZONS[:4]:
            k = i + h
            if k >= WARM:
                continue
            seg = M[i:k + 1] if bench else P[i:k + 1, j]
            r = math.log(seg[-1] / seg[0])
            path = np.log(seg / seg[0])
            dd = float(1 - math.exp(-float(np.max(np.maximum.accumulate(path) - path))))
            hz = fc["horizons"][name]
            args = (L, tk, fc["made"], h, hz["lq"], hz["lq_raw"], hz["p_up"], F.naive_lq(fc["naive_sigma_d"], h), r, dd,
                    days[WARM - 1], float(seg[0]), float(seg[-1]), days[k])
            if V3:
                fledger.add_backtest(*args, feats=ml.row_features(feats, fc, name) if feats else None)
            elif NEW:
                fledger.add_backtest(*args, feats=feats if h == 21 else None)
            else:
                fledger.add_backtest(*args)
bt_secs = time.time() - t0

state = {"calib": {}, "tilt": None, "tilt_at": None, "range": None}
evals, timeline = [], []
pending_eval = []                    # (due index, h, version, lq, nv, i, j)
for i in range(WARM, WARM + DAYS):
    today = days[i]
    # 1. grade and learn (the pipeline does this every 6 hours; once a day here)
    fledger.grade(L, series_upto(i), market_last=ords[i], today=today, last_complete=today)
    fledger.compact(L, today)                      # what saving the ledger does after every pipeline run
    calib = learn.calibration(L, min_n=200, min_dates=20)
    tilt = state["tilt"]
    if state["tilt_at"] is None or (today - state["tilt_at"]).days >= 7:
        tilt = learn.tilt_gate(tilt, learn.tilt_weights(L))
        state["tilt_at"] = today
    rl = (ml.range_learner(L, calib, prev=state["range"]) if "prev" in inspect.signature(ml.range_learner).parameters
          else ml.range_learner(L, calib)) if V3 else None
    state.update(calib=calib, tilt=tilt, range=rl)
    applied = {k: v for k, v in calib.items() if v.get("status") == "applied"}
    smodel = ml.SignalModel.build(tilt, L) if V3 else None
    rmodel = ml.RangeModel.build(rl, calib, L) if V3 else None
    day_feats = [feats_at(i, j) for j in range(N)]
    if smodel is not None and hasattr(smodel, "set_day"):
        smodel.set_day(day_feats)                  # centred on the day's stocks, as the pipeline does
    # 2. forecast every stock with what has been learned, and without (for comparison); record the learned one
    bfc = F.build(BENCH, ords[: i + 1], M[: i + 1].tolist(), beta=1.0, paths=PATHS, calib=applied)
    if bfc:
        bf = bench_feats_at(i)
        for name, h in F.HORIZONS:
            hz = bfc["horizons"][name]
            fledger.record(L, "quant", BENCH, bfc["made"], h, hz["lq"], lq_raw=hz["lq_raw"], pup=hz["p_up"],
                           naive=F.naive_lq(bfc["naive_sigma_d"], h),
                           feats=ml.row_features(bf, bfc, name) if V3 else bf)
    for j, tk in enumerate(names):
        hist = P[: i + 1, j].tolist()
        feats = day_feats[j]
        if V3:
            def build(calib, tilt_ret, tilt_days, tilt_parts, tk=tk, hist=hist, j=j):
                return F.build(tk, ords[: i + 1], hist, beta=float(beta[j]), paths=PATHS, calib=calib,
                               tilt_ret=tilt_ret, tilt_days=tilt_days, tilt_parts=tilt_parts)
            fc, tr = ml.forecast_with_learning(build, feats, applied, smodel, rmodel)
        else:
            tr, parts, td = learn.tilt_for(tilt, feats)
            fc = F.build(tk, ords[: i + 1], hist, beta=float(beta[j]), paths=PATHS, calib=applied, tilt_ret=tr, tilt_days=td)
        if not fc:
            continue
        base = fc if not tr else F.build(tk, ords[: i + 1], hist, beta=float(beta[j]), paths=PATHS)
        for name, h in F.HORIZONS:
            hz = fc["horizons"][name]
            extra = {"tl": hz["tilt"]} if hz.get("tilt") else None
            fledger.record(L, "quant", tk, fc["made"], h, hz["lq"], lq_raw=hz["lq_raw"], pup=hz["p_up"],
                           naive=F.naive_lq(fc["naive_sigma_d"], h),
                           feats=ml.row_features(feats, fc, name) if V3 else feats, extra=extra)
            if h in EVAL_H and i + h < T:
                nv = F.naive_lq(fc["naive_sigma_d"], h)
                pending_eval.append((i + h, h, "learn", hz["lq"], nv, i, j))
                pending_eval.append((i + h, h, "base", base["horizons"][name]["lq_raw"], nv, i, j))
    if (i - WARM) % 21 == 0:
        timeline.append({"day": i - WARM, "date": today.isoformat(),
                         "calib": {k: {x: v.get(x) for x in ("status", "from", "scale", "shift", "n", "dates")}
                                   for k, v in calib.items() if k in ("1w", "1m", "3m")},
                         "range": {k: {x: v.get(x) for x in ("status", "holdout_gain_vs_single", "scale_spread", "n")}
                                   for k, v in (rl or {}).items()},
                         "tilt": {x: (tilt or {}).get(x) for x in ("on", "status", "t", "test_gain", "recent_gain",
                                                                    "test_months", "n", "months", "weights", "model",
                                                                    "candidates", "best")}})
    if (i - WARM) % 50 == 0:
        print(f"K={K} day {i - WARM}/{DAYS} {time.time() - t0:.0f}s tilt_on={bool((tilt or {}).get('on'))}", flush=True)

# 3. score every forecast whose outcome is inside the simulation
for due, h, ver, lq, nv, i, j in pending_eval:
    if due >= T:
        continue
    r = math.log(P[due, j] / P[i, j])
    evals.append({"m": (i - WARM) // 21, "h": h, "v": ver, **scores(lq, r, nv)})

json.dump({"K": K, "JUMPX": JUMPX, "REGIME": REGIME, "NONLIN": NONLIN, "V3": V3, "NEW": NEW, "N": N, "DAYS": DAYS, "PATHS": PATHS, "seed": SEED, "bt_secs": round(bt_secs, 1),
           "secs": round(time.time() - t0, 1), "evals": evals, "timeline": timeline}, open(OUT, "w"))
print("done", OUT, round(time.time() - t0), "s")
