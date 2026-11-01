"""Machine learning for the price forecasts: which algorithms compete, how they're tested, how they're used.

Two learners. Neither touches a forecast until it has improved forecasts made after the data it learned from.

1. The signal learner nudges a forecast's median by how the stock is likely to do relative to the other stocks
   forecast the same day. Four candidates compete: ridge regression, gradient-boosted trees, a neural network (the
   average of three small networks), and the average of those three. In the largest study of machine learning
   for stock returns (Gu, Kelly & Xiu 2020), trees and neural networks beat linear models given decades of data on
   thousands of stocks; with less data, heavily regularized linear models are often as good or better, and averaging
   models is a robust default. So the app doesn't assume a winner. Each candidate is refit month by month on
   earlier months only and scored on the next; the candidate used for each month is the one that had scored best
   on the months before it, so even the choice of algorithm is tested on data it didn't see; and that whole
   procedure must pass a significance test before it nudges anything. Predictions are shrunk toward zero by
   independent months/(months + 6), so with little evidence they say little.
2. The range learner (conditional calibration) learns, per horizon, how much a forecast's 10–90% range needed
   stretching to hold the outcome, as a function of the stock's situation (volatility level, recent volatility
   versus its usual and versus the market, the biggest recent daily move, distance from the 52-week high,
   liquidity, beta, an earnings report inside the window), with gradient-boosted quantile trees. It replaces the
   single per-horizon stretch only if it scored better on the newest forecasts it wasn't fitted on.

Features are computed the same way for live forecasts and for the walk-forward backtest (prices and volumes known
on the anchor day), so both learners have data from the first day. Fitted models are never stored: they are refit
deterministically from the graded record when needed, so what is used is exactly what was tested.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import math
import time
import warnings

import numpy as np

from .prices import BENCH

FEAT_V = 3
PRICE_FEATS = ["mom", "rev", "rev_w", "hi52", "lvol", "vol_rel", "vol_chg", "maxret", "beta", "liq"]
SIGNAL_FEATS = ["sig", "val", "claude", "ins", "bub"]
FEATS = PRICE_FEATS + SIGNAL_FEATS
RANGE_FEATS = ["lvol", "vol_rel", "vol_chg", "maxret", "hi52", "liq", "beta", "ein"]
LABELS = {"mom": "12-month momentum vs the S&P 500", "rev": "last month's move vs the S&P 500",
          "rev_w": "last week's move vs the S&P 500", "hi52": "distance from its 52-week high",
          "lvol": "its volatility", "vol_rel": "its volatility vs the market's", "vol_chg": "recent vs usual volatility",
          "maxret": "its biggest daily jump last month", "beta": "its beta", "liq": "trading volume (liquidity)",
          "sig": "smart-money signals", "val": "valuation", "claude": "Claude's stance", "ins": "insider net buying",
          "bub": "bubble-risk score", "ein": "an earnings report in the window"}
CANDIDATES = ["ridge", "trees", "neural_net", "average"]
NAMES = {"ridge": "Ridge regression", "trees": "Gradient-boosted trees", "neural_net": "Neural network",
         "average": "Average of all three", "none": "None"}
MAX_ROWS = 20000                 # training rows per fit (a fixed sample by stock and date when there are more)
HP_V = 2                         # the candidates' settings: bump when they change (stored test results are redone)


# ---------------------------------------------------------------- features

def _clean(a) -> np.ndarray:
    return np.asarray([np.nan if x is None else float(x) for x in a], dtype=float)


def price_features(closes, bench, volumes=None) -> dict:
    """Price and volume features known on the anchor day (`closes` oldest first, last = the anchor close; `bench`:
    the S&P 500's closes on the same dates; `volumes` aligned with `closes`). Returns relative to the benchmark
    where the market's own move would otherwise dominate."""
    f: dict = {"v": FEAT_V}
    if closes is None or bench is None or len(closes) != len(bench) or len(closes) < 64:
        return f
    c, b = _clean(closes[-253:]), _clean(bench[-253:])
    ok = np.isfinite(c) & (c > 0) & np.isfinite(b) & (b > 0)
    if not ok[-64:].all():
        return f
    rc, rb = np.diff(np.log(c[-64:])), np.diff(np.log(b[-64:]))
    f["rev"] = math.log(c[-1] / c[-22]) - math.log(b[-1] / b[-22])
    f["rev_w"] = math.log(c[-1] / c[-6]) - math.log(b[-1] / b[-6])
    s21, b21 = float(np.std(rc[-21:])), float(np.std(rb[-21:]))
    if s21 > 0:
        f["lvol"] = math.log(s21 * math.sqrt(252))
        if b21 > 0:
            f["vol_rel"] = math.log(s21 / b21)
    f["maxret"] = float(np.max(rc[-21:]))
    if len(c) >= 253 and ok.all():
        lc, lb = np.log(c), np.log(b)
        f["mom"] = (lc[-22] - lc[-253]) - (lb[-22] - lb[-253])
        f["hi52"] = float(lc[-1] - np.max(lc[-252:]))
        ra, rba = np.diff(lc), np.diff(lb)
        s252 = float(np.std(ra))
        if s252 > 0 and s21 > 0:
            f["vol_chg"] = math.log(s21 / s252)
        vb = float(np.var(rba))
        if vb > 0:
            f["beta"] = float(np.clip(np.cov(ra, rba, bias=True)[0, 1] / vb, -1.0, 4.0))
    if volumes is not None and len(volumes) == len(closes):
        v = _clean(volumes[-21:])
        dv = c[-21:] * v
        dv = dv[np.isfinite(dv) & (dv > 0)]
        if len(dv) >= 10:
            f["liq"] = math.log10(float(np.median(dv)))
    return f


def signal_features(e_scores: list[dict], val: dict | None, bub: dict | None, claude_stance: str | None) -> dict:
    f: dict = {}
    s = sum(((x.get("score") or 0) / 100.0) * (1 if x.get("direction") == "bullish" else -1 if x.get("direction") == "bearish" else 0)
            for x in e_scores)
    f["sig"] = max(-2.0, min(2.0, s))
    if val and val.get("rich_pctl") is not None:
        f["val"] = val["rich_pctl"] / 100 - 0.5
    f["claude"] = {"favorable": 1.0, "watch": 0.5, "neutral": 0.0, "unfavorable": -1.0}.get(claude_stance or "", 0.0)
    ins_b = sum(1 for x in e_scores if x.get("kind") == "insider" and x.get("action") == "buy" and x.get("signal") != "noise")
    ins_s = sum(1 for x in e_scores if x.get("kind") == "insider" and x.get("action") in ("sell", "sale_notice"))
    f["ins"] = max(-3, min(3, ins_b - ins_s)) / 3
    if bub and bub.get("score") is not None:
        f["bub"] = min(1.0, bub["score"] / 10)
    return f


def earn_in(fc: dict, name: str) -> int:
    """1 if an earnings reaction day falls inside this horizon of the forecast."""
    hz = (fc.get("horizons") or {}).get(name) or {}
    end = hz.get("date")
    return int(bool(end) and any(e.get("kind") == "earnings" and e.get("date") and e["date"] <= end
                                 for e in fc.get("events") or []))


def row_features(feats: dict, fc: dict, name: str) -> dict:
    """What is stored with one horizon's forecast: the stock's features plus that horizon's earnings flag."""
    return {**feats, "ein": earn_in(fc, name)}


# ---------------------------------------------------------------- shared helpers

def block_days(h: int) -> int:
    """Length of an independent period for horizon h: one horizon (in calendar days), and at least a month."""
    return max(30, int(math.ceil(h * 7 / 5)))


def block(made: str, h: int) -> int:
    return dt.date.fromisoformat(str(made)[:10]).toordinal() // block_days(h)


def periods(made, h: int) -> int:
    """Independent periods covered by a set of forecast dates: period-length blocks counted from the first date,
    so the count grows steadily as days are added (blocks fixed to the calendar flip between counts as a rolling
    window moves) and stretches with no forecasts don't count."""
    ds = sorted({str(m)[:10] for m in made})
    if not ds:
        return 0
    d0, B = dt.date.fromisoformat(ds[0]).toordinal(), block_days(h)
    return len({(dt.date.fromisoformat(d).toordinal() - d0) // B for d in ds})


def _hash_order(keys: list[str]) -> np.ndarray:
    return np.asarray([int(hashlib.sha1(k.encode()).hexdigest()[:8], 16) for k in keys], dtype=np.int64)


def _sample(idx: np.ndarray, order: np.ndarray, cap: int = MAX_ROWS) -> np.ndarray:
    """A fixed sample of at most `cap` rows (the same rows every time the data is the same)."""
    if len(idx) <= cap:
        return idx
    return np.sort(idx[np.argsort(order[idx], kind="stable")[:cap]])


def _matrix(rows: list[dict], keys: list[str]) -> np.ndarray:
    return np.asarray([[float(r["f"][k]) if r["f"].get(k) is not None else np.nan for k in keys] for r in rows],
                      dtype=float)


def _date_demeaned(y: np.ndarray, made: np.ndarray) -> np.ndarray:
    out = y.astype(float).copy()
    for d in np.unique(made):
        m = made == d
        out[m] = y[m] - float(np.mean(y[m]))
    return out


def _quiet():
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", message=".*overflow.*")


# ---------------------------------------------------------------- the candidates

class Fitted:
    """One fitted candidate: predict(X) with missing values allowed. `shrink` scales its predictions toward 0."""

    def __init__(self, name: str, shrink: float, mu: np.ndarray, sd: np.ndarray, parts: dict, med: np.ndarray):
        self.name, self.shrink, self.mu, self.sd, self.parts, self.med = name, shrink, mu, sd, parts, med

    def _z(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mu) / self.sd
        Z[~np.isfinite(Z)] = 0.0
        return np.clip(Z, -5, 5)

    def _raw(self, kind: str, X: np.ndarray) -> np.ndarray:
        if kind == "ridge":
            return self._z(X) @ self.parts["ridge"]
        if kind == "trees":
            use = self.parts["use"]
            if self.parts["trees"] is None:
                return np.zeros(len(X))
            return self.parts["trees"].predict(X[:, use]) * self.shrink
        if kind == "neural_net":
            Z = self._z(X)
            return np.mean([m.predict(Z) for m in self.parts["neural_net"]], axis=0) * self.parts["ys"] * self.shrink
        raise ValueError(kind)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self.name == "average":
            return np.mean([self._raw(k, X) for k in ("ridge", "trees", "neural_net")], axis=0)
        return self._raw(self.name, X)


def _fit_ridge(Z: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    n, k = Z.shape
    A = np.hstack([np.ones((n, 1)), Z])
    P = np.eye(k + 1) * lam
    P[0, 0] = 0.0
    b = np.linalg.solve(A.T @ A + P, A.T @ y)
    return b[1:]


def fit(name: str, X: np.ndarray, y: np.ndarray, periods: int, b0: float = 6.0, seeds: int = 3) -> Fitted:
    """Fit one candidate ("ridge", "trees", "neural_net" or "average", which fits all three)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    _quiet()
    n, k = X.shape
    mu, sd, med = np.zeros(k), np.ones(k), np.zeros(k)
    for j in range(k):                          # missing values sit at the average (z = 0) for ridge and the network
        col = X[np.isfinite(X[:, j]), j]
        if len(col):
            mu[j], med[j] = float(np.mean(col)), float(np.median(col))
            if len(col) > 1 and float(np.std(col)) > 0:
                sd[j] = float(np.std(col))
    shrink = periods / (periods + b0)
    out = Fitted(name, shrink, mu, sd, {}, med)
    Z = out._z(X)
    need = ("ridge", "trees", "neural_net") if name == "average" else (name,)
    if "ridge" in need:
        # a weight keeps about periods/(periods + b0) of its unshrunk size: the penalty fades as evidence grows
        out.parts["ridge"] = _fit_ridge(Z, y, n * b0 / max(1, periods))
    if "trees" in need:
        # trees use the raw values (missing ones get their own branch); signals with no spread are left out
        use = np.asarray([j for j in range(k) if len(np.unique(X[np.isfinite(X[:, j]), j])) >= 2], dtype=int)
        out.parts["use"] = use
        out.parts["trees"] = HistGradientBoostingRegressor(
            max_depth=3, max_leaf_nodes=8, learning_rate=0.05, max_iter=60, min_samples_leaf=max(100, int(0.03 * n)),
            max_features=0.5, l2_regularization=1.0, early_stopping=False,
            random_state=0).fit(X[:, use], y) if len(use) else None
    if "neural_net" in need:
        ys = float(np.std(y)) or 1.0
        out.parts["ys"] = ys
        out.parts["neural_net"] = [MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1e-2, learning_rate_init=1e-3,
                                                max_iter=30, batch_size=min(512, n), random_state=s).fit(Z, y / ys)
                                   for s in range(seeds)]
    return out


# ---------------------------------------------------------------- the signal learner's competition

def rows_for(L: dict, h: int = 21) -> list[dict]:
    """Graded forecasts with this version's features: live ones and the walk-forward backtest's (not the S&P 500's
    own, which is the yardstick). Where the backtest re-ran a stock on a day that also has a live forecast, only the
    live one is used: the same outcome must not count twice."""
    rows, live = [], set()
    for src in ("quant", "quant_bt"):
        for r in L.get("graded", {}).get(src, {}).get(str(h), []):
            f = r.get("f")
            if f and f.get("v") == FEAT_V and "er" in r and r.get("t") != BENCH:
                if src == "quant":
                    live.add((r["t"], r["made"]))
                elif (r["t"], r["made"]) in live:
                    continue
                r.setdefault("er0", r["er"])      # the error against the median *before* any tilt (no feedback loop)
                rows.append(r)
    rows.sort(key=lambda r: r["made"])
    return rows


def _market_surprise(L: dict, h: int) -> dict[str, float]:
    """The S&P 500's own forecast error on each date (live first, then the backtest's)."""
    out: dict[str, float] = {}
    for src in ("quant_bt", "quant"):
        for r in L.get("graded", {}).get(src, {}).get(str(h), []):
            if r.get("t") == BENCH and "er" in r:
                out[r["made"]] = float(r.get("er0", r["er"]))
    return out


def targets(rows: list[dict], L: dict, h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(made dates, raw errors, learning target). The target is how a stock did *relative to the others*: its error,
    minus its beta times the market's own surprise that day (so beta can't learn one period's market move), minus
    the day's average. Stock-level signals can predict that; the market's move they can't."""
    made = np.asarray([r["made"] for r in rows])
    raw = np.asarray([r["er0"] for r in rows], dtype=float)
    mkt = _market_surprise(L, h)
    beta = np.asarray([float(r["f"]["beta"]) if r["f"].get("beta") is not None else 1.0 for r in rows])
    # A day without the index's own record (an older ledger, say): the market's surprise is estimated from that day's
    # stocks (their average error over their average beta). Treating it as zero would leave each stock's beta times
    # the market's move in the target, which the learner could then fit.
    by: dict[str, list[int]] = {}
    for i, m in enumerate(made):
        if m not in mkt:
            by.setdefault(m, []).append(i)
    for m, idx in by.items():
        mb = float(np.mean(beta[idx]))
        if len(idx) >= 10 and abs(mb) >= 0.3:
            mkt[m] = float(np.mean(raw[idx])) / mb
    adj = raw - beta * np.asarray([mkt.get(m, 0.0) for m in made])
    return made, raw, _date_demeaned(adj, made)


def _centered(p: np.ndarray, made: np.ndarray) -> np.ndarray:
    """Predictions minus their average across the stocks forecast the same day: a nudge says how one stock should
    do relative to the others, never which way the whole market goes."""
    return _date_demeaned(p, made)


def compete(L: dict, h: int = 21, min_n: int = 600, min_train_months: int = 6, min_test_months: int = 6,
            b0: float = 6.0, min_gain: float = 0.002, min_t: float = 2.5, seconds: float | None = None) -> dict:
    """The weekly competition. For each month (after `min_train_months`), every candidate is fitted on everything made
    before it (minus a gap the length of the horizon, so no outcome overlaps) and scored on that month: the reduction
    in squared error of how each stock did relative to the others that day. The candidate *used* for a month is the
    one with the best average on the months before it (the average of all three until three months are scored).
    That walk-forward record must have, over at least `min_test_months`, a mean gain above `min_gain`, t ≥ 2.5
    (standard error inflated by √2 for overlapping months; 2.5 because the test repeats weekly) and a positive last
    six months. The model put in use is the candidate with the best average over all test months."""
    t0 = time.monotonic()
    rows = rows_for(L, h)
    months = sorted({r["made"][:7] for r in rows})
    complete = months[:-1]           # the newest month's forecasts aren't all graded yet: it isn't a test month
    n_live = sum(1 for r in L.get("graded", {}).get("quant", {}).get(str(h), [])
                 if (r.get("f") or {}).get("v") == FEAT_V and "er" in r)
    res = {"h": h, "n": len(rows), "n_live": n_live, "months": len(months), "accepted": False, "fv": FEAT_V,
           "feats": FEATS, "candidates": {}, "model": "none"}
    if len(rows) < min_n or len(complete) < min_train_months + min_test_months:
        res["status"] = (f"learning: needs {min_n}+ graded {h}-day forecasts over "
                         f"{min_train_months + min_test_months}+ months (has {len(rows)} over {len(months)})")
        return res
    X = _matrix(rows, FEATS)
    made, raw, y = targets(rows, L, h)
    order = _hash_order([f"{r['t']}:{r['made']}" for r in rows])
    gap = int(math.ceil(h * 1.4))
    gains: dict[str, list[float]] = {c: [] for c in CANDIDATES}
    tested: list[str] = []
    # A month's result depends only on its training and test rows, which stop changing once its forecasts are
    # graded (old rows are dropped a whole month at a time): stored results are reused, so a weekly run refits
    # only the latest months.
    store = L.setdefault("ml_cache", {})
    if store.get("v") != [HP_V, FEAT_V]:
        store.clear()
        store.update(v=[HP_V, FEAT_V], folds={})
    folds = store.setdefault("folds", {})
    kept = set()
    for m in complete[min_train_months:]:
        if seconds is not None and time.monotonic() - t0 > seconds:
            res["status"] = "postponed: the weekly test ran out of time this run and will finish on a later one"
            res["postponed"] = True
            return res
        start = dt.date.fromisoformat(m + "-01")
        cut = (start - dt.timedelta(days=gap)).isoformat()
        tr = np.flatnonzero(made < cut)
        te = np.flatnonzero(np.asarray([x[:7] == m for x in made]))
        if len(tr) < min_n * 0.5 or len(te) < 20:
            continue
        m0 = float(np.mean(y[te] ** 2))
        if m0 <= 0:
            continue
        # what the result depends on: the rows (counts, date range), their outcomes and their features
        # (y is demeaned by day, so it sums to zero: its sum of squares and the raw errors are the checksums)
        key = [len(tr), str(made[tr[0]]), str(made[tr[-1]]), round(float(np.sum(y[tr] ** 2)), 6),
               round(float(np.sum(raw[tr])), 6), round(float(np.nansum(X[tr])), 3), len(te), str(made[te[0]]),
               str(made[te[-1]]), round(float(np.sum(y[te] ** 2)), 6), round(float(np.sum(raw[te])), 6),
               round(float(np.nansum(X[te])), 3), b0]
        fk = f"{h}:{m}"
        kept.add(fk)
        hit = folds.get(fk)
        if hit and hit.get("key") == key:
            g = hit["gains"]
        else:
            tr = _sample(tr, order)
            cap = 0.25 * float(np.std(y[tr]))
            f_all = fit("average", X[tr], y[tr], periods(made[tr], h), b0)
            preds = {k: f_all._raw(k, X[te]) for k in ("ridge", "trees", "neural_net")}
            preds["average"] = np.mean([preds["ridge"], preds["trees"], preds["neural_net"]], axis=0)
            # scored exactly as it would be used: centred on the day's average, then capped
            g = {c: round(1 - float(np.mean((y[te] - np.clip(_centered(preds[c], made[te]), -cap, cap)) ** 2)) / m0, 7)
                 for c in CANDIDATES}
            folds[fk] = {"key": key, "gains": g}
        for c in CANDIDATES:
            gains[c].append(float(g[c]))
        tested.append(m)
    for fk in [k for k in folds if k.startswith(f"{h}:") and k not in kept]:
        del folds[fk]
    k = len(tested)
    res["test_months"] = k
    if k < min_test_months:
        res["status"] = f"learning: {k} of {min_test_months} test months so far"
        return res
    meta, picks = [], []
    for i in range(k):
        pick = "average" if i < 3 else max(CANDIDATES, key=lambda c: float(np.mean(gains[c][:i])))
        picks.append(pick)
        meta.append(gains[pick][i])

    def stats(g: list[float]) -> tuple[float, float]:
        a = np.asarray(g)
        se = float(a.std(ddof=1) / math.sqrt(len(a))) * math.sqrt(2) if len(a) > 1 else float("inf")
        return float(a.mean()), (float(a.mean()) / se if se > 0 else 0.0)

    for c in CANDIDATES:
        mg, tc = stats(gains[c])
        res["candidates"][c] = {"gain": round(mg, 5), "t": round(tc, 2), "recent": round(float(np.mean(gains[c][-6:])), 5)}
    mean, t = stats(meta)
    recent = float(np.mean(meta[-6:]))
    best = max(CANDIDATES, key=lambda c: float(np.mean(gains[c])))
    # few test months make t itself noisy: the bar is the stricter of 2.5 and the one-sided 0.5% point of Student's t
    from scipy.stats import t as student
    t_need = max(min_t, float(student.ppf(0.995, k - 1)))
    res.update(test_gain=round(mean, 5), t=round(t, 2), t_needed=round(t_need, 2), recent_gain=round(recent, 5),
               monthly_gains=[round(float(x), 4) for x in meta[-12:]], picks=picks[-12:], best=best,
               cap=round(0.25 * float(np.std(y)), 5), periods=periods(made, h), last_month=tested[-1])
    if mean > min_gain and t >= t_need and recent > 0:
        res.update(accepted=True, model=best,
                   status=f"applied: {NAMES[best].lower()} won the monthly test and improved forecasts it never saw")
    else:
        res["status"] = "rejected: no algorithm reliably improved forecasts it never saw (no nudge is applied)"
    return res


_MODEL_CACHE: dict = {}


class SignalModel:
    """The signal learner in use: refit (deterministically) on the whole graded record when first needed."""

    def __init__(self, tw: dict, fitted: Fitted):
        self.tw, self.fitted, self.h = tw, fitted, int(tw.get("h") or 21)
        self.cap = float(tw.get("cap") or 0.02)
        self.center = 0.0

    def _row(self, feats: dict) -> np.ndarray:
        return np.asarray([[float(feats[k]) if feats.get(k) is not None and math.isfinite(float(feats[k])) else np.nan
                            for k in FEATS]])

    def set_day(self, feats_list: list[dict]) -> float:
        """The average raw prediction across the day's stocks: subtracted from each, as in the test."""
        if feats_list:
            X = np.vstack([self._row(f) for f in feats_list])
            p = self.fitted.predict(X)
            self.center = float(np.mean(p[np.isfinite(p)])) if np.isfinite(p).any() else 0.0
        return self.center

    @classmethod
    def build(cls, tw: dict | None, L: dict, b0: float = 6.0) -> "SignalModel | None":
        if not tw or not tw.get("accepted") or not tw.get("on") or tw.get("fv") != FEAT_V \
                or tw.get("model") not in CANDIDATES:
            return None
        h = int(tw.get("h") or 21)
        rows = rows_for(L, h)
        if len(rows) < 100:
            return None
        key = (tw["model"], h, len(rows), rows[-1]["made"], rows[0]["made"])
        if key not in _MODEL_CACHE:
            X = _matrix(rows, FEATS)
            made, _, y = targets(rows, L, h)
            idx = _sample(np.arange(len(rows)), _hash_order([f"{r['t']}:{r['made']}" for r in rows]))
            _MODEL_CACHE.clear()
            _MODEL_CACHE[key] = fit(tw["model"], X[idx], y[idx], periods(made, h), b0)
        return cls(tw, _MODEL_CACHE[key])

    def tilt(self, feats: dict) -> tuple[float, list[dict], int]:
        """(log-return nudge over the tested horizon, what drove it, that horizon in sessions)."""
        x = self._row(feats)
        if not np.isfinite(x).any():
            return 0.0, [], self.h
        # each signal's share: how much the nudge changes if that signal were at its usual value (one batch)
        have = [j for j in range(len(FEATS)) if np.isfinite(x[0, j])]
        Xs = np.repeat(x, len(have) + 1, axis=0)
        for r, j in enumerate(have, start=1):
            Xs[r, j] = self.fitted.med[j]
        out = self.fitted.predict(Xs)
        p = float(out[0])
        parts = []
        for r, j in enumerate(have, start=1):
            c = p - float(out[r])
            if abs(c) > 1e-6:
                k = FEATS[j]
                parts.append({"k": k, "label": LABELS[k][0].upper() + LABELS[k][1:], "h_ret": round(c, 5)})
        tot = float(np.clip(p - self.center, -self.cap, self.cap)) if math.isfinite(p) else 0.0
        return tot, sorted(parts, key=lambda q: -abs(q["h_ret"]))[:4], self.h


# ---------------------------------------------------------------- the range learner (conditional calibration)

RANGE_HORIZONS = (("1d", 1), ("1w", 5), ("1m", 21), ("3m", 63))


def _need(rows: list[dict], shift: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    er = np.asarray([r["er"] for r in rows], dtype=float)
    w10 = np.maximum(np.asarray([r["w10"] for r in rows], dtype=float), 1e-6)
    w90 = np.maximum(np.asarray([r["w90"] for r in rows], dtype=float), 1e-6)
    e2 = er - shift
    return np.where(e2 >= 0, e2 / w90, -e2 / w10), er, w10, w90


def _pinball3(er, w10, w90, scale, shift) -> float:
    tot = 0.0
    for tau, q in ((0.1, shift - scale * w10), (0.5, np.full_like(er, shift)), (0.9, shift + scale * w90)):
        u = er - q
        tot += float(np.mean(np.maximum(tau * u, (tau - 1) * u)))
    return tot / 3


class _Q80:
    """The 80th percentile of how much a range needed stretching, given the stock's situation. A location-scale
    model: gradient-boosted trees learn the typical (log) stretch from the features, and one constant turns that
    into an 80th percentile. It uses every outcome rather than only the misses, so it is steadier than fitting the
    percentile directly, and much faster."""

    def __init__(self, X: np.ndarray, need: np.ndarray, made: np.ndarray | None = None, h: int = 21):
        _quiet()
        target = np.log(need + 0.05)
        self.use = np.asarray([j for j in range(X.shape[1]) if len(np.unique(X[np.isfinite(X[:, j]), j])) >= 2], dtype=int)
        self.m = self._trees(X, target)
        # The constant comes from forecasts the trees were not fitted on (three blocks of dates, each scored by trees
        # fitted on the other two, minus a gap the length of the horizon): in-sample the trees look too good, the
        # constant comes out too small and the ranges too narrow.
        ratio = None
        if made is not None and self.m is not None and len(need) >= 600:
            ds = sorted(set(made.tolist()))
            edges = [ds[0], ds[len(ds) // 3], ds[2 * len(ds) // 3], "9999"]
            gap = dt.timedelta(days=int(math.ceil(h * 1.4)))
            parts = []
            for a, b in zip(edges[:-1], edges[1:]):
                te = (made >= a) & (made < b)
                lo_ = (dt.date.fromisoformat(a) - gap).isoformat()
                hi_ = (dt.date.fromisoformat(b) + gap).isoformat() if b != "9999" else "9999"
                tr = (made < lo_) | (made >= hi_)
                if te.sum() < 100 or tr.sum() < 300:
                    parts = []
                    break
                m = self._trees(X[tr], target[tr])
                parts.append((need[te] + 0.05) / self._typical(X[te], m))
            if parts:
                ratio = np.concatenate(parts)
        if ratio is None:
            ratio = (need + 0.05) / self._typical(X)
        self.c = float(np.quantile(ratio, 0.8))

    def _trees(self, X: np.ndarray, target: np.ndarray):
        from sklearn.ensemble import HistGradientBoostingRegressor
        if not len(self.use):
            return None
        return HistGradientBoostingRegressor(
            max_depth=2, max_leaf_nodes=4, learning_rate=0.05, max_iter=100, min_samples_leaf=max(200, len(target) // 30),
            l2_regularization=1.0, early_stopping=False, random_state=0).fit(X[:, self.use], target)

    def _typical(self, X: np.ndarray, m=None) -> np.ndarray:
        X = np.atleast_2d(X)
        m = self.m if m is None else m
        if m is None:
            return np.ones(len(X))
        return np.exp(m.predict(X[:, self.use]))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.maximum(self.c * self._typical(X) - 0.05, 0.0)


def _fit_q80(X: np.ndarray, need: np.ndarray, made=None, h: int = 21) -> _Q80:
    return _Q80(X, need, None if made is None else np.asarray(made), h)


def range_rows(L: dict, src: str, h: int, days: int) -> list[dict]:
    rows = [r for r in L.get("graded", {}).get(src, {}).get(str(h), [])
            if "w10" in r and "er" in r and (r.get("f") or {}).get("v") == FEAT_V]
    if not rows:
        return []
    last = max(r["made"] for r in rows)
    cut = (dt.date.fromisoformat(last) - dt.timedelta(days=days)).isoformat()
    return sorted([r for r in rows if r["made"] >= cut], key=lambda r: r["made"])


def range_learner(L: dict, calib: dict, prev: dict | None = None, min_n: int = 1000, min_periods: int = 6,
                  b0: float = 6.0, lo: float = 0.6, hi: float = 2.0, min_edge: float = 0.005, keep_edge: float = 0.0,
                  window_days: int = 460) -> dict:
    """Per horizon: does a conditional stretch beat the single one on the newest 30% of days? Fitted on older
    forecasts only (whose outcomes were known before those days), it must lower the 10/50/90 pinball loss there by
    at least `min_edge` versus whatever the single calibration would apply (itself, or no change if held); once on,
    it stays on while it does no worse than that (`keep_edge`), so the ranges don't flip week to week. It learns from
    live forecasts once there are enough for it, and until then from the walk-forward backtest."""
    from .forecast import trading_days_after
    out = {}
    for name, h in RANGE_HORIZONS:
        g = calib.get(name) or {}
        win = max(window_days, block_days(h) * 10)
        src, rows = "quant", range_rows(L, "quant", h, win)
        if len(rows) < min_n or periods([r["made"] for r in rows], h) < min_periods:
            bt = range_rows(L, "quant_bt", h, max(win, window_days * 3))
            if len(bt) >= min_n and periods([r["made"] for r in bt], h) >= min_periods:
                src, rows, win = "quant_bt", bt, max(win, window_days * 3)
        days = sorted({r["made"] for r in rows})
        per = periods(days, h)
        base = {"n": len(rows), "periods": per, "from": "backtest" if src == "quant_bt" else "live", "window": win,
                "status": "learning"}
        if len(rows) < min_n or per < min_periods or g.get("status") not in ("applied", "held"):
            out[name] = {**base, "need": {"n": min_n, "periods": min_periods}}
            continue
        shift = float(g.get("shift") or 0.0) if g.get("status") == "applied" else 0.0
        cut = days[int(len(days) * 0.7)]
        cut_d = dt.date.fromisoformat(cut)
        ends = {d: trading_days_after(dt.date.fromisoformat(d), h)[-1] for d in days if d < cut}
        old = [r for r in rows if r["made"] < cut and ends[r["made"]] <= cut_d]
        new = [r for r in rows if r["made"] >= cut]
        if len(old) < min_n // 2 or len(new) < 200:
            out[name] = {**base, "need": {"n": min_n, "periods": min_periods}}
            continue
        p_old = periods([r["made"] for r in old], h)
        w_old = p_old / (p_old + b0)
        # the hold-out's median nudge, like everything else it uses, comes from the older forecasts only
        sh_old = 0.0
        if g.get("status") == "applied":
            from .learn import _fit_scale
            sh_old = _fit_scale(old, w_old, lo, hi, p_old / (p_old + 60.0))[1]
        need_o, _, _, _ = _need(old, sh_old)
        need_n, er_n, w10_n, w90_n = _need(new, sh_old)
        s_glob_old = float(np.clip(1 + (float(np.quantile(need_o, 0.8)) - 1) * w_old, lo, hi))
        model = _fit_q80(_matrix(old, RANGE_FEATS), need_o, [r["made"] for r in old], h)
        q = model.predict(_matrix(new, RANGE_FEATS))
        s_cond = np.clip(1 + (q - 1) * w_old, lo, hi)
        ref_scale = s_glob_old if g.get("status") == "applied" else 1.0
        loss_ref = _pinball3(er_n, w10_n, w90_n, ref_scale, sh_old)
        loss_cond = _pinball3(er_n, w10_n, w90_n, s_cond, sh_old)
        edge = 1 - loss_cond / loss_ref if loss_ref > 0 else 0.0
        res = {**base, "holdout_gain_vs_single": round(edge, 5), "holdout_n": len(new),
               "cov80_holdout_single": float(np.mean(need_n <= ref_scale)), "cov80_holdout": float(np.mean(need_n <= s_cond)),
               "scale_spread": [round(float(np.quantile(s_cond, 0.1)), 3), round(float(np.quantile(s_cond, 0.9)), 3)],
               "single_scale": round(float(g.get("scale", 1.0)), 4), "shift": round(shift, 5)}
        was_on = ((prev or {}).get(name) or {}).get("status") == "applied"
        res["status"] = "applied" if edge >= min_edge or (was_on and edge >= keep_edge) else "held"
        out[name] = res
    return out


class RangeModel:
    """The range learner in use: per horizon, a quantile model refit on the graded record when first needed."""

    def __init__(self, parts: dict):
        self.parts = parts                         # name → (model, weight, shift)

    @classmethod
    def build(cls, rl: dict | None, calib: dict, L: dict, b0: float = 6.0, window_days: int = 460) -> "RangeModel | None":
        parts = {}
        for name, h in RANGE_HORIZONS:
            st = (rl or {}).get(name) or {}
            if st.get("status") != "applied":
                continue
            src = "quant_bt" if st.get("from") == "backtest" else "quant"
            rows = range_rows(L, src, h, int(st.get("window") or max(window_days, block_days(h) * 10)))
            if len(rows) < 200:
                continue
            key = ("range", name, len(rows), rows[-1]["made"], rows[0]["made"], st.get("shift"))
            if key not in _RANGE_CACHE:
                if len(_RANGE_CACHE) >= 2 * len(RANGE_HORIZONS):   # one run's worth (the record moves daily)
                    _RANGE_CACHE.clear()
                shift = float(st.get("shift") or 0.0)
                need, _, _, _ = _need(rows, shift)
                made = [r["made"] for r in rows]
                per = periods(made, h)
                _RANGE_CACHE[key] = (_fit_q80(_matrix(rows, RANGE_FEATS), need, made, h), per / (per + b0), shift)
            parts[name] = _RANGE_CACHE[key]
        return cls(parts) if parts else None

    def calib_for(self, feats: dict, fc: dict, calib: dict, lo: float = 0.6, hi: float = 2.0) -> dict:
        """The stock's own calibration: the single per-horizon stretch, replaced where the range learner is on."""
        out = dict(calib or {})
        for name, (model, w, shift) in self.parts.items():
            x = row_features(feats, fc, name)
            X = np.asarray([[float(x[k]) if x.get(k) is not None and math.isfinite(float(x[k])) else np.nan
                             for k in RANGE_FEATS]])
            q = float(model.predict(X)[0])
            out[name] = {"status": "applied", "scale": round(float(np.clip(1 + (q - 1) * w, lo, hi)), 4),
                         "shift": shift, "cond": True}
        return out


_RANGE_CACHE: dict = {}


def forecast_with_learning(build, feats: dict, calib: dict, smodel: "SignalModel | None",
                           rmodel: "RangeModel | None") -> tuple[dict | None, float]:
    """One stock's forecast with whatever the learners have earned the right to change. `build(calib=…, tilt_ret=…,
    tilt_days=…, tilt_parts=…)` makes the forecast. The range learner needs the forecast's own event dates (is an
    earnings report inside the window?), so when it is on the forecast is built twice. Returns (forecast, tilt)."""
    tr, parts, td = smodel.tilt(feats) if smodel else (0.0, [], 21)
    fc = build(calib=calib, tilt_ret=tr, tilt_days=td, tilt_parts=parts)
    if fc and rmodel:
        own = rmodel.calib_for(feats, fc, calib)
        if own != calib:
            fc = build(calib=own, tilt_ret=tr, tilt_days=td, tilt_parts=parts) or fc
    return fc, tr


def version(tw: dict | None, rl: dict | None) -> str:
    """Changes whenever what the learners would apply changes (so cached forecasts are rebuilt)."""
    a = (tw or {}).get("model") if (tw or {}).get("on") and (tw or {}).get("accepted") else None
    b = {k: v.get("status") for k, v in (rl or {}).items()}
    return hashlib.sha1(repr((a, sorted(b.items()), FEAT_V)).encode()).hexdigest()[:8]
