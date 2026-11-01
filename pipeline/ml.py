"""Machine learning for the price forecasts: which algorithms compete, how they're tested, how they're used.

Three learners. None touches a forecast until it has improved forecasts made after the data it learned from.

1. The signal learner nudges a forecast's median by how the stock is likely to do relative to the other stocks
   forecast the same day. Nine candidates compete: ridge regression, robust (Huber) regression, spline regression,
   gradient-boosted trees, a random forest, CatBoost, a neural network (three small networks, averaged), the average
   of all seven, and the fused model: the three with the best record so far, averaged (re-chosen every month from
   months whose outcomes were in, so the fusion itself is tested on months it didn't see). In the largest study of machine learning for stock returns (Gu, Kelly & Xiu 2020), trees
   and neural networks beat linear models given decades of data on thousands of stocks; with less data, heavily
   regularized linear models are often as good or better, and averaging models is a robust default. So the app
   doesn't assume a winner. Each candidate is refit month by month on earlier months only and scored on the next;
   the candidate used for each month is the one that had scored best on the months before it, so even the choice
   of algorithm is tested on data it didn't see; and that whole procedure must pass a significance test before it
   nudges anything. Predictions are shrunk toward zero by independent months/(months + 6), so with little
   evidence they say little.
2. The range learner (conditional calibration) learns, per horizon, how much a forecast's 10–90% range needed
   stretching to hold the outcome, as a function of the stock's situation (volatility level, recent volatility
   versus its usual and versus the market, the biggest recent daily move, distance from the 52-week high,
   liquidity, beta, an earnings report inside the window, and how often the stock's own ranges missed before), with
   gradient-boosted trees. It replaces the single per-horizon stretch only if it scored better on the newest
   forecasts it wasn't fitted on.
3. The coverage keeper (adaptive conformal inference, Gibbs & Candès 2021) widens or narrows every range a little
   after each day's results come in (more misses than the 20% an 80% range allows: wider; fewer: narrower), so
   coverage stays near 80% when markets change. It is replayed over the live record, using only results known
   before each day, and is used only where that replay scored better than leaving the ranges alone.

Features are computed the same way for live forecasts and for the walk-forward backtest (prices and volumes known
on the anchor day), so the learners have data from the first day. News, filings and financial statements are known
only for live forecasts (blank in the backtest), so they start to count as live months accumulate. Fitted models are never stored: they are refit
deterministically from the graded record when needed, so what is used is exactly what was tested.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import math
import os
import time
import warnings

import numpy as np

from .prices import BENCH

FEAT_V = 3
PRICE_FEATS = ["mom", "rev", "rev_w", "hi52", "lvol", "vol_rel", "vol_chg", "maxret", "beta", "liq"]
NEWS_FEATS = ["news", "tone", "ev8k", "flag"]                 # headlines and 8-K filings known at the anchor close
FUND_FEATS = ["growth", "margin", "fcfy", "dilution"]         # the latest financial statements filed before it
SIGNAL_FEATS = ["sig", "val", "claude", "ins", "bub"] + NEWS_FEATS + FUND_FEATS
FEATS = PRICE_FEATS + SIGNAL_FEATS
RANGE_FEATS = ["lvol", "vol_rel", "vol_chg", "maxret", "hi52", "liq", "beta", "ein"]
RANGE_X = RANGE_FEATS + ["miss", "emiss"]  # + the stock's own record of misses, and of misses caused by earnings
RANGE_V = 3
LABELS = {"mom": "12-month momentum vs the S&P 500", "rev": "last month's move vs the S&P 500",
          "rev_w": "last week's move vs the S&P 500", "hi52": "distance from its 52-week high",
          "lvol": "its volatility", "vol_rel": "its volatility vs the market's", "vol_chg": "recent vs usual volatility",
          "maxret": "its biggest daily jump last month", "beta": "its beta", "liq": "trading volume (liquidity)",
          "sig": "smart-money signals", "val": "valuation", "claude": "Claude's stance", "ins": "insider net buying",
          "bub": "bubble-risk score", "ein": "an earnings report in the window",
          "news": "news coverage (headlines in the last week)", "tone": "the tone of last week's headlines",
          "ev8k": "material 8-K filings in the last 10 days", "flag": "a red-flag 8-K in the last 90 days",
          "growth": "revenue growth (latest filing)", "margin": "operating margin (latest filing)",
          "fcfy": "free-cash-flow yield", "dilution": "share issuance (dilution)",
          "miss": "how often its ranges missed before"}
# The candidates, simplest first. Each catches a different kind of pattern; "average" combines them all.
BASE = ["ridge", "huber", "spline", "trees", "random_forest", "catboost", "neural_net"]


def _have(module: str) -> bool:
    try:
        __import__(module)
        return True
    except Exception:  # noqa: BLE001 – a broken install (not just a missing one) must not stop the pipeline
        return False


def _catboost_works() -> bool:
    """CatBoost imports and can fit a tiny model here (a broken install, or a CPU it doesn't support, fails at the
    first fit, not at import). Without it, the others compete without it."""
    if not _have("catboost"):
        return False
    try:
        from catboost import CatBoostRegressor
        rng = np.random.default_rng(0)
        X = rng.normal(size=(60, 3))
        CatBoostRegressor(iterations=3, depth=2, verbose=False, allow_writing_files=False, thread_count=1).fit(
            X, X[:, 0] + rng.normal(0, 0.1, 60))
        return True
    except Exception:  # noqa: BLE001
        return False


AVAILABLE = [b for b in BASE if b != "catboost" or _catboost_works()]   # CatBoost is an extra library
CANDIDATES = AVAILABLE + ["average", "fusion"]
NAMES = {"ridge": "Ridge regression", "huber": "Robust regression (Huber)", "spline": "Spline regression",
         "trees": "Gradient-boosted trees", "random_forest": "Random forest", "catboost": "CatBoost",
         "neural_net": "Neural network", "average": "Average of all", "fusion": "Fused model (best three)",
         "none": "None"}
FUSE_K = 3                       # the fused model averages this many of the best-so-far algorithms
FUSION_MIN_MONTHS = 9            # test months as a real fusion (three months graded first) before it can be put in use
MAX_ROWS = 20000                 # training rows per fit (a fixed sample by stock and date when there are more)
HP_V = 5                         # the candidates' settings and the test: bump when they change (results are redone)
THREADS = int(os.environ.get("SMR_ML_THREADS") or 0) or -1     # processor cores the forests may use (-1: all)


def _light() -> bool:
    """Fewer trees and networks (the test suite sets SMR_ML_LIGHT=1 to keep its runs short; never in the pipeline)."""
    return os.environ.get("SMR_ML_LIGHT") == "1"


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


ROUTINE_8K = {"2.02", "9.01", "5.07", "7.01"}   # results (counted as earnings), exhibits, votes, investor decks


def _ts(s) -> dt.datetime | None:
    try:
        d = dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)
    except (TypeError, ValueError):
        return None


def _filed_ts(s) -> dt.datetime | None:
    """When an 8-K was accepted. The live feed's times carry a real UTC offset; the SEC submissions list's
    acceptanceDateTime is New York clock time despite its "Z" (a 4:30pm ET filing reads "T16:30:00.000Z")."""
    import re
    from .util import ET
    s = str(s or "").strip()
    if re.search(r"[+-]\d{2}:?\d{2}$", s):
        return _ts(s)
    try:
        d = dt.datetime.fromisoformat(s.rstrip("Z")[:19])
    except ValueError:
        return None
    return d.replace(tzinfo=ET).astimezone(dt.timezone.utc)


def context_features(x: dict | None, cut: str, last: str) -> dict:
    """News, filing and financial-statement signals known at the anchor close (`cut`, a UTC timestamp; `last`, the
    anchor session's date). Anything published after the close is left out, and a signal whose source wasn't
    refreshed recently enough to trust is left blank (missing, which the learners treat as "no information"), never
    guessed.
      news      headlines about the company GDELT had seen in the week before the app's last news check before the
                close (a check at most 36 hours before it; up to 250 headlines, log scale)
      tone      their average tone, -1 to 1 (a small word list: "beats", "upgrade" … "misses", "probe" …)
      ev8k      8-K filings with a material item in the 10 days before the close (results, exhibits, votes and
                investor decks don't count); only with a filing record refreshed in the week before
      flag      1 if a red-flag item (bankruptcy, accelerated debt, delisting notice, restatement) was filed in the
                90 days before, else 0 (same condition)
      growth, margin, fcfy, dilution: revenue growth, operating margin, free-cash-flow yield and share issuance from
                the latest financial statements, if everything they could include was filed before the anchor day
                (on a day new statements come out, these are blank)"""
    f: dict = {}
    if not x:
        return f
    c = _ts(cut)
    if c is None:
        return f
    # the latest news check made before the close (at most 36 hours before it): everything it counted was published
    # before the close, and nothing after the close can change it
    best = None
    for sg in x.get("news_sigs") or []:
        when = _ts(sg[0]) if isinstance(sg, (list, tuple)) and len(sg) >= 3 else None
        if when is not None and c - dt.timedelta(hours=36) <= when <= c and (best is None or when > best[0]):
            best = (when, sg)
    if best is not None:
        n, tone = best[1][1], best[1][2]
        if isinstance(n, (int, float)) and isinstance(tone, (int, float)) and math.isfinite(n) and math.isfinite(tone):
            if len(best[1]) >= 4 and best[1][3]:
                n = max(float(n), 250.0)               # a full answer: at least as many as it can hold
            f["news"] = math.log1p(max(0.0, float(n)))
            f["tone"] = float(np.clip(tone, -1.0, 1.0))
    fa = _ts(x.get("filings_at"))
    fls = x.get("filings_feat") if x.get("filings_feat") is not None else x.get("filings")
    if fls is not None and fa is not None and fa >= c - dt.timedelta(days=7):
        n8, red = 0, 0
        for fl in fls:
            at = _filed_ts(fl.get("at")) if fl.get("at") else None
            day = str(fl.get("date") or "")[:10]
            if at is not None:
                if at > c:
                    continue
                age = (c - at).total_seconds() / 86400
            else:
                if not day or day >= last:            # filed on the anchor day at an unknown time: not counted
                    continue
                try:
                    age = (dt.date.fromisoformat(last) - dt.date.fromisoformat(day)).days
                except ValueError:
                    continue
            items = [i for i in fl.get("items") or [] if isinstance(i, dict)]
            if age <= 10 and any(i.get("item") not in ROUTINE_8K for i in items):
                n8 += 1
            if age <= 90 and any(i.get("sev") == "red" for i in items):
                red = 1
        f["ev8k"] = float(min(n8, 5))
        # no red flag found: "none" only if the record reaches back the full 90 days
        try:
            reach = (dt.date.fromisoformat(last) - dt.date.fromisoformat(str(x.get("filings_from"))[:10])).days
        except ValueError:
            reach = 10 ** 6
        if red or reach >= 90:
            f["flag"] = float(red)
    fx = x.get("feat") or {}
    v = x.get("val") or {}
    # (dated by the newest filing the numbers could include: unknown means blank)
    if v and fx.get("val_date") == last and fx.get("stmt_filed") and fx["stmt_filed"] < last:
        for k, src, lo, hi in (("growth", "growth", -0.5, 1.5), ("margin", "op_margin", -0.6, 0.6),
                               ("fcfy", "fcf_yield", -0.2, 0.2), ("dilution", "dilution", -0.2, 0.5)):
            val = v.get(src)
            if isinstance(val, (int, float)) and math.isfinite(val):
                f[k] = float(np.clip(val, lo, hi))
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
        if kind == "huber":
            return self._z(X) @ self.parts["huber"] * self.parts["ys"]
        if kind == "spline":
            sp, bmu, bsd, coef = self.parts["spline"]
            return _spline_basis(sp, self._z(X), np.isfinite(X), bmu, bsd)[0] @ coef
        if kind in ("trees", "random_forest", "catboost"):
            m = self.parts.get(kind)             # None: no signal had any spread, or the library failed to load
            if m is None:
                return np.zeros(len(X))
            return np.asarray(m.predict(X[:, self.parts["use"]]), dtype=float) * self.shrink
        if kind == "neural_net":
            Z = self._z(X)
            return np.mean([m.predict(Z) for m in self.parts["neural_net"]], axis=0) * self.parts["ys"] * self.shrink
        raise ValueError(kind)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self.name in ("average", "fusion"):
            return np.mean([self._raw(k, X) for k in self.parts["bases"]], axis=0)
        return self._raw(self.name, X)


def _spline_knots(Z: np.ndarray, obs: np.ndarray, n: int = 5) -> np.ndarray:
    """Knots at each signal's quantiles among the stocks that have it (missing values, stored as 0, would pile up
    on one knot); evenly spaced where a signal has too few distinct values for distinct quantiles."""
    K = np.empty((n, Z.shape[1]))
    for j in range(Z.shape[1]):
        v = Z[obs[:, j], j]
        q = np.quantile(v, np.linspace(0, 1, n)) if len(v) else np.zeros(n)
        if not len(v) or np.any(np.diff(q) <= 1e-9):
            lo, hi = (float(v.min()), float(v.max())) if len(v) else (-1.0, 1.0)
            if hi - lo <= 1e-9:
                lo, hi = lo - 1.0, hi + 1.0
            q = np.linspace(lo, hi, n)
        K[:, j] = q
    return K


def _spline_basis(sp, Z: np.ndarray, obs: np.ndarray, bmu=None, bsd=None):
    """Each signal's spline basis, standardized on the stocks that have the signal; a missing signal's basis is 0
    (its average), so a missing value means "no information", as it does for ridge."""
    B = sp.transform(Z)
    nb = B.shape[1] // Z.shape[1]
    if bmu is None:
        bmu, bsd = np.zeros(B.shape[1]), np.ones(B.shape[1])
        for j in range(Z.shape[1]):
            c = slice(j * nb, (j + 1) * nb)
            o = obs[:, j]
            if o.sum() > 1:
                bmu[c] = B[o, c].mean(axis=0)
                sd_ = B[o, c].std(axis=0)
                bsd[c] = np.where(sd_ > 0, sd_, 1.0)
    out = (B - bmu) / bsd
    for j in range(Z.shape[1]):
        out[~obs[:, j], j * nb:(j + 1) * nb] = 0.0
    return out, bmu, bsd, nb


def _fit_ridge(Z: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    n, k = Z.shape
    A = np.hstack([np.ones((n, 1)), Z])
    P = np.eye(k + 1) * lam
    P[0, 0] = 0.0
    b = np.linalg.solve(A.T @ A + P, A.T @ y)
    return b[1:]


def fit(name: str, X: np.ndarray, y: np.ndarray, periods: int, b0: float = 6.0, seeds: int = 3,
        bases: list[str] | None = None) -> Fitted:
    """Fit one candidate (a name in BASE; "average", which fits every available one; or "fusion", which fits the
    `bases` it averages). Their settings are fixed
    in advance and deliberately cautious (heavy shrinkage, shallow trees, big leaves, a small network): tuning
    them on the same record the test uses would flatter the test.
      ridge          linear; the penalty fades as evidence grows
      huber          linear, but a few huge errors (an earnings gap, a takeover) can't drag the fit around
      spline         each signal's effect may bend (only extreme values matter, say), no combinations
      trees          gradient-boosted trees: thresholds and combinations of signals
      random_forest  many trees on random halves of the data, averaged: steadier on noisy data
      catboost       boosted trees built to over-learn less on small, noisy data (symmetric trees, ordered boosting)
      neural_net     smooth curved patterns (three small networks, averaged)
    Trees, forests and networks are shrunk toward zero by periods/(periods + b0); the linear ones through their
    penalties."""
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import HuberRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import SplineTransformer
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
    if name == "fusion":
        if not bases or not set(bases) <= set(AVAILABLE):
            raise ValueError(f"fusion of {bases}")
        need = tuple(b for b in AVAILABLE if b in set(bases))
    else:
        need = tuple(AVAILABLE) if name == "average" else (name,)
    if name not in BASE and name not in ("average", "fusion"):
        raise ValueError(name)
    out.parts["bases"] = list(need)
    lam = n * b0 / max(1, periods)          # a weight keeps about periods/(periods + b0) of its unshrunk size
    ys = float(np.std(y)) or 1.0
    out.parts["ys"] = ys
    # trees and forests use the raw values (missing ones get their own branch); signals with no spread are left out
    use = np.asarray([j for j in range(k) if len(np.unique(X[np.isfinite(X[:, j]), j])) >= 2], dtype=int)
    out.parts["use"] = use
    leaf = max(100, int(0.03 * n))
    if "ridge" in need:
        out.parts["ridge"] = _fit_ridge(Z, y, lam)
    if "huber" in need:
        # on the unit scale the Huber loss matches squared error for ordinary errors, so the same penalty applies
        hb = HuberRegressor(epsilon=1.35, alpha=lam, max_iter=300).fit(Z, y / ys)
        out.parts["huber"] = hb.coef_
    if "spline" in need:
        obs = np.isfinite(X)
        sp = SplineTransformer(knots=_spline_knots(Z, obs), degree=3, extrapolation="linear",
                               include_bias=False).fit(Z)
        B, bmu, bsd, nb = _spline_basis(sp, Z, obs)
        # six basis columns per signal: the penalty grows with them, so a signal's curve is held about as firmly
        # as ridge holds its single slope
        out.parts["spline"] = (sp, bmu, bsd, _fit_ridge(B, y, lam * nb))
    if "trees" in need:
        out.parts["trees"] = HistGradientBoostingRegressor(
            max_depth=3, max_leaf_nodes=8, learning_rate=0.05, max_iter=60, min_samples_leaf=leaf,
            max_features=0.5, l2_regularization=1.0, early_stopping=False,
            random_state=0).fit(X[:, use], y) if len(use) else None
    light = _light()
    if "random_forest" in need:
        out.parts["random_forest"] = RandomForestRegressor(
            n_estimators=40 if light else 150, max_depth=6, min_samples_leaf=leaf, max_features=0.5, max_samples=0.5,
            n_jobs=THREADS, random_state=0).fit(X[:, use], y) if len(use) else None
    if "catboost" in need:
        # (whether CatBoost works here is checked when the module loads; a failure on this data fails the build: an
        # average or fused model without one of its members isn't the one that was tested. The weekly test is
        # retried on the next run, and a model in use is left off until it builds.)
        from catboost import CatBoostRegressor
        out.parts["catboost"] = CatBoostRegressor(
            iterations=50 if light else 150, depth=4, learning_rate=0.05, l2_leaf_reg=30.0, rsm=0.5,
            random_seed=0, verbose=False, thread_count=THREADS,
            allow_writing_files=False).fit(X[:, use], y) if len(use) else None
    if "neural_net" in need:
        out.parts["neural_net"] = [MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1e-2, learning_rate_init=1e-3,
                                                max_iter=30, batch_size=min(512, n), random_state=s).fit(Z, y / ys)
                                   for s in range(1 if light else seeds)]
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


def row_groups(rows: list[dict], L: dict, h: int = 21) -> np.ndarray:
    """Each row's day and source ("2026-03-02|q" live, "…|b" backtest): the stocks a day is compared across. Live
    and backtest forecasts of the same day are different sets of stocks, so each is centred on its own average."""
    live = {id(r) for r in L.get("graded", {}).get("quant", {}).get(str(h), [])}
    return np.asarray([r["made"] + ("|q" if id(r) in live else "|b") for r in rows])


def _market_surprise(L: dict, h: int) -> dict[str, float]:
    """The S&P 500's own forecast error on each date (live first, then the backtest's)."""
    out: dict[str, float] = {}
    for src in ("quant_bt", "quant"):
        for r in L.get("graded", {}).get(src, {}).get(str(h), []):
            if r.get("t") == BENCH and "er" in r:
                out[r["made"]] = float(r.get("er0", r["er"]))
    return out


def targets(rows: list[dict], L: dict, h: int, groups: np.ndarray | None = None
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(made dates, raw errors, learning target). The target is how a stock did *relative to the others*: its error,
    minus its beta times the market's own surprise that day (so beta can't learn one period's market move), minus
    the average of the stocks it is compared with (`groups`: the same day and source; by default the same day).
    Stock-level signals can predict that; the market's move they can't."""
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
    return made, raw, _date_demeaned(adj, made if groups is None else groups)


def _centered(p: np.ndarray, made: np.ndarray) -> np.ndarray:
    """Predictions minus their average across the stocks forecast the same day: a nudge says how one stock should
    do relative to the others, never which way the whole market goes."""
    return _date_demeaned(p, made)


def compete(L: dict, h: int = 21, min_n: int = 600, min_train_months: int = 6, min_test_months: int = 6,
            b0: float = 6.0, min_gain: float = 0.002, min_t: float = 2.5, seconds: float | None = None) -> dict:
    """The weekly competition. For each month (after `min_train_months`), every candidate is fitted on everything made
    before it (minus a gap the length of the horizon, so no outcome overlaps) and scored on that month: the reduction
    in squared error of how each stock did relative to the others that day. The fused candidate is, each month, the
    average of the three algorithms with the best record on the months before it whose outcomes were all in (the
    average of all of them until three months are), so fusing is tested like everything else; it can be put in use
    only after `FUSION_MIN_MONTHS` test months as a real fusion. The candidate *used* for a month is the one with the
    best average on the months before it (the average of all of them until three months are scored).
    That walk-forward record must have, over at least `min_test_months`, a mean gain above `min_gain`, t ≥ 2.5
    (standard error: the larger of the plain one inflated by √2 for overlapping months and a Newey-West one; 2.5
    because the test repeats weekly) and a positive last six months. The choice for a month uses only months whose
    outcomes were all known when it began. The model put in use is the candidate with the best average over all
    test months."""
    t0 = time.monotonic()
    rows = rows_for(L, h)
    months = sorted({r["made"][:7] for r in rows})
    complete = months[:-1]           # the newest month's forecasts aren't all graded yet: it isn't a test month
    n_live = sum(1 for r in L.get("graded", {}).get("quant", {}).get(str(h), [])
                 if (r.get("f") or {}).get("v") == FEAT_V and "er" in r)
    res = {"h": h, "n": len(rows), "n_live": n_live, "months": len(months), "accepted": False, "fv": FEAT_V,
           "hp": [HP_V, CANDIDATES], "feats": FEATS, "candidates": {}, "model": "none"}
    if len(rows) < min_n or len(complete) < min_train_months + min_test_months:
        res["status"] = (f"learning: needs {min_n}+ graded {h}-day forecasts over "
                         f"{min_train_months + min_test_months}+ months (has {len(rows)} over {len(months)})")
        return res
    X = _matrix(rows, FEATS)
    grp = row_groups(rows, L, h)
    made, raw, y = targets(rows, L, h, grp)
    order = _hash_order([f"{r['t']}:{r['made']}" for r in rows])
    gap = int(math.ceil(h * 1.4))
    gains: dict[str, list[float]] = {c: [] for c in CANDIDATES}
    tested: list[str] = []
    known_by: list[str] = []          # when each test month's last outcome was in (the month after its last forecast)
    fused: list[bool] = []            # whether the fused candidate was a real fusion that month (three months known)
    # A month's result depends only on its training and test rows, which stop changing once its forecasts are
    # graded (old rows are dropped a whole month at a time): stored results are reused, so a weekly run refits
    # only the latest months.
    store = L.setdefault("ml_cache", {})
    if store.get("v") != [HP_V, FEAT_V, CANDIDATES, FEATS]:     # (any change to what is tested starts afresh)
        store.clear()
        store.update(v=[HP_V, FEAT_V, CANDIDATES, FEATS], folds={})
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
        # the fused candidate for this month: the best three on the months graded before it began
        n_known = sum(1 for j in range(len(tested)) if known_by[j] < m + "-01")
        fset = _fuse_set(gains, n_known)
        hit = folds.get(fk)
        if hit and hit.get("key") == key and hit.get("fset") == fset:
            g = hit["gains"]
        else:
            tr = _sample(tr, order)
            cap = 0.25 * float(np.std(y[tr]))
            # (an algorithm that fails to fit raises: the weekly test stops and is retried on the next run, with the
            # months already done kept, rather than scoring the failure as "no prediction")
            f_all = fit("average", X[tr], y[tr], periods(made[tr], h), b0)
            preds = {k: f_all._raw(k, X[te]) for k in AVAILABLE}
            preds["average"] = f_all.predict(X[te])         # the candidates that fitted (all of them, normally)
            preds["fusion"] = np.mean([preds[k] for k in fset], axis=0)
            # scored exactly as it would be used: centred on the average of the stocks it is compared with, then capped
            g = {c: round(1 - float(np.mean((y[te] - np.clip(_centered(preds[c], grp[te]), -cap, cap)) ** 2)) / m0, 7)
                 for c in CANDIDATES}
            folds[fk] = {"key": key, "gains": g, "fset": fset}
        fused.append(n_known >= 3)
        for c in CANDIDATES:
            gains[c].append(float(g[c]))
        tested.append(m)
        known_by.append(_outcome_known(str(made[te[-1]]), h))
    for fk in [k for k in folds if k.startswith(f"{h}:") and k not in kept]:
        del folds[fk]
    k = len(tested)
    res["test_months"] = k
    if k < min_test_months:
        res["status"] = f"learning: {k} of {min_test_months} test months so far"
        return res
    picks = _policy(gains, tested, known_by, fused)
    meta = [gains[p][i] for i, p in enumerate(picks)]

    stats = _mean_t
    for c in CANDIDATES:
        mg, tc = stats(gains[c])
        res["candidates"][c] = {"gain": round(mg, 5), "t": round(tc, 2), "recent": round(float(np.mean(gains[c][-6:])), 5)}
    mean, t = stats(meta)
    recent = float(np.mean(meta[-6:]))
    best = max(CANDIDATES, key=lambda c: float(np.mean(gains[c])))
    n_fused = sum(fused)
    res["fusion"] = _fuse_set(gains, k)            # what the fused model would average now (by the whole record)
    res["fusion_months"] = n_fused
    if best == "fusion" and n_fused < FUSION_MIN_MONTHS:
        # not yet: the fusion needs FUSION_MIN_MONTHS months as a real fusion before it can be put in use
        res["fusion_waiting"] = FUSION_MIN_MONTHS - n_fused
        best = max([c for c in CANDIDATES if c != "fusion"], key=lambda c: float(np.mean(gains[c])))
    # few test months make t itself noisy: the bar is the stricter of 2.5 and the one-sided 0.5% point of Student's t
    from scipy.stats import t as student
    t_need = max(min_t, float(student.ppf(0.995, k - 1)))
    res.update(test_gain=round(mean, 5), t=round(t, 2), t_needed=round(t_need, 2), recent_gain=round(recent, 5),
               monthly_gains=[round(float(x), 4) for x in meta[-12:]], picks=picks[-12:], best=best,
               cap=round(0.25 * float(np.std(y)), 5), periods=periods(made, h), last_month=tested[-1])
    own_g, own_t = stats(gains[best])
    res.update(best_gain=round(own_g, 5), best_t=round(own_t, 2))
    if mean > min_gain and t >= t_need and recent > 0 and own_g > min_gain and own_t >= t_need:
        what = (f"the fused model ({', '.join(NAMES[b].lower() for b in res['fusion'])})" if best == "fusion"
                else NAMES[best].lower())
        res.update(accepted=True, model=best,
                   status=f"applied: {what} won the monthly test and improved forecasts it never saw")
    elif mean > min_gain and t >= t_need and recent > 0:
        # the month-by-month choices passed, but the one that would be put in use hasn't passed on its own record
        res["status"] = (f"rejected: the monthly choices passed, but {NAMES.get(best, best).lower()}, which would be "
                         f"put in use, hasn't on its own record (t {own_t:.1f}; {t_need:.1f} needed)")
    else:
        res["status"] = "rejected: no algorithm reliably improved forecasts it never saw (no nudge is applied)"
    return res


_MODEL_CACHE: dict = {}


def _policy(gains: dict[str, list[float]], tested: list[str], known_by: list[str], fused: list[bool]) -> list[str]:
    """The algorithm used for each test month: the one with the best record on the earlier test months whose
    outcomes were all in when it began ("average" until three are), and the fused model only once it had
    FUSION_MIN_MONTHS of those as a real fusion. The same rule decides what is put in use, so the test scores
    exactly the policy that is deployed."""
    picks = []
    for i in range(len(tested)):
        known = sum(1 for j in range(i) if known_by[j] < tested[i] + "-01")
        pool = [c for c in CANDIDATES if c != "fusion" or sum(fused[:known]) >= FUSION_MIN_MONTHS]
        picks.append("average" if known < 3 else max(pool, key=lambda c: float(np.mean(gains[c][:known]))))
    return picks


def _fuse_set(gains: dict[str, list[float]], n: int) -> list[str]:
    """The algorithms the fused model averages, chosen on the first `n` test months: the `FUSE_K` with the best
    average gain there (all of them until three months are in). Ties go to the simpler algorithm."""
    if n < 3:
        return list(AVAILABLE)
    ranked = sorted(AVAILABLE, key=lambda c: (-float(np.mean(gains[c][:n])), AVAILABLE.index(c)))
    return sorted(ranked[:FUSE_K], key=AVAILABLE.index)


def _mean_t(g: list[float]) -> tuple[float, float]:
    """(mean, t) of monthly test gains. The standard error is the larger of the plain one inflated by √2 (a month's
    outcomes run into the next month's) and a Newey-West one over four months, which also catches slower runs of
    luck."""
    a = np.asarray(g, dtype=float)
    n = len(a)
    if n < 2:
        return float(a.mean()) if n else 0.0, 0.0
    d = a - a.mean()
    lrv = float(np.mean(d * d))
    for lag in range(1, min(4, n - 1) + 1):
        lrv += 2 * (1 - lag / 5) * float(np.sum(d[lag:] * d[:-lag])) / n
    se = max(float(a.std(ddof=1)) / math.sqrt(n) * math.sqrt(2), math.sqrt(max(lrv, 0.0) / n))
    return float(a.mean()), (float(a.mean()) / se if se > 0 else 0.0)


def _outcome_known(made: str, h: int) -> str:
    """The day a forecast made on `made` for `h` sessions is graded."""
    from .forecast import trading_days_after
    return trading_days_after(dt.date.fromisoformat(made[:10]), h)[-1].isoformat()


def current(tw: dict | None) -> bool:
    """Whether a learner decision was made by this version's test: the same signals, the same candidates and
    settings. An older one (say, "average" when it averaged three models, not seven) is never applied; a fused one
    must name members this install has."""
    if not (bool(tw) and tw.get("fv") == FEAT_V and tw.get("hp") == [HP_V, CANDIDATES] and tw.get("model") in CANDIDATES
            and tw.get("feats", FEATS) == FEATS):
        return False
    if tw["model"] == "fusion":
        fs = tw.get("fusion") or []
        return 1 <= len(fs) <= len(AVAILABLE) and set(fs) <= set(AVAILABLE)
    return True


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
        if not tw or not tw.get("accepted") or not tw.get("on") or not current(tw):
            return None
        h = int(tw.get("h") or 21)
        rows = rows_for(L, h)
        if len(rows) < 100:
            return None
        bases = list(tw.get("fusion") or []) if tw["model"] == "fusion" else None
        key = (tw["model"], tuple(bases or ()), h, len(rows), rows[-1]["made"], rows[0]["made"])
        if key not in _MODEL_CACHE:
            X = _matrix(rows, FEATS)
            made, _, y = targets(rows, L, h, row_groups(rows, L, h))
            idx = _sample(np.arange(len(rows)), _hash_order([f"{r['t']}:{r['made']}" for r in rows]))
            _MODEL_CACHE.clear()
            _MODEL_CACHE[key] = fit(tw["model"], X[idx], y[idx], periods(made, h), b0, bases=bases)
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


MISS_PRIOR, MISS_K = 0.2, 10.0       # an 80% range misses 20% of the time; a stock's own record counts as n/(n+10)


def _earn_miss(r: dict) -> int:
    """A graded forecast that missed its range because of an earnings report (the Results tab's breakdown)."""
    return int(not r.get("in80", 1) and (r.get("why") or {}).get("c") == "earnings")


def miss_rates(rows: list[dict], history: list[dict], h: int, flag=None, prior: float = MISS_PRIOR) -> np.ndarray:
    """For each row, how often the same stock's raw 80% range had missed among its forecasts graded before that row
    was made (outcome date earlier than the row's date: known then), shrunk toward 20% by n/(n + 10). The stocks
    the model keeps getting wrong get wider ranges from the range learner, if the test says it helps. With `flag`
    (a row → 0/1) and its `prior`, the same for another kind of miss (those caused by earnings, say)."""
    from .forecast import trading_days_after
    flag = flag or (lambda r: 1 - int(r["in80r"]))
    by_t: dict[str, list[tuple[str, int]]] = {}
    ends: dict[str, str] = {}
    for r in history:
        if "in80r" not in r:
            continue
        d = r["made"]
        if d not in ends:
            ends[d] = trading_days_after(dt.date.fromisoformat(d), h)[-1].isoformat()
        # known once graded: the later of the calendar date and the window's recorded end (a stock whose prices skip
        # sessions is graded later than the calendar says)
        e = max(ends[d], str(r.get("end") or "")[:10])
        by_t.setdefault(r["t"], []).append((e, flag(r)))
    cum: dict[str, tuple[list[str], np.ndarray]] = {}
    for t, lst in by_t.items():
        lst.sort()
        cum[t] = ([e for e, _ in lst], np.cumsum([m for _, m in lst]))
    import bisect
    out = np.full(len(rows), prior)
    for i, r in enumerate(rows):
        got = cum.get(r["t"])
        if not got:
            continue
        k = bisect.bisect_left(got[0], r["made"])          # graded strictly before this forecast was made
        if k:
            out[i] = (float(got[1][k - 1]) + prior * MISS_K) / (k + MISS_K)
    return out


def miss_now(history: list[dict], h: int, today: dt.date) -> dict[str, dict[str, float]]:
    """Each stock's current miss rates (every graded forecast counts: all their outcomes are known): {"miss": …,
    "emiss": …}."""
    probe = [{"t": t, "made": (today + dt.timedelta(days=1)).isoformat()} for t in sorted({r["t"] for r in history})]
    return {"miss": {p["t"]: float(v) for p, v in zip(probe, miss_rates(probe, history, h))},
            "emiss": {p["t"]: float(v) for p, v in zip(probe, miss_rates(probe, history, h, _earn_miss, 0.0))}}


def _range_matrix(rows: list[dict], history: list[dict], h: int) -> np.ndarray:
    # (the earnings one is shrunk toward zero, not toward the record's rate: a stock with none then reads 0 however
    # many forecasts it has had, so the input can't stand in for how long a stock has been followed)
    return np.hstack([_matrix(rows, RANGE_FEATS), miss_rates(rows, history, h)[:, None],
                      miss_rates(rows, history, h, _earn_miss, 0.0)[:, None]])


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
                "status": "learning", "v": RANGE_V}
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
        hist = L.get("graded", {}).get(src, {}).get(str(h), [])
        model = _fit_q80(_range_matrix(old, hist, h), need_o, [r["made"] for r in old], h)
        q = model.predict(_range_matrix(new, hist, h))
        s_cond = np.clip(1 + (q - 1) * w_old, lo, hi)
        ref_scale = s_glob_old if g.get("status") == "applied" else 1.0
        loss_ref = _pinball3(er_n, w10_n, w90_n, ref_scale, sh_old)
        loss_cond = _pinball3(er_n, w10_n, w90_n, s_cond, sh_old)
        edge = 1 - loss_cond / loss_ref if loss_ref > 0 else 0.0
        res = {**base, "holdout_gain_vs_single": round(edge, 5), "holdout_n": len(new),
               "cov80_holdout_single": float(np.mean(need_n <= ref_scale)), "cov80_holdout": float(np.mean(need_n <= s_cond)),
               "scale_spread": [round(float(np.quantile(s_cond, 0.1)), 3), round(float(np.quantile(s_cond, 0.9)), 3)],
               "single_scale": round(float(g.get("scale", 1.0)), 4), "shift": round(shift, 5)}
        res["status"] = _range_status(edge, (prev or {}).get(name), min_edge, keep_edge)
        out[name] = res
    return out


def _range_status(edge: float, prev: dict | None, min_edge: float, keep_edge: float) -> str:
    """On when the test's edge clears the bar; one already on (by this version's test: an older version's decision
    starts over) stays on while it clears the lower bar."""
    was_on = (prev or {}).get("status") == "applied" and (prev or {}).get("v") == RANGE_V
    return "applied" if edge >= min_edge or (was_on and edge >= keep_edge) else "held"


class RangeModel:
    """The range learner in use: per horizon, a quantile model refit on the graded record when first needed."""

    def __init__(self, parts: dict, misses: dict | None = None):
        self.parts = parts                         # name → (model, weight, shift)
        self.misses = misses or {}                 # name → {"miss": {ticker: rate}, "emiss": {ticker: rate}}

    @classmethod
    def build(cls, rl: dict | None, calib: dict, L: dict, b0: float = 6.0, window_days: int = 460,
              today: dt.date | None = None) -> "RangeModel | None":
        parts, misses = {}, {}
        today = today or dt.date.today()
        for name, h in RANGE_HORIZONS:
            st = (rl or {}).get(name) or {}
            if st.get("status") != "applied" or st.get("v") != RANGE_V:     # decided by this version's test only
                continue
            src = "quant_bt" if st.get("from") == "backtest" else "quant"
            rows = range_rows(L, src, h, int(st.get("window") or max(window_days, block_days(h) * 10)))
            if len(rows) < 200:
                continue
            hist = L.get("graded", {}).get(src, {}).get(str(h), [])
            key = ("range", name, len(rows), rows[-1]["made"], rows[0]["made"], st.get("shift"), len(hist))
            if key not in _RANGE_CACHE:
                if len(_RANGE_CACHE) >= 2 * len(RANGE_HORIZONS):   # one run's worth (the record moves daily)
                    _RANGE_CACHE.clear()
                shift = float(st.get("shift") or 0.0)
                need, _, _, _ = _need(rows, shift)
                made = [r["made"] for r in rows]
                per = periods(made, h)
                _RANGE_CACHE[key] = (_fit_q80(_range_matrix(rows, hist, h), need, made, h), per / (per + b0), shift,
                                     miss_now(hist, h, today))
            model, w, shift, mm = _RANGE_CACHE[key]
            parts[name] = (model, w, shift)
            misses[name] = mm
        return cls(parts, misses) if parts else None

    def calib_for(self, feats: dict, fc: dict, calib: dict, lo: float = 0.6, hi: float = 2.0, t: str | None = None
                  ) -> dict:
        """The stock's own calibration: the single per-horizon stretch, replaced where the range learner is on."""
        out = dict(calib or {})
        for name, (model, w, shift) in self.parts.items():
            x = row_features(feats, fc, name)
            mm = self.misses.get(name) or {}
            X = np.asarray([[float(x[k]) if x.get(k) is not None and math.isfinite(float(x[k])) else np.nan
                             for k in RANGE_FEATS] + [(mm.get("miss") or {}).get(t, MISS_PRIOR),
                                                       (mm.get("emiss") or {}).get(t, 0.0)]])
            q = float(model.predict(X)[0])
            out[name] = {"status": "applied", "scale": round(float(np.clip(1 + (q - 1) * w, lo, hi)), 4),
                         "shift": shift, "cond": True}
        return out


_RANGE_CACHE: dict = {}


# ---------------------------------------------------------------- the coverage keeper (adaptive conformal)

KEEPER_HORIZONS = (("1d", 1), ("1w", 5), ("1m", 21), ("3m", 63))


def coverage_keeper(L: dict, prev: dict | None = None, target: float = 0.8, gamma: float = 0.05, lo: float = 0.67,
                    hi: float = 1.5, min_days: int = 40, min_edge: float = 0.002, window_days: int = 365) -> dict:
    """Adaptive conformal inference (Gibbs & Candès 2021) on the ranges the app actually published. Per horizon, a
    multiplier m on every range's width: when a day's forecasts are graded, log m moves by γ × (that day's miss rate
    − 20%), so more misses than an 80% range allows widen the next ranges and fewer narrow them. Over time coverage
    is pulled to 80% whatever the market does (the guarantee needs no model of why ranges miss).

    It is computed by replaying the live record in order: each day's forecasts get the multiplier built from the
    results known before that day (a forecast's result is known a horizon after it was made), with the multiplier
    they were actually published with taken out. The replay is also the test: the keeper is used only if the ranges
    it would have published scored better (10/50/90 pinball loss) than the ones without it, by at least `min_edge`
    to switch on; once on, it stays on while it does no worse. The score counts the last `window_days` of the
    replay (the keeper is for the market as it is now; what it would have done years ago matters less). γ is smaller
    for longer horizons, whose daily results overlap and so repeat each other (γ × √(spacing/horizon)), and a day
    with fewer forecasts than usual moves the multiplier less (its miss rate is noisier)."""
    from .forecast import trading_days_after
    out = {}
    for name, h in KEEPER_HORIZONS:
        rows = [r for r in L.get("graded", {}).get("quant", {}).get(str(h), [])
                if r.get("pw10") is not None and r.get("pw90") is not None and "err" in r and r.get("t") != BENCH]
        by_day: dict[str, list[dict]] = {}
        for r in rows:
            by_day.setdefault(r["made"], []).append(r)
        days = sorted(by_day)
        base = {"n": len(rows), "days": len(days), "status": "learning", "m": 1.0}
        if len(days) < min_days:
            out[name] = {**base, "need": {"days": min_days}}
            continue
        med_n = float(np.median([len(by_day[d]) for d in days]))
        full = [d for d in days if len(by_day[d]) >= 0.5 * med_n]     # a part-day (a short run) isn't a step
        spacing = max(1.0, (dt.date.fromisoformat(full[-1]) - dt.date.fromisoformat(full[0])).days * 5 / 7
                      / max(1, len(full) - 1))
        g = gamma * math.sqrt(min(1.0, spacing / h))
        ends = {d: trading_days_after(dt.date.fromisoformat(d), h)[-1].isoformat() for d in days}
        since = (dt.date.fromisoformat(days[-1]) - dt.timedelta(days=window_days)).isoformat()
        logm, pending = 0.0, []
        q_with = q_without = 0.0
        in_with = in_without = n_win = 0
        for d in days:
            pending.sort()
            while pending and pending[0][0] < d:           # results known before this day's close
                _, miss, wd = pending.pop(0)
                logm = float(np.clip(logm + g * wd * (miss - (1 - target)), math.log(lo), math.log(hi)))
            m = math.exp(logm)
            rr = by_day[d]
            err = np.asarray([r["err"] for r in rr], dtype=float)
            used = np.asarray([float(r.get("aci") or 1.0) for r in rr])
            w10 = np.maximum(np.asarray([r["pw10"] for r in rr], dtype=float) / used, 1e-6)
            w90 = np.maximum(np.asarray([r["pw90"] for r in rr], dtype=float) / used, 1e-6)
            inside = (err >= -m * w10) & (err <= m * w90)
            if d >= since:
                q_with += _pinball3(err, w10, w90, m, 0.0) * len(rr)
                q_without += _pinball3(err, w10, w90, 1.0, 0.0) * len(rr)
                in_with += int(inside.sum())
                in_without += int(((err >= -w10) & (err <= w90)).sum())
                n_win += len(rr)
            pending.append((ends[d], 1.0 - float(inside.mean()), min(1.0, len(rr) / med_n)))
        for _, miss, wd in sorted(pending):                # every row here is graded: all its results are known
            logm = float(np.clip(logm + g * wd * (miss - (1 - target)), math.log(lo), math.log(hi)))
        edge = 1 - q_with / q_without if q_without > 0 else 0.0
        was_on = ((prev or {}).get(name) or {}).get("status") == "applied"
        res = {**base, "m": round(math.exp(logm), 4), "gamma": round(g, 4), "replay_gain": round(edge, 5),
               "window_days": window_days, "window_n": n_win,
               "cov80_with": round(in_with / max(1, n_win), 4), "cov80_without": round(in_without / max(1, n_win), 4)}
        res["status"] = "applied" if edge >= min_edge or (was_on and edge >= 0) else "held"
        out[name] = res
    return out


def keeper_multipliers(keeper: dict | None) -> dict[str, float]:
    return {k: float(v["m"]) for k, v in (keeper or {}).items() if v.get("status") == "applied" and v.get("m")}


def _keep(calib: dict | None, aci: dict[str, float] | None) -> dict:
    """The calibration plus the coverage keeper's multipliers, as a factor of their own ("keep") at every horizon the
    keeper grades (1 where it is off). The forecast multiplies the calibration's stretch by it, each interpolated
    across horizons separately: the keeper scales whatever stretch a horizon has (its own or its neighbours'), never
    replaces it, and never moves a median or a horizon where it is off."""
    out = {k: dict(v) for k, v in (calib or {}).items()}
    if aci:
        for name, _ in KEEPER_HORIZONS:
            out.setdefault(name, {})["keep"] = round(float(aci.get(name, 1.0)), 4)
    return out


def forecast_with_learning(build, feats: dict, calib: dict, smodel: "SignalModel | None",
                           rmodel: "RangeModel | None", aci: dict[str, float] | None = None, t: str | None = None
                           ) -> tuple[dict | None, float]:
    """One stock's forecast with whatever the learners have earned the right to change. `build(calib=…, tilt_ret=…,
    tilt_days=…, tilt_parts=…)` makes the forecast. The range learner needs the forecast's own event dates (is an
    earnings report inside the window?), so when it is on the forecast is built twice. The coverage keeper's
    multipliers (`aci`, per horizon) go on top of either. Returns (forecast, tilt)."""
    tr, parts, td = smodel.tilt(feats) if smodel else (0.0, [], 21)
    fc = build(calib=_keep(calib, aci), tilt_ret=tr, tilt_days=td, tilt_parts=parts)
    if fc and rmodel:
        own = rmodel.calib_for(feats, fc, calib, t=t)
        if own != calib:
            fc = build(calib=_keep(own, aci), tilt_ret=tr, tilt_days=td, tilt_parts=parts) or fc
    return fc, tr


def version(tw: dict | None, rl: dict | None, keeper: dict | None = None) -> str:
    """Changes whenever what the learners would apply changes (so cached forecasts are rebuilt). The keeper's
    multipliers count to the nearest percent."""
    a = ((tw or {}).get("model"), tuple((tw or {}).get("fusion") or ()) if (tw or {}).get("model") == "fusion" else ()) \
        if (tw or {}).get("on") and (tw or {}).get("accepted") and current(tw) else None
    b = {k: (v.get("status"), v.get("v")) for k, v in (rl or {}).items()}
    c = {k: round(m, 2) for k, m in keeper_multipliers(keeper).items()}
    return hashlib.sha1(repr((a, sorted(b.items()), sorted(c.items()), FEAT_V)).encode()).hexdigest()[:8]
