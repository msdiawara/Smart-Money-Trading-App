"""Measured predictability.

Every number here answers one question as honestly as the data allows: *when signals like this were disclosed
before, how often did the stock beat the S&P 500 (SPY) over the next 63 trading days (~3 months), and how sure
can we be about that?*

* Base rates: historical frequencies with 95% Wilson intervals and sample sizes.
* A model for insider purchases (the largest official data set): trained on older filings, tested on newer
  ones it never saw (with a gap so the 3-month outcomes can't overlap). If it can't beat simply quoting the
  base rate on that test, the app says "no measurable predictability" and falls back to the base rate.
* Projection ranges: where the price has typically landed after 3 months, given how much it swings.
  Ranges, not targets: they carry no view on direction.
* Screen backtests: how the price-based idea screens did in every past month we have data for.

Entry is always the close of the first trading day *after* the filing/disclosure date (what a copier could
actually do); the benchmark is measured over exactly the same days.
"""
from __future__ import annotations

import bisect
import datetime as dt
import math
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from .util import date_str, log, now_et, now_utc, parse_date, read_json

HOLD = 63
BENCH = "SPY"
Z95 = 1.96

# Cross-sectional features only. Market-wide features (index trend, index return) are the same for every filing on
# a given day, so a few months of test data can't validate them; they'd mostly learn noise about market timing.
INSIDER_FEATURES = ["max_rank", "n_owners", "log_value", "cluster30", "routine_share", "plan",
                    "r1m", "r3m", "mom6_1", "vol3m", "dist_hi", "dist_lo", "log_adv"]
FEATURE_LABELS = {
    "max_rank": "seniority of the buyer", "n_owners": "insiders on the same filing", "log_value": "size of the purchase",
    "cluster30": "insiders buying within 30 days", "routine_share": "routine (same month every year) buyers",
    "plan": "pre-planned (10b5-1) purchase", "r1m": "1-month price move", "r3m": "3-month price move",
    "mom6_1": "6-month momentum", "vol3m": "volatility", "dist_hi": "distance below 52-week high",
    "dist_lo": "distance above 52-week low", "log_adv": "trading volume (company size proxy)",
    "spy_r3m": "S&P 500 3-month move", "spy_trend": "S&P 500 above its 200-day average",
}


# ---------------------------------------------------------------- statistics helpers

def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    c = p + z * z / (2 * n)
    adj = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - adj) / den), min(1.0, (c + adj) / den)


BLOCK_DAYS = 91      # ~13 weeks: one 63-trading-day outcome window


def block_of(d) -> int | None:
    d = d if isinstance(d, dt.date) else parse_date(d)
    return d.toordinal() // BLOCK_DAYS if d else None


def clustered_interval(outcomes, blocks) -> tuple[float, float, int]:
    """95% interval for a share when outcomes from the same ~quarter move together (they share the market's and
    sectors' moves over overlapping 3-month windows). Cluster-robust standard error over 13-week blocks and a t
    critical value on half the blocks' degrees of freedom, because neighbouring blocks overlap in their outcome
    windows too. In simulation with a common factor (docs/METHODS.md) this covered the true rate 95% of the time,
    where a plain Wilson interval covered it about a third of the time. Never narrower than Wilson.
    Returns (lo, hi, number of blocks)."""
    from scipy.stats import t as student_t
    y = np.asarray(outcomes, dtype=float)
    n = len(y)
    k = int(y.sum())
    wl, wh = wilson(k, n)
    keys = [b for b in blocks]
    uniq = sorted(set(keys))
    G = len(uniq)
    if G < 3:
        return wl, wh, G
    p = y.mean()
    idx = {b: i for i, b in enumerate(uniq)}
    s = np.zeros(G)
    np.add.at(s, [idx[b] for b in keys], y - p)
    se = math.sqrt(G / (G - 1) * float(np.sum(s ** 2)) / (n * n))
    tc = float(student_t.ppf(0.975, max(1, (G - 1) // 2)))
    return min(wl, max(0.0, p - tc * se)), max(wh, min(1.0, p + tc * se)), G


def rate_summary(outcomes: list[bool], excess: list[float] | None = None, dates: list | None = None) -> dict:
    n = len(outcomes)
    if not n:
        return {"n": 0}
    k = int(sum(outcomes))
    if dates is not None and len(dates) == n:
        lo, hi, g = clustered_interval(outcomes, [block_of(d) for d in dates])
    else:
        (lo, hi), g = wilson(k, n), None
    out = {"n": n, "p": k / n, "lo": lo, "hi": hi, "periods": g, "few_periods": g is not None and g < 3}
    if excess:
        ex = np.asarray(excess, dtype=float)
        out.update(mean_excess=float(np.mean(ex)), median_excess=float(np.median(ex)))
    return out


def auc_se(auc: float, n_pos: int, n_neg: int) -> float:
    """Standard error of an AUC (Hanley & McNeil 1982). Test filings cluster in time, so this is optimistic;
    it's used only as a minimum bar, together with beating the base rate on log loss."""
    if n_pos < 1 or n_neg < 1:
        return 1.0
    q1, q2 = auc / (2 - auc), 2 * auc * auc / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc * auc) + (n_neg - 1) * (q2 - auc * auc)) / (n_pos * n_neg)
    return math.sqrt(max(var, 1e-12))


class StratAUC:
    """AUC counted only over pairs in the same stratum (ties count half), optionally with observation weights.
    The sort is done once, so re-weighting for a bootstrap costs O(n) per draw."""

    def __init__(self, y: np.ndarray, p: np.ndarray, groups) -> None:
        y = np.asarray(y).astype(bool)
        p = np.asarray(p, dtype=float)
        _, g = np.unique(np.asarray(groups).astype(str), return_inverse=True)
        o = np.lexsort((p, g))
        self.o = o
        gs, ps = g[o], p[o]
        self.ys = y[o]
        new_blk = np.r_[True, (gs[1:] != gs[:-1]) | (ps[1:] != ps[:-1])]
        self.blk = np.cumsum(new_blk) - 1
        first = np.flatnonzero(new_blk)
        bg = gs[first]
        new_grp = np.r_[True, bg[1:] != bg[:-1]]
        self.grp_of_blk = np.cumsum(new_grp) - 1
        self.grp_start = np.flatnonzero(new_grp)
        self.nb = len(first)

    def __call__(self, w: np.ndarray | None = None) -> float | None:
        ws = np.ones(len(self.o)) if w is None else np.asarray(w, dtype=float)[self.o]
        neg = np.bincount(self.blk, weights=ws * ~self.ys, minlength=self.nb)
        pos = np.bincount(self.blk, weights=ws * self.ys, minlength=self.nb)
        cum = np.cumsum(neg)
        before_grp = np.r_[0.0, cum][self.grp_start][self.grp_of_blk]
        below = cum - neg - before_grp
        num = float(np.sum(pos * (below + 0.5 * neg)))
        den = float(np.sum(np.bincount(self.grp_of_blk, weights=pos) * np.bincount(self.grp_of_blk, weights=neg)))
        return num / den if den > 0 else None


def within_week_auc(y: np.ndarray, p: np.ndarray, groups: np.ndarray) -> float | None:
    """Ranking skill among filings in the same stratum (by default the same week; the gate uses week x size).
    Whether a stock beats the S&P 500 depends heavily on what the whole market does next, which no filing
    predicts; comparing only same-week filings removes that shared market move and measures pure stock-picking."""
    if len(y) == 0:
        return None
    return StratAUC(y, p, groups)()


def pigeonhole_bootstrap(y: np.ndarray, p: np.ndarray, p_ref: np.ndarray, weeks: list[str], tickers: list[str],
                         strata, reps: int = 400, seed: int = 0) -> dict:
    """95% intervals for the stratified AUC and the Brier skill that respect two kinds of dependence at once:
    filings in the same week share the market's move, and filings in the same stock (an insider buying spree,
    several insiders in a row) share one outcome. Weeks and stocks are resampled independently and each filing
    gets the product of the two draws (Owen's pigeonhole bootstrap, which is conservative for this case)."""
    rng = np.random.default_rng(seed)
    _, wi = np.unique(np.asarray(weeks), return_inverse=True)
    _, ti = np.unique(np.asarray(tickers), return_inverse=True)
    nw, nt = int(wi.max()) + 1, int(ti.max()) + 1
    auc = StratAUC(y, p, strata)
    yf = np.asarray(y, dtype=float)
    e1, e0 = (np.asarray(p) - yf) ** 2, (np.asarray(p_ref) - yf) ** 2
    aucs, skills = [], []
    for _ in range(reps):
        w = rng.poisson(1.0, nw)[wi] * rng.poisson(1.0, nt)[ti]
        if w.sum() < 10:
            continue
        a = auc(w)
        if a is None:
            continue
        aucs.append(a)
        b, b0 = float(np.dot(w, e1)), float(np.dot(w, e0))
        skills.append(1 - b / b0 if b0 > 0 else 0.0)
    if len(aucs) < reps // 2:
        return {"auc_ci": [0.0, 1.0], "skill_ci": [-1.0, 1.0], "weeks": nw, "stocks": nt}
    return {"auc_ci": [float(np.quantile(aucs, 0.025)), float(np.quantile(aucs, 0.975))],
            "skill_ci": [float(np.quantile(skills, 0.025)), float(np.quantile(skills, 0.975))], "weeks": nw,
            "stocks": nt}


def _label_in_force(has_skill: bool, auc: float | None, skill: float | None) -> str:
    if not has_skill:
        return "None detected"
    lab = predictability_label(auc, skill)
    return "Low (weaker in today's re-check)" if lab == "None detected" else lab


def predictability_label(auc: float | None, skill: float | None) -> str:
    if auc is None or skill is None or auc < 0.52 or skill <= 0:
        return "None detected"
    if auc < 0.55:
        return "Low"
    if auc < 0.60:
        return "Moderate"
    return "High (unusually high: treat with caution)"


# ---------------------------------------------------------------- price panel (whole-market history)

class Panel:
    """Read-only numpy view of the whole-market daily closes (MarketMatrix) for fast look-ups."""

    def __init__(self, matrix):
        self.dates = np.asarray(matrix.dates, dtype=np.int64)
        self.close = matrix.close
        self.volume = matrix.vol
        self.col = matrix.col
        self.T = len(self.dates)

    def ok(self) -> bool:
        return self.T > 130 and BENCH in self.col

    def ix_on_or_before(self, d: dt.date) -> int | None:
        k = bisect.bisect_right(self.dates, d.toordinal()) - 1
        return k if k >= 0 else None

    def ix_after(self, d: dt.date) -> int | None:
        k = bisect.bisect_right(self.dates, d.toordinal())
        return k if k < self.T else None

    def c(self, t: str) -> np.ndarray | None:
        j = self.col.get(t)
        return None if j is None else self.close[:, j]

    def v(self, t: str) -> np.ndarray | None:
        j = self.col.get(t)
        return None if j is None else self.volume[:, j]

    def date(self, i: int) -> str:
        return dt.date.fromordinal(int(self.dates[i])).isoformat()


def _val(c: np.ndarray, i: int, back: int = 3) -> float | None:
    """Close at row i, or the nearest earlier valid one within `back` rows."""
    for k in range(i, max(-1, i - back - 1), -1):
        if 0 <= k < len(c) and not np.isnan(c[k]) and c[k] > 0:      # zero/negative prices are bad data
            return float(c[k])
    return None


def _val_fwd(c: np.ndarray, i: int, fwd: int = 3) -> tuple[int, float] | None:
    for k in range(i, min(len(c), i + fwd + 1)):
        if not np.isnan(c[k]) and c[k] > 0:
            return k, float(c[k])
    return None


def price_features(panel: Panel, t: str, i: int) -> dict | None:
    """Features known at the close of row i (no look-ahead)."""
    c, v, s = panel.c(t), panel.v(t), panel.c(BENCH)
    if c is None or s is None or i < 63:
        return None
    p = _val(c, i)
    if p is None or p <= 0:
        return None
    lo_i = max(0, i - 251)
    win = c[lo_i:i + 1]
    valid = win[~np.isnan(win)]
    if len(valid) < 60:
        return None

    def back(n: int) -> float | None:
        return _val(c, i - n) if i - n >= 0 else None

    p21, p63, p126 = back(21), back(63), back(126)
    lr = np.diff(np.log(valid[-64:]))
    vol = float(np.std(lr, ddof=1) * math.sqrt(252)) if len(lr) > 20 else None
    dv = (c[max(0, i - 19):i + 1] * v[max(0, i - 19):i + 1])
    dv = dv[~np.isnan(dv)]
    adv = float(np.mean(dv)) if len(dv) else None
    s_now, s_63 = _val(s, i), _val(s, i - 63)
    s_win = s[max(0, i - 199):i + 1]
    s_win = s_win[~np.isnan(s_win)]
    return {
        "price": p,
        "r1m": (p / p21 - 1) if p21 else None,
        "r3m": (p / p63 - 1) if p63 else None,
        "mom6_1": (p21 / p126 - 1) if (p21 and p126) else None,
        "vol3m": vol,
        "dist_hi": p / float(np.max(valid)) - 1,
        "dist_lo": p / float(np.min(valid)) - 1,
        "log_adv": math.log10(adv + 1) if adv else None,
        "adv": adv,
        "spy_r3m": (s_now / s_63 - 1) if (s_now and s_63) else None,
        "spy_trend": (None if not s_now or len(s_win) < 150 else (1.0 if s_now > float(np.mean(s_win)) else 0.0)),
    }


def _last_valid(c: np.ndarray, lo: int, hi: int) -> int | None:
    seg = c[lo:hi + 1]
    ok = np.flatnonzero(~np.isnan(seg) & (seg > 0))
    return lo + int(ok[-1]) if len(ok) else None


def forward_excess(panel: Panel, t: str, j: int, hold: int = HOLD, info: dict | None = None) -> tuple[float, float] | None:
    """(stock return, stock minus SPY) entering at row j's close, exiting `hold` rows later.

    A stock that stops trading for good inside the window (taken over, delisted, bankrupt, renamed) is measured to
    its last traded close against SPY over the same days, i.e. the proceeds are assumed to sit in the index
    afterwards. Dropping those stocks instead would leave out many of the worst outcomes (survivorship bias).
    `info["delisted"]` is set when that happened."""
    c, s = panel.c(t), panel.c(BENCH)
    if c is None or s is None or j + hold >= panel.T:
        return None
    a = _val_fwd(c, j, 3)
    if not a:
        return None
    j0 = a[0]                       # same entry and exit rows for the stock and the benchmark
    k_exit = j0 + hold
    if k_exit >= panel.T:
        return None
    b = _val(c, k_exit, 5)
    k_end = k_exit
    if b is None:
        k_last = _last_valid(c, j0 + 1, k_exit)
        if k_last is None or _last_valid(c, k_exit, panel.T - 1) is not None:
            return None             # a gap in the data, not a delisting: can't grade (yet)
        b, k_end = float(c[k_last]), k_last
        if info is not None:
            info["delisted"] = True
    sa, sb = _val(s, j0), _val(s, k_end)
    if not sa or not sb:
        return None
    r = b / a[1] - 1
    return r, r - (sb / sa - 1)


def delisted_names(panel: Panel, quiet_rows: int = 5) -> set[str]:
    """Common-stock-looking tickers that stopped trading before the end of the stored history."""
    valid = ~np.isnan(panel.close)
    if not len(valid):
        return set()
    last = panel.T - 1 - np.argmax(valid[::-1], axis=0)
    has = valid.any(axis=0)
    pat = re.compile(r"^[A-Z]{1,5}(\.[A-Z])?$")       # letters, optionally a share class (BRK.B)
    out = set()
    for t, j in panel.col.items():
        if t == BENCH or not has[j] or last[j] >= panel.T - 1 - quiet_rows or not pat.match(t):
            continue
        if len(t) == 5 and t[-1] in "WURQ":          # warrants, units, rights, old bankruptcy suffix
            continue
        out.add(t)
    return out


# ---------------------------------------------------------------- insiders: history → data set

_CEO = re.compile(r"\b(chief executive|ceo)\b", re.I)
_SENIOR = re.compile(r"\b(chief financial|cfo|principal financial|president|chief operating|coo|chairman|chairwoman|"
                     r"executive chair|chair of the board)\b", re.I)
_CSUITE = re.compile(r"\b(chief [a-z]+ officer|evp|executive vice president|svp|senior vice president|general counsel)\b",
                     re.I)


def dera_rank(relationship: str | None, title: str | None) -> int:
    """Same seniority scale as the live Form 4 parser: 5 CEO, 4 Chair/CFO/President/COO, 3 C-suite or
    director, 2 other officer or 10% owner, 1 other."""
    t, rel = title or "", (relationship or "").lower()
    if _CEO.search(t):
        return 5
    if _SENIOR.search(t):
        return 4
    if _CSUITE.search(t) or "director" in rel:
        return 3
    if "officer" in rel or "ten" in rel or "10%" in rel:
        return 2
    return 1


def load_dera_rows(heavy_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for p in sorted((heavy_dir / "dera").glob("*.json.gz")):
        part = read_json(p) or []
        if isinstance(part, list):
            rows.extend(part)
    return rows


def months_window_start(months: dict) -> str | None:
    """First month covered by the insider trade history (YYYY-MM)."""
    w = months.get("_window_start")
    if isinstance(w, str):
        return w
    firsts = [v[0] for k, v in months.items() if not k.startswith("_") and isinstance(v, list) and v]
    return min(firsts) if firsts else None


def _routine(months: dict, owner: str, d: dt.date, window_start: str | None = None) -> float:
    """Cohen-Malloy-Pomorski: 1 = traded in the same calendar month in each of the 3 prior years, 0 = has traded in
    each of those years but not always that month, 0.5 = can't tell (too little history, or the month three years
    back is before the data starts)."""
    hist = months.get(f"cik:{owner}") or []
    ym = set(hist)
    ws = window_start or months_window_start(months)
    if not ws or f"{d.year - 3}-{d.month:02d}" < ws:
        return 0.5
    years = {int(x[:4]) for x in ym if int(x[:4]) < d.year}
    if not {d.year - 1, d.year - 2, d.year - 3} <= years:
        return 0.5                      # unknown
    return 1.0 if all(f"{d.year - k}-{d.month:02d}" in ym for k in (1, 2, 3)) else 0.0


def insider_dataset(rows: list[dict], panel: Panel, months: dict, code: str = "P") -> list[dict]:
    """One sample per (ticker, filing date): insider features + price features at the filing-day close +
    the outcome over the following 63 trading days (None if not known yet)."""
    trades = [r for r in rows if r.get("code") == code and r.get("ticker") and (r.get("value") or 0) >= 1000]
    groups: dict[tuple, list] = defaultdict(list)
    by_ticker: dict[str, list] = defaultdict(list)
    for r in trades:
        t = r["ticker"].replace(" ", "").upper()
        f = parse_date(r.get("filed"))
        if not f or t not in panel.col:
            continue
        groups[(t, f)].append(r)
        by_ticker[t].append((f, r.get("owner") or ""))
    for t in by_ticker:
        by_ticker[t].sort()
    ws = months_window_start(months)
    out = []
    for (t, f), g in groups.items():
        i = panel.ix_on_or_before(f)
        if i is None:
            continue
        pf = price_features(panel, t, i)
        if not pf:
            continue
        owners = {r.get("owner") for r in g}
        lst = by_ticker[t]
        a = bisect.bisect_left(lst, (f - dt.timedelta(days=30), ""))
        b = bisect.bisect_right(lst, (f, "￿"))
        cluster = len({o for _, o in lst[a:b]})
        rs = [_routine(months, r.get("owner") or "", parse_date(r.get("traded")) or f, ws) for r in g]
        feat = {
            "max_rank": max(dera_rank(r.get("role"), r.get("title")) for r in g),
            "n_owners": len(owners),
            "log_value": math.log10(sum(r.get("value") or 0 for r in g) + 1),
            "cluster30": cluster,
            "routine_share": float(np.mean(rs)) if rs else 0.5,
            "plan": 1.0 if any(r.get("plan") for r in g) else 0.0,
            **{k: pf[k] for k in ("r1m", "r3m", "mom6_1", "vol3m", "dist_hi", "dist_lo", "log_adv", "spy_r3m",
                                  "spy_trend")},
        }
        j = panel.ix_after(f)
        fx = forward_excess(panel, t, j) if j is not None else None
        out.append({"ticker": t, "filed": f.isoformat(), "x": feat,
                    "y": (fx[1] > 0) if fx else None, "excess": fx[1] if fx else None, "ret": fx[0] if fx else None})
    out.sort(key=lambda s: s["filed"])
    return out


def _matrix(samples: list[dict], feats: list[str]) -> np.ndarray:
    return np.asarray([[np.nan if s["x"].get(k) is None else float(s["x"][k]) for k in feats] for s in samples],
                      dtype=float)


# ---------------------------------------------------------------- the model

GATE_SPEC = "2026-10 week-x-size strata, pigeonhole bootstrap, both halves"


def train_model(samples: list[dict], feats: list[str] = INSIDER_FEATURES, min_n: int = 400,
                decision: bool | None = None) -> tuple[dict, dict | None]:
    """Time-ordered evaluation: train on the oldest ~70%, skip ~3 months so outcomes can't overlap, test on the
    newest filings. Returns (public summary, private bundle for live predictions or None).

    `decision` is a pass/fail already in force (the app re-decides once per calendar quarter, so daily retraining
    doesn't give a lucky streak 90 chances to pass); today's numbers are still computed and reported."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    done = [s for s in samples if s["y"] is not None]
    if len(done) < min_n:
        return {"status": "insufficient", "n": len(done),
                "note": f"Only {len(done)} past filings with a known 3-month outcome; need {min_n} to test a model."}, None
    X = _matrix(done, feats)
    y = np.asarray([1 if s["y"] else 0 for s in done])
    dates = [s["filed"] for s in done]
    cut = int(len(done) * 0.7)
    gap_to = (parse_date(dates[cut]) + dt.timedelta(days=100)).isoformat()
    test_start = next((k for k in range(cut, len(done)) if dates[k] >= gap_to), len(done))
    if len(done) - test_start < 150:
        cut = int(len(done) * 0.6)
        gap_to = (parse_date(dates[cut]) + dt.timedelta(days=100)).isoformat()
        test_start = next((k for k in range(cut, len(done)) if dates[k] >= gap_to), len(done))
    if len(done) - test_start < 100 or cut < 250:
        return {"status": "insufficient", "n": len(done),
                "note": "Not enough recent history to hold out a fair test period yet."}, None
    Xtr, ytr, Xte, yte = X[:cut], y[:cut], X[test_start:], y[test_start:]
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)

    def fill(A: np.ndarray, m: np.ndarray) -> np.ndarray:
        A = A.copy()
        idx = np.where(np.isnan(A))
        A[idx] = np.take(m, idx[1])
        return A

    Ftr, Fte = fill(Xtr, med), fill(Xte, med)
    mu, sd = Ftr.mean(axis=0), Ftr.std(axis=0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    base = float(ytr.mean())
    p0 = np.full(len(yte), base)
    ref = {"brier": float(brier_score_loss(yte, p0)), "log_loss": float(log_loss(yte, p0, labels=[0, 1]))}
    cands = {}
    lr = LogisticRegression(C=0.3, max_iter=3000)
    lr.fit((Ftr - mu) / sd, ytr)
    cands["logistic"] = (lr, lr.predict_proba((Fte - mu) / sd)[:, 1])
    hgb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.04, max_iter=250, min_samples_leaf=60,
                                         l2_regularization=1.0, early_stopping=True, validation_fraction=0.2,
                                         random_state=0)
    hgb.fit(Ftr, ytr)
    cands["gradient_boosting"] = (hgb, hgb.predict_proba(Fte)[:, 1])
    tested = {}
    for name, (_, p) in cands.items():
        p = np.clip(p, 1e-4, 1 - 1e-4)
        tested[name] = {"auc": float(roc_auc_score(yte, p)) if len(set(yte)) > 1 else None,
                        "brier": float(brier_score_loss(yte, p)), "log_loss": float(log_loss(yte, p, labels=[0, 1]))}
        tested[name]["skill"] = 1 - tested[name]["brier"] / ref["brier"] if ref["brier"] else None
    # One pre-registered model is judged (no picking the winner on the test data, which would flatter it); the
    # gradient-boosting model is reported alongside for comparison only.
    best = "logistic"
    weeks = [f"{parse_date(dates[test_start + k]).isocalendar()[0]}-{parse_date(dates[test_start + k]).isocalendar()[1]:02d}"
             for k in range(len(yte))]
    tick = [done[test_start + k].get("ticker") or f"?{k}" for k in range(len(yte))]
    # Compare filings only with others from the same week AND the same size third: a size factor (small caps
    # beating large caps for a few months) can't then pass for stock-picking skill.
    if "log_adv" in feats:
        la = Fte[:, feats.index("log_adv")]
        terc = np.digitize(la, np.quantile(la, [1 / 3, 2 / 3]))
    else:
        terc = np.zeros(len(yte), dtype=int)
    strata = np.asarray([f"{w}|{s}" for w, s in zip(weeks, terc)])
    p_b = np.clip(cands[best][1], 1e-4, 1 - 1e-4)
    boot = pigeonhole_bootstrap(yte, p_b, p0, weeks, tick, strata)
    auc_ww = within_week_auc(yte, p_b, strata)
    # Skill must be real on data the model never saw: a better Brier score than quoting the training base rate,
    # and a stratified AUC whose 95% interval (resampling weeks and stocks) stays above a coin flip. Otherwise the
    # app says "no measurable predictability" and shows base rates only. (Brier rather than log loss for the gate:
    # base rates drift between periods and log loss punishes that level shift hard enough to hide genuine ranking
    # skill; both are reported.)
    half = len(yte) // 2
    halves = [within_week_auc(yte[:half], p_b[:half], strata[:half]),
              within_week_auc(yte[half:], p_b[half:], strata[half:])]
    # Real skill has to show up in both the earlier and the later half of the test period, not just on average.
    big_enough = len(yte) >= 300 and boot["weeks"] >= 10 and boot["stocks"] >= 60
    passed = bool(big_enough and tested[best]["brier"] < ref["brier"] - 1e-4 and boot["auc_ci"][0] > 0.5
                  and auc_ww is not None and auc_ww > 0.5 and all(h is not None and h > 0.5 for h in halves))
    has_skill = passed if decision is None else bool(decision)
    p_best = np.clip(cands[best][1], 1e-4, 1 - 1e-4)
    # calibration on the held-out period, and what following the top / bottom fifth would have done
    order = np.argsort(p_best)
    bins = []
    for chunk in np.array_split(order, 5):
        if not len(chunk):
            continue
        k = int(yte[chunk].sum())
        lo, hi = wilson(k, len(chunk))
        bins.append({"pred": float(p_best[chunk].mean()), "obs": k / len(chunk), "lo": lo, "hi": hi, "n": int(len(chunk))})
    ex_te = np.asarray([done[test_start + k]["excess"] for k in range(len(yte))], dtype=float)
    q = max(1, len(order) // 5)
    top, bottom = order[-q:], order[:q]
    summary = {
        "status": "ok", "chosen": best if has_skill else "base_rate", "has_skill": bool(has_skill), "judged": best,
        "n_train": int(cut), "n_test": int(len(yte)), "train_period": [dates[0], dates[cut - 1]],
        "test_period": [dates[test_start], dates[-1]], "base_rate_train": base,
        "base_rate_test": float(yte.mean()), "reference": ref, "tested": tested,
        "auc": auc_ww if auc_ww is not None else tested[best]["auc"], "auc_pooled": tested[best]["auc"],
        "auc_ci": boot["auc_ci"], "skill_ci": boot["skill_ci"], "test_weeks": boot["weeks"],
        "test_stocks": boot["stocks"], "auc_halves": halves, "test_big_enough": big_enough,
        "passed_today": passed, "gate_spec": GATE_SPEC,
        "skill": tested[best]["skill"],
        "predictability": _label_in_force(has_skill, auc_ww, tested[best]["skill"]),
        "calibration": bins,
        "top_fifth": {"beat_rate": float(yte[top].mean()), "mean_excess": float(np.nanmean(ex_te[top])), "n": int(len(top))},
        "bottom_fifth": {"beat_rate": float(yte[bottom].mean()), "mean_excess": float(np.nanmean(ex_te[bottom])),
                         "n": int(len(bottom))},
        "features": feats, "trained": now_utc().isoformat(),
    }
    if best == "logistic":
        summary["effects"] = sorted(({"feature": f, "label": FEATURE_LABELS.get(f, f), "weight": float(w)}
                                     for f, w in zip(feats, lr.coef_[0])), key=lambda d: -abs(d["weight"]))
    if not has_skill:
        return summary, None
    # refit the chosen model on everything for live use
    Xall = fill(X, med)
    mu_a, sd_a = Xall.mean(axis=0), Xall.std(axis=0)
    sd_a = np.where(sd_a > 1e-9, sd_a, 1.0)
    if best == "logistic":
        final = LogisticRegression(C=0.3, max_iter=3000).fit((Xall - mu_a) / sd_a, y)
    else:
        final = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.04, max_iter=250, min_samples_leaf=60,
                                               l2_regularization=1.0, early_stopping=True, validation_fraction=0.2,
                                               random_state=0).fit(Xall, y)
    p_all = (final.predict_proba((Xall - mu_a) / sd_a)[:, 1] if best == "logistic" else final.predict_proba(Xall)[:, 1])
    bundle = {"kind": best, "model": final, "median": med, "mu": mu_a, "sd": sd_a, "features": feats,
              "calibration": bins, "base_rate": float(y.mean()), "trained": summary["trained"],
              # the refit model's own fifths: a live score is placed by rank, then shown with what that fifth did
              # in the held-out test (the refit model's raw numbers aren't on the test model's scale)
              "rank_edges": [float(v) for v in np.quantile(p_all, [0.2, 0.4, 0.6, 0.8])]}
    return summary, bundle


def save_bundle(path: Path, bundle: dict | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if bundle is None:
        path.unlink(missing_ok=True)
        return
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def load_bundle(path: Path) -> dict | None:
    """Only ever reads the file this pipeline wrote itself into its own cache."""
    if not path.exists():
        return None
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)  # noqa: S301 – our own artifact
    except Exception as e:  # noqa: BLE001 – a stale/corrupt model just means base rates for now
        log.warning("model bundle unreadable (%s); using base rates", e)
        return None


def predict(bundle: dict, x: dict) -> float:
    feats = bundle["features"]
    row = np.asarray([[np.nan if x.get(k) is None else float(x[k]) for k in feats]], dtype=float)
    row = np.where(np.isnan(row), bundle["median"], row)
    if bundle["kind"] == "logistic":
        row = (row - bundle["mu"]) / bundle["sd"]
    return float(bundle["model"].predict_proba(row)[0, 1])


def calibration_band(bundle_or_summary: dict, p: float) -> dict | None:
    """'When the model said about this much before, here's what actually happened' (held-out data)."""
    bins = bundle_or_summary.get("calibration") or []
    if not bins:
        return None
    edges = bundle_or_summary.get("rank_edges")
    if edges and len(bins) == len(edges) + 1:
        q = int(np.searchsorted(np.asarray(edges), p, side="right"))
        best = bins[q]
        return {"obs": best["obs"], "lo": best["lo"], "hi": best["hi"], "n": best["n"], "fifth": q + 1}
    best = min(bins, key=lambda b: abs(b["pred"] - p))
    return {"obs": best["obs"], "lo": best["lo"], "hi": best["hi"], "n": best["n"]}


# ---------------------------------------------------------------- base rates

def base_rates(insider_buys: list[dict], insider_sells: list[dict], congress: list[dict]) -> dict:
    """Historical beat-SPY frequencies for the signal families the app shows (only outcomes already known)."""
    def take(samples, cond=lambda s: True):
        sel = [s for s in samples if s.get("y") is not None and cond(s)]
        return rate_summary([bool(s["y"]) for s in sel], [s["excess"] for s in sel if s.get("excess") is not None],
                            [s["filed"] for s in sel])

    x = lambda s, k: s["x"].get(k)  # noqa: E731
    out = {
        "insider_buy": take(insider_buys),
        "insider_buy_senior": take(insider_buys, lambda s: (x(s, "max_rank") or 0) >= 4),
        "insider_buy_cluster": take(insider_buys, lambda s: (x(s, "cluster30") or 0) >= 3),
        "insider_buy_weakness": take(insider_buys, lambda s: (x(s, "r3m") or 0) <= -0.20),
        # exactly the 'insiders buying into weakness' screen: 2+ insiders within 30 days, down 15%+ at the filing
        "insider_buy_cluster_weakness": take(insider_buys, lambda s: (x(s, "cluster30") or 0) >= 2
                                             and x(s, "r3m") is not None and x(s, "r3m") <= -0.15),
        "insider_buy_small": take(insider_buys, lambda s: (x(s, "log_adv") or 9) < 6.3),   # < ~$2M/day traded
        "insider_buy_large": take(insider_buys, lambda s: (x(s, "log_adv") or 0) >= 8),    # > ~$100M/day traded
        "insider_sell": take(insider_sells),
        "congress_buy": take(congress),
    }
    return {k: v for k, v in out.items() if v.get("n")}


def congress_samples(hist_rows: list[dict], events: list[dict], panel: Panel) -> list[dict]:
    rows = [r for r in hist_rows if r.get("action") == "buy" and r.get("ticker")]
    rows += [{"ticker": e["ticker"], "filed": e.get("filed_at"), "actor_id": e["actor"]["id"]}
             for e in events if e.get("kind") == "congress" and e.get("action") == "buy" and e.get("ticker")]
    seen, out = set(), []
    for r in rows:
        f = parse_date(r.get("filed") or r.get("filed_at"))
        t = (r.get("ticker") or "").upper()
        k = (r.get("actor_id"), t, f)
        if not f or k in seen or t not in panel.col:
            continue
        seen.add(k)
        j = panel.ix_after(f)
        fx = forward_excess(panel, t, j) if j is not None else None
        out.append({"ticker": t, "filed": f.isoformat(), "x": {}, "y": (fx[1] > 0) if fx else None,
                    "excess": fx[1] if fx else None})
    return out


def base_rate_key(e: dict) -> str | None:
    k, a = e.get("kind"), e.get("action")
    if k == "insider" and a == "buy":
        return "insider_buy"
    if k == "insider" and a == "sell":
        return "insider_sell"
    if k == "congress" and a == "buy":
        return "congress_buy"
    return None


# ---------------------------------------------------------------- live view for one event

def live_insider_groups(events: list[dict], routine=None) -> dict[str, dict]:
    """Insider features for live purchases, built exactly like the training rows: one sample per (ticker, filing
    date) summing every purchase filed that day, and the 30-day cluster counted from filings up to that date only.
    `routine(event) -> True/False/None`. Returns {event id: features}."""
    groups: dict[tuple, list] = defaultdict(list)
    by_ticker: dict[str, list] = defaultdict(list)
    for e in events:
        if e.get("kind") != "insider" or e.get("action") != "buy" or not e.get("ticker") or (e.get("value") or 0) < 1000:
            continue
        f = parse_date(e.get("filed_at"))
        if not f:
            continue
        groups[(e["ticker"], f)].append(e)
        by_ticker[e["ticker"]].append((f, (e.get("actor") or {}).get("id") or e["id"]))
    for t in by_ticker:
        by_ticker[t].sort()
    out = {}
    for (t, f), g in groups.items():
        lst = by_ticker[t]
        a = bisect.bisect_left(lst, (f - dt.timedelta(days=30), ""))
        b = bisect.bisect_right(lst, (f, "\uffff"))
        rs = []
        for e in g:
            r = routine(e) if routine else None
            rs.append(0.5 if r is None else (1.0 if r else 0.0))
        d_any = [e.get("details") or {} for e in g]
        feat = {"max_rank": max(((e.get("actor") or {}).get("rank") or 1) for e in g),
                "n_owners": len({(e.get("actor") or {}).get("id") or e["id"] for e in g}),
                "log_value": math.log10(sum(e.get("value") or 0 for e in g) + 1),
                "cluster30": len({o for _, o in lst[a:b]}),
                "routine_share": float(np.mean(rs)) if rs else 0.5,
                # training uses the Form 4 checkbox (DERA's AFF10B5ONE), so a footnote mention doesn't count here
                "plan": 1.0 if any(d.get("plan_10b5_1") and d.get("plan_flag_source") == "checkbox" for d in d_any) else 0.0}
        for e in g:
            out[e["id"]] = feat
    return out


def event_view(e: dict, panel: Panel | None, bundle: dict | None, model_summary: dict | None, rates: dict,
               cluster_buyers: int = 0, routine: bool | None = None, ledger_rates: dict | None = None,
               group: dict | None = None) -> dict | None:
    """The 'how predictable is this?' block shown next to a trade. `group` = live_insider_groups()[e["id"]], the
    same per-day features the model was trained on (falls back to this filing alone)."""
    if not e.get("ticker") or e.get("action") not in ("buy", "sell", "new", "add", "stake", "trim", "exit", "sale_notice"):
        return None
    key = base_rate_key(e)
    br = rates.get(key) if key else None
    if not br and ledger_rates:
        br = ledger_rates.get(f"{e.get('kind')}:{e.get('action')}")
    view = {"question": "Chance the stock beats the S&P 500 over the next ~3 months",
            "base_rate": br, "base_rate_key": key}
    if (e.get("kind") == "insider" and e.get("action") == "buy" and bundle is not None and panel is not None
            and model_summary and model_summary.get("has_skill")):
        f = parse_date(e.get("filed_at")) or now_utc().date()
        i = panel.ix_on_or_before(f)
        pf = price_features(panel, e["ticker"], i) if i is not None else None
        if pf:
            d = e.get("details") or {}
            base = group or {"max_rank": (e.get("actor") or {}).get("rank") or 1, "n_owners": 1,
                             "log_value": math.log10((e.get("value") or 0) + 1), "cluster30": max(1, cluster_buyers),
                             "routine_share": 0.5 if routine is None else (1.0 if routine else 0.0),
                             "plan": 1.0 if (d.get("plan_10b5_1") and d.get("plan_flag_source") == "checkbox") else 0.0}
            x = {**base, **{k: pf.get(k) for k in ("r1m", "r3m", "mom6_1", "vol3m", "dist_hi", "dist_lo", "log_adv",
                                                   "spy_r3m", "spy_trend")}}
            p = predict(bundle, x)
            view.update(p=p, source="model", band=calibration_band(bundle, p),
                        predictability=model_summary.get("predictability"), auc=model_summary.get("auc"))
    if "p" not in view and br and br.get("n", 0) >= 30:
        view.update(p=br["p"], source="base_rate", band={"obs": br["p"], "lo": br["lo"], "hi": br["hi"], "n": br["n"]},
                    predictability="Base rate only")
    if "p" not in view:
        view.update(p=None, source="none", predictability="Not enough history yet")
    return view


# ---------------------------------------------------------------- projection ranges

def projection(price: float | None, vol_annual: float | None, days: int = HOLD) -> dict | None:
    """Typical price range after `days` trading days if the stock keeps swinging like it has (lognormal, no drift).
    Not a forecast of direction."""
    if (not price or not vol_annual or not math.isfinite(price) or not math.isfinite(vol_annual) or price <= 0
            or vol_annual <= 0):
        return None
    s = vol_annual * math.sqrt(days / 252)
    f = lambda z: round(price * math.exp(z * s), 4)  # noqa: E731
    return {"days": days, "p05": f(-1.645), "p16": f(-1.0), "p50": round(price, 4), "p84": f(1.0), "p95": f(1.645),
            "sigma": s}


# ---------------------------------------------------------------- price-based idea screen + its backtest

def liquid_universe(panel: Panel, i: int, allowed: set[str] | None = None, min_price: float = 5.0,
                    min_adv: float = 10e6) -> list[str]:
    import warnings
    lo = max(0, i - 19)
    c = panel.close[lo:i + 1]
    v = panel.volume[lo:i + 1]
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)     # tickers with no trades in the window
        adv = np.nanmean(c * v, axis=0)
    last = panel.close[i]
    ok = (~np.isnan(last)) & (last >= min_price) & (adv >= min_adv)
    names = [t for t, j in panel.col.items() if ok[j]]
    if allowed is not None:
        names = [t for t in names if t in allowed]
    return names


def momentum_screen(panel: Panel, i: int, universe: list[str], top_frac: float = 0.10) -> list[dict]:
    """Strong 6-month momentum (skipping the latest month) that is still within 5% of its 52-week high and not
    wildly volatile: the George–Hwang 52-week-high and Jegadeesh–Titman momentum effects together."""
    rows = []
    for t in universe:
        pf = price_features(panel, t, i)
        if not pf or pf["mom6_1"] is None or pf["vol3m"] is None:
            continue
        rows.append((t, pf))
    if len(rows) < 50:
        return []
    moms = np.asarray([pf["mom6_1"] for _, pf in rows])
    cut = float(np.quantile(moms, 1 - top_frac))
    picks = [{"ticker": t, **{k: pf[k] for k in ("mom6_1", "r1m", "dist_hi", "vol3m", "price", "adv")}}
             for t, pf in rows if pf["mom6_1"] >= cut and pf["dist_hi"] >= -0.05 and pf["vol3m"] <= 0.70]
    picks.sort(key=lambda d: -d["mom6_1"])
    return picks


def hac_mean_ci(x: np.ndarray, lags: int) -> tuple[float, float] | None:
    """95% interval for a mean of overlapping-window returns: Newey-West standard error, and a t critical value on
    fewer degrees of freedom than non-overlapping windows (deliberately conservative: with ~15 monthly cohorts a
    null screen's interval wrongly excluded zero about 3% of the time in simulation, vs ~7% with one more df)."""
    from scipy.stats import t as student_t
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 3:
        return None
    e = x - x.mean()
    v = float(e @ e) / n
    for L in range(1, min(lags, n - 1) + 1):
        v += 2 * (1 - L / (lags + 1)) * float(e[L:] @ e[:-L]) / n
    se = math.sqrt(max(v, float(e @ e) / n / (lags + 1)) / (n - 1))
    tc = float(student_t.ppf(0.975, max(1, (n - 1) // (lags + 2))))
    return float(x.mean() - tc * se), float(x.mean() + tc * se)


def backtest_momentum_screen(panel: Panel, allowed: set[str] | None = None, step: int = 21) -> dict:
    """Run the screen at every month-end with a known 3-month outcome. Cohort-level stats (cohorts overlap in
    time, so the honest sample size is the number of months, not the number of stocks).

    Stocks that later stopped trading are kept in the historical universe (the current SEC ticker list alone
    would only contain survivors), and graded to their last trade."""
    cohorts = []
    start = 130
    allowed_bt = (set(allowed) | delisted_names(panel)) if allowed is not None else None
    for i in range(start, panel.T - HOLD - 2, step):
        uni = liquid_universe(panel, i, allowed_bt)
        picks = momentum_screen(panel, i, uni)
        if not picks:
            continue
        outs, all_beat = [], []
        for d in picks:
            fx = forward_excess(panel, d["ticker"], i + 1)
            if fx:
                outs.append(fx[1])
        for t in uni[:4000]:
            fx = forward_excess(panel, t, i + 1)
            if fx:
                all_beat.append(fx[1] > 0)
        if len(outs) >= 5:
            cohorts.append({"date": panel.date(i), "n": len(outs), "beat": float(np.mean([o > 0 for o in outs])),
                            "mean_excess": float(np.mean(outs)), "all_beat": float(np.mean(all_beat)) if all_beat else None})
    if not cohorts:
        return {"status": "insufficient", "cohorts": 0}
    beats = np.asarray([c["beat"] for c in cohorts])
    ex = np.asarray([c["mean_excess"] for c in cohorts])
    lags = max(0, math.ceil(HOLD / step) - 1)            # 3-month outcomes measured monthly overlap 2 months
    ci = hac_mean_ci(ex, lags)
    beat_ci = hac_mean_ci(beats, lags)
    base = [c["all_beat"] for c in cohorts if c["all_beat"] is not None]
    return {"status": "ok", "cohorts": len(cohorts), "stocks": int(sum(c["n"] for c in cohorts)),
            "beat_rate": float(beats.mean()), "mean_excess": float(ex.mean()),
            "mean_excess_ci": list(ci) if ci else None,
            "beat_rate_ci": [max(0.0, beat_ci[0]), min(1.0, beat_ci[1])] if beat_ci else None,
            "independent_windows": len(ex) // (lags + 1),
            "all_stocks_beat_rate": float(np.mean(base)) if base else None,
            "months_positive": int((ex > 0).sum()), "period": [cohorts[0]["date"], cohorts[-1]["date"]],
            "by_month": cohorts[-12:]}


def quarter_label(d: dt.date) -> str:
    return f"{d.year}Q{(d.month - 1) // 3 + 1}"


def build_all(heavy_dir: Path, panel: Panel, months: dict, congress_hist: list[dict], events: list[dict],
              allowed: set[str] | None = None, gate: dict | None = None) -> tuple[dict, dict | None]:
    """Daily: rebuild data sets, base rates, the tested model and the screen backtest.

    The pass/fail decision on the model is taken once per calendar quarter and then held (`gate`, kept in the
    app's state): re-deciding every day on slightly different data would give chance many tries to pass."""
    rows = load_dera_rows(heavy_dir)
    buys = insider_dataset(rows, panel, months, "P")
    sells = insider_dataset(rows, panel, months, "S")
    cong = congress_samples(congress_hist, events, panel)
    today = now_et().date()
    q_now = quarter_label(today)
    frozen = gate if (gate and gate.get("quarter") == q_now and gate.get("spec") == GATE_SPEC) else None
    model_summary, bundle = train_model(buys, decision=frozen["has_skill"] if frozen else None)
    if model_summary.get("status") == "ok":
        new_gate = frozen or {"quarter": q_now, "spec": GATE_SPEC, "has_skill": bool(model_summary["passed_today"]),
                              "decided": today.isoformat(), "auc": model_summary.get("auc"),
                              "auc_ci": model_summary.get("auc_ci"), "n_test": model_summary.get("n_test")}
    else:
        new_gate = frozen
    model_summary["gate"] = new_gate
    out = {"built": now_utc().isoformat(), "rates": base_rates(buys, sells, cong), "gate": new_gate,
           "insider_model": model_summary, "screen": backtest_momentum_screen(panel, allowed),
           "samples": {"insider_buys": len(buys), "insider_buys_known": sum(1 for s in buys if s["y"] is not None),
                       "insider_sells": len(sells), "congress_buys": len(cong)},
           "price_history": [panel.date(0), panel.date(panel.T - 1)] if panel.T else None}
    log.info("quant: %d insider-buy samples (%d with outcomes), model=%s", len(buys), out["samples"]["insider_buys_known"],
             model_summary.get("chosen") or model_summary.get("status"))
    return out, bundle


def first_trading_day_after(panel: Panel, d: str | None) -> str | None:
    f = parse_date(d)
    if not f:
        return None
    j = panel.ix_after(f)
    return panel.date(j) if j is not None else None


__all__ = ["Panel", "wilson", "rate_summary", "projection", "event_view", "build_all", "load_bundle", "save_bundle",
           "momentum_screen", "liquid_universe", "price_features", "forward_excess", "insider_dataset", "train_model",
           "base_rates", "date_str", "first_trading_day_after"]
