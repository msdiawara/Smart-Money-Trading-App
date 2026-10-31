"""The forecast ledger: the price forecasts' report card.

Every forecast the app shows is written down when it is made (never edited, never backdated) and graded once
its horizon has passed, against the stock's actual closing price:

* coverage: did the price land inside the 50%, 80% and 90% ranges? A well-calibrated forecaster's 80% ranges
  contain the outcome ~80% of the time. This is the target the app holds itself to (and learns toward);
* direction: was "more likely up than down" right? Shown next to the "always say up" baseline, because over
  most windows more stocks rise than fall;
* skill: the quantile score (a proper scoring rule close to CRPS, Gneiting & Raftery 2007) versus a naive
  forecast (no drift, last year's volatility). Above 0% = better than naive;
* PIT histogram: where in its own forecast distribution each outcome fell (flat = calibrated).

Raw model outputs (before the learned calibration) are kept too, so the app can show what learning changed.
Sampling keeps it small: 1-day, 1-week and 1-month forecasts daily; 3-month weekly; 1-year monthly, for up to a
few hundred stocks. Every graded forecast counts in the monthly aggregates; the per-forecast records that the
learning uses keep a fixed random sample per day (outcomes on one day move together, so more of the same day
adds little) for about 15 months.

Forecasts given as price levels (Claude's, the blend) are re-based by any split since they were written: the
price history is split-adjusted, so a 2-for-1 split must halve the forecast too, or it would grade as a crash.
"""
from __future__ import annotations

import bisect
import datetime as dt
import hashlib
import math
from pathlib import Path

import numpy as np

from .forecast import HORIZONS, TAUS, last_complete_session, split_like
from .quant import clustered_interval
from .util import now_et, read_json, write_json

PATH = "forecast_ledger.json.gz"
HMAP = dict(HORIZONS)
SRCS = ("quant", "claude", "blend", "quant_bt")      # quant_bt: walk-forward backtest of the model on past data
KEEP_GRADED = 15000                 # hard cap per source and horizon
PER_DAY = {"quant": 30, "quant_bt": 40}  # sampled records kept per forecast day (others: all)
KEEP_DAYS = 460


def load(heavy_dir: Path) -> dict:
    L = read_json(heavy_dir / PATH)
    if not isinstance(L, dict) or L.get("v") != 1:
        L = {"v": 1}
    for k, d in (("pending", {}), ("graded", {}), ("agg", {}), ("last", {})):
        L.setdefault(k, d)
    return L


def save(heavy_dir: Path, L: dict) -> None:
    compact(L)
    write_json(heavy_dir / PATH, L)


def _hash(t: str, made: str) -> int:
    return int(hashlib.sha1(f"{t}:{made}".encode()).hexdigest()[:8], 16)


def compact(L: dict, today: dt.date | None = None) -> None:
    """Keep the graded records the learning needs: ~15 months, a fixed random sample per day (the same stocks
    at every horizon), and a hard cap. The monthly aggregates are not touched: they count everything."""
    today = today or now_et().date()
    cut = (today - dt.timedelta(days=KEEP_DAYS)).isoformat()
    for src, by_h in (L.get("graded") or {}).items():
        k = PER_DAY.get(src)
        for h, rows in list(by_h.items()):
            rows = [r for r in rows if (r.get("made") or "") >= cut]
            if k:
                by_day: dict[str, list] = {}
                for r in rows:
                    by_day.setdefault(r["made"], []).append(r)
                rows = []
                for d in sorted(by_day):
                    day = by_day[d]
                    if len(day) > k:
                        day = sorted(day, key=lambda r: _hash(r["t"], d))[:k]
                    rows.extend(day)
            else:
                rows.sort(key=lambda r: r["made"])
            by_h[h] = rows[-KEEP_GRADED:]


def _week(d: dt.date) -> str:
    y, w, _ = d.isocalendar()
    return f"{y}-W{w:02d}"


def sampled(h: int, made: dt.date, last: str | None) -> bool:
    """Whether a forecast at horizon h made on `made` should be recorded given the last recorded one's date."""
    if last is None:
        return True
    ld = dt.date.fromisoformat(last)
    if made <= ld:
        return False
    if h >= 252:
        return (made.year, made.month) != (ld.year, ld.month)
    if h >= 63:
        return _week(made) != _week(ld)
    return True


def record(L: dict, src: str, t: str, made: str, h: int, lq: list[float] | None, *, lq_raw: list[float] | None = None,
           pup: float | None = None, naive: list[float] | None = None, feats: dict | None = None,
           bubble: str | None = None, ref_date: str | None = None, pq: list[float] | None = None,
           extra: dict | None = None) -> bool:
    """First write wins. `made` is the anchor session (quant) or the forecast date (Claude); `ref_date` the
    session whose close the outcome is measured from (defaults to `made`). A forecast given as price levels
    (`pq`, 7 prices) is turned into returns from the reference close when it is graded."""
    fid = f"{src}:{t}:{h}:{made}"
    vals = pq if pq is not None else lq
    if fid in L["pending"] or not vals or len(vals) != len(TAUS):
        return False
    if any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in vals):
        return False
    if any(b < a for a, b in zip(vals, vals[1:])) or (pq is not None and min(pq) <= 0):
        return False
    key = f"{src}:{t}:{h}"
    if not sampled(h, dt.date.fromisoformat(made[:10]), L["last"].get(key)):
        return False
    e = {"t": t, "src": src, "h": h, "made": made[:10]}
    if pq is not None:
        e["pq"] = [round(float(x), 4) for x in pq]
    else:
        e["lq"] = [round(float(x), 5) for x in lq]
    if lq_raw:
        e["lqr"] = [round(float(x), 5) for x in lq_raw]
    if pup is not None and math.isfinite(pup):
        e["pup"] = round(float(pup), 4)
    if naive:
        e["nv"] = [round(float(x), 5) for x in naive]
    if feats:
        e["f"] = {k: round(float(v), 4) for k, v in feats.items() if v is not None and math.isfinite(v)}
    if bubble:
        e["b"] = bubble
    if ref_date and ref_date != made[:10]:
        e["ref"] = ref_date
    if extra:
        e.update(extra)
    L["pending"][fid] = e
    L["last"][key] = made[:10]
    return True


# ---------------------------------------------------------------- scoring

def pit(lq: list[float], r: float) -> float:
    """Where the outcome fell in the forecast's own distribution (0-1), piecewise linear between quantiles."""
    q = np.asarray(lq, dtype=float)
    p = TAUS
    if r <= q[0]:
        w = max(q[1] - q[0], 1e-6)
        return float(max(0.001, p[0] - (q[0] - r) / w * (p[1] - p[0])))
    if r >= q[-1]:
        w = max(q[-1] - q[-2], 1e-6)
        return float(min(0.999, p[-1] + (r - q[-1]) / w * (p[-1] - p[-2])))
    i = int(np.searchsorted(q, r, side="right")) - 1
    i = min(max(i, 0), len(q) - 2)
    span = q[i + 1] - q[i]
    return float(p[i] + (p[i + 1] - p[i]) * ((r - q[i]) / span if span > 0 else 0.5))


def qscore(lq: list[float], r: float) -> float:
    """2 × mean pinball loss over the 7 quantiles (≈ CRPS for a dense set of quantiles). Lower is better."""
    q = np.asarray(lq, dtype=float)
    u = r - q
    return float(2 * np.mean(np.maximum(TAUS * u, (TAUS - 1) * u)))


def s_needed(lq: list[float], r: float) -> float | None:
    """How much the 10–90% range (around the median) would have to be stretched to just contain r."""
    med = lq[3]
    if r >= med:
        w = lq[5] - med
    else:
        w = med - lq[1]
    if w <= 0:
        return None
    return float(abs(r - med) / w)


def _series_index(dates: list[int], d: dt.date, back: int = 3) -> int | None:
    """Row of the close on `d`, or the nearest earlier trading row within `back` calendar days."""
    o = d.toordinal()
    i = bisect.bisect_right(dates, o) - 1
    if i < 0 or o - dates[i] > back:
        return None
    return i


def _split_factor(e: dict, dates: list[int], c) -> float:
    """For a forecast stored as price levels: how much the (split-adjusted) history has been re-based since it was
    written. 1.0 normally; 0.5 after a 2-for-1 split."""
    last, ld = e.get("last"), e.get("ld")
    if not last or not ld:
        return 1.0
    try:
        i = _series_index(dates, dt.date.fromisoformat(ld))
    except ValueError:
        return 1.0
    if i is None:
        return 1.0
    f = float(c[i]) / float(last)
    if not math.isfinite(f) or not (0.005 < f < 200):
        return 1.0
    return f if abs(math.log(f)) > 0.02 else 1.0      # ignore tiny data revisions


def grade(L: dict, series, market_last: int | None = None, today: dt.date | None = None,
          last_complete: dt.date | None = None) -> int:
    """Grade every pending forecast whose horizon has passed. `series(t)` → (dates[ordinal], closes) or None.
    `market_last`: the latest date the price history covers (a stock missing from there is delisted or halted).
    Only completed sessions count (a partial intraday bar is never an outcome), and a window that contains what
    looks like an un-adjusted split waits up to a week for the history to be re-based."""
    now = now_et()
    today = today or now.date()
    lc = (last_complete or (last_complete_session(now) if today == now.date() else today)).toordinal()
    cache: dict[str, tuple | None] = {}
    done = 0
    for fid in list(L["pending"]):
        e = L["pending"][fid]
        h = int(e["h"])
        ref = dt.date.fromisoformat(e.get("ref") or e["made"])
        if (today - ref).days < h * 1.38:
            continue
        if e["t"] not in cache:
            cache[e["t"]] = series(e["t"])
        s = cache[e["t"]]
        late = (today - ref).days > h * 1.45 + 45
        if not s:
            if late:
                _void(L, fid, "no prices")
                done += 1
            continue
        dates, c = s
        n_ok = bisect.bisect_right(dates, lc)            # rows up to the last completed session
        dates, c = dates[:n_ok], c[:n_ok]
        j = _series_index(dates, ref)
        if j is None:
            if late or (dates and dates[-1] > ref.toordinal() + 10):
                _void(L, fid, "no price on the reference day")
                done += 1
            continue
        k = j + h
        delisted = False
        if k >= len(dates):
            stopped = market_last is not None and dates[-1] < market_last - 14
            if stopped and (today - ref).days > h * 1.45:
                k = len(dates) - 1                    # stopped trading: graded to its last close
                delisted = True
            else:
                if late and not stopped:
                    _void(L, fid, "price history ended")
                    done += 1
                continue
        p0, p1 = float(c[j]), float(c[k])
        if not (p0 > 0 and p1 > 0 and math.isfinite(p0) and math.isfinite(p1)):
            _void(L, fid, "bad price")
            done += 1
            continue
        seg = np.asarray(c[j:k + 1], dtype=float)
        if len(seg) > 1 and (today - dt.date.fromordinal(int(dates[k]))).days <= 7 and \
                any(split_like(float(x)) for x in np.diff(np.log(seg))):
            continue                                  # looks like a split not yet applied: wait for the re-base
        r = math.log(p1 / p0)
        f = _split_factor(e, dates, c)
        base = p0
        if e.get("pq"):
            e = {**e, "lq": [math.log(x * f / p0) for x in e["pq"]]}
            if e.get("last"):
                base = float(e["last"]) * f           # "chance it ends higher" was relative to the close it saw
        if e.get("qpq"):
            e = {**e, "lq_m": [math.log(x * f / p0) for x in e["qpq"]]}
        path = np.log(seg / p0)
        peak = np.maximum.accumulate(path)
        dd = float(1 - math.exp(-float(np.max(peak - path))))
        if f != 1.0:
            e = {**e, "split_f": round(f, 6)}
        _finish(L, fid, e, r, dd, delisted, today, p0, p1, dt.date.fromordinal(int(dates[k])), base=base)
        done += 1
    if done:
        compact(L, today)
    return done


def _void(L: dict, fid: str, why: str) -> None:
    e = L["pending"].pop(fid)
    L.setdefault("void", {})
    L["void"][why] = L["void"].get(why, 0) + 1
    del e


def _finish(L: dict, fid: str, e: dict, r: float, dd: float, delisted: bool, today: dt.date, p0: float, p1: float,
            end: dt.date, base: float | None = None) -> None:
    lq = e["lq"]
    g = {"t": e["t"], "made": e["made"], "r": round(r, 5), "dd": round(dd, 4)}
    if e["src"] in ("claude", "blend"):            # kept whole for Claude's post-mortems and the checks
        g.update(lq=[round(x, 5) for x in lq], p0=round(p0, 4), p1=round(p1, 4), end=end.isoformat(),
                 ref=e.get("ref") or e["made"], gd=today.isoformat())
        if e.get("ld"):
            g["ld"] = e["ld"]
        if e.get("split_f"):
            g["split_f"] = e["split_f"]
    p = pit(lq, r)
    g.update(pit=round(p, 4), in50=int(lq[2] <= r <= lq[4]), in80=int(lq[1] <= r <= lq[5]),
             in90=int(lq[0] <= r <= lq[6]), qs=round(qscore(lq, r), 6), err=round(r - lq[3], 5), up=int(r > 0))
    if e.get("pup") is not None and e["pup"] != 0.5:
        g["pup"] = e["pup"]
        b = base if base and base > 0 else p0
        g["hit"] = int((e["pup"] > 0.5) == (p1 > b))
    if e.get("nv"):
        g["qsn"] = round(qscore(e["nv"], r), 6)
    if e.get("lq_m"):
        g["qs_m"] = round(qscore(e["lq_m"], r), 6)     # the model's forecast from the same close, same outcome
    raw = e.get("lqr") or lq
    g["er"] = round(r - raw[3], 5)
    if e.get("tl"):
        g["er0"] = round(r - (raw[3] - float(e["tl"])), 5)
    g["w10"], g["w90"] = round(raw[3] - raw[1], 5), round(raw[5] - raw[3], 5)
    g["in80r"] = int(raw[1] <= r <= raw[5])
    if e.get("f"):
        g["f"] = e["f"]
    if e.get("b"):
        g["b"] = e["b"]
    if e.get("pd") is not None:
        g["pd"] = e["pd"]
    if delisted:
        g["delisted"] = 1
    src, h = e["src"], str(e["h"])
    L["graded"].setdefault(src, {}).setdefault(h, []).append(g)
    m = e["made"][:7]
    a = L["agg"].setdefault(src, {}).setdefault(h, {}).setdefault(m, {
        "n": 0, "in50": 0, "in80": 0, "in90": 0, "in80r": 0, "hits": 0, "nhit": 0, "up": 0, "qs": 0.0, "qsn": 0.0,
        "nqsn": 0, "pit": [0] * 10, "err": 0.0})
    a["n"] += 1
    a["in50"] += g["in50"]
    a["in80"] += g["in80"]
    a["in90"] += g["in90"]
    a["in80r"] += g["in80r"]
    a["up"] += g["up"]
    a["err"] = round(a["err"] + g["err"], 5)
    a["pit"][min(9, int(p * 10))] += 1
    if "hit" in g:
        a["hits"] += g["hit"]
        a["nhit"] += 1
    if "qsn" in g:
        a["qs"] = round(a["qs"] + g["qs"], 6)
        a["qsn"] = round(a["qsn"] + g["qsn"], 6)
        a["nqsn"] += 1
    del L["pending"][fid]


def add_backtest(L: dict, t: str, made: str, h: int, lq: list[float], lq_raw: list[float], pup: float,
                 naive: list[float], r: float, dd: float, today: dt.date, p0: float, p1: float, end: dt.date) -> bool:
    """A backtest forecast is graded at once (its outcome is already history); same scoring as live ones."""
    fid = f"quant_bt:{t}:{h}:{made}"
    if fid in L["pending"] or fid in L.setdefault("bt_seen", {}):
        return False
    e = {"t": t, "src": "quant_bt", "h": h, "made": made, "lq": lq, "lqr": lq_raw, "pup": pup, "nv": naive}
    L["pending"][fid] = e
    _finish(L, fid, e, r, dd, False, today, p0, p1, end)
    L["bt_seen"][fid] = 1
    return True


# ---------------------------------------------------------------- reporting

def _block(made: str, h: int) -> int:
    days = max(7, int(math.ceil(h * 7 / 5)))
    return dt.date.fromisoformat(made).toordinal() // days


def _rate(rows: list[dict], key: str, h: int) -> dict | None:
    rr = [r for r in rows if key in r]
    if not rr:
        return None
    y = [r[key] for r in rr]
    lo, hi, g = clustered_interval(y, [_block(r["made"], h) for r in rr])
    return {"p": float(np.mean(y)), "lo": lo, "hi": hi, "n": len(rr), "periods": g}


def summarize(rows: list[dict], h: int) -> dict:
    if not rows:
        return {"n": 0}
    out = {"n": len(rows)}
    for k in ("in50", "in80", "in90", "in80r", "hit", "up"):
        v = _rate(rows, k, h)
        if v:
            out[k] = v
    both = [r for r in rows if "qsn" in r]
    if both:
        qs, qsn = sum(r["qs"] for r in both), sum(r["qsn"] for r in both)
        out["skill"] = (1 - qs / qsn) if qsn > 0 else None
    hist = [0] * 10
    for r in rows:
        hist[min(9, int(r["pit"] * 10))] += 1
    out["pit"] = hist
    out["bias"] = float(np.mean([r["err"] for r in rows]))
    out["mae"] = float(np.median([abs(r["err"]) for r in rows]))
    return out


def metrics(L: dict, today: dt.date | None = None, recent_days: int = 120) -> dict:
    """Per source and horizon: everything graded (from the monthly aggregates), the recent window (from the
    graded records, with honest clustered intervals), and the month-by-month trend."""
    today = today or now_et().date()
    cut = (today - dt.timedelta(days=recent_days)).isoformat()
    out: dict = {"pending": len(L["pending"]), "void": L.get("void") or {}, "by": {}}
    pend: dict[str, int] = {}
    for e in L["pending"].values():
        pend[e["src"]] = pend.get(e["src"], 0) + 1
    out["pending_by_src"] = pend
    names = {str(h): name for name, h in HORIZONS}
    for src in SRCS:
        for h, months in (L["agg"].get(src) or {}).items():
            tot = {k: sum(m[k] for m in months.values()) for k in ("n", "in50", "in80", "in90", "in80r", "hits", "nhit",
                                                                     "up", "nqsn")}
            qs = sum(m["qs"] for m in months.values())
            qsn = sum(m["qsn"] for m in months.values())
            trend = []
            for mo in sorted(months):
                m = months[mo]
                if not m["n"]:
                    continue
                trend.append({"month": mo, "n": m["n"], "cov80": m["in80"] / m["n"], "cov50": m["in50"] / m["n"],
                              "cov80_raw": m["in80r"] / m["n"], "hit": (m["hits"] / m["nhit"]) if m["nhit"] else None,
                              "up": m["up"] / m["n"], "skill": (1 - m["qs"] / m["qsn"]) if m["qsn"] > 0 else None})
            rows = L["graded"].get(src, {}).get(h, [])
            recent = [r for r in rows if r["made"] >= cut]
            n = tot["n"]
            ci = _rate(rows, "in80", int(h))      # clustered interval from the kept records (overlapping windows)
            out["by"].setdefault(src, {})[names.get(h, h)] = {
                "h": int(h),
                "all": {"n": n, "cov50": tot["in50"] / n if n else None, "cov80": tot["in80"] / n if n else None,
                        "cov90": tot["in90"] / n if n else None, "cov80_raw": tot["in80r"] / n if n else None,
                        "hit": tot["hits"] / tot["nhit"] if tot["nhit"] else None, "up": tot["up"] / n if n else None,
                        "skill": (1 - qs / qsn) if qsn > 0 else None,
                        # the kept records' own rate and interval (they can differ from the all-time totals),
                        # and how many independent periods it rests on (under 3, it isn't clustered: don't lean on it)
                        "cov80_ci": ({"p": ci["p"], "lo": ci["lo"], "hi": ci["hi"], "n": ci["n"], "periods": ci["periods"]}
                                     if ci and ci["n"] >= 20 else None)},
                "recent": summarize(recent, int(h)), "trend": trend,
            }
    out["bubble_check"] = bubble_check(L)
    return out


def bubble_check(L: dict) -> dict:
    """Did the bubble-risk rating mean anything beyond volatility? For 3-month forecasts, by the rating the stock
    had when the forecast was made: the share with a 20%+ drop from a peak, next to the share the model expected
    from each stock's own volatility. A rating that only restates volatility shows no excess."""
    rows = L["graded"].get("quant", {}).get("63", [])
    out = {}
    for lvl in ("Low", "Moderate", "Elevated", "High"):
        rr = [r for r in rows if r.get("b") == lvl]
        if rr:
            y = [int(r["dd"] >= 0.2) for r in rr]
            lo, hi, g = clustered_interval(y, [_block(r["made"], 63) for r in rr])
            exp = [r["pd"] for r in rr if r.get("pd") is not None]
            out[lvl] = {"n": len(rr), "p_dd20": float(np.mean(y)), "lo": lo, "hi": hi, "periods": g,
                        "expected": float(np.mean(exp)) if len(exp) >= len(rr) * 0.8 else None}
    return out


def recent_misses(L: dict, src: str = "claude", days: int = 45, limit: int = 12, today: dt.date | None = None) -> list[dict]:
    """The biggest recent misses (outcome outside the 90% range), worst first: inputs for Claude's post-mortems."""
    today = today or now_et().date()
    cut = (today - dt.timedelta(days=days)).isoformat()
    rows = []
    for h, lst in (L["graded"].get(src) or {}).items():
        for r in lst:
            if (r.get("gd") or "") < cut or r.get("in90"):
                continue
            rows.append({**r, "h": int(h)})
    rows.sort(key=lambda r: -abs(r["pit"] - 0.5))
    return rows[:limit]
