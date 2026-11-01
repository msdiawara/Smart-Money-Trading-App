"""Results: what each forecast said, what happened, and why the misses missed (the app's Results tab).

A forecast *wins* when the price ends inside its 80% range: that is what the tool promises, and by design about 80%
of forecasts should win. Its direction call ("more likely up" or "more likely down") is scored separately, next to
what the forecasts themselves expected (the average of their own odds) and to "always say up".

Every miss is taken apart with what happened in its window (prices only, so the same method works for live forecasts
and the backtest):
  market      the S&P 500's own surprise over the window times the stock's beta
  earnings    the stock's move beyond the market on its earnings reaction day(s)
  news        its biggest move beyond the market on any other single day, when that day was a shock (4+ normal days'
              size; for a 1-day forecast, 2.5+)
  volatility  it swung far more than the forecast allowed throughout, with no single day to blame
  trend       a steady drift past the middle of the range, with no big day (what the signal learner tries to predict)
The biggest part that pushed the price the way it missed is the cause (half the miss or more; else volatility, if the
stock swung 1.5x more than allowed or the forecast was for one day; else a part with at least 30%; else trend). Next to each cause the page says what
the learning loop does about that kind of miss, from the learners' current state, so "how it will correct" is what it
actually does, not a promise.
"""
from __future__ import annotations

import bisect
import datetime as dt
import math

import numpy as np

from .forecast import HORIZONS, trading_days_after
from .prices import BENCH
from .quant import clustered_interval
from .util import iso, now_et, now_utc

Z80 = 1.2816
CAUSES = ["market", "earnings", "news", "volatility", "trend", "unknown", "pending"]
CAUSE_NAMES = {"market": "The whole market moved", "earnings": "An earnings report", "news": "A one-day shock",
               "volatility": "Much more volatile than expected", "trend": "A steady trend",
               "unknown": "Couldn't be taken apart (no prices)", "pending": "Not taken apart yet"}
NAMES = {str(h): name for name, h in HORIZONS}
H_OF = {name: h for name, h in HORIZONS}
ROUTINE = {"2.02", "9.01", "5.07", "7.01"}
SRC_LIVE = ("quant", "claude", "blend")


# ---------------------------------------------------------------- one miss, taken apart

def published(r: dict) -> tuple[float, float, float] | None:
    """(10th, 50th, 90th percentile) log returns the forecast was published with."""
    if r.get("lq") and len(r["lq"]) == 7:
        return float(r["lq"][1]), float(r["lq"][3]), float(r["lq"][5])
    if "err" not in r or "r" not in r:
        return None
    mid = float(r["r"]) - float(r["err"])
    w10, w90 = (r.get("pw10"), r.get("pw90")) if r.get("pw10") is not None else (r.get("w10"), r.get("w90"))
    if w10 is None or w90 is None:
        return None
    return mid - float(w10), mid, mid + float(w90)


def _reaction_days(ds: list[int], earnings: list[str], when: dict, spikes: list[str] | None = None) -> list[int]:
    """Positions, among the window's daily moves, of earnings reaction days. `ds`: the window's sessions with the
    anchor first (its close is the starting price, so a release before that day's open is already in it). A release
    before the open moves that session; after the close, the next one; unknown timing: both candidates (the bigger
    move is used). An estimated earnings day (a volume spike) is the reaction session itself."""
    out = []
    a, z = ds[0], ds[-1]
    for e in earnings:
        try:
            o = dt.date.fromisoformat(str(e)[:10]).toordinal()
        except ValueError:
            continue
        if o < a or o > z:
            continue
        k = bisect.bisect_left(ds, o)               # the first session on or after the release day
        on = k < len(ds) and ds[k] == o
        tm = when.get(str(e)[:10])
        if tm == "pre":
            if o == a:
                continue
            cands = [k]
        elif tm == "post":
            cands = [k + 1] if on else [k]
        else:
            cands = [k, k + 1] if on and o != a else [k + 1] if on else [k]
        out += [c - 1 for c in cands if 1 <= c < len(ds)]
    for e in spikes or []:
        try:
            o = dt.date.fromisoformat(str(e)[:10]).toordinal()
        except ValueError:
            continue
        k = bisect.bisect_left(ds, o)
        if 1 <= k < len(ds) and ds[k] == o:
            out.append(k - 1)
    return sorted(set(out))


def diagnose(r: dict, h: int, dates: list[int], closes: np.ndarray, spy: dict[int, float],
             mkt_err: float | None = None, earnings: list[str] | None = None, when: dict | None = None,
             filings: list[dict] | None = None, spikes: list[str] | None = None) -> dict:
    """Why one graded forecast missed its 80% range (or, for a win, the same parts for the record).

    `dates`/`closes`: the stock's sessions (ordinals) and closes; `spy`: the S&P 500 ETF's close by ordinal;
    `mkt_err`: the S&P 500's own forecast error for the same date and horizon (its surprise); `earnings`/`when`:
    known earnings release dates and whether before the open; `filings`: 8-Ks (date, items); `spikes`: estimated
    earnings days (volume spikes) for stocks without a filing record."""
    q = published(r)
    if q is None:
        return {"c": "unknown", "why": "no range"}
    q10, q50, q90 = q
    ret = float(r["r"])
    side = "above" if ret > q90 else "below" if ret < q10 else "inside"
    by = ((ret - q90) / max(q90 - q50, 1e-6) if side == "above" else (q10 - ret) / max(q50 - q10, 1e-6)
          if side == "below" else 0.0)
    base = {"side": side, "by": round(float(by), 2)}
    try:
        o0 = dt.date.fromisoformat(str(r.get("ref") or r["made"])[:10]).toordinal()
    except ValueError:
        return {**base, "c": "unknown"}
    i0 = bisect.bisect_left(dates, o0)
    if i0 >= len(dates) or dates[i0] != o0:
        return {**base, "c": "unknown", "why": "no price on the forecast's date"}
    if r.get("end"):
        o1 = dt.date.fromisoformat(r["end"][:10]).toordinal()
        i1 = bisect.bisect_right(dates, o1) - 1
    else:
        i1 = i0 + h
    i1 = min(i1, len(dates) - 1)
    if i1 <= i0:
        return {**base, "c": "unknown", "why": "no prices in the window"}
    c = np.asarray(closes[i0:i1 + 1], dtype=float)
    ds = list(dates[i0:i1 + 1])
    ok = np.isfinite(c) & (c > 0)
    ok &= np.asarray([d in spy for d in ds])
    if ok.sum() < 2 or not ok[0]:
        return {**base, "c": "unknown", "why": "gaps in the prices"}
    c = c[ok]
    ds = [d for d, k in zip(ds, ok) if k]
    m = np.asarray([spy[d] for d in ds], dtype=float)
    rs, rm = np.diff(np.log(c)), np.diff(np.log(m))
    b_rec = (r.get("f") or {}).get("beta")
    beta_est = None
    if not isinstance(b_rec, (int, float)) or not math.isfinite(b_rec):
        beta_est = _beta_before(dates, closes, spy, i0)    # (Claude's rows don't carry the model's signals)
    beta = float(np.clip(float(b_rec if beta_est is None else beta_est), 0.3, 2.5))
    e = rs - beta * rm
    window = ds[1:]                                   # the sessions whose moves are in the window (e[k]: into window[k])
    sig_h = (q90 - q10) / (2 * Z80)                   # the spread the forecast allowed over the whole horizon
    sig_d = max(sig_h / math.sqrt(max(h, 1)), 1e-5)
    err = ret - q50
    mkt = beta * (mkt_err if mkt_err is not None else float(np.sum(rm)))
    sp_days = _quarterly(spikes or [], dates)
    eidx = _reaction_days(ds, earnings or [], when or {}, sp_days)
    from_spike = set(_reaction_days(ds, [], {}, sp_days)) - set(_reaction_days(ds, earnings or [], when or {}))
    # an unknown-timing release has two candidate days: the bigger move counts, not both
    epos, used = [], set()
    for k in eidx:
        if k in used:
            continue
        pair = [j for j in (k, k + 1) if j in eidx and j not in used] if k + 1 in eidx else [k]
        best = max(pair, key=lambda j: abs(e[j]))
        epos.append(best)
        used.update(pair)
    earn = float(sum(e[k] for k in epos))
    rest_idx = [k for k in range(len(e)) if k not in set(epos)]
    news, jd, jz = 0.0, None, 0.0
    if rest_idx:
        k = max(rest_idx, key=lambda j: abs(e[j]))
        jz = abs(float(e[k])) / sig_d
        if jz >= (4.0 if h >= 5 else 2.5):
            news, jd = float(e[k]), window[k]
    vol = float(np.sqrt(np.mean(np.square(rs)))) / sig_d if len(rs) else None
    parts = {"market": mkt, "earnings": earn, "news": news}
    parts["trend"] = err - mkt - earn - news
    sgn = 1.0 if err >= 0 else -1.0
    push = {k: v * sgn for k, v in parts.items() if k != "trend" and v * sgn > 0}
    big = max(push, key=push.get) if push else None
    share = (push[big] / abs(err)) if big and abs(err) > 1e-9 else 0.0
    if big and share >= 0.5:
        cause = big
    elif (vol is not None and vol >= 1.5) or h < 5:     # (a one-day forecast has no "trend": the day was just big)
        cause = "volatility"
    elif big and share >= 0.3:
        cause = big
    else:
        cause = "trend"
    out = {**base, "c": cause, "err": round(err, 4), "parts": {k: round(v, 4) for k, v in parts.items()},
           "beta": round(beta, 2)}
    if vol is not None:
        out["vol"] = round(vol, 2)
    if epos:
        k = max(epos, key=lambda j: abs(e[j]))
        out["ed"], out["em"] = dt.date.fromordinal(window[k]).isoformat(), round(float(e[k]), 4)
        if k in from_spike:
            out["est"] = 1                               # an earnings day estimated from a volume spike
    if jd is not None:
        out["jd"], out["jm"], out["jz"] = dt.date.fromordinal(jd).isoformat(), round(news, 4), round(jz, 1)
        f8 = _filing_near(filings or [], jd)
        if f8:
            out["f8"] = f8
    if mkt_err is not None:
        out["mkt_err"] = round(float(mkt_err), 4)
    out["mkt"] = round(float(np.sum(rm)), 4)
    if beta_est is not None:
        out["beta_est"] = 1                              # estimated from the year of prices before the forecast
    return out


def _beta_before(dates: list[int], closes, spy: dict[int, float], i0: int, n: int = 252) -> float:
    """The stock's beta to the S&P 500 over the year of sessions before index `i0` (1 when there's too little)."""
    lo = max(0, i0 - n)
    c = np.asarray(closes[lo:i0 + 1], dtype=float)
    m = np.asarray([spy.get(d, np.nan) for d in dates[lo:i0 + 1]], dtype=float)
    ok = np.isfinite(c) & (c > 0) & np.isfinite(m) & (m > 0)
    if ok.sum() < 120:
        return 1.0
    a, b = np.diff(np.log(c[ok])), np.diff(np.log(m[ok]))
    vb = float(np.var(b))
    return float(np.cov(a, b)[0, 1] / vb) if vb > 0 else 1.0


def _quarterly(spikes: list[str], dates: list[int]) -> list[str]:
    """Estimated earnings days (volume spikes) that keep a quarterly rhythm: another one 45-80 sessions before or
    after. A one-off spike (a takeover bid, a drug approval) is more likely news than earnings."""
    pos = []
    for s in spikes:
        try:
            o = dt.date.fromisoformat(str(s)[:10]).toordinal()
        except ValueError:
            continue
        k = bisect.bisect_left(dates, o)
        if k < len(dates) and dates[k] == o:
            pos.append((k, str(s)[:10]))
    ks = [k for k, _ in pos]
    return [d for k, d in pos if any(45 <= abs(k - j) <= 80 for j in ks if j != k)]


def _filing_near(filings: list[dict], day: int) -> dict | None:
    """An 8-K filed on the shock's day or the day before (its first material item)."""
    for f in filings:
        try:
            o = dt.date.fromisoformat(str(f.get("date") or "")[:10]).toordinal()
        except ValueError:
            continue
        if day - 3 <= o <= day:
            for it in f.get("items") or []:
                if isinstance(it, dict) and it.get("item") not in ROUTINE:
                    return {"item": it.get("item"), "label": it.get("label"), "date": f.get("date")}
    return None


def _window_end(r: dict, h: int) -> str:
    """The last session of a graded forecast's window (its recorded end, else h sessions after its date)."""
    if r.get("end"):
        return str(r["end"])[:10]
    return trading_days_after(dt.date.fromisoformat(str(r.get("ref") or r["made"])[:10]), h)[-1].isoformat()


def known_earnings(x: dict | None, made: str) -> tuple[list[str], dict] | None:
    """The stock's earnings dates from its filings, if that record reaches back to the forecast's date (the SEC
    record kept covers recent years: on an older date, no filings would look like no earnings at all). None: use
    estimated earnings days (volume spikes) instead."""
    known = (x or {}).get("past_earnings") or []
    if known and min(known) <= str(made)[:10]:
        return known, (x or {}).get("earnings_when") or {}
    return None


def panel_view(P) -> tuple[list[int], dict[int, float]] | None:
    """A price history's sessions and the S&P 500 ETF's close by session (worked out once per history)."""
    spy = P.c(BENCH)
    if spy is None:
        return None
    dates = [int(d) for d in P.dates]
    return dates, {d: float(v) for d, v in zip(dates, spy) if v is not None and math.isfinite(float(v)) and v > 0}


def diagnose_panel(r: dict, h: int, P, view: tuple[list[int], dict[int, float]] | None, mkt_err: float | None = None,
                   x: dict | None = None, spikes: list[str] | None = None) -> dict:
    """`diagnose` with prices from a quant.Panel (the backtest's price histories); `view` from panel_view(P)."""
    c = P.c(r["t"])
    if c is None or view is None:
        return {"c": "unknown", "why": "no prices"}
    x = x or {}
    ke = known_earnings(x, r["made"])
    return diagnose(r, h, view[0], np.asarray(c, dtype=float), view[1], mkt_err, ke[0] if ke else [],
                    ke[1] if ke else {}, x.get("filings") or [], None if ke else spikes)


# ---------------------------------------------------------------- filling in the ledger's misses

def market_errors(L: dict, src: str) -> dict[tuple[str, int], float]:
    """The S&P 500's own forecast error by (date, horizon) for a source (live sources use the model's own)."""
    s = "quant_bt" if src == "quant_bt" else "quant"
    out = {}
    for hk, rows in (L.get("graded", {}).get(s) or {}).items():
        for r in rows:
            if r.get("t") == BENCH and "err" in r:
                out[(r["made"], int(hk))] = float(r["err"])
    return out


def fill_why(L: dict, src: str, get, ctx: dict | None = None, limit: int = 4000, misses_only: bool = True,
             spike_fn=None, today: dt.date | None = None) -> int:
    """Diagnose the graded misses of a source that don't have a diagnosis yet (newest first), storing it on the row
    ("why"), so each is worked out once. `get(t)` → (dates, closes, volumes) or None. `spike_fn(closes, volumes,
    dates)` → candidate earnings days (volume spikes), for stocks whose filing record doesn't reach back far enough;
    each miss uses only the ones up to its window's end, as they'd have been picked then."""
    from .insight import pick_spikes
    ctx = ctx or {}
    spy_s = get(BENCH)
    if not spy_s:
        return 0
    spy = {int(d): float(c) for d, c in zip(spy_s[0], spy_s[1]) if c is not None and math.isfinite(float(c)) and c > 0}
    merr = market_errors(L, src)
    today = (today or now_et().date()).isoformat()
    todo = []
    for hk, rows in (L.get("graded", {}).get(src) or {}).items():
        for r in rows:
            if r.get("t") == BENCH or (misses_only and r.get("in80", 1)):
                continue
            w = r.get("why")
            # one that couldn't be taken apart for want of prices is tried again on a later day, up to three times
            if w is not None and not (w.get("c") == "unknown" and w.get("try", 3) < 3 and w.get("on", "") < today):
                continue
            todo.append((r.get("end") or r["made"], int(hk), r))
    todo.sort(key=lambda x: x[0], reverse=True)
    series: dict = {}
    spikes: dict = {}
    n = 0
    for _, h, r in todo[:limit]:
        t = r["t"]
        if t not in series:
            series[t] = get(t)
        s = series[t]
        tries = int((r.get("why") or {}).get("try", 0)) + 1
        if not s:
            r["why"] = {"c": "unknown", "why": "no prices", "try": tries, "on": today}
            n += 1
            continue
        x = ctx.get(t) or {}
        ke = known_earnings(x, r["made"])
        sp = None
        if ke is None and spike_fn is not None and s[2] is not None:
            if t not in spikes:
                spikes[t] = spike_fn(np.asarray(s[1], dtype=float), np.asarray(s[2], dtype=float),
                                     np.asarray(s[0], dtype=np.int64))
            sp = pick_spikes(spikes[t], _window_end(r, h))
        ref = str(r.get("ref") or r["made"])[:10]
        w = diagnose(r, h, list(s[0]), np.asarray(s[1], dtype=float), spy, merr.get((ref, h)),
                     ke[0] if ke else [], ke[1] if ke else {}, x.get("filings") or [], sp)
        if w.get("c") == "unknown":
            w.update(**{"try": tries, "on": today})
        r["why"] = w
        n += 1
    return n


# ---------------------------------------------------------------- the page's numbers

def _blocks(rows: list[dict], h: int) -> list[int]:
    days = max(7, int(math.ceil(h * 7 / 5)))
    return [dt.date.fromisoformat(r["made"][:10]).toordinal() // days for r in rows]


def _rate(rows: list[dict], key: str, h: int) -> dict | None:
    rr = [r for r in rows if key in r]
    if not rr:
        return None
    y = [float(r[key]) for r in rr]
    lo, hi, g = clustered_interval(y, _blocks(rr, h))
    return {"p": round(float(np.mean(y)), 4), "lo": round(lo, 4), "hi": round(hi, 4), "n": len(rr), "periods": g}


def score(rows: list[dict], h: int) -> dict:
    """Win rate (inside the 80% range), direction (said vs expected vs "always up") and skill over a set of rows."""
    if not rows:
        return {"n": 0}
    out = {"n": len(rows), "win": _rate(rows, "in80", h)}
    d = [r for r in rows if "hit" in r and r.get("pup") is not None]
    if d:
        out["dir"] = _rate(d, "hit", h)
        out["dir_expected"] = round(float(np.mean([max(r["pup"], 1 - r["pup"]) for r in d])), 4)
        out["always_up"] = round(float(np.mean([r.get("up", 0) for r in d])), 4)
    sk = [r for r in rows if r.get("qsn") and r.get("qs") is not None]
    if sk:
        out["skill"] = round(1 - sum(r["qs"] for r in sk) / sum(r["qsn"] for r in sk), 4)
    return out


def _terciles(rows: list[dict], key: str) -> tuple[float, float] | None:
    v = [float(r["f"][key]) for r in rows if (r.get("f") or {}).get(key) is not None]
    if len(v) < 60:
        return None
    return float(np.quantile(v, 1 / 3)), float(np.quantile(v, 2 / 3))


def groups(rows: list[dict], h: int, mret: dict, industry: dict) -> list[dict]:
    """Where the forecasts do better or worse: the same scores for slices of the record."""
    out = []

    def add(kind: str, label: str, sel: list[dict]):
        if len(sel) >= 20:
            out.append({"kind": kind, "label": label, **score(sel, h)})
    with_e = [r for r in rows if (r.get("f") or {}).get("ein") == 1]
    without = [r for r in rows if (r.get("f") or {}).get("ein") == 0]
    add("earnings", "An earnings report inside the window", with_e)
    add("earnings", "No earnings report inside", without)
    for key, kind, names in (("lvol", "volatility", ("Calm stocks (lowest third by volatility)", "Middle third",
                                                      "Volatile stocks (highest third)")),
                             ("liq", "liquidity", ("Thinly traded (lowest third by dollar volume)", "Middle third",
                                                   "Heavily traded (highest third)"))):
        tq = _terciles(rows, key)
        if tq:
            a, b = tq
            add(kind, names[0], [r for r in rows if (r.get("f") or {}).get(key) is not None and r["f"][key] <= a])
            add(kind, names[1], [r for r in rows if (r.get("f") or {}).get(key) is not None and a < r["f"][key] <= b])
            add(kind, names[2], [r for r in rows if (r.get("f") or {}).get(key) is not None and r["f"][key] > b])
    up = [r for r in rows if mret.get((r["made"], h)) is not None and mret[(r["made"], h)] > 0]
    dn = [r for r in rows if mret.get((r["made"], h)) is not None and mret[(r["made"], h)] <= 0]
    add("market", "The S&P 500 rose over the window", up)
    add("market", "The S&P 500 fell over the window", dn)
    add("nudge", "Nudged by the signal learner", [r for r in rows if r.get("tl")])
    by_i: dict[str, list] = {}
    for r in rows:
        ind = industry.get(r["t"])
        if ind:
            by_i.setdefault(ind, []).append(r)
    for ind, sel in sorted(by_i.items(), key=lambda kv: -len(kv[1]))[:8]:
        if len(sel) >= 40:
            add("industry", ind, sel)
    return out


def _price(p0: float | None, x: float | None) -> float | None:
    if p0 is None or x is None or not math.isfinite(p0) or p0 <= 0:
        return None
    return round(p0 * math.exp(x), 4 if p0 < 1 else 2)


def _p0(r: dict, get, cache: dict) -> float | None:
    if r.get("p0"):
        return float(r["p0"])
    t = r["t"]
    if t not in cache:
        cache[t] = get(t)
    s = cache[t]
    if not s:
        return None
    try:
        o = dt.date.fromisoformat(str(r.get("ref") or r["made"])[:10]).toordinal()
    except ValueError:
        return None
    k = bisect.bisect_left(list(s[0]), o)
    if k < len(s[0]) and s[0][k] == o and s[1][k] is not None and math.isfinite(float(s[1][k])):
        return float(s[1][k])
    return None


def row_view(r: dict, h: int, get, cache: dict, full: bool = False) -> dict:
    """One graded forecast as the page lists it (prices from its anchor close)."""
    q = published(r)
    p0 = _p0(r, get, cache)
    end = r.get("end") or trading_days_after(dt.date.fromisoformat(str(r.get("ref") or r["made"])[:10]), h)[-1].isoformat()
    v = {"t": r["t"], "h": NAMES.get(str(h), str(h)), "made": r["made"][:10], "end": end,
         "win": int(r.get("in80", 0)), "ret": round(math.expm1(float(r["r"])), 4)}
    if "hit" in r:
        v["dir"] = int(r["hit"])
        v["pup"] = r.get("pup")
    if p0:
        v["p0"] = round(p0, 4 if p0 < 1 else 2)
        v["p1"] = _price(p0, float(r["r"]))
        if q:
            v["lo"], v["mid"], v["hi"] = (_price(p0, x) for x in q)
    if full and r.get("why"):
        w = r["why"]
        v["why"] = {k: w[k] for k in ("c", "side", "by", "err", "parts", "vol", "ed", "em", "jd", "jm", "jz", "f8", "mkt",
                                      "mkt_err", "beta") if k in w}
    if r.get("delisted"):
        v["delisted"] = 1
    return v


# ---------------------------------------------------------------- how each kind of miss is being corrected

def _pc(x: float | None, d: int = 0, sign: bool = False) -> str:
    if x is None:
        return "?"
    s = f"{x * 100:+.{d}f}%" if sign else f"{x * 100:.{d}f}%"
    return s


def corrections(lr: dict) -> dict:
    """What the learning loop is doing right now, per horizon (from its last decisions)."""
    out = {}
    cal, rng, kp, tilt = lr.get("calib") or {}, lr.get("range") or {}, lr.get("keeper") or {}, lr.get("tilt") or {}
    for name, h in HORIZONS:
        c, g, k = cal.get(name) or {}, rng.get(name) or {}, kp.get(name) or {}
        out[name] = {
            "calibration": {"status": c.get("status", "learning"), "scale": c.get("scale"), "shift": c.get("shift"),
                            "from": c.get("from"), "gain": c.get("holdout_gain"),
                            "cov_before": c.get("cov80_holdout_raw"), "cov_after": c.get("cov80_holdout")},
            "range": {"status": g.get("status", "learning") if name in RANGE_H else "n/a",
                      "gain": g.get("holdout_gain_vs_single"), "spread": g.get("scale_spread"), "from": g.get("from")},
            "keeper": {"status": k.get("status", "learning") if name in KEEPER_H else "n/a", "m": k.get("m"),
                       "gain": k.get("replay_gain")}}
    out["signal"] = {"status": tilt.get("status"), "on": bool(tilt.get("on")), "model": tilt.get("model"),
                     "fusion": tilt.get("fusion"), "fusion_months": tilt.get("fusion_months"),
                     "fusion_waiting": tilt.get("fusion_waiting"), "t": tilt.get("t"), "t_needed": tilt.get("t_needed"),
                     "test_months": tilt.get("test_months"), "best": tilt.get("best")}
    return out


RANGE_H = ("1d", "1w", "1m", "3m")       # the horizons the range learner and the coverage keeper run at
KEEPER_H = RANGE_H
ADJ = {"1d": "1-day", "1w": "1-week", "1m": "1-month", "3m": "3-month", "1y": "1-year"}


def _model_fix(cause: str, name: str, cor: dict, why: dict | None) -> str:
    """What the learning loop does about this kind of miss at this horizon, for the model's own forecasts."""
    from .ml import NAMES as ALGO
    c = cor.get(name) or {}
    cal, rng, kp = c.get("calibration") or {}, c.get("range") or {}, c.get("keeper") or {}
    sig = cor.get("signal") or {}
    hz = ADJ.get(name, name)

    def keeper() -> str:
        if name not in KEEPER_H:
            return f"the coverage keeper doesn't run for {hz} ranges (their results come in too slowly to steer by)"
        if kp.get("status") == "applied":
            return f"the coverage keeper has {'widened' if (kp.get('m') or 1) >= 1 else 'narrowed'} {hz} ranges ×{kp['m']:.2f} after the latest results"
        if kp.get("status") == "held":
            return f"the coverage keeper is watching {hz} ranges (off for now: its replay didn't beat leaving them alone)"
        return "the coverage keeper starts once it has 40 days of live results"

    def ranges() -> str:
        if name not in RANGE_H:
            return f"the range learner doesn't run for {hz} ranges (too few independent {hz} outcomes to test it)"
        if rng.get("status") == "applied":
            return f"the range learner widens {hz} ranges for the stocks and situations that need it (it beat one width for all in its test)"
        if rng.get("status") == "held":
            return f"the range learner tested wider {hz} ranges for situations like this and found they didn't help yet"
        return f"the range learner is still collecting {hz} forecasts to test this"

    def calib() -> str:
        if cal.get("status") == "applied" and cal.get("scale"):
            return f"the calibration has set {hz} ranges ×{cal['scale']:.2f} from how often they missed"
        if cal.get("status") == "held":
            return f"the calibration's correction for {hz} ranges is held back (it didn't help the newest forecasts)"
        return f"the calibration is still collecting {hz} results"

    def signal() -> str:
        if sig.get("on"):
            m = sig.get("model")
            what = (f"the fused model ({', '.join(ALGO.get(b, b) for b in sig.get('fusion') or [])})" if m == "fusion"
                    else ALGO.get(m, m))
            return f"the signal learner is on ({what}) and nudges forecasts by what its test found"
        if sig.get("t") is not None and sig.get("t_needed"):
            return (f"the signal learner is off: no algorithm has passed its test yet (t {sig['t']:.1f}; "
                    f"{sig['t_needed']:.1f} needed)")
        return "the signal learner is off until an algorithm passes its test"
    w = why or {}
    if cause == "market":
        return (f"No stock-level signal can call the market. {keeper()[0].upper() + keeper()[1:]}, and the S&P 500's "
                f"own surprise is taken out of what the signal learner learns from.")
    if cause == "earnings":
        return (f"Each stock's earnings-day risk is re-measured from its own reactions at every forecast, and "
                f"{ranges()}; how often a stock's misses came from earnings is one of its inputs.")
    if cause == "news":
        f8 = f" (the day of an 8-K: {w['f8']['label'].lower()})" if w.get("f8") and w["f8"].get("label") else ""
        return (f"One-day shocks{f8} are why the ranges have fat tails: {calib()}; news coverage and 8-K filings are "
                f"inputs the signal learner tests.")
    if cause == "volatility":
        return f"The volatility estimate updates with every close, and {ranges()} (recent against usual volatility is one of its inputs)."
    if cause == "trend":
        return f"A steady drift is what the signal learner tries to predict: {signal()}."
    return _OTHER.get(cause, _OTHER["unknown"])


_OTHER = {"pending": "Not taken apart yet: live misses are taken apart within the hour after they're graded, backtest ones "
                     "as the backtest runs.",
          "unknown": "The prices for that window weren't available, so it couldn't be taken apart."}


def fix_text(cause: str, name: str, cor: dict, why: dict | None = None, src: str = "quant") -> str:
    """What the learning loop does about this kind of miss at this horizon, said plainly, for whose forecast it was."""
    if cause in _OTHER:
        return _OTHER[cause]
    if src == "claude":
        lead = "No stock-level view can call the market. " if cause == "market" else ""
        return (f"{lead}Claude's ranges are its own: the model's learners don't change them. Its biggest misses get a "
                f"post-mortem (up to two a day), the causes at least two post-mortems agree on go into its next "
                f"forecasts, and how much its forecasts count in the combined forecast follows its measured record.")
    model = _model_fix(cause, name, cor, why)
    if src == "blend":
        return (f"The combined forecast weighs the model and Claude by their measured skill, re-weighted as results "
                f"come in. For the model's part: {model[0].lower() + model[1:]}")
    return model


# ---------------------------------------------------------------- the whole page

def _tot(months: dict) -> dict:
    """Sums of a source and horizon's monthly totals (every graded forecast, as the Records tab counts them)."""
    t = {k: 0 for k in ("n", "in80", "hits", "nhit", "up", "npexp", "hitx")}
    t.update(qs=0.0, qsn=0.0, pexp=0.0)
    for m in months.values():
        for k in ("n", "in80", "hits", "nhit", "up"):
            t[k] += int(m.get(k) or 0)
        t["qs"] += float(m.get("qs") or 0.0)
        t["qsn"] += float(m.get("qsn") or 0.0)
        if "pexp" in m:                                 # (recorded from this version on)
            t["pexp"] += float(m["pexp"])
            t["npexp"] += int(m.get("npexp") or 0)
            t["hitx"] += int(m.get("hitx") or 0)
    return t


def _add(a: dict, b: dict) -> dict:
    return {k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b)}


def _totals(t: dict, rows: list[dict]) -> dict:
    """The all-time figures. What the forecasts expected comes from the same months when they recorded it (from
    this version on), else from the forecasts kept in detail."""
    out: dict = {"n": int(t["n"])}
    if t["n"]:
        out.update(win=round(t["in80"] / t["n"], 4), n_miss=int(t["n"] - t["in80"]),
                   always_up=round(t["up"] / t["n"], 4))
    if t["nhit"]:
        out.update(dir=round(t["hits"] / t["nhit"], 4), dir_n=int(t["nhit"]))
    # expected against right on the same calls: all of them once every month records what was expected; until then
    # the calls that recorded it (if they're 30%+ of them), else the forecasts kept in detail
    if t["npexp"] and t["npexp"] == t["nhit"]:
        out.update(dir_expected=round(t["pexp"] / t["npexp"], 4), dir_expected_from="all")
    elif t["npexp"] and t["npexp"] >= 0.3 * t["nhit"]:
        out.update(dir_expected=round(t["pexp"] / t["npexp"], 4), dir_cmp=round(t["hitx"] / t["npexp"], 4),
                   dir_cmp_n=int(t["npexp"]), dir_expected_from="recorded")
    else:
        d = [r for r in rows if "hit" in r and r.get("pup") is not None]
        if d:
            out.update(dir_expected=round(float(np.mean([max(r["pup"], 1 - r["pup"]) for r in d])), 4),
                       dir_cmp=round(float(np.mean([r["hit"] for r in d])), 4), dir_cmp_n=len(d),
                       dir_expected_from="kept")
    if t["qsn"] > 0:
        out["skill"] = round(1 - t["qs"] / t["qsn"], 4)
    return out


def _trend(by_month: dict) -> list[dict]:
    out = []
    for mo in sorted(by_month):
        m = by_month[mo]
        if not m.get("n"):
            continue
        out.append({"m": mo, "n": int(m["n"]), "win": round(m["in80"] / m["n"], 4),
                    "miss": int(m["n"] - m["in80"]), "dir": round(m["hits"] / m["nhit"], 4) if m.get("nhit") else None})
    return out


def _cause(r: dict) -> str:
    return (r.get("why") or {}).get("c") or "pending"


def build(L: dict, get, lr: dict, ctx: dict | None = None, today: dt.date | None = None, recent_days: int = 90,
          per_h_misses: int = 40, per_h_rows: int = 60) -> dict:
    """The Results page's data (data/results.json). The win rates, direction and skill count every graded forecast
    (the monthly totals, as on the Records tab); the slices, the cause breakdown and the lists use the forecasts the
    ledger keeps in detail (a fixed random sample of up to 50 a day per horizon, 150 for the backtest)."""
    today = today or now_et().date()
    ctx = ctx or {}
    industry = {t: x.get("industry") for t, x in ctx.items() if x.get("industry")}
    cor = corrections(lr)
    cut = (today - dt.timedelta(days=recent_days)).isoformat()
    cut_ci = (today - dt.timedelta(days=460)).isoformat()
    srcs = {}
    cache: dict = {}
    for src in ("quant", "claude", "blend", "quant_bt"):
        agg = (L.get("agg") or {}).get(src) or {}
        by_h = L.get("graded", {}).get(src) or {}
        kept = {int(hk): [r for r in rows if r.get("t") != BENCH and "in80" in r] for hk, rows in by_h.items()}
        if not any(m.get("n") for months in agg.values() for m in months.values()) and not any(kept.values()):
            continue
        mret = {}
        for hk, rows in (L.get("graded", {}).get("quant_bt" if src == "quant_bt" else "quant") or {}).items():
            for r in rows:
                if r.get("t") == BENCH and "r" in r:
                    mret[(r["made"], int(hk))] = float(r["r"])
        per_h, misses, listed = {}, [], []
        pooled = {k: 0 for k in ("n", "in80", "hits", "nhit", "up", "npexp")}
        pooled.update(qs=0.0, qsn=0.0, pexp=0.0)
        pooled_month: dict[str, dict] = {}
        cause_all: dict[str, int] = {}
        cause_recent: dict[str, int] = {}
        lesson: dict[str, dict[str, int]] = {}
        all_rows: list[dict] = []
        for name, h in HORIZONS:
            months = agg.get(str(h)) or {}
            rows = kept.get(h) or []
            t = _tot(months)
            if not t["n"] and not rows:
                continue
            pooled = _add(pooled, t)
            for mo, m in months.items():
                pooled_month[mo] = _add(pooled_month.get(mo, {}), {k: m.get(k) or 0 for k in ("n", "in80", "hits", "nhit")})
            sc = _totals(t, rows)
            ci = _rate([r for r in rows if r["made"] >= cut_ci], "in80", h)
            if ci and ci["n"] >= 20 and ci["periods"] >= 6:
                sc["win_ci"] = ci                    # (an interval on fewer independent periods isn't one to lean on)
            rec = [r for r in rows if (r.get("end") or r["made"]) >= cut]
            sc["recent"] = score(rec, h) if rec else {"n": 0}
            sc["groups"] = groups(rows, h, mret, industry)
            sc["trend"] = _trend(months)[-36:]
            if months:
                sc["first"] = min(months)
            mrows = sorted((r for r in rows if not r.get("in80")), key=lambda r: (r.get("end") or r["made"]), reverse=True)
            mc: dict[str, int] = {}
            for r in mrows:
                cz = _cause(r)
                mc[cz] = mc.get(cz, 0) + 1
                cause_all[cz] = cause_all.get(cz, 0) + 1
                if (r.get("end") or r["made"]) >= cut:
                    cause_recent[cz] = cause_recent.get(cz, 0) + 1
                lesson.setdefault(cz, {})[name] = lesson.get(cz, {}).get(name, 0) + 1
            sc.update(causes=mc, kept=len(rows), kept_miss=len(mrows))
            sc["fix"] = {cz: fix_text(cz, name, cor, None, src) for cz in mc}
            per_h[name] = sc
            for r in mrows[: per_h_misses if src != "quant_bt" else per_h_misses * 5 // 8]:
                v = row_view(r, h, get, cache, full=True)
                fx = fix_text(_cause(r), name, cor, r.get("why"), src)
                if fx != sc["fix"].get(_cause(r)):       # (the page uses the horizon's text unless this one differs)
                    v["fix"] = fx
                misses.append(v)
            newest = sorted(rows, key=lambda r: (r.get("end") or r["made"]), reverse=True)[:per_h_rows]
            listed += [row_view(r, h, get, cache) for r in newest]
            all_rows += rows
        head = _totals(pooled, all_rows)
        rec_pool = [r for r in all_rows if (r.get("end") or r["made"]) >= cut]
        if rec_pool:
            head["recent"] = {"n": len(rec_pool), "win": round(float(np.mean([r["in80"] for r in rec_pool])), 4)}
        if pooled_month:
            head["first"] = min(pooled_month)
        misses.sort(key=lambda v: v["end"], reverse=True)
        listed.sort(key=lambda v: v["end"], reverse=True)
        known = sum(n for cz, n in cause_all.items() if cz != "pending")
        lessons = [{"cause": cz, "name": CAUSE_NAMES.get(cz, cz), "n": n, "share": round(n / known, 3) if known else None,
                    "top_h": max(lesson[cz], key=lesson[cz].get),
                    "fix": {nm: fix_text(cz, nm, cor, None, src) for nm in lesson[cz]}}
                   for cz, n in sorted(cause_all.items(), key=lambda kv: CAUSES.index(kv[0]) if kv[0] in CAUSES else 99)
                   if cz != "pending"]
        srcs[src] = {"headline": head, "by_h": per_h, "trend": _trend(pooled_month)[-36:],
                     "causes": {"all": cause_all, "recent": cause_recent}, "lessons": lessons,
                     "misses": misses, "graded": listed}
    return {"generated_at": iso(now_utc()), "recent_days": recent_days, "definitions": {
        "win": "the price ended inside the forecast's 80% range (about 80% should, by design)",
        "dir": "the forecast said more likely up (or down) and the price went that way",
        "dir_expected": "what the forecasts expected: the average of their own odds for the side they picked",
        "skill": "how much better its quantile score was than a random walk's (above 0 is better)",
        "counts": "the rates count every graded forecast (the S&P 500's own included, as on the Records tab); the slices, "
                  "causes and lists use the forecasts kept in detail: a fixed random sample of up to 50 a day per "
                  "horizon (150 for the backtest)"},
        "cause_names": CAUSE_NAMES, "sources": srcs, "corrections": cor}
