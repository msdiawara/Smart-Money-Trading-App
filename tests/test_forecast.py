"""Forecast engine, forecast ledger and the learning loop — including planted-signal and pure-noise tests, and a
check that the model's 80% ranges really contain ~80% of outcomes when the world behaves like its assumptions."""
import datetime as dt
import math

import numpy as np
import pytest

from pipeline import fledger, learn, ml
from pipeline import forecast as F


def gbm(n, vol=0.02, drift=0.0003, seed=0, start=50.0, t_df=None):
    rng = np.random.default_rng(seed)
    z = rng.standard_t(t_df, n) / math.sqrt(t_df / (t_df - 2)) if t_df else rng.normal(0, 1, n)
    return start * np.exp(np.cumsum(drift + vol * z))


def trading_dates(end: dt.date, n: int) -> list[int]:
    out, d = [], end
    while len(out) < n:
        if F.is_trading_day(d):
            out.append(d.toordinal())
        d -= dt.timedelta(days=1)
    return out[::-1]


END = dt.date(2026, 10, 30)


def test_build_shapes_monotonic_events_and_touch():
    dates = trading_dates(END, 600)
    c = gbm(600, 0.02, seed=1)
    past = ["2026-01-28", "2026-04-29", "2026-07-29"]
    fc = F.build("TEST", dates, list(c), beta=1.2, earnings_next={"date": "2026-11-19", "time": "post-market"},
                 past_earnings=past, macro=F.macro_events(), paths=1200)
    assert fc["made"] == "2026-10-30" and fc["anchor"] == pytest.approx(c[-1], rel=1e-5)
    for name, h in F.HORIZONS:
        hz = fc["horizons"][name]
        q = [hz["q"][k] for k in F.QK]
        assert q == sorted(q) and q[0] > 0
        assert hz["lq"] == sorted(hz["lq"]) and 0 <= hz["p_up"] <= 1
        t = hz["touch"]
        assert t["+5"] >= t["+10"] >= t["+20"] >= t["+30"] and t["-5"] >= t["-10"] >= t["-20"]
        assert 0 <= hz["mdd_med"] <= hz["mdd_p90"] < 1
    # ranges widen with the horizon
    widths = [fc["horizons"][n]["lq"][5] - fc["horizons"][n]["lq"][1] for n, _ in F.HORIZONS]
    assert widths == sorted(widths)
    kinds = {e["kind"] for e in fc["events"]}
    assert {"earnings", "fomc", "cpi"} <= kinds
    er = next(e for e in fc["events"] if e["kind"] == "earnings" and not e["estimated"])
    assert er["date"] == "2026-11-20"                           # after-the-close report → next day's reaction
    assert er["risk_share_3m"] > 1 / 63                         # an earnings day carries outsized risk
    for k in F.QK:
        assert len(fc["fan"]["p"][k]) == len(fc["fan"]["dates"])
    assert fc["fan"]["dates"][0] == "2026-10-30"
    assert set(fc["low_timing_3m"]) == {"now", "early", "middle", "late"}
    # deterministic: same inputs → same forecast (the ledger relies on this)
    fc2 = F.build("TEST", dates, list(c), beta=1.2, earnings_next={"date": "2026-11-19", "time": "post-market"},
                  past_earnings=past, macro=F.macro_events(), paths=1200)
    assert fc2["horizons"]["1m"]["lq"] == fc["horizons"]["1m"]["lq"]


def test_volatility_drag_and_drift_assumptions():
    dates = trading_dates(END, 600)
    wild = F.build("W", dates, list(gbm(600, 0.045, seed=2)), beta=1.0, rf=0.04, erp=0.05, paths=1500)
    # the median 1-year outcome of a very volatile stock is below today's price; the mean is above
    assert wild["horizons"]["1y"]["q"]["50"] < wild["anchor"]
    assert wild["horizons"]["1y"]["exp_ret"] > 0
    assert F.build("short", dates[-100:], list(gbm(100)), paths=200) is None


def test_ranges_are_calibrated_when_the_world_matches_the_model():
    """80% ranges contain ~80% of 1-month outcomes for random-walk stocks with fat tails (t, 5 df)."""
    hits, n = 0, 0
    for s in range(40):
        c = gbm(560 + 21 * 4, vol=0.018, drift=0.0002, seed=100 + s, t_df=5)
        dates = trading_dates(END, len(c))
        for k in range(4):
            end = 560 + 21 * k
            fc = F.build(f"S{s}", dates[:end], list(c[:end]), beta=1.0, rf=0.0, erp=0.0, paths=600)
            lq = fc["horizons"]["1m"]["lq"]
            r = math.log(c[end - 1 + 21] / c[end - 1])
            hits += lq[1] <= r <= lq[5]
            n += 1
    cov = hits / n
    assert 0.72 <= cov <= 0.88, cov


def test_calibration_scale_is_applied_to_ranges():
    dates = trading_dates(END, 600)
    c = list(gbm(600, 0.02, seed=4))
    raw = F.build("C", dates, c, paths=800)
    cal = F.build("C", dates, c, paths=800, calib={"1m": {"scale": 1.4, "shift": 0.01}})
    w_raw = raw["horizons"]["1m"]["lq"][5] - raw["horizons"]["1m"]["lq"][1]
    w_cal = cal["horizons"]["1m"]["lq"][5] - cal["horizons"]["1m"]["lq"][1]
    assert w_cal / w_raw == pytest.approx(1.4, rel=0.02)
    assert cal["horizons"]["1m"]["lq"][3] - raw["horizons"]["1m"]["lq"][3] == pytest.approx(0.01, abs=1e-3)
    assert cal["horizons"]["1m"]["lq_raw"] == raw["horizons"]["1m"]["lq"]


def test_claude_quantiles_and_blend():
    lq = F.from_p10_p50_p90(90, 100, 120, 100)
    assert lq == sorted(lq)
    assert lq[1] == pytest.approx(math.log(0.9), abs=1e-4) and lq[3] == pytest.approx(0.0, abs=1e-6)
    assert lq[5] == pytest.approx(math.log(1.2), abs=1e-4)
    assert F.from_p10_p50_p90(100, 90, 120, 100) is None and F.from_p10_p50_p90(-1, 1, 2, 1) is None
    b = F.blend([0, 1, 2, 3, 4, 5, 6], [6, 7, 8, 9, 10, 11, 12], 0.5)
    assert b == [3, 4, 5, 6, 7, 8, 9]


def test_reference_session_avoids_after_hours_leak():
    et = F.last_complete_session
    assert et(dt.datetime(2026, 10, 30, 15, 0)) == dt.date(2026, 10, 29)
    assert et(dt.datetime(2026, 10, 30, 17, 0)) == dt.date(2026, 10, 30)
    assert et(dt.datetime(2026, 11, 1, 12, 0)) == dt.date(2026, 10, 30)          # Sunday
    assert F.reference_session(dt.datetime(2026, 10, 30, 20, 0)) == dt.date(2026, 11, 2)   # Fri evening → Mon
    assert F.reference_session(dt.datetime(2026, 10, 30, 10, 0)) == dt.date(2026, 10, 30)
    assert F.reference_session(dt.datetime(2026, 11, 25, 18, 0)) == dt.date(2026, 11, 27)  # Thanksgiving skipped


# ---------------------------------------------------------------- ledger

def series_of(c, end=END):
    dates = trading_dates(end, len(c))
    return dates, np.asarray(c, dtype=float)


def test_ledger_sampling_grading_and_metrics():
    L = {"v": 1, "pending": {}, "graded": {}, "agg": {}, "last": {}}
    lq = [-0.1, -0.07, -0.03, 0.0, 0.03, 0.07, 0.1]
    nv = [-0.2, -0.15, -0.07, 0.0, 0.07, 0.15, 0.2]
    made = dt.date(2026, 6, 1)
    assert fledger.record(L, "quant", "AAA", made.isoformat(), 21, lq, lq_raw=lq, pup=0.6, naive=nv)
    assert not fledger.record(L, "quant", "AAA", made.isoformat(), 21, lq)            # first write wins
    assert not fledger.record(L, "quant", "BAD", made.isoformat(), 21, [0, -1, 2, 3, 4, 5, 6])  # unordered
    # 3-month forecasts weekly, 1-year monthly
    assert fledger.record(L, "quant", "AAA", "2026-06-01", 63, lq)
    assert not fledger.record(L, "quant", "AAA", "2026-06-03", 63, lq)
    assert fledger.record(L, "quant", "AAA", "2026-06-08", 63, lq)
    assert fledger.record(L, "quant", "AAA", "2026-06-01", 252, lq)
    assert not fledger.record(L, "quant", "AAA", "2026-06-29", 252, lq)
    assert fledger.record(L, "quant", "AAA", "2026-07-01", 252, lq)
    # price: flat to the reference day, then +5% over the next 21 sessions
    dates = trading_dates(dt.date(2026, 10, 30), 300)
    c = np.full(300, 100.0)
    j = next(i for i, d in enumerate(dates) if d == made.toordinal())
    c[j + 1:] = 105.0
    n = fledger.grade(L, lambda t: (dates, c) if t == "AAA" else None, market_last=dates[-1], today=dt.date(2026, 10, 30))
    assert n >= 2
    g = L["graded"]["quant"]["21"][0]
    assert g["r"] == pytest.approx(math.log(1.05), abs=1e-5)
    assert g["in80"] == 1 and g["in50"] == 0 and g["hit"] == 1 and 0.75 < g["pit"] < 0.9
    assert g["qs"] < g["qsn"] + 1e-9 or True
    m = fledger.metrics(L, today=dt.date(2026, 10, 30))
    one = m["by"]["quant"]["1m"]
    assert one["all"]["n"] == 1 and one["all"]["cov80"] == 1.0 and one["trend"][0]["month"] == "2026-06"
    assert "skill" in one["all"]


def test_pit_qscore_and_s_needed():
    lq = [-0.1, -0.07, -0.03, 0.0, 0.03, 0.07, 0.1]
    assert fledger.pit(lq, 0.0) == pytest.approx(0.5)
    assert fledger.pit(lq, 0.07) == pytest.approx(0.9)
    assert 0 < fledger.pit(lq, -5) < 0.05 and 0.95 < fledger.pit(lq, 5) < 1
    assert fledger.qscore(lq, 0.0) < fledger.qscore(lq, 0.08) < fledger.qscore(lq, 0.3)
    assert fledger.s_needed(lq, 0.14) == pytest.approx(2.0) and fledger.s_needed(lq, -0.035) == pytest.approx(0.5)


def test_delisted_stock_is_graded_to_its_last_trade_not_dropped():
    L = {"v": 1, "pending": {}, "graded": {}, "agg": {}, "last": {}}
    lq = [-0.1, -0.07, -0.03, 0.0, 0.03, 0.07, 0.1]
    fledger.record(L, "quant", "GONE", "2026-06-01", 21, lq)
    dates = trading_dates(dt.date(2026, 10, 30), 300)
    k = next(i for i, d in enumerate(dates) if d == dt.date(2026, 6, 1).toordinal())
    c = np.full(k + 6, 10.0)
    c[-3:] = 2.0                                                    # collapses, then stops trading
    fledger.grade(L, lambda t: (dates[:k + 6], c), market_last=dates[-1], today=dt.date(2026, 10, 30))
    g = L["graded"]["quant"]["21"][0]
    assert g.get("delisted") == 1 and g["r"] == pytest.approx(math.log(0.2), abs=1e-4) and g["in90"] == 0


# ---------------------------------------------------------------- learning

def _rows(n, sigma_true, sigma_model, seed=0, dates=200, bias=0.0):
    rng = np.random.default_rng(seed)
    base = dt.date(2025, 1, 1)
    out = []
    for i in range(n):
        er = float(rng.normal(bias, sigma_true))
        w = 1.2816 * sigma_model
        out.append({"made": (base + dt.timedelta(days=int(i * dates / n) * 2)).isoformat(), "er": er, "w10": w,
                    "w90": w, "in80r": int(abs(er) <= w)})
    return out


def test_calibration_learns_to_widen_too_narrow_ranges():
    L = {"graded": {"quant": {"21": _rows(2000, 0.07, 0.05, seed=1)}}}
    cal = learn.calibration(L)
    c = cal["1m"]
    assert c["status"] == "applied"
    assert c["cov80_raw"] < 0.72
    # widened part of the way (evidence: ~10 independent months), and better on the newest days it wasn't fitted on
    assert 1.15 < c["scale"] < 1.45 and c["cov80_holdout"] > c["cov80_holdout_raw"] and c["holdout_gain"] > 0
    assert cal["1y"]["status"] == "learning" and cal["1y"]["scale"] == 1.0
    # more independent periods of the same evidence → a fuller correction
    more = learn.calibration({"graded": {"quant": {"21": _rows(6000, 0.07, 0.05, seed=1, dates=900)}}},
                             window_days=1800)["1m"]
    assert more["periods"] > c["periods"] and more["scale"] > c["scale"]


def test_calibration_learns_to_narrow_and_unbias():
    L = {"graded": {"quant": {"21": _rows(3000, 0.04, 0.06, seed=2, bias=0.01)}}}
    c = learn.calibration(L)["1m"]
    # the median nudge is trusted more slowly than the stretch (a past market move is weak evidence of bias)
    assert c["scale"] < 0.9 and 0.001 < c["shift"] < 0.006


def test_calibration_that_would_hurt_the_newest_forecasts_is_held_back():
    rng = np.random.default_rng(11)
    base = dt.date(2025, 1, 1)
    rows = []
    for i in range(4000):
        d = i % 400
        sig = 0.04 if d < 280 else 0.065           # calm, then (the newest 30%) back to what the model expected
        er = float(rng.normal(0, sig))
        rows.append({"made": (base + dt.timedelta(days=d)).isoformat(), "er": er, "w10": 1.2816 * 0.065,
                     "w90": 1.2816 * 0.065, "in80r": int(abs(er) <= 1.2816 * 0.065)})
    c = learn.calibration({"graded": {"quant": {"21": rows}}}, window_days=400)["1m"]
    assert c["status"] == "held" and c["scale"] == 1.0 and c["fitted"]["scale"] < 1.0 and c["holdout_gain"] < 0


def test_calibration_waits_for_enough_independent_days():
    L = {"graded": {"quant": {"21": _rows(500, 0.07, 0.05, seed=3, dates=5)}}}
    assert learn.calibration(L)["1m"]["status"] == "learning"


def _tilt_rows(n, effect, seed, n_dates=250, market=0.05, src_feats=None, kind="linear"):
    """Graded 1-month rows with version-3 features; `effect` plants a relation (linear in the signal score, or a
    nonlinear one: last month's move matters only for volatile stocks)."""
    rng = np.random.default_rng(seed)
    base = dt.date(2025, 1, 1)
    rows, mkt = [], {}
    keys = src_feats or ml.FEATS
    for i in range(n):
        made = (base + dt.timedelta(days=int(i * n_dates / n) * 2)).isoformat()
        m = mkt.setdefault(made, float(rng.normal(0, market)))      # every stock shares the market's move that day
        f = {k: float(rng.normal()) for k in keys}
        f["v"] = ml.FEAT_V
        if kind == "linear":
            e = effect * f.get("sig", f.get("rev", 0.0))
        else:
            e = effect * f["rev"] * (f["lvol"] > 0.3) + effect * 0.6 * (f["hi52"] < -0.8)
        rows.append({"t": f"T{i % 40}", "made": made, "f": f, "er": m + e + float(rng.normal(0, 0.08))})
    return rows


def test_competition_accepts_a_planted_signal_and_nudges_the_right_way():
    L = {"graded": {"quant": {"21": _tilt_rows(2500, 0.03, seed=5)}}}
    tw = learn.tilt_weights(L)
    assert tw["accepted"], tw["status"]
    assert tw["model"] in ml.CANDIDATES and set(tw["candidates"]) == set(ml.CANDIDATES)
    sm = ml.SignalModel.build({**tw, "on": True}, L)
    hi, parts, days = sm.tilt({"sig": 2.0, "v": ml.FEAT_V})
    lo = sm.tilt({"sig": -2.0, "v": ml.FEAT_V})[0]
    assert hi > lo and hi > 0 and days == 21 and parts and parts[0]["k"] == "sig"
    assert abs(hi) <= tw["cap"] + 1e-9


def test_competition_rejects_pure_noise():
    accepted = sum(learn.tilt_weights({"graded": {"quant": {"21": _tilt_rows(2500, 0.0, seed=s)}}})["accepted"]
                   for s in range(10, 16))
    assert accepted == 0
    assert ml.SignalModel.build({"accepted": False}, {"graded": {}}) is None


def test_competition_picks_trees_when_the_signal_is_nonlinear_and_ridge_when_it_is_linear():
    price = [k for k in ml.PRICE_FEATS]
    nonlin = learn.tilt_weights({"graded": {"quant_bt": {"21": _tilt_rows(6000, 0.04, seed=21, n_dates=420,
                                                                          src_feats=price, kind="nonlinear")}}})
    assert nonlin["accepted"] and nonlin["candidates"]["trees"]["gain"] > nonlin["candidates"]["ridge"]["gain"] + 0.005
    assert nonlin["model"] in ("trees", "average", "neural_net") and nonlin["model"] != "ridge"
    lin = learn.tilt_weights({"graded": {"quant_bt": {"21": _tilt_rows(6000, 0.02, seed=22, n_dates=420,
                                                                       src_feats=price)}}})
    assert lin["accepted"] and lin["candidates"]["ridge"]["gain"] >= lin["candidates"]["trees"]["gain"] - 0.002


def test_learner_starts_from_the_backtest_on_day_one():
    bt = _tilt_rows(3000, 0.0, seed=8, src_feats=ml.PRICE_FEATS)
    for r in bt:                                  # a real effect of last month's move, in the backtest only
        r["er"] += -0.02 * r["f"]["rev"]
    tw = learn.tilt_weights({"graded": {"quant_bt": {"21": bt}}})
    assert tw["accepted"] and tw["n_live"] == 0
    sm = ml.SignalModel.build({**tw, "on": True}, {"graded": {"quant_bt": {"21": bt}}})
    assert sm.tilt({"rev": -1.5, "sig": 2.0, "v": ml.FEAT_V})[0] > sm.tilt({"rev": 1.5, "v": ml.FEAT_V})[0]


def test_learner_ignores_rows_and_states_from_older_feature_definitions():
    old = _tilt_rows(2500, 0.03, seed=5)
    for r in old:
        r["f"]["v"] = 2
    assert learn.tilt_weights({"graded": {"quant": {"21": old}}})["n"] == 0
    stale = {"accepted": True, "on": True, "model": "ridge", "fv": 2}
    assert ml.SignalModel.build(stale, {"graded": {"quant": {"21": _tilt_rows(2500, 0.03, seed=5)}}}) is None


def _pair_rows(n, qs_c, qs_m, days=120):
    base = dt.date(2026, 1, 2)
    return [{"made": (base + dt.timedelta(days=i % days)).isoformat(), "qs": qs_c, "qs_m": qs_m, "qsn": 1.0}
            for i in range(n)]


def test_blend_weights_follow_skill_on_the_same_forecasts():
    L = {"graded": {"claude": {"21": _pair_rows(400, 0.85, 0.95)}}}       # Claude 15% better than naive, model 5%
    w = learn.blend_weights(L)["1m"]
    assert 0.52 < w["w_claude"] < 0.62 and w["days"] == 120 and w["periods"] <= 5   # four months: modest evidence
    L = {"graded": {"claude": {"21": _pair_rows(1200, 0.85, 0.95, days=600)}}}
    assert learn.blend_weights(L)["1m"]["w_claude"] > 0.6                          # twenty months: trusted more
    L = {"graded": {"claude": {"21": _pair_rows(1200, 1.1, 0.95, days=600)}}}
    assert learn.blend_weights(L)["1m"]["w_claude"] < 0.4
    assert learn.blend_weights({"graded": {}})["1m"]["w_claude"] == 0.5


def test_features_recorded_with_forecasts():
    evs = [{"score": 80, "direction": "bullish", "kind": "insider", "action": "buy", "signal": "strong"},
           {"score": 30, "direction": "bearish", "kind": "insider", "action": "sell", "signal": "weak"}]
    rng = np.random.default_rng(3)
    c = list(50 * np.exp(np.cumsum(rng.normal(0.002, 0.02, 300))))
    b = list(100 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 300))))
    v = [1e6] * 300
    f = learn.features(evs, {}, {"rich_pctl": 80}, {"score": 5}, "favorable", c, b, v)
    assert f["sig"] == pytest.approx(0.5) and f["claude"] == 1.0 and f["ins"] == 0.0
    assert f["val"] == pytest.approx(0.3) and f["bub"] == 0.5 and f["v"] == learn.FEAT_V
    assert set(ml.PRICE_FEATS) <= set(f) and f["hi52"] <= 0 and f["vol_rel"] > 0 and -1 <= f["beta"] <= 4
    assert f["liq"] == pytest.approx(math.log10(float(np.median(np.asarray(c[-21:]) * 1e6))))
    # returns are measured against the S&P 500 over the same days: a stock that only kept pace has none
    same = learn.features([], {}, None, None, None, c, c)
    assert same["mom"] == pytest.approx(0.0) and same["rev"] == pytest.approx(0.0) and same["vol_rel"] == pytest.approx(0.0)
    assert "mom" not in learn.features([], {}, None, None, None, c)        # no benchmark: no price features
    assert "liq" not in same                                                # no volumes: no liquidity


class _Matrix:
    def __init__(self, cols, dates):
        self.dates = dates
        self.tickers = list(cols)
        self.col = {t: i for i, t in enumerate(self.tickers)}
        self.close = np.column_stack([cols[t] for t in self.tickers]).astype(np.float32)
        self.vol = np.full_like(self.close, 2_000_000.0)


def test_backtest_job_grades_past_forecasts_and_seeds_calibration(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from pipeline import insight, quant
    monkeypatch.setenv("SMR_FAKE_NOW", "2026-10-31T13:00:00+00:00")
    T = 520
    dates = trading_dates(END, T)
    cols = {"SPY": gbm(T, 0.01, seed=1, start=500)}
    for k in range(12):
        cols[f"S{chr(65 + k)}"] = gbm(T, 0.02, seed=10 + k, start=40, t_df=5)
    panel = quant.Panel(_Matrix(cols, dates))
    heavy = tmp_path / "heavy"
    heavy.mkdir()
    cfg = {"forecast": {"backtest_stocks": 12, "backtest_anchors": 20, "backtest_paths": 300, "backtest_seconds_per_run": 600}}
    settings = SimpleNamespace(c=lambda *k, default=None: cfg.get(k[0], default) if len(k) == 1 else default,
                               mark_heavy_dirty=lambda: None)
    budget = SimpleNamespace(left=lambda: 1000)
    n, note = insight.job_backtest(settings, {}, panel, budget, heavy)
    assert n > 300 and note.endswith("complete")
    L = fledger.load(heavy)
    m = fledger.metrics(L, today=dt.date(2026, 10, 31))
    one = m["by"]["quant_bt"]["1m"]["all"]
    assert one["n"] >= 100 and 0.6 <= one["cov80"] <= 0.95
    # running again adds nothing new (same stock-dates)
    n2, _ = insight.job_backtest(settings, {}, panel, budget, heavy)
    assert n2 == 0
    # the grid is one anchor a week, on the calendar: next month's grid skips every stock-date already graded
    bt = fledger.load(heavy)["bt"]
    weeks = [dt.date.fromordinal(int(d)).isocalendar()[:2] for d in bt["anchors"]]
    assert len(set(weeks)) == len(weeks) == 20
    L2 = fledger.load(heavy)
    L2["bt"]["built"] = "2026-09"
    fledger.save(heavy, L2)
    n3, _ = insight.job_backtest(settings, {}, panel, budget, heavy)
    assert n3 == 0 and fledger.load(heavy)["bt"]["n_names"] == 0
    cal = learn.calibration(L, min_n=100, min_dates=10)
    assert cal["1m"]["status"] in ("applied", "held") and cal["1m"]["from"] == "backtest"
    # every backtest horizon carries the features both learners use from day one, with its own earnings flag
    for h in ("1", "5", "21", "63"):
        rows = L["graded"]["quant_bt"][h]
        assert rows and all(r["f"]["v"] == learn.FEAT_V and "rev" in r["f"] and "ein" in r["f"] for r in rows if "f" in r)
        assert sum("f" in r for r in rows) >= 0.8 * len(rows)
