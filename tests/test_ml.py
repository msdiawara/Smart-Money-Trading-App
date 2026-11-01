"""The machine-learning layer (pipeline/ml.py): the range learner, the deployed models, and how the pipeline uses
them. The algorithm competition itself is tested in test_forecast.py."""
import datetime as dt
import json
from types import SimpleNamespace

import numpy as np
import pytest

from pipeline import fledger, insight, learn, ml
from pipeline import forecast as F


def _range_rows(conditional: bool, seed: int = 3, n_dates: int = 400, per: int = 40) -> dict:
    """1-month rows whose true spread depends on recent-vs-usual volatility (or doesn't)."""
    rng = np.random.default_rng(seed)
    base = dt.date(2024, 1, 2)
    rows = []
    for d in range(n_dates):
        made = (base + dt.timedelta(days=d * 7 // 5)).isoformat()
        for i in range(per):
            f = {k: float(rng.normal()) for k in ml.PRICE_FEATS}
            f.update(v=ml.FEAT_V, ein=int(rng.random() < 0.3))
            k = (0.7 if f["vol_chg"] > 0.5 else 1.3) if conditional else 1.0
            er = float(rng.normal(0, 0.06 / 1.2816 * k))
            rows.append({"t": f"S{i}", "made": made, "f": f, "er": er, "w10": 0.06, "w90": 0.06,
                         "in80r": int(abs(er) <= 0.06)})
    return {"graded": {"quant": {"21": rows}}}


def test_range_learner_is_used_only_where_it_beats_one_stretch_for_all():
    flat = _range_rows(False)
    rl = ml.range_learner(flat, learn.calibration(flat))
    assert rl["1m"]["status"] == "held" and ml.RangeModel.build(rl, {}, flat) is None
    cond = _range_rows(True)
    cal = learn.calibration(cond)
    rl = ml.range_learner(cond, cal)
    assert rl["1m"]["status"] == "applied" and rl["1m"]["holdout_gain_vs_single"] > 0.005
    assert rl["1d"]["status"] == "learning"                     # no 1-day rows at all
    rm = ml.RangeModel.build(rl, cal, cond)
    fc = {"horizons": {"1m": {"date": "2026-01-30"}}, "events": []}
    spike = rm.calib_for({"vol_chg": 1.5, "v": ml.FEAT_V}, fc, cal)["1m"]
    calm = rm.calib_for({"vol_chg": -1.0, "v": ml.FEAT_V}, fc, cal)["1m"]
    assert spike["scale"] < 0.9 < 1.1 < calm["scale"] and spike["status"] == "applied" and spike.get("cond")
    assert rm.calib_for({}, fc, cal)["1w"] == cal.get("1w")      # other horizons keep the single stretch


def test_range_learner_waits_for_enough_independent_months():
    small = _range_rows(True, n_dates=60)
    rl = ml.range_learner(small, learn.calibration(small, min_n=100, min_dates=10))
    assert rl["1m"]["status"] == "learning"


def test_forecast_with_learning_rebuilds_with_the_stocks_own_ranges():
    rng = np.random.default_rng(1)
    dates = [d.toordinal() for d in F.trading_days_after(dt.date(2024, 1, 2), 400)]
    closes = list(50 * np.exp(np.cumsum(rng.normal(0, 0.02, 400))))
    calls = []

    def build(calib, tilt_ret, tilt_days, tilt_parts):
        calls.append(calib)
        return F.build("ACME", dates, closes, calib=calib, tilt_ret=tilt_ret, tilt_days=tilt_days,
                       tilt_parts=tilt_parts, paths=300)

    class _RM:
        def calib_for(self, feats, fc, calib):
            assert fc["horizons"]["1m"]["date"]                    # sees the first build's dates
            return {**calib, "1m": {"status": "applied", "scale": 1.5, "shift": 0.0}}

    class _SM:
        def tilt(self, feats):
            return 0.01, [{"k": "rev", "label": "x", "h_ret": 0.01}], 21

    base, _ = ml.forecast_with_learning(build, {}, {}, None, None)
    fc, tr = ml.forecast_with_learning(build, {}, {}, _SM(), _RM())
    assert len(calls) == 3 and calls[-1]["1m"]["scale"] == 1.5 and tr == 0.01
    w = lambda f: f["horizons"]["1m"]["lq"][5] - f["horizons"]["1m"]["lq"][1]
    assert w(fc) > 1.3 * w(base) and fc["tilt"]["ret"] == 0.01


def test_every_stock_with_a_forecast_is_listed(tmp_path):
    out = tmp_path / "out"
    (tmp_path / "cache").mkdir()
    rng = np.random.default_rng(2)
    dates = [d.toordinal() for d in F.trading_days_after(dt.date(2024, 1, 2), 300)]
    fc = F.build("ACME", dates, list(40 * np.exp(np.cumsum(rng.normal(0, 0.02, 300)))), paths=300)
    from pipeline.util import write_json
    write_json(tmp_path / "cache" / insight.FC_CACHE, {"by_ticker": {"ACME": fc}})
    settings = SimpleNamespace(out_dir=out, cache_dir=tmp_path / "cache",
                               c=lambda *k, default=None: default)
    store = SimpleNamespace(series=lambda t: None)
    n = insight.publish_stocks(settings, {"quotes": {"ACME": {"name": "Acme Corp", "price": 41.0}}},
                               {"ACME": {"name": "Acme Corp", "val": {"label": "Cheap"}, "conv": {"rating": "positive"}}},
                               None, store)
    idx = json.loads((out / "forecasts.json").read_text())
    assert n == 1 and idx["n"] == 1
    row = idx["stocks"][0]
    assert row["t"] == "ACME" and row["val"] == "Cheap" and row["rating"] == "positive"
    assert row["m1"]["lo"] < row["m1"]["mid"] < row["m1"]["hi"] and 0 <= row["m1"]["up"] <= 1
    assert row["close"] == fc["anchor"] and row["made"] == fc["made"] and idx["enabled"] is True
    # a page that can't be rebuilt next run keeps its last good version, and its line in the list
    real = insight._blend_view
    insight._blend_view = lambda *a, **k: 1 / 0
    try:
        insight.publish_stocks(settings, {"quotes": {}}, {"ACME": {"name": "Acme Corp"}}, None, store)
    finally:
        insight._blend_view = real
    idx2 = json.loads((out / "forecasts.json").read_text())
    assert [r["t"] for r in idx2["stocks"]] == ["ACME"] and idx2["stocks"][0]["m1"] == row["m1"]
    # forecasts switched off: the list says so (the app shows that instead of "no forecasts yet")
    off = SimpleNamespace(out_dir=out, cache_dir=tmp_path / "cache",
                          c=lambda *k, default=None: {"enabled": False} if k[0] == "forecast" else default)
    insight.publish_stocks(off, {"quotes": {}}, {"ACME": {"name": "Acme Corp"}}, None, store)
    idx3 = json.loads((out / "forecasts.json").read_text())
    assert idx3["enabled"] is False and idx3["stocks"] == []


def test_the_self_check_watches_the_all_forecasts_list(tmp_path):
    from pipeline import selfcheck
    from pipeline.util import iso, now_utc
    a = selfcheck.Audit()
    selfcheck._check_index(a, tmp_path, 5)
    assert a.checks[-1]["status"] == "fail"                                  # missing
    (tmp_path / "forecasts.json").write_text(json.dumps({"generated_at": iso(now_utc()), "stocks": [{"t": "A"}] * 5}))
    selfcheck._check_index(a, tmp_path, 5)
    assert a.checks[-1]["status"] == "pass"
    selfcheck._check_index(a, tmp_path, 40)
    assert a.checks[-1]["status"] == "warn"                                  # pages and list disagree
    (tmp_path / "forecasts.json").write_text(json.dumps({"generated_at": "2026-01-01T00:00:00Z", "stocks": []}))
    selfcheck._check_index(a, tmp_path, 0)
    assert a.checks[-1]["status"] == "fail"                                  # not rewritten this run


def test_weekly_competition_is_time_boxed_and_postponed_not_half_applied(tmp_path, monkeypatch):
    monkeypatch.setenv("SMR_FAKE_NOW", "2026-10-30T20:30:00+00:00")
    heavy = tmp_path / "heavy"
    heavy.mkdir()
    rng = np.random.default_rng(4)
    rows = []
    base = dt.date(2025, 1, 2)
    for d in range(300):
        made = (base + dt.timedelta(days=d * 7 // 5)).isoformat()
        for i in range(10):
            f = {k: float(rng.normal()) for k in ml.PRICE_FEATS}
            f["v"] = ml.FEAT_V
            rows.append({"t": f"S{i}", "made": made, "f": f, "er": float(rng.normal(0, 0.08)), "w10": 0.1, "w90": 0.1})
    L = fledger.load(heavy)
    L["graded"]["quant_bt"] = {"21": rows}
    fledger.save(heavy, L)
    settings = SimpleNamespace(c=lambda *k, default=None: default, cache_dir=tmp_path, mark_heavy_dirty=lambda: None)
    state = {"learn": {"tilt": {"fv": ml.FEAT_V, "on": False, "status": "old"}, "tilt_at": "2026-10-01T00:00:00Z"}}
    calls = []
    real = ml.compete

    def slow(L, **kw):
        calls.append(kw.get("seconds"))
        return {"postponed": True, "status": "postponed"}
    monkeypatch.setattr(ml, "compete", slow)
    get = SimpleNamespace(series=lambda t: None)
    insight.job_grade_learn(settings, state, None, get, heavy, SimpleNamespace(left=lambda: 400))
    assert calls and calls[0] <= 300 and state["learn"]["tilt"]["status"] == "old"      # kept; retried next run
    assert state["learn"]["tilt_at"] == "2026-10-01T00:00:00Z"
    calls.clear()
    insight.job_grade_learn(settings, state, None, get, heavy, SimpleNamespace(left=lambda: 150))
    assert not calls                                                      # not enough time left: not even started
    monkeypatch.setattr(ml, "compete", real)
    insight.job_grade_learn(settings, state, None, get, heavy, SimpleNamespace(left=lambda: 2000))
    assert state["learn"]["tilt"]["status"].startswith("learning") or "candidates" in state["learn"]["tilt"]
    assert state["learn"]["tilt_at"] != "2026-10-01T00:00:00Z"


def test_price_features_need_enough_clean_history():
    assert ml.price_features([1.0] * 30, [1.0] * 30) == {"v": ml.FEAT_V}
    c = list(np.linspace(10, 20, 100))
    c[-3] = float("nan")
    assert "rev" not in ml.price_features(c, [1.0] * 100)
    assert ml.price_features(list(np.linspace(10, 20, 100)), [1.0] * 99) == {"v": ml.FEAT_V}   # misaligned
    f = ml.price_features(list(np.linspace(10, 20, 100)), list(np.linspace(5, 6, 100)))
    assert "rev" in f and "mom" not in f                     # under a year: no 12-month features


def test_earnings_flag_follows_the_forecasts_own_events():
    fc = {"horizons": {"1w": {"date": "2026-11-06"}, "1m": {"date": "2026-11-27"}},
          "events": [{"kind": "fomc", "date": "2026-11-04"}, {"kind": "earnings", "date": "2026-11-20"}]}
    assert ml.earn_in(fc, "1w") == 0 and ml.earn_in(fc, "1m") == 1
    assert ml.row_features({"rev": 0.1}, fc, "1m") == {"rev": 0.1, "ein": 1}


@pytest.mark.parametrize("model", ["ridge", "trees", "neural_net", "average"])
def test_every_algorithm_fits_and_predicts_with_missing_values(model):
    rng = np.random.default_rng(5)
    X = rng.normal(size=(800, len(ml.FEATS)))
    X[:, ml.FEATS.index("val")] = np.nan                     # a signal nobody has
    X[rng.random(X.shape) < 0.2] = np.nan
    y = 0.02 * np.nan_to_num(X[:, 1]) + rng.normal(0, 0.05, 800)
    f = ml.fit(model, X, y, periods=10)
    p = f.predict(X[:5])
    assert p.shape == (5,) and np.all(np.isfinite(p))
    assert np.isfinite(f.predict(np.full((1, len(ml.FEATS)), np.nan))).all()


def test_stored_month_results_are_reused_only_while_the_data_is_unchanged():
    from tests.test_forecast import _tilt_rows
    rows = _tilt_rows(2500, 0.03, seed=5)
    L = {"graded": {"quant": {"21": rows}}}
    first = ml.compete(L)
    assert L["ml_cache"]["folds"] and ml.compete(L)["candidates"] == first["candidates"]
    calls = []
    real = ml.fit

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)
    import pipeline.ml as m
    m.fit, saved = counting, m.fit
    try:
        ml.compete(L)
        assert not calls                                  # nothing changed: every month reused
        last = sorted({r["made"][:7] for r in rows})[-2]  # the newest month is still being graded: not tested
        k = max(i for i, r in enumerate(rows) if r["made"][:7] == last)
        rows[k]["er"] += 0.5                              # a test outcome changes (same counts, same dates)
        rows[k]["er0"] = rows[k]["er"]
        ml.compete(L)
        assert len(calls) == 1                            # only that month is refitted
        rows[0]["f"]["mom"] = 9.0                         # a feature in the oldest training rows changes
        calls.clear()
        ml.compete(L)
        assert len(calls) >= 2                            # every month trained on it is refitted
    finally:
        m.fit = saved


# ---------------------------------------------------------------- fifth review: the statistics reviewer's findings

def test_the_target_is_how_a_stock_did_against_the_others():
    """The S&P 500's own rows aren't trained on; each stock's error has its beta times the market's surprise that
    day taken out, then the day's average."""
    f = lambda b: {"v": ml.FEAT_V, "beta": b, "mom": 0.1}            # noqa: E731
    rows = [{"t": "SPY", "made": "2026-01-05", "er": 0.04, "f": f(1.0)},
            {"t": "AAA", "made": "2026-01-05", "er": 0.08, "f": f(2.0)},
            {"t": "BBB", "made": "2026-01-05", "er": 0.00, "f": f(0.5)},
            {"t": "AAA", "made": "2026-01-06", "er": 0.01, "f": f(2.0)},
            {"t": "BBB", "made": "2026-01-06", "er": -0.01, "f": f(0.5)}]
    L = {"graded": {"quant_bt": {"21": rows}}}
    got = ml.rows_for(L, 21)
    assert [r["t"] for r in got] == ["AAA", "BBB", "AAA", "BBB"]
    _, _, y = ml.targets(got, L, 21)
    # day 1: AAA 0.08 − 2×0.04 = 0, BBB 0 − 0.5×0.04 = −0.02, so +0.01 / −0.01; day 2 has no index row
    assert np.allclose(y, [0.01, -0.01, 0.01, -0.01])


def test_nudges_are_centred_on_the_days_stocks():
    from tests.test_forecast import _tilt_rows
    rows = _tilt_rows(2500, 0.03, seed=5)
    L = {"graded": {"quant": {"21": rows}}}
    res = ml.compete(L)
    assert res["accepted"], res["status"]
    sm = ml.SignalModel.build({**res, "on": True}, L)
    day = [{**r["f"], "sig": r["f"]["sig"] + 1.0} for r in rows[-300:]]   # a day when every score is high
    c = sm.set_day(day)
    raw = sm.fitted.predict(np.vstack([sm._row(f) for f in day]))
    assert c == pytest.approx(float(raw.mean())) and c > 0
    tilts = np.asarray([sm.tilt(f)[0] for f in day])
    assert abs(tilts.mean()) < 0.25 * sm.cap and (tilts > 0).any() and (tilts < 0).any()


def test_the_bar_is_stricter_with_few_test_months():
    from scipy.stats import t as student
    from tests.test_forecast import _tilt_rows
    res = ml.compete({"graded": {"quant": {"21": _tilt_rows(2500, 0.0, seed=4)}}})
    k = res["test_months"]
    assert k >= 6 and res["t_needed"] == pytest.approx(max(2.5, float(student.ppf(0.995, k - 1))), abs=0.01)
    assert res["t_needed"] > 2.5 and res["last_month"] and not res["accepted"]


def test_period_counts_are_steady_and_skip_gaps():
    d0 = dt.date(2025, 1, 1)
    days = [(d0 + dt.timedelta(days=i)).isoformat() for i in range(400)]
    assert {ml.periods(days[i:i + 90], 21) for i in range(300)} == {3}   # calendar blocks flip between 3 and 4
    assert ml.periods([days[0], days[1], days[300]], 21) == 2            # ten idle months aren't evidence
    assert ml.periods([], 21) == 0 and ml.periods(days[:200:5], 63) == 3  # 3-month blocks: 89 days each


def test_compaction_always_keeps_the_index():
    day = "2026-09-01"
    rows = [{"t": f"S{i:03d}", "made": day, "er": 0.0} for i in range(400)] + [{"t": "SPY", "made": day, "er": 0.0}]
    L = {"graded": {"quant_bt": {"21": rows}, "quant": {"21": [dict(r) for r in rows]}}}
    fledger.compact(L, dt.date(2026, 10, 31))
    for src, k in (("quant_bt", 150), ("quant", 50)):
        kept = L["graded"][src]["21"]
        assert len(kept) == k + 1 and sum(r["t"] == "SPY" for r in kept) == 1


def test_the_range_constant_comes_from_forecasts_the_trees_did_not_see():
    rng = np.random.default_rng(11)

    def draw(n, start):
        X = rng.normal(size=(n, len(ml.RANGE_FEATS)))
        need = np.abs(rng.normal(0, 1, n)) * np.exp(0.5 * X[:, 0])        # the spread depends on one signal
        made = np.asarray([(start + dt.timedelta(days=int(i * 600 / n))).isoformat() for i in range(n)])
        return X, need, made
    X, need, made = draw(12000, dt.date(2023, 1, 2))
    model = ml._fit_q80(X, need, made, 21)
    X2, need2, _ = draw(20000, dt.date(2025, 1, 2))
    assert 0.77 <= float(np.mean(need2 <= model.predict(X2))) <= 0.83      # 80% of new ones inside


def _fc_env(tmp_path, monkeypatch):
    """One stock, one run of the daily forecast job, with the signal learner switched on."""
    from pipeline.util import now_et, read_json
    monkeypatch.setenv("SMR_FAKE_NOW", "2026-10-30T20:30:00+00:00")
    anchor = F.last_complete_session(now_et())
    days = [d for d in F.trading_days_after(anchor - dt.timedelta(days=440), 300) if d <= anchor]
    rng = np.random.default_rng(7)
    ser = {t: {"dates": [d.toordinal() for d in days], "v": [1e6] * len(days),
               "c": list(40 * np.exp(np.cumsum(rng.normal(0, 0.02, len(days)))))} for t in ("ACME", "SPY")}
    (tmp_path / "cache").mkdir()
    (tmp_path / "heavy").mkdir()
    settings = SimpleNamespace(c=lambda *k, default=None: {"paths": 300} if k[0] == "forecast" else default,
                               cache_dir=tmp_path / "cache", mark_heavy_dirty=lambda: None)
    tilt = {"on": True, "accepted": True, "model": "ridge", "fv": ml.FEAT_V, "h": 21, "cap": 0.02, "status": "applied"}
    state = {"quotes": {"ACME": {"price": 41.0, "beta": 1.0}}, "events": [],
             "learn": {"tilt": tilt, "tilt_at": "2026-10-28T00:00:00Z", "range": {}}}
    store = SimpleNamespace(series=lambda t: ser.get(t))

    def run():
        return insight.job_forecasts(settings, state, store, SimpleNamespace(left=lambda: 600), {"ACME": {}},
                                     ["ACME"], tmp_path / "heavy")

    def fc():
        return read_json(tmp_path / "cache" / insight.FC_CACHE)["by_ticker"]["ACME"]
    return run, fc


class _Nudge:
    def __init__(self, fail=False):
        self.fail = fail

    def set_day(self, feats):
        return 0.0

    def tilt(self, feats):
        if self.fail:
            raise ValueError("a bad row")
        return 0.015, [{"k": "rev", "label": "Last month's move vs the S&P 500", "h_ret": 0.015}], 21


def test_a_learner_that_cant_be_rebuilt_is_retried_on_the_next_run(tmp_path, monkeypatch):
    run, fc = _fc_env(tmp_path, monkeypatch)

    def boom(cls, *a, **k):
        raise MemoryError("couldn't refit")
    monkeypatch.setattr(ml.SignalModel, "build", classmethod(boom))
    n, note = run()                                                        # ACME and the S&P 500
    assert n == 2 and "made without the signal" in note
    assert fc()["cv"].endswith("!") and not fc()["horizons"]["1m"].get("tilt")
    monkeypatch.setattr(ml.SignalModel, "build", classmethod(lambda cls, *a, **k: _Nudge()))
    n, note = run()
    assert n == 2 and "made without" not in note                          # rebuilt, now with the learner
    assert not fc()["cv"].endswith("!") and fc()["horizons"]["1m"]["tilt"] == pytest.approx(0.015)
    assert run()[0] == 0                                                   # and then left alone


def test_a_learner_error_on_one_stock_falls_back_to_the_plain_forecast(tmp_path, monkeypatch):
    run, fc = _fc_env(tmp_path, monkeypatch)
    monkeypatch.setattr(ml.SignalModel, "build", classmethod(lambda cls, *a, **k: _Nudge(fail=True)))
    n, note = run()
    assert n == 2 and "1 made without" in note and fc()["cv"].endswith("!")


# ---------------------------------------------------------------- the verification review's findings

def test_a_day_without_the_index_row_still_has_the_market_taken_out():
    rng = np.random.default_rng(3)
    betas = rng.uniform(0.5, 2.0, 40)
    rows = []
    for d, mkt in (("2026-01-05", 0.06), ("2026-01-06", -0.04)):
        for i, b in enumerate(betas):
            rows.append({"t": f"S{i}", "made": d, "er": float(b * mkt + rng.normal(0, 0.002)),
                         "f": {"v": ml.FEAT_V, "beta": float(b)}})
    rows.append({"t": "SPY", "made": "2026-01-05", "er": 0.06, "f": {"v": ml.FEAT_V, "beta": 1.0}})
    L = {"graded": {"quant": {"21": rows}}}
    got = ml.rows_for(L, 21)
    _, _, y = ml.targets(got, L, 21)
    made = np.asarray([r["made"] for r in got])
    assert np.abs(y[made == "2026-01-05"]).max() < 0.01        # with the index's own record
    assert np.abs(y[made == "2026-01-06"]).max() < 0.01        # without it: estimated from that day's stocks
    assert np.corrcoef(y[made == "2026-01-06"], betas)[0, 1] ** 2 < 0.5   # no beta left to learn


def test_a_stock_day_in_both_the_live_record_and_the_backtest_counts_once():
    f = {"v": ml.FEAT_V, "beta": 1.0}
    live = [{"t": "AAA", "made": "2026-03-02", "er": 0.01, "f": f}]
    bt = [{"t": "AAA", "made": "2026-03-02", "er": 0.01, "f": f}, {"t": "BBB", "made": "2026-03-02", "er": 0.0, "f": f}]
    got = ml.rows_for({"graded": {"quant": {"21": live}, "quant_bt": {"21": bt}}}, 21)
    assert sorted((r["t"], r is live[0]) for r in got) == [("AAA", True), ("BBB", False)]


def test_a_month_still_being_graded_is_not_a_test_month():
    from tests.test_forecast import _tilt_rows
    rows = _tilt_rows(2500, 0.0, seed=4)
    first = ml.compete({"graded": {"quant": {"21": rows}}})
    nxt = dt.date.fromisoformat(rows[-1]["made"][:8] + "01") + dt.timedelta(days=32)
    stub = [{**rows[i], "t": f"N{i}", "made": nxt.replace(day=1).isoformat()} for i in range(25)]
    second = ml.compete({"graded": {"quant": {"21": rows + stub}}})
    # a graded forecast from the 1st means every forecast of the month before is graded: that month now counts,
    # the one-day stub doesn't
    assert second["last_month"] < nxt.isoformat()[:7] and second["test_months"] == first["test_months"] + 1
    more = stub + [{**rows[i], "t": f"M{i}", "made": nxt.replace(day=2).isoformat()} for i in range(25)]
    third = ml.compete({"graded": {"quant": {"21": rows + more}}})
    assert third["last_month"] == second["last_month"]
    g = learn.tilt_gate({"passes": 1, "last_month": second["last_month"]}, {**third, "accepted": True})
    assert g["passes"] == 1 and not g["on"]                     # more days of a new month aren't a completed month


def test_the_row_cap_drops_whole_months():
    rows = []
    for mo in range(1, 8):
        for d in range(1, 21):
            rows += [{"t": f"S{i}", "made": f"2026-{mo:02d}-{d:02d}", "er": 0.0} for i in range(10)]
    L = {"graded": {"claude": {"5": list(rows)}}}
    saved = fledger.KEEP_GRADED
    fledger.KEEP_GRADED = 1000                                  # 1,400 rows, 200 a month
    try:
        fledger.compact(L, dt.date(2026, 8, 15))
    finally:
        fledger.KEEP_GRADED = saved
    kept = L["graded"]["claude"]["5"]
    assert len(kept) == 1000 and min(r["made"] for r in kept) == "2026-03-01"


def test_the_coverage_warning_rests_on_the_last_fifteen_months_only():
    def rows(start, n, hit):
        d0 = dt.date.fromisoformat(start)
        return [{"t": f"S{i % 30}", "made": (d0 + dt.timedelta(days=i // 30)).isoformat(), "in80": hit, "er": 0.0}
                for i in range(n)]
    old, new = rows("2023-01-02", 6000, 0), rows("2026-01-05", 6000, 1)
    L = {"pending": {}, "graded": {"quant": {"21": old + new}}, "agg": {"quant": {"21": {}}}}
    m = fledger.metrics(L, today=dt.date(2026, 12, 31))           # (nothing in the 120-day "recent" window)
    assert m["by"]["quant"]["1m"]["all"]["cov80_ci"]["p"] == 1.0     # the 2023 misses don't count


def test_the_index_gets_no_stock_versus_stock_nudge_and_goes_first(tmp_path, monkeypatch):
    run, fc = _fc_env(tmp_path, monkeypatch)
    seen = []

    class Spy(_Nudge):
        def tilt(self, feats):
            seen.append(feats.get("rev"))
            return super().tilt(feats)
    monkeypatch.setattr(ml.SignalModel, "build", classmethod(lambda cls, *a, **k: Spy()))
    real = insight.forecast.build
    order = []

    def spy_build(t, *a, **k):
        order.append(t)
        return real(t, *a, **k)
    monkeypatch.setattr(insight.forecast, "build", spy_build)
    run()
    from pipeline.util import read_json
    by = read_json(tmp_path / "cache" / insight.FC_CACHE)["by_ticker"]
    assert order[0] == "SPY" and len(seen) == 1                   # only ACME was asked for a nudge
    assert not by["SPY"]["horizons"]["1m"].get("tilt") and by["ACME"]["horizons"]["1m"]["tilt"] == pytest.approx(0.015)


def test_forecasts_made_without_a_learner_are_redone_only_once_it_can_be_built(tmp_path, monkeypatch):
    run, fc = _fc_env(tmp_path, monkeypatch)
    monkeypatch.setattr(ml.SignalModel, "build", classmethod(lambda cls, *a, **k: None))   # on, but no record here
    n, note = run()
    assert n == 2 and fc()["cv"].endswith("!") and "made without the signal" in note
    assert run()[0] == 0                                           # still can't be built: nothing redone
    monkeypatch.setattr(ml.SignalModel, "build", classmethod(lambda cls, *a, **k: _Nudge()))
    assert run()[0] == 2 and not fc()["cv"].endswith("!")


def test_the_backtest_queue_keeps_its_dates_when_days_are_added_to_the_history(tmp_path, monkeypatch):
    from tests.test_forecast import END, _Matrix, gbm, trading_dates
    from pipeline import quant
    monkeypatch.setenv("SMR_FAKE_NOW", "2026-10-31T13:00:00+00:00")
    T = 520
    dates = trading_dates(END, T)
    cols = {"SPY": gbm(T, 0.01, seed=1, start=500), "SA": gbm(T, 0.02, seed=10, start=40)}
    heavy = tmp_path / "heavy"
    heavy.mkdir()
    short = quant.Panel(_Matrix({k: v[1:] for k, v in cols.items()}, dates[1:]))
    want = [int(short.dates[i]) for i in (300, 350, 400)]       # queued while the history was one day shorter
    L = fledger.load(heavy)
    L["bt"] = {"v": 3, "built": "2026-10", "anchors": want, "todo": [["SA", d] for d in want]}
    fledger.save(heavy, L)
    cfg = {"forecast": {"backtest_paths": 200}}
    settings = SimpleNamespace(c=lambda *k, default=None: cfg.get(k[0], default) if len(k) == 1 else default,
                               mark_heavy_dirty=lambda: None)
    insight.job_backtest(settings, {}, quant.Panel(_Matrix(cols, dates)), SimpleNamespace(left=lambda: 1000), heavy)
    made = {r["made"] for r in fledger.load(heavy)["graded"]["quant_bt"]["5"]}
    assert made == {dt.date.fromordinal(d).isoformat() for d in want}


def test_the_range_learner_holds_steady_and_starts_from_the_backtest():
    cond = _range_rows(True)
    cal = learn.calibration(cond)
    on = ml.range_learner(cond, cal)["1m"]
    assert on["status"] == "applied"
    # a later week where its edge has shrunk below the bar to switch on, but it's still no worse: it stays on
    weak = {"1m": {**on, "status": "applied"}}
    import pipeline.ml as m
    real = m._pinball3
    shrink = (1 - 0.002) / (1 - on["holdout_gain_vs_single"])          # its edge becomes 0.2%: under the 0.5% bar
    m._pinball3 = lambda er, w10, w90, scale, shift: real(er, w10, w90, scale, shift) * (
        1.0 if np.ndim(scale) == 0 else shrink)
    try:
        assert ml.range_learner(cond, cal, prev=weak)["1m"]["status"] == "applied"
        assert ml.range_learner(cond, cal)["1m"]["status"] == "held"          # not on before: needs the full bar
    finally:
        m._pinball3 = real
    # live forecasts too few for it, the backtest plenty: it learns from the backtest meanwhile
    bt = _range_rows(True)["graded"]["quant"]["21"]
    mixed = {"graded": {"quant": {"21": bt[-400:]}, "quant_bt": {"21": bt}}}
    rl = ml.range_learner(mixed, learn.calibration(mixed))["1m"]
    assert rl["from"] == "backtest" and rl["status"] == "applied"
    assert ml.RangeModel.build({"1m": rl}, {}, mixed) is not None


def test_calibration_stays_on_the_backtest_until_live_grades_span_enough_months():
    bt = _range_rows(False, n_dates=300)["graded"]["quant"]["21"]
    last = dt.date.fromisoformat(bt[-1]["made"])
    live = [{**r, "made": (last + dt.timedelta(days=1 + i // 40)).isoformat()} for i, r in enumerate(bt[:1000])]
    L = {"graded": {"quant": {"21": live}, "quant_bt": {"21": bt}}}       # 1,000 live grades from 25 days: 1 period
    c = learn.calibration(L)["1m"]
    assert c["from"] == "backtest" and c["status"] in ("applied", "held")
    assert ml.range_learner(L, learn.calibration(L))["1m"]["from"] == "backtest"
