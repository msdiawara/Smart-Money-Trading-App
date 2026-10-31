"""The predictability layer on synthetic markets: it must find a planted signal, must NOT claim skill on noise,
and every number it reports must be computable from data available at the time."""
import datetime as dt
import math
import types

import numpy as np
import pytest

from pipeline import ledger, market, quant


def make_panel(n_days=620, n_tickers=320, seed=7):
    rng = np.random.default_rng(seed)
    start = dt.date(2024, 1, 2)
    days, d = [], start
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d.toordinal())
        d += dt.timedelta(days=1)
    names = ["SPY"] + [f"T{i:03d}" for i in range(n_tickers)]
    rets = rng.normal(0.0003, 0.018, size=(n_days, len(names)))
    rets[:, 0] = rng.normal(0.0004, 0.009, size=n_days)
    close = (50 + 100 * rng.random(len(names))) * np.exp(np.cumsum(rets, axis=0))
    close[:, 0] = 400 * np.exp(np.cumsum(rets[:, 0]))
    vol = np.full_like(close, 1e6)
    m = types.SimpleNamespace(dates=days, close=close.astype(np.float32), vol=vol.astype(np.float32),
                              col={t: j for j, t in enumerate(names)})
    return quant.Panel(m), names


def synthetic_dera(panel, names, planted: bool, n=3200, seed=11):
    """Insider buys. If planted, CEO buys are much more likely when the stock goes on to beat SPY."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n):
        t = names[1 + rng.integers(0, len(names) - 1)]
        i = int(rng.integers(140, panel.T - 70))
        f = dt.date.fromordinal(int(panel.dates[i]))
        fx = quant.forward_excess(panel, t, i + 1)
        win = fx is not None and fx[1] > 0
        if planted:
            ceo = rng.random() < (0.75 if win else 0.25)
        else:
            ceo = rng.random() < 0.5
        rows.append({"acc": f"a{k}", "code": "P", "owner": f"{k % 900:010d}", "owner_name": "X",
                     "role": "Officer" if ceo else "Director", "title": "CEO" if ceo else "",
                     "issuer": "0", "ticker": t, "filed": f.isoformat(), "traded": f.isoformat(),
                     "value": float(rng.integers(20_000, 2_000_000)), "plan": False})
    return rows


def test_model_finds_a_planted_signal():
    panel, names = make_panel()
    samples = quant.insider_dataset(synthetic_dera(panel, names, planted=True), panel, {})
    assert len(samples) > 2500 and all(s["filed"] <= s2["filed"] for s, s2 in zip(samples, samples[1:]))
    summary, bundle = quant.train_model(samples)
    assert summary["status"] == "ok" and summary["has_skill"] and bundle is not None
    assert summary["auc"] > 0.6 and summary["auc_ci"][0] > 0.5
    assert summary["top_fifth"]["beat_rate"] > summary["bottom_fifth"]["beat_rate"]
    assert summary["test_period"][0] > summary["train_period"][1]          # tested only on later filings
    assert len(summary["calibration"]) == 5
    # live prediction for a CEO buy vs a director buy on the same stock/day
    x = dict(samples[-1]["x"])
    p_ceo = quant.predict(bundle, {**x, "max_rank": 5})
    p_dir = quant.predict(bundle, {**x, "max_rank": 3})
    assert 0 < p_dir < p_ceo < 1


def test_model_refuses_to_claim_skill_on_noise():
    panel, names = make_panel(seed=3)
    samples = quant.insider_dataset(synthetic_dera(panel, names, planted=False, seed=5), panel, {})
    summary, bundle = quant.train_model(samples)
    assert summary["status"] == "ok"
    assert not summary["has_skill"] and bundle is None and summary["predictability"] == "None detected"


def test_features_never_look_ahead():
    panel, names = make_panel(n_days=300, n_tickers=5)
    i = 200
    before = quant.price_features(panel, "T001", i)
    panel.close[i + 1:, :] *= 3.0                    # change the future
    after = quant.price_features(panel, "T001", i)
    assert before == after


def test_forward_excess_uses_same_window_for_stock_and_spy():
    panel, names = make_panel(n_days=300, n_tickers=3)
    j = 100
    r, ex = quant.forward_excess(panel, "T001", j)
    c, s = panel.c("T001"), panel.c("SPY")
    assert math.isclose(r, c[j + 63] / c[j] - 1, rel_tol=1e-6)
    assert math.isclose(ex, r - (s[j + 63] / s[j] - 1), rel_tol=1e-6)
    panel.close[j, panel.col["T001"]] = np.nan       # missing entry day → next valid day, SPY moves with it
    r2, ex2 = quant.forward_excess(panel, "T001", j)
    assert math.isclose(r2, c[j + 64] / c[j + 1] - 1, rel_tol=1e-6)


def test_base_rates_and_event_view_without_model():
    panel, names = make_panel()
    buys = quant.insider_dataset(synthetic_dera(panel, names, planted=False), panel, {})
    rates = quant.base_rates(buys, [], [])
    br = rates["insider_buy"]
    assert br["n"] > 2500 and br["lo"] < br["p"] < br["hi"]
    # outcomes in the same quarter share SPY's move (these synthetic stocks don't track SPY at all), so the honest
    # interval is far wider than a Wilson interval that treats every filing as independent
    wl, wh = quant.wilson(round(br["p"] * br["n"]), br["n"])
    assert br["hi"] - br["lo"] > 2 * (wh - wl) and br["periods"] >= 5
    e = {"id": "x", "kind": "insider", "action": "buy", "ticker": "T005", "filed_at": "2025-06-02",
         "actor": {"rank": 5}, "value": 1e5, "details": {}}
    v = quant.event_view(e, panel, None, {"has_skill": False}, rates)
    assert v["source"] == "base_rate" and v["p"] == br["p"] and v["band"]["n"] == br["n"]
    assert quant.event_view({**e, "kind": "fund", "action": "new"}, panel, None, None, rates)["source"] == "none"


def test_projection_ranges():
    p = quant.projection(100.0, 0.40)
    assert p["p05"] < p["p16"] < p["p50"] == 100.0 < p["p84"] < p["p95"]
    assert math.isclose(math.log(p["p84"] / 100), -math.log(p["p16"] / 100), rel_tol=1e-3)
    assert quant.projection(None, 0.3) is None and quant.projection(10, None) is None


def test_momentum_screen_backtest_and_market_context():
    panel, names = make_panel()
    bt = quant.backtest_momentum_screen(panel)
    assert bt["status"] == "ok" and bt["cohorts"] >= 10 and 0 <= bt["beat_rate"] <= 1
    assert bt["all_stocks_beat_rate"] is not None
    ctx = market.context(panel, None, {"VIXCLS": {"value": 33.0, "date": "2026-10-29", "change_1m": 8.0},
                                       "T10Y2Y": {"value": -0.2, "date": "2026-10-29", "change_1m": 0.1}})
    assert ctx["breadth"]["universe"] >= 200 and 0 <= ctx["breadth"]["above_200d"] <= 1
    assert ctx["regime"]["label"] in ("Risk-on", "Mixed", "Risk-off")
    assert any("VIX" in r for r in ctx["regime"]["negatives"]) and any("curve" in r for r in ctx["regime"]["negatives"])
    assert "not endorsed" in ctx["fred_notice"]


def test_ledger_records_once_grades_and_scores(monkeypatch):
    panel, names = make_panel()
    led = {"entries": {}}
    made = panel.date(200)
    assert ledger.record(led, "ev:1", ticker="T001", made=made, p=0.7, source="model", kind="insider", action="buy")
    assert not ledger.record(led, "ev:1", ticker="T001", made=made, p=0.1, source="model")     # can't be revised
    for k in range(2, 40):
        ledger.record(led, f"ev:{k}", ticker=names[k], made=made, p=0.55, source="model", kind="insider", action="buy")
    ledger.record(led, "ev:gone", ticker="NOPE", made=made, p=0.5, source="model", kind="insider", action="buy")
    today = dt.date.fromordinal(int(panel.dates[200])) + dt.timedelta(days=260)
    n = ledger.grade(led, panel, today=today)
    assert n == 40 and led["entries"]["ev:gone"].get("void")          # 39 graded + 1 voided (never priced)
    e1 = led["entries"]["ev:1"]
    fx = quant.forward_excess(panel, "T001", panel.ix_after(dt.date.fromisoformat(made)))
    assert e1["out"] == (1 if fx[1] > 0 else 0) and e1["p"] == 0.7
    m = ledger.metrics(led)
    s = m["by_source"]["model"]
    assert s["n"] == 39 and s["n_prob"] == 39 and 0 <= s["brier"] <= 1 and "calibration" in s
    assert m["graded"] == 39 and m["pending"] == 0


def test_dera_rank_matches_live_scale():
    assert quant.dera_rank("Officer", "President & CEO") == 5
    assert quant.dera_rank("Officer", "Chief Financial Officer") == 4
    assert quant.dera_rank("Director", "") == 3
    assert quant.dera_rank("Officer", "VP Sales") == 2
    assert quant.dera_rank("TenPercentOwner", "") == 2
    assert quant.dera_rank("Other", "") == 1


@pytest.mark.parametrize("auc,n", [(0.55, 100), (0.70, 400)])
def test_auc_standard_error_shrinks_with_n(auc, n):
    assert quant.auc_se(auc, n, n) > quant.auc_se(auc, 4 * n, 4 * n)


def test_odd_data_never_crashes():
    panel, names = make_panel(n_days=300, n_tickers=60)
    j = panel.col["T001"]
    panel.close[:, j] = np.nan                       # never traded
    panel.close[150:, panel.col["T002"]] = 0.0       # zero prices (bad data)
    panel.close[200:, panel.col["T003"]] = np.nan    # delisted mid-way
    for t in ("T001", "T002", "T003", "NOPE"):
        assert quant.price_features(panel, t, 250) is None or isinstance(quant.price_features(panel, t, 250), dict)
        assert quant.forward_excess(panel, t, 100) is None or len(quant.forward_excess(panel, t, 100)) == 2
    e = {"id": "z", "kind": "insider", "action": "buy", "ticker": "T001", "filed_at": None, "actor": {}, "value": None,
         "details": None}
    assert quant.event_view(e, panel, None, None, {})["source"] == "none"
    assert quant.event_view({**e, "ticker": None}, panel, None, None, {}) is None
    assert quant.momentum_screen(panel, 250, quant.liquid_universe(panel, 250)) == [] or True
    ctx = market.context(panel, None, None)
    assert ctx["regime"]["label"] in ("Risk-on", "Mixed", "Risk-off")
    assert market.context(None, None, {})["regime"]["label"] == "Mixed"


def test_zero_entry_price_is_treated_as_missing():
    panel, names = make_panel(n_days=300, n_tickers=5)
    j = panel.col["T001"]
    panel.close[100:104, j] = 0.0
    assert quant.forward_excess(panel, "T001", 100) is None            # no valid entry within 3 days
    panel.close[103, j] = 50.0
    r = quant.forward_excess(panel, "T001", 100)
    assert r is not None and math.isfinite(r[0])
