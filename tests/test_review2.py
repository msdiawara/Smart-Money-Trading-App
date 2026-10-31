"""Regression tests for the second independent review (statistics, Claude billing, ledger integrity).
Each test pins one finding so it can't come back."""
import datetime as dt
import json
import time
import types

import numpy as np
import pytest
import requests

from pipeline import analyst as A
from pipeline import analyst_main, ideas, jobs, ledger, quant
from pipeline.http import HttpClient, Response
from pipeline.util import read_json, write_json
from tests.test_analyst import VIEW
from tests.test_quant import make_panel


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    monkeypatch.setenv("SMR_FAKE_NOW", "2026-10-30T14:30:00+00:00")


# ---------------------------------------------------------------- M2: intervals that allow for shared moves

def test_clustered_interval_widens_only_when_outcomes_move_together():
    rng = np.random.default_rng(0)
    weeks = np.repeat(np.arange(100), 60)
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=int(7 * w)) for w in weeks]
    indep = rng.random(6000) < 0.5
    r = quant.rate_summary(list(indep), None, dates)
    wl, wh = quant.wilson(int(indep.sum()), 6000)
    assert r["lo"] <= wl and r["hi"] >= wh and (r["hi"] - r["lo"]) < 3 * (wh - wl)
    f = np.repeat(rng.normal(0, 0.15, 8), 13 * 60)[:6000]                 # quarter-long shared shocks
    shared = (rng.normal(0, 0.3, 6000) + f) > 0
    r2 = quant.rate_summary(list(shared), None, dates)
    assert (r2["hi"] - r2["lo"]) > 3 * (wh - wl) and r2["periods"] >= 7


# ---------------------------------------------------------------- M3: momentum interval with overlapping months

def test_hac_interval_is_wider_than_naive_for_overlapping_windows():
    rng = np.random.default_rng(1)
    shocks = rng.normal(0, 0.03, 40)
    x = np.convolve(shocks, np.ones(3), "valid")                            # 3-month windows measured monthly
    lo, hi = quant.hac_mean_ci(x, 2)
    se = x.std(ddof=1) / np.sqrt(len(x))
    assert hi - lo > 2 * 1.96 * se and lo < x.mean() < hi


# ---------------------------------------------------------------- M4: the gate on clustered and size-driven noise

def _clustered_null(seed, weeks=110, per_week=10, n_tickers=600):
    rng = np.random.default_rng(seed)
    idio = rng.uniform(0.25, 0.65, n_tickers)
    log_adv = rng.normal(6.8, 1.0, n_tickers)
    u = rng.normal(0, 1, (weeks + 20, n_tickers)) * (idio / np.sqrt(52))
    size_factor = rng.normal(0, 0.02, weeks + 20)          # small vs large caps drift apart some months
    out = []
    for w0 in range(weeks):
        for t in rng.integers(0, n_tickers, per_week):
            x = {"max_rank": int(rng.integers(1, 6)), "log_value": rng.normal(5, .7), "r1m": rng.normal(0, .1),
                 "r3m": rng.normal(0, .2), "mom6_1": rng.normal(0, .3), "vol3m": float(idio[t]), "dist_hi": -.1,
                 "dist_lo": .2, "log_adv": float(log_adv[t]), "routine_share": .5, "plan": 0.0, "n_owners": 1}
            for j in range(1 + int(rng.poisson(3))):            # a buying spree: several filings, one outcome
                w = w0 + int(rng.integers(0, 3))
                if w >= weeks:
                    continue
                ex = float(u[w + 1:w + 14, t].sum() - (log_adv[t] - 6.8) * size_factor[w + 1:w + 14].sum())
                out.append({"ticker": f"T{t}", "filed": (dt.date(2024, 11, 4) + dt.timedelta(days=7 * w)).isoformat(),
                            "x": {**x, "cluster30": min(j + 1, 6)}, "y": ex > 0, "excess": ex})
    out.sort(key=lambda s: s["filed"])
    return out


def test_gate_refuses_clustered_noise_and_holds_its_decision():
    passed = 0
    for seed in range(4):
        summ, bundle = quant.train_model(_clustered_null(seed))
        assert summ["status"] == "ok" and "test_stocks" in summ
        passed += summ["has_skill"]
    assert passed == 0
    # a decision already in force for the quarter is kept, whatever today's re-check says
    summ, bundle = quant.train_model(_clustered_null(9), decision=True)
    assert summ["has_skill"] and not summ["passed_today"] and bundle is not None
    assert summ["predictability"].startswith("Low")


def test_stratified_auc_matches_pairwise_definition():
    rng = np.random.default_rng(2)
    y = (rng.random(300) < .4).astype(int)
    p = np.round(rng.random(300), 1)
    g = rng.integers(0, 7, 300)
    num = den = 0.0
    for k in set(g):
        m = g == k
        for a in np.flatnonzero(m & (y == 1)):
            for b in np.flatnonzero(m & (y == 0)):
                den += 1
                num += 1.0 if p[a] > p[b] else 0.5 if p[a] == p[b] else 0.0
    assert quant.within_week_auc(y, p, g) == pytest.approx(num / den)


# ---------------------------------------------------------------- M7: live probabilities placed by rank

def test_calibration_band_uses_the_refit_models_fifths():
    bins = [{"pred": .40 + .02 * i, "obs": .30 + .1 * i, "lo": 0, "hi": 1, "n": 100} for i in range(5)]
    b = {"calibration": bins, "rank_edges": [.50, .52, .54, .56]}
    assert quant.calibration_band(b, .49)["fifth"] == 1 and quant.calibration_band(b, .49)["obs"] == pytest.approx(.30)
    assert quant.calibration_band(b, .99)["fifth"] == 5 and quant.calibration_band(b, .53)["fifth"] == 3


# ---------------------------------------------------------------- M5: routine needs the month 3 years back in the data

def test_routine_is_unknown_when_three_years_back_is_outside_the_data():
    months = {"_window_start": "2022-10", "cik:1": [f"{y}-{m:02d}" for y in range(2022, 2027) for m in (3, 6, 9, 12)
                                                     if f"{y}-{m:02d}" >= "2022-10"]}
    assert quant._routine(months, "1", dt.date(2025, 3, 10)) == 0.5          # 2022-03 isn't in the data
    assert quant._routine(months, "1", dt.date(2025, 12, 10)) == 1.0
    from pipeline import track_records
    assert track_records.is_routine(months, "cik:1", "2025-03-10") is None
    assert track_records.is_routine(months, "cik:1", "2025-12-10") is True


# ---------------------------------------------------------------- M6: live features built like the training rows

def test_live_insider_features_aggregate_the_day_like_training():
    ev = [{"id": "a", "kind": "insider", "action": "buy", "ticker": "ACME", "filed_at": "2026-10-20", "value": 50_000,
           "actor": {"id": "c1", "rank": 3}, "details": {"plan_10b5_1": True, "plan_flag_source": "footnote"}},
          {"id": "b", "kind": "insider", "action": "buy", "ticker": "ACME", "filed_at": "2026-10-20", "value": 150_000,
           "actor": {"id": "c2", "rank": 5}, "details": {}},
          {"id": "c", "kind": "insider", "action": "buy", "ticker": "ACME", "filed_at": "2026-10-29", "value": 10_000,
           "actor": {"id": "c3", "rank": 2}, "details": {}},
          {"id": "d", "kind": "insider", "action": "buy", "ticker": "ACME", "filed_at": "2026-12-15", "value": 10_000,
           "actor": {"id": "c4", "rank": 2}, "details": {}}]
    g = quant.live_insider_groups(ev)
    assert g["a"] == g["b"] and g["a"]["n_owners"] == 2 and g["a"]["max_rank"] == 5
    assert g["a"]["log_value"] == pytest.approx(np.log10(200_001)) and g["a"]["plan"] == 0.0   # footnote ≠ checkbox
    assert g["a"]["cluster30"] == 2 and g["c"]["cluster30"] == 3 and g["d"]["cluster30"] == 1   # only earlier filings


# ---------------------------------------------------------------- L1: delisted stocks are graded, not dropped

def test_delisted_stock_is_graded_to_its_last_trade_and_kept_in_the_backtest():
    panel, names = make_panel(n_days=300, n_tickers=4)
    c = panel.c("T001")
    c[150:] = np.nan                                      # stops trading for good on row 149
    info = {}
    r = quant.forward_excess(panel, "T001", 120, info=info)
    assert r is not None and info.get("delisted")
    assert r[0] == pytest.approx(c[149] / c[120] - 1, rel=1e-5)
    c2 = panel.c("T002")
    c2[170:190] = np.nan                                  # a long gap over the exit, then trading resumes: not graded
    assert quant.forward_excess(panel, "T002", 120) is None
    panel.col["DEAD"], panel.col["GAPP"] = panel.col["T001"], panel.col["T002"]     # real tickers are letters
    names = quant.delisted_names(panel)
    assert "DEAD" in names and "GAPP" not in names and "T001" not in names


# ---------------------------------------------------------------- M8: ledger dates and stable ids

def test_amended_filing_keeps_the_original_ledger_id():
    old = {"id": "f4:1:P", "kind": "insider", "issuer_cik": "9", "actor": {"id": "x"}, "action": "buy",
           "trade_date": "2026-10-01", "first_seen": "2026-10-02T10:00:00Z", "details": {}, "source": {}}
    new = {**old, "id": "f4:2:P", "details": {"amendment": True}, "source": {"accession": "2"}}
    st = {"events": [old]}
    assert jobs.replace_amended(st, new) and st["events"][0]["ledger_id"] == "f4:1:P"


def test_idea_registry_keeps_first_seen_across_short_gaps():
    reg, d0 = {}, dt.date(2026, 10, 1)
    mk = lambda: {"lists": {"momentum": {"items": [{"id": "mom:ACME"}]}}}  # noqa: E731
    ideas.apply_registry(mk(), reg, d0)
    ideas.apply_registry({"lists": {"momentum": {"items": []}}}, reg, d0 + dt.timedelta(days=2))
    back = ideas.apply_registry(mk(), reg, d0 + dt.timedelta(days=5))
    assert back["lists"]["momentum"]["items"][0]["first_seen"] == "2026-10-01"
    later = ideas.apply_registry(mk(), reg, d0 + dt.timedelta(days=60))
    assert later["lists"]["momentum"]["items"][0]["first_seen"] == "2026-11-30"


def test_ledger_hit_rate_needs_varying_probabilities():
    rows = [{"out": i % 2, "p": 0.47, "made": "2026-01-05", "ex": 0.0} for i in range(40)]
    assert ledger._score(rows)["hit_rate"] is None
    rows2 = [{**r, "p": 0.4 + 0.01 * (i % 20)} for i, r in enumerate(rows)]
    assert ledger._score(rows2)["hit_rate"] is not None


# ---------------------------------------------------------------- M9: the weakness screen uses its own rule

def test_weakness_screen_measures_the_fall_at_the_filing():
    panel, names = make_panel(n_days=320, n_tickers=3)
    i = panel.T - 1
    t = "T001"
    c = panel.c(t)
    c[i - 63:i + 1] = np.linspace(c[i - 63], c[i - 63] * 0.7, 64)          # down 30% into the latest filing
    f = dt.date.fromordinal(int(panel.dates[i])).isoformat()
    ev = [{"id": k, "kind": "insider", "action": "buy", "ticker": t, "filed_at": f, "value": 20_000,
           "actor": {"id": k, "name": k, "role": "Director"}} for k in ("a", "b")]
    import os
    os.environ["SMR_FAKE_NOW"] = f + "T20:00:00+00:00"
    out = ideas.insider_weakness(ev, {t: {"ret_3m": 5.0}}, panel)       # today's quote says +5%: ignored
    assert out and out[0]["measured_at"] == f and "30%" in out[0]["title"]


# ---------------------------------------------------------------- H1: billing never lost, never doubled

class _Seq:
    """Fake Messages API: answers from a list of callables/responses in order."""

    def __init__(self, steps):
        self.steps, self.posts = list(steps), 0

    def __call__(self, method, url, headers, data, timeout):
        self.posts += 1
        s = self.steps.pop(0)
        return s(data) if callable(s) else s


def _ok(usage, content, stop):
    return lambda data: Response(200, {}, json.dumps({"content": content, "stop_reason": stop, "model": data["model"],
                                                      "usage": usage}).encode(), "u")


U = {"input_tokens": 100_000, "output_tokens": 10_000}
PAUSE = [{"type": "server_tool_use", "id": "s", "name": "web_search", "input": {"query": "q"}}]
ANSWER = [{"type": "tool_use", "id": "t", "name": "submit_view", "input": VIEW}]


def _run(fake, **kw):
    c = A.Claude(HttpClient("t", transport=fake), "k")
    args = dict(model="claude-opus-5-5", effort="max", user="x", schema=A.TRADE_SCHEMA, tool_name="submit_view",
                tool_desc="d", max_tokens=20000, max_searches=6)
    args.update(kw)
    return c.run(**args)


def test_usage_is_kept_when_a_later_turn_fails():
    bad = Response(400, {}, b'{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long"}}', "u")
    r = _run(_Seq([_ok(U, PAUSE, "pause_turn"), bad]))
    assert r.answer is None and r.cost == pytest.approx(A.cost_usd("claude-opus-5-5", U)) and "400" in r.error


def test_a_lost_response_is_never_resent():
    def timeout(data):
        raise requests.exceptions.ReadTimeout("read timed out")
    fake = _Seq([timeout, _ok(U, ANSWER, "tool_use")])
    with pytest.raises(A.ClaudeError) as ei:
        _run(fake)
    assert fake.posts == 1 and ei.value.maybe_billed


def test_spend_limit_429_stops_and_overload_retries():
    no_ra = Response(429, {}, b'{"type":"error","error":{"type":"rate_limit_error","message":"limit"}}', "u")
    with pytest.raises(A.ClaudeError) as ei:
        _run(_Seq([no_ra]))
    assert ei.value.stop_run
    broke = Response(400, {}, b'{"type":"error","error":{"type":"invalid_request_error","message":"Your credit balance is too low to access the Claude API."}}', "u")
    with pytest.raises(A.ClaudeError) as ei:
        _run(_Seq([broke]))
    assert ei.value.stop_run and ei.value.transient           # stop the run, no strike against the task
    over = Response(529, {}, b'{"type":"error","error":{"type":"overloaded_error"}}', "u")
    fake = _Seq([over, _ok(U, ANSWER, "tool_use")])
    assert _run(fake).answer == VIEW and fake.posts == 2


def test_per_task_cost_cap_and_search_allowance():
    seen = []

    def pause(data):
        seen.append(next(t for t in data["tools"] if t.get("name") == "web_search")["max_uses"])
        u = {**U, "server_tool_use": {"web_search_requests": 4}}
        return _ok(u, PAUSE, "pause_turn")(data)
    r = _run(_Seq([pause, pause, pause, pause]), max_cost=2.5)
    assert r.answer is None and "cost cap" in r.error and r.cost >= 2.5
    assert seen[0] == 6 and seen[1] == 2 and seen[2] == 1           # remaining searches, never below 1


def test_max_tokens_is_reported():
    r = _run(_Seq([_ok(U, [{"type": "text", "text": "thinking..."}], "max_tokens")]))
    assert r.hit_max_tokens and r.answer is None


# ---------------------------------------------------------------- H2: the budget doesn't drift

def test_budget_reservations_release_exactly():
    b = A.Budget(daily_usd=10.0, spend={})
    r1, r2 = b.reserve("deep"), b.reserve("deep")
    b.charge("deep", "m", 1.5)                  # changes the estimate while r2 is still out
    b.release(r1)
    b.release(r2)
    assert b.reserved == 0 and b.today()["usd"] == pytest.approx(1.5) and b.left() == pytest.approx(8.5)
    assert "reserved" not in json.dumps(b.spend)


# ---------------------------------------------------------------- L6: brief time written without quotes in YAML

def test_brief_slot_accepts_yaml_sexagesimal():
    assert analyst_main.slot_hm("07:45") == (7, 45) and analyst_main.slot_hm(465) == (7, 45)
    assert analyst_main.slot_label(465) == "07:45" and analyst_main.slot_hm("25:00") is None


# ---------------------------------------------------------------- L3: never publish NaN

def test_json_output_never_contains_nan(tmp_path):
    write_json(tmp_path / "x.json", {"a": float("nan"), "b": [np.float64("inf"), 1.0]})
    assert json.loads((tmp_path / "x.json").read_text()) == {"a": None, "b": [None, 1.0]}
    assert quant.projection(float("nan"), 0.3) is None and quant.projection(10.0, float("inf")) is None


# ---------------------------------------------------------------- M10/L7: workflow guards

def test_analyst_workflow_never_starts_from_an_unrestored_memory():
    import yaml
    from pathlib import Path
    wf = Path(__file__).resolve().parents[1] / ".github" / "workflows"
    an = yaml.safe_load((wf / "analyst.yml").read_text())
    steps = an["jobs"]["analyze"]["steps"]
    guard = next(s for s in steps if s.get("id") == "guard")
    assert "cache-matched-key" in guard["env"]["MATCHED"] and "gh cache list" in guard["run"]
    run = next(s for s in steps if s.get("id") == "analyst")
    assert run["if"] == "steps.guard.outputs.go == '1'"
    assert an["permissions"]["actions"] == "read"
    rf = (wf / "refresh.yml").read_text()
    assert "ANTHROPIC_API_KEY: ${{" not in rf and "HAS_ANTHROPIC_KEY" in rf


def test_ledger_entries_are_dated_when_written(tmp_path):
    led = {"entries": {}}
    ledger.record(led, "ev:x", ticker="ACME", made="2026-10-30", p=0.5, source="base_rate")
    assert not ledger.record(led, "ev:x", ticker="ACME", made="2026-10-31", p=0.9, source="base_rate")
    assert led["entries"]["ev:x"]["made"] == "2026-10-30" and led["entries"]["ev:x"]["p"] == 0.5
    _ = types, read_json
