"""Claude's price forecasts and post-mortems in the analyst workflow, end to end against a fake API, and their
journey into the forecast ledger (graded from the session after they were written)."""
import json
import time

import pytest

from pipeline import analyst as A
from pipeline import analyst_main, fledger, main
from pipeline.settings import Settings
from pipeline.util import read_json
from tests.fake_web import FakeWeb
from tests.test_analyst import FakeClaude


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("SMR_FAKE_NOW", "2026-10-30T20:30:00+00:00")
    monkeypatch.setenv("SEC_CONTACT_EMAIL", "me@example.com")
    monkeypatch.setenv("MASSIVE_API_KEY", "demo")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setattr(time, "sleep", lambda s: None)
    s = Settings.load(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out")
    s.config["funds"]["list"] = []
    s.config["refresh"]["run_budget_minutes"] = 60
    s.config["alerts"]["watch_tickers"] = ["NVDA"]
    s.config["analyst"] = {"deep_min_score": 101, "standard_min_score": 101, "ideas_per_day": 0, "brief_times_et": []}
    return s


def test_clean_forecast_rejects_nonsense():
    good = {"horizons": [{"h": "1m", "low_10": 90, "median": 100, "high_90": 110, "p_up": 55},
                         {"h": "3m", "low_10": 80, "median": 102, "high_90": 125, "p_up": 0.56},
                         {"h": "1w", "low_10": 99, "median": 98, "high_90": 101, "p_up": 0.5}],
            "scenarios": [{"name": "bull", "probability": 2, "price_3m": 130, "drivers": "x"},
                          {"name": "bear", "probability": 2, "price_3m": 80, "drivers": "y"}],
            "news_digest": [{"date": "2026-10-29", "headline": "h", "why_it_matters": "w", "url": "javascript:alert(1)"}],
            "stance": "favorable", "probability": 0.6}
    out = A.clean_forecast(good, 100.0)
    assert set(out["horizons"]) == {"1m", "3m"}                        # the unordered 1w row is dropped
    assert out["horizons"]["1m"]["p_up"] == pytest.approx(0.55)         # percent → fraction
    assert [s["probability"] for s in out["scenarios"]] == [0.5, 0.5]
    assert out["news_digest"][0]["url"] is None
    assert A.clean_forecast({"horizons": good["horizons"][:1]}, 100.0) is None   # no 3-month row → rejected


def test_forecast_task_end_to_end_into_the_ledger(env):
    web = FakeWeb()
    main.run(force={"all"}, transport=web, settings=env)
    assert (env.cache_dir / "forecasts.json.gz").exists() and (env.cache_dir / "stock_ctx.json.gz").exists()
    fake = FakeClaude()
    info = analyst_main.run(env, transport=fake)
    assert info["done"] >= 1, info
    calls = [c for c in fake.calls if any(t.get("name") == "submit_forecast" for t in c["tools"])]
    assert calls, "no forecast call made"
    prompts = [c["messages"][0]["content"][0]["text"] for c in calls]          # tasks run in parallel: any order
    payloads = [json.loads(t[t.rfind("\n{") + 1:]) for t in prompts]
    payload = next(x for x in payloads if x["ticker"] == "NVDA")                   # the watchlist stock is always planned
    prompt = prompts[payloads.index(payload)]
    assert payload["reference_session"] == "2026-11-02"                           # Fri evening → Mon
    assert payload["app_model_forecast"]["horizons"]["1m"]["p10"] > 0
    assert "your_forecast_track_record" in payload and "HOW TO FORECAST" in prompt
    store = read_json(env.cache_dir / "ai" / "analyses.json.gz")
    pf = store["price_forecasts"]["NVDA"]
    assert pf["ref_date"] == "2026-11-02" and pf["horizons"]["3m"]["median"] > 0 and pf["made"] == "2026-10-30"
    assert any(f["fid"].startswith("claude-fc:") for f in store["forecasts"])
    # the next refresh records Claude's forecast (and the blend) in the forecast ledger, by price level
    main.run(force=set(), transport=web, settings=env)
    L = fledger.load(env.heavy_dir)
    cl = [e for e in L["pending"].values() if e["src"] == "claude" and e["t"] == "NVDA"]
    bl = [e for e in L["pending"].values() if e["src"] == "blend" and e["t"] == "NVDA"]
    assert {e["h"] for e in cl} == {5, 21, 63, 252} and all(e.get("pq") and e["ref"] == "2026-11-02" for e in cl)
    assert bl and all(0 < e["w"] < 1 for e in bl)
    doc = json.loads((env.out_dir / "stock" / "NVDA.json").read_text())
    assert doc["claude"]["thesis"] and doc["blend"]["3m"]["q"]["50"] > 0
    # a second analyst run the same day doesn't forecast NVDA again
    n_before = len(calls)
    analyst_main.run(env, transport=fake)
    assert len([c for c in fake.calls if any(t.get("name") == "submit_forecast" for t in c["tools"])]) == n_before


def test_postmortem_turns_a_miss_into_a_lesson(env):
    web = FakeWeb()
    main.run(force={"all"}, transport=web, settings=env)
    miss = {"t": "NVDA", "h": 21, "made": "2026-09-01", "ref": "2026-09-02", "end": "2026-10-01", "p0": 180.0, "p1": 140.0,
            "ret_pct": -22.2, "p10": 165.0, "p50": 182.0, "p90": 200.0, "pit": 0.003, "path": []}
    (env.cache_dir / "claude_feedback.json.gz").unlink(missing_ok=True)
    from pipeline.util import write_json
    write_json(env.cache_dir / "claude_feedback.json.gz", {"track": {}, "misses": [miss]})
    env.config["analyst"]["forecasts_per_day"] = 0
    fake = FakeClaude()
    analyst_main.run(env, transport=fake)
    store = read_json(env.cache_dir / "ai" / "analyses.json.gz")
    assert store["lessons"] and store["lessons"][-1]["lesson"].startswith("Widen ranges")
    assert "pm:NVDA:21:2026-09-01" in store["postmortems"]
    # lessons reach the app
    main.run(force=set(), transport=web, settings=env)
    acc = json.loads((env.out_dir / "forecast_accuracy.json").read_text())
    assert acc["lessons"][0]["cause"] == "event_surprise"
