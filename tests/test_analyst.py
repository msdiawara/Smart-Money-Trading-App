"""Claude analyst: API handling (web search, pause_turn, fallbacks, strict tool answer), cost and budget
accounting, task planning, and the end-to-end flow into the app and the forecast ledger — all against a fake
Claude API (no network)."""
import json

import pytest

from pipeline import analyst as A
from pipeline import analyst_main, extras, main
from pipeline.http import HttpClient, Response
from pipeline.settings import Secrets, Settings
from pipeline.util import read_json
from tests.fake_web import FakeWeb

VIEW = {"stance": "favorable", "probability": 0.58, "confidence": "medium", "headline": "Three insiders buying the dip",
        "thesis": "Cluster of opportunistic buys after a 30% fall.", "bull_case": ["CEO and CFO bought"],
        "bear_case": ["Guidance cut last quarter"], "catalysts": [{"when": "2026-11-05", "what": "Q3 earnings"}],
        "watch_for": ["A fourth insider buy"], "invalidation": "Close below $20 on volume",
        "position_note": "At most half the ceiling until earnings", "key_facts_checked": ["Earnings on Nov 5"],
        "data_gaps": []}
BRIEF = {"regime_read": "Uptrend with narrow breadth.", "what_matters_this_week": ["CPI on Nov 12"], "risks": ["Rates"],
         "sector_view": "Tech leads.", "setups": [{"ticker": "ACME", "name": "Acme", "setup_type": "insider_conviction",
                                                    "thesis": "t", "why_now": "w", "trigger": "x", "invalidation": "y",
                                                    "probability": 0.56, "horizon_days": 63, "confidence": "low",
                                                    "risk_note": "small cap"}],
         "avoid": [{"ticker": "OLDC", "reason": "dilution"}], "bottom_line": "Be selective."}


POSTMORTEM = {"what_happened": "Shares fell 18% after the company cut guidance on Nov 12.", "cause": "event_surprise",
              "foreseeable": "partly", "what_was_missed": "Guidance risk was flagged by two suppliers.",
              "lesson": "Widen ranges into earnings when suppliers have warned on demand.", "range_advice": "wider"}


def forecast_answer(content) -> dict:
    text = content if isinstance(content, str) else "".join(b.get("text", "") for b in content)
    payload = json.loads(text[text.rfind("\n{") + 1:])
    p = float(payload["last_close"])
    rows = [{"h": h, "low_10": round(p * lo, 2), "median": round(p * mid, 2), "high_90": round(p * hi, 2), "p_up": pu}
            for h, lo, mid, hi, pu in (("1w", .95, 1.005, 1.06, .52), ("1m", .88, 1.01, 1.14, .53),
                                       ("3m", .80, 1.03, 1.28, .55), ("1y", .62, 1.08, 1.70, .58))]
    return {"horizons": rows, "path_notes": "Choppy into earnings, then trend resumes if guidance holds.",
            "dip_risk": {"window": "Around the next earnings report", "why": "Guidance risk", "p_dip10_3m": 0.32},
            "peak_view": {"window": "After earnings if margins hold", "why": "Estimates look beatable"},
            "key_dates": [{"date": "2026-11-19", "event": "Q3 earnings", "impact": "Biggest single-day risk"}],
            "scenarios": [{"name": "bull", "probability": 0.25, "price_3m": round(p * 1.3, 2), "drivers": "AI demand beats"},
                          {"name": "base", "probability": 0.5, "price_3m": round(p * 1.03, 2), "drivers": "In line"},
                          {"name": "bear", "probability": 0.25, "price_3m": round(p * 0.8, 2), "drivers": "Capex digestion"}],
            "valuation_view": {"label": "Expensive", "reasoning": "Premium multiple versus peers, partly earned by growth."},
            "bubble_view": {"level": "Elevated", "reasoning": "Large run-up with the AI basket; issuance modest."},
            "ai_exposure": "core",
            "news_digest": [{"date": "2026-10-29", "headline": "Results beat estimates", "why_it_matters": "Demand holding",
                             "url": "https://www.reuters.com/tech/nvidia-results"}],
            "stance": "favorable", "probability": 0.6, "confidence": "medium",
            "thesis": "Demand still outruns supply; the risk is a spending pause.", "risks": ["Capex digestion", "Export rules"],
            "what_would_change_my_mind": "Hyperscaler capex guidance cut.", "lessons_applied": []}


class FakeClaude:
    """Mimics the Messages API: a web-search turn that pauses, then the tool answer."""

    def __init__(self, reject_new_search_tool=False, never_answer=False):
        self.calls = []
        self.reject = reject_new_search_tool
        self.never = never_answer

    def __call__(self, method, url, headers, data, timeout):
        if "api.anthropic.com" not in url:
            return FakeWeb()(method, url, headers, data, timeout)
        self.calls.append(data)
        tools = data["tools"]
        tool = next(t for t in tools if t.get("name", "").startswith("submit"))
        web = next((t for t in tools if t.get("name") == "web_search"), None)
        if self.reject and web and web["type"] != "web_search_20250305":
            body = {"type": "error", "error": {"type": "invalid_request_error",
                                               "message": f"tools.0: unknown tool type '{web['type']}' for web_search"}}
            return Response(400, {}, json.dumps(body).encode(), url)
        usage = {"input_tokens": 3000, "output_tokens": 2000, "cache_creation_input_tokens": 1500,
                 "cache_read_input_tokens": 0}
        last = data["messages"][-1]
        if web and last["role"] == "user" and len(data["messages"]) == 1:
            content = [{"type": "server_tool_use", "id": "srv1", "name": "web_search", "input": {"query": "ACME news"}},
                       {"type": "web_search_tool_result", "tool_use_id": "srv1", "content": [
                           {"type": "web_search_result", "url": "https://www.sec.gov/acme-8k", "title": "ACME 8-K",
                            "page_age": "2 days ago", "encrypted_content": "x"},
                           {"type": "web_search_result", "url": "https://news.example.com/acme", "title": "ACME news",
                            "encrypted_content": "y"}]},
                       {"type": "text", "text": "Earnings are on Nov 5.", "citations": [
                           {"type": "web_search_result_location", "url": "https://news.example.com/acme",
                            "title": "ACME news", "cited_text": "Nov 5", "encrypted_index": "z"}]}]
            body = {"content": content, "stop_reason": "pause_turn", "model": data["model"],
                    "usage": {**usage, "server_tool_use": {"web_search_requests": 1}}}
            return Response(200, {}, json.dumps(body).encode(), url)
        if self.never:
            body = {"content": [{"type": "text", "text": "I think it's fine."}], "stop_reason": "end_turn",
                    "model": data["model"], "usage": usage}
            return Response(200, {}, json.dumps(body).encode(), url)
        if tool["name"] == "submit_forecast":
            answer = forecast_answer(data["messages"][0]["content"])
        elif tool["name"] == "submit_postmortem":
            answer = POSTMORTEM
        else:
            answer = BRIEF if tool["name"] == "submit_brief" else VIEW
        body = {"content": [{"type": "text", "text": "Done."},
                            {"type": "tool_use", "id": "tu1", "name": tool["name"], "input": answer}],
                "stop_reason": "tool_use", "model": data["model"], "usage": {**usage, "cache_read_input_tokens": 1500}}
        return Response(200, {}, json.dumps(body).encode(), url)


def test_client_handles_search_pause_and_answer():
    fake = FakeClaude()
    c = A.Claude(HttpClient("t", transport=fake), "sk-test")
    r = c.run(model="claude-opus-5-5", effort="max", user="x", schema=A.TRADE_SCHEMA, tool_name="submit_view",
              tool_desc="d", max_tokens=48000, max_searches=6)
    assert r.answer == VIEW and r.stop == "tool_use" and r.error is None
    first = fake.calls[0]
    assert first["output_config"] == {"effort": "max"} and first["model"] == "claude-opus-5-5"
    assert first["system"][0]["cache_control"] == {"type": "ephemeral"} and "HONESTY RULES" in first["system"][0]["text"]
    assert any(t.get("strict") for t in first["tools"]) and first["tools"][0]["max_uses"] == 6
    assert fake.calls[1]["messages"][-1]["role"] == "assistant"            # pause_turn continued unchanged
    assert [s["url"] for s in r.sources][0] == "https://news.example.com/acme"    # cited source first
    # cost from the API's own usage report: Opus 5.5 $4/$20 per M, cache write $5, cache read $0.20, $0.01/search
    expect = (6000 * 4 + 4000 * 20 + 3000 * 5 + 1500 * 0.20) / 1e6 + 0.01
    assert r.cost == pytest.approx(expect)


def test_client_falls_back_to_basic_search_tool_and_nudges():
    fake = FakeClaude(reject_new_search_tool=True)
    c = A.Claude(HttpClient("t", transport=fake), "k", web_tool="web_search_20260318")     # a newer version configured
    r = c.run(model="claude-opus-5-5", effort="high", user="x", schema=A.TRADE_SCHEMA, tool_name="submit_view",
              tool_desc="d", max_tokens=8000, max_searches=3)
    assert r.answer == VIEW and c.web_tool == "web_search_20250305"
    never = A.Claude(HttpClient("t", transport=FakeClaude(never_answer=True)), "k")
    r2 = never.run(model="claude-opus-5-5", effort="low", user="x", schema=A.TRADE_SCHEMA, tool_name="submit_view",
                   tool_desc="d", max_tokens=8000, max_searches=0)
    assert r2.answer is None and "no answer" in r2.error and r2.cost > 0      # tried once more, then gave up


def test_cleaning_bounds_answers():
    v = A.clean_trade({**VIEW, "probability": 93, "stance": "buy now!!"}, base_p=0.47)
    assert v["probability"] == 0.93 and v["stance"] == "neutral" and "flag" in v
    v2 = A.clean_trade({**VIEW, "probability": "n/a", "bull_case": ["x"] * 20}, base_p=None)
    assert v2["probability"] is None and len(v2["bull_case"]) == 6
    b = A.clean_brief({**BRIEF, "setups": BRIEF["setups"] * 12})
    assert len(b["setups"]) == 8 and b["setups"][0]["ticker"] == "ACME"


def test_budget_reserves_and_learns_costs(monkeypatch):
    b = A.Budget(daily_usd=1.0, spend={})
    assert b.can("deep")
    b.charge("deep", "claude-opus-5-5", 0.9)
    assert not b.can("deep") and b.left() == pytest.approx(0.1)
    assert b.estimate("deep") == 0.9 and b.can("standard")


def test_prices_and_fallback_are_conservative():
    u = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
    assert A.cost_usd("claude-opus-5-5", u) == pytest.approx(24.0)
    assert A.cost_usd("claude-fable-5-1", u) == pytest.approx(60.0)
    assert A.cost_usd("some-future-model", u) == pytest.approx(60.0)          # unknown → most expensive


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("SMR_FAKE_NOW", "2026-10-30T14:30:00+00:00")             # Fri 10:30am ET
    monkeypatch.setenv("SEC_CONTACT_EMAIL", "me@example.com")
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "av-test")
    for k in ("ALPACA_KEY_ID", "ALPACA_SECRET_KEY", "MASSIVE_API_KEY", "ANTHROPIC_API_KEY", "NTFY_TOPIC"):
        monkeypatch.delenv(k, raising=False)
    import time
    monkeypatch.setattr(time, "sleep", lambda s: None)
    s = Settings.load(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out")
    s.config["funds"]["list"] = [{"cik": "0001067983", "name": "Berkshire Hathaway", "expect": "BERKSHIRE HATHAWAY"}]
    s.config["analyst"] = {"deep_min_score": 60, "standard_min_score": 30, "daily_budget_usd": 5.0}
    return s


def test_end_to_end_analyst_into_app_and_ledger(env, monkeypatch):
    main.run(force={"all"}, transport=FakeWeb(), settings=env)
    q = json.loads((env.out_dir / "quotes.json").read_text())["quotes"]
    assert q["ACME"]["earnings"]["date"] == "2026-11-05" and q["ACME"]["short"]["days_to_cover"] == 2.5
    # no key → the analyst says so and spends nothing
    info = analyst_main.run(env, transport=FakeClaude())
    assert info["skipped"] == "no ANTHROPIC_API_KEY secret"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    env.secrets = Secrets.from_env()
    fake = FakeClaude()
    info = analyst_main.run(env, transport=fake)
    assert info["done"] >= 2 and not info["errors"], info
    store = read_json(env.cache_dir / "ai" / "analyses.json.gz")
    assert store["briefs"][-1]["slot"] == "07:45" and store["briefs"][-1]["brief"]["setups"][0]["ticker"] == "ACME"
    assert store["trades"] and all(v["view"]["stance"] in A.STANCES for v in store["trades"].values())
    spend = read_json(env.cache_dir / "ai" / "spend.json")
    day = list(spend["days"].values())[-1]
    assert 0 < day["usd"] <= 5.0 and day["calls"] == info["done"]
    deep = [c for c in fake.calls if c["output_config"]["effort"] == "max"]
    assert deep and all(c["model"] == "claude-opus-5-5" for c in deep)
    # same day again: nothing new to do (no duplicate brief, no re-analysis)
    again = analyst_main.run(env, transport=FakeClaude())
    assert again["planned"] == 0
    # the next data refresh shows Claude's views and records its probabilities for grading
    main.run(transport=FakeWeb(), settings=env)
    feed = json.loads((env.out_dir / "feed.json").read_text())["events"]
    shown = [e for e in feed if e.get("analysis")]
    assert shown and shown[0]["analysis"]["headline"] == VIEW["headline"]
    assert shown[0]["analysis"]["sources"][0]["url"].startswith("https://")
    ideas = json.loads((env.out_dir / "ideas.json").read_text())
    assert ideas["brief"]["brief"]["bottom_line"] == "Be selective."
    led = read_json(env.cache_dir / "ledger.json.gz")["entries"]
    assert any(k.startswith("claude:") and v["src"] == "claude" and v["p"] == 0.58 for k, v in led.items())
    assert any(k.startswith("claude-setup:") for k in led)
    health = json.loads((env.out_dir / "health.json").read_text())
    assert health["analyst"]["today"]["usd"] > 0 and health["analyst"]["deep_model"] == "claude-opus-5-5"


def test_budget_stops_spending(env, monkeypatch):
    main.run(force={"all"}, transport=FakeWeb(), settings=env)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    env.secrets = Secrets.from_env()
    env.config["analyst"] = {"deep_min_score": 0, "standard_min_score": 0, "daily_budget_usd": 0.05}
    info = analyst_main.run(env, transport=FakeClaude())
    spend = read_json(env.cache_dir / "ai" / "spend.json")
    assert list(spend["days"].values())[-1]["usd"] <= 0.05 + 0.2      # at most one call's overshoot
    assert info["planned"] >= 1 and info["done"] <= 1


def test_parsers_for_extras():
    settle, rows = extras.parse_finra("symbolCode|currentShortPositionQuantity|previousShortPositionQuantity|"
                                      "averageDailyVolumeQuantity|daysToCoverQuantity|settlementDate\n"
                                      "BRK/B|100|80|50|2.0|2026-10-15\n")
    assert settle == "2026-10-15" and rows["BRK.B"] == [100.0, 80.0, 50.0, 2.0]
    with pytest.raises(ValueError):
        extras.parse_finra("foo,bar\n1,2\n")
    cal = extras.parse_av_calendar("symbol,name,reportDate,fiscalDateEnding,estimate,currency\n"
                                   "BRK-B,Berkshire,2026-11-03,2026-09-30,,USD\n")
    assert cal["BRK.B"]["date"] == "2026-11-03"
