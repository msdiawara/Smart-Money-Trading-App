"""The self-audit that runs at the end of every refresh: passes on a healthy run, catches broken data."""
import json
import time

import pytest

from pipeline import main, selfcheck
from pipeline.settings import Settings
from tests.fake_web import FakeWeb


@pytest.fixture()
def run(tmp_path, monkeypatch):
    monkeypatch.setenv("SMR_FAKE_NOW", "2026-10-30T20:30:00+00:00")
    monkeypatch.setenv("SEC_CONTACT_EMAIL", "me@example.com")
    monkeypatch.setenv("MASSIVE_API_KEY", "demo")
    monkeypatch.setenv("NTFY_TOPIC", "smr-test-topic")
    monkeypatch.setattr(time, "sleep", lambda s: None)
    s = Settings.load(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out")
    s.config["funds"]["list"] = []
    s.config["refresh"]["run_budget_minutes"] = 60
    web = FakeWeb()
    info = main.run(force={"all"}, transport=web, settings=s)
    return s, info, web


def test_healthy_run_has_no_failures(run):
    s, info, _ = run
    a = json.loads((s.out_dir / "audit.json").read_text())
    assert a["counts"]["fail"] == 0, [c for c in a["checks"] if c["status"] == "fail"]
    ids = {c["id"] for c in a["checks"]}
    assert {"files", "feed:schema", "prices:sanity", "forecast:valid", "fledger:grading", "sizes", "config"} <= ids
    assert a["stock_files"]["forecasts"] >= 5


def test_catches_a_broken_forecast_and_a_corrupt_file_and_alerts_once(run):
    s, info, web = run
    p = s.out_dir / "stock" / "NVDA.json"
    doc = json.loads(p.read_text())
    doc["forecast"]["horizons"]["1m"]["q"]["90"] = doc["forecast"]["horizons"]["1m"]["q"]["10"] / 2   # out of order
    p.write_text(json.dumps(doc))
    (s.out_dir / "ideas.json").write_text("{not json")
    state = {"jobs": {}, "accuracy": {}, "analyst": {}}
    n0 = sum(1 for u in web.calls if "ntfy.sh" in u)
    a = selfcheck.run_and_publish(s, state, info, web_client(web))
    fails = {c["id"] for c in a["checks"] if c["status"] == "fail"}
    assert "forecast:valid" in fails and "file:ideas.json" in fails and a["overall"] == "fail"
    assert sum(1 for u in web.calls if "ntfy.sh" in u) == n0 + 1          # one push for the new failures
    selfcheck.run_and_publish(s, state, info, web_client(web))
    assert sum(1 for u in web.calls if "ntfy.sh" in u) == n0 + 1          # not again for the same failures


def test_stale_sources_and_overspend_are_reported(run):
    s, info, _ = run
    recent = "2026-10-30T20:25:00Z"
    state = {"jobs": {
        "insiders_live": {"last_run": recent, "last_ok": "2026-10-20T00:00:00Z", "consecutive_failures": 500, "error": "HTTP 503"},
        "headlines": {"last_run": recent, "last_ok": "2026-10-30T18:00:00Z", "consecutive_failures": 3, "error": "Timeout"},
        "companies": {"last_run": recent, "last_ok": "2026-10-29T12:00:00Z", "consecutive_failures": 1, "error": "x"},
        "congress_senate": {"last_run": recent, "last_ok": "2026-09-01T00:00:00Z", "consecutive_failures": 900, "error": "403"},
        "etoro": {"last_run": "2026-09-01T00:00:00Z", "last_ok": "2026-08-01T00:00:00Z", "consecutive_failures": 9, "error": "off"}},
        "analyst": {"today": {"usd": 9.0}, "last_run": {"started": "2026-10-30T20:00:00Z"}}}
    s.secrets.anthropic_key = "sk-ant-x"
    a = selfcheck.run_and_publish(s, state, info, None)
    by = {c["id"]: c for c in a["checks"]}
    assert by["sources:dead"]["status"] == "fail" and "insiders_live" in json.dumps(by["sources:dead"]["detail"])
    assert "congress_senate" not in json.dumps(by["sources:dead"]["detail"])        # known blocked: a warning only
    assert "headlines" in json.dumps(by["sources:failing"]["detail"])
    assert "companies" in json.dumps(by["sources:stale"]["detail"])
    assert "etoro" not in json.dumps(a["checks"])                                    # not being tried: ignored
    assert by["analyst:budget"]["status"] == "fail"


def test_a_throttled_failure_is_alerted_later_not_dropped(run):
    s, info, web = run
    (s.out_dir / "ideas.json").write_text("{not json")
    state = {"jobs": {}, "accuracy": {}, "analyst": {}, "audit": {"alerted_at": "2026-10-30T19:00:00Z", "alerted": []}}
    n0 = sum(1 for u in web.calls if "ntfy.sh" in u)
    selfcheck.run_and_publish(s, state, info, web_client(web))
    assert sum(1 for u in web.calls if "ntfy.sh" in u) == n0                        # throttled: not yet…
    assert "file:ideas.json" not in state["audit"]["alerted"]
    state["audit"]["alerted_at"] = "2026-10-30T10:00:00Z"                          # …six hours later
    (s.out_dir / "ideas.json").write_text("{not json")
    selfcheck.run_and_publish(s, state, info, web_client(web))
    assert sum(1 for u in web.calls if "ntfy.sh" in u) == n0 + 1 and "file:ideas.json" in state["audit"]["alerted"]


def test_missing_stock_pages_and_bad_claude_blocks_fail(run):
    s, info, _ = run
    files = sorted((s.out_dir / "stock").glob("*.json"))
    doc = json.loads(files[0].read_text())
    doc["claude"] = {"horizons": {"1m": {"low_10": 120, "median": 100, "high_90": 130}}}
    files[0].write_text(json.dumps(doc))
    state = {"jobs": {}, "accuracy": {}, "analyst": {}, "stock_pages": {"names": len(files) * 3, "written": len(files)}}
    a = selfcheck.run_and_publish(s, state, info, None)
    fails = {c["id"] for c in a["checks"] if c["status"] == "fail"}
    assert {"stock:missing", "claude:valid"} <= fails


def web_client(web):
    from pipeline.http import HttpClient
    return HttpClient(sec_user_agent="t", transport=web)
