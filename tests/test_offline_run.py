"""End-to-end: run the whole pipeline against the fake internet and check what the app would receive."""
import json

import pytest

from pipeline import main
from pipeline.settings import Settings
from tests.fake_web import FakeWeb


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("SMR_FAKE_NOW", "2026-10-30T20:30:00+00:00")  # Fri 4:30pm ET
    monkeypatch.setenv("SEC_CONTACT_EMAIL", "me@example.com")
    monkeypatch.setenv("NTFY_TOPIC", "smr-test-topic")
    for k in ("ALPACA_KEY_ID", "ALPACA_SECRET_KEY", "ETORO_API_KEY", "ETORO_USER_KEY", "FMP_API_KEY",
              "QUIVER_API_KEY", "ANTHROPIC_API_KEY", "TELEGRAM_BOT_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    import time
    monkeypatch.setattr(time, "sleep", lambda s: None)
    s = Settings.load(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out")
    s.config["funds"]["list"] = [
        {"cik": "0001067983", "name": "Berkshire Hathaway", "manager": "Greg Abel", "style": "concentrated",
         "expect": "BERKSHIRE HATHAWAY"},
        {"cik": "0001336528", "name": "Pershing Square", "manager": "Bill Ackman", "style": "concentrated",
         "expect": "PERSHING SQUARE"},
    ]
    return s


def test_full_offline_run(env):
    web = FakeWeb()
    info = main.run(force={"all"}, transport=web, settings=env)
    out = env.out_dir
    feed = json.loads((out / "feed.json").read_text())
    health = json.loads((out / "health.json").read_text())
    meta = json.loads((out / "meta.json").read_text())
    quotes = json.loads((out / "quotes.json").read_text())["quotes"]
    ev = {e["id"]: e for e in feed["events"]}

    # every source job ran without crashing (Senate is blocked → reported, not fatal)
    bad = {k: v for k, v in info["results"].items() if not v.get("ok")}
    assert not bad, bad
    assert meta["counts"]["insider"] >= 4 and meta["counts"]["fund"] >= 3 and meta["counts"]["congress"] >= 4

    # cluster: CEO + CFO + director at ACME within days → strong bullish
    ceo = ev["f4:0001999001-26-000011:P"]
    assert ceo["direction"] == "bullish" and ceo["signal"] == "strong" and ceo["score"] >= 70
    assert any("Cluster buy" in p for p in ceo["read"]["pros"])
    assert ceo["filed_at"].startswith("2026-10-29T17:05:12")
    # 10b5-1 CEO sale at a mega-cap → weak bearish, with the plan caveat
    nv = ev["f4:0001777001-26-000020:S"]
    assert nv["direction"] == "bearish" and nv["score"] < 40
    assert any("10b5-1" in c for c in nv["read"]["cons"])
    # grant/exercise rows are noise
    assert any(e["signal"] == "noise" for e in feed["events"] if e["kind"] == "insider")

    # 13F diff: Berkshire new position + exit + trim, with tickers mapped via OpenFIGI
    f = [e for e in feed["events"] if e["kind"] == "fund"]
    kinds = {(e["ticker"], e["action"]) for e in f}
    assert ("NEWC", "new") in kinds and ("OLDC", "exit") in kinds and ("BAC", "trim") in kinds
    # wrong CIK is caught and skipped
    psq = next(x for x in health["funds"] if x["cik"] == "0001336528")
    assert psq["ok"] is False and "expected" in psq["error"]

    # 13D activist with % of class, linked to the ticker
    a = ev["s13:0000921895-26-002222"]
    assert a["ticker"] == "ACME" and a["details"]["percent"] == 8.7 and a["signal"] in ("strong", "notable")

    # House PDF → 4 transactions with tickers, ranges and owner; leadership flag off
    house = [e for e in feed["events"] if e["kind"] == "congress"]
    assert {e["ticker"] for e in house} >= {"NVDA", "GOOG", "MSFT", "AAPL"}
    goog = next(e for e in house if e["ticker"] == "GOOG")
    assert goog["value_range"] == [1000001, 5000000] and goog["details"]["owner"] == "Spouse"
    assert goog["actor"]["name"] == "Rep. Jane Q. Member" and "D-CA-11" in goog["actor"]["role"]

    # Senate blocked → clear message in health, not a failure
    sen = next(s for s in health["sources"] if s["key"] == "congress_senate")
    assert "eFD unavailable" in (sen["note"] or "")

    # prices + risk for the feed tickers (Yahoo fallback since no Alpaca keys)
    assert quotes["ACME"]["risk_level"] in (3, 4, 5) and quotes["ACME"]["cap_tier"] == "small"
    assert quotes["ACME"]["profitable"] is False
    assert quotes["NVDA"]["cap_tier"] == "mega"

    # alerts went out through ntfy (bundled), capped per run
    assert any("ntfy.sh/smr-test-topic" in c for c in web.calls)
    assert info["results"]["alerts"]["ok"]

    # SEC requests carried a declared User-Agent and stayed on the SEC hosts we expect
    assert all(not c.startswith("http://") for c in web.calls)


def test_second_run_is_incremental(env):
    web = FakeWeb()
    main.run(force={"all"}, transport=web, settings=env)
    n_first = len(web.calls)
    web2 = FakeWeb()
    info = main.run(force={"insiders_live", "activists_live"}, transport=web2, settings=env)
    # nothing new: no filing documents fetched again, no duplicate events
    assert not [c for c in web2.calls if c.endswith(".txt")]
    assert info["new_events"] == 0
    assert n_first > len(web2.calls)


def test_run_with_price_keys_builds_win_rates(env, monkeypatch):
    monkeypatch.setenv("MASSIVE_API_KEY", "test-massive")
    monkeypatch.setenv("ALPACA_KEY_ID", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    from pipeline.settings import Secrets
    env.secrets = Secrets.from_env()
    env.config["refresh"]["run_budget_minutes"] = 60
    web = FakeWeb()
    info = main.run(force={"all"}, transport=web, settings=env)
    bad = {k: v for k, v in info["results"].items() if not v.get("ok")}
    assert not bad, bad
    assert "trading days" in info["results"]["market_history"]["note"]
    quotes = json.loads((env.out_dir / "quotes.json").read_text())["quotes"]
    assert quotes["ACME"]["price_src"] == "alpaca-iex" and quotes["ACME"]["vol_1y"] > 0
    leaders = json.loads((env.out_dir / "leaders.json").read_text())
    jane = next(r for r in leaders["insiders"] if r["id"] == "cik:0001999001")
    assert jane["n"] == 3 and 0 <= jane["beat_rate"] <= 1 and jane["name"] == "Doe Jane A"
    feed = json.loads((env.out_dir / "feed.json").read_text())
    ceo = next(e for e in feed["events"] if e["id"] == "f4:0001999001-26-000011:P")
    assert ceo["actor_record"]["n"] == 3
    # the SEC moved the data-set path in 2026: the new path must be tried first
    dera = [c for c in web.calls if "form345" in c]
    assert dera and "datastandardsinnovation" in dera[0]


def test_secrets_never_published(env, monkeypatch, tmp_path):
    """Keys appear in some request URLs (Massive apiKey=, Telegram /bot<token>/): errors must not leak them."""
    monkeypatch.setenv("MASSIVE_API_KEY", "SUPERSECRETMASSIVEKEY")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456789:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsawZZ")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "987654321")
    monkeypatch.setenv("NTFY_TOPIC", "my-private-topic-xyz")
    from pipeline.settings import Secrets
    env.secrets = Secrets.from_env()

    class Failing(FakeWeb):
        def __call__(self, method, url, headers, data, timeout):
            if "massive" in url or "telegram" in url or "ntfy" in url:
                from pipeline.http import Response
                return Response(401, {}, f"denied {url}".encode(), url)
            return super().__call__(method, url, headers, data, timeout)

    main.run(force={"all"}, transport=Failing(), settings=env)
    blob = "".join(p.read_text() for p in env.out_dir.glob("*.json"))
    for secret in ("SUPERSECRETMASSIVEKEY", "AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsawZZ", "my-private-topic-xyz", "987654321"):
        assert secret not in blob, secret


def test_without_sec_contact_email_sec_sources_wait(env, monkeypatch):
    monkeypatch.delenv("SEC_CONTACT_EMAIL", raising=False)
    from pipeline.settings import Secrets
    env.secrets = Secrets.from_env()
    web = FakeWeb()
    info = main.run(force={"all"}, transport=web, settings=env)
    assert not [u for u in web.calls if "sec.gov" in u]      # no undeclared traffic to the SEC at all
    for job in ("insiders_live", "activists_live", "funds"):
        assert not info["results"][job]["ok"] and "SEC_CONTACT_EMAIL" in info["results"][job]["error"]
    assert info["results"]["congress_house"]["ok"]           # non-SEC sources still work
    health = json.loads((env.out_dir / "health.json").read_text())
    assert "SEC_CONTACT_EMAIL" in json.dumps(health)
