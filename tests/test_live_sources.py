"""The sources added in round three: Form 144 sale notices, transparent-ETF daily trades and FINRA daily
short-sale volume, each parsed from fixtures that mirror the real files, then run end to end offline."""
import json
import time

import pytest

from pipeline import etf, extras, form144, main, scoring
from pipeline.settings import Settings
from pipeline.util import read_json
from tests.fake_web import FakeWeb, ark_csv, finra_daily

FX = __import__("pathlib").Path(__file__).parent / "fixtures"


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("SMR_FAKE_NOW", "2026-10-30T14:30:00+00:00")             # Fri 10:30am ET
    monkeypatch.setenv("SEC_CONTACT_EMAIL", "me@example.com")
    monkeypatch.setenv("MASSIVE_API_KEY", "m")
    for k in ("ALPACA_KEY_ID", "ALPACA_SECRET_KEY", "ANTHROPIC_API_KEY", "NTFY_TOPIC", "ALPHAVANTAGE_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    s = Settings.load(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out")
    s.config["funds"]["list"] = []
    s.config["refresh"]["run_budget_minutes"] = 30
    return s


# ---------------------------------------------------------------- Form 144

def test_form144_parser_reads_both_namespace_styles():
    p = form144.parse_form144((FX / "form144_officer.xml").read_text())
    assert p["issuer_cik"] == "0001234567" and p["filer_cik"] == "0001999001" and p["seller"] == "Doe Jane A"
    assert p["relationships"] == ["Officer", "Director"] and p["shares"] == 400000 and p["value"] == 9004000.0
    assert p["approx_sale_date"] == "2026-11-03" and p["pct_outstanding"] == pytest.approx(0.6667, abs=1e-3)
    assert p["sold_past_3m"][0]["proceeds"] == 1400000.0 and p["plan_adoption_dates"] == []
    assert p["acquisitions"][0]["nature"] == "Founder Stock" and p["broker"].startswith("Morgan Stanley")
    q = form144.parse_form144((FX / "form144_plan.xml").read_text())          # `com:` prefix, 10b5-1 plan, nothing sold
    assert q["plan_adoption_dates"] == ["2026-03-14"] and q["sold_past_3m"] == [] and q["issuer_name"] == "NVIDIA CORP"
    with pytest.raises(ValueError):
        form144.parse_form144((FX / "sched13d.xml").read_text())


def test_form144_event_and_score():
    p = form144.parse_form144((FX / "form144_officer.xml").read_text())
    e = form144.build_event(p, "0001999001-26-000144", "2026-10-30T14:15:00Z", "https://sec.gov/x", "2026-10-30", "ACME")
    assert e["kind"] == "insider" and e["action"] == "sale_notice" and e["actor"]["id"] == "cik:0001999001"
    assert e["actor"]["role"] == "Officer, Director" and e["actor"]["rank"] == 3 and e["price"] == pytest.approx(22.51)
    scoring.score_event(e, scoring.Ctx())
    assert e["direction"] == "bearish" and e["signal"] in ("weak", "mild") and "notice of intent" in " ".join(e["read"]["cons"])
    assert any("0.67% of all shares" in x for x in e["read"]["pros"])
    q = form144.parse_form144((FX / "form144_plan.xml").read_text())
    e2 = form144.build_event(q, "0001777001-26-000144", "2026-10-30T15:30:00Z", "u", "2026-10-30", "NVDA")
    scoring.score_event(e2, scoring.Ctx())
    assert e2["details"]["plan_10b5_1"] and e2["score"] < e["score"]
    assert any("10b5-1" in c for c in e2["read"]["cons"]) and any("compensation" in c for c in e2["read"]["cons"])


# ---------------------------------------------------------------- transparent ETFs

def test_ark_holdings_parse_and_diff():
    day1, h1 = etf.parse_holdings(ark_csv(1))
    day2, h2 = etf.parse_holdings(ark_csv(2))
    assert day1 == "2026-10-30" and day2 == "2026-11-02" and "BAC" in h1 and "" not in h1   # cash row skipped
    assert h1["TSLA"]["shares"] == 3_000_000 and h1["TSLA"]["value"] == pytest.approx(1.2e9) and 60 < h1["TSLA"]["weight"] < 70
    assert etf.weight_total(h1) == pytest.approx(100, abs=0.2) and len(etf.positions(h1)) == 6
    trades = {t["ticker"]: t for t in etf.diff_holdings(h1, h2)}
    assert set(trades) == {"ACME", "NVDA", "BAC", "NEWC"}            # TSLA/BTC1/MSFT only moved with fund flow
    assert trades["ACME"]["action"] == "add" and trades["ACME"]["pct_change"] == pytest.approx(30, abs=0.6)
    assert trades["NVDA"]["action"] == "trim" and trades["BAC"]["action"] == "exit" and trades["NEWC"]["action"] == "new"
    assert abs(trades["ACME"]["flow"] - 0.01) < 0.002
    evs = etf.build_events(etf.DEFAULT_FUNDS[0], list(trades.values()), day2, "https://assets.ark-funds.com/x.csv")
    e = next(x for x in evs if x["ticker"] == "ACME")
    assert e["kind"] == "etf" and e["trade_date"] == "2026-10-30" and e["actor"]["id"] == "etf:ARKK"   # file dated Mon → trades Fri
    assert e["value"] == pytest.approx(0.29 * 2_000_000 * 22.5, rel=0.05)
    with pytest.raises(ValueError):
        etf.parse_holdings("<html>nope</html>")


def test_etf_scores_and_consensus():
    _, h1 = etf.parse_holdings(ark_csv(1))
    _, h2 = etf.parse_holdings(ark_csv(2))
    evs = etf.build_events(etf.DEFAULT_FUNDS[0], etf.diff_holdings(h1, h2), "2026-11-02", "u")
    for e in evs:
        scoring.score_event(e, scoring.Ctx(etf_consensus=lambda e: 1))
    by = {e["ticker"]: e for e in evs}
    assert by["NEWC"]["direction"] == "bullish" and by["BAC"]["direction"] == "bearish"
    assert by["ACME"]["score"] > by["NVDA"]["score"] and any("other fund" in p for p in by["ACME"]["read"]["pros"])
    assert all(1 <= e["score"] <= 99 and e["read"]["summary"] for e in evs)


# ---------------------------------------------------------------- FINRA daily short volume

def test_short_volume_parse_and_view():
    day, rows = extras.parse_short_volume(finra_daily("20261029"))
    assert day == "2026-10-29" and rows["ACME"][0] / rows["ACME"][1] == pytest.approx(0.62, abs=0.001)
    hist = {d: [40.0 + (i % 3), 100.0] for i, d in enumerate(f"2026-10-{k:02d}" for k in range(1, 21))}
    hist["2026-10-29"] = [62.0, 100.0]
    v = extras.short_volume_view(hist)
    assert v["pct"] == 62.0 and 40 <= v["avg20_pct"] <= 42 and v["asof"] == "2026-10-29" and v["days"] == 21
    assert extras.short_volume_view({"2026-10-29": [1.0, 0.0]}) is None
    with pytest.raises(ValueError):
        extras.parse_short_volume("a,b,c\n1,2,3")


# ---------------------------------------------------------------- end to end, offline

def test_new_sources_flow_into_the_app(env):
    fake = FakeWeb()
    main.run(force={"all"}, transport=fake, settings=env)
    feed = json.loads((env.out_dir / "feed.json").read_text())["events"]
    notices = [e for e in feed if e["id"].startswith("f144:")]
    assert len(notices) == 2 and all(e["action"] == "sale_notice" for e in notices)
    acme = next(e for e in notices if e["ticker"] == "ACME")
    assert acme["actor"]["name"] == "Doe Jane A" and acme["details"]["approx_sale_date"] == "2026-11-03"
    assert not [e for e in feed if e["kind"] == "etf"]            # first snapshot only
    st = read_json(env.cache_dir / "state.json.gz")
    assert st["etf"]["urls"]["ARKK"].endswith("ARKK_HOLDINGS.csv") and "ARKW" in st["etf"]["errors"]
    health = json.loads((env.out_dir / "health.json").read_text())
    jobs = {s["key"]: s for s in health["sources"]}
    assert not jobs["sale_notices"]["error"] and not jobs["etf_daily"]["error"] and "first snapshot" in jobs["etf_daily"]["note"]
    assert jobs["sale_notices"]["items"] == 2
    # daily short volume reached the quotes (ACME's 62% day vs ~41% average)
    q = json.loads((env.out_dir / "quotes.json").read_text())["quotes"]
    assert q["ACME"]["short_volume"]["pct"] == 62.0 and q["ACME"]["short_volume"]["asof"] == "2026-10-29"
    assert q["ACME"]["short_volume"]["avg20_pct"] is not None
    assert q["ACME"].get("rel_volume") and q["ACME"]["rel_volume"]["ratio"] > 0
    # next day: the fund's new file turns into trades
    fake.ark_version = 2
    import os
    os.environ["SMR_FAKE_NOW"] = "2026-11-02T13:30:00+00:00"
    main.run(force={"etf_daily"}, transport=fake, settings=env)
    feed = json.loads((env.out_dir / "feed.json").read_text())["events"]
    trades = [e for e in feed if e["kind"] == "etf"]
    assert {e["ticker"] for e in trades} == {"ACME", "NVDA", "BAC", "NEWC"}
    assert all(e.get("score") and e.get("read") for e in trades)
    assert next(e for e in trades if e["ticker"] == "ACME")["action"] == "add"
    # same file again: no duplicates
    main.run(force={"etf_daily"}, transport=fake, settings=env)
    feed2 = json.loads((env.out_dir / "feed.json").read_text())["events"]
    assert len([e for e in feed2 if e["kind"] == "etf"]) == len(trades)
    led = read_json(env.cache_dir / "ledger.json.gz")["entries"]
    assert any(k.startswith("ev:etf:ARKK") for k in led) and any(k.startswith("ev:f144:") for k in led)


def test_short_volume_retries_todays_file_but_not_holidays(tmp_path, monkeypatch):
    from pipeline.http import HttpClient, Response
    monkeypatch.setenv("SMR_FAKE_NOW", "2026-10-30T22:30:00+00:00")      # Fri 6:30pm ET: today's file is due
    monkeypatch.setattr(time, "sleep", lambda s: None)
    asked = []

    def send(method, url, headers, data, timeout):
        asked.append(url)
        day = url[-12:-4]
        if day in ("20261029", "20261028"):
            return Response(200, {}, finra_daily(day).encode(), url)
        return Response(403, {}, b"<Error>NoSuchKey</Error>", url)       # 10/30 not posted yet; 10/27-ish "holidays"
    http = HttpClient("t", transport=send)
    n, note = extras.refresh_short_volume(http, tmp_path, {"ACME"})
    have = read_json(tmp_path / "short_volume.json.gz")
    assert n == 2 and have["days"]["2026-10-29"] == "2026-10-29"
    assert "2026-10-30" not in have["days"]                               # will be asked again next run
    assert have["days"].get("2026-10-27") == "missing"                    # a past day with no file: not asked again
    asked.clear()
    extras.refresh_short_volume(http, tmp_path, {"ACME"})
    assert any(u.endswith("20261030.txt") for u in asked) and not any(u.endswith("20261027.txt") for u in asked)


def test_short_volume_backfills_tickers_that_arrive_later(tmp_path, monkeypatch):
    from pipeline.http import HttpClient, Response
    monkeypatch.setenv("SMR_FAKE_NOW", "2026-10-30T14:30:00+00:00")
    monkeypatch.setattr(time, "sleep", lambda s: None)
    asked = []

    def send(method, url, headers, data, timeout):
        asked.append(url[-12:-4])
        day = url[-12:-4]
        if day in ("20261029", "20261028", "20261027", "20261026", "20261023", "20261022"):
            return Response(200, {}, finra_daily(day).encode(), url)
        return Response(403, {}, b"<Error>NoSuchKey</Error>", url)
    http = HttpClient("t", transport=send)
    extras.refresh_short_volume(http, tmp_path, {"ACME"})
    have = read_json(tmp_path / "short_volume.json.gz")
    assert len(have["by_ticker"]["ACME"]) == 6 and "NVDA" not in have["by_ticker"]
    asked.clear()
    extras.refresh_short_volume(http, tmp_path, {"ACME", "NVDA"})       # NVDA enters the feed: known days re-read for it
    have = read_json(tmp_path / "short_volume.json.gz")
    assert len(have["by_ticker"]["NVDA"]) == 6 and len(have["by_ticker"]["ACME"]) == 6
    assert asked.count("20261029") == 1 and "20261022" in asked


# ---------------------------------------------------------------- fixes from the fourth independent review

def test_etf_ignores_partial_and_stale_files(tmp_path, monkeypatch):
    from pipeline.http import HttpClient, Response
    monkeypatch.setattr(time, "sleep", lambda s: None)
    served = {"body": ark_csv(1)}
    http = HttpClient("t", transport=lambda m, u, h, d, t: Response(200, {}, served["body"].encode(), u))
    fund = [etf.DEFAULT_FUNDS[0]]
    st = {}
    etf.refresh(http, tmp_path, fund, st)                                   # first snapshot (10/30)
    served["body"] = "\n".join(ark_csv(2).splitlines()[:4])                  # next day's file cut to 3 rows
    evs, note = etf.refresh(http, tmp_path, fund, st)
    assert not evs and "incomplete" in note and st["etf"]["errors"]["ARKK"] == "incomplete holdings file"
    assert read_json(etf.snapshot_path(tmp_path))["ARKK"]["date"] == "2026-10-30"      # snapshot kept
    served["body"] = "\n".join(ark_csv(2).splitlines()[:6])                  # 5 of 6 rows: weights ~95%, rows 83%
    evs, note = etf.refresh(http, tmp_path, fund, st)
    assert not evs and "incomplete" in note                                   # caught by the weight total, not the row count
    served["body"] = ark_csv(2)
    evs, note = etf.refresh(http, tmp_path, fund, st)
    assert {e["ticker"] for e in evs} == {"ACME", "NVDA", "BAC", "NEWC"} and "ARKK" not in st["etf"]["errors"]
    served["body"] = ark_csv(1)                                              # an old file served again
    evs, note = etf.refresh(http, tmp_path, fund, st)
    assert not evs and "older file" in note and read_json(etf.snapshot_path(tmp_path))["ARKK"]["date"] == "2026-11-02"


def test_etf_accepts_a_real_shrink_once_the_same_file_persists(tmp_path, monkeypatch):
    from pipeline.http import HttpClient, Response
    monkeypatch.setattr(time, "sleep", lambda s: None)
    monkeypatch.setenv("SMR_FAKE_NOW", "2026-11-02T22:00:00+00:00")
    small = "\n".join(ark_csv(2).splitlines()[:4]) + "\n"             # a fund that really is down to 3 names
    small = small.replace("11/02/2026", "11/03/2026")
    served = {"body": ark_csv(2)}
    http = HttpClient("t", transport=lambda m, u, h, d, t: Response(200, {}, served["body"].encode(), u))
    fund, st = [etf.DEFAULT_FUNDS[0]], {}
    etf.refresh(http, tmp_path, fund, st)
    served["body"] = small
    evs, note = etf.refresh(http, tmp_path, fund, st)
    assert not evs and "incomplete" in note and st["etf"]["pending"]["ARKK"]["date"] == "2026-11-03"
    monkeypatch.setenv("SMR_FAKE_NOW", "2026-11-03T01:30:00+00:00")   # 3.5 hours later, same file still served
    evs, note = etf.refresh(http, tmp_path, fund, st)
    assert {e["ticker"] for e in evs} >= {"BTC1", "MSFT"} and all(e["action"] == "exit" for e in evs if e["ticker"] in ("BTC1", "MSFT"))
    assert "ARKK" not in st["etf"]["errors"] and "ARKK" not in st["etf"]["pending"]
    assert read_json(etf.snapshot_path(tmp_path))["ARKK"]["date"] == "2026-11-03"


def test_etf_trade_date_uses_the_market_calendar():
    import datetime as dt
    days = [dt.date(2026, 11, 24), dt.date(2026, 11, 25), dt.date(2026, 11, 27)]    # Thanksgiving 11/26 closed
    cal = [d.toordinal() for d in days]
    evs = etf.build_events(etf.DEFAULT_FUNDS[0], [{"ticker": "X", "action": "new", "shares": 10, "delta": 10, "price": 1.0,
                                                   "value": 10.0, "weight": 1.0, "prev_weight": None, "pct_change": None,
                                                   "company": "X", "cusip": None, "flow": 0.0}], "2026-11-27", "u", cal)
    assert evs[0]["trade_date"] == "2026-11-25"
    assert etf.prev_trading_day(dt.date(2026, 11, 27), None) == dt.date(2026, 11, 25)   # no calendar: NYSE holidays by rule
    # the stored calendar always lags a day or two: days past its end are trading days unless weekend/holiday
    ends_fri = [dt.date(2026, 10, 30).toordinal()]
    assert etf.prev_trading_day(dt.date(2026, 11, 3), ends_fri) == dt.date(2026, 11, 2)
    assert etf.prev_trading_day(dt.date(2026, 11, 2), ends_fri) == dt.date(2026, 10, 30)
    assert etf.prev_trading_day(dt.date(2026, 11, 30), [dt.date(2026, 11, 25).toordinal()]) == dt.date(2026, 11, 27)


def test_form144_amendment_replaces_the_notice_it_names():
    from pipeline import jobs
    xml = (FX / "form144_officer.xml").read_text()
    older = form144.build_event(form144.parse_form144(xml), "0001999001-26-000100", "2026-10-03T14:00:00Z", "u", "2026-10-03", "ACME")
    newer = form144.build_event(form144.parse_form144(xml), "0001999001-26-000144", "2026-10-30T14:00:00Z", "u", "2026-10-30", "ACME")
    amend_xml = xml.replace("<submissionType>144</submissionType>",
                            "<submissionType>144/A</submissionType><previousAccessionNumber>0001999001-26-000100</previousAccessionNumber>")
    p = form144.parse_form144(amend_xml)
    assert p["doc_type"] == "144/A" and p["previous_accession"] == "0001999001-26-000100"
    state = {"events": [newer, older], "seen": {}}
    fake_sub = type("S", (), {"header": {"FILED AS OF DATE": "20261031"}, "first_xml": lambda self, **k: amend_xml,
                              "accepted_at": lambda self: None})()
    import pipeline.sec as sec
    orig = sec.fetch_submission
    sec.fetch_submission = lambda http, cik, acc: fake_sub
    try:
        n = jobs.process_form144(None, state, "1999001", "0001999001-26-000150", None,
                                 {"by_cik": {"0001234567": {"ticker": "ACME"}}}, [])
    finally:
        sec.fetch_submission = orig
    ids = [e["id"] for e in state["events"]]
    assert n == 0 and "f144:0001999001-26-000150" in ids and "f144:0001999001-26-000100" not in ids
    assert "f144:0001999001-26-000144" in ids                                   # the newer notice is untouched
    amended = next(e for e in state["events"] if e["id"].endswith("000150"))
    assert amended["ledger_id"] == "f144:0001999001-26-000100" and amended["details"]["amends"] == "0001999001-26-000100"
    # a second amendment names the original again: it replaces the first amendment, not a fresh card
    sec.fetch_submission = lambda http, cik, acc: fake_sub
    try:
        n2 = jobs.process_form144(None, state, "1999001", "0001999001-26-000160", None,
                                  {"by_cik": {"0001234567": {"ticker": "ACME"}}}, [])
    finally:
        sec.fetch_submission = orig
    ids = [e["id"] for e in state["events"]]
    assert n2 == 0 and ids.count("f144:0001999001-26-000160") == 1 and "f144:0001999001-26-000150" not in ids and len(ids) == 2


def test_relative_volume_uses_twenty_prior_days():
    import numpy as np
    import types
    from pipeline import quant
    T = 60
    V = np.ones((T, 2)) * 100.0
    V[-21:-1, 1] = np.arange(40, 60)           # the 20 days before the last: mean 49.5
    V[-1, 1] = 99.0
    days = list(range(700000, 700000 + T))
    C = np.ones((T, 2)) * 10.0
    P = quant.Panel(types.SimpleNamespace(dates=days, close=C, vol=V, col={"SPY": 0, "X": 1}))
    rv = extras.relative_volume(P, "X")
    assert rv["ratio"] == pytest.approx(99 / 49.5, rel=1e-6) and rv["avg20"] == pytest.approx(49.5)
