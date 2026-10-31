"""Regression tests for the issues found in the independent code review (one test per fix)."""
import datetime as dt
import json
import time

import pytest

from pipeline import congress, form4, jobs, scoring, sec, track_records
from pipeline.http import HttpClient, HttpError, Response
from pipeline.prices import BENCH, MarketMatrix, PriceStore
from pipeline.state import empty_state, job_due, job_record


def transport_from(routes):
    """routes: list of (substring, status, body) — first match wins."""
    calls = []

    def send(method, url, headers, data, timeout):
        calls.append((method, url))
        for sub, status, body in routes:
            if sub in url:
                raw = body if isinstance(body, bytes) else (json.dumps(body) if not isinstance(body, str) else body).encode()
                return Response(status, {}, raw, url)
        return Response(404, {}, b"not found", url)
    send.calls = calls
    return send


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    monkeypatch.setenv("SMR_FAKE_NOW", "2026-10-30T20:30:00+00:00")


# ---------------------------------------------------------------- H3: splits

def _matrix_with_split(tmp_path):
    m = MarketMatrix(tmp_path / "m.npz")
    days = [dt.date(2026, 10, d) for d in (19, 20, 21, 22, 23, 26, 27)]   # split executes on the 26th
    for d in days[:5]:
        m.add_day(d, [("XYZ", 100.0 + d.day % 3, 1000), ("SPY", 500.0, 1e6)])
    for d in days[5:]:
        m.add_day(d, [("XYZ", 50.0 + d.day % 3 / 2, 2000), ("SPY", 501.0, 1e6)])
    e = days[5].toordinal()
    m.fetched = [e - 3] * 5 + [e + 1, e + 2]        # pre-split rows downloaded before the split happened
    return m, days


def test_split_rebases_rows_downloaded_before_the_split(tmp_path):
    m, days = _matrix_with_split(tmp_path)
    split = {"id": "S1", "ticker": "XYZ", "execution_date": "2026-10-26", "split_from": 1, "split_to": 2}
    assert m.apply_splits([split]) == ["XYZ"]
    c = m.series("XYZ")["c"]
    assert all(49 <= x <= 52 for x in c), c                        # no fake -50% crash any more
    assert m.series("SPY")["c"][0] == 500.0                          # other tickers untouched
    assert m.apply_splits([split]) == []                             # applied once only
    m.save()
    m2 = MarketMatrix(tmp_path / "m.npz")
    assert "S1" in m2.applied and m2.fetched == m.fetched and m2.series("XYZ")["c"] == c


def test_split_already_reflected_in_download_is_left_alone(tmp_path):
    m = MarketMatrix(tmp_path / "m.npz")
    for d in (21, 22, 26, 27):
        m.add_day(dt.date(2026, 10, d), [("XYZ", 50.0, 1000)])       # backfilled after the split: adjusted
    assert m.apply_splits([{"id": "S2", "ticker": "XYZ", "execution_date": "2026-10-26",
                            "split_from": 1, "split_to": 2}]) == []
    assert m.series("XYZ")["c"] == [50.0] * 4


def test_split_waits_for_a_post_split_day_and_checks_same_day_downloads(tmp_path):
    m = MarketMatrix(tmp_path / "m.npz")
    m.add_day(dt.date(2026, 10, 22), [("XYZ", 100.0, 1)])
    m.add_day(dt.date(2026, 10, 23), [("XYZ", 101.0, 1)])
    e = dt.date(2026, 10, 26).toordinal()
    m.fetched = [e - 2, e]                                             # 2nd row downloaded on the split day
    sp = {"id": "S3", "ticker": "XYZ", "execution_date": "2026-10-26", "split_from": 1, "split_to": 2}
    assert m.apply_splits([sp]) == [] and "S3" not in m.applied        # nothing post-split to compare yet
    m.add_day(dt.date(2026, 10, 26), [("XYZ", 50.4, 1)])
    m.fetched[-1] = e + 1
    assert m.apply_splits([sp]) == ["XYZ"]
    assert m.series("XYZ")["c"] == [50.0, 50.5, 50.4]                 # same-day row was still unadjusted


def test_price_store_heals_a_provider_rebase(tmp_path):
    st = PriceStore(tmp_path)
    days = [dt.date(2026, 9, 1) + dt.timedelta(days=i) for i in range(60)]
    days = [d for d in days if d.weekday() < 5]
    st.put("XYZ", [(d, 100.0, 10) for d in days], "yahoo")
    # provider re-adjusted after a 1:2 split and we only re-downloaded the last 10 days
    st.put("XYZ", [(d, 50.0, 20) for d in days[-10:]], "yahoo")
    assert set(st.series("XYZ")["c"]) == {50.0}
    # ordinary overlap (same numbers) changes nothing
    st.put("ABC", [(d, 10.0 + i, 1) for i, d in enumerate(days)], "yahoo")
    before = list(st.series("ABC")["c"])
    st.put("ABC", [(d, 10.0 + i, 1) for i, d in enumerate(days)][-10:], "yahoo")
    assert st.series("ABC")["c"] == before


# ---------------------------------------------------------------- M11: same window as SPY

def test_forward_return_uses_the_benchmark_calendar(tmp_path):
    st = PriceStore(tmp_path)
    days = [d for d in (dt.date(2026, 1, 1) + dt.timedelta(days=i) for i in range(200)) if d.weekday() < 5]
    st.put(BENCH, [(d, 100.0 + i, 1) for i, d in enumerate(days)], "t")
    gappy = [(d, 10.0 + i, 1) for i, d in enumerate(days) if not (20 <= i < 30)]   # 10 missing days
    st.put("GAP", gappy, "t")
    r, rb, entry = track_records.forward_return(st, "GAP", days[4], hold=40)
    # entry = day 5, exit = day 45 on SPY's calendar → stock 15→55, SPY 105→145
    assert entry == days[5].isoformat()
    assert abs(r - (55 / 15 - 1)) < 1e-9 and abs(rb - (145 / 105 - 1)) < 1e-9


# ---------------------------------------------------------------- M10: per-line holdings

def test_holdings_change_is_computed_per_ownership_line():
    owner = {"cik": "0000000001", "name": "DOE JANE", "is_officer": True, "officer_title": "CEO"}
    tx = lambda shares, after, direct, nature=None: {
        "code": "P", "shares": shares, "price": 10.0, "date": "2026-10-28", "derivative": False, "acq_disp": "A",
        "shares_after": after, "direct": direct, "nature": nature, "security": "Common Stock",
        "footnote_ids": [], "timeliness": None}
    parsed = {"owners": [owner], "issuer": {"cik": "0000000099", "name": "ACME", "ticker": "ACME"},
              "footnotes": {}, "plan_10b5_1": False, "plan_in_footnotes": False, "doc_type": "4",
              "transactions": [tx(1000, 101000, "D"), tx(1000, 2000, "I", "By Trust")]}
    ev = form4.build_events(parsed, "0000000001-26-000001", None, "u", "2026-10-29")[0]
    # direct 100,000→101,000 and trust 1,000→2,000: 2,000 bought on 101,000 held before = +1.98%
    assert ev["holdings_after"] == 103000
    assert abs(ev["holdings_change_pct"] - 2000 / 101000 * 100) < 0.01


# ---------------------------------------------------------------- Form 4/A

def test_amendment_replaces_the_original_instead_of_duplicating():
    state = empty_state()
    base = {"kind": "insider", "issuer_cik": "0000000099", "actor": {"id": "cik:1"}, "action": "buy",
            "trade_date": "2026-10-28", "details": {}, "source": {"accession": "A1"}}
    new_ids = []
    jobs.add_events(state, [dict(base, id="f4:A1:P", quote_at_filing={"price": 9})], new_ids)
    first_seen = state["events"][0]["first_seen"]
    amended = dict(base, id="f4:A2:P", details={"amendment": True}, source={"accession": "A2"})
    assert jobs.replace_amended(state, amended) is True
    assert [e["id"] for e in state["events"]] == ["f4:A2:P"]
    assert state["events"][0]["first_seen"] == first_seen and state["events"][0]["details"]["replaces"] == "A1"
    assert state["events"][0]["quote_at_filing"] == {"price": 9}


# ---------------------------------------------------------------- H1: bad items don't stall the feed

def test_an_item_that_keeps_failing_is_given_up_after_two_tries():
    state = empty_state()
    assert jobs.failed_twice(state, "f4", "acc1") is False
    assert jobs.failed_twice(state, "f4", "acc1") is True


def test_unexpected_parse_errors_in_live_feed_are_contained(monkeypatch, tmp_path):
    from pipeline.settings import Settings
    s = Settings.load(cache_dir=tmp_path / "c", out_dir=tmp_path / "o")
    entries = [sec.FeedEntry(form="4", name="X", cik="0000000001", role="Reporting", accession=f"000000000{i}-26-000001",
                             updated="2026-10-30T12:00:00-04:00", link="") for i in (1, 2)]
    monkeypatch.setattr(jobs.sec, "current_feed", lambda *a, **k: entries)
    calls = []

    def boom(http, settings, state, cik, acc, upd, new_ids):
        calls.append(acc)
        if acc.startswith("0000000001"):
            raise KeyError("weird document")          # not an HttpError/ValueError
        return 0
    monkeypatch.setattr(jobs, "process_form4", boom)
    state = empty_state()
    http = HttpClient("t", transport=transport_from([]))
    for _ in range(3):
        jobs.job_insiders_live(http, s, state, jobs.Budget(60), [])
    # the bad filing was tried twice then skipped; the good one wasn't blocked by it
    assert calls.count("0000000001-26-000001") == 2
    assert "0000000001-26-000001" in state["seen"]["f4"]


# ---------------------------------------------------------------- H5: first 13F under a CIK

def test_first_13f_records_a_baseline_without_flooding_new_positions(monkeypatch, tmp_path):
    from pipeline.figi import FigiMapper
    from pipeline.settings import Settings
    s = Settings.load(cache_dir=tmp_path / "c", out_dir=tmp_path / "o")
    s.config["funds"]["list"] = [{"cik": "0002026053", "name": "Pershing Square", "expect": "PERSHING"}]
    monkeypatch.setattr(jobs.sec, "submissions", lambda http, cik: {
        "name": "PERSHING SQUARE HOLDCO, L.P.", "filings": {"recent": {}}})
    monkeypatch.setattr(jobs.sec, "recent_filings", lambda sub, forms: [
        {"accessionNumber": "ACC1", "form": "13F-HR", "filingDate": "2026-08-14", "reportDate": "2026-06-30"}]
        if "13F-HR" in forms else [])
    monkeypatch.setattr(jobs, "load_13f", lambda *a: {
        "accession": "ACC1", "period": "2026-06-30", "filed": "2026-08-14", "n_positions": 10,
        "total_long_value": 1e9, "positions": {"X": {"cusip": "X", "issuer": "X CORP", "value": 1e8, "shares": 1}}})
    http = HttpClient("t", transport=transport_from([]))
    state = empty_state()
    n, note = jobs.job_funds(http, s, state, jobs.Budget(600), [], FigiMapper(http, s.cache_dir))
    assert n == 0 and not state["events"]
    st = state["funds"]["0002026053"]
    assert st["processed_accession"] == "ACC1" and "next quarterly 13F" in st["info"] and not st["error"]


# ---------------------------------------------------------------- M6: leadership

LEGISLATORS = [
    {"name": {"first": "Mike", "last": "Johnson", "official_full": "Mike Johnson"}, "id": {"bioguide": "J000299"},
     "terms": [{"type": "rep", "state": "LA", "district": 4, "party": "Republican"}],
     "leadership_roles": [{"title": "Speaker of the House", "chamber": "house", "start": "2025-01-03"}]},
    {"name": {"first": "Dusty", "last": "Johnson", "official_full": "Dusty Johnson"}, "id": {"bioguide": "J000301"},
     "terms": [{"type": "rep", "state": "SD", "district": 0, "party": "Republican"}]},
    {"name": {"first": "Ron", "last": "Johnson", "official_full": "Ron Johnson"}, "id": {"bioguide": "J000293"},
     "terms": [{"type": "sen", "state": "WI", "party": "Republican"}]},
    {"name": {"first": "Yvette", "last": "Clarke", "official_full": "Yvette D. Clarke"}, "id": {"bioguide": "C001067"},
     "terms": [{"type": "rep", "state": "NY", "district": 9, "party": "Democrat"}]},
    {"name": {"first": "Mark", "last": "Warner", "official_full": "Mark R. Warner"}, "id": {"bioguide": "W000805"},
     "terms": [{"type": "sen", "state": "VA", "party": "Democrat"}],
     "leadership_roles": [{"title": "Senate Democratic Conference Vice Chair", "chamber": "senate", "start": "2025-01-03"}]},
]


def test_leadership_comes_from_ids_not_surnames():
    http = HttpClient("t", transport=transport_from([("congress-legislators", 200, LEGISLATORS)]))
    d = congress.load_legislators(http)
    assert d["leaders"] == {"J000299": "Speaker of the House"}           # vice chair isn't top leadership
    speaker = congress.member_actor("house", "Mike", "Johnson", "LA04", d)
    dusty = congress.member_actor("house", "Dusty", "Johnson", "SD00", d)
    ron = congress.member_actor("senate", "Ron", "Johnson", None, d)
    clarke = congress.member_actor("house", "Yvette", "Clarke", "NY09", d)
    assert speaker["leadership"] == "Speaker of the House"
    assert not dusty["leadership"] and not ron["leadership"] and not clarke["leadership"]
    # same surname, no state and an unmatched first name → don't guess
    assert congress.match_member(d, "Johnson", "Hank", "house", None) is None


# ---------------------------------------------------------------- M8: Senate bot walls and ids

def test_efd_bot_check_pages_are_detected():
    page = '<form><input name="csrfmiddlewaretoken" value="tok"></form>'
    http = HttpClient("t", transport=transport_from([
        ("/search/report/data/", 200, "<html>Please verify you are a human</html>"),
        ("/search/home/", 200, page)]))
    with pytest.raises(congress.EfdBlocked):
        congress.efd_recent_ptrs(http)
    with pytest.raises(congress.EfdBlocked):
        congress.parse_efd_ptr_html("<html><head><title>Access Denied</title></head><body>Transaction Date</body></html>")
    with pytest.raises(congress.EfdBlocked):     # session expired → bounced to the agreement page
        congress.parse_efd_ptr_html("<html><head><title>eFD: Home</title></head><body>I understand</body></html>")


def test_senate_ids_match_across_official_and_fallback_sources():
    official = {"trade_date": "2026-10-01", "ticker": "AAPL", "action": "buy", "owner": "Spouse", "lo": 1001.0}
    fallback = {"trade_date": "2026-10-01", "ticker": "AAPL", "action": "buy", "owner": "SP", "lo": 1001}
    other_owner = dict(official, owner="Self")
    assert jobs.senate_key("Tuberville", official) == jobs.senate_key("Tuberville", fallback)
    assert jobs.senate_key("Tuberville", official) != jobs.senate_key("Tuberville", other_owner)


def test_every_congress_buy_is_kept_for_win_rates(tmp_path):
    from pipeline.settings import Settings
    s = Settings.load(cache_dir=tmp_path / "c", out_dir=tmp_path / "o")
    ev = {"id": "house:1:0", "actor": {"id": "congress:a", "name": "Rep. A", "role": "House"}, "ticker": "AAPL",
          "action": "buy", "trade_date": "2026-10-01", "filed_at": "2026-10-20", "value_range": [1001, 15000]}
    jobs.remember_congress(s, [ev, dict(ev, id="house:1:1", action="sell")])
    jobs.remember_congress(s, [ev])                                    # no duplicates
    from pipeline.util import read_json
    rows = read_json(s.heavy_dir / "congress_hist.json.gz")
    assert [r["id"] for r in rows] == ["house:1:0"] and (s.heavy_dir / ".dirty").exists()


# ---------------------------------------------------------------- M12: errors are reported, retried sooner

def test_daily_index_missing_vs_blocked():
    missing = HttpClient("t", transport=transport_from([("daily-index", 404, "Not Found")]))
    assert sec.daily_master_index(missing, dt.date(2026, 11, 26)) is None
    blocked = HttpClient("t", transport=transport_from([("daily-index", 403, "Your Request Originates from an Undeclared Automated Tool")]))
    with pytest.raises(HttpError):
        sec.daily_master_index(blocked, dt.date(2026, 10, 29))


def test_failed_slow_jobs_retry_sooner():
    state = empty_state()
    job_record(state, "track_records", False, error="boom")
    state["jobs"]["track_records"]["last_run"] = "2026-10-30T20:00:00+00:00"   # 30 min ago
    assert job_due(state, "track_records", 1440)                              # 15-min retry, not 24h
    job_record(state, "track_records", True)
    state["jobs"]["track_records"]["last_run"] = "2026-10-30T20:00:00+00:00"
    assert not job_due(state, "track_records", 1440)


def test_retries_stop_when_the_run_is_out_of_time():
    t = transport_from([("example.com", 503, "busy")])
    http = HttpClient("t", transport=t, deadline=time.monotonic() + 5)
    with pytest.raises(HttpError):
        http.get("https://example.com/x")
    assert len(t.calls) == 1


# ---------------------------------------------------------------- M9: frozen price context

def test_price_context_is_frozen_at_first_sight():
    e = {"id": "x", "ticker": "ACME"}
    at_filing = {"price": 10.0, "lo_52w": 9.8, "hi_52w": 30.0, "ret_3m": -25.0, "risk_level": 3, "as_of": "t0"}
    q1 = scoring.frozen_quote(e, at_filing)
    later = {"price": 29.0, "lo_52w": 9.8, "hi_52w": 30.0, "ret_3m": 150.0, "risk_level": 4, "as_of": "t1"}
    q2 = scoring.frozen_quote(e, later)
    assert q1["price"] == q2["price"] == 10.0 and q2["ret_3m"] == -25.0
    assert q2["risk_level"] == 4                     # risk itself stays current
    pros, cons = [], []
    assert scoring._price_context(q2, pros, cons, True) > 0 and any("52-week low" in p for p in pros)


# ---------------------------------------------------------------- M13: tickers mapped later

def test_fund_tickers_are_filled_in_once_openfigi_answers(monkeypatch, tmp_path):
    from pipeline.figi import FigiMapper
    from pipeline.settings import Settings
    s = Settings.load(cache_dir=tmp_path / "c", out_dir=tmp_path / "o")
    s.config["funds"]["list"] = []
    state = empty_state()
    state["events"] = [{"id": "f13:A:037833100", "kind": "fund", "ticker": None, "details": {"cusip": "037833100"}}]
    http = HttpClient("t", transport=transport_from([("openfigi", 200, [{"data": [
        {"ticker": "AAPL", "name": "APPLE INC", "securityType": "Common Stock"}]}])]))
    jobs.job_funds(http, s, state, jobs.Budget(60), [], FigiMapper(http, s.cache_dir))
    assert state["events"][0]["ticker"] == "AAPL"


# ---------------------------------------------------------------- backstop never skips a business day

def test_backstop_walks_forward_through_missed_days(monkeypatch):
    monkeypatch.setenv("SMR_FAKE_NOW", "2026-10-30T14:00:00+00:00")     # Friday 10am ET
    state = empty_state()
    assert jobs.backstop_next_day(state) == dt.date(2026, 10, 29)       # first run: yesterday
    state["backstop_done"] = "2026-10-26"                               # Monday done, Tue+Wed missed
    assert jobs.backstop_next_day(state) == dt.date(2026, 10, 27)
    state["backstop_done"] = "2026-10-28"
    assert jobs.backstop_next_day(state) == dt.date(2026, 10, 29)
    state["backstop_done"] = "2026-10-29"
    assert jobs.backstop_next_day(state) is None
    state["backstop_pending"] = "2026-10-29"                            # half-done day comes first
    assert jobs.backstop_next_day(state) == dt.date(2026, 10, 29)
    monkeypatch.setenv("SMR_FAKE_NOW", "2026-11-02T14:00:00+00:00")     # Monday: Friday is next
    state = dict(empty_state(), backstop_done="2026-10-29")
    assert jobs.backstop_next_day(state) == dt.date(2026, 10, 30)


def test_quiver_names_with_suffixes_keep_the_surname():
    http = HttpClient("t", transport=transport_from([("quiverquant", 200, [
        {"Representative": "John Smith Jr.", "House": "Senate", "TransactionDate": "2026-10-01", "ReportDate": "2026-10-20",
         "Ticker": "AAPL", "Transaction": "Purchase", "Range": "$1,001 - $15,000"}])]))
    rows = congress.quiver_latest(http, "key")
    assert rows[0]["last"] == "Smith" and rows[0]["first"] == "John"


def test_workflows_share_identical_cache_paths():
    """actions/cache only finds a cache if the path list is identical; the analyst reads the refresh's state cache."""
    import yaml
    from pathlib import Path
    wf = Path(__file__).resolve().parents[1] / ".github" / "workflows"
    refresh = yaml.safe_load((wf / "refresh.yml").read_text())
    analyst = yaml.safe_load((wf / "analyst.yml").read_text())

    def paths(doc, prefix):
        out = []
        for st in doc["jobs"][next(iter(doc["jobs"]))]["steps"]:
            w = st.get("with") or {}
            key = str(w.get("key", "")) + str(w.get("restore-keys", ""))
            if "actions/cache" in st.get("uses", "") and prefix in key:
                out.append(tuple(x.strip() for x in str(w["path"]).strip().splitlines()))
        return out
    state_paths = paths(refresh, "smr-state-") + paths(analyst, "smr-state-")
    ai_paths = paths(refresh, "smr-ai-") + paths(analyst, "smr-ai-")
    assert len(state_paths) == 3 and len(set(state_paths)) == 1, state_paths       # restore + save + analyst restore
    assert state_paths[0] == (".cache/*", "!.cache/heavy", "!.cache/ai")
    assert len(ai_paths) == 3 and set(ai_paths) == {(".cache/ai",)}
    # every action is a Node 24 major (GitHub removed Node 20 from runners on 16 Sep 2026)
    for doc in (refresh, analyst):
        for st in doc["jobs"][next(iter(doc["jobs"]))]["steps"]:
            u = st.get("uses", "")
            if u.startswith("actions/"):
                major = int(u.split("@v")[1])
                assert major >= {"checkout": 5, "setup-python": 6, "configure-pages": 6, "upload-pages-artifact": 5,
                                 "deploy-pages": 5}.get(u.split("/")[1].split("@")[0], 5), u
