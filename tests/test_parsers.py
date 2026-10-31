import datetime as dt
from pathlib import Path

import pytest

from pipeline import congress, f13, form4, sched13, sec
from pipeline.util import business_days_between, parse_date

FX = Path(__file__).parent / "fixtures"


def read(name):
    return (FX / name).read_text(encoding="utf-8")


# ---------------------------------------------------------------- Form 4

def test_form4_ceo_buy_aggregates_rows():
    p = form4.parse_form4(read("form4_ceo_buy.xml"))
    assert p["issuer"] == {"cik": "0001234567", "name": "Acme Robotics Inc", "ticker": "ACME"}
    assert p["plan_10b5_1"] is False
    evs = form4.build_events(p, "0001999001-26-000011", "2026-10-29T17:05:12-04:00", "u", "2026-10-29")
    assert len(evs) == 1
    e = evs[0]
    assert e["action"] == "buy" and e["actor"]["role"].startswith("CEO") and e["actor"]["rank"] == 5
    assert e["shares"] == 30000
    assert e["value"] == pytest.approx(20000 * 24.10 + 10000 * 24.70)
    assert e["price"] == pytest.approx(e["value"] / 30000)
    assert e["holdings_after"] == 80000
    assert e["holdings_change_pct"] == pytest.approx(30000 / 50000 * 100)  # 50k before → +60%
    assert e["trade_date"] == "2026-10-27" and e["trade_date_end"] == "2026-10-28"
    assert e["lag_days"] == 2 and e["details"]["late"] is False
    assert "weighted average" in e["details"]["footnotes"][0]


def test_form4_plan_sale_with_exercise_and_two_owners():
    p = form4.parse_form4(read("form4_plan_sale.xml"))
    assert p["plan_10b5_1"] is True and p["plan_in_footnotes"] is True
    evs = form4.build_events(p, "0001888002-26-000004", None, "u", "2026-10-28")
    kinds = {e["action"]: e for e in evs}
    sale = kinds["sell"]
    assert sale["details"]["plan_10b5_1"] is True
    assert sale["details"]["exercise_linked"] is True
    assert sale["details"]["indirect"] is True and sale["details"]["ownership_nature"] == "By Trust"
    assert "joint filer" in sale["actor"]["name"]
    assert sale["holdings_change_pct"] == pytest.approx(-5000 / 25000 * 100)
    other = [e for e in evs if e["action"] not in ("buy", "sell")]
    assert other and "Option exercise" in other[0]["action_label"]


def test_owner_roles():
    assert form4.owner_role({"is_officer": True, "officer_title": "President & CEO"})[1] == 5
    assert form4.owner_role({"is_director": True})[0] == "Director"
    r, k = form4.owner_role({"is_ten_pct": True})
    assert r == "10% owner" and k == 2


# ---------------------------------------------------------------- feeds & indexes

def test_current_feed_parsing_and_dedupe():
    entries = sec.parse_current_feed(read("current_feed.xml").encode("latin-1"))
    assert len(entries) == 4
    e = entries[0]
    assert (e.form, e.cik, e.role, e.accession) == ("4", "0001999001", "Reporting", "0001999001-26-000011")
    assert len({x.accession for x in entries if x.form in ("4", "4/A")}) == 2
    assert entries[3].form == "424B2"  # the feed's type filter is a prefix match → caller filters


def test_master_index_parsing(monkeypatch):
    class R:
        status = 200
        def text(self, enc="utf-8"):
            return read("master.idx")
    class H:
        def get(self, url, **kw):
            assert url.endswith("/2026/QTR4/master.20261029.idx")
            return R()
    rows = sec.daily_master_index(H(), dt.date(2026, 10, 29))
    assert len(rows) == 4
    assert rows[2]["form"] == "SCHEDULE 13D" and rows[2]["accession"] == "0000921895-26-002222"


def test_submission_text_extracts_xml_and_acceptance():
    txt = ("<SEC-DOCUMENT>0001999001-26-000011.txt : 20261029\n<ACCEPTANCE-DATETIME>20261029170512\n"
           "CONFORMED SUBMISSION TYPE:\t4\nFILED AS OF DATE:\t\t20261029\n"
           "<DOCUMENT>\n<TYPE>4\n<SEQUENCE>1\n<FILENAME>wk-form4.xml\n<TEXT>\n<XML>\n" + read("form4_ceo_buy.xml") +
           "\n</XML>\n</TEXT>\n</DOCUMENT>\n</SEC-DOCUMENT>")
    sub = sec.parse_submission_text("0001999001-26-000011", txt)
    assert sub.accepted_at().isoformat().startswith("2026-10-29T17:05:12")
    assert sub.header["FILED AS OF DATE"] == "20261029"
    x = sub.first_xml(root_contains="ownershipDocument")
    assert x and form4.parse_form4(x)["issuer"]["ticker"] == "ACME"


# ---------------------------------------------------------------- 13F

def test_13f_units_aggregate_and_diff():
    cur_rows = f13.parse_info_table(read("infotable_2026q2.xml"))
    prev_rows = f13.parse_info_table(read("infotable_2026q1.xml"))
    assert len(cur_rows) == 6 and cur_rows[0]["cusip"] == "037833100"
    assert f13.detect_multiplier(cur_rows, dt.date(2026, 8, 14)) == 1
    cur = f13.aggregate(cur_rows, 1)
    prev = f13.aggregate(prev_rows, 1)
    assert cur["060505104"]["shares"] == 495_000_000  # two rows (DFND + OTR) summed
    assert "78462F103:PUT" in cur and cur["78462F103:PUT"]["weight"] is None
    f13.rank_positions(cur)
    f13.rank_positions(prev)
    ch = {c["cusip"]: c for c in f13.diff(cur, prev)}
    assert ch["65249B109"]["kind"] == "new"
    assert ch["680223104"]["kind"] == "exit"
    assert ch["060505104"]["kind"] == "trim" and ch["060505104"]["pct_change"] == pytest.approx(-29.29, abs=0.01)
    assert ch["166764100"]["kind"] == "hold"
    sig = {c["cusip"] for c in f13.significant(list(ch.values()))}
    assert {"65249B109", "680223104", "060505104"} <= sig and "166764100" not in sig


def test_13f_thousands_detection():
    rows = f13.parse_info_table(read("infotable_thousands.xml"))
    assert f13.detect_multiplier(rows, dt.date(2022, 11, 14)) == 1000
    # a post-2023 filer that kept reporting thousands is caught by the implied-price check
    assert f13.detect_multiplier(rows, dt.date(2025, 5, 15)) == 1000


# ---------------------------------------------------------------- 13D/G

def test_schedule13d_fuzzy_parse():
    info = sched13.parse_schedule13(read("sched13d.xml"))
    assert info["issuer_name"] == "Acme Robotics Inc" and info["issuer_cik"] == "0001234567"
    assert info["event_date"] == "2026-10-21"
    assert info["max_percent"] == 8.7 and info["max_shares"] == 5_400_000
    assert len(info["persons"]) == 2 and info["persons"][0]["name"] == "Starboard Value LP"
    assert "undervalued" in info["purpose"]
    assert sched13.is_13d("SCHEDULE 13D/A") and sched13.is_13g("SC 13G")


# ---------------------------------------------------------------- Congress

def test_house_ptr_text_parsing():
    txs = congress.parse_house_ptr_text(read("house_ptr.txt"))
    assert len(txs) == 4
    nv, goog, msft, aapl = txs
    assert nv["ticker"] == "NVDA" and nv["owner"] == "Spouse" and nv["action"] == "buy"
    assert (nv["lo"], nv["hi"]) == (250001, 500000) and nv["asset_type"] == "Stock"
    assert goog["ticker"] == "GOOG" and goog["asset_type"] == "Stock option"
    assert goog["description"] and "call options" in goog["description"] and "1/15/2027" in goog["description"]
    assert (goog["lo"], goog["hi"]) == (1000001, 5000000)
    assert msft["action"] == "sell" and msft["action_label"] == "Sale (partial)" and msft["owner"] == "Joint"
    assert (msft["lo"], msft["hi"]) == (15001, 50000)
    assert aapl["ticker"] == "AAPL" and aapl["owner"] == "Self" and aapl["trade_date"] == "2026-09-29"


def test_senate_html_parsing():
    rows = congress.parse_efd_ptr_html(read("senate_ptr.html"))
    assert len(rows) == 2
    assert rows[0]["ticker"] == "AMCR" and rows[0]["action"] == "buy" and rows[0]["owner"] == "Spouse"
    assert rows[1]["ticker"] is None and rows[1]["action"] == "sell" and rows[1]["description"] == "Matured"


def test_amounts_and_actions():
    assert congress.parse_amount("$1,001 - $15,000") == (1001, 15000)
    assert congress.parse_amount("Over $50,000,000") == (50000000, None)
    assert congress.tx_action("Sale (Partial)") == ("sell", "Sale (partial)")
    assert congress.tx_action("P")[0] == "buy"


# ---------------------------------------------------------------- utils

def test_dates():
    assert parse_date("03-JAN-2023") == dt.date(2023, 1, 3)
    assert parse_date("10/28/2026") == dt.date(2026, 10, 28)
    assert parse_date("20261028") == dt.date(2026, 10, 28)
    assert business_days_between(dt.date(2026, 10, 30), dt.date(2026, 11, 3)) == 2  # Fri → Tue
