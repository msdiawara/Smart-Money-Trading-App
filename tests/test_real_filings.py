"""Parsers run against REAL documents (public SEC / House / Senate records collected from public test suites;
see fixtures/real/SOURCES.md in the project archive for exact origins)."""
import datetime as dt
from pathlib import Path

import pytest

from pipeline import congress, f13, form4, sched13, sec

R = Path(__file__).parent / "fixtures" / "real"


def txt(name):
    return (R / name).read_text(encoding="utf-8", errors="replace")


def test_form4_x0609_checkbox_true_and_sale_value():
    p = form4.parse_form4(txt("aapl_form4_2026-10-01_X0609_M-F-S-G_aff10b5One-true.xml"))
    assert p["plan_10b5_1"] is True and p["issuer"]["ticker"] == "AAPL"
    sale = next(e for e in form4.build_events(p, "a", None, "u", p["period"]) if e["action"] == "sell")
    assert sale["details"]["plan_10b5_1"] is True and sale["value"] == pytest.approx(63_845_841, rel=1e-4)
    other = next(e for e in form4.build_events(p, "a", None, "u", p["period"]) if e["action"] not in ("buy", "sell"))
    assert other["value"] is None  # tax withholding/gifts never show a misleading $ total


def test_form4_pre_checkbox_footnote_and_explicit_false():
    old = form4.parse_form4(txt("aapl_form4_2022-05-04_X0306_S_pre-checkbox-10b5-1-footnotes.xml"))
    assert old["plan_10b5_1"] is None and old["plan_in_footnotes"] is True
    ev = form4.build_events(old, "a", None, "u", old["period"])[0]
    assert ev["details"]["plan_10b5_1"] is True and ev["details"]["plan_flag_source"] == "footnote"
    assert "SVP" in ev["actor"]["role"]
    new = form4.parse_form4(txt("374water_form4_2025-04-30_X0508_aff10b5One-false.xml"))
    assert new["plan_10b5_1"] is False


def test_form4_exercise_and_sell_weighted_average():
    p = form4.parse_form4(txt("snow_form4_2022-12-13_X0306_M-S_weighted-avg-footnotes.xml"))
    sale = next(e for e in form4.build_events(p, "a", None, "u", p["period"]) if e["action"] == "sell")
    assert sale["details"]["exercise_linked"] is True and sale["shares"] == 200000
    assert any("weighted-average" in n.lower() for n in sale["details"]["footnotes"])


def test_13f_units_detected_from_prices_not_dates():
    late = f13.parse_info_table(txt("bullstreet_13F-HR_2022Q4_infotable_thousands-filed-after-cutover.xml"))
    assert f13.detect_multiplier(late, dt.date(2023, 2, 28)) == 1000      # filed after the rule change, still thousands
    early = f13.parse_info_table(txt("kahn_13F-HR_2022Q1_infotable.xml"))
    assert f13.detect_multiplier(early, dt.date(2022, 5, 13)) == 1        # filed before it, already dollars
    cover = f13.parse_cover(txt("mfn_13F-HRA_2025Q4_primary_doc.xml"))
    assert cover["is_amendment"] and cover["amendment_type"] == "NEW HOLDINGS" and cover["period"] == "2025-12-31"


@pytest.mark.parametrize("name", sorted(p.name for p in R.glob("schedule13*.xml")))
def test_schedule13_real_filings(name):
    i = sched13.parse_schedule13(txt(name))
    assert i["issuer_name"] and i["issuer_cik"] and i["event_date"]
    assert i["persons"] and i["max_percent"] is not None
    assert i["submission_type"].startswith("SCHEDULE 13")


def test_house_real_encrypted_pdf():
    t = congress.parse_house_ptr_text(congress.pdf_text((R / "house-ptr-20025000-encrypted.pdf").read_bytes()))
    assert [x["ticker"] for x in t] == ["COP", "HXL", "JPM", "LULU"]
    assert all(x["owner"] == "Joint" and (x["lo"], x["hi"]) == (15001, 50000) for x in t)
    assert [x["action"] for x in t] == ["buy", "sell", "buy", "sell"]
    assert t[2]["asset"] == "JP Morgan Chase & Co. Common Stock"  # name wrapped across two lines in the PDF


def test_senate_real_pages():
    rows = congress.parse_efd_ptr_html(txt("efd_ptr_0068462f-ee01-4550-98c9-b4437019d615_hoeven_2017-05-03.html"))
    assert len(rows) == 6 and rows[0]["ticker"] == "IBM" and (rows[0]["lo"], rows[0]["hi"]) == (50001, 100000)
    assert congress.parse_efd_ptr_html(txt("efd_paper_d145efef-1925-4ec2-b435-118c9a57a394_boozman_ptr_2020-02-07.html")) == []


def test_real_edgar_atom_feed():
    es = sec.parse_current_feed((R / "getcurrent_type4_atom.xml").read_bytes())
    assert len(es) == 100 and len({e.accession for e in es}) == 46
    assert {e.role for e in es} >= {"Reporting", "Issuer"} and all(len(e.cik) == 10 for e in es)
