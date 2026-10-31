"""Valuation & bubble engine: TTM arithmetic from SEC-style facts, multiples, reverse DCF, relative valuation,
bubble checklist and the AI theme gauge."""
import datetime as dt
import math

import numpy as np
import pytest

from pipeline import valuation as V


def fact(start, end, val, filed, form="10-Q", accn="0000000000-26-000001"):
    d = {"end": end, "val": val, "filed": filed, "form": form, "accn": accn}
    if start:
        d["start"] = start
    return d


def companyfacts_calendar():
    """Calendar-year company: FY2025 revenue 1,000; Q1-Q3 2025 = 230/240/250; Q1-Q3 2026 = 300/320/340."""
    rev = [
        fact("2024-01-01", "2024-12-31", 800, "2025-02-20", "10-K"),
        fact("2025-01-01", "2025-12-31", 1000, "2026-02-20", "10-K"),
        fact("2025-01-01", "2025-03-31", 230, "2025-05-01"), fact("2025-01-01", "2025-06-30", 470, "2025-08-01"),
        fact("2025-01-01", "2025-09-30", 720, "2025-11-01"),
        fact("2026-01-01", "2026-03-31", 300, "2026-05-01"), fact("2026-01-01", "2026-06-30", 620, "2026-08-01"),
        fact("2026-01-01", "2026-09-30", 960, "2026-11-01"),
        fact("2025-07-01", "2025-09-30", 250, "2025-11-01"), fact("2026-07-01", "2026-09-30", 340, "2026-11-01"),
    ]
    ni = [fact("2025-01-01", "2025-12-31", 100, "2026-02-20", "10-K"), fact("2025-01-01", "2025-09-30", 70, "2025-11-01"),
          fact("2026-01-01", "2026-09-30", 110, "2026-11-01")]
    return {"facts": {
        "us-gaap": {
            "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": rev}},
            "NetIncomeLoss": {"units": {"USD": ni}},
            "OperatingIncomeLoss": {"units": {"USD": [fact("2025-01-01", "2025-12-31", 150, "2026-02-20", "10-K"),
                                                      fact("2025-01-01", "2025-09-30", 100, "2025-11-01"),
                                                      fact("2026-01-01", "2026-09-30", 160, "2026-11-01")]}},
            "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [
                fact("2025-01-01", "2025-12-31", 180, "2026-02-20", "10-K"), fact("2025-01-01", "2025-09-30", 120, "2025-11-01"),
                fact("2026-01-01", "2026-09-30", 150, "2026-11-01")]}},
            "PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": [
                fact("2025-01-01", "2025-12-31", 40, "2026-02-20", "10-K"), fact("2025-01-01", "2025-09-30", 30, "2025-11-01"),
                fact("2026-01-01", "2026-09-30", 45, "2026-11-01")]}},
            "CashAndCashEquivalentsAtCarryingValue": {"units": {"USD": [fact(None, "2026-09-30", 200, "2026-11-01")]}},
            "LongTermDebt": {"units": {"USD": [fact(None, "2026-09-30", 300, "2026-11-01")]}},
            "StockholdersEquity": {"units": {"USD": [fact(None, "2026-09-30", 500, "2026-11-01")]}},
            "WeightedAverageNumberOfDilutedSharesOutstanding": {"units": {"shares": [
                fact("2026-07-01", "2026-09-30", 102, "2026-11-01")]}},
        },
        "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
            fact(None, "2026-10-25", 60, "2026-11-01", accn="A2"), fact(None, "2026-10-25", 40, "2026-11-01", accn="A2"),
            fact(None, "2025-10-24", 90, "2025-11-01", accn="A1")]}}},
    }}


def test_ttm_year_to_date_method_and_growth():
    m = V.extract_metrics(companyfacts_calendar())
    assert m["rev"] == pytest.approx(1000 + 960 - 720)            # 1,240
    assert m["rev_method"] == "year-to-date" and m["rev_end"] == "2026-09-30"
    # the TTM a year earlier needs Jan-Sep 2024, which isn't filed here: no growth figure rather than a wrong one
    assert m["rev_prev"] is None
    js = companyfacts_calendar()
    js["facts"]["us-gaap"]["RevenueFromContractWithCustomerExcludingAssessedTax"]["units"]["USD"].append(
        fact("2024-01-01", "2024-09-30", 600, "2024-11-01"))
    m2 = V.extract_metrics(js)
    assert m2["rev_prev"] == pytest.approx(800 + 720 - 600)        # 920 → growth 1240/920 − 1
    assert V.fundamentals_view(m2, 10.0)["growth"] == pytest.approx(1240 / 920 - 1)
    assert m["ni"] == pytest.approx(100 + 110 - 70)
    assert m["oi"] == pytest.approx(150 + 160 - 100)
    assert m["ocf"] - m["capex"] == pytest.approx((180 + 150 - 120) - (40 + 45 - 30))
    assert m["cash"] == 200 and m["debt"] == 300 and m["equity"] == 500
    assert m["shares"] == 100                                      # two share classes on the cover page, added
    assert m["shares_prev"] == 90


def test_ttm_restatement_latest_filing_wins():
    per = V.merge_duration({"Revenues": [fact("2025-01-01", "2025-12-31", 100, "2026-02-01", "10-K"),
                                         fact("2025-01-01", "2025-12-31", 90, "2026-08-01", "10-K/A")]})
    assert V.ttm(per)["value"] == 90


def test_ttm_four_quarters_fallback_and_stale_annual():
    q = [fact("2025-10-01", "2025-12-31", 10, "2026-02-01"), fact("2026-01-01", "2026-03-31", 11, "2026-05-01"),
         fact("2026-04-01", "2026-06-30", 12, "2026-08-01"), fact("2026-07-01", "2026-09-30", 13, "2026-11-01")]
    r = V.ttm(V.merge_duration({"Revenues": q}))
    assert r["value"] == 46 and r["method"] == "four quarters"
    old = V.merge_duration({"Revenues": [fact("2023-01-01", "2023-12-31", 50, "2024-02-01", "10-K")]})
    assert V.ttm(old, as_of=dt.date(2026, 10, 30)) is None       # too old to call "trailing"


def test_52_53_week_fiscal_year():
    rev = [fact("2024-09-29", "2025-09-27", 400, "2025-11-01", "10-K"),
           fact("2025-09-28", "2025-12-27", 120, "2026-01-30"), fact("2024-09-29", "2024-12-28", 100, "2025-01-30"),
           fact("2023-10-01", "2024-09-28", 380, "2024-11-01", "10-K")]
    r = V.ttm(V.merge_duration({"Revenues": rev}))
    assert r["value"] == 420 and r["method"] == "year-to-date"


def test_foreign_currency_is_skipped():
    js = {"facts": {"ifrs-full": {"Revenue": {"units": {"TWD": [fact("2025-01-01", "2025-12-31", 1e12, "2026-04-01", "20-F")]}}}}}
    m = V.extract_metrics(js)
    assert m["currency"] == "TWD" and "rev" not in m and "USD" in m["note"]


def test_share_class_sanity_falls_back_to_diluted():
    js = companyfacts_calendar()
    js["facts"]["dei"]["EntityCommonStockSharesOutstanding"]["units"]["shares"] = [
        fact(None, "2026-10-25", 30, "2026-11-01", accn="A2")]          # only one class reported: far too few
    m = V.extract_metrics(js)
    assert m["shares"] == 102


def test_multiples_and_reverse_dcf():
    m = V.extract_metrics(companyfacts_calendar())
    v = V.fundamentals_view(m, 20.0)
    assert v["mcap"] == 2000 and v["ev"] == 2000 + 300 - 200
    assert v["ps"] == pytest.approx(2000 / 1240)
    assert v["pe"] == pytest.approx(2000 / 140)
    d = V.reverse_dcf(1000, 50, r=0.09, g_term=0.03)
    assert d["growth"] is not None and 0 < d["growth"] < 0.2
    # check the solver: plugging the growth back in reproduces the value
    g, f, tot = d["growth"], 50, 0.0
    for t in range(1, 11):
        f *= 1 + g
        tot += f / 1.09 ** t
    tot += f * 1.03 / 0.06 / 1.09 ** 10
    assert tot == pytest.approx(1000, rel=1e-6)
    assert V.reverse_dcf(1e9, 1, 0.09, 0.03)["growth"] is None         # priced beyond 100%/yr
    assert V.reverse_dcf(1000, -5) is None


def _universe(n=400, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        g = float(rng.uniform(-0.1, 0.6))
        om = float(rng.uniform(-0.1, 0.4))
        fair = math.exp(0.5 + 2.5 * g + 3.0 * om)
        rows.append({"t": f"T{i}", "sic": 7372 if i % 2 else 3674, "ev_s": fair * math.exp(rng.normal(0, 0.3)),
                     "growth": g, "op_margin": om, "gross_margin": 0.6})
    return rows


def test_relative_valuation_flags_the_planted_outlier():
    rows = _universe()
    rich = {"t": "RICH", "sic": 7372, "growth": 0.2, "op_margin": 0.1, "gross_margin": 0.6}
    rich["ev_s"] = math.exp(0.5 + 2.5 * 0.2 + 3.0 * 0.1) * 4          # 4× its "fair" multiple
    cheap = {**rich, "t": "CHEAP", "ev_s": rich["ev_s"] / 16}           # ¼ of fair
    fit = V.fit_relative(rows + [rich, cheap])
    assert fit and fit["r2"] > 0.6
    resid = np.asarray(fit["resid"])
    e_rich = math.log(rich["ev_s"]) - math.log(V.predict_fair(fit, rich))
    e_cheap = math.log(cheap["ev_s"]) - math.log(V.predict_fair(fit, cheap))
    assert V.label_for(V._pctl(resid, e_rich)) == "Very expensive"
    assert V.label_for(V._pctl(resid, e_cheap)) == "Very cheap"


def test_value_all_end_to_end_labels_and_financials():
    companies, quotes = {}, {}
    rng = np.random.default_rng(3)
    for i in range(260):
        g = float(rng.uniform(0, 0.5))
        om = float(rng.uniform(0.0, 0.3))
        rev = 1000.0
        evs = math.exp(0.3 + 2 * g + 3 * om + rng.normal(0, 0.25))
        t = f"S{i}"
        companies[str(i)] = {"t": t, "sub": {"sic": 3674 if i % 3 else 2834, "sic_desc": "Semis"},
                             "m": {"rev": rev, "rev_prev": rev / (1 + g), "oi": rev * om, "ni": rev * om * 0.8,
                                   "gp": rev * 0.5, "shares": 100.0, "cash": 0.0, "debt": 0.0, "equity": 500.0,
                                   "ocf": rev * om, "capex": 10.0}}
        quotes[t] = {"price": evs * rev / 100.0}
    for i in range(30):
        t = f"B{i}"
        companies[f"b{i}"] = {"t": t, "sub": {"sic": 6022}, "m": {"rev": 500.0, "ni": 50.0, "shares": 10.0,
                                                                 "equity": 400.0 * (1 + i / 30), "cash": 0, "debt": 0}}
        quotes[t] = {"price": 50.0}
    out = V.value_all(companies, quotes)
    bt = out["by_ticker"]
    assert out["meta"]["fit_n"] >= 200 and len(bt) == 290
    labels = {v["label"] for v in bt.values()}
    assert {"Very cheap", "Fairly valued", "Very expensive"} <= labels
    assert bt["B0"]["basis"].startswith("P/E and P/B")
    assert all(0 <= v["rich_pctl"] <= 100 for v in bt.values() if v["rich_pctl"] is not None)


def test_submissions_parse_items_age_and_offerings():
    sub = {"name": "ACME", "sic": "3674", "sicDescription": "Semiconductors", "tickers": ["ACME"],
           "filings": {"recent": {
               "accessionNumber": ["a1", "a2", "a3", "a4", "a5"],
               "form": ["8-K", "10-Q", "424B5", "S-3", "8-K"],
               "filingDate": ["2026-10-29", "2026-10-28", "2026-06-01", "2026-05-01", "2026-04-30"],
               "items": ["2.02,9.01", "", "", "", "5.02"], "acceptanceDateTime": ["2026-10-29T16:05:00.000Z"] * 5,
               "primaryDocument": ["x.htm"] * 5, "reportDate": ["", "2026-09-30", "", "", ""]},
               "files": [{"name": "CIK-001.json", "filingFrom": "2004-03-01", "filingTo": "2015-01-01"}]}}
    p = V.parse_submissions(sub, today=dt.date(2026, 10, 30))
    assert p["sic"] == 3674 and p["first_filed"] == "2004-03-01"
    assert p["k8"][0]["items"] == ["2.02", "9.01"] and p["periodic"]["form"] == "10-Q"
    assert p["offerings_12m"] == 2


def _path(n, daily, vol=0.01, seed=0, start=10.0):
    rng = np.random.default_rng(seed)
    r = daily + rng.normal(0, vol, n)
    return start * np.exp(np.cumsum(r))


def test_bubble_checklist_levels():
    spy = _path(520, 0.0003, 0.008, seed=1, start=500)
    hot = _path(520, 0.0035, 0.025, seed=2)          # ~+400% in 2 years
    calm = 10 * (spy / spy[0]) * np.exp(np.cumsum(np.random.default_rng(3).normal(0, 0.004, 520)))
    st_hot = V.price_path_stats(hot, spy)
    st_calm = V.price_path_stats(calm, spy)
    assert st_hot["net_2y"] > 1.5 and abs(st_calm["net_2y"]) < 0.6
    b_hot = V.bubble_risk(st_hot, {"rich_pctl": 95, "ps": 40, "dilution": 0.12}, {"offerings_12m": 2},
                          {"sellers": 4, "buyers": 0}, {"level": "core", "label": "Core AI"}, "High")
    b_calm = V.bubble_risk(st_calm, {"rich_pctl": 40}, {}, {}, {"level": "none"}, "High")
    assert b_hot["level"] == "High" and b_calm["level"] in ("Low", "Moderate")
    assert any(s["title"] == "Issuing stock" for s in b_hot["signs"])
    assert b_hot["gsy_crash_2y"] == pytest.approx(0.80)
    assert V.gsy_crash(0.4) is None and V.gsy_crash(1.0) == pytest.approx(0.53)
    assert V.gsy_crash(0.875) == pytest.approx((0.36 + 0.53) / 2)


class _P:
    """Minimal stand-in for quant.Panel."""
    def __init__(self, cols: dict, T: int):
        self.T = T
        self.col = {t: i for i, t in enumerate(cols)}
        self.close = np.column_stack([cols[t] for t in cols])
        self.dates = np.arange(738000, 738000 + T)

    def c(self, t):
        j = self.col.get(t)
        return None if j is None else self.close[:, j]

    def date(self, i):
        return dt.date.fromordinal(int(self.dates[i])).isoformat()


def test_ai_theme_and_factor_loadings():
    T = 520
    spy = _path(T, 0.0003, 0.008, seed=1, start=500)
    rng = np.random.default_rng(9)
    ai_f = np.cumsum(rng.normal(0.002, 0.012, T))
    cols = {"SPY": spy}
    for i in range(8):
        cols[f"AI{i}"] = 20 * np.exp(np.log(spy / spy[0]) + ai_f + np.cumsum(rng.normal(0, 0.01, T)))
    cols["UTIL"] = 30 * np.exp(np.log(spy / spy[0]) + 0.8 * ai_f + np.cumsum(rng.normal(0, 0.006, T)))
    cols["BORING"] = 40 * np.exp(np.log(spy / spy[0]) + np.cumsum(rng.normal(0, 0.01, T)))
    P = _P(cols, T)
    basket = [f"AI{i}" for i in range(8)]
    th = V.ai_theme(P, basket)
    assert th and th["n"] == 8 and th["stats"]["net_2y"] > 0.5 and th["level"] in ("Moderate", "Elevated", "High")
    lo = V.ai_factor_loadings(P, basket, ["UTIL", "BORING"])
    assert lo["UTIL"]["beta"] > 0.5 and lo["UTIL"]["t"] > 3
    assert abs(lo["BORING"]["beta"]) < 0.3
    ex = V.ai_exposure("UTIL", 4911, {"core": ["AI0"]}, lo["UTIL"])
    assert ex["level"] == "linked"
    assert V.ai_exposure("AI0", None, {"core": ["AI0"]})["level"] == "core"
    assert V.ai_exposure("XYZ", 7372, {})["level"] == "sector"
