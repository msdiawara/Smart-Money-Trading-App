"""News: company-name cleaning, GDELT queries/answers (including its plain-text errors), 8-K items and flags."""
import datetime as dt
import json

from pipeline import news as N
from pipeline import conviction as C


def test_clean_name_and_query():
    assert N.clean_name("NVIDIA CORP") == "Nvidia"
    assert N.clean_name("Alphabet Inc.") == "Alphabet"
    assert N.clean_name("Advanced Micro Devices, Inc.") == "Advanced Micro Devices"
    assert N.clean_name("BERKSHIRE HATHAWAY INC /DE/") == "Berkshire Hathaway"
    assert N.clean_name("AMD") == "AMD"
    q = N.gdelt_query("NVIDIA CORP", "NVDA")
    assert '"Nvidia"' in q and '"NVDA"' in q and "sourcelang:english" in q and "(stock OR" in q
    assert "(stock OR" not in N.gdelt_query("Advanced Micro Devices Inc", "AMD")       # specific enough already
    assert '"GE"' not in (N.gdelt_query("General Electric Co", "GE") or "")            # too-short tickers left out
    assert N.gdelt_query(None, "X") is None
    assert "maxrecords=40" in N.gdelt_url(q) and "timespan=7d" in N.gdelt_url(q)


def test_parse_gdelt_ranks_reputable_dedupes_and_reads_errors():
    arts = {"articles": [
        {"url": "https://randomblog.example/a", "title": "Nvidia stock soars", "seendate": "20261030T150000Z", "domain": "randomblog.example"},
        {"url": "https://www.reuters.com/x", "title": "Nvidia beats estimates", "seendate": "20261030T120000Z", "domain": "reuters.com"},
        {"url": "https://www.cnbc.com/y", "title": "Nvidia beats estimates!", "seendate": "20261030T110000Z", "domain": "cnbc.com"},
        {"url": "https://www.wsj.com/z", "title": "Chip stocks slide", "seendate": "20261029T090000Z", "domain": "wsj.com"},
        {"url": "https://apnews.com/w", "title": "AI spending questioned", "seendate": "20261028T090000Z", "domain": "apnews.com"},
        {"url": "http://insecure.example/q", "title": "insecure", "seendate": "20261030T090000Z", "domain": "insecure.example"},
    ]}
    items, err = N.parse_gdelt(json.dumps(arts))
    assert err is None
    assert [i["domain"] for i in items] == ["reuters.com", "wsj.com", "apnews.com"]          # dedupe + reputable only
    assert items[0]["seen"] == "2026-10-30T12:00:00Z"
    assert N.parse_gdelt("Please limit requests to one every 5 seconds")[1].startswith("Please limit")
    assert N.parse_gdelt("{}") == ([], None)
    few = {"articles": [{"url": "https://randomblog.example/a", "title": "A", "seendate": "20261030T150000Z", "domain": "randomblog.example"}]}
    assert len(N.parse_gdelt(json.dumps(few))[0]) == 1                                   # better something than nothing


def test_8k_items_flags_and_earnings_dates():
    rec = {"cik": "0001234567", "sub": {"k8": [
        {"d": "2026-10-29", "acc": "0001234567-26-000050", "items": ["2.02", "9.01"], "form": "8-K"},
        {"d": "2026-09-15", "acc": "0001234567-26-000040", "items": ["4.02"], "form": "8-K"},
        {"d": "2026-01-10", "acc": "0001234567-26-000001", "items": ["3.01"], "form": "8-K"},
        {"d": "2026-07-29", "acc": "0001234567-26-000030", "items": ["2.02", "9.01"], "form": "8-K"}]}}
    live = [{"cik": "0001234567", "acc": "0001234567-26-000060", "items": ["5.02"], "at": "2026-10-30T16:10:00-04:00", "form": "8-K"},
            {"cik": "0009999999", "acc": "x", "items": ["1.03"], "at": "2026-10-30T16:00:00-04:00"}]
    fl = N.filings_for(rec, live, "0001234567")
    assert fl[0]["acc"] == "0001234567-26-000060" and fl[0]["items"][0]["label"].startswith("Director or officer")
    q = next(f for f in fl if f["acc"].endswith("50"))
    assert q["title"] == "Quarterly results" and all(i["item"] != "9.01" for i in q["items"])
    assert q["url"].endswith("/000123456726000050/0001234567-26-000050-index.htm")
    red = N.red_flags(fl, today=dt.date(2026, 10, 30))
    assert [r["item"] for r in red] == ["4.02"]                        # the Jan delisting notice is older than 90 days
    assert N.earnings_dates(rec["sub"]["k8"]) == ["2026-07-29", "2026-10-29"]
    assert N.items_from_summary("<b>Filed:</b> 2026-10-30 Item 2.02: Results&lt;br&gt;Item 9.01: Exhibits") == ["2.02", "9.01"]


def test_conviction_needs_every_check():
    good = dict(quote={"risk_level": 3, "adv_usd": 5e7}, val={"label": "Fairly valued"}, bub={"level": "Moderate"},
                flags=[], claude={"stance": "favorable", "probability": 0.62, "confidence": "medium", "base": 0.5},
                support={"what": "CEO purchase", "edge": 0.03})
    r = C.rate("X", **good)
    assert r["rating"] == "top" and r["passed"] == r["of"] == 6
    assert C.rate("X", **{**good, "val": {"label": "Very expensive"}})["rating"] == "caution"
    assert C.rate("X", **{**good, "claude": {**good["claude"], "probability": 0.55}})["rating"] == "positive"
    assert C.rate("X", **{**good, "claude": None, "support": None})["rating"] == "neutral"
    assert C.rate("X", **{**good, "flags": [{"item": "4.02", "label": "Restatement", "date": "2026-10-01"}]})["rating"] == "red"
    hi_base = {**good, "claude": {**good["claude"], "base": 0.56}}       # 62% isn't 8 points above a 56% base rate
    assert C.rate("X", **hi_base)["rating"] == "positive"
    assert C.rate("X", **{**good, "quote": {"risk_level": 3, "adv_usd": 1e6}})["rating"] != "top"
