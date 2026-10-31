"""Write the JSON files the phone app reads (web/data/*.json)."""
from __future__ import annotations

from .settings import Settings
from .util import iso, now_utc, write_json

SOURCES = [
    {"key": "insiders_live", "name": "Insider trades (Form 4)", "origin": "SEC EDGAR live feed", "official": True,
     "freshness": "minutes after SEC accepts the filing (insiders have 2 business days to file)"},
    {"key": "sale_notices", "name": "Planned insider sales (Form 144)", "origin": "SEC EDGAR live feed", "official": True,
     "freshness": "minutes after filing; filed before the sale, usually days ahead of the Form 4"},
    {"key": "etf_daily", "name": "Active ETF daily trades (ARK)", "origin": "the funds' own daily holdings files",
     "official": True, "freshness": "daily; positions at the start of the day, i.e. yesterday's trades"},
    {"key": "daily_backstop", "name": "Missed-filing check", "origin": "SEC EDGAR daily index", "official": True,
     "freshness": "once a day for the previous business day"},
    {"key": "activists_live", "name": "Activist & 5%+ stakes (13D/13G)", "origin": "SEC EDGAR live feed", "official": True,
     "freshness": "minutes after filing (13D due within 5 business days)"},
    {"key": "funds", "name": "Hedge funds / superinvestors (13F)", "origin": "SEC EDGAR submissions API", "official": True,
     "freshness": "quarterly; holdings up to 45 days old when filed"},
    {"key": "congress_house", "name": "House trades (PTR)", "origin": "House Clerk disclosures", "official": True,
     "freshness": "checked every 30 min; members have up to 45 days to disclose"},
    {"key": "congress_senate", "name": "Senate trades (PTR)", "origin": "Senate eFD, else FMP/Quiver fallback",
     "official": True, "freshness": "hourly; up to 45-day disclosure lag"},
    {"key": "etoro", "name": "eToro Popular Investors", "origin": "eToro official Public API", "official": True,
     "freshness": "every 15 min (live portfolios)"},
    {"key": "market_history", "name": "Daily price history", "origin": "Massive (ex-Polygon) whole-market end-of-day",
     "official": False, "freshness": "nightly after midnight ET; ~2 years kept"},
    {"key": "splits", "name": "Stock-split adjustments", "origin": "Massive splits reference data", "official": False,
     "freshness": "twice a day; re-bases stored history so splits don't look like crashes"},
    {"key": "prices", "name": "Latest prices & risk", "origin": "Alpaca IEX real-time (indicative) or Yahoo (unofficial)",
     "official": False, "freshness": "every 15 min in market hours"},
    {"key": "reference", "name": "Company size & tickers", "origin": "SEC XBRL frames + ticker file", "official": True,
     "freshness": "daily"},
    {"key": "track_records", "name": "Win rates (computed)", "origin": "SEC insider data sets + prices", "official": True,
     "freshness": "daily rebuild (new quarters of SEC data arrive quarterly)"},
    {"key": "short_interest", "name": "Short interest", "origin": "FINRA (free, twice a month)", "official": True,
     "freshness": "positions as of mid-month and month-end, published ~7 business days later"},
    {"key": "earnings", "name": "Next earnings dates", "origin": "Alpha Vantage earnings calendar", "official": False,
     "freshness": "daily (needs a free Alpha Vantage key)"},
    {"key": "predictability", "name": "Predictability (base rates + tested model)",
     "origin": "SEC insider data sets + whole-market prices", "official": True, "freshness": "rebuilt daily"},
    {"key": "market_context", "name": "Market backdrop", "origin": "Whole-market price history (+ FRED macro)",
     "official": False, "freshness": "hourly"},
    {"key": "macro", "name": "Macro: rates, VIX, credit spreads", "origin": "St. Louis Fed (FRED)", "official": True,
     "freshness": "twice a day (needs a free FRED key)"},
    {"key": "ideas", "name": "Idea screens", "origin": "Computed from the feed and price history", "official": False,
     "freshness": "hourly"},
    {"key": "ledger", "name": "Forecast grading", "origin": "The app's own forecast ledger", "official": False,
     "freshness": "every 6 hours"},
    {"key": "alerts", "name": "Push alerts", "origin": "ntfy / Telegram", "official": False, "freshness": "each run"},
    {"key": "companies", "name": "Company financials & industry", "origin": "SEC XBRL companyfacts + submissions APIs",
     "official": True, "freshness": "refreshed when a new 10-Q/10-K is filed (checked every 3 days), else monthly"},
    {"key": "filings_live", "name": "Company announcements (8-K)", "origin": "SEC EDGAR live feed", "official": True,
     "freshness": "every 10 min; minutes after filing"},
    {"key": "headlines", "name": "News headlines", "origin": "GDELT Project DOC 2.0 API (gdeltproject.org)",
     "official": False, "freshness": "hourly batches (GDELT asks for one request per 5 seconds)"},
    {"key": "context", "name": "Valuation & bubble risk", "origin": "Computed from SEC financials and stored prices",
     "official": False, "freshness": "every run"},
    {"key": "forecasts", "name": "Price forecasts", "origin": "Computed (filtered historical simulation)", "official": False,
     "freshness": "after each new daily close"},
    {"key": "forecast_grading", "name": "Forecast grading & learning", "origin": "The app's own forecast ledger",
     "official": False, "freshness": "every 6 hours"},
    {"key": "backtest", "name": "Forecast backtest", "origin": "The model re-run on past dates (whole-market history)",
     "official": False, "freshness": "rebuilt monthly, in batches"},
]


def _slim(e: dict) -> dict:
    d = dict(e)
    det = dict(d.get("details") or {})
    if "owners" in det and len(det["owners"]) > 4:
        det["owners"] = det["owners"][:4]
    d["details"] = det
    if d.get("quant"):                      # the phone already knows the question; keep the feed small
        d["quant"] = {k: v for k, v in d["quant"].items() if k != "question"}
    return d


def leaders(state: dict) -> dict:
    tr = state.get("track", {})
    ins = (tr.get("insiders") or {}).get("records", {})
    cong = (tr.get("congress") or {}).get("records", {})
    funds = (tr.get("funds") or {}).get("records", {})

    def top(recs: dict, min_n: int, limit: int):
        rows = [{"id": k, **v} for k, v in recs.items() if (v.get("n") or v.get("quarters") or 0) >= min_n]
        rows.sort(key=lambda r: (r.get("rank_score") or 0, r.get("n") or r.get("quarters") or 0), reverse=True)
        return rows[:limit]
    return {
        "generated_at": iso(now_utc()),
        "method": {"hold_days": 63, "entry": "close of the first trading day after the filing/disclosure date",
                   "benchmark": "SPY", "ranking": "Wilson lower bound of beat-the-S&P rate (punishes tiny samples)"},
        "insiders": top(ins, 3, 60),
        "congress": top(cong, 5, 40),
        "funds": top(funds, 2, 40),
        "etoro": (state.get("etoro") or {}).get("leaders", []),
        "meta": {k: {kk: vv for kk, vv in (v or {}).items() if kk != "records"} for k, v in tr.items() if isinstance(v, dict)},
    }


def _etf_funds(settings: Settings) -> list[dict]:
    from .etf import fund_list
    try:
        return fund_list(settings)
    except Exception:  # noqa: BLE001 – a malformed config block must not break publishing
        return []


def health(settings: Settings, state: dict, run: dict) -> dict:
    jobs = state.get("jobs", {})
    srcs = []
    for s in SOURCES:
        j = jobs.get(s["key"], {})
        srcs.append({**s, "last_run": j.get("last_run"), "last_ok": j.get("last_ok"), "error": j.get("error"),
                     "items": j.get("items"), "note": j.get("note"), "seconds": j.get("seconds"),
                     "failures": j.get("consecutive_failures", 0)})
    funds = [{"cik": c, "sec_name": v.get("sec_name"), "ok": v.get("name_ok"), "error": v.get("error") or v.get("notice"), "info": v.get("info"),
              "last_13f": v.get("last_13f"), "period": v.get("period")} for c, v in (state.get("funds") or {}).items()]
    an = state.get("analyst") or {}
    acfg = settings.c("analyst", default={}) or {}
    analyst = {"last_run": an.get("last_run"), "today": an.get("today"), "month_usd": an.get("month_usd"),
               "analyses": an.get("analyses"), "briefs": an.get("briefs"),
               "budget_usd": acfg.get("daily_budget_usd", 3.0), "deep_model": acfg.get("deep_model", "claude-opus-5-5"),
               "deep_effort": acfg.get("deep_effort", "max")}
    et = state.get("etf") or {}
    etf_funds = [{"ticker": f["ticker"], "name": f.get("name"), "url": (et.get("urls") or {}).get(f["ticker"]),
                  "error": (et.get("errors") or {}).get(f["ticker"]), "last": (et.get("last") or {}).get(f["ticker"])}
                 for f in _etf_funds(settings)] if settings.c("etf", "enabled", default=True) else []
    return {"generated_at": iso(now_utc()), "integrations": settings.secrets.present(), "sources": srcs,
            "analyst": analyst, "funds": funds, "etf_funds": etf_funds, "last_run": run, "runs": [{k: r.get(k) for k in ("started", "finished", "new_events")}
                                                      for r in state.get("runs", [])[:12]]}


def write_all(settings: Settings, state: dict, run: dict) -> None:
    out = settings.out_dir
    events = [_slim(e) for e in state["events"]]
    tickers = {e["ticker"] for e in events if e.get("ticker")}
    quotes = {t: q for t, q in (state.get("quotes") or {}).items() if t in tickers or t == "SPY"}
    gen = iso(now_utc())
    write_json(out / "feed.json", {"generated_at": gen, "events": events})
    write_json(out / "quotes.json", {"generated_at": gen, "quotes": quotes})
    write_json(out / "leaders.json", leaders(state))
    write_json(out / "health.json", health(settings, state, run))
    q = state.get("quant") or {}
    write_json(out / "market.json", {"generated_at": gen, **(state.get("market") or {})})
    write_json(out / "ideas.json", {"generated_at": gen, "brief": state.get("brief"),
                                     "lists": (state.get("ideas") or {}).get("lists") or {},
                                     "conviction": state.get("conviction")})
    write_json(out / "accuracy.json", {"generated_at": gen, "ledger": state.get("accuracy"),
                                        "model": q.get("insider_model"), "rates": q.get("rates"),
                                        "screen": q.get("screen"), "samples": q.get("samples"),
                                        "price_history": q.get("price_history"), "built": q.get("built")})
    counts = {}
    for e in events:
        counts[e["kind"]] = counts.get(e["kind"], 0) + 1
    write_json(out / "meta.json", {"generated_at": gen, "version": int(now_utc().timestamp()), "counts": counts,
                                   "new_events": run.get("new_events", 0), "app": settings.c("app", "name")})
