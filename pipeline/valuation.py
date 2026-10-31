"""Valuation (cheap or expensive?) and bubble risk, from the SEC's own XBRL financial data and stored prices.

Per company (cached in .cache/companies.json.gz, refreshed when a new 10-Q/10-K appears or every few weeks):
  * trailing-twelve-month (TTM) revenue, profits, cash flow, plus cash, debt, equity and share count, built from
    the companyfacts API (last fiscal year + this year's year-to-date − last year's same year-to-date);
  * industry (SIC code), age as an SEC filer, recent 8-K filings and share offerings, from the submissions API.

Valuation verdict: how expensive the stock is *relative to its own fundamentals*. A cross-sectional regression
across every company the app tracks estimates the EV/sales multiple a company "usually" gets for its growth,
margins and industry (the relative-valuation approach taught by e.g. Damodaran); the stock's actual multiple
versus that fair multiple gives a percentile from "Very cheap" to "Very expensive". Banks and insurers are judged
on P/E and P/B instead. A reverse DCF shows how much cash-flow growth the price already assumes.
Valuation is a poor timing tool: expensive stocks can stay expensive for years.

Bubble risk: a transparent checklist of the warning signs in Greenwood, Shleifer & You, "Bubbles for Fama"
(Journal of Financial Economics, 2019): a big 2-year run-up net of the market, an accelerating rise, rising
volatility and turnover, share issuance and young firms; plus valuation stretch, insider selling and how tightly
the stock trades with the AI theme. It is a checklist, not a fitted model: the app grades it over time
(stocks rated High vs Low: how often did they fall 20%+?) on the Records tab.
"""
from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

import numpy as np

from .http import HttpClient, HttpError
from .util import log, now_et, now_utc, iso, parse_date, parse_iso, read_json, write_json

COMPANYFACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
CACHE = "companies.json.gz"

# Concept fallbacks. Revenue takes the largest of these for each period (they are all "total revenue"-like and
# companies switch tags over time); the others take the first concept that has the period.
REVENUE = [("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"), ("us-gaap", "Revenues"),
           ("us-gaap", "SalesRevenueNet"), ("us-gaap", "RevenueFromContractWithCustomerIncludingAssessedTax"),
           ("us-gaap", "SalesRevenueGoodsNet"), ("us-gaap", "RevenuesNetOfInterestExpense"),
           ("ifrs-full", "Revenue")]
NET_INCOME = [("us-gaap", "NetIncomeLoss"), ("us-gaap", "ProfitLoss"),
              ("us-gaap", "NetIncomeLossAvailableToCommonStockholdersBasic"),
              ("ifrs-full", "ProfitLossAttributableToOwnersOfParent"), ("ifrs-full", "ProfitLoss")]
OP_INCOME = [("us-gaap", "OperatingIncomeLoss"), ("ifrs-full", "ProfitLossFromOperatingActivities")]
GROSS = [("us-gaap", "GrossProfit"), ("ifrs-full", "GrossProfit")]
OCF = [("us-gaap", "NetCashProvidedByUsedInOperatingActivities"),
       ("us-gaap", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"),
       ("ifrs-full", "CashFlowsFromUsedInOperatingActivities")]
CAPEX = [("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment"), ("us-gaap", "PaymentsToAcquireProductiveAssets"),
         ("ifrs-full", "PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities")]
SBC = [("us-gaap", "ShareBasedCompensation"), ("us-gaap", "AllocatedShareBasedCompensationExpense")]
DIVIDENDS = [("us-gaap", "PaymentsOfDividendsCommonStock"), ("us-gaap", "PaymentsOfDividends"),
             ("us-gaap", "DividendsCommonStockCash"), ("ifrs-full", "DividendsPaidClassifiedAsFinancingActivities")]
CASH = [("us-gaap", "CashAndCashEquivalentsAtCarryingValue"),
        ("us-gaap", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"), ("us-gaap", "Cash"),
        ("ifrs-full", "CashAndCashEquivalents")]
ST_INVEST = [("us-gaap", "ShortTermInvestments"), ("us-gaap", "MarketableSecuritiesCurrent"),
             ("us-gaap", "AvailableForSaleSecuritiesDebtSecuritiesCurrent")]
EQUITY = [("us-gaap", "StockholdersEquity"),
          ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
          ("ifrs-full", "EquityAttributableToOwnersOfParent"), ("ifrs-full", "Equity")]
SHARES_COVER = [("dei", "EntityCommonStockSharesOutstanding")]
SHARES_BS = [("us-gaap", "CommonStockSharesOutstanding")]
SHARES_DILUTED = [("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding"),
                  ("ifrs-full", "AdjustedWeightedAverageShares")]
DEBT_TOTAL = [("us-gaap", "LongTermDebt"), ("us-gaap", "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities")]
DEBT_NONCUR = [("us-gaap", "LongTermDebtNoncurrent"), ("us-gaap", "LongTermDebtAndCapitalLeaseObligations")]
DEBT_CUR = [("us-gaap", "LongTermDebtCurrent"), ("us-gaap", "LongTermDebtAndCapitalLeaseObligationsCurrent")]
DEBT_SHORT = [("us-gaap", "ShortTermBorrowings"), ("us-gaap", "CommercialPaper")]
DEBT_CUR_ALL = [("us-gaap", "DebtCurrent")]

PERIODIC = {"10-K", "10-Q", "10-K/A", "10-Q/A", "20-F", "20-F/A", "40-F", "10-KT"}
OFFERING_FORMS = {"S-1", "S-1/A", "S-3", "S-3ASR", "F-1", "F-3", "F-3ASR", "424B1", "424B2", "424B3", "424B4", "424B5"}

SIC_GROUPS = {  # broad industry names for 2-digit SIC codes (for peers and plain-English labels)
    "10": "Metal mining", "13": "Oil & gas", "20": "Food & beverages", "28": "Chemicals & pharma",
    "35": "Computers & machinery", "36": "Electronics & semiconductors", "37": "Vehicles & aerospace",
    "38": "Instruments & medical devices", "48": "Telecom & media", "49": "Utilities & power",
    "50": "Wholesale", "51": "Wholesale", "53": "Retail", "56": "Retail (apparel)", "58": "Restaurants",
    "59": "Retail", "60": "Banks", "61": "Lending", "62": "Brokers & asset managers", "63": "Insurance",
    "64": "Insurance brokers", "65": "Real estate", "67": "Holding & investment companies",
    "70": "Hotels", "73": "Software & IT services", "78": "Entertainment", "80": "Health services",
    "87": "Engineering & research services",
}
# SIC codes whose business is mostly computing, chips or software: "sector-level" AI exposure
AI_SIC = {3570, 3571, 3572, 3576, 3577, 3661, 3663, 3669, 3672, 3674, 3679, 7370, 7371, 7372, 7373, 7374}
LABELS = ["Very cheap", "Cheap", "Fairly valued", "Expensive", "Very expensive"]

# Greenwood, Shleifer & You (2019): probability of a 40%+ crash within 2 years for US industries, by 2-year
# return net of the market (their Table, as also summarised by Schwab). Below a 50% run-up the study has no row.
GSY_CRASH = [(0.50, 0.20), (0.75, 0.36), (1.00, 0.53), (1.25, 0.76), (1.50, 0.80)]


# ---------------------------------------------------------------- XBRL helpers

def _d(s) -> dt.date | None:
    try:
        return dt.date.fromisoformat(str(s)[:10])
    except (TypeError, ValueError):
        return None


def _facts(js: dict, concepts: list[tuple[str, str]], unit: str = "USD") -> dict[str, list[dict]]:
    """{concept: [fact...]} for the concepts present, in `unit` (money: USD only; other currencies are skipped,
    since mixing them with a USD share price would be wrong)."""
    out = {}
    facts = js.get("facts") or {}
    for tax, name in concepts:
        units = ((facts.get(tax) or {}).get(name) or {}).get("units") or {}
        rows = units.get(unit)
        if rows:
            out[name] = rows
    return out


def currency_of(js: dict) -> str | None:
    """The reporting currency, judged from where revenue/net income facts live (None if nothing is found)."""
    facts = js.get("facts") or {}
    for tax, name in REVENUE + NET_INCOME:
        units = ((facts.get(tax) or {}).get(name) or {}).get("units") or {}
        if units:
            return "USD" if "USD" in units else sorted(units)[0]
    return None


def _clean_duration(rows: list[dict]) -> dict[tuple[dt.date, dt.date], float]:
    """(start, end) → value, keeping the most recently filed value for each period (restatements win)."""
    best: dict[tuple, tuple[str, float]] = {}
    for r in rows:
        s, e, v = _d(r.get("start")), _d(r.get("end")), r.get("val")
        if s is None or e is None or v is None or not isinstance(v, (int, float)) or not math.isfinite(v):
            continue
        if e <= s:
            continue
        k = (s, e)
        f = str(r.get("filed") or "")
        if k not in best or f >= best[k][0]:
            best[k] = (f, float(v))
    return {k: v for k, (_, v) in best.items()}


def merge_duration(groups: dict[str, list[dict]], how: str = "first", order: list[str] | None = None) -> dict:
    """Combine several concepts into one period → value map ('max' for revenue, 'first' = priority order)."""
    out: dict[tuple, float] = {}
    names = order or list(groups)
    for name in names:
        if name not in groups:
            continue
        for k, v in _clean_duration(groups[name]).items():
            if k not in out:
                out[k] = v
            elif how == "max":
                out[k] = max(out[k], v)
    return out


def _near(a: dt.date, b: dt.date, days: int) -> bool:
    return abs((a - b).days) <= days


def _dur(k: tuple[dt.date, dt.date]) -> int:
    return (k[1] - k[0]).days + 1


def ttm(periods: dict[tuple[dt.date, dt.date], float], as_of: dt.date | None = None) -> dict | None:
    """Trailing twelve months ending at the latest period end on or before `as_of`.

    Methods, in order: a fiscal year ending then; last fiscal year + current year-to-date − prior year-to-date;
    four consecutive quarters; else the latest fiscal year if it is under 15 months old ('annual')."""
    items = [(k, v) for k, v in periods.items() if as_of is None or k[1] <= as_of + dt.timedelta(days=10)]
    if not items:
        return None
    ends = sorted({k[1] for k, _ in items})
    latest = ends[-1]
    if ((as_of or now_utc().date()) - latest).days > 460:          # too old to call "trailing"
        return None
    annual = [(k, v) for k, v in items if 350 <= _dur(k) <= 380]
    for k, v in annual:
        if _near(k[1], latest, 3):
            return {"value": v, "end": latest.isoformat(), "method": "fiscal year"}
    prior_annual = sorted([(k, v) for k, v in annual if k[1] < latest - dt.timedelta(days=3)], key=lambda x: x[0][1])
    if prior_annual:
        (a_s, a_e), a_v = prior_annual[-1]
        cur = [(k, v) for k, v in items if _near(k[1], latest, 3) and _near(k[0], a_e + dt.timedelta(days=1), 7)]
        for (c_s, c_e), c_v in cur:
            d = _dur((c_s, c_e))
            prev = [(k, v) for k, v in items if _near(k[1], c_e - dt.timedelta(days=364), 10) and abs(_dur(k) - d) <= 10]
            if prev:
                p_v = prev[0][1]
                return {"value": a_v + c_v - p_v, "end": latest.isoformat(), "method": "year-to-date"}
    quarters = sorted([(k, v) for k, v in items if 80 <= _dur(k) <= 100], key=lambda x: x[0][1])
    chain = [q for q in quarters if _near(q[0][1], latest, 3)]
    if chain:
        cur = chain[-1]
        seq = [cur]
        for _ in range(3):
            want_end = seq[-1][0][0] - dt.timedelta(days=1)
            nxt = [q for q in quarters if _near(q[0][1], want_end, 7)]
            if not nxt:
                break
            seq.append(nxt[-1])
        if len(seq) == 4:
            return {"value": sum(v for _, v in seq), "end": latest.isoformat(), "method": "four quarters"}
    if annual:
        (a_s, a_e), a_v = sorted(annual, key=lambda x: x[0][1])[-1]
        ref = as_of or now_utc().date()
        if (ref - a_e).days <= 460:
            return {"value": a_v, "end": a_e.isoformat(), "method": "annual"}
    return None


def ttm_pair(periods: dict) -> tuple[dict | None, dict | None]:
    """(latest TTM, the TTM one year earlier) — the second only if it really ends ~a year before the first."""
    cur = ttm(periods)
    if not cur:
        return None, None
    end = _d(cur["end"])
    prev = ttm(periods, as_of=end - dt.timedelta(days=365))
    if prev and not _near(_d(prev["end"]), end - dt.timedelta(days=365), 25):
        prev = None
    return cur, prev


def latest_instant(groups: dict[str, list[dict]], order: list[str], as_of: dt.date | None = None,
                   sum_classes: bool = False, with_count: bool = False):
    """Most recent instant value (balance-sheet items, share counts). `sum_classes`: a cover page that lists
    several share classes at the same date in the same filing reports one value per class; add them."""
    best = None
    for name in order:
        rows = groups.get(name) or []
        by: dict[tuple, dict] = {}
        for r in rows:
            e, v = _d(r.get("end")), r.get("val")
            if e is None or v is None or r.get("start") or not isinstance(v, (int, float)) or not math.isfinite(v):
                continue
            if as_of is not None and e > as_of:
                continue
            k = (e, str(r.get("accn") or ""))
            by.setdefault(k, {"end": e, "filed": str(r.get("filed") or ""), "vals": []})["vals"].append(float(v))
        if not by:
            continue
        top_end = max(x["end"] for x in by.values())
        cands = [x for x in by.values() if x["end"] == top_end]
        x = max(cands, key=lambda c: c["filed"])
        val = sum(x["vals"]) if sum_classes else max(x["vals"])
        if best is None or top_end > best[1]:
            best = (val, top_end, len(x["vals"]))
    if not best:
        return None
    return (best[0], best[1].isoformat(), best[2]) if with_count else (best[0], best[1].isoformat())


def _at(groups: dict, order: list[str], when: dt.date, tol: int = 7) -> float | None:
    """The value reported for `when` (± tol days), first concept in priority order that has one, latest filing
    wins. Balance-sheet items are all read at one date, so a tag a company stopped using years ago (or last
    year's short-term investments) can't be mixed into today's balance sheet."""
    for name in order:
        best = None
        for r in groups.get(name) or []:
            e, v = _d(r.get("end")), r.get("val")
            if e is None or v is None or r.get("start") or not isinstance(v, (int, float)) or not math.isfinite(v):
                continue
            if abs((e - when).days) > tol:
                continue
            f = str(r.get("filed") or "")
            if best is None or f > best[0]:
                best = (f, float(v))
        if best is not None:
            return best[1]
    return None


def _instant_at(groups: dict, order: list[str], when: dt.date, tol: int = 60, sum_classes: bool = False) -> float | None:
    r = latest_instant(groups, order, as_of=when + dt.timedelta(days=tol), sum_classes=sum_classes)
    if r and _near(_d(r[1]), when, tol):
        return r[0]
    return None


def extract_metrics(js: dict) -> dict:
    """Everything the valuation needs from one companyfacts document."""
    cur = currency_of(js)
    out: dict = {"currency": cur}
    if cur not in (None, "USD"):
        out["note"] = f"reports in {cur}; valuation needs USD figures"
        return out
    rev = merge_duration(_facts(js, REVENUE), how="max")
    r_cur, r_prev = ttm_pair(rev)

    def one(concepts):
        g = _facts(js, concepts)
        return ttm(merge_duration(g, order=[n for _, n in concepts]))
    ni, oi, gp, ocf, capex, sbc, div = (one(c) for c in (NET_INCOME, OP_INCOME, GROSS, OCF, CAPEX, SBC, DIVIDENDS))
    bs = _facts(js, CASH + ST_INVEST + EQUITY + DEBT_TOTAL + DEBT_NONCUR + DEBT_CUR + DEBT_SHORT + DEBT_CUR_ALL)
    names = lambda lst: [n for _, n in lst]  # noqa: E731
    anchors = [x for x in (latest_instant(bs, names(EQUITY)), latest_instant(bs, names(CASH))) if x]
    bs_date = max((_d(x[1]) for x in anchors), default=None)
    if bs_date is None:
        dd = [x for x in (latest_instant(bs, names(DEBT_TOTAL)), latest_instant(bs, names(DEBT_NONCUR))) if x]
        bs_date = max((_d(x[1]) for x in dd), default=None)
    cash = sti = eq = debt = None
    if bs_date is not None:
        cash, sti, eq = _at(bs, names(CASH), bs_date), _at(bs, names(ST_INVEST), bs_date), _at(bs, names(EQUITY), bs_date)
        debt = _debt_at(bs, bs_date)
    sh_cover = _facts(js, SHARES_COVER, unit="shares")
    sh_bs = _facts(js, SHARES_BS, unit="shares")
    sh_dil = _facts(js, SHARES_DILUTED, unit="shares")
    cov = latest_instant(sh_cover, ["EntityCommonStockSharesOutstanding"], sum_classes=True, with_count=True)
    dil = ttm_like_latest(sh_dil)
    classes = cov[2] if cov else 1
    shares = (cov[0], cov[1]) if cov else latest_instant(sh_bs, ["CommonStockSharesOutstanding"])
    unclear = None
    if shares and dil and classes >= 2 and not (0.6 <= shares[0] / dil[0] <= 1.6):
        # several share classes that don't add up to the diluted count: classes with different economics
        # (Berkshire's A and B, say). One ticker's price times a share count would be wrong, so don't value it.
        shares, unclear = None, "several share classes with different prices; market value can't be computed"
    elif shares and dil and not (0.5 <= shares[0] / dil[0] <= 2.0):
        shares = (dil[0], dil[1])
    if not shares and dil and not unclear:
        shares = dil
    shares_prev = None
    if shares:
        when = _d(shares[1]) - dt.timedelta(days=365)
        shares_prev = (_instant_at(sh_cover, ["EntityCommonStockSharesOutstanding"], when, sum_classes=True)
                       or _instant_at(sh_bs, ["CommonStockSharesOutstanding"], when))
        if shares_prev and not (0.3 <= shares[0] / shares_prev <= 3.0):
            shares_prev = None                      # a split or class change, not issuance
    out.update({
        "rev": r_cur and r_cur["value"], "rev_prev": r_prev and r_prev["value"],
        "rev_end": r_cur and r_cur["end"], "rev_method": r_cur and r_cur["method"],
        "ni": ni and ni["value"], "oi": oi and oi["value"], "gp": gp and gp["value"],
        "ocf": ocf and ocf["value"], "capex": capex and capex["value"], "sbc": sbc and sbc["value"],
        "div": div and abs(div["value"]),
        "cash": (cash or 0.0) + (sti or 0.0) if (cash is not None or sti is not None) else None,
        "debt": debt, "equity": eq, "bs_end": bs_date.isoformat() if bs_date else None,
        "shares": shares and shares[0], "shares_end": shares and shares[1], "shares_prev": shares_prev,
        "shares_prev_end": when.isoformat() if shares and shares_prev else None, "share_classes": classes,
    })
    if unclear:
        out["note"] = unclear
    return out


def ttm_like_latest(groups: dict) -> tuple[float, str] | None:
    """Latest weighted-average diluted share count (a quarterly figure if there is one, else annual)."""
    per = merge_duration(groups)
    if not per:
        return None
    k = max(per, key=lambda k: (k[1], -_dur(k)))
    return per[k], k[1].isoformat()


def _debt_at(bs: dict, when: dt.date) -> float | None:
    """Total debt on the balance-sheet date. LongTermDebt already includes the current portion, so only
    short-term borrowings are added to it; DebtCurrent already includes short-term borrowings, so they are not
    added again."""
    names = lambda lst: [n for _, n in lst]  # noqa: E731
    tot, short = _at(bs, names(DEBT_TOTAL), when), _at(bs, names(DEBT_SHORT), when)
    if tot is not None:
        return tot + (short or 0.0)
    non, cur_all = _at(bs, names(DEBT_NONCUR), when), _at(bs, names(DEBT_CUR_ALL), when)
    cur = cur_all if cur_all is not None else ((_at(bs, names(DEBT_CUR), when) or 0.0) + (short or 0.0))
    if non is None and not cur:
        return None
    return (non or 0.0) + (cur or 0.0)


# ---------------------------------------------------------------- submissions: industry, age, 8-Ks, offerings

def parse_submissions(sub: dict, today: dt.date | None = None) -> dict:
    today = today or now_et().date()
    r = (sub.get("filings") or {}).get("recent") or {}
    n = len(r.get("accessionNumber") or [])
    col = lambda k, i: (r.get(k) or [None] * n)[i] if i < len(r.get(k) or []) else None  # noqa: E731
    k8, periodic, offerings = [], None, 0
    first = None
    for i in range(n):
        form = col("form", i) or ""
        fd = col("filingDate", i) or ""
        if fd and (first is None or fd < first):
            first = fd
        if form in ("8-K", "8-K/A") and len(k8) < 30:
            items = [x.strip() for x in str(col("items", i) or "").split(",") if x.strip()]
            k8.append({"d": fd, "acc": col("accessionNumber", i), "items": items, "form": form,
                       "at": col("acceptanceDateTime", i), "doc": col("primaryDocument", i),
                       "desc": col("primaryDocDescription", i)})
        if form in PERIODIC and periodic is None:
            periodic = {"form": form, "filed": fd, "acc": col("accessionNumber", i), "period": col("reportDate", i)}
        if form in OFFERING_FORMS and fd and (today - (_d(fd) or today)).days <= 365:
            offerings += 1
    for f in (sub.get("filings") or {}).get("files") or []:      # older filings live in extra files
        ff = f.get("filingFrom")
        if ff and (first is None or ff < first):
            first = ff
    sic = sub.get("sic")
    try:
        sic = int(sic) if sic not in (None, "") else None
    except (TypeError, ValueError):
        sic = None
    return {"name": sub.get("name"), "sic": sic, "sic_desc": sub.get("sicDescription"),
            "fye": sub.get("fiscalYearEnd"), "category": sub.get("category"),
            "tickers": (sub.get("tickers") or [])[:4], "first_filed": first, "periodic": periodic,
            "k8": k8, "offerings_12m": offerings}


# ---------------------------------------------------------------- refresh (time-boxed, prioritised)

def load_cache(cache_dir: Path) -> dict:
    c = read_json(cache_dir / CACHE)
    return c if isinstance(c, dict) and isinstance(c.get("by_cik"), dict) else {"by_cik": {}}


def save_cache(cache_dir: Path, cache: dict) -> None:
    write_json(cache_dir / CACHE, cache)


def mark_stale(cache: dict, cik: str, what: str = "sub") -> None:
    """A live feed saw a new filing for this company: refetch its submissions (and facts, for 10-Q/10-K)."""
    rec = cache["by_cik"].get(str(cik).zfill(10))
    if rec is not None:
        rec[f"stale_{what}"] = True


def refresh(http: HttpClient, cache_dir: Path, tickers: list[str], tick: dict, budget_seconds: float,
            max_companies: int = 150, today: dt.date | None = None) -> tuple[int, str]:
    """Fetch/refresh SEC data for the given tickers (already in priority order) within the time budget."""
    import time
    t0 = time.monotonic()
    today = today or now_et().date()
    cache = load_cache(cache_dir)
    cool = parse_iso(cache.get("sec_cooldown_until"))
    if cool and cool > now_utc():
        return 0, f"waiting until {iso(cool)}: the SEC asked the app to slow down"
    by_t = tick.get("by_ticker") or {}
    want: list[tuple[int, str, str, bool, bool]] = []      # (rank, cik, ticker, need_sub, need_facts)
    seen = set()
    for rank, t in enumerate(tickers):
        info = by_t.get(t)
        if not info or not info.get("cik"):
            continue
        cik = str(info["cik"]).zfill(10)
        if cik in seen:
            continue
        seen.add(cik)
        rec = cache["by_cik"].get(cik) or {}
        sub_age = _age_days(rec.get("sub_fetched"), today)
        fact_age = _age_days(rec.get("facts_fetched"), today)
        per = (rec.get("sub") or {}).get("periodic") or {}
        new_periodic = bool(per.get("filed") and rec.get("facts_fetched") and per["filed"] > rec["facts_fetched"][:10])
        need_sub = rec.get("stale_sub") or sub_age is None or sub_age >= 3
        need_facts = (rec.get("stale_facts") or fact_age is None or fact_age >= 30 or new_periodic) \
            and not (rec.get("facts_error") and fact_age is not None and fact_age < 14)
        if need_sub or need_facts:
            urgency = 0 if (sub_age is None or fact_age is None) else 1 if (rec.get("stale_sub") or new_periodic) else 2
            want.append((urgency * 100000 + rank, cik, t, need_sub, need_facts))
    want.sort()
    done = errors = 0
    for _, cik, t, need_sub, need_facts in want[:max_companies]:
        if time.monotonic() - t0 > budget_seconds:
            break
        rec = cache["by_cik"].setdefault(cik, {"cik": cik})
        rec["t"] = t
        try:
            if need_sub:
                sub = http.get_json(SUBMISSIONS.format(cik=cik))
                rec["sub"] = parse_submissions(sub, today)
                rec["sub_fetched"] = iso(now_utc())
                rec.pop("stale_sub", None)
                per = rec["sub"].get("periodic") or {}
                if per.get("filed") and rec.get("facts_fetched") and per["filed"] > rec["facts_fetched"][:10]:
                    need_facts = True
            if need_facts:
                try:
                    js = http.get_json(COMPANYFACTS.format(cik=cik))
                    rec["m"] = extract_metrics(js)
                    rec.pop("facts_error", None)
                except HttpError as e:
                    if e.status != 404:
                        raise
                    rec["m"] = {"note": "no XBRL financial data at the SEC (fund, trust or new filer)"}
                    rec["facts_error"] = "404"
                rec["facts_fetched"] = iso(now_utc())
                rec.pop("stale_facts", None)
            done += 1
        except HttpError as e:
            errors += 1
            rec["error"] = f"HttpError {e.status}"
            if e.status in (403, 429):              # SEC rate-limit page: stop now and back off (shared IPs)
                cache["sec_cooldown_until"] = iso(now_utc() + dt.timedelta(minutes=30))
                log.warning("SEC returned %s: pausing company data for 30 minutes", e.status)
                break
            log.warning("company data %s (%s): %s", t, cik, e.status)
        except Exception as e:  # noqa: BLE001 – one company failing must not stop the rest
            errors += 1
            rec["error"] = f"{type(e).__name__}: {str(e)[:160]}"
            log.warning("company data %s (%s): %s", t, cik, e)
    _prune(cache, today)
    save_cache(cache_dir, cache)
    have = sum(1 for r in cache["by_cik"].values() if (r.get("m") or {}).get("rev") is not None)
    left = max(0, len(want) - done - errors)
    return done, (f"{have} companies with financials on file" + (f"; {left} queued" if left else "")
                  + (f"; {errors} errors" if errors else ""))


def _age_days(ts: str | None, today: dt.date) -> int | None:
    d = parse_iso(ts) if ts else None
    return None if d is None else (today - d.date()).days


def _prune(cache: dict, today: dt.date, keep_days: int = 120) -> None:
    for cik in [c for c, r in cache["by_cik"].items() if (_age_days(r.get("sub_fetched"), today) or 0) > keep_days]:
        cache["by_cik"].pop(cik, None)


# ---------------------------------------------------------------- valuation

def _w(x, lo, hi):
    return None if x is None or not math.isfinite(x) else min(hi, max(lo, x))


def fundamentals_view(m: dict, price: float | None) -> dict | None:
    """Multiples and quality numbers for one company at the current price."""
    if not m or not price or price <= 0 or not m.get("shares") or m["shares"] <= 0:
        return None
    mcap = price * m["shares"]
    rev, ni, oi, gp = m.get("rev"), m.get("ni"), m.get("oi"), m.get("gp")
    ocf, capex, sbc = m.get("ocf"), m.get("capex"), m.get("sbc")
    cash, debt, eq = m.get("cash") or 0.0, m.get("debt") or 0.0, m.get("equity")
    ev = mcap + debt - cash
    fcf = (ocf - (capex or 0.0)) if ocf is not None else None
    v = {"mcap": mcap, "ev": ev, "fcf": fcf}
    pos = lambda a, b: (a / b) if (a is not None and b is not None and b > 0) else None  # noqa: E731
    v["pe"] = pos(mcap, ni)
    v["ps"] = pos(mcap, rev)
    v["ev_s"] = pos(ev, rev) if ev > 0 else None
    v["ev_ebit"] = pos(ev, oi) if ev > 0 else None
    v["pb"] = pos(mcap, eq)
    v["fcf_yield"] = (fcf / mcap) if fcf is not None else None
    v["growth"] = (rev / m["rev_prev"] - 1) if (rev and m.get("rev_prev") and m["rev_prev"] > 0) else None
    v["gross_margin"] = (gp / rev) if (gp is not None and rev and rev > 0) else None
    v["op_margin"] = (oi / rev) if (oi is not None and rev and rev > 0) else None
    v["net_margin"] = (ni / rev) if (ni is not None and rev and rev > 0) else None
    v["fcf_margin"] = (fcf / rev) if (fcf is not None and rev and rev > 0) else None
    v["sbc_pct"] = (sbc / rev) if (sbc is not None and rev and rev > 0) else None
    v["dilution"] = (m["shares"] / m["shares_prev"] - 1) if m.get("shares_prev") else None
    if v["dilution"] is not None and not m.get("split_checked"):
        ratio = m["shares"] / m["shares_prev"]
        if any(abs(ratio / k - 1) < 0.04 for k in (1.5, 2, 3, 4, 5, 10, 1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 10)):
            v["dilution"] = None                    # looks like a split with no split record: not issuance
    v["net_cash"] = cash - debt
    v["div_yield"] = min(0.2, m["div"] / mcap) if m.get("div") and mcap > 0 else None
    return v


def adjust_for_splits(m: dict, t: str, splits: dict | None, upto: str | None = None) -> dict:
    """Share counts re-based for splits after the date they were reported (a 10-for-1 split after the last
    10-Q would otherwise make the company look 10× cheaper until the next filing), counting only splits on or
    before `upto`, the date of the price the count is paired with (a pre-split price goes with pre-split shares)."""
    ev = (splits or {}).get(t)
    if ev is None or not m.get("shares"):
        return m

    def after(d: str | None) -> float:
        f = 1.0
        for ed, k in ev:
            e = str(ed)[:10]
            if d and e > d[:10] and (not upto or e <= str(upto)[:10]) and k and k > 0:
                f *= float(k)
        return f
    out = dict(m, split_checked=True)
    fs = after(m.get("shares_end"))
    out["shares"] = m["shares"] * fs
    if m.get("shares_prev"):
        out["shares_prev"] = m["shares_prev"] * after(m.get("shares_prev_end"))
    if fs != 1.0:
        out["split_adj"] = fs
    return out


def reverse_dcf(ev: float | None, fcf: float | None, r: float = 0.09, g_term: float = 0.03, years: int = 10) -> dict | None:
    """The 10-year free-cash-flow growth rate that the current enterprise value implies (then g_term forever)."""
    if not ev or ev <= 0 or fcf is None or fcf <= 0 or r <= g_term:
        return None

    def value(g: float) -> float:
        tot, f = 0.0, fcf
        for t in range(1, years + 1):
            f *= (1 + g)
            tot += f / (1 + r) ** t
        return tot + f * (1 + g_term) / (r - g_term) / (1 + r) ** years
    lo, hi = -0.5, 1.0
    if value(hi) < ev:
        return {"growth": None, "above": hi, "r": r, "g_term": g_term, "years": years}
    if value(lo) > ev:
        return {"growth": None, "below": lo, "r": r, "g_term": g_term, "years": years}
    for _ in range(80):
        mid = (lo + hi) / 2
        if value(mid) < ev:
            lo = mid
        else:
            hi = mid
    return {"growth": (lo + hi) / 2, "r": r, "g_term": g_term, "years": years}


def is_financial(sic: int | None) -> bool:
    return sic is not None and 6000 <= sic <= 6799


def _pctl(values: np.ndarray, x: float) -> float:
    """Share of `values` below x (ties count half), in 0-100."""
    if not len(values):
        return float("nan")
    lo = np.searchsorted(values, x, side="left")
    hi = np.searchsorted(values, x, side="right")
    return float((lo + hi) / 2 / len(values) * 100)


def label_for(pctl: float | None) -> str | None:
    if pctl is None or not math.isfinite(pctl):
        return None
    return LABELS[0] if pctl < 10 else LABELS[1] if pctl < 30 else LABELS[2] if pctl <= 70 else LABELS[3] if pctl < 90 else LABELS[4]


def fit_relative(rows: list[dict], min_n: int = 120, ridge: float = 1.0) -> dict | None:
    """Cross-sectional fit of log(EV/sales) on growth, margins and industry. rows: dicts with ev_s, growth,
    op_margin, gross_margin, sic. Returns the coefficients and the residual distribution."""
    use = [r for r in rows if r.get("ev_s") and r["ev_s"] > 0 and r.get("growth") is not None
           and r.get("op_margin") is not None and not is_financial(r.get("sic"))]
    if len(use) < min_n:
        return None
    groups: dict[str, int] = {}
    for r in use:
        g = _sic2(r.get("sic"))
        groups[g] = groups.get(g, 0) + 1
    dummies = sorted(g for g, n in groups.items() if n >= 15 and g != "other")
    med_gm = float(np.median([r["gross_margin"] for r in use if r.get("gross_margin") is not None] or [0.4]))

    def feats(r):
        g = _w(r.get("growth"), -0.5, 1.5)
        om = _w(r.get("op_margin"), -0.6, 0.6)
        gm = r.get("gross_margin")
        gm_miss = 1.0 if gm is None else 0.0
        gm = _w(gm if gm is not None else med_gm, 0.0, 1.0)
        x = [g, g * g, om, max(om, 0) * g, gm, gm_miss]
        s2 = _sic2(r.get("sic"))
        x += [1.0 if s2 == d else 0.0 for d in dummies]
        return x
    X = np.asarray([feats(r) for r in use], dtype=float)
    y = np.log(np.asarray([r["ev_s"] for r in use], dtype=float))
    y = np.clip(y, np.log(0.05), np.log(200))
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    A = np.hstack([np.ones((len(Z), 1)), Z])
    P = np.eye(A.shape[1]) * ridge
    P[0, 0] = 0.0
    beta = np.linalg.solve(A.T @ A + P, A.T @ y)
    resid = y - A @ beta
    r2 = 1 - float(np.var(resid)) / float(np.var(y)) if np.var(y) > 0 else None
    return {"beta": beta.tolist(), "mu": mu.tolist(), "sd": sd.tolist(), "dummies": dummies, "med_gm": med_gm,
            "resid": np.sort(resid).tolist(), "n": len(use), "r2": r2, "feats": feats}


def _sic2(sic) -> str:
    try:
        return str(int(sic)).zfill(4)[:2]
    except (TypeError, ValueError):
        return "other"


def predict_fair(fit: dict, r: dict) -> float | None:
    if not fit or r.get("growth") is None or r.get("op_margin") is None:
        return None
    x = np.asarray(fit["feats"](r), dtype=float)
    z = (x - np.asarray(fit["mu"])) / np.asarray(fit["sd"])
    return float(np.exp(fit["beta"][0] + z @ np.asarray(fit["beta"][1:])))


ADR_NOTE = ("A foreign company filing form 20-F: its US shares are usually ADRs, and the SEC data doesn't give "
            "how many ordinary shares each ADR represents, so it isn't valued here")


def value_all(companies: dict, quotes: dict, cfg: dict | None = None, splits: dict | None = None) -> dict[str, dict]:
    """{ticker: valuation view}. `companies` is the cache's by_cik map; quotes give the current prices;
    `splits` ({ticker: [[date, share factor]]}) re-bases share counts reported before a split."""
    cfg = cfg or {}
    r_disc, g_term = float(cfg.get("discount_rate", 0.09)), float(cfg.get("terminal_growth", 0.03))
    rows, aliases, notes = [], [], {}
    for rec in companies.values():
        t, m = rec.get("t"), rec.get("m") or {}
        sub = rec.get("sub") or {}
        tks = [t] + [x for x in (sub.get("tickers") or []) if x and x != t and x in quotes]
        form = str((sub.get("periodic") or {}).get("form") or "")
        if form.startswith("20-F") or m.get("note"):
            for tt in tks:
                if tt:
                    notes[tt] = ADR_NOTE if form.startswith("20-F") else m["note"]
            continue
        for k, tt in enumerate(tks):
            q = quotes.get(tt) or {}
            fv = fundamentals_view(adjust_for_splits(m, tt, splits, q.get("asof")), q.get("price"))
            if fv:
                (rows if k == 0 else aliases).append({"t": tt, "sic": sub.get("sic"), "m": m, "sub": sub, **fv})
    fit = fit_relative(rows)
    resid = np.asarray(fit["resid"]) if fit else None
    # market-wide distributions for plain percentiles
    dist = {}
    for k in ("pe", "ps", "ev_s", "pb", "fcf_yield", "ev_ebit"):
        vals = np.sort(np.asarray([r[k] for r in rows if r.get(k) is not None and math.isfinite(r[k])], dtype=float))
        dist[k] = vals
    fin = [r for r in rows if is_financial(r.get("sic"))]
    fin_pe = np.sort(np.asarray([r["pe"] for r in fin if r.get("pe")], dtype=float))
    fin_pb = np.sort(np.asarray([r["pb"] for r in fin if r.get("pb")], dtype=float))
    out = {}
    for r in rows + aliases:
        v = {k: r.get(k) for k in ("mcap", "ev", "fcf", "pe", "ps", "ev_s", "ev_ebit", "pb", "fcf_yield", "growth",
                                     "gross_margin", "op_margin", "net_margin", "fcf_margin", "sbc_pct", "dilution",
                                     "net_cash", "div_yield")}
        m = r["m"]
        v.update(period_end=m.get("rev_end"), method=m.get("rev_method"), rev=m.get("rev"), ni=m.get("ni"),
                 industry=r["sub"].get("sic_desc"), sic=r.get("sic"),
                 group=SIC_GROUPS.get(_sic2(r.get("sic")), r["sub"].get("sic_desc")))
        pct = {k: round(_pctl(dist[k], r[k]), 1) for k in dist if r.get(k) is not None and len(dist[k]) >= 50}
        v["market_pctl"] = pct
        basis, rich = None, None
        if is_financial(r.get("sic")):
            parts = [x for x in (_pctl(fin_pe, r["pe"]) if r.get("pe") and len(fin_pe) >= 25 else None,
                                 _pctl(fin_pb, r["pb"]) if r.get("pb") and len(fin_pb) >= 25 else None) if x is not None]
            if parts:
                rich, basis = float(np.mean(parts)), "P/E and P/B versus other financial companies"
        elif fit is not None:
            fair = predict_fair(fit, r)
            if fair and r.get("ev_s"):
                e = math.log(min(200, max(0.05, r["ev_s"]))) - math.log(fair)
                rich = _pctl(resid, e)
                v.update(fair_ev_s=fair, premium=math.exp(e) - 1)
                basis = "EV/sales versus companies with similar growth, margins and industry"
        if rich is None and r.get("ev_s") and len(dist["ev_s"]) >= 50:
            rich, basis = _pctl(dist["ev_s"], r["ev_s"]), "EV/sales versus all tracked companies (growth unknown)"
        v["rich_pctl"] = None if rich is None else round(rich, 1)
        v["label"] = label_for(rich)
        v["basis"] = basis
        v["dcf"] = reverse_dcf(r.get("ev") if not is_financial(r.get("sic")) else r.get("mcap"), r.get("fcf"),
                               r_disc, g_term)
        out[r["t"]] = v
    for tt, note in notes.items():
        out.setdefault(tt, {"label": None, "note": note})
    meta = {"n": len(rows), "fit_n": fit["n"] if fit else 0, "r2": fit["r2"] if fit else None,
            "financials": len(fin), "not_valued": len(notes), "asof": iso(now_utc())}
    return {"by_ticker": out, "meta": meta}


# ---------------------------------------------------------------- AI exposure

def ai_exposure(t: str, sic: int | None, ai_cfg: dict, factor: dict | None = None, claude: str | None = None) -> dict:
    """How tied a stock is to the AI trade: curated list (core / platform / power & infrastructure), the
    industry code, and how much its daily moves load on an AI-basket factor (net of the market)."""
    tiers = {"core": "Core AI (chips, AI infrastructure, pure plays)", "platform": "AI platform (hyperscalers, big software)",
             "power": "AI build-out (power, data centres, networking)"}
    why = []
    level = "none"
    for tier in ("core", "platform", "power"):
        if t in set(ai_cfg.get(tier) or []):
            level = tier
            why.append(tiers[tier])
            break
    if level == "none" and sic in AI_SIC:
        level = "sector"
        why.append("Its industry (computing, chips or software) is directly exposed to AI spending")
    if factor and factor.get("beta") is not None and factor.get("t") is not None:
        if factor["beta"] >= 0.5 and factor["t"] >= 3:
            why.append(f"Its daily moves track the AI basket (loading {factor['beta']:.2f}, beyond the market's own moves)")
            if level in ("none", "sector"):
                level = "linked"
    if claude in ("core", "adjacent", "low", "none") and claude:
        why.append(f"Claude's read: {claude} exposure")
        if claude == "core" and level in ("none", "sector", "linked"):
            level = "core"
    label = {"core": "Core AI", "platform": "AI platform", "power": "AI build-out", "sector": "AI-exposed sector",
             "linked": "Trades with AI stocks", "none": "Low AI exposure"}[level]
    return {"level": level, "label": label, "why": why, "factor": factor}


def ai_factor_loadings(panel, basket: list[str], tickers: list[str], days: int = 252) -> dict[str, dict]:
    """Regress each stock's daily log returns on SPY and (AI basket − SPY) over the last `days`."""
    from .quant import BENCH
    if panel is None or panel.T < days + 5:
        return {}
    i1 = panel.T
    i0 = i1 - days - 1
    spy = panel.c(BENCH)
    if spy is None:
        return {}
    rs = np.diff(np.log(spy[i0:i1]))
    mem = []
    for b in basket:
        c = panel.c(b)
        if c is None:
            continue
        seg = c[i0:i1]
        if np.isnan(seg).mean() > 0.1:
            continue
        mem.append(np.diff(np.log(seg)))
    if len(mem) < 5:
        return {}
    ai = np.nanmean(np.vstack(mem), axis=0)
    f = ai - rs
    out = {}
    for t in tickers:
        c = panel.c(t)
        if c is None:
            continue
        y = np.diff(np.log(c[i0:i1]))
        ok = np.isfinite(y) & np.isfinite(rs) & np.isfinite(f)
        if ok.sum() < days * 0.8:
            continue
        X = np.column_stack([np.ones(ok.sum()), rs[ok], f[ok]])
        yy = y[ok]
        try:
            b, *_ = np.linalg.lstsq(X, yy, rcond=None)
        except np.linalg.LinAlgError:
            continue
        res = yy - X @ b
        s2 = float(res @ res) / max(1, len(yy) - 3)
        try:
            cov = s2 * np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            continue
        se = math.sqrt(max(cov[2, 2], 1e-18))
        out[t] = {"beta": round(float(b[2]), 3), "t": round(float(b[2]) / se, 2), "mkt_beta": round(float(b[1]), 3)}
    return out


# ---------------------------------------------------------------- bubble risk (stock level)

def price_path_stats(c: np.ndarray, spy: np.ndarray | None, v: np.ndarray | None = None) -> dict | None:
    """Run-up, acceleration, volatility and turnover trends from aligned daily closes (oldest first)."""
    ok = np.isfinite(c) & (c > 0)
    if ok.sum() < 260:
        return None
    idx = np.flatnonzero(ok)
    c2 = c[idx]
    n = len(c2)
    s2 = spy[idx] if spy is not None else None
    out: dict = {"days": int(n)}

    def net(k):
        if n <= k:
            return None
        r = c2[-1] / c2[-1 - k] - 1
        if s2 is not None and np.isfinite(s2[-1]) and np.isfinite(s2[-1 - k]) and s2[-1 - k] > 0:
            return float((1 + r) / (s2[-1] / s2[-1 - k]) - 1)
        return float(r)
    span2 = min(n - 1, 504)
    out["net_1y"] = net(252)
    out["net_2y"] = net(span2) if span2 >= 400 else None
    out["raw_2y"] = float(c2[-1] / c2[-1 - span2] - 1) if span2 >= 400 else None
    out["raw_1y"] = float(c2[-1] / c2[-253] - 1) if n > 252 else None
    out["span_2y_days"] = span2
    out["ret_6m"] = float(c2[-1] / c2[-127] - 1) if n > 127 else None
    lg_total = math.log(c2[-1] / c2[-1 - span2]) if span2 >= 250 else None
    lg_recent = math.log(c2[-1] / c2[-127]) if n > 127 else None
    if lg_total and lg_total > 0 and lg_recent is not None:
        out["accel_share"] = float(lg_recent / lg_total)          # share of the run-up made in the last 6 months
    r = np.diff(np.log(c2))
    if len(r) >= 300:
        recent, before = r[-63:], r[-315:-63]
        out["vol_now"] = float(np.std(recent, ddof=1) * math.sqrt(252))
        out["vol_before"] = float(np.std(before, ddof=1) * math.sqrt(252))
        out["vol_ratio"] = out["vol_now"] / out["vol_before"] if out["vol_before"] > 0 else None
    if v is not None:
        vv = v[idx]
        if len(vv) >= 315:
            a, b = vv[-63:], vv[-315:-63]
            a, b = a[np.isfinite(a) & (a > 0)], b[np.isfinite(b) & (b > 0)]
            if len(a) > 40 and len(b) > 150:
                out["turnover_ratio"] = float(np.mean(a) / np.mean(b))
    hi = float(np.max(c2[-252:]))
    out["from_high"] = float(c2[-1] / hi - 1)
    return out


def gsy_crash(net_2y: float | None) -> float | None:
    """Historical crash frequency for an industry with this 2-year net-of-market run-up (interpolated)."""
    if net_2y is None or net_2y < GSY_CRASH[0][0]:
        return None
    pts = GSY_CRASH
    if net_2y >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= net_2y <= x1:
            return y0 + (y1 - y0) * (net_2y - x0) / (x1 - x0)
    return None


def bubble_risk(stats: dict | None, val: dict | None, sub: dict | None, insiders: dict | None,
                ai: dict | None, theme_level: str | None, today: dt.date | None = None) -> dict:
    """Checklist score → Low / Moderate / Elevated / High, with every sign listed."""
    today = today or now_et().date()
    signs: list[dict] = []
    pts = 0.0

    def add(p: float, title: str, detail: str, kind: str = "warn"):
        nonlocal pts
        pts += p
        signs.append({"points": round(p, 2), "title": title, "detail": detail, "kind": kind})
    st = stats or {}
    # Greenwood, Shleifer & You count a run-up only when both the raw and the net-of-market return clear the bar
    run2 = _both(st.get("net_2y"), st.get("raw_2y"))
    run1 = _both(st.get("net_1y"), st.get("raw_1y"))
    big = False
    if run2 is not None:
        months = round(st.get("span_2y_days", 504) / 21)
        desc = (f"Up {st['raw_2y']*100:.0f}% over {months} months ({st['net_2y']*100:.0f}% more than the S&P 500)"
                if st.get("raw_2y") is not None and st.get("net_2y") is not None else "")
        if run2 >= 1.5:
            add(3, "Huge run-up", desc)
        elif run2 >= 1.0:
            add(2.5, "Very large run-up", desc)
        elif run2 >= 0.5:
            add(1.5, "Large run-up", desc)
        elif run2 >= 0.25:
            add(0.5, "Strong run-up", desc)
        big = run2 >= 0.5
    elif run1 is not None:
        desc = (f"Up {st['raw_1y']*100:.0f}% in a year ({st['net_1y']*100:.0f}% more than the S&P 500)"
                if st.get("raw_1y") is not None and st.get("net_1y") is not None else "")
        if run1 >= 1.0:
            add(2, "Very large 1-year run-up", desc)
        elif run1 >= 0.5:
            add(1, "Large 1-year run-up", desc)
        big = run1 >= 0.5
    if big and (st.get("accel_share") or 0) >= 0.5:
        add(1, "Accelerating", f"{st['accel_share']*100:.0f}% of the gain came in the last 6 months")
    if big and (st.get("vol_ratio") or 0) >= 1.3:
        add(0.5, "Volatility rising", f"Swings are {st['vol_ratio']:.1f}× the prior year's")
    if big and (st.get("turnover_ratio") or 0) >= 1.5:
        add(0.5, "Trading frenzy", f"Volume is {st['turnover_ratio']:.1f}× the prior year's average")
    v = val or {}
    dil = v.get("dilution")
    offers = (sub or {}).get("offerings_12m") or 0
    if (dil is not None and dil >= 0.05) or offers >= 2:
        bits = []
        if dil is not None and dil >= 0.05:
            bits.append(f"share count up {dil*100:.0f}% in a year")
        if offers >= 2:
            bits.append(f"{offers} offering filings in 12 months")
        add(1, "Issuing stock", "; ".join(bits).capitalize())
    first = parse_date((sub or {}).get("first_filed"))
    if big and first and (today - first).days < 5 * 365:
        add(0.5, "Young company", f"First SEC filing {first.year}")
    rich = v.get("rich_pctl")
    if rich is not None and rich >= 90:
        add(1.5, "Very expensive", f"Pricier than {rich:.0f}% of companies relative to its fundamentals")
    elif rich is not None and rich >= 75:
        add(0.75, "Expensive", f"Pricier than {rich:.0f}% of companies relative to its fundamentals")
    if (v.get("ps") or 0) >= 25:
        add(0.5, "Extreme sales multiple", f"Market value is {v['ps']:.0f}× yearly sales")
    ins = insiders or {}
    if ins.get("sellers", 0) >= 3 and not ins.get("buyers"):
        add(0.5, "Insiders selling", f"{ins['sellers']} insiders sold or filed to sell in 90 days; none bought")
    lvl = (ai or {}).get("level")
    if lvl in ("core", "platform", "power", "linked") and theme_level in ("Elevated", "High"):
        add(1 if lvl == "core" else 0.5, "Tied to a stretched AI theme",
            f"{(ai or {}).get('label')}; the AI basket's bubble gauge reads {theme_level}")
    if (st.get("from_high") or 0) <= -0.3 and big:
        signs.append({"points": 0, "title": "Already falling hard", "kind": "info",
                      "detail": f"{-st['from_high']*100:.0f}% below its 52-week high: a deflating run-up"})
    level = "High" if pts >= 5.5 else "Elevated" if pts >= 3.5 else "Moderate" if pts >= 1.5 else "Low"
    return {"level": level, "score": round(pts, 2), "signs": signs,
            "gsy_crash_2y": gsy_crash(run2) if run2 is not None else None, "runup_2y": run2,
            "stats": {k: st.get(k) for k in ("net_1y", "net_2y", "raw_1y", "raw_2y", "accel_share", "vol_ratio",
                                               "turnover_ratio", "from_high", "ret_6m", "span_2y_days")}}


def _both(net: float | None, raw: float | None) -> float | None:
    """The run-up that clears both bars: the smaller of the raw and net-of-market returns."""
    if net is None:
        return None
    return net if raw is None else min(net, raw)


def insider_flow(events: list[dict], t: str, days: int = 90, upto: str | None = None) -> dict:
    """Distinct insiders buying / selling in the last `days` (filed on or before `upto`, a UTC ISO time)."""
    cutoff = (now_utc() - dt.timedelta(days=days)).date().isoformat()
    buyers, sellers = set(), set()
    for e in events:
        if e.get("ticker") != t or e.get("kind") != "insider" or (e.get("filed_at") or "")[:10] < cutoff:
            continue
        if upto is not None:
            fa = parse_iso(e.get("filed_at"))
            if fa is None or iso(fa) > upto:
                continue
        if e.get("action") == "buy" and e.get("signal") != "noise":
            buyers.add(e["actor"]["id"])
        elif e.get("action") in ("sell", "sale_notice"):
            sellers.add(e["actor"]["id"])
    return {"buyers": len(buyers), "sellers": len(sellers)}


# ---------------------------------------------------------------- AI theme gauge (market level)

def ai_theme(panel, basket: list[str], vals: dict[str, dict] | None = None) -> dict | None:
    """Equal-weight AI basket vs the S&P 500: the same warning signs as above, at the theme level, plus breadth
    and how far it has come off its high (is it still inflating, or deflating?)."""
    from .quant import BENCH
    if panel is None or panel.T < 300:
        return None
    spy = panel.c(BENCH)
    members, rets = [], []
    for t in basket:
        c = panel.c(t)
        if c is None:
            continue
        ok = np.isfinite(c) & (c > 0)
        if ok[-260:].mean() < 0.95:
            continue
        lr = np.diff(np.log(np.where(ok, c, np.nan)))
        members.append(t)
        rets.append(lr)
    if len(members) < 5 or spy is None:
        return None
    R = np.expm1(np.vstack(rets))                     # simple returns: an equal-weight basket rebalanced daily
    daily = np.nanmean(R, axis=0)
    daily = np.where(np.isfinite(daily), daily, 0.0)
    idx = np.concatenate([[1.0], np.cumprod(1.0 + daily)])
    st = price_path_stats(idx, spy)
    breadth = []
    for t in members:
        c = panel.c(t)
        last = c[-1]
        ma = np.nanmean(c[-200:])
        if np.isfinite(last) and np.isfinite(ma):
            breadth.append(last > ma)
    worst = []
    for t, lr in zip(members, rets):
        seg = lr[-21:]
        k = int(np.nanargmin(seg)) if np.isfinite(seg).any() else None
        if k is not None and seg[k] <= -0.08:
            worst.append({"t": t, "day": panel.date(panel.T - 21 + k), "move": float(math.expm1(seg[k]))})
    worst.sort(key=lambda w: w["move"])
    vlist = [vals.get(t) for t in members] if vals else []
    vlist = [v for v in vlist if v]
    med_ev_s = float(np.median([v["ev_s"] for v in vlist if v.get("ev_s")])) if any(v.get("ev_s") for v in vlist) else None
    very = sum(1 for v in vlist if v.get("label") == "Very expensive")
    pts = 0.0
    run2 = _both((st or {}).get("net_2y"), (st or {}).get("raw_2y"))
    if run2 is not None:
        pts += 3 if run2 >= 1.5 else 2.5 if run2 >= 1.0 else 1.5 if run2 >= 0.5 else 0.5 if run2 >= 0.25 else 0
    if (st or {}).get("accel_share", 0) >= 0.5 and (run2 or 0) >= 0.5:
        pts += 1
    if vlist and very / len(vlist) >= 0.4:
        pts += 1.5
    elif vlist and very / len(vlist) >= 0.2:
        pts += 0.75
    if (st or {}).get("vol_ratio", 0) >= 1.3:
        pts += 0.5
    level = "High" if pts >= 5 else "Elevated" if pts >= 3 else "Moderate" if pts >= 1.5 else "Low"
    fh = (st or {}).get("from_high")
    phase = ("Deflating" if fh is not None and fh <= -0.2 else "Wobbling" if fh is not None and fh <= -0.1
             else "Inflating" if (run2 or 0) >= 0.5 else "Calm")
    return {"level": level, "score": round(pts, 2), "phase": phase, "members": members, "n": len(members),
            "stats": st, "breadth_above_200d": (sum(breadth) / len(breadth)) if breadth else None,
            "big_down_days": worst[:8], "median_ev_s": med_ev_s,
            "very_expensive_share": (very / len(vlist)) if vlist else None,
            "gsy_crash_2y": gsy_crash(run2), "asof": panel.date(panel.T - 1),
            "series": [round(float(x), 4) for x in idx[-260:][::5]]}
