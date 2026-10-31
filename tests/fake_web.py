"""A fake internet for offline end-to-end runs: maps the pipeline's URLs to fixture responses.

Used by tests (and to render demo data for UI screenshots). Nothing here is shipped to the live app."""
from __future__ import annotations

import datetime as dt
import io
import json
import math
import re
import zipfile
from pathlib import Path

from pipeline.http import Response

FX = Path(__file__).parent / "fixtures"


def _sgml(acc: str, accepted: str, filed: str, docs: list[tuple[str, str, str]]) -> str:
    out = [f"<SEC-DOCUMENT>{acc}.txt : {filed}", f"<ACCEPTANCE-DATETIME>{accepted}",
           f"ACCESSION NUMBER:\t\t{acc}", f"FILED AS OF DATE:\t\t{filed}"]
    for typ, fname, xml in docs:
        out.append(f"<DOCUMENT>\n<TYPE>{typ}\n<SEQUENCE>1\n<FILENAME>{fname}\n<TEXT>\n<XML>\n{xml}\n</XML>\n</TEXT>\n</DOCUMENT>")
    out.append("</SEC-DOCUMENT>")
    return "\n".join(out)


def form4_xml(issuer_cik, issuer, ticker, owner_cik, owner, title, code, rows, director=False, plan=0,
              after=None):
    tx = []
    held = after or 100000
    for d, sh, px in rows:
        ad = "A" if code == "P" else "D"
        tx.append(f"""<nonDerivativeTransaction><securityTitle><value>Common Stock</value></securityTitle>
<transactionDate><value>{d}</value></transactionDate><transactionCoding><transactionFormType>4</transactionFormType>
<transactionCode>{code}</transactionCode><equitySwapInvolved>0</equitySwapInvolved></transactionCoding>
<transactionAmounts><transactionShares><value>{sh}</value></transactionShares><transactionPricePerShare><value>{px}</value>
</transactionPricePerShare><transactionAcquiredDisposedCode><value>{ad}</value></transactionAcquiredDisposedCode></transactionAmounts>
<postTransactionAmounts><sharesOwnedFollowingTransaction><value>{held}</value></sharesOwnedFollowingTransaction></postTransactionAmounts>
<ownershipNature><directOrIndirectOwnership><value>D</value></directOrIndirectOwnership></ownershipNature></nonDerivativeTransaction>""")
    return f"""<ownershipDocument><schemaVersion>X0508</schemaVersion><documentType>4</documentType>
<periodOfReport>{rows[0][0]}</periodOfReport><aff10b5One>{plan}</aff10b5One>
<issuer><issuerCik>{issuer_cik}</issuerCik><issuerName>{issuer}</issuerName><issuerTradingSymbol>{ticker}</issuerTradingSymbol></issuer>
<reportingOwner><reportingOwnerId><rptOwnerCik>{owner_cik}</rptOwnerCik><rptOwnerName>{owner}</rptOwnerName></reportingOwnerId>
<reportingOwnerRelationship><isDirector>{1 if director else 0}</isDirector><isOfficer>{1 if title else 0}</isOfficer>
<isTenPercentOwner>0</isTenPercentOwner><isOther>0</isOther><officerTitle>{title}</officerTitle></reportingOwnerRelationship></reportingOwner>
<nonDerivativeTable>{''.join(tx)}</nonDerivativeTable></ownershipDocument>"""


# accession → (cik, accepted, filed, xml)
FILINGS = {
    "0001999001-26-000011": ("1234567", "20261029170512", "20261029", (FX / "form4_ceo_buy.xml").read_text()),
    "0001999002-26-000003": ("1234567", "20261029181010", "20261029",
                             form4_xml("0001234567", "Acme Robotics Inc", "ACME", "0001999002", "LEE KAREN",
                                       "Chief Financial Officer", "P", [("2026-10-28", 8000, 24.55)], after=20000)),
    "0001999003-26-000001": ("1234567", "20261030091500", "20261030",
                             form4_xml("0001234567", "Acme Robotics Inc", "ACME", "0001999003", "PATEL RAVI", "",
                                       "P", [("2026-10-29", 4000, 25.10)], director=True, after=4000)),
    "0001888002-26-000004": ("1888002", "20261028090144", "20261028", (FX / "form4_plan_sale.xml").read_text()),
    "0001777001-26-000020": ("1045810", "20261030161200", "20261030",
                             form4_xml("0001045810", "NVIDIA CORP", "NVDA", "0001777001", "HUANG JEN HSUN",
                                       "President and CEO", "S", [("2026-10-28", 75000, 182.4)], director=True,
                                       plan=1, after=850000000)),
    "0000921895-26-002222": ("1234567", "20261029160000", "20261029", (FX / "sched13d.xml").read_text()),
}


# Form 144 notices (accession → (filer cik, accepted, filed, xml))
F144 = {
    "0001999001-26-000144": ("1999001", "20261030101500", "20261030", (FX / "form144_officer.xml").read_text()),
    "0001777001-26-000144": ("1777001", "20261030113000", "20261030", (FX / "form144_plan.xml").read_text()),
}


def ark_csv(version: int = 1) -> str:
    """ARK-style holdings file: header, rows dated the *next* trading day, disclaimer paragraphs at the end.
    Version 2 is the following day's file: ACME added, NVDA trimmed, BAC sold out, NEWC opened, flow +1%."""
    day = "10/30/2026" if version == 1 else "11/02/2026"
    f = 1.01 if version == 2 else 1.0           # creations scale every position by ~1%
    rows = [("TESLA INC", "TSLA", "88160R101", 3_000_000 * f, 400.0),
            ("ACME ROBOTICS INC", "ACME", "00123A100", (2_000_000 * f) * (1.30 if version == 2 else 1.0), 22.5),
            ("NVIDIA CORP", "NVDA", "67066G104", (900_000 * f) * (0.80 if version == 2 else 1.0), 182.0),
            ("BIG TECH CORP", "BTC1", "00000B100", 500_000 * f, 190.0),
            ("MICROSOFT CORP", "MSFT", "594918104", 400_000 * f, 420.0)]
    if version == 1:
        rows.append(("BANK OF AMERICA", "BAC", "060505104", 1_500_000, 40.0))
    else:
        rows.append(("NEWCO HOLDINGS", "NEWC", "65249B109", 700_000, 110.0))
    cash = 1_000_000.0
    total = sum(sh * px for _, _, _, sh, px in rows) + cash
    lines = ["date,fund,company,ticker,cusip,shares,\"market value ($)\",\"weight (%)\""]
    for co, t, cu, sh, px in rows:
        lines.append(f"{day},ARKK,{co},{t},{cu},\"{sh:,.0f}\",\"${sh * px:,.2f}\",{sh * px / total * 100:.2f}%")
    lines.append(f"{day},ARKK,MONEY MARKET FUND,,,\"0\",\"${cash:,.2f}\",{cash / total * 100:.2f}%")   # cash row without a ticker
    lines += ["", "\"Investors should carefully consider the investment objectives and risks before investing.\"",
              "\"Holdings are subject to change without notice.\""]
    return "\n".join(lines)


def finra_daily(day: str) -> str:
    seed = int(day[-2:])
    rows = ["Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market"]
    for k, sym in enumerate(["ACME", "NVDA", "AAPL", "BAC", "MSFT"] + [f"Z{i:04d}" for i in range(1200)]):
        tot = 1_000_000 + k * 100
        short = int(tot * (0.62 if (sym == "ACME" and day.endswith("29")) else 0.40 + (seed % 5) * 0.01))
        rows.append(f"{day}|{sym}|{short}|0|{tot}|B,Q,N")
    return "\n".join(rows) + "\n"


def feed_xml(form_prefix: str) -> str:
    entries = []
    if form_prefix == "144":
        for acc, (cik, accepted, filed, xml) in F144.items():
            t = dt.datetime.strptime(accepted, "%Y%m%d%H%M%S").strftime("%Y-%m-%dT%H:%M:%S-04:00")
            entries.append(f"""<entry><title>144 - OWNER ({cik.zfill(10)}) (Reporting)</title>
<link rel="alternate" type="text/html" href="https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}/{acc}-index.htm"/>
<updated>{t}</updated><category scheme="https://www.sec.gov/" label="form type" term="144"/>
<id>urn:tag:sec.gov,2008:accession-number={acc}</id></entry>""")
        return f'<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><title>Latest</title>{"".join(entries)}</feed>'
    for acc, (cik, accepted, filed, xml) in FILINGS.items():
        is13 = "edgarSubmission" in xml
        if form_prefix == "4" and is13:
            continue
        if form_prefix.startswith(("SCHEDULE", "SC")) and not is13:
            continue
        form = "SCHEDULE 13D" if is13 else "4"
        t = dt.datetime.strptime(accepted, "%Y%m%d%H%M%S").strftime("%Y-%m-%dT%H:%M:%S-04:00")
        roles = [("Subject", "Acme Robotics Inc", "0001234567"), ("Filed by", "STARBOARD VALUE LP", "0001517137")] if is13 \
            else [("Reporting", "OWNER", cik.zfill(10))]
        for role, name, c in roles:
            entries.append(f"""<entry><title>{form} - {name} ({c}) ({role})</title>
<link rel="alternate" type="text/html" href="https://www.sec.gov/Archives/edgar/data/{int(c)}/{acc.replace('-', '')}/{acc}-index.htm"/>
<updated>{t}</updated><category scheme="https://www.sec.gov/" label="form type" term="{form}"/>
<id>urn:tag:sec.gov,2008:accession-number={acc}</id></entry>""")
    return f'<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><title>Latest</title>{"".join(entries)}</feed>'


def house_zip(year: int) -> bytes:
    xml = f"""<?xml version="1.0" encoding="utf-8"?><FinancialDisclosure>
<Member><Prefix>Hon.</Prefix><Last>Member</Last><First>Jane Q.</First><Suffix /><FilingType>P</FilingType>
<StateDst>CA11</StateDst><Year>{year}</Year><FilingDate>10/07/{year}</FilingDate><DocID>20029999</DocID></Member>
<Member><Prefix>Hon.</Prefix><Last>Other</Last><First>Sam</First><Suffix /><FilingType>O</FilingType>
<StateDst>TX02</StateDst><Year>{year}</Year><FilingDate>05/15/{year}</FilingDate><DocID>10061234</DocID></Member>
</FinancialDisclosure>"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(f"{year}FD.xml", xml)
        z.writestr(f"{year}FD.txt", "ignored")
    return buf.getvalue()


def house_pdf() -> bytes:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    y = 760
    for line in (FX / "house_ptr.txt").read_text().splitlines():
        c.setFont("Helvetica", 8)
        c.drawString(30, y, line)
        y -= 11
        if y < 40:
            c.showPage()
            y = 760
    c.save()
    return buf.getvalue()


def chart(sym: str, start_price: float, drift: float, vol: float, n: int = 300) -> dict:
    base = dt.datetime(2026, 10, 30, 20, 0, tzinfo=dt.timezone.utc)
    ts, closes = [], []
    p = start_price
    seed = sum(ord(ch) for ch in sym)
    days = []
    d = base
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d -= dt.timedelta(days=1)
    days.reverse()
    for i, day in enumerate(days):
        shock = math.sin(i * 0.7 + seed) * vol + math.cos(i * 0.23 + seed * 3) * vol * 0.6
        p *= math.exp(drift + shock)
        ts.append(int(day.timestamp()))
        closes.append(round(p, 4))
    return {"chart": {"result": [{"meta": {"regularMarketPrice": closes[-1], "regularMarketTime": ts[-1]},
                                  "timestamp": ts, "indicators": {"quote": [{"close": closes, "volume": [2_000_000] * n}],
                                                                   "adjclose": [{"adjclose": closes}]}}], "error": None}}


CHARTS = {"SPY": (520, 0.0004, 0.006), "ACME": (31, -0.0012, 0.022), "NVDA": (120, 0.0015, 0.018),
          "BTC1": (190, 0.0003, 0.012), "AAPL": (210, 0.0002, 0.010), "BAC": (40, 0.0003, 0.011),
          "NEWC": (110, 0.0006, 0.015), "GOOG": (160, 0.0006, 0.012), "MSFT": (420, 0.0003, 0.009),
          "CVX": (150, 0.0001, 0.010), "AMCR": (9.5, -0.0002, 0.012)}


_CLOSES: dict[str, dict[str, float]] = {}


def close_on(sym: str, day: str) -> float | None:
    if sym not in _CLOSES:
        c = chart(sym, *CHARTS[sym])["chart"]["result"][0]
        _CLOSES[sym] = {dt.datetime.fromtimestamp(t, dt.timezone.utc).date().isoformat(): v
                        for t, v in zip(c["timestamp"], c["indicators"]["quote"][0]["close"])}
    return _CLOSES[sym].get(day)


def dera_zip(y: int, q: int) -> bytes | None:
    """Tiny SEC 'Insider Transactions Data Set' with a few open-market buys (for track records)."""
    trades = {  # (year, quarter): [(acc, filed, traded, owner_cik, owner, ticker, issuer_cik, shares, price, code)]
        (2025, 4): [("0001999001-25-000001", "05-DEC-2025", "03-DEC-2025", "0001999001", "DOE JANE A", "ACME", "0001234567", 5000, 30.0, "P"),
                    ("0001999002-25-000001", "09-DEC-2025", "05-DEC-2025", "0001999002", "LEE KAREN", "ACME", "0001234567", 3000, 29.5, "P")],
        (2026, 1): [("0001999001-26-000001", "12-FEB-2026", "10-FEB-2026", "0001999001", "DOE JANE A", "ACME", "0001234567", 4000, 27.0, "P"),
                    ("0001777001-26-000001", "20-FEB-2026", "18-FEB-2026", "0001777001", "HUANG JEN HSUN", "NVDA", "0001045810", 1000, 140.0, "S")],
        (2026, 2): [("0001999001-26-000005", "14-MAY-2026", "12-MAY-2026", "0001999001", "DOE JANE A", "ACME", "0001234567", 2000, 26.0, "P"),
                    ("0001888002-26-000001", "03-JUN-2026", "01-JUN-2026", "0001888002", "SMITH JOHN", "BTC1", "0000320193", 500, 200.0, "P")],
    }.get((y, q))
    if trades is None:
        return None
    sub = ["ACCESSION_NUMBER\tFILING_DATE\tPERIOD_OF_REPORT\tDOCUMENT_TYPE\tISSUERCIK\tISSUERNAME\tISSUERTRADINGSYMBOL\tAFF10B5ONE"]
    own = ["ACCESSION_NUMBER\tRPTOWNERCIK\tRPTOWNERNAME\tRPTOWNER_RELATIONSHIP\tRPTOWNER_TITLE"]
    nd = ["ACCESSION_NUMBER\tNONDERIV_TRANS_SK\tSECURITY_TITLE\tTRANS_DATE\tTRANS_CODE\tTRANS_SHARES\tTRANS_PRICEPERSHARE\tTRANS_ACQUIRED_DISP_CD"]
    for i, (acc, filed, traded, ocik, oname, t, icik, sh, px, code) in enumerate(trades):
        sub.append(f"{acc}\t{filed}\t{traded}\t4\t{icik}\t{t} INC\t{t}\t0")
        own.append(f"{acc}\t{ocik}\t{oname}\tOfficer\tCEO")
        nd.append(f"{acc}\t{i}\tCommon Stock\t{traded}\t{code}\t{sh}\t{px}\t{'A' if code == 'P' else 'D'}")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("SUBMISSION.tsv", "\n".join(sub) + "\n")
        z.writestr("REPORTINGOWNER.tsv", "\n".join(own) + "\n")
        z.writestr("NONDERIV_TRANS.tsv", "\n".join(nd) + "\n")
    return buf.getvalue()


def companyfacts(cik: str) -> dict | None:
    """SEC companyfacts for the fake companies: calendar-year filers with a 10-Q for Q3 2026."""
    spec = {"0001234567": (400e6, 0.10, -0.05, 60e6), "0001045810": (180e9, 0.55, 0.55, 24.4e9),
            "0000320193": (400e9, 0.06, 0.30, 15e9)}.get(cik)
    if not spec:
        return None
    rev_fy, g, om, shares = spec
    def dur(start, end, val, filed, form="10-Q"):
        return {"start": start, "end": end, "val": val, "filed": filed, "form": form, "accn": f"acc-{end}"}
    q25 = rev_fy / 4
    rev = [dur("2024-01-01", "2024-12-31", rev_fy / (1 + g), "2025-02-20", "10-K"),
           dur("2025-01-01", "2025-12-31", rev_fy, "2026-02-20", "10-K"),
           dur("2025-01-01", "2025-09-30", q25 * 3, "2025-11-01"), dur("2024-01-01", "2024-09-30", q25 * 3 / (1 + g), "2024-11-01"),
           dur("2026-01-01", "2026-09-30", q25 * 3 * (1 + g), "2026-10-28")]
    def scaled(m):
        return [{**r, "val": r["val"] * m} for r in rev]
    return {"cik": int(cik), "facts": {
        "us-gaap": {"RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": rev}},
                    "OperatingIncomeLoss": {"units": {"USD": scaled(om)}}, "NetIncomeLoss": {"units": {"USD": scaled(om * 0.8)}},
                    "GrossProfit": {"units": {"USD": scaled(0.6)}},
                    "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": scaled(max(om, 0.02) + 0.05)}},
                    "PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": scaled(0.04)}},
                    "CashAndCashEquivalentsAtCarryingValue": {"units": {"USD": [{"end": "2026-09-30", "val": rev_fy * 0.2, "filed": "2026-10-28", "accn": "a"}]}},
                    "LongTermDebt": {"units": {"USD": [{"end": "2026-09-30", "val": rev_fy * 0.1, "filed": "2026-10-28", "accn": "a"}]}},
                    "StockholdersEquity": {"units": {"USD": [{"end": "2026-09-30", "val": rev_fy * 0.8, "filed": "2026-10-28", "accn": "a"}]}},
                    "WeightedAverageNumberOfDilutedSharesOutstanding": {"units": {"shares": [
                        dur("2026-07-01", "2026-09-30", shares * 1.01, "2026-10-28")]}}},
        "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
            {"end": "2026-10-20", "val": shares, "filed": "2026-10-28", "accn": "b"},
            {"end": "2025-10-20", "val": shares * 0.97, "filed": "2025-10-28", "accn": "c"}]}}}}}


def company_submissions(cik: str) -> dict | None:
    info = {"0001234567": ("Acme Robotics Inc", 3569, "General Industrial Machinery", ["ACME"], "2019-04-01"),
            "0001045810": ("NVIDIA CORP", 3674, "Semiconductors & Related Devices", ["NVDA"], "1998-01-20"),
            "0000320193": ("Big Tech Corp", 3571, "Electronic Computers", ["BTC1"], "1994-01-26")}.get(cik)
    if not info:
        return None
    name, sic, desc, tickers, first = info
    acc = lambda n: f"{cik}-26-{n:06d}"  # noqa: E731
    rows = [("8-K", "2026-10-29", "2.02,9.01"), ("10-Q", "2026-10-28", ""), ("8-K", "2026-08-15", "5.02"),
            ("8-K", "2026-07-29", "2.02,9.01"), ("8-K", "2026-04-29", "2.02,9.01"), ("8-K", "2026-01-28", "2.02,9.01")]
    if cik == "0001234567":
        rows.insert(1, ("8-K", "2026-10-15", "4.02"))
    r = {"accessionNumber": [acc(i + 1) for i in range(len(rows))], "form": [x[0] for x in rows],
         "filingDate": [x[1] for x in rows], "items": [x[2] for x in rows],
         "acceptanceDateTime": [x[1] + "T16:05:00.000Z" for x in rows], "primaryDocument": ["doc.htm"] * len(rows),
         "primaryDocDescription": [x[0] for x in rows], "reportDate": ["2026-09-30" if x[0] == "10-Q" else "" for x in rows]}
    return {"cik": cik, "name": name, "sic": str(sic), "sicDescription": desc, "tickers": tickers, "fiscalYearEnd": "1231",
            "filings": {"recent": r, "files": [{"name": "old.json", "filingFrom": first, "filingTo": "2015-01-01"}]}}


def gdelt(query: str) -> str:
    if "Nvidia" in query or "NVDA" in query:
        return json.dumps({"articles": [
            {"url": "https://www.reuters.com/tech/nvidia-results", "title": "Nvidia results top estimates as AI demand holds",
             "seendate": "20261030T120000Z", "domain": "reuters.com", "sourcecountry": "United States"},
            {"url": "https://www.cnbc.com/nvidia-ai-bubble", "title": "Is the AI trade a bubble? Nvidia in focus",
             "seendate": "20261029T150000Z", "domain": "cnbc.com", "sourcecountry": "United States"},
            {"url": "https://www.wsj.com/chips-slide", "title": "Chip stocks slide after Alphabet's cash-flow miss",
             "seendate": "20261029T100000Z", "domain": "wsj.com", "sourcecountry": "United States"}]})
    if "Acme" in query:
        return "Please limit requests to one every 5 seconds or contact kalev.leetaru5@gmail.com for larger queries."
    return "{}"


def eightk_feed() -> str:
    return ('<?xml version="1.0" encoding="ISO-8859-1" ?><feed xmlns="http://www.w3.org/2005/Atom">'
            '<entry><title>8-K - NVIDIA CORP (0001045810) (Filer)</title>'
            '<link rel="alternate" type="text/html" href="https://www.sec.gov/Archives/edgar/data/1045810/000104581026000099/0001045810-26-000099-index.htm"/>'
            '<summary type="html"> &lt;b&gt;Filed:&lt;/b&gt; 2026-10-30 &lt;b&gt;AccNo:&lt;/b&gt; 0001045810-26-000099 Item 7.01: Regulation FD</summary>'
            '<updated>2026-10-30T16:20:00-04:00</updated><category scheme="https://www.sec.gov/" label="form type" term="8-K"/>'
            '<id>urn:tag:sec.gov,2008:accession-number=0001045810-26-000099</id></entry>'
            '<entry><title>8-K - SOMEONE ELSE INC (0000000042) (Filer)</title>'
            '<summary type="html">x</summary><updated>2026-10-30T16:00:00-04:00</updated>'
            '<id>urn:tag:sec.gov,2008:accession-number=0000000042-26-000001</id></entry></feed>')


class FakeWeb:
    def __init__(self):
        self.calls: list[str] = []
        self.ark_version = 1          # bump to 2 to serve the next day's ARK holdings file

    def __call__(self, method, url, headers, data, timeout):
        self.calls.append(url)
        J = lambda obj, st=200: Response(st, {"Content-Type": "application/json"}, json.dumps(obj).encode(), url)  # noqa: E731
        T = lambda txt, st=200: Response(st, {}, txt.encode("latin-1", errors="replace"), url)  # noqa: E731
        if "browse-edgar?action=getcurrent" in url and "type=8-K" in url:
            return T(eightk_feed() if "start=0" in url else '<feed xmlns="http://www.w3.org/2005/Atom"></feed>')
        m = re.search(r"/api/xbrl/companyfacts/CIK(\d{10})\.json", url)
        if m:
            cf = companyfacts(m.group(1))
            return J(cf) if cf else J({"message": "not found"}, 404)
        if "api.gdeltproject.org/api/v2/doc/doc" in url:
            from urllib.parse import unquote
            return T(gdelt(unquote(re.search(r"query=([^&]+)", url).group(1))))
        if "browse-edgar?action=getcurrent" in url:
            m = re.search(r"type=([^&]*)", url)
            prefix = m.group(1).replace("%20", " ") if m else ""
            start = int(re.search(r"start=(\d+)", url).group(1))
            return T(feed_xml(prefix) if start == 0 else '<feed xmlns="http://www.w3.org/2005/Atom"></feed>')
        m = re.search(r"/Archives/edgar/data/\d+/\d+/(\d{10}-\d{2}-\d{6})\.txt$", url)
        if m:
            acc = m.group(1)
            if acc in FILINGS:
                cik, accepted, filed, xml = FILINGS[acc]
                return T(_sgml(acc, accepted, filed, [("4" if "ownership" in xml else "SCHEDULE 13D", "doc.xml", xml)]))
            if acc in F13:
                return T(F13[acc])
            if acc in F144:
                cik, accepted, filed, xml = F144[acc]
                return T(_sgml(acc, accepted, filed, [("144", "primary_doc.xml", xml)]))
            return T("not found", 404)
        if "company_tickers_exchange.json" in url:
            rows = [[1234567, "Acme Robotics Inc", "ACME", "Nasdaq"], [1045810, "NVIDIA CORP", "NVDA", "Nasdaq"],
                    [320193, "Big Tech Corp", "BTC1", "Nasdaq"], [1067983, "BERKSHIRE HATHAWAY INC", "BRK.B", "NYSE"]]
            return J({"fields": ["cik", "name", "ticker", "exchange"], "data": rows})
        if "/api/xbrl/frames/dei/EntityCommonStockSharesOutstanding" in url:
            return J({"data": [{"cik": 1234567, "val": 60_000_000, "end": "2026-09-30"},
                               {"cik": 1045810, "val": 24_400_000_000, "end": "2026-07-25"},
                               {"cik": 320193, "val": 15_000_000_000, "end": "2026-09-27"}]})
        if "/api/xbrl/frames/us-gaap/NetIncomeLoss" in url:
            return J({"data": [{"cik": 1234567, "val": -12_000_000}, {"cik": 1045810, "val": 72_000_000_000},
                               {"cik": 320193, "val": 95_000_000_000}]})
        if "/api/xbrl/frames/" in url:
            return J({"data": []})
        m = re.search(r"submissions/CIK(\d{10})\.json", url)
        if m:
            cik = m.group(1)
            if cik == "0001067983":
                return J({"cik": cik, "name": "BERKSHIRE HATHAWAY INC", "filings": {"recent": {
                    "accessionNumber": ["0000950123-26-008888", "0000950123-26-005555"],
                    "filingDate": ["2026-08-14", "2026-05-15"], "reportDate": ["2026-06-30", "2026-03-31"],
                    "form": ["13F-HR", "13F-HR"], "primaryDocument": ["primary_doc.xml", "primary_doc.xml"]}}})
            if cik == "0001336528":
                return J({"cik": cik, "name": "SOMEBODY ELSE LLC", "filings": {"recent": {"accessionNumber": [], "form": []}}})
            cs = company_submissions(cik)
            if cs:
                return J(cs)
            return J({"cik": cik, "name": "UNKNOWN", "filings": {"recent": {}}}, 404)
        if "daily-index" in url:
            return T((FX / "master.idx").read_text())
        if "api.openfigi.com" in url:
            table = {"037833100": "AAPL", "060505104": "BAC", "166764100": "CVX", "65249B109": "NEWC",
                     "680223104": "OLDC", "78462F103": "SPY"}
            return J([{"data": [{"ticker": table.get(j["idValue"]), "name": j["idValue"], "securityType": "Common Stock",
                                 "exchCode": "US"}]} if j["idValue"] in table else {"warning": "No identifier found."}
                      for j in data])
        if "disclosures-clerk.house.gov" in url and url.endswith("FD.zip"):
            return Response(200, {}, house_zip(int(re.search(r"(\d{4})FD\.zip", url).group(1))), url)
        if "disclosures-clerk.house.gov" in url and url.endswith(".pdf"):
            return Response(200, {}, house_pdf(), url)
        if "efdsearch.senate.gov" in url:
            return T("Access Denied", 403)
        if "congress-legislators" in url:
            return J([{"name": {"first": "Jane", "last": "Member", "official_full": "Jane Q. Member"},
                       "id": {"bioguide": "M000001"}, "terms": [{"type": "rep", "state": "CA", "district": 11, "party": "Democrat"}]}])
        if "finance.yahoo.com/v8/finance/chart/" in url:
            sym = url.split("/chart/")[1].split("?")[0].replace("-", ".")
            if sym in CHARTS:
                return J(chart(sym, *CHARTS[sym]))
            return J({"chart": {"result": None, "error": {"code": "Not Found"}}}, 404)
        if "ntfy.sh" in url:
            return T("ok")
        if "assets.ark-funds.com" in url:
            if "ARKK_HOLDINGS.csv" in url:
                return T(ark_csv(self.ark_version))
            return T("<html>not found</html>", 404)
        m = re.search(r"cdn\.finra\.org/equity/regsho/daily/CNMSshvol(\d{8})\.txt", url)
        if m:
            day = m.group(1)
            if day in ("20261029", "20261028", "20261027", "20261026", "20261023", "20261022"):
                return T(finra_daily(day))
            return T("<Error>NoSuchKey</Error>", 403)
        m = re.search(r"cdn\.finra\.org/equity/otcmarket/biweekly/shrt(\d{8})\.csv", url)
        if m:
            if m.group(1) != "20261015":
                return T("<Error>NoSuchKey</Error>", 403)
            head = ("accountingYearMonthNumber|symbolCode|issueName|issuerServicesGroupExchangeCode|marketClassCode|"
                    "currentShortPositionQuantity|previousShortPositionQuantity|stockSplitFlag|averageDailyVolumeQuantity|"
                    "daysToCoverQuantity|revisionFlag|changePercent|changePreviousNumber|settlementDate")
            rows = [f"20261015|{sym}|{sym} INC|A|NYSE|{1_000_000 + k * 1000}|{900_000 + k * 1000}||{500_000}|2.5||11.1|100000|2026-10-15"
                    for k, sym in enumerate(["ACME", "NVDA", "AAPL"] + [f"Z{i:04d}" for i in range(600)])]
            return T("\n".join([head] + rows))
        if "alphavantage.co/query" in url and "EARNINGS_CALENDAR" in url:
            return T("symbol,name,reportDate,fiscalDateEnding,estimate,currency,timeOfTheDay\n"
                     "ACME,Acme Robotics,2026-11-05,2026-09-30,0.12,USD,post-market\n"
                     "NVDA,NVIDIA Corp,2026-11-19,2026-10-31,1.05,USD,post-market\n"
                     "NVDA,NVIDIA Corp,2027-02-25,2027-01-31,1.10,USD,post-market\n"
                     "OLD,Old Co,2026-10-01,2026-09-30,,USD,\n")
        if "api.massive.com/stocks/v1/splits" in url:
            # one split for a ticker we don't store, one long before our history starts: both must be no-ops
            return J({"status": "OK", "results": [
                {"id": "Esplit1", "ticker": "ZZZZ", "execution_date": "2026-09-15", "split_from": 1, "split_to": 4,
                 "adjustment_type": "forward_split", "historical_adjustment_factor": 0.25},
                {"id": "Esplit2", "ticker": "AAPL", "execution_date": "2020-08-31", "split_from": 1, "split_to": 4,
                 "adjustment_type": "forward_split", "historical_adjustment_factor": 0.25}]})
        m = re.search(r"/v2/aggs/grouped/locale/us/market/stocks/(\d{4}-\d{2}-\d{2})", url)
        if m:
            day = m.group(1)
            if day < "2024-10-30":
                return J({"status": "NOT_AUTHORIZED"}, 403)
            res = [{"T": sym, "c": close_on(sym, day), "v": 1_000_000} for sym in CHARTS if close_on(sym, day)]
            return J({"status": "OK", "resultsCount": len(res), "results": res})
        if "data.alpaca.markets/v2/stocks/snapshots" in url:
            syms = re.search(r"symbols=([^&]+)", url).group(1).split(",")
            out = {}
            for sym in syms:
                c = close_on(sym, "2026-10-30") if sym in CHARTS else None
                if c:
                    out[sym] = {"latestTrade": {"p": round(c * 1.002, 2), "t": "2026-10-30T19:55:00Z"},
                                "dailyBar": {"c": c, "t": "2026-10-30T04:00:00Z"}}
            return J(out)
        m = re.search(r"insider-transactions-data-sets/(\d{4})q(\d)_form345\.zip", url)
        if m:
            if "datastandardsinnovation" not in url:
                return T("moved", 404)
            z = dera_zip(int(m.group(1)), int(m.group(2)))
            return Response(200, {}, z, url) if z else T("not found", 404)
        return T(f"unmapped {url}", 404)


def _infotable_txt(acc: str, filed: str, table: str, period: str) -> str:
    cover = f"""<edgarSubmission xmlns="http://www.sec.gov/edgar/thirteenffiler"><headerData><submissionType>13F-HR</submissionType>
<filerInfo><periodOfReport>{period}</periodOfReport></filerInfo></headerData><formData><coverPage><reportCalendarOrQuarter>{period}</reportCalendarOrQuarter>
<isAmendment>false</isAmendment><filingManager><name>Berkshire Hathaway Inc</name></filingManager><reportType>13F HOLDINGS REPORT</reportType></coverPage>
<summaryPage><tableEntryTotal>5</tableEntryTotal><tableValueTotal>100</tableValueTotal></summaryPage></formData></edgarSubmission>"""
    return _sgml(acc, filed.replace("-", "") + "160000", filed.replace("-", ""),
                 [("13F-HR", "primary_doc.xml", cover), ("INFORMATION TABLE", "infotable.xml",
                                                         (FX / table).read_text().split("?>", 1)[-1])])


F13 = {
    "0000950123-26-008888": _infotable_txt("0000950123-26-008888", "2026-08-14", "infotable_2026q2.xml", "06-30-2026"),
    "0000950123-26-005555": _infotable_txt("0000950123-26-005555", "2026-05-15", "infotable_2026q1.xml", "03-31-2026"),
}
