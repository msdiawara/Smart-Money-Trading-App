"""SEC EDGAR access: real-time current-events feed, daily index backstop, full-submission documents,
submissions API, XBRL frames and the ticker map. All official and free (fair access: <=10 req/s with a
declared User-Agent, which http.HttpClient enforces)."""
from __future__ import annotations

import datetime as dt
import html
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from urllib.parse import quote

from .http import HttpClient, HttpError
from .util import ET as TZ_ET
from .util import log, parse_iso

ATOM = "{http://www.w3.org/2005/Atom}"
SEC = "https://www.sec.gov"


@dataclass
class FeedEntry:
    form: str
    name: str
    cik: str
    role: str              # Issuer / Reporting / Subject / Filed by / Filer
    accession: str
    updated: str | None     # ISO time the filing was accepted/updated
    link: str

    @property
    def folder(self) -> str:
        return f"{SEC}/Archives/edgar/data/{int(self.cik)}/{self.accession.replace('-', '')}"


_TITLE = re.compile(r"^(?P<form>.+?) - (?P<name>.*) \((?P<cik>\d{4,10})\) \((?P<role>[^)]+)\)\s*$")
_ACC = re.compile(r"(\d{10}-\d{2}-\d{6})")


def parse_current_feed(xml_bytes: bytes) -> list[FeedEntry]:
    root = ET.fromstring(xml_bytes)
    out: list[FeedEntry] = []
    for e in root.findall(f"{ATOM}entry"):
        title = (e.findtext(f"{ATOM}title") or "").strip()
        m = _TITLE.match(title)
        cat = e.find(f"{ATOM}category")
        form = (cat.get("term") if cat is not None else None) or (m.group("form") if m else "")
        link_el = e.find(f"{ATOM}link")
        link = link_el.get("href") if link_el is not None else ""
        id_txt = e.findtext(f"{ATOM}id") or ""
        acc_m = _ACC.search(id_txt) or _ACC.search(link) or _ACC.search(e.findtext(f"{ATOM}summary") or "")
        if not acc_m:
            continue
        out.append(FeedEntry(
            form=form.strip(),
            name=html.unescape(m.group("name")) if m else title,
            cik=(m.group("cik") if m else "").zfill(10),
            role=m.group("role") if m else "",
            accession=acc_m.group(1),
            updated=e.findtext(f"{ATOM}updated"),
            link=link,
        ))
    return out


def current_feed(http: HttpClient, form_prefix: str, pages: int = 5, count: int = 100,
                 stop_at: set[str] | None = None) -> list[FeedEntry]:
    """Newest-first entries from EDGAR's real-time 'Latest Filings' Atom feed.

    The `type` filter is a *prefix* match, so callers must filter exact form types themselves.
    Stops paging once a page contains only accessions in `stop_at` (already processed)."""
    entries: list[FeedEntry] = []
    for page in range(pages):
        url = (f"{SEC}/cgi-bin/browse-edgar?action=getcurrent&CIK=&type={quote(form_prefix)}&company="
               f"&dateb=&owner=include&start={page * count}&count={count}&output=atom")
        batch = parse_current_feed(http.get(url).content)
        if not batch:
            break
        entries.extend(batch)
        if stop_at is not None and all(b.accession in stop_at for b in batch):
            break
        if len(batch) < count:
            break
    return entries


# ---------------------------------------------------------------- daily index (backstop)

_SEC_BLOCK = ("Undeclared Automated Tool", "Request Rate Threshold", "Your Request Originates")


def daily_master_index(http: HttpClient, day: dt.date) -> list[dict] | None:
    """Rows of master.YYYYMMDD.idx (pipe-delimited): cik, company, form, date_filed, filename.
    Returns None if the file doesn't exist (weekend/holiday/not yet published). Real failures — SEC blocking
    us, server errors after retries, network errors — raise, so the run reports them instead of hiding them."""
    q = (day.month - 1) // 3 + 1
    url = f"{SEC}/Archives/edgar/daily-index/{day.year}/QTR{q}/master.{day:%Y%m%d}.idx"
    resp = http.get(url, ok_statuses=(404, 403))
    if resp.status in (404, 403):
        body = resp.text("latin-1")
        if resp.status == 403 and any(m in body for m in _SEC_BLOCK):
            raise HttpError(403, url, "SEC refused the request (check SEC_CONTACT_EMAIL / request rate)")
        return None
    txt = resp.text("latin-1")
    rows = []
    started = False
    for line in txt.splitlines():
        if not started:
            if line.startswith("-----"):
                started = True
            continue
        parts = line.split("|")
        if len(parts) != 5 or not parts[0].strip().isdigit():
            continue
        cik, company, form, date_filed, filename = (p.strip() for p in parts)
        acc_m = _ACC.search(filename)
        rows.append({"cik": cik.zfill(10), "company": company, "form": form, "date_filed": date_filed,
                     "filename": filename, "accession": acc_m.group(1) if acc_m else None})
    return rows


# ---------------------------------------------------------------- full submission documents

@dataclass
class Submission:
    accession: str
    header: dict           # parsed SGML header fields (ACCEPTANCE-DATETIME, FILED AS OF DATE, ...)
    documents: list[dict]  # {type, filename, description, text, xml}

    def accepted_at(self) -> dt.datetime | None:
        v = self.header.get("ACCEPTANCE-DATETIME")
        if v and re.fullmatch(r"\d{14}", v):
            return dt.datetime.strptime(v, "%Y%m%d%H%M%S").replace(tzinfo=TZ_ET)
        return None

    def first_xml(self, *, root_contains: str | None = None, skip_filenames: tuple = ()) -> str | None:
        for d in self.documents:
            x = d.get("xml")
            if not x:
                continue
            if any(d.get("filename", "").lower() == s for s in skip_filenames):
                continue
            if root_contains and root_contains.lower() not in x[:3000].lower():
                continue
            return x
        return None


_DOC = re.compile(r"<DOCUMENT>(.*?)</DOCUMENT>", re.S | re.I)
_XML = re.compile(r"<XML>\s*(.*?)\s*</XML>", re.S | re.I)


def _tag(block: str, name: str) -> str:
    m = re.search(rf"<{name}>([^\n<]*)", block, re.I)
    return m.group(1).strip() if m else ""


def parse_submission_text(acc: str, txt: str) -> Submission:
    head_end = txt.find("<DOCUMENT>")
    head = txt[:head_end if head_end > 0 else 4000]
    header: dict = {}
    for line in head.splitlines():
        line = line.strip()
        m = re.match(r"<(ACCEPTANCE-DATETIME)>(\d+)", line)
        if m:
            header[m.group(1)] = m.group(2)
            continue
        if ":" in line and not line.startswith("<"):
            k, _, v = line.partition(":")
            k, v = k.strip().upper(), v.strip()
            if k and v and k not in header:
                header[k] = v
    docs = []
    for block in _DOC.findall(txt):
        xm = _XML.search(block)
        xml = xm.group(1) if xm else None
        if xml and xml.lstrip().startswith("<?xml"):
            # keep declaration-free XML so ElementTree accepts str input with encoding decl
            xml = re.sub(r"^\s*<\?xml[^>]*\?>", "", xml, count=1)
        docs.append({"type": _tag(block, "TYPE"), "filename": _tag(block, "FILENAME"),
                     "description": _tag(block, "DESCRIPTION"), "xml": xml,
                     "text": None if xml else block[:200000]})
    return Submission(accession=acc, header=header, documents=docs)


def fetch_submission(http: HttpClient, cik: str, accession: str) -> Submission:
    url = f"{SEC}/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/{accession}.txt"
    return parse_submission_text(accession, http.get(url).text("latin-1"))


def filing_index_url(cik: str, accession: str) -> str:
    return f"{SEC}/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/{accession}-index.htm"


# ---------------------------------------------------------------- APIs

def submissions(http: HttpClient, cik: str) -> dict:
    return http.get_json(f"https://data.sec.gov/submissions/CIK{str(cik).zfill(10)}.json")


def recent_filings(sub: dict, forms: set[str]) -> list[dict]:
    r = (sub.get("filings") or {}).get("recent") or {}
    keys = list(r.keys())
    n = len(r.get("accessionNumber", []))
    out = []
    for i in range(n):
        row = {k: (r[k][i] if i < len(r[k]) else None) for k in keys}
        if row.get("form") in forms:
            out.append(row)
    return out


def frames(http: HttpClient, taxonomy: str, tag: str, unit: str, period: str) -> list[dict]:
    url = f"https://data.sec.gov/api/xbrl/frames/{taxonomy}/{tag}/{unit}/{period}.json"
    try:
        return http.get_json(url).get("data") or []
    except HttpError as e:
        if e.status == 404:
            return []
        raise


def ticker_map(http: HttpClient) -> dict:
    """{'by_ticker': {T: {cik,name,exchange}}, 'by_cik': {cik10: {ticker,name,exchange}}}"""
    d = http.get_json(f"{SEC}/files/company_tickers_exchange.json")
    fields = d.get("fields") or ["cik", "name", "ticker", "exchange"]
    by_t, by_c = {}, {}
    for row in d.get("data") or []:
        rec = dict(zip(fields, row))
        t = (rec.get("ticker") or "").upper()
        c = str(rec.get("cik") or "").zfill(10)
        if not t:
            continue
        by_t.setdefault(t, {"cik": c, "name": rec.get("name"), "exchange": rec.get("exchange")})
        # first ticker listed for a CIK is its primary share class
        by_c.setdefault(c, {"ticker": t, "name": rec.get("name"), "exchange": rec.get("exchange")})
    return {"by_ticker": by_t, "by_cik": by_c}


def accepted_iso(entry_updated: str | None) -> str | None:
    d = parse_iso(entry_updated)
    return d.isoformat() if d else None


def log_feed_stats(name: str, entries: list[FeedEntry]) -> None:
    log.info("%s: %d feed entries (%d unique accessions)", name, len(entries),
             len({e.accession for e in entries}))
