"""Congress (STOCK Act periodic transaction reports).

House: official House Clerk index ZIP + PDFs (parsed here).
Senate: official eFD first; it is known to block many cloud IPs, so optional fallbacks (FMP, Quiver) are
used when their keys are configured. Every event links to the original filing."""
from __future__ import annotations

import datetime as dt
import io
import re
import zipfile
import xml.etree.ElementTree as ET
from html import unescape

from .http import HttpClient, HttpError
from .util import clean, date_str, log, now_utc, parse_date, slug, to_float

HOUSE = "https://disclosures-clerk.house.gov"
EFD = "https://efdsearch.senate.gov"
LEGISLATORS = "https://unitedstates.github.io/congress-legislators/legislators-current.json"

AMOUNT_RX = re.compile(r"(?:Over\s+)?\$\s?([\d,]+)(?:\s*-\s*\$?\s?([\d,]+))?", re.I)
OWNER = {"SP": "Spouse", "JT": "Joint", "DC": "Dependent child", "": "Self", "SELF": "Self"}


def parse_amount(txt: str | None) -> tuple[float | None, float | None]:
    if not txt:
        return None, None
    t = txt.replace("–", "-").replace("—", "-")
    m = AMOUNT_RX.search(t)
    if not m:
        return None, None
    lo = to_float(m.group(1))
    hi = to_float(m.group(2)) if m.group(2) else None
    if hi is None and re.search(r"over", t, re.I):
        return lo, None
    return lo, hi if hi is not None else lo


def tx_action(t: str | None) -> tuple[str, str]:
    s = (t or "").strip().lower()
    if s.startswith("p") or "purchase" in s or s == "buy":
        return "buy", "Purchase"
    if "partial" in s:
        return "sell", "Sale (partial)"
    if s.startswith("s") or "sale" in s or s == "sell":
        return "sell", "Sale"
    if s.startswith("e") or "exchange" in s:
        return "exchange", "Exchange"
    return "other", (t or "Other").strip().title()


# ---------------------------------------------------------------- member directory (party etc.)

def load_legislators(http: HttpClient) -> dict:
    """{'by_last': {lastname_lower: [ {name, party, state, chamber, district, bioguide} ]}}"""
    try:
        data = http.get_json(LEGISLATORS)
    except (HttpError, ValueError) as e:
        log.warning("legislators list unavailable: %s", e)
        return {"by_last": {}}
    by_last: dict[str, list] = {}
    leaders: dict[str, str] = {}
    today = now_utc().date().isoformat()
    for p in data:
        nm = p.get("name") or {}
        term = (p.get("terms") or [{}])[-1]
        rec = {"first": nm.get("first"), "last": nm.get("last"), "official": nm.get("official_full"),
               "nickname": nm.get("nickname"), "party": term.get("party"), "state": term.get("state"),
               "chamber": term.get("type"), "district": term.get("district"),
               "bioguide": (p.get("id") or {}).get("bioguide")}
        by_last.setdefault((nm.get("last") or "").lower(), []).append(rec)
        for role in p.get("leadership_roles") or []:     # maintained by the @unitedstates project
            if str(role.get("end") or "9999") >= today and is_top_leadership(role.get("title") or ""):
                leaders[rec["bioguide"]] = role["title"]
    return {"by_last": by_last, "leaders": leaders}


def is_top_leadership(title: str) -> bool:
    """Party leadership as used in the research: Speaker, floor leaders, whips, caucus/conference chairs."""
    t = title.lower()
    if any(x in t for x in ("vice chair", "deputy", "steering", "outreach", "communications", "policy committee",
                            "pro tempore", "committee chair")):
        return False
    return any(x in t for x in ("speaker", "leader", "whip", "caucus chair", "conference chair"))


def match_member(directory: dict, last: str, first: str = "", chamber: str | None = None,
                 state: str | None = None) -> dict | None:
    cands = directory.get("by_last", {}).get((last or "").lower().strip(), [])
    if chamber:
        cands = [c for c in cands if (c.get("chamber") == ("rep" if chamber == "house" else "sen"))] or cands
    if state:
        cands = [c for c in cands if c.get("state") == state] or cands
    if len(cands) > 1 and first:
        f = first.lower().strip()[:3]
        cands = [c for c in cands if (c.get("first") or "").lower().startswith(f)
                 or (c.get("nickname") or "").lower().startswith(f)] or cands
    return cands[0] if len(cands) == 1 else None   # still ambiguous → don't guess who it is


def member_actor(chamber: str, first: str, last: str, state_dst: str | None, directory: dict) -> dict:
    state = (state_dst or "")[:2] or None
    m = match_member(directory, last, first, chamber, state)
    title = "Rep." if chamber == "house" else "Sen."
    name = f"{title} {clean(first)} {clean(last)}".strip()
    if m and m.get("official"):
        name = f"{title} {m['official']}"
    party = (m or {}).get("party")
    pchar = {"Democrat": "D", "Republican": "R", "Independent": "I"}.get(party or "", "")
    where = state_dst or (m or {}).get("state") or ""
    if chamber == "house" and state_dst and len(state_dst) > 2:
        where = f"{state_dst[:2]}-{state_dst[2:].lstrip('0') or 'AL'}"
    role = ("House" if chamber == "house" else "Senate") + (f" · {pchar}-{where}" if pchar else f" · {where}" if where else "")
    bio = (m or {}).get("bioguide")
    return {"name": name, "id": f"congress:{slug(clean(last) + '-' + clean(first))}", "role": role,
            "type": "congress", "party": party, "bioguide": bio,
            "leadership": (directory.get("leaders") or {}).get(bio) if bio else None}


# ---------------------------------------------------------------- House (official)

def house_index(http: HttpClient, year: int) -> list[dict]:
    url = f"{HOUSE}/public_disc/financial-pdfs/{year}FD.zip"
    raw = http.get(url, timeout=90).content
    z = zipfile.ZipFile(io.BytesIO(raw))
    name = next((n for n in z.namelist() if n.lower().endswith(".xml")), None)
    if not name:
        return []
    root = ET.fromstring(z.read(name))
    out = []
    for m in root.iter("Member"):
        rec = {c.tag: clean(c.text) for c in m}
        if rec.get("FilingType", "").upper() != "P":
            continue
        out.append({"doc_id": rec.get("DocID"), "first": rec.get("First", ""), "last": rec.get("Last", ""),
                    "prefix": rec.get("Prefix", ""), "state_dst": rec.get("StateDst", ""),
                    "year": rec.get("Year") or str(year), "filing_date": date_str(parse_date(rec.get("FilingDate")))})
    return out


def house_pdf_url(year: str | int, doc_id: str) -> str:
    return f"{HOUSE}/public_disc/ptr-pdfs/{year}/{doc_id}.pdf"


_TX_ANCHOR = re.compile(
    r"(?P<type>\bP\b|\bS\b(?:\s*\(partial\))?|\bE\b)\s+(?P<d1>\d{1,2}/\d{1,2}/\d{4})\s+(?P<d2>\d{1,2}/\d{1,2}/\d{4})\s+"
    r"(?P<amt>(?:Over\s+)?\$[\d,]+(?:\s*-\s*(?:\$[\d,]+)?)?|Spouse/DC Over \$1,000,000)", re.I)
_TICKER = re.compile(r"\(([A-Z][A-Z0-9.\-]{0,6})\)")
_ASSET_TYPE = re.compile(r"\[([A-Z]{2})\]")
_BOILER = re.compile(r"(Filing ID #\S+|Clerk of the House.*|Page \d+ of \d+|ID\s+Owner\s+Asset.*|Type\s+Date.*|"
                     r"\$200\?|Cap\.?|Gains\s*>|Transactions?$|Filer Information.*|Name:.*|Status:.*|"
                     r"State/District:.*|F\s*I\s*$|T\s*$)", re.I)
ASSET_TYPES = {"ST": "Stock", "OP": "Stock option", "EF": "ETF", "MF": "Mutual fund", "CS": "Corporate bond",
               "GS": "Government security", "OT": "Other", "CT": "Crypto", "PS": "Stock (private)",
               "RS": "Restricted stock", "AB": "Asset-backed security", "OI": "Options (other)",
               "SA": "Stock appreciation right", "PE": "Private equity", "HN": "Hedge fund"}


def parse_house_ptr_text(text: str) -> list[dict]:
    """Parse text extracted from an e-filed House PTR PDF into transactions.

    Layout per row: [owner] asset name … (TICKER) [TYPE]  P|S|S (partial)|E  trade-date  notify-date  $low - $high
    followed by 'Filing Status' / 'Description' / 'Subholding Of' lines. Asset names and amounts can wrap."""
    lines = [ln.rstrip() for ln in text.replace("\x00", "").splitlines()]
    body = "\n".join(lines)
    # drop the signature/certification block
    cut = re.search(r"\n\s*(?:I\s*P\s*O|C\s*S\s*\n|Certification and Signature|\* For the complete list)", body)
    if cut:
        body = body[:cut.start()]
    anchors = list(_TX_ANCHOR.finditer(body))
    out = []
    prev_end = 0
    for i, m in enumerate(anchors):
        nxt = anchors[i + 1].start() if i + 1 < len(anchors) else len(body)
        tail = body[m.end():nxt]
        # The asset's first line sits on the same text line as type/date/amount (cells are top-aligned);
        # anything on earlier lines belongs to the previous row's status/description block.
        line_start = body.rfind("\n", 0, m.start()) + 1
        asset_txt = body[line_start:m.start()].strip()
        if not asset_txt:
            prev_lines = [ln.strip() for ln in body[prev_end:line_start].splitlines() if ln.strip()]
            if prev_lines and not re.match(r"^(F\s*S|Filing Status|D\s*:|Description|S\s*O\s*:|Subholding)",
                                           prev_lines[-1], re.I) and not _BOILER.fullmatch(prev_lines[-1]):
                asset_txt = prev_lines[-1]
        prev_end = m.end()
        # the ticker / [type] can wrap onto the line after the amounts
        tail_lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
        amt = m.group("amt")
        extra_asset = []
        desc = None
        status = None
        for ln in tail_lines[:8]:
            if amt.rstrip().endswith("-"):
                um = re.search(r"(\$[\d,]+)\s*$", ln)
                if um:
                    amt = amt + " " + um.group(1)
                    rest = ln[:um.start()].strip()
                    if rest:
                        extra_asset.append(rest)
                    continue
            if re.match(r"^(F\s*S|Filing Status)\s*:?", ln, re.I):
                status = re.sub(r"^(F\s*S|Filing Status)\s*:?\s*", "", ln, flags=re.I)
                continue
            if re.match(r"^(D|Description)\s*:", ln, re.I):
                desc = re.sub(r"^(D|Description)\s*:\s*", "", ln, flags=re.I)
                continue
            if desc is not None and not re.match(r"^(S\s*O|Subholding|L\s*:|C\s*:)", ln, re.I) and len(desc) < 600 \
                    and not _TX_ANCHOR.search(ln) and status is not None:
                desc += " " + ln
                continue
            if status is None and desc is None and (ln.startswith("(") or ln.startswith("[") or _TICKER.search(ln)):
                extra_asset.append(ln)
        full_asset = clean(asset_txt + " " + " ".join(extra_asset))
        owner = ""
        om = re.match(r"^(SP|JT|DC)\s+", full_asset)
        if om:
            owner = om.group(1)
            full_asset = full_asset[om.end():]
        tick = _TICKER.findall(full_asset)
        atype = _ASSET_TYPE.search(full_asset)
        name = clean(_ASSET_TYPE.sub("", _TICKER.sub("", full_asset))).strip(" -–")
        lo, hi = parse_amount(amt)
        action, label = tx_action(m.group("type"))
        out.append({
            "owner": OWNER.get(owner, owner), "asset": name[:200], "ticker": tick[-1] if tick else None,
            "asset_type": ASSET_TYPES.get(atype.group(1), atype.group(1)) if atype else None,
            "action": action, "action_label": label,
            "trade_date": date_str(parse_date(m.group("d1"))), "notified": date_str(parse_date(m.group("d2"))),
            "amount_text": clean(amt), "lo": lo, "hi": hi, "status": status, "description": clean(desc) if desc else None,
        })
    return out


def pdf_text(raw: bytes) -> str:
    import pdfplumber
    parts = []
    with pdfplumber.open(io.BytesIO(raw)) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    return "\n".join(parts)


# ---------------------------------------------------------------- Senate (official eFD)

class EfdBlocked(RuntimeError):
    pass


def efd_recent_ptrs(http: HttpClient, days: int = 14, max_reports: int = 100) -> list[dict]:
    """Search eFD for recent periodic transaction reports. Raises EfdBlocked when the site refuses us."""
    try:
        home = http.get(f"{EFD}/search/home/", ok_statuses=(403,))
    except HttpError as e:
        raise EfdBlocked(str(e))
    if home.status == 403:
        raise EfdBlocked("HTTP 403 (bot protection)")
    m = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', home.text())
    if not m:
        raise EfdBlocked("no CSRF token on the agreement page")
    token = m.group(1)
    hdr = {"Referer": f"{EFD}/search/home/", "Content-Type": "application/x-www-form-urlencoded"}
    http.post(f"{EFD}/search/home/", data={"prohibition_agreement": "1", "csrfmiddlewaretoken": token}, headers=hdr)
    cookies = http._session.cookies  # noqa: SLF001 (same session keeps the agreement)
    csrf = cookies.get("csrftoken") or cookies.get("csrf") or token
    start = (now_utc() - dt.timedelta(days=days)).strftime("%m/%d/%Y 00:00:00")
    form = {"start": "0", "length": str(max_reports), "report_types": "[11]", "filer_types": "[]",
            "submitted_start_date": start, "submitted_end_date": "", "candidate_state": "", "senator_state": "",
            "office_id": "", "first_name": "", "last_name": "", "csrfmiddlewaretoken": csrf}
    r = http.post(f"{EFD}/search/report/data/", data=form,
                  headers={"Referer": f"{EFD}/search/", "X-CSRFToken": csrf,
                           "Content-Type": "application/x-www-form-urlencoded"}, ok_statuses=(403,))
    if r.status == 403:
        raise EfdBlocked("search API returned 403")
    try:
        payload = r.json()
    except ValueError:
        raise EfdBlocked("search returned a web page instead of data (bot check)") from None
    out = []
    for row in payload.get("data") or []:
        first, last, filer, link_html, filed = (row + [""] * 5)[:5]
        lm = re.search(r'href="([^"]+)"', link_html or "")
        if not lm:
            continue
        href = lm.group(1)
        out.append({"first": clean(first), "last": clean(last), "filer": clean(filer),
                    "url": EFD + href if href.startswith("/") else href,
                    "paper": "/paper/" in href, "report_id": href.rstrip("/").split("/")[-1],
                    "title": clean(re.sub(r"<[^>]+>", "", link_html)), "filed": date_str(parse_date(filed))})
    return out


def parse_efd_ptr_html(html: str) -> list[dict]:
    m = re.search(r"<title>\s*([^<]*)</title>", html, re.I)
    title = (m.group(1) if m else "").strip().lower()
    if not (title.startswith("efd") and "report" in title):   # bot check / agreement page, not a report
        raise EfdBlocked("report page didn't load (bot check or expired session)")
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S | re.I)
    out = []
    for r in rows:
        cells = [clean(unescape(re.sub(r"<[^>]+>", " ", c))) for c in re.findall(r"<td[^>]*>(.*?)</td>", r, re.S | re.I)]
        if len(cells) < 8 or not re.match(r"\d{1,2}/\d{1,2}/\d{4}", cells[1]):
            continue
        # #, Transaction Date, Owner, Ticker, Asset Name, Asset Type, Type, Amount, Comment
        lo, hi = parse_amount(cells[7])
        action, label = tx_action(cells[6])
        ticker = cells[3] if cells[3] not in ("--", "", "N/A") else None
        out.append({"trade_date": date_str(parse_date(cells[1])), "owner": cells[2] or "Self",
                    "ticker": ticker.upper() if ticker else None, "asset": cells[4], "asset_type": cells[5],
                    "action": action, "action_label": label, "amount_text": cells[7], "lo": lo, "hi": hi,
                    "description": cells[8] if len(cells) > 8 and cells[8] not in ("--", "") else None})
    return out


# ---------------------------------------------------------------- fallbacks (keyed)

def fmp_latest(http: HttpClient, key: str, chamber: str, pages: int = 3) -> list[dict]:
    path = "senate-latest" if chamber == "senate" else "house-latest"
    out = []
    for p in range(pages):
        rows = http.get_json(f"https://financialmodelingprep.com/stable/{path}?page={p}&limit=100&apikey={key}")
        if not isinstance(rows, list) or not rows:
            break
        for r in rows:
            lo, hi = parse_amount(r.get("amount"))
            action, label = tx_action(r.get("type"))
            sym = (r.get("symbol") or "").upper() or None
            out.append({"first": r.get("firstName") or "", "last": r.get("lastName") or "",
                        "office": r.get("office"), "state_dst": r.get("district") or r.get("state"),
                        "trade_date": date_str(parse_date(r.get("transactionDate"))),
                        "filed": date_str(parse_date(r.get("disclosureDate") or r.get("dateRecieved"))),
                        "owner": r.get("owner") or "Self", "ticker": sym if sym and sym not in ("--", "N/A") else None,
                        "asset": r.get("assetDescription") or "", "asset_type": r.get("assetType"),
                        "action": action, "action_label": label, "amount_text": r.get("amount"), "lo": lo, "hi": hi,
                        "description": r.get("comment"), "url": r.get("link")})
    return out


def quiver_latest(http: HttpClient, key: str) -> list[dict]:
    url = "https://api.quiverquant.com/beta/live/congresstrading"
    try:
        rows = http.get_json(url, headers={"Authorization": f"Token {key}", "Accept": "application/json"})
    except HttpError as e:
        if e.status in (401, 403):
            rows = http.get_json(url, headers={"Authorization": f"Bearer {key}", "Accept": "application/json"})
        else:
            raise
    out = []
    for r in rows or []:
        chamber = "senate" if str(r.get("House", "")).lower().startswith("sen") else "house"
        name = clean(r.get("Representative") or r.get("Name") or "")
        name = re.sub(r",?\s+(Jr\.?|Sr\.?|II|III|IV)$", "", name, flags=re.I)   # "John Smith Jr." → Smith
        first, _, last = name.rpartition(" ") if " " in name else ("", "", name)
        lo, hi = parse_amount(r.get("Range") or str(r.get("Amount") or ""))
        action, label = tx_action(r.get("Transaction"))
        out.append({"chamber": chamber, "first": first, "last": last, "state_dst": r.get("State"),
                    "trade_date": date_str(parse_date(r.get("TransactionDate"))),
                    "filed": date_str(parse_date(r.get("ReportDate") or r.get("Filed"))),
                    "owner": r.get("Owner") or "", "ticker": (r.get("Ticker") or "").upper() or None,
                    "asset": r.get("Description") or r.get("Company") or "", "asset_type": r.get("TickerType"),
                    "action": action, "action_label": label, "amount_text": r.get("Range"), "lo": lo, "hi": hi,
                    "description": r.get("Comments"), "party": r.get("Party"), "url": None})
    return out


# ---------------------------------------------------------------- events

def make_event(src: str, key: str, tx: dict, actor: dict, filed: str | None, url: str | None) -> dict:
    td, fd = parse_date(tx.get("trade_date")), parse_date(filed)
    lag = (fd - td).days if (td and fd) else None
    lo, hi = tx.get("lo"), tx.get("hi")
    return {
        "id": f"{src}:{key}",
        "kind": "congress",
        "ticker": tx.get("ticker"),
        "company": tx.get("asset") or tx.get("ticker"),
        "actor": actor,
        "action": tx.get("action"),
        "action_label": tx.get("action_label"),
        "trade_date": tx.get("trade_date"),
        "filed_at": filed,
        "lag_days": lag,
        "value_range": [lo, hi],
        "value": ((lo or 0) + (hi or lo or 0)) / 2 if lo else None,
        "details": {"owner": tx.get("owner"), "asset_type": tx.get("asset_type"), "amount_text": tx.get("amount_text"),
                    "description": tx.get("description"), "status": tx.get("status"),
                    "chamber": "House" if src.startswith("house") else "Senate" if src.startswith("senate") else None,
                    "late": lag is not None and lag > 45},
        "source": {"name": {"house": "House Clerk · PTR", "senate": "Senate eFD · PTR", "fmp": "FMP (from official PTR)",
                            "quiver": "Quiver (from official PTR)"}.get(src.split("-")[0], src), "url": url},
    }
