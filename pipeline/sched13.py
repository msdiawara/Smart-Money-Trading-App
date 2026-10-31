"""Schedules 13D / 13G (beneficial owners of more than 5%).

Since 18 Dec 2024 these are filed as structured XML (form types 'SCHEDULE 13D', 'SCHEDULE 13D/A',
'SCHEDULE 13G', 'SCHEDULE 13G/A'). The schema has had several minor versions, so fields are located by
element *local names* (several candidates each) instead of hard-coded paths."""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from .util import clean, date_str, parse_date, to_float, xml_bytes

FORMS_13D = {"SCHEDULE 13D", "SCHEDULE 13D/A", "SC 13D", "SC 13D/A"}
FORMS_13G = {"SCHEDULE 13G", "SCHEDULE 13G/A", "SC 13G", "SC 13G/A"}

_CANDS = {
    "issuer_name": ["issuername", "nameofissuer"],
    "issuer_cik": ["issuercik"],
    "cusip": ["issuercusip", "cusipnumber", "cusip", "issuercusips"],
    "class_title": ["securitiesclasstitle", "classofsecurities", "titleofclassofsecurities", "securityclasstitle"],
    "event_date": ["dateofevent", "eventdaterequiresfilingthisstatement", "eventdate"],
    "submission_type": ["submissiontype"],
    "purpose": ["transactionpurpose", "purposeoftransaction", "item4"],
    "funds_source": ["fundssource", "sourceoffunds"],
}
_PERSON_NAME = ["reportingpersonname", "nameofreportingperson", "nameofreportingpersons"]
_PERSON_AMOUNT = ["aggregateamountowned", "aggregateamountbeneficiallyowned", "amountbeneficiallyowned",
                  "reportingpersonbeneficiallyownedaggregatenumberofshares"]
_PERSON_PCT = ["percentofclass", "classpercent", "percentofclassrepresented", "percentofclassrepresentedbyamountinrow"]


def _local(tag: str) -> str:
    return tag.split("}", 1)[-1].lower() if isinstance(tag, str) else ""


def _first(root: ET.Element, names: list[str]) -> str | None:
    for el in root.iter():
        if _local(el.tag) in names:
            txt = clean("".join(el.itertext()))
            if txt:
                return txt
    return None


def parse_schedule13(xml_text: str) -> dict:
    root = ET.fromstring(xml_bytes(xml_text))
    out = {k: _first(root, v) for k, v in _CANDS.items()}
    # reporting persons: for every name element, read amount/percent from its own (innermost) group
    parent = {c: p for p in root.iter() for c in p}
    persons = []
    for el in root.iter():
        if _local(el.tag) not in _PERSON_NAME:
            continue
        grp = parent.get(el, root)
        sub = {}
        for c in grp.iter():
            ln = _local(c.tag)
            if ln not in sub:
                sub[ln] = c
        pct_el = next((sub[n] for n in _PERSON_PCT if n in sub), None)
        amt_el = next((sub[n] for n in _PERSON_AMOUNT if n in sub), None)
        nm = clean("".join(el.itertext()))
        if not nm or any(p["name"] == nm for p in persons):
            continue
        persons.append({
            "name": nm,
            "shares": to_float(clean("".join(amt_el.itertext()))) if amt_el is not None else None,
            "percent": to_float(re.sub(r"[%\s]", "", clean("".join(pct_el.itertext())))) if pct_el is not None else None,
        })
    # Keep the innermost matches (outer containers repeat the first person); de-dup done above.
    out["persons"] = persons[:12]
    out["event_date"] = date_str(parse_date(out.get("event_date")))
    pcts = [p["percent"] for p in persons if p.get("percent") is not None]
    shares = [p["shares"] for p in persons if p.get("shares") is not None]
    out["max_percent"] = max(pcts) if pcts else None
    out["max_shares"] = max(shares) if shares else None
    if out.get("purpose"):
        out["purpose"] = out["purpose"][:900]
    return out


def is_13d(form: str) -> bool:
    return form.upper() in FORMS_13D


def is_13g(form: str) -> bool:
    return form.upper() in FORMS_13G
