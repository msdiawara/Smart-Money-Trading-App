"""SEC Form 144: an insider's advance notice that they intend to sell restricted or control stock.

Required before selling more than 5,000 shares or $50,000 of such stock in any 3-month period, and filed on EDGAR
as structured XML since April 2023 (root <edgarSubmission> with <submissionType>144</submissionType>). It arrives
*before* the sale, often days before the Form 4 that reports it, so it's the earliest official sign that a sale
is coming. Most are routine (compensation shares, 10b5-1 plans), which the scorer says plainly.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

from .form4 import _strip_ns
from .util import clean, date_str, parse_date, title_name, to_float, xml_bytes

RELATIONSHIP_RANK = {"officer": 2, "director": 3, "10%": 2, "ten percent": 2, "stockholder": 2, "member of immediate family": 1}
_SENIOR = ("chief executive", "ceo", "president", "chief financial", "cfo", "chair")


def _t(node: ET.Element | None, path: str) -> str | None:
    if node is None:
        return None
    v = clean(node.findtext(path))
    return v or None


def parse_form144(xml_text: str) -> dict:
    root = _strip_ns(ET.fromstring(xml_bytes(xml_text)))
    if root.tag != "edgarSubmission":
        raise ValueError(f"not a Form 144 submission: {root.tag}")
    sub_type = _t(root, "headerData/submissionType") or ""
    if not sub_type.startswith("144"):
        raise ValueError(f"not a Form 144: {sub_type!r}")
    fd = root.find("formData")
    if fd is None:
        raise ValueError("Form 144 without formData")
    issuer = fd.find("issuerInfo")
    sec_info = fd.find("securitiesInformation")
    rels = [clean(r.text) for r in (issuer.findall("relationshipsToIssuer/relationshipToIssuer") if issuer is not None else [])
            if clean(r.text)]
    acquisitions = []
    for a in fd.findall("securitiesToBeSold"):
        acquisitions.append({"date": date_str(parse_date(_t(a, "acquiredDate"))), "nature": _t(a, "natureOfAcquisitionTransaction"),
                             "from": _t(a, "nameOfPersonfromWhomAcquired"), "shares": to_float(_t(a, "amountOfSecuritiesAcquired")),
                             "payment": _t(a, "natureOfPayment"), "gift": (_t(a, "isGiftTransaction") or "").upper() == "Y"})
    sold = []
    if (_t(fd, "nothingToReportFlagOnSecuritiesSoldInPast3Months") or "N").upper() != "Y":
        for s in fd.findall("securitiesSoldInPast3Months"):
            sold.append({"seller": title_name(_t(s, "sellerDetails/name") or ""), "date": date_str(parse_date(_t(s, "saleDate"))),
                         "shares": to_float(_t(s, "amountOfSecuritiesSold")), "proceeds": to_float(_t(s, "grossProceeds"))})
    plan_dates = [date_str(parse_date(clean(p.text))) for p in fd.findall("noticeSignature/planAdoptionDates/planAdoptionDate")
                  if clean(p.text)]
    shares = to_float(_t(sec_info, "noOfUnitsSold"))
    value = to_float(_t(sec_info, "aggregateMarketValue"))
    outstanding = to_float(_t(sec_info, "noOfUnitsOutstanding"))
    return {
        "doc_type": sub_type,
        "previous_accession": _t(root, "headerData/previousAccessionNumber"),      # what a 144/A amends
        "filer_cik": (_t(root, "headerData/filerInfo/filer/filerCredentials/cik") or "").zfill(10),
        "issuer_cik": (_t(issuer, "issuerCik") or "").zfill(10),
        "issuer_name": _t(issuer, "issuerName"),
        "seller": title_name(_t(issuer, "nameOfPersonForWhoseAccountTheSecuritiesAreToBeSold") or ""),
        "relationships": rels,
        "class_title": _t(sec_info, "securitiesClassTitle"),
        "broker": _t(sec_info, "brokerOrMarketmakerDetails/name"),
        "shares": shares, "value": value, "outstanding": outstanding,
        "pct_outstanding": (shares / outstanding * 100) if shares and outstanding else None,
        "approx_sale_date": date_str(parse_date(_t(sec_info, "approxSaleDate"))),
        "exchange": _t(sec_info, "securitiesExchangeName"),
        "acquisitions": acquisitions[:6],
        "sold_past_3m": sold[:12],
        "sold_past_3m_proceeds": sum(s["proceeds"] or 0 for s in sold) if sold else 0.0,
        "plan_adoption_dates": [d for d in plan_dates if d],
        "notice_date": date_str(parse_date(_t(fd, "noticeSignature/noticeDate"))),
        "remarks": _t(fd, "remarks"),
    }


def role_and_rank(relationships: list[str]) -> tuple[str, int]:
    """Readable role and a seniority rank on the Form 4 scale (5 = CEO … 1 = other). Form 144 only states the
    relationship category, so officers rank 2 unless the text names a senior title."""
    if not relationships:
        return "Insider", 1
    labels, rank = [], 1
    for r in relationships:
        low = r.lower()
        if any(s in low for s in _SENIOR):
            rank = max(rank, 4)
        base = next((v for k, v in RELATIONSHIP_RANK.items() if k in low), 1)
        rank = max(rank, base)
        labels.append(r.title() if r.isupper() else r)
    return ", ".join(dict.fromkeys(labels)), rank


def build_event(parsed: dict, accession: str, filed_at: str | None, filing_url: str, filed_date: str | None,
                ticker: str | None) -> dict:
    role, rank = role_and_rank(parsed["relationships"])
    amended = parsed["doc_type"].endswith("/A")
    shares, value = parsed["shares"], parsed["value"]
    return {
        "id": f"f144:{accession}",
        "kind": "insider",
        "ticker": ticker,
        "company": parsed["issuer_name"],
        "issuer_cik": parsed["issuer_cik"],
        "actor": {"name": parsed["seller"] or "Unknown", "id": f"cik:{parsed['filer_cik']}" if parsed["filer_cik"].strip("0") else "",
                  "role": role, "rank": rank, "type": "insider"},
        "action": "sale_notice",
        "action_label": "Plans to sell (Form 144 notice)" + (" (amended)" if amended else ""),
        "trade_date": parsed["approx_sale_date"] or filed_date,
        "filed_at": filed_at,
        "lag_days": None,
        "shares": shares,
        "price": round(value / shares, 4) if value and shares else None,
        "value": value,
        "holdings_after": None,
        "holdings_change_pct": None,
        "details": {
            "form": parsed["doc_type"], "notice": True, "amendment": amended, "codes": [],
            "amends": parsed.get("previous_accession"),
            "security": parsed["class_title"], "broker": parsed["broker"], "exchange": parsed["exchange"],
            "pct_outstanding": round(parsed["pct_outstanding"], 4) if parsed["pct_outstanding"] is not None else None,
            "approx_sale_date": parsed["approx_sale_date"], "notice_date": parsed["notice_date"],
            "plan_10b5_1": bool(parsed["plan_adoption_dates"]), "plan_flag_source": "plan adoption date" if parsed["plan_adoption_dates"] else None,
            "plan_adoption_dates": parsed["plan_adoption_dates"],
            "acquisitions": parsed["acquisitions"], "sold_past_3m": parsed["sold_past_3m"],
            "sold_past_3m_proceeds": parsed["sold_past_3m_proceeds"], "remarks": parsed["remarks"],
        },
        "source": {"name": f"SEC EDGAR · Form {parsed['doc_type']}", "url": filing_url, "accession": accession},
    }
