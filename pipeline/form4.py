"""Parse SEC Form 4 / 4/A ownership XML (root <ownershipDocument>) into trade events."""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any

from .util import business_days_between, clean, date_str, nice_title, parse_date, title_name, to_bool, to_float, xml_bytes

CODE_LABELS = {
    "P": "Open-market buy", "S": "Open-market sale", "A": "Grant/award", "D": "Returned to company",
    "F": "Shares withheld for tax", "I": "Plan transaction", "M": "Option exercise", "C": "Conversion",
    "E": "Short derivative expired", "H": "Derivative expired/cancelled", "O": "Out-of-money exercise",
    "X": "Option exercise", "G": "Gift", "L": "Small acquisition", "W": "Inheritance",
    "Z": "Voting-trust deposit/withdrawal", "J": "Other", "K": "Equity swap", "U": "Tender in takeover",
    "V": "Voluntary early report",
}
DISCRETIONARY = {"P", "S"}  # trades where the insider chose to buy/sell in the market


def _strip_ns(root: ET.Element) -> ET.Element:
    for el in root.iter():
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    return root


def _val(node: ET.Element | None, path: str) -> str | None:
    """Ownership XML wraps most values as <x><value>..</value><footnoteId/></x>."""
    if node is None:
        return None
    el = node.find(path)
    if el is None:
        return None
    v = el.find("value")
    txt = (v.text if v is not None else el.text)
    txt = clean(txt)
    return txt or None


def _footnote_ids(node: ET.Element | None) -> list[str]:
    if node is None:
        return []
    return [f.get("id") for f in node.iter("footnoteId") if f.get("id")]


def parse_form4(xml_text: str) -> dict:
    root = _strip_ns(ET.fromstring(xml_bytes(xml_text)))
    if root.tag != "ownershipDocument":
        raise ValueError(f"not an ownershipDocument: {root.tag}")
    issuer = root.find("issuer")
    footnotes = {f.get("id"): clean("".join(f.itertext())) for f in root.findall("footnotes/footnote")}
    owners = []
    for ro in root.findall("reportingOwner"):
        rel = ro.find("reportingOwnerRelationship")
        owners.append({
            "cik": clean(ro.findtext("reportingOwnerId/rptOwnerCik")).zfill(10),
            "name": clean(ro.findtext("reportingOwnerId/rptOwnerName")),
            "is_director": bool(to_bool(rel.findtext("isDirector")) if rel is not None else False),
            "is_officer": bool(to_bool(rel.findtext("isOfficer")) if rel is not None else False),
            "is_ten_pct": bool(to_bool(rel.findtext("isTenPercentOwner")) if rel is not None else False),
            "is_other": bool(to_bool(rel.findtext("isOther")) if rel is not None else False),
            "officer_title": clean(rel.findtext("officerTitle")) if rel is not None else "",
            "other_text": clean(rel.findtext("otherText")) if rel is not None else "",
        })

    def tx(node: ET.Element, derivative: bool) -> dict:
        amounts = node.find("transactionAmounts")
        post = node.find("postTransactionAmounts")
        fids = _footnote_ids(node)
        d = {
            "derivative": derivative,
            "security": _val(node, "securityTitle"),
            "date": date_str(parse_date(_val(node, "transactionDate"))),
            "code": clean(node.findtext("transactionCoding/transactionCode")).upper() or None,
            "form_type": clean(node.findtext("transactionCoding/transactionFormType")) or None,
            "equity_swap": to_bool(node.findtext("transactionCoding/equitySwapInvolved")),
            "timeliness": _val(node, "transactionTimeliness"),
            "shares": to_float(_val(amounts, "transactionShares")),
            "price": to_float(_val(amounts, "transactionPricePerShare")),
            "acq_disp": (_val(amounts, "transactionAcquiredDisposedCode") or "").upper() or None,
            "shares_after": to_float(_val(post, "sharesOwnedFollowingTransaction")),
            "value_after": to_float(_val(post, "valueOwnedFollowingTransaction")),
            "direct": (_val(node, "ownershipNature/directOrIndirectOwnership") or "D").upper(),
            "nature": _val(node, "ownershipNature/natureOfOwnership"),
            "footnote_ids": fids,
        }
        if derivative:
            d["exercise_price"] = to_float(_val(node, "conversionOrExercisePrice"))
            d["expiration"] = date_str(parse_date(_val(node, "expirationDate")))
            d["underlying"] = _val(node, "underlyingSecurity/underlyingSecurityTitle")
            d["underlying_shares"] = to_float(_val(node, "underlyingSecurity/underlyingSecurityShares"))
        return d

    txs = [tx(n, False) for n in root.findall("nonDerivativeTable/nonDerivativeTransaction")]
    txs += [tx(n, True) for n in root.findall("derivativeTable/derivativeTransaction")]

    aff = root.findtext("aff10b5One")
    plan_flag = to_bool(aff) if aff is not None else None
    all_notes = " ".join(footnotes.values()) + " " + clean(root.findtext("remarks"))
    plan_in_notes = bool(re.search(r"10b5-?1", all_notes, re.I))

    return {
        "schema": clean(root.findtext("schemaVersion")),
        "doc_type": clean(root.findtext("documentType")),
        "period": date_str(parse_date(root.findtext("periodOfReport"))),
        "original_date": date_str(parse_date(root.findtext("dateOfOriginalSubmission"))),
        "not_subject_16": to_bool(root.findtext("notSubjectToSection16")),
        "plan_10b5_1": plan_flag,
        "plan_in_footnotes": plan_in_notes,
        "issuer": {
            "cik": clean(issuer.findtext("issuerCik") if issuer is not None else "").zfill(10),
            "name": clean(issuer.findtext("issuerName") if issuer is not None else ""),
            "ticker": clean(issuer.findtext("issuerTradingSymbol") if issuer is not None else "").upper(),
        },
        "owners": owners,
        "transactions": txs,
        "footnotes": footnotes,
        "remarks": clean(root.findtext("remarks")),
        "signature_date": date_str(parse_date(root.findtext("ownerSignature/signatureDate"))),
    }


# ---------------------------------------------------------------- roles

_ROLE_RULES = [
    (re.compile(r"\b(chief executive|ceo)\b", re.I), "CEO", 5),
    (re.compile(r"\b(executive chair|chairman|chairwoman|chair of the board|chairperson)\b", re.I), "Chair", 4),
    (re.compile(r"\b(chief financial|cfo|principal financial)\b", re.I), "CFO", 4),
    (re.compile(r"\bpresident\b", re.I), "President", 4),
    (re.compile(r"\b(chief operating|coo)\b", re.I), "COO", 4),
    (re.compile(r"\b(chief [a-z]+ officer|c[a-z]o)\b", re.I), "C-suite", 3),
    (re.compile(r"\b(evp|executive vice president|svp|senior vice president|general counsel)\b", re.I), "EVP", 3),
]


def owner_role(o: dict) -> tuple[str, int]:
    """Readable role and a seniority rank (5 = CEO … 1 = other)."""
    labels: list[str] = []
    rank = 1
    title = o.get("officer_title") or ""
    if o.get("is_officer") or title:
        matched = False
        for rx, lab, r in _ROLE_RULES:
            if rx.search(title):
                labels.append(lab if lab not in ("C-suite", "EVP") else nice_title(title)[:40])
                rank = max(rank, r)
                matched = True
                break
        if not matched:
            labels.append(nice_title(title)[:40] if title else "Officer")
            rank = max(rank, 2)
    if o.get("is_director"):
        labels.append("Director")
        rank = max(rank, 3)
    if o.get("is_ten_pct"):
        labels.append("10% owner")
        rank = max(rank, 2)
    if o.get("is_other") and not labels:
        labels.append(o.get("other_text") or "Other")
    seen, uniq = set(), []
    for lab in labels:
        if lab.lower() not in seen:
            uniq.append(lab)
            seen.add(lab.lower())
    return ", ".join(uniq) or "Insider", rank


# ---------------------------------------------------------------- events

def _group_key(t: dict) -> str:
    code = t.get("code") or "?"
    if code in DISCRETIONARY:
        return code
    return "other"


def build_events(parsed: dict, accession: str, filed_at: str | None, filing_url: str,
                 filed_date: str | None) -> list[dict]:
    """One event per filing per action group (open-market buys, open-market sales, other)."""
    owners = parsed["owners"] or [{"cik": "", "name": "Unknown"}]
    primary = owners[0]
    role, rank = owner_role(primary)
    if len(owners) > 1:
        roles = [owner_role(o) for o in owners]
        rank = max(r for _, r in roles)
    actor_name = title_name(primary["name"])
    if len(owners) > 1:
        actor_name += f" + {len(owners) - 1} joint filer" + ("s" if len(owners) > 2 else "")

    issuer = parsed["issuer"]
    footnotes = parsed["footnotes"]
    plan = bool(parsed.get("plan_10b5_1")) or (parsed.get("plan_10b5_1") is None and parsed.get("plan_in_footnotes"))
    groups: dict[str, list[dict]] = {}
    for t in parsed["transactions"]:
        groups.setdefault(_group_key(t), []).append(t)

    events = []
    for gkey, rows in groups.items():
        common_rows = [r for r in rows if not r["derivative"]] or rows
        shares = sum((r["shares"] or 0) for r in common_rows)
        priced = [r for r in common_rows if r["shares"] and r["price"]]
        value = sum(r["shares"] * r["price"] for r in priced) if priced else None
        vwap = (value / sum(r["shares"] for r in priced)) if priced else None
        dates = sorted(d for d in (r["date"] for r in rows) if d)
        last = common_rows[-1]
        # Holdings are reported per ownership line (direct, each trust/family holding, each class), so the
        # before/after comparison is done line by line and then added up — never one line's total vs all rows.
        lines: dict[tuple, dict] = {}
        for r in common_rows:
            k = ((r.get("direct") or "D"), (r.get("nature") or "").strip().lower(), (r.get("security") or "").strip().lower())
            L = lines.setdefault(k, {"shares": 0.0, "after": None})
            L["shares"] += r["shares"] or 0
            if r.get("shares_after") is not None:
                L["after"] = r["shares_after"]
        known = [L for L in lines.values() if L["after"] is not None]
        after = sum(L["after"] for L in known) if known else None
        moved = sum(L["shares"] for L in known)
        if gkey == "P":
            action, label = "buy", "Open-market buy"
        elif gkey == "S":
            action, label = "sell", "Open-market sale"
        else:
            codes = sorted({r["code"] or "?" for r in rows})
            acq = [r for r in common_rows if r.get("acq_disp") == "A"]
            action = "other"
            label = " + ".join(CODE_LABELS.get(c, c) for c in codes)
            if len(acq) == len(common_rows):
                action = "acquire"
            elif not acq:
                action = "dispose"
        if gkey == "other":  # grants/exercises/tax/gifts: a $ total would be misleading
            value = vwap = None
        chg_pct = None
        if after is not None and moved:
            if action in ("buy", "acquire"):
                before = after - moved
                chg_pct = (moved / before * 100.0) if before > 0 else None
            elif action in ("sell", "dispose"):
                before = after + moved
                chg_pct = -(moved / before * 100.0) if before > 0 else None
        fids = []
        for r in rows:
            fids.extend(r.get("footnote_ids") or [])
        notes = [footnotes[f] for f in dict.fromkeys(fids) if f in footnotes][:3]
        trade_d = parse_date(dates[0]) if dates else None
        filed_d = parse_date(filed_date)
        lag = business_days_between(trade_d, filed_d) if (trade_d and filed_d) else None
        late = any((r.get("timeliness") or "").upper() == "L" for r in rows) or (lag is not None and lag > 2)
        indirect = [r for r in rows if r.get("direct") == "I"]
        exercise_linked = gkey == "S" and any(r.get("code") in ("M", "X", "C") for r in parsed["transactions"])
        ev = {
            "id": f"f4:{accession}:{gkey}",
            "kind": "insider",
            "ticker": issuer["ticker"] or None,
            "company": issuer["name"],
            "issuer_cik": issuer["cik"],
            "actor": {"name": actor_name, "id": f"cik:{primary.get('cik')}", "role": role, "rank": rank,
                      "type": "insider"},
            "action": action,
            "action_label": label,
            "trade_date": dates[0] if dates else parsed.get("period"),
            "trade_date_end": dates[-1] if len(dates) > 1 else None,
            "filed_at": filed_at or filed_date,
            "lag_days": lag,
            "shares": round(shares, 4) if shares else None,
            "price": round(vwap, 4) if vwap else None,
            "value": round(value, 2) if value else None,
            "holdings_after": after,
            "holdings_change_pct": round(chg_pct, 2) if chg_pct is not None else None,
            "details": {
                "form": parsed.get("doc_type") or "4",
                "codes": sorted({r["code"] for r in rows if r.get("code")}),
                "rows": len(rows),
                "security": last.get("security"),
                "plan_10b5_1": plan,
                "plan_flag_source": ("checkbox" if parsed.get("plan_10b5_1") is not None else
                                     "footnote" if parsed.get("plan_in_footnotes") else None),
                "indirect": bool(indirect),
                "ownership_nature": (indirect[0].get("nature") if indirect else None),
                "late": late,
                "exercise_linked": exercise_linked,
                "derivative_only": all(r["derivative"] for r in rows),
                "owners": [{"name": title_name(o["name"]), "cik": o["cik"], "role": owner_role(o)[0]}
                           for o in owners],
                "footnotes": [n[:400] for n in notes],
                "amendment": (parsed.get("doc_type") or "").endswith("/A"),
            },
            "source": {"name": "SEC EDGAR · Form " + (parsed.get("doc_type") or "4"), "url": filing_url,
                       "accession": accession},
        }
        events.append(ev)
    return events


def summarize_codes(parsed: dict) -> dict[str, Any]:
    out: dict[str, int] = {}
    for t in parsed["transactions"]:
        out[t.get("code") or "?"] = out.get(t.get("code") or "?", 0) + 1
    return out
