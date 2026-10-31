"""Form 13F-HR: parse information tables, detect value units, aggregate positions, diff quarters."""
from __future__ import annotations

import datetime as dt
import statistics
import xml.etree.ElementTree as ET

from .util import clean, date_str, parse_date, to_float, xml_bytes

DOLLARS_SINCE = dt.date(2023, 1, 3)  # SEC switched 13F values from $ thousands to whole dollars


def _strip(root: ET.Element) -> ET.Element:
    for el in root.iter():
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    return root


def _find_text(el: ET.Element, *names: str) -> str:
    for n in names:
        v = el.findtext(n)
        if v is not None and v.strip():
            return v.strip()
    return ""


def parse_cover(xml_text: str) -> dict:
    """13F primary_doc.xml (edgarSubmission): period, amendment info, manager name, totals."""
    root = _strip(ET.fromstring(xml_bytes(xml_text)))
    out = {"period": None, "is_amendment": False, "amendment_type": None, "manager": None,
           "entry_total": None, "value_total": None, "report_type": None}
    for el in root.iter():
        tag = el.tag
        txt = (el.text or "").strip()
        if not txt:
            continue
        if tag in ("periodOfReport", "reportCalendarOrQuarter") and not out["period"]:
            out["period"] = date_str(parse_date(txt))
        elif tag == "isAmendment":
            out["is_amendment"] = txt.lower() in ("true", "1", "y", "yes")
        elif tag == "amendmentType":
            out["amendment_type"] = txt.upper()
        elif tag == "name" and out["manager"] is None:
            out["manager"] = txt
        elif tag == "tableEntryTotal":
            out["entry_total"] = to_float(txt)
        elif tag == "tableValueTotal":
            out["value_total"] = to_float(txt)
        elif tag == "reportType":
            out["report_type"] = txt
    return out


def parse_info_table(xml_text: str) -> list[dict]:
    root = _strip(ET.fromstring(xml_bytes(xml_text)))
    rows = []
    for it in root.iter("infoTable"):
        rows.append({
            "issuer": clean(_find_text(it, "nameOfIssuer")),
            "class": clean(_find_text(it, "titleOfClass")),
            "cusip": clean(_find_text(it, "cusip")).upper(),
            "figi": clean(_find_text(it, "figi")) or None,
            "value": to_float(_find_text(it, "value")) or 0.0,
            "shares": to_float(_find_text(it, "shrsOrPrnAmt/sshPrnamt")) or 0.0,
            "share_type": clean(_find_text(it, "shrsOrPrnAmt/sshPrnamtType")).upper() or "SH",
            "put_call": clean(_find_text(it, "putCall")).upper() or None,
            "discretion": clean(_find_text(it, "investmentDiscretion")) or None,
        })
    return rows


def detect_multiplier(rows: list[dict], filed: dt.date | None) -> int:
    """1 if values are whole dollars, 1000 if in thousands.

    Filings since 3 Jan 2023 should be in dollars, but some filers kept reporting thousands; an implied
    share price far below $1 across most holdings gives that away."""
    prices = [r["value"] / r["shares"] for r in rows
              if r["shares"] > 0 and r["value"] > 0 and r["share_type"] == "SH" and not r["put_call"]]
    med = statistics.median(prices) if prices else None
    if med is not None:
        if med < 0.5:      # e.g. a $150 stock showing $0.15/share → values are in thousands
            return 1000
        if med >= 2.0:
            return 1
    # ambiguous or no priced rows: fall back to the rule in force on the filing date
    return 1000 if (filed is not None and filed < DOLLARS_SINCE) else 1


def aggregate(rows: list[dict], multiplier: int) -> dict[str, dict]:
    """Sum rows per CUSIP (+ put/call). Key 'CUSIP' for shares, 'CUSIP:PUT'/'CUSIP:CALL' for options."""
    pos: dict[str, dict] = {}
    for r in rows:
        if not r["cusip"]:
            continue
        key = r["cusip"] + (f":{r['put_call']}" if r["put_call"] else "")
        p = pos.setdefault(key, {"cusip": r["cusip"], "issuer": r["issuer"], "class": r["class"],
                                 "put_call": r["put_call"], "shares": 0.0, "value": 0.0,
                                 "share_type": r["share_type"], "figi": r["figi"]})
        p["shares"] += r["shares"]
        p["value"] += r["value"] * multiplier
    total_long = sum(p["value"] for p in pos.values() if not p["put_call"]) or 0.0
    for p in pos.values():
        p["weight"] = (p["value"] / total_long * 100.0) if (total_long and not p["put_call"]) else None
    return pos


def rank_positions(pos: dict[str, dict]) -> list[dict]:
    longs = [p for p in pos.values() if not p["put_call"]]
    longs.sort(key=lambda p: p["value"], reverse=True)
    for i, p in enumerate(longs, 1):
        p["rank"] = i
    return longs


def diff(cur: dict[str, dict], prev: dict[str, dict] | None) -> list[dict]:
    """Per-position changes between two quarters (long stock positions only)."""
    changes = []
    prev = prev or {}
    keys = {k for k, p in cur.items() if not p["put_call"]} | {k for k, p in prev.items() if not p["put_call"]}
    for k in keys:
        c, p = cur.get(k), prev.get(k)
        if c and not p:
            kind = "new"
            dsh = c["shares"]
            pct_chg = None
        elif p and not c:
            kind = "exit"
            dsh = -p["shares"]
            pct_chg = -100.0
        else:
            dsh = c["shares"] - p["shares"]
            if p["shares"] <= 0:
                pct_chg = None
            else:
                pct_chg = dsh / p["shares"] * 100.0
            if abs(dsh) < 1e-9 or (pct_chg is not None and abs(pct_chg) < 0.5):
                kind = "hold"
            else:
                kind = "add" if dsh > 0 else "trim"
        base = c or p
        changes.append({
            "key": k, "cusip": base["cusip"], "issuer": base["issuer"], "class": base["class"],
            "kind": kind, "shares": c["shares"] if c else 0.0, "prev_shares": p["shares"] if p else 0.0,
            "delta_shares": dsh, "pct_change": pct_chg,
            "value": c["value"] if c else 0.0, "prev_value": p["value"] if p else 0.0,
            "weight": c.get("weight") if c else 0.0, "prev_weight": p.get("weight") if p else 0.0,
            "rank": c.get("rank") if c else None,
        })
    return changes


def significant(changes: list[dict], limit: int = 15) -> list[dict]:
    """Pick the changes worth showing: new/exited meaningful stakes and big adds/trims."""
    keep = []
    for ch in changes:
        w, pw = ch["weight"] or 0.0, ch["prev_weight"] or 0.0
        if ch["kind"] == "new" and (w >= 0.5 or (ch["rank"] or 999) <= 20):
            keep.append(ch)
        elif ch["kind"] == "exit" and pw >= 0.5:
            keep.append(ch)
        elif ch["kind"] in ("add", "trim") and abs(ch["pct_change"] or 0) >= 20 and max(w, pw) >= 1.0:
            keep.append(ch)
        elif ch["kind"] in ("add", "trim") and abs(w - pw) >= 1.0:
            keep.append(ch)
    keep.sort(key=lambda c: abs((c["value"] or 0) - (c["prev_value"] or 0)), reverse=True)
    return keep[:limit]
