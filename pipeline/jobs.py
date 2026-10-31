"""The individual refresh jobs. Each returns (items_processed, note) and raises on hard failure;
main.py records the outcome per job for the Data Health screen."""
from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

from . import congress, f13, form4, form144, sched13, sec
from .etoro import Etoro, diff_positions, leader_card, pick_leaders
from .figi import FigiMapper
from .http import HttpClient, HttpError
from .settings import Settings
from .state import is_seen, mark_seen
from .util import (date_str, iso, log, now_et, now_utc, parse_date, parse_iso, read_json, title_name,
                   to_float, write_json)

F4_FORMS = {"4", "4/A"}
F144_FORMS = {"144", "144/A"}


def note_failure(state: dict, ns: str, key: str, limit: int = 2) -> bool:
    """Count a failure for one item; True once it has failed `limit` times (stop retrying it). Broken
    documents get 2 tries; network/server errors get more, since those are usually temporary."""
    fails = state.setdefault("fails", {}).setdefault(ns, {})
    fails[key] = int(fails.get(key, 0)) + 1
    if fails[key] >= limit:
        fails.pop(key, None)
        return True
    return False


def failed_twice(state: dict, ns: str, key: str) -> bool:
    return note_failure(state, ns, key, 2)


def _limit(err: Exception) -> int:
    return 6 if isinstance(err, (HttpError, OSError)) else 2


class Budget:
    def __init__(self, seconds: float):
        self.t0 = time.monotonic()
        self.seconds = seconds

    def left(self) -> float:
        return self.seconds - (time.monotonic() - self.t0)

    def ok(self, need: float = 20) -> bool:
        return self.left() > need


# ---------------------------------------------------------------- shared helpers

def add_events(state: dict, events: list[dict], new_ids: list[str]) -> int:
    idx = {e["id"]: i for i, e in enumerate(state["events"])}
    n = 0
    for e in events:
        e.setdefault("first_seen", iso(now_utc()))
        if e["id"] in idx:
            old = state["events"][idx[e["id"]]]
            e["first_seen"] = old.get("first_seen", e["first_seen"])
            state["events"][idx[e["id"]]] = e
        else:
            state["events"].append(e)
            new_ids.append(e["id"])
            n += 1
    return n


def _filed_at(sub: sec.Submission | None, entry_updated: str | None, fallback_date: str | None) -> str | None:
    if sub:
        acc = sub.accepted_at()
        if acc:
            return acc.isoformat()
    if entry_updated:
        return entry_updated
    return fallback_date


def process_form4(http: HttpClient, settings: Settings, state: dict, cik: str, accession: str,
                  entry_updated: str | None, new_ids: list[str]) -> int:
    sub = sec.fetch_submission(http, cik, accession)
    xml = sub.first_xml(root_contains="ownershipDocument")
    if not xml:
        mark_seen(state, "f4", accession)
        return 0
    parsed = form4.parse_form4(xml)
    filed_date = sub.header.get("FILED AS OF DATE")
    filed_date = date_str(parse_date(filed_date)) if filed_date else (entry_updated or "")[:10] or None
    url = sec.filing_index_url(parsed["issuer"]["cik"] or cik, accession)
    evs = form4.build_events(parsed, accession, _filed_at(sub, entry_updated, filed_date), url, filed_date)
    min_v = settings.c("insiders", "min_value_usd", default=0) or 0
    keep_other = settings.c("insiders", "show_non_discretionary", default=True)
    evs = [e for e in evs if (e["action"] in ("buy", "sell") and (e.get("value") or 0) >= min_v)
           or (keep_other and e["action"] not in ("buy", "sell"))]
    mark_seen(state, "f4", accession)
    if (parsed.get("doc_type") or "").endswith("/A"):
        evs = [e for e in evs if not replace_amended(state, e)]
    return add_events(state, evs, new_ids)


def replace_amended(state: dict, ev: dict) -> bool:
    """A Form 4/A restates an earlier Form 4. If we already show the original, swap it for the corrected
    version in place (no duplicate card, no second alert). Returns True when it replaced something."""
    for i, old in enumerate(state["events"]):
        if (old.get("kind") == "insider" and old["id"] != ev["id"]
                and old.get("issuer_cik") == ev.get("issuer_cik")
                and old["actor"].get("id") == ev["actor"].get("id")
                and old.get("action") == ev.get("action")
                and old.get("trade_date") == ev.get("trade_date")
                and not (old.get("details") or {}).get("amendment")):
            ev["first_seen"] = old.get("first_seen", ev.get("first_seen"))
            ev["ledger_id"] = old.get("ledger_id") or old["id"]       # its forecast stays the original one
            ev["details"]["replaces"] = (old.get("source") or {}).get("accession")
            for k in ("quote_at_filing",):
                if old.get(k) is not None:
                    ev[k] = old[k]
            state["events"][i] = ev
            return True
    return False


# ---------------------------------------------------------------- 1. insiders (live feed)

def job_insiders_live(http: HttpClient, settings: Settings, state: dict, budget: Budget, new_ids: list) -> tuple[int, str]:
    seen = set(state["seen"].get("f4", {}))
    entries = sec.current_feed(http, "4", pages=15, stop_at=seen)
    todo, picked = [], set()
    for e in entries:
        if e.form in F4_FORMS and e.accession not in seen and e.accession not in picked:
            picked.add(e.accession)
            todo.append(e)
    todo.reverse()  # oldest first so the feed fills in order
    n = done = 0
    for e in todo:
        if not budget.ok(25):
            break
        try:
            n += process_form4(http, settings, state, e.cik, e.accession, e.updated, new_ids)
            done += 1
        except Exception as err:  # noqa: BLE001 – one bad filing must never stall the feed
            log.warning("form4 %s failed: %s", e.accession, err)
            if note_failure(state, "f4", e.accession, _limit(err)):
                mark_seen(state, "f4", e.accession)
    return n, f"{done}/{len(todo)} new filings from the live feed"


# ---------------------------------------------------------------- 1b. planned insider sales (Form 144)

def process_form144(http: HttpClient, state: dict, cik: str, accession: str, entry_updated: str | None,
                    tick: dict, new_ids: list[str]) -> int:
    sub = sec.fetch_submission(http, cik, accession)
    xml = sub.first_xml(root_contains="edgarSubmission")
    if not xml:
        mark_seen(state, "f144", accession)
        return 0
    parsed = form144.parse_form144(xml)
    filed_date = sub.header.get("FILED AS OF DATE")
    filed_date = date_str(parse_date(filed_date)) if filed_date else (entry_updated or "")[:10] or None
    ticker = (tick.get("by_cik", {}).get(parsed["issuer_cik"]) or {}).get("ticker")
    url = sec.filing_index_url(parsed["filer_cik"] if parsed["filer_cik"].strip("0") else cik, accession)
    ev = form144.build_event(parsed, accession, _filed_at(sub, entry_updated, filed_date), url, filed_date, ticker)
    mark_seen(state, "f144", accession)
    if ev["details"]["amendment"]:
        # an amended notice replaces the card it amends: by the accession it names, else the same person's latest
        # un-amended notice for the same issuer
        prev_acc = parsed.get("previous_accession")
        for i, old in enumerate(state["events"]):
            if not old["id"].startswith("f144:"):
                continue
            # the named original, or an earlier amendment of it (a chain of 144/As names the original each time)
            hit = (old["id"] == f"f144:{prev_acc}" or old["details"].get("amends") == prev_acc) if prev_acc else (
                old.get("issuer_cik") == ev["issuer_cik"] and old["actor"].get("id") == ev["actor"].get("id")
                and not old["details"].get("amendment"))
            if hit:
                ev["first_seen"] = old.get("first_seen", ev.get("first_seen"))
                ev["ledger_id"] = old.get("ledger_id") or old["id"]
                if old.get("quote_at_filing") is not None:
                    ev["quote_at_filing"] = old["quote_at_filing"]
                state["events"][i] = ev
                return 0
    return add_events(state, [ev], new_ids)


def job_form144_live(http: HttpClient, settings: Settings, state: dict, budget: Budget, new_ids: list,
                     tick: dict) -> tuple[int, str]:
    seen = set(state["seen"].get("f144", {}))
    entries = sec.current_feed(http, "144", pages=6, stop_at=seen)
    todo, picked = [], set()
    for e in entries:
        if e.form in F144_FORMS and e.accession not in seen and e.accession not in picked:
            picked.add(e.accession)
            todo.append(e)
    todo.reverse()
    n = done = 0
    for e in todo:
        if not budget.ok(25):
            break
        try:
            n += process_form144(http, state, e.cik, e.accession, e.updated, tick, new_ids)
            done += 1
        except Exception as err:  # noqa: BLE001 – one bad filing must never stall the feed
            log.warning("form144 %s failed: %s", e.accession, err)
            if note_failure(state, "f144", e.accession, _limit(err)):
                mark_seen(state, "f144", e.accession)
    return n, f"{done}/{len(todo)} new sale notices from the live feed"


# ---------------------------------------------------------------- 2. activists / >5% stakes

def process_13dg(http: HttpClient, state: dict, cik: str, accession: str, form: str, entry_updated: str | None,
                 names: dict, tick: dict, fund_ciks: dict, new_ids: list) -> int:
    sub = sec.fetch_submission(http, cik, accession)
    xml = sub.first_xml(root_contains="edgarSubmission") or sub.first_xml()
    info = {}
    if xml:
        try:
            info = sched13.parse_schedule13(xml)
        except Exception as err:  # noqa: BLE001 – schema drift shouldn't kill the run
            log.warning("13D/G parse %s: %s", accession, err)
    subject = names.get("Subject") or names.get("Subject Company") or {}
    filer = names.get("Filed by") or names.get("Reporting") or names.get("Filer") or {}
    issuer_cik = (info.get("issuer_cik") or subject.get("cik") or "").zfill(10)
    if not filer.get("cik") and names.get("_rows"):          # nightly index: pick the CIK that isn't the issuer
        other = [r for r in names["_rows"] if r["cik"] != issuer_cik]
        if other:
            filer = other[0]
    company = info.get("issuer_name") or subject.get("name") or ""
    ticker = (tick.get("by_cik", {}).get(issuer_cik) or {}).get("ticker")
    persons = info.get("persons") or []
    actor_name = title_name(filer.get("name") or (persons[0]["name"] if persons else "Unknown filer"))
    filed_date = date_str(parse_date(sub.header.get("FILED AS OF DATE"))) or (entry_updated or "")[:10]
    is13d = sched13.is_13d(form)
    filer_cik = (filer.get("cik") or "").zfill(10)
    ev = {
        "id": f"s13:{accession}",
        "kind": "activist",
        "ticker": ticker,
        "company": company,
        "issuer_cik": issuer_cik,
        "actor": {"name": actor_name, "id": f"cik:{filer_cik}", "role": "Activist (13D)" if is13d else "5%+ holder (13G)",
                  "type": "fund" if filer_cik in fund_ciks else "holder"},
        "action": "stake",
        "action_label": ("Activist stake" if is13d else "Passive 5%+ stake") + (" (amended)" if form.endswith("/A") else ""),
        "trade_date": info.get("event_date") or filed_date,
        "filed_at": _filed_at(sub, entry_updated, filed_date),
        "lag_days": None,
        "shares": info.get("max_shares"),
        "price": None,
        "value": None,
        "details": {"schedule": "13D" if is13d else "13G", "form": form, "amendment": form.endswith("/A"),
                    "percent": info.get("max_percent"), "class_title": info.get("class_title"),
                    "cusip": info.get("cusip"), "persons": persons[:6], "purpose": info.get("purpose"),
                    "funds_source": info.get("funds_source"), "filer_cik": filer_cik,
                    "parsed": bool(xml)},
        "source": {"name": f"SEC EDGAR · {form}",
                   "url": sec.filing_index_url(filer_cik if filer_cik.strip("0") else cik, accession),
                   "accession": accession},
    }
    mark_seen(state, "s13", accession)
    return add_events(state, [ev], new_ids)


def job_activists_live(http: HttpClient, settings: Settings, state: dict, budget: Budget, new_ids: list,
                       tick: dict) -> tuple[int, str]:
    seen = set(state["seen"].get("s13", {}))
    entries = sec.current_feed(http, "SCHEDULE 13", pages=2, stop_at=seen)
    entries += sec.current_feed(http, "SC 13", pages=1, stop_at=seen)
    forms = sched13.FORMS_13D | sched13.FORMS_13G
    by_acc: dict[str, dict] = {}
    for e in entries:
        if e.form.upper() not in forms or e.accession in seen:
            continue
        d = by_acc.setdefault(e.accession, {"form": e.form.upper(), "updated": e.updated, "cik": e.cik, "names": {}})
        d["names"][e.role] = {"name": e.name, "cik": e.cik}
    fund_ciks = {f["cik"]: f for f in settings.c("funds", "list", default=[])}
    n = 0
    for acc, d in list(by_acc.items())[::-1]:
        if not budget.ok(25):
            break
        try:
            n += process_13dg(http, state, d["cik"], acc, d["form"], d["updated"], d["names"], tick, fund_ciks, new_ids)
        except Exception as err:  # noqa: BLE001
            log.warning("13D/G %s failed: %s", acc, err)
            if note_failure(state, "s13", acc, _limit(err)):
                mark_seen(state, "s13", acc)
    return n, f"{len(by_acc)} new 13D/13G filings"


# ---------------------------------------------------------------- 3. daily backstop

def _prev_weekday(d: dt.date) -> dt.date:
    d -= dt.timedelta(days=1)
    while d.weekday() >= 5:
        d -= dt.timedelta(days=1)
    return d


def backstop_next_day(state: dict) -> dt.date | None:
    """The next business day whose SEC daily index hasn't been checked yet (None = up to date). Catches up
    on missed days (up to a week back) one day at a time, so a skipped or failed day is never lost."""
    if state.get("backstop_pending"):
        return parse_date(state["backstop_pending"])
    yday = _prev_weekday(now_et().date())
    done = parse_date(state.get("backstop_done"))
    if not done:
        return yday
    if done >= yday:
        return None
    d = max(done, yday - dt.timedelta(days=8)) + dt.timedelta(days=1)
    while d.weekday() >= 5:
        d += dt.timedelta(days=1)
    return d if d <= yday else None


def job_backstop(http: HttpClient, settings: Settings, state: dict, budget: Budget, new_ids: list,
                 tick: dict) -> tuple[int, str]:
    day = backstop_next_day(state)
    if day is None:
        return 0, "up to date"
    rows = sec.daily_master_index(http, day)
    if rows is None:
        yday = _prev_weekday(now_et().date())
        if day < yday or now_et().hour >= 12:      # still nothing hours later → a market holiday
            state["backstop_done"] = day.isoformat()
            state["backstop_pending"] = None
            return 0, f"no SEC index for {day} (market holiday)"
        return 0, f"SEC hasn't posted the {day} index yet; will check again within the hour"
    seen4 = set(state["seen"].get("f4", {}))
    seen13 = set(state["seen"].get("s13", {}))
    missing4 = [r for r in rows if r["form"] in F4_FORMS and r["accession"] and r["accession"] not in seen4]
    # de-dup (each Form 4 is listed under issuer and owner)
    uniq, seen_acc = [], set()
    for r in missing4:
        if r["accession"] not in seen_acc:
            seen_acc.add(r["accession"])
            uniq.append(r)
    n = done = 0
    for r in uniq[:900]:
        if not budget.ok(40):
            break
        try:
            n += process_form4(http, settings, state, r["cik"], r["accession"], None, new_ids)
            done += 1
        except Exception as err:  # noqa: BLE001
            log.warning("backstop form4 %s: %s", r["accession"], err)
            if note_failure(state, "f4", r["accession"], _limit(err)):
                mark_seen(state, "f4", r["accession"])
    seen144 = set(state["seen"].get("f144", {}))
    m144, done144 = {}, 0
    for r in rows:
        if r["form"] in F144_FORMS and r["accession"] and r["accession"] not in seen144:
            m144.setdefault(r["accession"], r["cik"])
    for acc, c in m144.items():
        if not budget.ok(40):
            break
        done144 += 1
        try:
            n += process_form144(http, state, c, acc, None, tick, new_ids)
        except Exception as err:  # noqa: BLE001
            log.warning("backstop form144 %s: %s", acc, err)
            if note_failure(state, "f144", acc, _limit(err)):
                mark_seen(state, "f144", acc)
    forms13 = sched13.FORMS_13D | sched13.FORMS_13G
    m13 = {}
    for r in rows:
        if r["form"].upper() in forms13 and r["accession"] and r["accession"] not in seen13:
            d = m13.setdefault(r["accession"], {"form": r["form"].upper(), "cik": r["cik"], "names": {"_rows": []}})
            d["names"]["_rows"].append({"cik": r["cik"], "name": r["company"]})
    fund_ciks = {f["cik"]: f for f in settings.c("funds", "list", default=[])}
    done13 = 0
    for acc, d in m13.items():
        if not budget.ok(40):
            break
        done13 += 1
        try:
            n += process_13dg(http, state, d["cik"], acc, d["form"], None, d["names"], tick, fund_ciks, new_ids)
        except Exception as err:  # noqa: BLE001
            log.warning("backstop 13D/G %s: %s", acc, err)
            if note_failure(state, "s13", acc, _limit(err)):
                mark_seen(state, "s13", acc)
    complete = done >= len(uniq) and done13 >= len(m13) and done144 >= len(m144)
    state["backstop_pending"] = None if complete else day.isoformat()
    if complete:
        state["backstop_done"] = day.isoformat()
    return n, f"{day}: {len(uniq)} Form 4s missed by the live feed ({done} processed), {len(m13)} 13D/G, {len(m144)} Form 144" + \
        ("" if complete else " — continuing next run")


# ---------------------------------------------------------------- 4. funds (13F)

def _norm(s: str) -> str:
    import re as _re
    s = _re.sub(r"[^A-Z0-9 ]+", " ", s.upper())
    s = _re.sub(r"\b(LLC|L ?P|INC|CORP|CO|LTD|MA|DE|NY|THE)\b", " ", s)
    return " ".join(s.split())


def _f13_cache(settings: Settings, accession: str) -> Path:
    return settings.heavy_dir / "f13" / f"{accession}.json.gz"


def load_13f(http: HttpClient, settings: Settings, cik: str, row: dict) -> dict | None:
    acc = row["accessionNumber"]
    path = _f13_cache(settings, acc)
    cached = read_json(path)
    if cached:
        return cached
    sub = sec.fetch_submission(http, cik, acc)
    cover_xml = sub.first_xml(root_contains="edgarSubmission")
    table_xml = sub.first_xml(root_contains="informationTable")
    if not table_xml:
        return None
    cover = f13.parse_cover(cover_xml) if cover_xml else {}
    rows = f13.parse_info_table(table_xml)
    filed = parse_date(row.get("filingDate"))
    mult = f13.detect_multiplier(rows, filed)
    pos = f13.aggregate(rows, mult)
    ranked = f13.rank_positions(pos)
    keep = {p["cusip"]: p for p in ranked[:400]}
    for k, p in pos.items():  # keep options too, for context
        if p["put_call"] and len(keep) < 600:
            keep[k] = p
    data = {"accession": acc, "form": row.get("form"), "filed": row.get("filingDate"),
            "period": cover.get("period") or row.get("reportDate"), "cover": cover, "multiplier": mult,
            "total_long_value": sum(p["value"] for p in pos.values() if not p["put_call"]),
            "n_positions": len([p for p in pos.values() if not p["put_call"]]), "positions": keep}
    write_json(path, data)
    settings.mark_heavy_dirty()
    return data


def job_funds(http: HttpClient, settings: Settings, state: dict, budget: Budget, new_ids: list,
              figi: FigiMapper) -> tuple[int, str]:
    funds = settings.c("funds", "list", default=[])
    fstate = state.setdefault("funds", {})
    n = 0
    notes = []
    for f in funds:
        if not budget.ok(60):
            notes.append("time budget reached")
            break
        cik = f["cik"]
        st = fstate.setdefault(cik, {})
        try:
            sub = sec.submissions(http, cik)
        except Exception as e:  # noqa: BLE001 – one fund failing never blocks the others
            st["error"] = str(e)[:200]
            continue
        names = [sub.get("name") or ""] + [x.get("name", "") for x in (sub.get("formerNames") or [])]
        expect = _norm(f.get("expect") or "")
        st["sec_name"] = sub.get("name")
        st["name_ok"] = (not expect) or any(expect in _norm(n) for n in names)
        if not st["name_ok"]:
            st["error"] = f"CIK {cik} belongs to '{sub.get('name')}', expected '{expect}' — skipped"
            continue
        filings = sec.recent_filings(sub, {"13F-HR", "13F-HR/A"})
        notices = sec.recent_filings(sub, {"13F-NT", "13F-NT/A"})
        st["last_13f"] = filings[0]["filingDate"] if filings else None
        if notices and (not filings or notices[0]["filingDate"] > filings[0]["filingDate"]):
            st["notice"] = (f"Latest is a 13F-NT ({notices[0]['filingDate']}): holdings are reported by another "
                            "manager — update this fund's CIK in config.yaml")
        else:
            st["notice"] = None
        if not filings:
            st["error"] = st.get("notice") or "no 13F-HR filings found"
            continue
        latest = filings[0]
        originals = [r for r in filings if r["form"] == "13F-HR"]
        if latest["accessionNumber"] == st.get("processed_accession"):
            st["error"] = None
            continue
        try:
            cur = load_13f(http, settings, cik, originals[0] if originals else latest)
            prev_row = next((r for r in originals[1:] if (r.get("reportDate") or "") < ((cur or {}).get("period") or "9999")), None)
            prev = load_13f(http, settings, cik, prev_row) if (cur and prev_row) else None
        except Exception as err:  # noqa: BLE001
            st["error"] = f"could not read 13F: {err}"[:200]
            continue
        if not cur:
            st["error"] = "could not read information table"
            continue
        if prev is None:  # first filing under this CIK: nothing to compare → record a baseline, no "new" flood
            st.update(processed_accession=latest["accessionNumber"], period=cur.get("period"), filed=cur.get("filed"),
                      n_positions=cur.get("n_positions"), total_value=cur.get("total_long_value"),
                      error=None, info="First filing seen under this CIK: changes appear from its next quarterly 13F")
            continue
        figi.map([p["cusip"] for p in list(cur["positions"].values())[:150]] +
                 ([p["cusip"] for p in list(prev["positions"].values())[:150]] if prev else []), max_requests=30)
        changes = f13.diff(cur["positions"], prev["positions"] if prev else None)
        sig = f13.significant(changes)
        period_end = parse_date(cur.get("period"))
        filed = cur.get("filed")
        age = (parse_date(filed) - period_end).days if (period_end and parse_date(filed)) else None
        evs = []
        for ch in sig:
            m = figi.lookup(ch["cusip"]) or {}
            evs.append({
                "id": f"f13:{cur['accession']}:{ch['cusip']}",
                "kind": "fund",
                "ticker": m.get("ticker"),
                "company": title_name(ch["issuer"]),
                "actor": {"name": f["name"], "id": f"cik:{cik}", "role": f.get("manager", ""), "type": "fund"},
                "action": ch["kind"],
                "action_label": {"new": "New position", "add": "Added", "trim": "Trimmed", "exit": "Sold out"}[ch["kind"]],
                "trade_date": cur.get("period"),
                "filed_at": filed,
                "lag_days": age,
                "shares": ch["shares"] or ch["prev_shares"],
                "price": (ch["value"] / ch["shares"]) if ch["shares"] else None,
                "value": ch["value"] or ch["prev_value"],
                "holdings_change_pct": ch["pct_change"],
                "details": {"change": ch["kind"], "fund_cik": cik, "period": cur.get("period"),
                            "weight": round(ch["weight"] or 0, 3), "prev_weight": round(ch["prev_weight"] or 0, 3),
                            "rank": ch["rank"], "pct_change": ch["pct_change"], "delta_shares": ch["delta_shares"],
                            "prev_shares": ch["prev_shares"], "cusip": ch["cusip"], "class": ch["class"],
                            "fund_positions": cur.get("n_positions"), "fund_value": cur.get("total_long_value"),
                            "style": f.get("style"), "days_since_quarter_end": (now_utc().date() - period_end).days if period_end else None,
                            "amended_later": latest["form"].endswith("/A"), "units_multiplier": cur.get("multiplier")},
                "source": {"name": "SEC EDGAR · Form 13F-HR", "url": sec.filing_index_url(cik, cur["accession"]),
                           "accession": cur["accession"]},
            })
        n += add_events(state, evs, new_ids)
        st.update(processed_accession=latest["accessionNumber"], period=cur.get("period"), filed=filed,
                  n_positions=cur.get("n_positions"), total_value=cur.get("total_long_value"), error=None, info=None,
                  top=[{"cusip": p["cusip"], "issuer": p["issuer"], "weight": p.get("weight"), "value": p["value"]}
                       for p in list(cur["positions"].values())[:15] if not p.get("put_call")])
    # positions whose ticker couldn't be looked up earlier (OpenFIGI down or rate-limited) get another try
    missing = [e for e in state["events"] if e.get("kind") == "fund" and not e.get("ticker")
               and (e.get("details") or {}).get("cusip")]
    if missing and budget.ok(30):
        figi.map([e["details"]["cusip"] for e in missing], max_requests=10)
        for e in missing:
            m = figi.lookup(e["details"]["cusip"]) or {}
            if m.get("ticker"):
                e["ticker"] = m["ticker"]
    figi.save()
    bad = [c for c, s in fstate.items() if s.get("error")]
    return n, (f"{len(funds)} funds checked, {len(bad)} with issues" + (f"; {'; '.join(notes)}" if notes else ""))


def in_13f_deadline_window(today: dt.date | None = None) -> bool:
    d = today or now_et().date()
    return (d.month in (2, 5, 8, 11) and 1 <= d.day <= 18)


# ---------------------------------------------------------------- 5. congress — House (official)

def job_house(http: HttpClient, settings: Settings, state: dict, budget: Budget, new_ids: list,
              directory: dict) -> tuple[int, str]:
    year = now_et().year
    years = [year] + ([year - 1] if now_et().month <= 2 else [])
    cutoff = now_utc().date() - dt.timedelta(days=60)
    todo = []
    for y in years:
        for r in congress.house_index(http, y):
            if not r["doc_id"] or is_seen(state, "house", r["doc_id"]):
                continue
            fd = parse_date(r["filing_date"])
            if fd and fd < cutoff:
                state.setdefault("house_backfill", {})[r["doc_id"]] = r  # for track records later
                mark_seen(state, "house", r["doc_id"])
                continue
            todo.append(r)
    todo.sort(key=lambda r: r["filing_date"] or "")
    n = parsed = scanned = 0
    for r in todo[:60]:
        if not budget.ok(30):
            break
        url = congress.house_pdf_url(r["year"], r["doc_id"])
        try:
            raw = http.get(url, timeout=60).content
            text = congress.pdf_text(raw)
            txs = congress.parse_house_ptr_text(text)
        except HttpError as e:
            if e.status == 404 or note_failure(state, "house", r["doc_id"], 6):
                mark_seen(state, "house", r["doc_id"])
            continue
        except Exception as err:  # noqa: BLE001 – a malformed PDF shouldn't stop the run
            log.warning("house PDF %s: %s", r["doc_id"], err)
            if not note_failure(state, "house", r["doc_id"], 2):
                continue
            text, txs = "", []
        actor = congress.member_actor("house", r["first"], r["last"], r["state_dst"], directory)
        if not txs:
            scanned += 1
            label = ("Scanned paper filing — open the PDF" if len(text.strip()) < 200
                     else "Couldn't read this report automatically — open the PDF")
            ev = congress.make_event("house", f"{r['doc_id']}:0", {"asset": label, "action": "other",
                                                                    "action_label": "Paper filing" if len(text.strip()) < 200 else "Unreadable report"},
                                     actor, r["filing_date"], url)
            ev["details"]["unparsed"] = True
            n += add_events(state, [ev], new_ids)
        evs = [congress.make_event("house", f"{r['doc_id']}:{i}", t, actor, r["filing_date"], url) for i, t in enumerate(txs)]
        n += add_events(state, evs, new_ids)
        remember_congress(settings, evs)
        parsed += 1
        mark_seen(state, "house", r["doc_id"])
    return n, f"{parsed} reports parsed ({scanned} scanned/unparsed), {max(0, len(todo) - parsed)} waiting"


# ---------------------------------------------------------------- 6. congress — Senate

def senate_key(last: str, t: dict) -> str:
    """Same id whether the trade came from eFD, Quiver or FMP, so switching sources never duplicates."""
    from .util import slug
    what = t.get("ticker") or slug(t.get("asset") or "")[:24]
    o = (t.get("owner") or "").strip().lower()
    owner = ("spouse" if o.startswith("sp") else "joint" if o.startswith("jo") else
             "child" if o.startswith(("ch", "de")) else "self")
    return slug(f"{last}-{t.get('trade_date')}-{what}-{t.get('action')}-{owner}-{int(t.get('lo') or 0)}")


def job_senate(http: HttpClient, settings: Settings, state: dict, budget: Budget, new_ids: list,
               directory: dict) -> tuple[int, str]:
    sec_ = settings.secrets
    n = 0
    reason = None
    try:
        reports = congress.efd_recent_ptrs(http, days=21)
        for r in reports:
            if is_seen(state, "senate", r["report_id"]):
                continue
            actor = congress.member_actor("senate", r["first"], r["last"], None, directory)
            if r["paper"]:
                ev = congress.make_event("senate", f"paper-{r['report_id']}",
                                         {"asset": "Paper filing — open on eFD", "action": "other",
                                          "action_label": "Paper filing"}, actor, r["filed"], r["url"])
                ev["details"]["unparsed"] = True
                n += add_events(state, [ev], new_ids)
                mark_seen(state, "senate", r["report_id"])
                continue
            if not budget.ok(30):
                break
            html = http.get(r["url"]).text()
            evs = [congress.make_event("senate", senate_key(r["last"], t), t, actor, r["filed"], r["url"])
                   for t in congress.parse_efd_ptr_html(html)]
            n += add_events(state, evs, new_ids)
            remember_congress(settings, evs)
            mark_seen(state, "senate", r["report_id"])
        return n, f"official eFD: {len(reports)} recent reports"
    except Exception as e:  # noqa: BLE001 – bot walls return 403s *or* HTML challenges; either way, fall back
        reason = str(e)[:120]
    rows, src = [], None
    try:
        if sec_.quiver_key:
            rows, src = [r for r in congress.quiver_latest(http, sec_.quiver_key) if r["chamber"] == "senate"], "quiver"
        elif sec_.fmp_key:
            rows, src = congress.fmp_latest(http, sec_.fmp_key, "senate"), "fmp"
    except HttpError as e:
        return 0, f"eFD unavailable ({reason}); {('Quiver' if sec_.quiver_key else 'FMP')} fallback failed: HTTP {e.status}"
    if not src:
        return 0, f"official eFD unavailable from this server ({reason}); add QUIVER_API_KEY (or FMP_API_KEY) for Senate trades"
    cutoff = (now_utc() - dt.timedelta(days=settings.c("app", "feed_days", default=120))).date().isoformat()
    rows = [r for r in rows if (r.get("filed") or r.get("trade_date") or "") >= cutoff]   # only what the feed keeps
    evs = []
    for r in rows:
        actor = congress.member_actor("senate", r.get("first", ""), r.get("last", ""), r.get("state_dst"), directory)
        evs.append(congress.make_event("senate", senate_key(r.get("last", ""), r), r, actor, r.get("filed"), r.get("url")))
        evs[-1]["source"]["name"] = f"{src.upper()} (from the official Senate PTR)"
    n += add_events(state, evs, new_ids)
    remember_congress(settings, evs)
    return n, f"eFD unavailable ({reason}); used {src.upper()} fallback ({len(rows)} rows)"


def remember_congress(settings: Settings, evs: list[dict]) -> None:
    """Every disclosed congressional buy goes into the long-term history used for win rates."""
    rows = [{"id": e["id"], "actor_id": e["actor"]["id"], "actor_name": e["actor"]["name"], "role": e["actor"].get("role"),
             "ticker": e["ticker"], "action": e["action"], "trade_date": e["trade_date"], "filed": e.get("filed_at"),
             "lo": (e.get("value_range") or [None])[0]}
            for e in evs if e.get("ticker") and e.get("action") == "buy"]
    if not rows:
        return
    path = settings.heavy_dir / "congress_hist.json.gz"
    hist = read_json(path) or []
    have = {h.get("id") for h in hist}
    new = [r for r in rows if r["id"] not in have]
    if not new:          # nothing new → don't rewrite (and re-upload) the large cache
        return
    hist.extend(new)
    write_json(path, hist[-60000:])
    settings.mark_heavy_dirty()


# ---------------------------------------------------------------- 7. eToro

def job_etoro(http: HttpClient, settings: Settings, state: dict, budget: Budget, new_ids: list) -> tuple[int, str]:
    s = settings.secrets
    cfg = settings.c("etoro")
    api = Etoro(http, s.etoro_api_key, s.etoro_user_key)
    est = state.setdefault("etoro", {})
    last_rank = parse_iso(est.get("ranked_at"))
    if not est.get("leaders") or not last_rank or (now_utc() - last_rank).total_seconds() > 6 * 3600:
        rows = api.rankings(cfg["period"], cfg["max_risk_score"], cfg["min_copiers"])
        est["leaders"] = [leader_card(r) for r in pick_leaders(rows, cfg["top_n"], cfg["min_trades"])]
        est["ranked_at"] = iso(now_utc())
        est["universe"] = len(rows)
    snaps = est.setdefault("snapshots", {})
    instr = est.setdefault("instruments", {})
    n = 0
    pending: list[tuple[dict, str, dict, str]] = []
    for L in est["leaders"]:
        if not budget.ok(20):
            break
        u = L["username"]
        try:
            port = api.live_portfolio(u)
        except HttpError as e:
            L["error"] = f"portfolio HTTP {e.status}"
            continue
        positions = list(port.get("positions") or [])
        opened, closed, snap = diff_positions(snaps.get(u), positions)
        snaps[u] = snap
        for pid, p in opened:
            pending.append((L, pid, p, "open"))
        for pid, p in closed:
            pending.append((L, pid, p, "close"))
    need = [int(p["instrumentId"]) for _, _, p, _ in pending if p.get("instrumentId") and str(p["instrumentId"]) not in instr]
    if need:
        for k, v in api.instruments(need).items():
            instr[str(k)] = v
    for L, pid, p, kind in pending:
        meta = instr.get(str(p.get("instrumentId"))) or {}
        sym = (meta.get("symbol") or "").upper() or None
        ev = {
            "id": f"etoro:{L['username']}:{pid}:{kind}",
            "kind": "etoro",
            "ticker": sym,
            "company": meta.get("name") or sym or f"instrument {p.get('instrumentId')}",
            "actor": {"name": L.get("name") or L["username"], "id": f"etoro:{L['username']}",
                      "role": f"eToro Popular Investor · win rate {L.get('win_ratio') or 0:.0f}% · risk {L.get('risk_score')}",
                      "type": "etoro"},
            "action": kind,
            "action_label": ("Opened " if kind == "open" else "Closed ") + ("long" if p.get("isBuy", True) else "short"),
            "trade_date": (p.get("openTimestamp") or iso(now_utc()))[:10] if kind == "open" else now_utc().date().isoformat(),
            "filed_at": iso(now_utc()),
            "lag_days": 0,
            "price": to_float(p.get("openRate")),
            "value": None,
            "details": {"is_buy": p.get("isBuy", True), "leverage": p.get("leverage"),
                        "investment_pct": to_float(p.get("investmentPct")), "net_profit_pct": to_float(p.get("netProfit")),
                        "asset_type": meta.get("type"), "win_ratio": L.get("win_ratio"), "gain_pct": L.get("gain_pct"),
                        "risk_score": L.get("risk_score"), "copiers": L.get("copiers"),
                        "opened_at": p.get("openTimestamp")},
            "source": {"name": "eToro Public API", "url": L.get("url")},
        }
        n += add_events(state, [ev], new_ids)
    return n, f"{len(est['leaders'])} investors followed, {len(pending)} position changes"


# ---------------------------------------------------------------- 8. House backfill (for Congress win rates)

def job_house_backfill(http: HttpClient, settings: Settings, state: dict, budget: Budget, directory: dict,
                       max_docs: int = 150) -> int:
    """Parse older House PTRs (queued by job_house) into a compact history used only for track records."""
    queue: dict = state.setdefault("house_backfill", {})
    path = settings.heavy_dir / "congress_hist.json.gz"
    hist = read_json(path) or []
    done = 0
    for doc_id in sorted(queue, key=lambda k: queue[k].get("filing_date") or "", reverse=True)[:max_docs]:
        if not budget.ok(150):
            break
        r = queue[doc_id]
        try:
            raw = http.get(congress.house_pdf_url(r["year"], doc_id), timeout=60).content
            txs = congress.parse_house_ptr_text(congress.pdf_text(raw))
        except HttpError as e:
            if e.status != 404:
                continue
            txs = []
        except Exception:  # noqa: BLE001
            txs = []
        actor = congress.member_actor("house", r["first"], r["last"], r["state_dst"], directory)
        for t in txs:
            if t.get("ticker"):
                hist.append({"actor_id": actor["id"], "actor_name": actor["name"], "ticker": t["ticker"],
                             "action": t["action"], "trade_date": t["trade_date"], "filed": r["filing_date"],
                             "lo": t["lo"], "hi": t["hi"]})
        queue.pop(doc_id, None)
        done += 1
    if done:
        write_json(path, hist[-60000:])
        settings.mark_heavy_dirty()
    return done
