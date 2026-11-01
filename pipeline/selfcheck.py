"""The app audits itself at the end of every refresh and publishes the result (data/audit.json).

Checks: every published file exists, parses and is fresh; the feed's structure; price sanity; every forecast is a
valid, ordered distribution anchored on a real price; valuation numbers are finite and in range; both ledgers
are being graded (nothing stuck) and the 80% ranges are near target once there's enough data; each data source
ran recently enough for its schedule; the Claude analyst's health and spend; cache sizes; config sanity; and,
when GitHub gives the run a token, whether the automated test suite last passed.

A new failure sends one push alert (ntfy/Telegram, if set up). The phone app runs its own device-side checks
on top of this whenever it refreshes (see web/app.js, selfCheck).
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

from .util import iso, log, now_et, now_utc, parse_iso, redact, write_json

REQUIRED = {"feed.json": ["generated_at", "events"], "quotes.json": ["generated_at", "quotes"],
            "meta.json": ["generated_at", "version"], "health.json": ["generated_at", "sources"],
            "market.json": ["generated_at"], "ideas.json": ["generated_at"], "accuracy.json": ["generated_at"],
            "leaders.json": ["generated_at"], "forecast_accuracy.json": ["generated_at"]}
KINDS = {"insider", "fund", "activist", "congress", "etoro", "etf"}
SIGNALS = {"strong", "notable", "mild", "weak", "noise"}
VAL_LABELS = {"Very cheap", "Cheap", "Fairly valued", "Expensive", "Very expensive"}
BUBBLE = {"Low", "Moderate", "Elevated", "High"}
QK = ["05", "10", "25", "50", "75", "90", "95"]


class Audit:
    def __init__(self):
        self.checks: list[dict] = []

    def add(self, area: str, cid: str, status: str, msg: str, detail=None) -> None:
        c = {"area": area, "id": cid, "status": status, "msg": msg}
        if detail:
            c["detail"] = detail if isinstance(detail, (str, int, float)) else json.loads(json.dumps(detail, default=str))
        self.checks.append(c)

    def ok(self, area, cid, msg, detail=None):
        self.add(area, cid, "pass", msg, detail)

    def warn(self, area, cid, msg, detail=None):
        self.add(area, cid, "warn", msg, detail)

    def fail(self, area, cid, msg, detail=None):
        self.add(area, cid, "fail", msg, detail)


def _age_min(ts: str | None) -> float | None:
    d = parse_iso(ts) if ts else None
    return None if d is None else (now_utc() - d).total_seconds() / 60


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def check_files(a: Audit, out: Path) -> dict:
    docs = {}
    for name, keys in REQUIRED.items():
        p = out / name
        if not p.exists():
            a.fail("files", f"file:{name}", f"{name} is missing")
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except ValueError as e:
            a.fail("files", f"file:{name}", f"{name} isn't valid JSON ({e})")
            continue
        miss = [k for k in keys if k not in d]
        age = _age_min(d.get("generated_at"))
        if miss:
            a.fail("files", f"file:{name}", f"{name} lacks {', '.join(miss)}")
            continue
        if age is None or age > 30:          # every file is rewritten each run: an old one means its writer failed
            a.fail("files", f"file:{name}", f"{name} wasn't rewritten this run ({age and round(age)} minutes old)")
        docs[name] = d
    if len(docs) == len(REQUIRED) and not any(c["status"] == "fail" and c["area"] == "files" for c in a.checks):
        a.ok("files", "files", f"All {len(REQUIRED)} data files present, valid and rewritten this run")
    return docs


def _business_days_since(d) -> int:
    import datetime as dt
    from .forecast import is_trading_day
    today, n, x = now_et().date(), 0, d
    while x < today:
        x += dt.timedelta(days=1)
        if is_trading_day(x):
            n += 1
    return n


def check_data_age(a: Audit, docs: dict) -> None:
    """Is the data itself moving? (The files are rewritten every run even when a source has died.)"""
    import datetime as dt
    ev = (docs.get("feed.json") or {}).get("events") or []
    newest = max((str(e.get("first_seen") or e.get("filed_at") or "")[:10] for e in ev), default="")
    if newest:
        n = _business_days_since(dt.date.fromisoformat(newest))
        if n > 5:
            a.fail("freshness", "fresh:feed", f"No new trades for {n} business days (newest {newest}): the filing feeds may have stopped")
        elif n > 2:
            a.warn("freshness", "fresh:feed", f"No new trades for {n} business days (newest {newest})")
        else:
            a.ok("freshness", "fresh:feed", f"Newest trade seen {newest}")
    qs = ((docs.get("quotes.json") or {}).get("quotes") or {}).values()
    asof = max((str(q.get("asof") or "")[:10] for q in qs if q.get("asof")), default="")
    if asof:
        n = _business_days_since(dt.date.fromisoformat(asof))
        if n > 4:
            a.fail("freshness", "fresh:prices", f"Prices haven't updated for {n} business days (newest {asof})")
        elif n > 2:
            a.warn("freshness", "fresh:prices", f"Prices are {n} business days old (newest {asof})")
        else:
            a.ok("freshness", "fresh:prices", f"Newest price {asof}")


def check_feed(a: Audit, feed: dict | None) -> None:
    if not feed:
        return
    ev = feed.get("events") or []
    ids = [e.get("id") for e in ev]
    dup = len(ids) - len(set(ids))
    bad = []
    for e in ev:
        if not e.get("id") or e.get("kind") not in KINDS or not isinstance(e.get("actor"), dict) or not e["actor"].get("name"):
            bad.append(e.get("id") or "?")
            continue
        sc = e.get("score")
        if sc is not None and not (0 <= sc <= 100):
            bad.append(e["id"])
        if e.get("signal") not in SIGNALS:
            bad.append(e["id"])
    if dup:
        a.fail("feed", "feed:dupes", f"{dup} duplicate trade ids in the feed")
    if bad:
        a.fail("feed", "feed:schema", f"{len(bad)} trades have missing or invalid fields", bad[:10])
    if not dup and not bad:
        a.ok("feed", "feed:schema", f"{len(ev)} trades, all well-formed and unique")


def check_quotes(a: Audit, quotes: dict | None) -> None:
    if not quotes:
        return
    qs = quotes.get("quotes") or {}
    bad, odd, stale = [], [], []
    today = now_et().date()
    for t, q in qs.items():
        p = q.get("price")
        if p is not None and (not _finite(p) or p <= 0):
            bad.append(t)
            continue
        hi, lo = q.get("hi_52w"), q.get("lo_52w")
        if p and hi and lo and (p > hi * 3 or p < lo / 3):
            odd.append(t)
        asof_d = None
        if q.get("asof"):
            try:
                import datetime as dt
                asof_d = dt.date.fromisoformat(str(q["asof"])[:10])
            except ValueError:
                asof_d = None
        if asof_d and (today - asof_d).days > 5:
            stale.append(t)
    if bad:
        a.fail("prices", "prices:invalid", f"{len(bad)} stocks have an invalid price", bad[:10])
    if odd:
        a.warn("prices", "prices:jumps", f"{len(odd)} prices are far outside their 52-week range (possible bad data or split)", odd[:10])
    if len(stale) > max(5, len(qs) * 0.2):
        a.warn("prices", "prices:stale", f"{len(stale)} of {len(qs)} prices are over 5 days old", stale[:10])
    if not bad and not odd:
        a.ok("prices", "prices:sanity", f"{len(qs)} prices positive and within plausible ranges")


def check_stock_files(a: Audit, out: Path, state: dict | None = None, forecasts_on: bool = True) -> dict:
    d = out / "stock"
    files = sorted(d.glob("*.json")) if d.exists() else []
    sp = (state or {}).get("stock_pages") or {}
    if sp.get("names") and len(files) < 0.9 * sp["names"]:
        a.fail("forecasts", "stock:missing", f"Only {len(files)} of {sp['names']} stock pages were published")
    elif sp.get("failed"):
        a.warn("forecasts", "stock:errors", f"{sp['failed']} stock pages couldn't be rebuilt this run (the last good ones are kept)")
    if not files:
        a.warn("forecasts", "stock:none", "No per-stock files yet (forecasts appear once prices are stored)")
        return {"n": 0}
    bad_fc, bad_val, bad_cl, n_fc, n_val = [], [], [], 0, 0
    made: list[str] = []
    for f in files:
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
        except ValueError:
            bad_fc.append(f.stem)
            continue
        fc = doc.get("forecast")
        if fc:
            n_fc += 1
            if not _forecast_ok(fc):
                bad_fc.append(f.stem)
            else:
                made.append(fc["made"])
        v = doc.get("valuation")
        if v:
            n_val += 1
            if (v.get("label") is not None and v["label"] not in VAL_LABELS) or \
               (v.get("rich_pctl") is not None and not (0 <= v["rich_pctl"] <= 100)) or \
               any(v.get(k) is not None and (not _finite(v[k]) or v[k] <= 0) for k in ("ps", "pe", "ev_s", "pb")):
                bad_val.append(f.stem)
        b = doc.get("bubble")
        if b and b.get("level") not in BUBBLE:
            bad_val.append(f.stem)
        if not _claude_ok(doc.get("claude"), doc.get("blend")):
            bad_cl.append(f.stem)
    if bad_cl:
        a.fail("forecasts", "claude:valid", f"{len(bad_cl)} Claude or combined forecasts are malformed", bad_cl[:10])
    if bad_fc:
        a.fail("forecasts", "forecast:valid", f"{len(bad_fc)} forecasts are malformed (unordered, non-positive or misdated)", bad_fc[:10])
    else:
        a.ok("forecasts", "forecast:valid", f"{n_fc} forecasts: every range ordered, positive and anchored on a real price")
    _check_index(a, out, n_fc)
    if forecasts_on:
        _check_forecast_age(a, made, n_fc, len(files))
    else:
        a.ok("forecasts", "forecast:age", "Price forecasts are switched off in config")
    if bad_val:
        a.fail("valuation", "valuation:valid", f"{len(bad_val)} valuation/bubble records are out of range", bad_val[:10])
    elif n_val:
        a.ok("valuation", "valuation:valid", f"{n_val} valuations with finite, in-range numbers")
    return {"n": len(files), "forecasts": n_fc, "valuations": n_val}


def _check_index(a: Audit, out: Path, n_fc: int) -> None:
    """The All forecasts list (forecasts.json): there, rewritten this run, and listing every stock with a forecast."""
    p = out / "forecasts.json"
    try:
        d = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except ValueError:
        d = None
    if not isinstance(d, dict) or not isinstance(d.get("stocks"), list):
        a.fail("forecasts", "forecast:index", "The All forecasts list (forecasts.json) is missing or unreadable")
        return
    age = _age_min(d.get("generated_at"))
    n = len(d["stocks"])
    if age is None or age > 30:
        a.fail("forecasts", "forecast:index", f"The All forecasts list wasn't rewritten this run ({age and round(age)} minutes old)")
    elif abs(n - n_fc) > max(2, 0.02 * n_fc):
        a.warn("forecasts", "forecast:index", f"The All forecasts list has {n} stocks but {n_fc} stock pages have a forecast")
    else:
        a.ok("forecasts", "forecast:index", f"The All forecasts list has all {n} stocks with a forecast")


def _sessions_behind(made: str, last) -> int:
    """Trading sessions between a forecast's anchor day and the latest finished session."""
    import datetime as dt
    from .forecast import is_trading_day
    try:
        d = dt.date.fromisoformat(str(made)[:10])
    except ValueError:
        return 99
    n = 0
    while d < last and n < 99:
        d += dt.timedelta(days=1)
        n += is_trading_day(d)
    return n


def _check_forecast_age(a: Audit, made: list[str], n_fc: int, n_files: int) -> None:
    """Forecasts are rebuilt after every new close, so most should sit on the latest finished session. A few
    old ones are normal (halted or delisted stocks); most of them old means the forecast job has stalled."""
    if n_files and not n_fc:
        a.warn("forecasts", "forecast:none", f"{n_files} stock pages but none has a forecast (prices may be missing)")
        return
    if not made:
        return
    from .forecast import last_complete_session
    last = last_complete_session(now_et())
    behind = sorted(_sessions_behind(m, last) for m in made)
    old = sum(b >= 2 for b in behind)
    med = behind[len(behind) // 2]
    if med >= 2:
        a.fail("forecasts", "forecast:age", f"Most forecasts are {med} sessions old (latest close {last}): "
               "the forecast job looks stalled")
    elif old > max(3, 0.2 * len(made)):
        a.warn("forecasts", "forecast:age", f"{old} of {len(made)} forecasts are 2+ sessions old (latest close {last})")
    else:
        a.ok("forecasts", "forecast:age", f"Forecasts are up to date with the {last} close")


def _claude_ok(cf: dict | None, bl: dict | None) -> bool:
    try:
        for row in ((cf or {}).get("horizons") or {}).values():
            v = [row["low_10"], row["median"], row["high_90"]]
            if any(not _finite(x) or x <= 0 for x in v) or v != sorted(v):
                return False
        for row in (bl or {}).values():
            q = [row["q"][k] for k in QK]
            if any(not _finite(x) or x <= 0 for x in q) or q != sorted(q):
                return False
        return True
    except (KeyError, TypeError):
        return False


def _forecast_ok(fc: dict) -> bool:
    try:
        if not (_finite(fc["anchor"]) and fc["anchor"] > 0):
            return False
        prev_date = fc["made"]
        for name in ("1d", "1w", "1m", "3m", "1y"):
            h = fc["horizons"][name]
            q = [h["q"][k] for k in QK]
            if any(not _finite(x) or x <= 0 for x in q) or q != sorted(q):
                return False
            if not (0 <= h["p_up"] <= 1) or h["date"] <= prev_date:
                return False
            prev_date = h["date"]
        fan = fc["fan"]
        n = len(fan["dates"])
        if fan["dates"] != sorted(fan["dates"]) or any(len(fan["p"][k]) != n for k in QK):
            return False
        for i in range(n):
            col = [fan["p"][k][i] for k in QK]
            if col != sorted(col) or col[0] <= 0:
                return False
        return True
    except (KeyError, TypeError):
        return False


def check_ledgers(a: Audit, state: dict, settings) -> None:
    from . import fledger
    acc = state.get("accuracy") or {}
    old = acc.get("oldest_pending")
    if old:
        import datetime as dt
        days = (now_et().date() - dt.date.fromisoformat(old)).days
        if days > 63 * 1.5 + 60:
            a.warn("ledger", "ledger:stuck", f"A signal forecast from {old} is still ungraded after {days} days")
    m = state.get("fc_metrics") or {}
    L = fledger.load(settings.heavy_dir)
    import datetime as dt
    today = now_et().date()
    stuck = [k for k, e in L["pending"].items()
             if (today - dt.date.fromisoformat(e.get("ref") or e["made"])).days > e["h"] * 1.45 + 60]
    if stuck:
        a.warn("ledger", "fledger:stuck", f"{len(stuck)} price forecasts are past their horizon but not graded", stuck[:8])
    else:
        a.ok("ledger", "fledger:grading", f"Forecast ledger: {len(L['pending'])} waiting, nothing stuck")
    for src in ("quant", "claude"):
        for name, v in ((m.get("by") or {}).get(src) or {}).items():
            rec = v.get("recent") or {}
            c = (rec.get("in80") or {})
            if rec.get("n", 0) >= 200 and c.get("p") is not None and not (0.70 <= c["p"] <= 0.90):
                a.warn("accuracy", f"cov:{src}:{name}",
                       f"{'Model' if src == 'quant' else 'Claude'} {name} 80% ranges held {c['p']*100:.0f}% of recent outcomes "
                       f"(target 80%); the calibration step is adjusting")


CHRONIC = {"congress_senate"}      # the Senate site blocks cloud servers (documented): never more than a warning
# jobs with no "refresh" entry of their own: minutes, or the config key whose cadence they follow
DEFAULT_EVERY = {"prices": "prices_off_hours", "reference": "fundamentals", "market_history": 240,
                 "context": 5, "claude_forecasts": 5, "conviction": 5, "stock_pages": 5, "accuracy_page": 5,
                 "alerts": 5}


def check_sources(a: Audit, state: dict, settings) -> None:
    """A source that keeps failing is a warning; one still being tried but with no success for three days (or for
    twelve of its own intervals, whichever is longer) is a failure, which sends a push notice."""
    R = settings.c("refresh", default={}) or {}
    jobs = state.get("jobs") or {}
    stale, failing, dead = [], [], []
    for name, j in jobs.items():
        every = R.get(name, DEFAULT_EVERY.get(name))
        if isinstance(every, str):
            every = R.get(every)
        if not isinstance(every, (int, float)) or every <= 0:
            every = j.get("every")                 # the cadence the run itself scheduled the job with
        if not isinstance(every, (int, float)) or every <= 0:
            continue
        err = str(j.get("error") or "")
        if "SEC_CONTACT_EMAIL" in err:
            continue
        tried = _age_min(j.get("last_run"))
        if tried is None or tried > max(every * 3, 120):
            continue                               # not being attempted any more (switched off, or not due)
        age = _age_min(j.get("last_ok"))
        cf = j.get("consecutive_failures", 0)
        if cf >= 3 and age is not None and age > max(every * 12, 72 * 60) and name not in CHRONIC:
            dead.append(f"{name}: no success for {round(age / 60)}h ({redact(err)[:100]})")
        elif cf >= 3:
            failing.append(f"{name}: {redact(err)[:120]}")
        elif age is not None and age > max(every * 6, 180) and age > 24 * 60:
            stale.append(f"{name} (last OK {round(age / 60)}h ago)")
    if dead:
        a.fail("sources", "sources:dead", f"{len(dead)} data sources have been down for days", dead[:8])
    if failing:
        a.warn("sources", "sources:failing", f"{len(failing)} data sources failing repeatedly", failing[:8])
    if stale:
        a.warn("sources", "sources:stale", f"{len(stale)} data sources haven't succeeded for over a day", stale[:8])
    if not failing and not stale and not dead:
        a.ok("sources", "sources", "Every scheduled source succeeded recently")


def check_analyst(a: Audit, state: dict, settings) -> None:
    an = state.get("analyst") or {}
    has = settings.secrets.present().get("claude_api")
    if not has:
        a.ok("analyst", "analyst", "Claude analyst not set up (optional)")
        return
    lr = an.get("last_run") or {}
    age = _age_min(lr.get("started"))
    budget = float((settings.c("analyst", default={}) or {}).get("daily_budget_usd", 3.0))
    spent = (an.get("today") or {}).get("usd") or 0
    if age is None:
        a.warn("analyst", "analyst:run", "The Claude analyst hasn't run yet")
    elif age > 180:
        a.warn("analyst", "analyst:run", f"The Claude analyst last ran {round(age / 60)} hours ago")
    elif lr.get("errors"):
        a.warn("analyst", "analyst:errors", f"Claude's last run had {len(lr['errors'])} errors", [redact(x) for x in lr["errors"][:3]])
    else:
        a.ok("analyst", "analyst:run", "Claude analyst ran recently without errors")
    if spent > budget * 1.05:
        a.fail("analyst", "analyst:budget", f"Claude spend today ${spent:.2f} is over the ${budget:.2f} budget")


def check_sizes(a: Audit, settings) -> None:
    small = sum(f.stat().st_size for f in settings.cache_dir.glob("*") if f.is_file())
    heavy = sum(f.stat().st_size for f in settings.heavy_dir.rglob("*") if f.is_file())
    site = sum(f.stat().st_size for f in settings.out_dir.rglob("*") if f.is_file())
    det = {"state_cache_mb": round(small / 1e6, 1), "heavy_cache_mb": round(heavy / 1e6, 1), "site_data_mb": round(site / 1e6, 1)}
    if small > 60e6 or heavy > 2.5e9 or site > 400e6:
        a.warn("caches", "sizes", "A cache or the published data is getting large", det)
    else:
        a.ok("caches", "sizes", f"Caches {det['state_cache_mb']} MB + {det['heavy_cache_mb']} MB, site data {det['site_data_mb']} MB", det)


def check_config(a: Audit, settings) -> None:
    probs = []
    an = settings.c("analyst", default={}) or {}
    if not 0 < float(an.get("daily_budget_usd", 3) or 0) <= 100:
        probs.append("analyst.daily_budget_usd should be between 0 and 100")
    if an.get("deep_effort", "max") not in ("low", "medium", "high", "xhigh", "max"):
        probs.append("analyst.deep_effort must be low/medium/high/xhigh/max")
    fc = settings.c("forecast", default={}) or {}
    if not 200 <= int(fc.get("paths", 2000) or 0) <= 20000:
        probs.append("forecast.paths should be 200-20000")
    for t in (settings.c("bubble", "core", default=[]) or []):
        if not isinstance(t, str) or not t.strip():
            probs.append("bubble.core has a blank ticker")
            break
    if probs:
        a.warn("config", "config", "Config values to fix", probs)
    else:
        a.ok("config", "config", "Config values are in range")


def check_tests(a: Audit, http) -> None:
    tok, repo = os.environ.get("GITHUB_TOKEN"), os.environ.get("GITHUB_REPOSITORY")
    if not tok or not repo or http is None:
        return
    from .util import register_secret
    register_secret(tok)
    branch = os.environ.get("GITHUB_REF_NAME") or "main"
    try:
        d = http.get_json(f"https://api.github.com/repos/{repo}/actions/workflows/tests.yml/runs"
                          f"?branch={branch}&status=completed&per_page=5",
                          headers={"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json"},
                          retries=0, timeout=15)
    except Exception as e:  # noqa: BLE001
        a.warn("tests", "tests", f"Couldn't read the test workflow's status ({type(e).__name__})")
        return
    runs = [r for r in (d.get("workflow_runs") or []) if r.get("conclusion") not in ("cancelled", "skipped", None)]
    if not runs:
        a.warn("tests", "tests", f"No finished test run on {branch} yet")
        return
    r = runs[0]
    concl = r.get("conclusion")
    when = r.get("updated_at")
    if concl == "success":
        a.ok("tests", "tests", f"Automated tests passed on {branch} ({when[:10] if when else ''})")
    else:
        a.fail("tests", "tests", f"Automated tests {concl} on {branch} on {when[:10] if when else '?'}", r.get("html_url"))


def check_run(a: Audit, run: dict) -> None:
    res = run.get("results") or {}
    bad = {k: v.get("error") for k, v in res.items() if not v.get("ok") and "SEC_CONTACT_EMAIL" not in str(v.get("error"))}
    st = run.get("http") or {}
    hosts = [f"{h}: {s.get('errors')}/{s.get('requests')} failed" for h, s in st.items()
             if s.get("requests", 0) >= 10 and s.get("errors", 0) / max(1, s.get("requests", 1)) > 0.3]
    if bad:
        a.warn("run", "run:jobs", f"{len(bad)} jobs failed this run", {k: redact(str(v))[:160] for k, v in list(bad.items())[:8]})
    else:
        a.ok("run", "run:jobs", f"All {len(res)} jobs this run succeeded")
    if hosts:
        a.warn("run", "run:http", "High error rate from some sites", hosts[:6])


def run_and_publish(settings, state: dict, run: dict, http=None) -> dict:
    a = Audit()
    out = settings.out_dir
    docs = check_files(a, out)
    check_feed(a, docs.get("feed.json"))
    check_quotes(a, docs.get("quotes.json"))
    check_data_age(a, docs)
    files = check_stock_files(a, out, state, (settings.c("forecast", default={}) or {}).get("enabled", True))
    for fn in (check_ledgers, check_sources, check_analyst):
        try:
            fn(a, state, settings)
        except Exception as e:  # noqa: BLE001 – a broken check is itself a finding
            a.warn("audit", f"check:{fn.__name__}", f"The {fn.__name__} check crashed: {type(e).__name__}: {redact(str(e))[:120]}")
    check_sizes(a, settings)
    check_config(a, settings)
    check_tests(a, http)
    check_run(a, run)
    counts = {s: sum(1 for c in a.checks if c["status"] == s) for s in ("pass", "warn", "fail")}
    overall = "fail" if counts["fail"] else "warn" if counts["warn"] else "pass"
    doc = {"generated_at": iso(now_utc()), "overall": overall, "counts": counts, "checks": a.checks, "stock_files": files}
    write_json(out / "audit.json", doc)
    old = state.get("audit") or {}
    failing = sorted(c["id"] for c in a.checks if c["status"] == "fail")
    # a failure is "alerted" once a notice about it actually went out; until then it stays pending (throttled,
    # quiet hours, or a failed send), and it is forgotten once it clears, so a relapse alerts again
    alerted = {k for k in (old.get("alerted") or []) if k in failing}
    new = [c for c in a.checks if c["status"] == "fail" and c["id"] not in alerted]
    state["audit"] = {"overall": overall, "counts": counts, "failing": failing, "at": doc["generated_at"],
                      "alerted_at": old.get("alerted_at"), "alerted": sorted(alerted)}
    last = parse_iso(old.get("alerted_at"))
    throttled = last is not None and (now_utc() - last).total_seconds() < 6 * 3600
    if new and not throttled and http is not None and settings.c("selfcheck", "alert_on_fail", default=True):
        try:
            from .alerts import send_system
            if send_system(settings, http, f"Self-check: {len(new)} new problem{'s' if len(new) > 1 else ''}",
                           "\n".join(f"• {c['msg']}" for c in new[:6]), "#/status"):
                state["audit"]["alerted_at"] = iso(now_utc())
                state["audit"]["alerted"] = sorted(alerted | {c["id"] for c in new})
        except Exception as e:  # noqa: BLE001
            log.warning("self-check alert failed: %s", e)
    _heartbeat(settings, http, overall)
    log.info("self-check: %s (%s)", overall, counts)
    return doc


def _heartbeat(settings, http, overall: str) -> None:
    """Optional dead-man's switch: ping a monitoring URL (e.g. a free healthchecks.io check, secret HEARTBEAT_URL)
    at the end of every run. If the refresh stops running altogether, that service notices the silence; nothing
    inside the app can, because nothing inside it runs."""
    url = getattr(settings.secrets, "heartbeat_url", None)
    if not url or http is None or not str(url).startswith("https://"):
        return
    try:
        http.get(url + ("/fail" if overall == "fail" and "hc-ping.com" in url else ""), retries=0, timeout=10)
    except Exception as e:  # noqa: BLE001
        log.warning("heartbeat ping failed: %s", type(e).__name__)
