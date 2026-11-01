"""Entry point: python -m pipeline.main [--force job,job] [--verbose]

Runs every ~5 minutes on GitHub Actions. Each job has its own refresh interval (config.yaml → refresh) and
the whole run has a time budget, so slow sources never block fast ones."""
from __future__ import annotations

import argparse
import datetime as dt
import time
import traceback

from . import (alerts, congress, etf, extras, fundamentals, ideas, insight, jobs, ledger, market, publish, quant, risk,
               sec, selfcheck, track_records)
from .figi import FigiMapper
from .http import HttpClient
from .prices import (BENCH, MarketMatrix, PriceClient, PriceStore, backfill_market, load_splits, massive_splits,
                     record_splits, to_num)
from .scoring import Ctx, score_event
from .settings import Settings
from .state import job_due, job_record, load_state, save_state
from .util import (ET, iso, is_market_hours, log, now_et, now_utc, parse_date, parse_iso, read_json, redact,
                   setup_logging, write_json)


def run(force: set[str] | None = None, transport=None, settings: Settings | None = None) -> dict:
    force = force or set()
    settings = settings or Settings.load()
    s = settings.secrets
    s.register_all()
    http = HttpClient(sec_user_agent=s.sec_user_agent(), transport=transport)
    state = load_state(settings.cache_dir)
    R = settings.c("refresh")
    budget = jobs.Budget(R.get("run_budget_minutes", 9) * 60)
    http.deadline = time.monotonic() + budget.left() + 60
    new_ids: list[str] = []
    started = now_utc()
    results = {}

    def due(name: str, minutes: float) -> bool:
        j = state["jobs"].get(name)
        if j is not None and minutes and minutes > 0:
            j["every"] = minutes                   # lets the self-check judge every job against its own cadence
        return job_due(state, name, minutes, force=("all" in force or name in force))

    def do(name: str, fn, *a, **kw):
        t0 = time.monotonic()
        try:
            items, note = fn(*a, **kw)
            note = redact(str(note)) if note is not None else None      # notes are published on the Data tab
            job_record(state, name, True, items=items, note=note, extra={"seconds": round(time.monotonic() - t0, 1)})
            results[name] = {"ok": True, "items": items, "note": note}
            log.info("✓ %s: %s (%s)", name, items, note)
        except Exception as e:  # noqa: BLE001 – one source failing must not stop the others
            msg = redact(f"{type(e).__name__}: {e}")
            job_record(state, name, False, error=msg, extra={"seconds": round(time.monotonic() - t0, 1)})
            results[name] = {"ok": False, "error": msg}
            log.error("✗ %s: %s\n%s", name, msg, redact(traceback.format_exc(limit=3)))
        try:  # checkpoint: if the runner is killed later, finished work (and "already seen" marks) survive
            save_state(settings.cache_dir, state)
        except Exception as e:  # noqa: BLE001
            log.warning("checkpoint failed: %s", e)

    # The SEC requires automated tools to declare a contact email. Without it, SEC sources wait (and say so on the
    # Data tab) instead of sending undeclared traffic from GitHub's shared IPs, which can get them blocked.
    sec_ok = s.has_sec_contact()
    SEC_MSG = "Waiting for the SEC_CONTACT_EMAIL secret (the SEC requires a contact email from automated tools)"

    def need_sec(name: str) -> bool:
        if sec_ok:
            return True
        job_record(state, name, False, error=SEC_MSG)
        results[name] = {"ok": False, "error": SEC_MSG}
        return False

    # ---- reference data (cached daily)
    tick = read_json(settings.cache_dir / "tickers.json.gz") or {}
    fund_meta_path = settings.cache_dir / "fundamentals.json.gz"
    funda = read_json(fund_meta_path) or {}
    if due("reference", R.get("fundamentals", 1440)) or (not tick and sec_ok):
        def ref():
            nonlocal tick, funda
            leg = congress.load_legislators(http)
            write_json(settings.cache_dir / "legislators.json.gz", leg)
            if not sec_ok:
                raise RuntimeError(SEC_MSG)
            tick = sec.ticker_map(http)
            write_json(settings.cache_dir / "tickers.json.gz", tick)
            funda = fundamentals.load_fundamentals(http)
            write_json(fund_meta_path, funda)
            return len(tick.get("by_ticker", {})), f"{len(funda)} companies with size data"
        do("reference", ref)
    directory = read_json(settings.cache_dir / "legislators.json.gz") or {"by_last": {}}

    # ---- fast, high-value sources first
    if due("insiders_live", R["insiders_live"]) and need_sec("insiders_live"):
        do("insiders_live", jobs.job_insiders_live, http, settings, state, budget, new_ids)
    if due("sale_notices", R.get("sale_notices", 10)) and need_sec("sale_notices"):
        do("sale_notices", jobs.job_form144_live, http, settings, state, budget, new_ids, tick)
    if due("activists_live", R["activists_live"]) and need_sec("activists_live"):
        do("activists_live", jobs.job_activists_live, http, settings, state, budget, new_ids, tick)
    if s.etoro_api_key and s.etoro_user_key and settings.c("etoro", "enabled") and due("etoro", R["etoro"]) and budget.ok(60):
        do("etoro", jobs.job_etoro, http, settings, state, budget, new_ids)
    if settings.c("congress", "house") and due("congress_house", R["congress_house"]) and budget.ok(90):
        do("congress_house", jobs.job_house, http, settings, state, budget, new_ids, directory)
    if settings.c("congress", "senate") and due("congress_senate", R["congress_senate"]) and budget.ok(60):
        do("congress_senate", jobs.job_senate, http, settings, state, budget, new_ids, directory)
    fund_every = R["funds_deadline_window"] if jobs.in_13f_deadline_window() else R["funds"]
    figi = FigiMapper(http, settings.cache_dir, s.openfigi_key)
    if due("funds", fund_every) and budget.ok(120) and need_sec("funds"):
        do("funds", jobs.job_funds, http, settings, state, budget, new_ids, figi)
    # Yesterday's SEC daily index catches anything the live feed missed. Runs once a new business day is
    # available (from 6am ET), keeps going every run while a day is half-done, and catches up missed days.
    if (now_et().hour >= 6 and jobs.backstop_next_day(state) is not None
            and (state.get("backstop_pending") or due("daily_backstop", 60)) and budget.ok(120)
            and need_sec("daily_backstop")):
        do("daily_backstop", jobs.job_backstop, http, settings, state, budget, new_ids, tick)

    # ---- tidy the event store
    trim_events(state, settings)

    # ---- whole-market daily history (Massive free plan: past days only, 5 requests/min)
    matrix = MarketMatrix(settings.heavy_dir / "market.npz") if s.massive_key else None
    added = {"rows": 0}
    if matrix is not None:
        missing = matrix.missing_days()
        if missing and (due("market_history", 0) if len(missing) > 3 else due("market_history", 240)) and budget.ok(150):
            def mh():
                try:
                    n = backfill_market(http, s.massive_key, matrix,
                                        budget_seconds=max(45, min(240, budget.left() - 200)))
                finally:                         # keep any days fetched before a failure
                    if matrix.dirty:
                        matrix.save()
                        settings.mark_heavy_dirty()
                added["rows"] = n
                left = len(matrix.missing_days())
                return n, (f"{len(matrix.dates)} trading days × {len(matrix.tickers)} tickers stored"
                           + (f"; backfilling, {left} days to go" if left > 3 else ""))
            do("market_history", mh)
    # splits right after new rows arrive (and before forecasts and grading), so a split is never read as a crash
    if matrix is not None and matrix.dates and (due("splits", R.get("splits", 720)) or added["rows"]) and budget.ok(60):
        def splits_job():
            last = parse_date(state.get("splits_since") or (state.get("market") or {}).get("splits_since"))
            since = (last - dt.timedelta(days=14)) if last else dt.date.fromordinal(matrix.dates[0])
            sp = massive_splits(http, s.massive_key, since)
            record_splits(settings.cache_dir, [(x.get("ticker"), x.get("execution_date"),
                                                (to_num(x.get("split_to")) or 0) / (to_num(x.get("split_from")) or 1))
                                               for x in sp if x.get("ticker") and x.get("execution_date")])
            adj = matrix.apply_splits(sp)
            if matrix.dirty:
                matrix.save()
                settings.mark_heavy_dirty()
            state["splits_since"] = now_et().date().isoformat()
            return len(adj), (f"{len(sp)} splits since {since}; re-based "
                              + (", ".join(adj[:8]) + ("…" if len(adj) > 8 else "") if adj else "none needed"))
        do("splits", splits_job)

    # ---- prices + risk for every ticker in the feed
    store = PriceStore(settings.cache_dir)
    pc = PriceClient(http, store, s.alpaca_key_id, s.alpaca_secret, matrix=matrix)
    every = R["prices_market_hours"] if is_market_hours() else R["prices_off_hours"]
    tickers = sorted({e["ticker"] for e in state["events"] if e.get("ticker")})
    fresh_tickers = sorted({e["ticker"] for e in state["events"] if e["id"] in set(new_ids) and e.get("ticker")})
    if (due("prices", every) or fresh_tickers) and budget.ok(45):
        def price_job():
            want = tickers if due("prices", every) else fresh_tickers
            n1 = pc.refresh_daily(want, days=400)
            n2 = pc.refresh_latest(want)
            record_splits(settings.cache_dir, pc.split_events)
            update_quotes(state, store, want, funda, tick, load_splits(settings.cache_dir))
            store.prune(set(tickers))
            store.save()
            return n1 + n2, f"{len(want)} tickers via {pc.source}" + (f"; {len(pc.errors)} errors" if pc.errors else "")
        do("prices", price_job)

    # ---- transparent active ETFs: the funds' daily holdings files, diffed into trades (after the market
    # calendar is loaded, so a holiday before the file's date doesn't shift the trade date)
    if settings.c("etf", "enabled", default=True) and due("etf_daily", R.get("etf_daily", 120)) and budget.ok(60):
        def etf_job():
            cal = list(matrix.dates) if (matrix is not None and matrix.dates) else None
            evs, note = etf.refresh(http, settings.cache_dir, etf.fund_list(settings), state, cal)
            return jobs.add_events(state, evs, new_ids), note
        do("etf_daily", etf_job)

    # ---- track records (weekly, incremental, time-boxed)
    if due("track_records", R["track_records"]) and budget.ok(240):
        def track_job():
            hist = PriceStore(settings.cache_dir)
            hist.path = settings.heavy_dir / "prices_hist.json.gz"
            hist.data = read_json(hist.path, {}) or {}
            hpc = PriceClient(http, hist, s.alpaca_key_id, s.alpaca_secret, matrix=matrix)
            tr = state.setdefault("track", {})
            parts = []
            if not sec_ok:
                parts.append("insiders: " + SEC_MSG)
            elif hpc.matrix is not None:  # thousands of tickers: needs the whole-market history (free Massive key)
                ins = track_records.build_insider_records(http, settings, hist, hpc, budget)
                write_json(settings.cache_dir / "insider_months.json.gz", ins.pop("months"))
                tr["insiders"] = ins
                parts.append(f"insiders: {len(ins['records'])} people from {', '.join(ins['quarters'][:2])}…")
            else:
                parts.append("insiders: needs a free Massive key (whole-market price history)")
            bf = jobs.job_house_backfill(http, settings, state, budget, directory)
            parts.append(f"house backfill: {bf} reports")
            hist_rows = read_json(settings.heavy_dir / "congress_hist.json.gz") or []
            tr["congress"] = track_records.build_congress_records(state["events"], hist_rows, hist, hpc, budget)
            parts.append(f"congress: {len(tr['congress']['records'])} members")
            if sec_ok:
                tr["funds"] = track_records.build_fund_records(http, settings, figi, hist, hpc, budget, jobs.load_13f)
                parts.append(f"funds: {len(tr['funds']['records'])}")
            if hist.dirty:
                hist.save()
                settings.mark_heavy_dirty()
            return sum(len(v.get("records", {})) for v in tr.values() if isinstance(v, dict)), "; ".join(parts)
        do("track_records", track_job)

    # ---- measured predictability (daily): data sets, base rates, the out-of-sample-tested model, screen backtest
    panel = quant.Panel(matrix) if (matrix is not None and matrix.dates) else None
    allowed = set((tick.get("by_ticker") or {}).keys()) or None
    model_path = settings.heavy_dir / "quant_insider.pkl"
    if panel is not None and panel.ok() and due("predictability", R.get("predictability", 1440)) and budget.ok(120):
        def predict_job():
            q, bundle_new = quant.build_all(settings.heavy_dir, panel,
                                            read_json(settings.cache_dir / "insider_months.json.gz") or {},
                                            read_json(settings.heavy_dir / "congress_hist.json.gz") or [],
                                            state["events"], allowed, state.get("quant_gate"))
            state["quant_gate"] = q.get("gate")
            quant.save_bundle(model_path, bundle_new)
            settings.mark_heavy_dirty()
            state["quant"] = q
            m = q["insider_model"]
            note = (f"insider model: {m.get('predictability')} (test AUC {m['auc']:.3f}, {m['n_test']} held-out filings)"
                    if m.get("status") == "ok" and m.get("auc") is not None else f"insider model: {m.get('note') or m.get('status')}")
            return q["samples"]["insider_buys_known"], note
        do("predictability", predict_job)
    bundle = quant.load_bundle(model_path) if panel is not None else None

    if s.fred_key and due("macro", R.get("macro", 720)) and budget.ok(30):
        def macro_job():
            state["fred"] = market.fetch_fred(http, s.fred_key) or state.get("fred") or {}
            return len(state["fred"]), "St. Louis Fed (FRED) series"
        do("macro", macro_job)
    if due("market_context", R.get("market_context", 60)) or "regime" not in (state.get("market") or {}):
        def market_job():
            state["market"] = market.context(panel, allowed, state.get("fred"))
            return 1, f"regime: {state['market']['regime']['label']}"
        do("market_context", market_job)

    # ---- short interest (FINRA, twice a month) and next earnings dates (Alpha Vantage, daily)
    if due("short_interest", R.get("short_interest", 720)) and budget.ok(60):
        do("short_interest", extras.refresh_short_interest, http, settings.cache_dir)
    if s.alphavantage_key and due("earnings", R.get("earnings", 1440)) and budget.ok(30):
        do("earnings", extras.refresh_earnings, http, settings.cache_dir, s.alphavantage_key)
    # daily short-sale volume (FINRA Reg SHO, posted by 6pm ET): checked every few hours, one file per trading day
    if due("short_volume", R.get("short_volume", 180)) and budget.ok(60):
        watch_t = {str(t).upper() for t in settings.c("alerts", "watch_tickers", default=[]) or []}
        do("short_volume", extras.refresh_short_volume, http, settings.cache_dir, set(tickers) | watch_t)
    extras.attach(state.get("quotes") or {}, settings.cache_dir, panel if (panel is not None and panel.ok()) else None)

    # ---- per-stock research: company financials (SEC), news, valuation & bubble risk, price forecasts
    pan = panel if (panel is not None and panel.ok()) else None
    prio = insight.priority_tickers(state, settings)
    if due("companies", R.get("companies", 30)) and budget.ok(90) and need_sec("companies"):
        do("companies", insight.job_companies, http, settings, budget, tick, prio)
    if due("filings_live", R.get("filings_live", 10)) and budget.ok(45) and need_sec("filings_live"):
        do("filings_live", insight.job_filings_live, http, settings, tick, prio)
    if due("headlines", R.get("headlines", 60)) and budget.ok(60):
        do("headlines", insight.job_headlines, http, settings, budget, state, prio)
    sctx: dict = {}          # per-stock research context (valuation, bubble, news…)

    def ctx_job():
        nonlocal sctx
        sctx = insight.build_context(settings, state, pan, store, tick, prio)
        vm = state.get("valuation_meta") or {}
        return len(sctx), (f"{vm.get('n', 0)} stocks valued ({vm.get('fit_n', 0)} in the fundamentals fit"
                          + (f", R² {vm['r2']:.2f}" if vm.get("r2") is not None else "") + ")")
    do("context", ctx_job)
    if due("forecasts", R.get("forecasts", 30)) and budget.ok(60):
        do("forecasts", insight.job_forecasts, settings, state, store, budget, sctx, prio, settings.heavy_dir)
    # ---- deep history (Alpaca's free plan, back to 2016): built once, a couple of minutes a run, for the backtest
    deep = None
    fcfg = settings.c("forecast", default={}) or {}
    if pan is not None and s.alpaca_key_id and s.alpaca_secret and fcfg.get("deep_history", True):
        from .deep import DeepHistory
        deep = DeepHistory(settings.heavy_dir, start=str(fcfg.get("deep_start", "2016-01-01")))
        if not deep.done and due("deep_history", R.get("deep_history", 60)) and budget.ok(240):
            from .deep import listed_ciks as _listed
            listed_ciks = _listed(tick)              # companies still trading on a main exchange (renamed, not gone)

            def deep_job():
                try:
                    return deep.step(http, s.alpaca_key_id, s.alpaca_secret,
                                     seconds=max(30.0, min(float(fcfg.get("deep_seconds_per_run", 150)),
                                                           budget.left() - 330)),
                                     massive_key=s.massive_key, listed_ciks=listed_ciks)
                finally:
                    settings.mark_heavy_dirty()           # progress is saved even when a step fails
            do("deep_history", deep_job)
    bt_left = ((state.get("jobs") or {}).get("backtest") or {}).get("note") or ""
    if pan is not None and (due("backtest", R.get("backtest", 1440)) or "to go" in bt_left) and budget.ok(120):
        # the deep matrix (one to two hundred MB) is read from disk only when the backtest runs
        deep_pan = quant.Panel(deep) if deep is not None and deep.done else None
        do("backtest", insight.job_backtest, settings, state, pan, budget, settings.heavy_dir, allowed, sctx, deep_pan)

    # ---- score everything (cheap; context changes as new trades arrive)
    months = read_json(settings.cache_dir / "insider_months.json.gz") or {}
    ctx = build_ctx(state, settings, months, directory)
    recs = actor_records(state)
    led = ledger.load(settings.cache_dir)
    qsum = state.get("quant") or {}
    rates, lrates, msum = qsum.get("rates") or {}, ledger.kind_rates(led), qsum.get("insider_model")
    quotes = state.get("quotes") or {}
    live_groups = quant.live_insider_groups(state["events"], ctx.insider_routine)
    today_et = now_et().date().isoformat()
    for e in state["events"]:
        score_event(e, ctx)
        e.setdefault("score0", e.get("score"))      # as first scored: later filings change cluster counts
        r = recs(e)
        if r:
            e["actor_record"] = r
        else:
            e.pop("actor_record", None)
        qt = quotes.get(e.get("ticker") or "") or {}
        e["projection"] = quant.projection(qt.get("price"), qt.get("vol_1y"))
        ins = e.get("kind") == "insider"
        try:
            e["quant"] = quant.event_view(e, panel, bundle, msum, rates,
                                          ctx.insider_cluster(e).get("buyers", 0) if ins else 0,
                                          ctx.insider_routine(e) if ins else None, lrates, live_groups.get(e["id"]))
        except Exception as err:  # noqa: BLE001 – a bad feature row must not stop scoring
            log.warning("quant view %s: %s", e["id"], err)
            e["quant"] = None
        if e.get("ticker") and e.get("action") in TRACKED and e.get("signal") != "noise":
            # dated the day it is written down (never backdated to when the trade was first seen: after a cache
            # loss that would let part of the outcome be known before the forecast is recorded)
            ledger.record(led, f"ev:{e.get('ledger_id') or e['id']}", ticker=e["ticker"], made=today_et,
                          p=(e.get("quant") or {}).get("p"), source=(e.get("quant") or {}).get("source") or "none",
                          kind=e.get("kind"), action=e.get("action"))
    fresh = [e for e in state["events"] if e["id"] in set(new_ids)]

    # ---- ideas (screens) and Claude's work from the analyst workflow
    if due("ideas", R.get("ideas", 60)) or not state.get("ideas"):
        def ideas_job():
            state["ideas"] = ideas.apply_registry(ideas.build(state["events"], quotes, panel, allowed, qsum),
                                                  state.setdefault("idea_registry", {}))
            n = sum(len(v["items"]) for v in state["ideas"]["lists"].values())
            return n, ", ".join(f"{k}: {len(v['items'])}" for k, v in state["ideas"]["lists"].items())
        do("ideas", ideas_job)
    for key, lst in ((state.get("ideas") or {}).get("lists") or {}).items():
        trk = lst.get("track") or {}
        for it in lst.get("items", []):
            # a probability only from a track record of this exact rule: 30+ graded cases (10+ monthly cohorts
            # for the momentum screen, whose sample size is counted in months)
            enough = (trk.get("n") or 0) >= (10 if trk.get("unit") == "months" else 30)
            ledger.record(led, f"idea:{it['id']}:{it.get('first_seen')}", ticker=it["ticker"], made=today_et,
                          p=trk.get("p") if enough else None, source="idea", kind="idea", action=key)
    merge_ai(state, led, settings)
    do("claude_forecasts", lambda: (insight.merge_claude_forecasts(settings, state, settings.heavy_dir),
                                    "Claude's price forecasts written to the forecast ledger"))
    if due("forecast_grading", R.get("forecast_grading", 360)):
        do("forecast_grading", insight.job_grade_learn, settings, state, pan, store, settings.heavy_dir, budget)
    if (due("results", R.get("results", 60)) or not (settings.out_dir / "results.json").exists()) and budget.ok(30):
        do("results", insight.job_results, settings, state, pan, store, settings.heavy_dir, sctx)
    do("conviction", lambda: (lambda n: (n, f"{n} Top conviction"))(insight.apply_conviction(settings, state, sctx, led)))
    if due("ledger", R.get("ledger", 360)):
        do("ledger", lambda: (ledger.grade(led, panel, store), f"{len(led['entries'])} forecasts on record"))
    ledger.save(settings.cache_dir, led)
    state["accuracy"] = ledger.metrics(led)

    # ---- alerts
    picked = alerts.select(settings, state, fresh)
    if picked or state["alerts"].get("queue"):
        do("alerts", lambda: (lambda r: (r["sent"], f"queued {r['queued']}, via {', '.join(r['channels']) or 'none'}"))(
            alerts.deliver(settings, http, state, picked)))

    # ---- publish
    run_info = {"started": iso(started), "finished": iso(now_utc()), "new_events": len(new_ids),
                "results": results, "http": http.stats_summary()}
    state["runs"] = ([run_info] + state.get("runs", []))[:30]
    figi.save()
    do("stock_pages", lambda: (insight.publish_stocks(settings, state, sctx, pan, store), "one file per stock"))
    publish.write_all(settings, state, run_info)
    from .util import write_json as _wj
    do("accuracy_page", lambda: (_wj(settings.out_dir / "forecast_accuracy.json",
                                     insight.accuracy_doc(settings, state, state.get("accuracy"))) or 1,
                                 "forecast accuracy written"))
    try:
        selfcheck.run_and_publish(settings, state, run_info, http)
    except Exception as e:  # noqa: BLE001 – the audit must never break publishing
        log.error("self-check failed to run: %s", redact(str(e)))
    save_state(settings.cache_dir, state)
    log.info("run done in %.0fs: %d new events", (now_utc() - started).total_seconds(), len(new_ids))
    return run_info


TRACKED = {"buy", "sell", "new", "add", "trim", "exit", "stake", "open", "close", "sale_notice"}


def merge_ai(state: dict, led: dict, settings: Settings) -> None:
    """Attach the analyst workflow's latest output (restored read-only from its own cache)."""
    ai = read_json(settings.cache_dir / "ai" / "analyses.json.gz") or {}
    views = ai.get("trades") or {}
    keep = ("tier", "model", "effort", "made", "sources", "cost", "base_p")
    for e in state["events"]:
        v = views.get(e["id"])
        if v and v.get("view"):
            e["analysis"] = {**v["view"], **{k: v.get(k) for k in keep}}
        else:
            e.pop("analysis", None)
    idea_views = ai.get("ideas") or {}
    for lst in ((state.get("ideas") or {}).get("lists") or {}).values():
        for it in lst.get("items", []):
            v = idea_views.get(it["id"])
            if v and v.get("view"):
                it["analysis"] = {**v["view"], **{k: v.get(k) for k in keep}}
            else:
                it.pop("analysis", None)
    briefs = ai.get("briefs") or []
    state["brief"] = briefs[-1] if briefs else None
    for f in ai.get("forecasts") or []:
        ledger.record(led, f["fid"], ticker=f.get("ticker"), made=f.get("made"), p=f.get("p"), source="claude",
                      kind=f.get("kind"), action=f.get("action"), stance=f.get("stance"),
                      horizon=int(f.get("h") or quant.HOLD))
    spend = read_json(settings.cache_dir / "ai" / "spend.json") or {}
    runs = ai.get("runs") or []
    day = now_et().date().isoformat()
    state["analyst"] = {"last_run": runs[-1] if runs else None, "today": (spend.get("days") or {}).get(day),
                        "month_usd": round(sum(v.get("usd") or 0 for k, v in (spend.get("days") or {}).items()
                                               if k[:7] == day[:7]), 2),
                        "analyses": len(views), "briefs": len(briefs)}


def trim_events(state: dict, settings: Settings) -> None:
    max_n = settings.c("app", "max_events", default=2500)
    days = settings.c("app", "feed_days", default=120)
    cutoff = (now_utc() - dt.timedelta(days=days)).date().isoformat()
    ev = [e for e in state["events"] if (e.get("filed_at") or e.get("trade_date") or "9999")[:10] >= cutoff]

    def key(e):
        return sort_ts(e)
    ev.sort(key=key, reverse=True)
    if len(ev) > max_n:
        # keep every scored-strong/notable event, drop the oldest noise first
        keep = [e for e in ev if e.get("signal") not in ("noise", "weak")]
        rest = [e for e in ev if e.get("signal") in ("noise", "weak")]
        ev = (keep + rest)[:max_n]
        ev.sort(key=key, reverse=True)
    state["events"] = ev


def sort_ts(e: dict) -> float:
    """Comparable timestamp for an event: exact acceptance time when known, else 4pm ET on the filing date."""
    v = e.get("filed_at") or e.get("trade_date") or ""
    d = parse_iso(v) if len(v) > 10 else None
    if d is None:
        day = parse_date(v)
        if day is None:
            return 0.0
        d = dt.datetime(day.year, day.month, day.day, 16, 0, tzinfo=ET)
    return d.timestamp()


def split_factor_after(events: list | None, d: str | None, upto: str | None = None) -> float:
    """Shares multiply by this for splits executed after date `d` (a share count reported before a 10-for-1
    split is a tenth of today's), counting only splits on or before `upto`, the date of the price they're paired
    with (a price from before the split still goes with the old share count)."""
    f = 1.0
    for ed, k in events or []:
        e = str(ed)[:10]
        if d and e > str(d)[:10] and (not upto or e <= str(upto)[:10]) and k and k > 0:
            f *= float(k)
    return f


def update_quotes(state: dict, store: PriceStore, tickers: list[str], funda: dict, tick: dict,
                  splits: dict | None = None) -> None:
    q = state.setdefault("quotes", {})
    by_t = tick.get("by_ticker", {})
    for t in tickers + [BENCH]:
        m = risk.metrics(store, t)
        latest = store.latest.get(t) or {}
        price = latest.get("price") or (m or {}).get("last_close")
        cik = (by_t.get(t) or {}).get("cik")
        f = funda.get(cik) if cik else None
        mcap = fundamentals.market_cap(f, price)
        px_date = latest.get("asof") or ((m or {}).get("last_close_date") and
                                         dt.date.fromordinal(m["last_close_date"]).isoformat())
        if mcap and f and (splits or {}).get(t):
            mcap *= split_factor_after(splits[t], f.get("shares_end"), px_date)
        prof = fundamentals.is_profitable(f)
        rl = risk.risk_level(m, mcap, prof, is_fund=not bool(cik))
        if not m and not price:
            continue
        q[t] = {
            "price": price, "asof": latest.get("asof") or ((m or {}).get("last_close_date") and
                                                             dt.date.fromordinal(m["last_close_date"]).isoformat()),
            "price_src": latest.get("src") or (store.series(t) or {}).get("src"),
            "name": (by_t.get(t) or {}).get("name"), "exchange": (by_t.get(t) or {}).get("exchange"),
            "mcap": mcap, "cap_tier": risk.cap_tier(mcap), "profitable": prof,
            "vol_1y": (m or {}).get("vol_1y"), "mdd_1y": (m or {}).get("mdd_1y"), "beta": (m or {}).get("beta"),
            "lo_52w": (m or {}).get("lo_52w"), "hi_52w": (m or {}).get("hi_52w"),
            "ret_1m": (m or {}).get("ret_1m"), "ret_3m": (m or {}).get("ret_3m"), "ret_1y": (m or {}).get("ret_1y"),
            "adv_usd": (m or {}).get("adv_usd"),
            "risk_level": rl.get("level"), "risk_label": rl.get("label"), "risk_factors": rl.get("factors"),
            "updated": iso(now_utc()),
        }


def build_ctx(state: dict, settings: Settings, months: dict, directory: dict | None = None) -> Ctx:
    quotes = state.get("quotes", {})
    events = state["events"]
    by_issuer: dict[str, list] = {}
    fund_new: dict[tuple, set] = {}
    etf_same_day: dict[tuple, set] = {}
    for e in events:
        if e["kind"] == "insider" and e.get("action") in ("buy", "sell", "sale_notice"):
            by_issuer.setdefault(e.get("issuer_cik") or e.get("ticker") or "", []).append(e)
        if e["kind"] == "fund" and e.get("action") in ("new", "add") and e.get("ticker"):
            fund_new.setdefault((e["ticker"], e["details"].get("period")), set()).add(e["actor"]["id"])
        if e["kind"] == "etf" and e.get("ticker"):
            side = "buy" if e.get("action") in ("new", "add") else "sell"
            key = (e["ticker"], e["details"].get("file_date"), side, (e.get("actor") or {}).get("role"))   # same family
            etf_same_day.setdefault(key, set()).add(e["actor"]["id"])
    tr = state.get("track", {})
    ins = (tr.get("insiders") or {}).get("records", {})
    cong = (tr.get("congress") or {}).get("records", {})
    funds_tr = (tr.get("funds") or {}).get("records", {})
    funds_cfg = {f["cik"]: f for f in settings.c("funds", "list", default=[])}
    leaders = dict((directory or {}).get("leaders") or {})
    for bio in settings.c("congress", "leadership_extra", default=[]) or []:   # manual additions (bioguide IDs)
        leaders.setdefault(str(bio).upper(), "Congressional leadership")

    def cluster(e):
        d0 = parse_date(e.get("trade_date"))
        out = {"buyers": 0, "sellers": 0, "names": []}
        if not d0:
            return out
        buyers, sellers = set(), set()
        for o in by_issuer.get(e.get("issuer_cik") or e.get("ticker") or "", []):
            d1 = parse_date(o.get("trade_date"))
            if d1 and abs((d1 - d0).days) <= 30:
                (buyers if o["action"] == "buy" else sellers).add(o["actor"]["id"])   # a sale notice counts as a seller
        out.update(buyers=len(buyers), sellers=len(sellers))
        return out

    def fund_meta(cik):
        f = funds_cfg.get(cik)
        if not f:
            return None
        return {**f, "clone": funds_tr.get(f"cik:{cik}")}

    def leader(actor: dict) -> str | None:
        return actor.get("leadership") or leaders.get((actor.get("bioguide") or "").upper())

    return Ctx(
        quote=lambda t: quotes.get(t) if t else None,
        insider_cluster=cluster,
        insider_record=lambda aid: ins.get(aid),
        insider_routine=lambda e: track_records.is_routine(months, e["actor"].get("id", ""), e.get("trade_date")),
        fund_meta=fund_meta,
        fund_consensus=lambda e: len(fund_new.get((e.get("ticker"), e["details"].get("period")), set())),
        etf_consensus=lambda e: max(0, len(etf_same_day.get((e.get("ticker"), e["details"].get("file_date"),
                                                              "buy" if e.get("action") in ("new", "add") else "sell",
                                                              (e.get("actor") or {}).get("role")), set())) - 1),
        member_record=lambda aid: cong.get(aid),
        is_leader=leader,
    )


def actor_records(state: dict):
    tr = state.get("track", {})
    ins = (tr.get("insiders") or {}).get("records", {})
    cong = (tr.get("congress") or {}).get("records", {})
    funds = (tr.get("funds") or {}).get("records", {})
    etoro = {f"etoro:{L.get('username')}": L for L in (state.get("etoro") or {}).get("leaders", [])}

    def pick(r):
        return {k: r.get(k) for k in ("n", "win_rate", "beat_rate", "avg_ret", "avg_excess")} if r and r.get("n") else None

    def get(e):
        aid = e.get("actor", {}).get("id", "")
        if e["kind"] == "insider":
            return pick(ins.get(aid))
        if e["kind"] == "congress":
            return pick(cong.get(aid))
        if e["kind"] == "fund":
            f = funds.get(aid)
            legs = [q for q in (f or {}).get("legs", []) if not q.get("open")]
            if not legs:
                return None
            return {"n": len(legs), "win_rate": sum(q["ret"] > 0 for q in legs) / len(legs),
                    "beat_rate": f.get("beat_rate"), "avg_excess": sum(q["ret"] - q["spy"] for q in legs) / len(legs)}
        if e["kind"] == "etoro":
            L = etoro.get(aid)
            if L and L.get("trades"):
                return {"n": L.get("trades"), "win_rate": (L.get("win_ratio") or 0) / 100, "beat_rate": None,
                        "avg_excess": None}
        return None
    return get


def cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", default="", help="comma-separated jobs to run now (or 'all')")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    setup_logging(a.verbose)
    run(force={x.strip() for x in a.force.split(",") if x.strip()})


if __name__ == "__main__":
    cli()
