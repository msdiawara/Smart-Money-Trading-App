"""Glue for the per-stock research layer: company data, news, valuation & bubble risk, price forecasts, the
forecast ledger and learning loop, the conviction rating, and the per-stock files the app opens.

Called from main.run() in this order: priority list → company data / live 8-Ks / headlines (time-boxed jobs) →
context (valuation, bubble, AI exposure, news views) → forecasts (+ ledger records) → Claude forecasts merged →
grading & learning → conviction → publish.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import analyst as A
from . import conviction, fledger, forecast, learn, ml, news, valuation
from .prices import BENCH, _plausible, load_splits
from .util import iso, log, now_et, now_utc, read_json, write_json

FC_CACHE = "forecasts.json.gz"
CTX_CACHE = "stock_ctx.json.gz"
FEEDBACK = "claude_feedback.json.gz"
HMAP = dict(forecast.HORIZONS)


# ---------------------------------------------------------------- who matters most

def ai_cfg(settings) -> dict:
    b = settings.c("bubble", default={}) or {}
    return {k: [str(x).upper() for x in (b.get(k) or [])] for k in ("core", "platform", "power")}


def priority_tickers(state: dict, settings, ai: dict | None = None) -> list[str]:
    ai = ai or ai_cfg(settings)
    watch = [str(t).upper() for t in settings.c("alerts", "watch_tickers", default=[]) or []]
    ideas = [it.get("ticker") for lst in ((state.get("ideas") or {}).get("lists") or {}).values() for it in lst.get("items", [])]
    claude = [e.get("ticker") for e in state.get("events", []) if e.get("analysis")]
    feed = [e.get("ticker") for e in sorted(state.get("events", []), key=lambda e: (-(e.get("score") or 0), e.get("filed_at") or ""))]
    basket = ai.get("core", []) + ai.get("platform", []) + ai.get("power", [])
    out = []
    for t in watch + ideas + claude + basket + feed + sorted((state.get("quotes") or {}).keys()):
        if t and t not in out and _plausible(t):
            out.append(t)
    return out


# ---------------------------------------------------------------- jobs that fetch

def job_companies(http, settings, budget, tick: dict, prio: list[str]) -> tuple[int, str]:
    cfg = settings.c("valuation", default={}) or {}
    secs = max(20.0, min(float(cfg.get("seconds_per_run", 90)), budget.left() - 240))
    return valuation.refresh(http, settings.cache_dir, prio, tick, budget_seconds=secs,
                             max_companies=int(cfg.get("per_run", 150)))


def job_filings_live(http, settings, tick: dict, prio: list[str]) -> tuple[int, str]:
    by_t = tick.get("by_ticker") or {}
    ciks = {str(by_t[t]["cik"]).zfill(10) for t in prio if t in by_t and by_t[t].get("cik")}
    new, scanned = news.poll_live_8k(http, settings.cache_dir, ciks)
    if new:
        cache = valuation.load_cache(settings.cache_dir)
        for k in new:
            valuation.mark_stale(cache, k["cik"], "sub")
        valuation.save_cache(settings.cache_dir, cache)
    return len(new), f"{scanned} recent 8-Ks scanned; {len(new)} from companies in the app"


def job_headlines(http, settings, budget, state: dict, prio: list[str]) -> tuple[int, str]:
    cfg = settings.c("news", default={}) or {}
    if not cfg.get("gdelt", True):
        return 0, "GDELT headlines switched off in config"
    quotes = state.get("quotes") or {}
    cache = valuation.load_cache(settings.cache_dir)["by_cik"]
    names = {r.get("t"): (r.get("sub") or {}).get("name") for r in cache.values() if r.get("t")}
    for t, q in quotes.items():
        if q.get("name"):
            names.setdefault(t, q["name"])
    secs = max(15.0, min(float(cfg.get("seconds_per_run", 70)), budget.left() - 240))
    return news.refresh_headlines(http, settings.cache_dir, prio[: int(cfg.get("max_tickers", 300))], names,
                                  budget_seconds=secs, per_run=int(cfg.get("per_run", 12)),
                                  max_age_hours=float(cfg.get("max_age_hours", 12)))


# ---------------------------------------------------------------- per-stock context

def _aligned(panel, store, t: str):
    """(closes, spy closes, volumes) as aligned numpy arrays, oldest first."""
    if panel is not None:
        c = panel.c(t)
        if c is not None:
            return c, panel.c(BENCH), panel.v(t)
    s, b = store.series(t), store.series(BENCH)
    if not s or not b:
        return None, None, None
    bm = dict(zip(b["dates"], b["c"]))
    keep = [i for i, d in enumerate(s["dates"]) if d in bm]
    c = np.asarray([s["c"][i] for i in keep], dtype=float)
    spy = np.asarray([bm[s["dates"][i]] for i in keep], dtype=float)
    v = np.asarray([s["v"][i] for i in keep], dtype=float)
    return c, spy, v


def build_context(settings, state: dict, panel, store, tick: dict, prio: list[str]) -> dict:
    """Valuation, bubble risk, AI exposure and news views for every stock with a quote."""
    quotes = state.get("quotes") or {}
    cache = valuation.load_cache(settings.cache_dir)["by_cik"]
    by_t = tick.get("by_ticker") or {}
    by_cik_t = {}
    for rec in cache.values():
        by_cik_t[rec.get("cik")] = rec
    # valued at the last close (not a live intraday price): the page says so, and the learning features recorded
    # with each forecast then use only what was known at the forecast's anchor close
    closes = {t: _last_close(panel, store, t) for t in quotes}
    priced = {t: {**q, "price": closes[t][1], "asof": closes[t][0]} if closes.get(t) else q for t, q in quotes.items()}
    vals = valuation.value_all(cache, priced, settings.c("valuation", default={}) or {},
                               splits=load_splits(settings.cache_dir))
    V = vals["by_ticker"]
    for t, v in V.items():
        if closes.get(t) and v.get("label") is not None:
            v["price_date"] = closes[t][0]
    anchor_cut = _anchor_cut()
    aic = ai_cfg(settings)
    basket = [t for t in aic["core"] + aic["platform"] if t != BENCH]
    theme = valuation.ai_theme(panel, basket, V) if panel is not None else None
    universe = [t for t in quotes if t != BENCH]
    loads = valuation.ai_factor_loadings(panel, basket, universe) if panel is not None else {}
    nc = news.load(settings.cache_dir)
    ai_store = read_json(settings.cache_dir / "ai" / "analyses.json.gz") or {}
    cfc = ai_store.get("price_forecasts") or {}
    events = state.get("events", [])
    ctx = {}
    for t in universe:
        info = by_t.get(t) or {}
        cik = str(info.get("cik") or "").zfill(10) if info.get("cik") else None
        rec = by_cik_t.get(cik) if cik else None
        sub = (rec or {}).get("sub") or {}
        v = V.get(t)
        c, spy, vol = _aligned(panel, store, t)
        stats = valuation.price_path_stats(c, spy, vol) if c is not None else None
        claude_ai = (cfc.get(t) or {}).get("ai_exposure")
        ai = valuation.ai_exposure(t, sub.get("sic"), aic, loads.get(t), claude_ai)
        ins = valuation.insider_flow(events, t)
        bub = valuation.bubble_risk(stats, v, sub, ins, ai, (theme or {}).get("level"))
        ins_cut = valuation.insider_flow(events, t, upto=anchor_cut)
        bub_feat = bub if ins_cut == ins else valuation.bubble_risk(stats, v, sub, ins_cut, ai, (theme or {}).get("level"))
        fl = news.filings_for(rec, nc["live_8k"], cik) if cik else []
        flags = news.red_flags(fl)
        hl = ((nc["by_ticker"].get(t) or {}).get("items") or [])[:8]
        ctx[t] = {"name": quotes[t].get("name") or sub.get("name"), "val": v, "bubble": bub, "ai": ai,
                  "filings": fl, "flags": flags, "headlines": hl, "industry": sub.get("sic_desc"),
                  "first_filed": sub.get("first_filed"), "headlines_fetched": (nc["by_ticker"].get(t) or {}).get("fetched"),
                  "past_earnings": news.earnings_dates(sub.get("k8") or []),
                  "feat": {"val": (v or {}).get("rich_pctl"), "val_date": (v or {}).get("price_date"),
                           "val_filed": ((sub.get("periodic") or {}).get("filed") or None),
                           "bub": bub_feat.get("score") if (stats or v) else None},
                  "earnings_when": news.earnings_when(sub.get("k8") or [])}
        q = quotes[t]
        q["val"] = {"label": v.get("label"), "pctl": v.get("rich_pctl")} if v and v.get("label") else None
        q["bubble"] = bub["level"] if stats or v else None
        q["ai"] = ai["level"] if ai["level"] != "none" else None
        q["flag"] = flags[0]["label"] if flags else None
        if v and v.get("mcap"):                 # latest filing, split-adjusted: better than the frames estimate
            from .risk import cap_tier
            q["mcap"] = v["mcap"]
            q["cap_tier"] = cap_tier(v["mcap"])
    state.setdefault("market", {})
    if theme:
        state["market"]["ai_bubble"] = {**theme, "about": AI_ABOUT}
        st = theme.get("stats") or {}
        state["market"]["ai_bubble_summary"] = {
            "level": theme["level"], "phase": theme["phase"], "members": theme["n"],
            "net_2y_vs_sp500": st.get("net_2y"), "net_1y_vs_sp500": st.get("net_1y"), "from_high": st.get("from_high"),
            "breadth_above_200d": theme.get("breadth_above_200d"), "very_expensive_share": theme.get("very_expensive_share"),
            "historical_crash_rate_2y": theme.get("gsy_crash_2y"),
            "big_down_days": theme.get("big_down_days")[:5]}
    state["valuation_meta"] = vals["meta"]
    _write_ctx(settings, ctx)
    return ctx


AI_ABOUT = ("An equal-weight basket of the AI trade (chips, AI infrastructure and the big AI platforms; edit the list in "
            "config.yaml under bubble) measured with the warning signs from Greenwood, Shleifer & You's 'Bubbles for "
            "Fama' (2019): the 2-year run-up versus the S&P 500, acceleration, volatility, and valuations. In their US "
            "industry data, run-ups of 100%+ net of the market were followed by a 40%+ crash within two years about half "
            "the time, yet run-ups alone didn't predict lower average returns and peaks were hard to time. A gauge, not a "
            "forecast.")


def _last_close(panel, store, t: str) -> tuple[str, float] | None:
    """(date, close) of the latest stored close: the same series the forecasts are built from."""
    if panel is not None:
        c = panel.c(t)
        if c is not None:
            ok = np.flatnonzero(np.isfinite(c) & (c > 0))
            if len(ok):
                return panel.date(int(ok[-1])), float(c[ok[-1]])
    s = store.series(t) if store is not None else None
    if s and s.get("dates"):
        return dt.date.fromordinal(int(s["dates"][-1])).isoformat(), float(s["c"][-1])
    return None


def _anchor_cut() -> str:
    """The forecasts' anchor close as a UTC timestamp: anything filed after it isn't known at the anchor."""
    from .util import ET
    d = forecast.last_complete_session(now_et())
    h, m = forecast.close_hm(d)
    return iso(dt.datetime(d.year, d.month, d.day, h, m, tzinfo=ET).astimezone(dt.timezone.utc))


def _write_ctx(settings, ctx: dict) -> None:
    """Compact per-stock context for the analyst workflow (it reads the small cache, read-only)."""
    def slim_val(v):
        if not v:
            return None
        return {k: (round(v[k], 4) if isinstance(v.get(k), float) else v.get(k))
                for k in ("label", "rich_pctl", "basis", "pe", "ps", "ev_s", "fair_ev_s", "premium", "growth", "op_margin",
                          "gross_margin", "fcf_yield", "dilution", "group", "mcap") if v.get(k) is not None} | (
            {"implied_10y_fcf_growth": v["dcf"].get("growth")} if v.get("dcf") else {})
    out = {}
    for t, x in ctx.items():
        b = x.get("bubble") or {}
        out[t] = {"name": x.get("name"), "val": slim_val(x.get("val")),
                  "bubble": {"level": b.get("level"), "signs": [s["title"] + ": " + s["detail"] for s in b.get("signs", [])][:8]},
                  "ai": {"level": (x.get("ai") or {}).get("level"), "label": (x.get("ai") or {}).get("label")},
                  "filings": [{"date": f.get("date"), "title": f.get("title"), "items": [i["item"] for i in f.get("items", [])]}
                              for f in (x.get("filings") or [])[:6]],
                  "headlines": [{"date": (h.get("seen") or "")[:10], "source": h.get("domain"), "title": h.get("title")}
                                for h in (x.get("headlines") or [])[:6]],
                  "conv": {"rating": (x.get("conv") or {}).get("rating")}}
    write_json(settings.cache_dir / CTX_CACHE, {"generated_at": iso(now_utc()), "by_ticker": out})


# ---------------------------------------------------------------- forecasts

def _calib_version(calib: dict | None, tilt: dict | None) -> str:
    raw = json.dumps({"c": {k: [v.get("scale"), v.get("shift")] for k, v in (calib or {}).items()},
                      "t": (tilt or {}).get("weights") if (tilt or {}).get("accepted") else None}, sort_keys=True)
    return hashlib.sha1(raw.encode()).hexdigest()[:10]


def _bench_on(dates, bmap: dict) -> list[float] | None:
    """The benchmark's close on each of a stock's dates (the latest one on or before it), for relative features."""
    if not bmap or dates is None or not len(dates):
        return None
    keys = sorted(bmap)
    import bisect
    out = []
    for d in dates:
        k = bisect.bisect_right(keys, int(d)) - 1
        if k < 0 or int(d) - keys[k] > 4:
            return None
        out.append(bmap[keys[k]])
    return out


def _series_v(store, t: str, upto: dt.date):
    """(dates, closes, volumes) up to a day; volumes None when the store has none for every day."""
    s = store.series(t)
    if not s or not s.get("dates"):
        return None, None, None
    o = upto.toordinal()
    k = len(s["dates"])
    while k > 0 and s["dates"][k - 1] > o:
        k -= 1
    v = s.get("v")
    return s["dates"][:k], s["c"][:k], (v[:k] if v and len(v) == len(s["dates"]) else None)


def _series(store, t: str, upto: dt.date):
    s = store.series(t)
    if not s or not s.get("dates"):
        return None, None
    o = upto.toordinal()
    k = len(s["dates"])
    while k > 0 and s["dates"][k - 1] > o:
        k -= 1
    return s["dates"][:k], s["c"][:k]


def _claude_stance_asof(cf: dict | None, hist: list | None, cut: str) -> str | None:
    """Claude's stance as known at the anchor close (a newer one would leak what happened after it)."""
    if cf and (cf.get("made_at") or "") and cf["made_at"] <= cut:
        return cf.get("stance")
    for h in reversed(hist or []):
        when = h.get("made_at") or ((h.get("made") or "") + "T00:00:00Z")
        if when <= cut:
            return h.get("stance")
    return None


def job_forecasts(settings, state: dict, store, budget, ctx: dict, prio: list[str], heavy_dir: Path) -> tuple[int, str]:
    import time
    cfg = settings.c("forecast", default={}) or {}
    if not cfg.get("enabled", True):
        return 0, "forecasts switched off in config"
    t0 = time.monotonic()
    secs = max(20.0, min(float(cfg.get("seconds_per_run", 150)), budget.left() - 200))
    fcache = (read_json(settings.cache_dir / FC_CACHE) or {}).get("by_ticker") or {}
    L = fledger.load(heavy_dir)
    lr = state.get("learn") or {}
    calib = {k: v for k, v in (lr.get("calib") or {}).items() if v.get("status") == "applied"}
    tilt = learn.tilt_in_force(lr, now_utc())
    cv = _calib_version(calib, tilt) + ml.version(tilt, lr.get("range"))
    models: dict = {}
    failed: list[str] = []
    memo: dict = {}

    def learners():
        """The learners that have passed their tests, refit from the record the first time a stock needs them. Each
        is built on its own: one that can't be rebuilt is left out this run (the forecasts made without it are
        marked, so the next run rebuilds them) and the other still applies."""
        if not models:
            models["s"] = models["r"] = None
            try:
                models["s"] = ml.SignalModel.build(tilt, L)
                if models["s"] is not None:
                    # the nudge says how a stock should do relative to the others, as tested: centred on the
                    # average over the stocks the record follows (the same kind of group it was trained on)
                    day = [f for f in (feats_for(t) for t in ledger_set if t != BENCH) if f]
                    models["s"].set_day([f[-1] for f in day])
                elif (tilt or {}).get("on") and (tilt or {}).get("accepted") and tilt.get("fv") == ml.FEAT_V \
                        and tilt.get("model") in ml.CANDIDATES:
                    failed.append("signal")             # on, but its record isn't here (a lost ledger cache, say)
            except Exception as err:  # noqa: BLE001
                log.warning("signal learner: %s", err)
                models["s"] = None
                failed.append("signal")
            try:
                models["r"] = ml.RangeModel.build(lr.get("range"), lr.get("calib") or {}, L)
                if models["r"] is None and any(v.get("status") == "applied" for v in (lr.get("range") or {}).values()):
                    failed.append("range")
            except Exception as err:  # noqa: BLE001
                log.warning("range learner: %s", err)
                failed.append("range")
        return models["s"], models["r"]
    fred = state.get("fred") or {}
    rf = (fred.get("DFF") or {}).get("value")
    rf = float(rf) / 100 if isinstance(rf, (int, float)) and 0 <= rf < 25 else float(cfg.get("risk_free", 0.04))
    erp = float(cfg.get("equity_premium", 0.05))
    macro = forecast.macro_events(cfg.get("macro_events"))
    anchor_day = forecast.last_complete_session(now_et())
    cut = _anchor_cut()
    lo_cut = iso(dt.datetime.fromisoformat(cut.replace("Z", "+00:00")) - dt.timedelta(days=30))
    quotes = state.get("quotes") or {}
    ledger_set = set(prio[: int(cfg.get("ledger_max", 250))]) | {BENCH}
    paths = int(cfg.get("paths", 2000))
    ai_store = read_json(settings.cache_dir / "ai" / "analyses.json.gz") or {}
    cfc = ai_store.get("price_forecasts") or {}
    fch = ai_store.get("fc_history") or {}
    events_by_t: dict[str, list] = {}
    for e in state.get("events", []):
        fa = _utc_iso(e.get("filed_at"))
        if e.get("ticker") and fa and lo_cut <= fa <= cut:      # known at the anchor close, and recent
            events_by_t.setdefault(e["ticker"], []).append(e)
    order = [t for t in prio if t in quotes or t == BENCH] + [t for t in quotes if t not in set(prio)]
    b_dates, b_closes = _series(store, BENCH, anchor_day)
    bmap = dict(zip([int(d) for d in b_dates], [float(x) for x in b_closes])) if b_dates else {}

    def feats_for(t: str):
        """(dates, closes, last close's day, features) for a stock with fresh enough history, else None."""
        if t not in memo:
            memo[t] = None
            dates, closes, vols = _series_v(store, t, anchor_day)
            if dates and len(dates) >= 130:
                last = dt.date.fromordinal(int(dates[-1]))
                if (anchor_day - last).days <= 4:
                    x = ctx.get(t) or {}
                    fx = x.get("feat") or {}
                    val_ok = fx.get("val") is not None and fx.get("val_date") == last.isoformat() and \
                        not (fx.get("val_filed") and fx["val_filed"] >= last.isoformat())
                    val_feat = {"rich_pctl": fx["val"]} if val_ok else None
                    bub_feat = {"score": fx["bub"]} if fx.get("bub") is not None else None
                    stance = _claude_stance_asof(cfc.get(t), fch.get(t), cut)
                    evs0 = [{**e, "score": e.get("score0", e.get("score"))} for e in events_by_t.get(t, [])]
                    memo[t] = (dates, closes, last, learn.features(evs0, quotes.get(t) or {}, val_feat, bub_feat,
                                                                   stance, closes, _bench_on(dates, bmap), vols))
        return memo[t]
    built = recorded = skipped_stale = held = degraded = 0
    held_list: list[str] = []
    seen: set[str] = set()
    splits = load_splits(settings.cache_dir)
    # the S&P 500 first (its record is the market's surprise the learners measure against); forecasts made earlier on
    # this close without a learner that's on come last, and are redone only once the learners can be built again
    todo = list(dict.fromkeys([BENCH] + order))
    redo = [t for t in todo if str((fcache.get(t) or {}).get("cv", "")).endswith("!")]
    todo = [t for t in todo if t not in set(redo)] + redo
    for t in todo:
        if time.monotonic() - t0 > secs:
            break
        seen.add(t)
        dates, closes, _ = _series_v(store, t, anchor_day)
        if not dates or len(dates) < 130:
            continue
        last = dt.date.fromordinal(int(dates[-1]))
        if (anchor_day - last).days > 4:
            skipped_stale += 1
            fcache.pop(t, None)
            continue
        old = fcache.get(t)
        if old and old.get("made") == last.isoformat() and old.get("cv") == cv:
            continue
        if old and old.get("made") == last.isoformat() and old.get("cv") == cv + "!":
            learners()
            if failed:
                continue                      # the learner still can't be built: keep this one until it can
        tail = np.asarray(closes[-6:], dtype=float)
        tail_days = [dt.date.fromordinal(int(d)) for d in dates[-len(tail):]]
        if len(tail) > 1 and np.all(tail > 0) and any(
                forecast.split_like(float(x)) or forecast.split_near(splits.get(t), tail_days[i + 1], float(x))
                for i, x in enumerate(np.diff(np.log(tail)))):
            held += 1                         # looks like a split the history hasn't been re-based for yet:
            fcache.pop(t, None)               # take the old forecast down rather than show it beside a new price
            held_list.append(t)
            continue
        x = ctx.get(t) or {}
        q = quotes.get(t) or {}
        got = feats_for(t)
        if not got:
            continue
        feats = got[-1]

        def build(calib, tilt_ret, tilt_days, tilt_parts, t=t, dates=dates, closes=closes, q=q, x=x):
            return forecast.build(t, dates, closes, beta=q.get("beta"), rf=rf, erp=erp, earnings_next=q.get("earnings"),
                                  past_earnings=x.get("past_earnings"), macro=macro, calib=calib, tilt_ret=tilt_ret,
                                  tilt_days=tilt_days, tilt_parts=tilt_parts,
                                  div_yield=(x.get("val") or {}).get("div_yield"), paths=paths)
        plain = False
        try:
            sm, rm = learners()
            # the S&P 500 gets no stock-versus-stock nudge (it isn't in what the learner was trained or centred on)
            fc, _ = ml.forecast_with_learning(build, feats, calib, None if t == BENCH else sm, rm)
        except Exception as err:  # noqa: BLE001 – a learner failing on one stock: forecast it without the learners
            log.warning("forecast %s with the learners: %s", t, err)
            plain = True
            try:
                fc = build(calib=calib, tilt_ret=0.0, tilt_days=21, tilt_parts=[])
            except Exception as err2:  # noqa: BLE001 – one stock's bad data must not stop the rest
                log.warning("forecast %s: %s", t, err2)
                continue
        if fc is None:
            continue
        degraded += bool(plain or failed)
        fc["cv"] = cv + ("!" if plain or failed else "")      # made without a learner that's on: rebuilt next run
        fc["next_earnings"] = next((e.get("announce") for e in fc["events"] if e["kind"] == "earnings"), None)
        fcache[t] = fc
        built += 1
        if t in ledger_set and last == anchor_day:
            for name, h in forecast.HORIZONS:
                hz = fc["horizons"][name]
                extra = {"tl": hz["tilt"]} if hz.get("tilt") else {}
                if h == 63:
                    extra["pd"] = hz.get("p_dd20")
                recorded += fledger.record(L, "quant", t, fc["made"], h, hz["lq"], lq_raw=hz["lq_raw"], pup=hz["p_up"],
                                           naive=forecast.naive_lq(fc["naive_sigma_d"], h),
                                           feats=ml.row_features(feats, fc, name),
                                           bubble=(x.get("bubble") or {}).get("level"), extra=extra or None)
    keep = set(quotes) | {BENCH}
    for t in [t for t in fcache if t not in keep]:
        fcache.pop(t, None)
    write_json(settings.cache_dir / FC_CACHE, {"generated_at": iso(now_utc()), "by_ticker": fcache})
    # paused stocks keep their note until the loop gets back to them (a time-limited run may not reach them all)
    held_now = {t: anchor_day.isoformat() for t in held_list}
    for t, d in (state.get("fc_held") or {}).items():
        if t not in seen and t not in held_now and str(d) >= (anchor_day - dt.timedelta(days=10)).isoformat():
            held_now[t] = d
    state["fc_held"] = held_now
    if recorded:
        fledger.save(heavy_dir, L)
        settings.mark_heavy_dirty()
    note = (f"; {degraded} made without the {' and '.join(failed) or 'learners'} (couldn't be rebuilt; retried next run)"
            if degraded else "")
    return built, (f"{len(fcache)} stocks forecast ({built} updated){note}; {recorded} forecasts written to the ledger"
                   + (f"; {skipped_stale} skipped (stale prices)" if skipped_stale else "")
                   + (f"; {held} held for a split re-base" if held else ""))


def _utc_iso(s: str | None) -> str | None:
    """A filed_at value as a comparable UTC ISO string (the same reading as everywhere else: naive = ET)."""
    from .util import parse_iso
    d = parse_iso(s) if s else None
    return iso(d) if d else None


_PLAIN = __import__("re").compile(r"^[A-Z]{1,5}(\.[A-Z])?$")


def earnings_proxy(c: np.ndarray, v: np.ndarray | None, dates: np.ndarray) -> list[str]:
    """Likely past earnings days from the price history alone (for stocks with no filing record, delisted ones
    included): a volume spike of 3× the prior month's median with a move twice the usual size, at most one per
    ~40 sessions. Only used by the backtest, so it models earnings risk like the live forecasts do."""
    if v is None:
        return []
    ok = np.isfinite(c) & (c > 0) & np.isfinite(v)
    idx = np.flatnonzero(ok)
    if len(idx) < 80:
        return []
    cc, vv = c[idx], v[idx]
    r = np.abs(np.diff(np.log(cc)))
    cands = []
    for k in range(60, len(r)):
        mv, mr = float(np.median(vv[k - 20:k + 1])), float(np.median(r[k - 60:k]))
        if mv > 0 and mr > 0 and vv[k + 1] >= 3 * mv and r[k] >= 2 * mr:
            cands.append((vv[k + 1] / mv, k + 1))
    picked: list[int] = []
    for _, k in sorted(cands, reverse=True):
        if all(abs(k - j) >= 40 for j in picked):
            picked.append(k)
    return sorted(dt.date.fromordinal(int(dates[idx[k]])).isoformat() for k in picked)


def _beta_at(c: np.ndarray, spy: np.ndarray, i: int, n: int = 252) -> float | None:
    a, b = c[max(0, i - n):i + 1], spy[max(0, i - n):i + 1]
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    if ok.sum() < 120:
        return None
    ra, rb = np.diff(np.log(a[ok])), np.diff(np.log(b[ok]))
    vb = float(np.var(rb))
    return float(np.cov(ra, rb)[0, 1] / vb) if vb > 0 else None


def job_backtest(settings, state: dict, panel, budget, heavy_dir: Path, allowed: set | None = None,
                 ctx: dict | None = None) -> tuple[int, str]:
    """Walk-forward backtest of the forecast model: once a week over the stored history, a sample of
    the stocks that were liquid *on that day* (delisted ones included, so it isn't flattered by survivors), each
    forecast with only the data available then (beta, earnings dates, Fed and CPI days as the live model sees
    them) and graded against what happened. Resumes across runs (time-boxed); live grades replace it."""
    import time
    from . import quant
    cfg = settings.c("forecast", default={}) or {}
    if panel is None or panel.T < 330 or not cfg.get("backtest", True):
        return 0, "needs ~15 months of whole-market price history (free Massive key)"
    t0 = time.monotonic()
    secs = max(15.0, min(float(cfg.get("backtest_seconds_per_run", 75)), budget.left() - 220))
    L = fledger.load(heavy_dir)
    bt = L.setdefault("bt", {})
    today = now_et().date()
    per_anchor = int(cfg.get("backtest_stocks", 150))
    if bt.get("built") != today.isoformat()[:7] or bt.get("v") != 3:      # the grid is extended once a month
        last = panel.T - 1
        # one anchor a week (its last session), fixed to the calendar: the same dates every month, so stock-dates
        # already graded are skipped and the record grows by about four new weeks a month (years of months, which
        # is what the signal learner's test needs)
        this_week = dt.date.fromordinal(int(panel.dates[last])).isocalendar()[:2]
        wk: dict = {}
        for i in range(260, last):
            w = dt.date.fromordinal(int(panel.dates[i])).isocalendar()[:2]
            if w != this_week:
                wk[w] = i
        anchors = sorted(wk.values(), reverse=True)[: int(cfg.get("backtest_anchors", 120))]
        seen = L.get("bt_seen") or {}
        # queued by date: days added to the price history between runs shift the row positions
        live_end = {}
        for t, j in panel.col.items():
            col = panel.close[:, j]
            nz = np.flatnonzero(np.isfinite(col) & (col > 0))
            live_end[t] = int(nz[-1]) if len(nz) else -1
        todo = []
        for i in anchors:
            day = dt.date.fromordinal(int(panel.dates[i])).isoformat()
            fits = [h for _, h in forecast.HORIZONS[:4] if i + h < panel.T]
            names = [t for t in quant.liquid_universe(panel, i, None)
                     if _PLAIN.match(t) and t != BENCH
                     and ((allowed is None or t in allowed) or live_end.get(t, -1) < last - 20)]
            names = sorted(names, key=lambda t: hashlib.sha1(f"{t}:{day}".encode()).hexdigest())[:per_anchor]
            todo += [[t, int(panel.dates[i])] for t in names + [BENCH]
                     if not all(f"quant_bt:{t}:{h}:{day}" in seen for h in fits)]
        bt.update(v=3, built=today.isoformat()[:7], anchors=[int(panel.dates[i]) for i in anchors], todo=todo,
                  n_names=len(todo))
        cut = (today - dt.timedelta(days=1600)).isoformat()
        L["bt_seen"] = {k: 1 for k in (L.get("bt_seen") or {}) if k.rsplit(":", 1)[-1] >= cut}
    macro = forecast.macro_events(cfg.get("macro_events"))
    paths = int(cfg.get("backtest_paths", 800))
    spy = panel.c(BENCH)
    ctx = ctx or {}
    proxies: dict[str, list[str]] = {}
    done = added = 0
    filled = _fill_bt_features(L, panel, spy)
    todo = bt.get("todo") or []
    pos = {int(d): k for k, d in enumerate(panel.dates)}
    while todo and time.monotonic() - t0 < secs:
        t, day_ord = todo.pop()
        i = pos.get(int(day_ord))
        c = panel.c(t)
        if c is None or i is None:
            continue
        hist = c[: i + 1]
        ok = np.isfinite(hist) & (hist > 0)
        if ok.sum() < 260 or not ok[-1]:
            continue
        dates = [int(d) for d in panel.dates[: i + 1][ok]]
        anchor = dt.date.fromordinal(int(panel.dates[i]))
        bfeat = _bt_features(hist, spy, i, ok, panel.v(t)) if t != BENCH else None
        known = (ctx.get(t) or {}).get("past_earnings")
        when = (ctx.get(t) or {}).get("earnings_when") or {}
        nxt = None
        if known:                             # filing dates: the next one is usually announced weeks ahead
            # (two weeks ahead: later ones may not have been announced yet; the model estimates those, as live)
            upcoming = [d for d in known if anchor.isoformat() < d <= (anchor + dt.timedelta(days=14)).isoformat()]
            if upcoming:
                nxt = {"date": upcoming[0], "time": when.get(upcoming[0], "post")}
        else:                                 # volume-spike estimates: the past only (no peeking at later spikes)
            if t not in proxies:
                proxies[t] = earnings_proxy(c, panel.v(t), panel.dates)
            known = proxies[t]
        past = [d for d in known if d <= anchor.isoformat()]
        beta = _beta_at(c, spy, i) if spy is not None and t != BENCH else 1.0
        try:
            fc = forecast.build(t, dates, hist[ok].tolist(), beta=beta, earnings_next=nxt, past_earnings=past,
                                macro=macro, paths=paths, rf=float(cfg.get("risk_free", 0.04)),
                                erp=float(cfg.get("equity_premium", 0.05)))
        except Exception as err:  # noqa: BLE001
            log.warning("backtest %s@%s: %s", t, anchor, err)
            continue
        done += 1
        if fc is None:
            continue
        made = fc["made"]
        for name, h in forecast.HORIZONS[:4]:                 # 1-year outcomes don't fit in two years of history
            k = i + h
            if k >= panel.T:
                continue
            seg = c[i:k + 1]
            if not (np.isfinite(seg).all() and np.all(seg > 0)):
                # stopped trading inside the window (and never again): graded to its last close, like live ones
                good = np.flatnonzero(np.isfinite(seg) & (seg > 0))
                if not len(good) or good[-1] == 0:
                    continue
                rest = c[i + good[-1] + 1:]
                if np.any(np.isfinite(rest) & (rest > 0)):
                    continue                                  # a gap in the data, not a delisting
                seg = seg[: good[-1] + 1]
                if not (np.isfinite(seg).all() and np.all(seg > 0)):
                    continue
                k = i + good[-1]
            r = math.log(c[k] / c[i])
            path = np.log(seg / c[i])
            dd = float(1 - math.exp(-float(np.max(np.maximum.accumulate(path) - path))))
            hz = fc["horizons"][name]
            added += fledger.add_backtest(L, t, made, h, hz["lq"], hz["lq_raw"], hz["p_up"],
                                          forecast.naive_lq(fc["naive_sigma_d"], h), r, dd, today, float(c[i]),
                                          float(c[k]), dt.date.fromordinal(int(panel.dates[k])),
                                          feats=ml.row_features(bfeat, fc, name) if bfeat else None)
    bt["todo"] = todo
    fledger.save(heavy_dir, L)
    settings.mark_heavy_dirty()
    left = len(todo)
    return added, (f"{added} backtest forecasts graded this run" + (f"; {left} stock-dates to go" if left else "; complete")
                   + (f"; signal features added to {filled} earlier ones" if filled else ""))


def _bt_features(hist: np.ndarray, spy: np.ndarray | None, i: int, ok: np.ndarray, vol: np.ndarray | None = None
                 ) -> dict | None:
    """Price and volume features known at a backtest anchor (row i), from the stock's valid closes (and volumes) up
    to it and the S&P 500's closes on the same days."""
    if spy is None:
        return None
    b = spy[: i + 1][ok]
    if not np.all(np.isfinite(b) & (b > 0)):
        return None
    v = vol[: i + 1][ok].tolist() if vol is not None else None
    f = ml.price_features(hist[ok].tolist(), b.tolist(), v)
    return f if "rev" in f else None


def _fill_bt_features(L: dict, panel, spy: np.ndarray | None, limit: int = 6000) -> int:
    """Backtest rows written before this version's features get them now (same prices and volumes, same anchor day).
    Whether an earnings report fell inside the window isn't known for these, so it stays blank."""
    if spy is None:
        return 0
    n = 0
    for h in ("1", "5", "21", "63"):
        for r in (L.get("graded", {}).get("quant_bt", {}).get(h) or []):
            if n >= limit:
                return n
            if (r.get("f") or {}).get("v") == ml.FEAT_V or r.get("f_tried") == ml.FEAT_V:
                continue
            c = panel.c(r["t"])
            i = panel.ix_on_or_before(dt.date.fromisoformat(r["made"]))
            f = None
            if c is not None and i is not None and panel.dates[i] == dt.date.fromisoformat(r["made"]).toordinal():
                hist = c[: i + 1]
                ok = np.isfinite(hist) & (hist > 0)
                if ok.any() and ok[-1]:
                    f = _bt_features(hist, spy, i, ok, panel.v(r["t"]))
            if f:
                r["f"] = {k: round(float(v), 4) for k, v in f.items()}
                r.pop("f_tried", None)
            else:
                r["f_tried"] = ml.FEAT_V          # not enough history on that date: don't look again
            n += 1
    return n


def merge_claude_forecasts(settings, state: dict, heavy_dir: Path) -> int:
    """Record Claude's price forecasts (and the model+Claude blend) in the forecast ledger. Each Claude entry also
    carries the model's own forecast from the same close, so the two are compared on exactly the same stocks,
    dates and outcomes; and the close it saw, so a later split re-bases its price levels."""
    ai = read_json(settings.cache_dir / "ai" / "analyses.json.gz") or {}
    pf = ai.get("price_forecasts") or {}
    if not pf:
        return 0
    fcache = (read_json(settings.cache_dir / FC_CACHE) or {}).get("by_ticker") or {}
    L = fledger.load(heavy_dir)
    bw = (state.get("learn") or {}).get("blend") or {}
    n = 0
    for t, rec in pf.items():
        made, ref, last, ld = rec.get("made"), rec.get("ref_date"), rec.get("last_close"), rec.get("last_close_date")
        if not made or not ref or not last or not ld:
            continue
        qf = fcache.get(t)
        same = bool(qf and qf.get("made") == ld)          # the model's forecast from the close Claude started from
        for name, row in (rec.get("horizons") or {}).items():
            h = HMAP.get(name)
            if not h:
                continue
            lq = forecast.from_p10_p50_p90(row["low_10"], row["median"], row["high_90"], float(last))
            if not lq:
                continue
            pq = [float(last) * math.exp(x) for x in lq]
            nv = forecast.naive_lq(qf["naive_sigma_d"], h) if qf else None
            extra = {"last": float(last), "ld": ld}
            qlq = (qf["horizons"].get(name) or {}).get("lq") if same else None
            if qlq:
                extra["qpq"] = [round(float(qf["anchor"]) * math.exp(x), 4) for x in qlq]
            n += fledger.record(L, "claude", t, made, h, None, pq=pq, pup=row.get("p_up"), naive=nv, ref_date=ref,
                                extra=extra)
            if qlq:
                w = float((bw.get(name) or {}).get("w_claude", 0.5))
                # in returns from each one's own close (the same close here), then back to prices
                bl = [w * a + (1 - w) * (b + math.log(float(qf["anchor"]) / float(last))) for a, b in zip(lq, qlq)]
                bq = [float(last) * math.exp(x) for x in bl]
                pu = None
                if row.get("p_up") is not None:
                    pu = w * row["p_up"] + (1 - w) * qf["horizons"][name]["p_up"]
                n += fledger.record(L, "blend", t, made, h, None, pq=bq, pup=pu, naive=nv, ref_date=ref,
                                    extra={"w": round(w, 3), "last": float(last), "ld": ld})
    if n:
        fledger.save(heavy_dir, L)
        settings.mark_heavy_dirty()
    return n


# ---------------------------------------------------------------- grading & learning

def _series_fn(panel, store):
    def get(t: str):
        if panel is not None:
            c = panel.c(t)
            if c is not None:
                ok = np.isfinite(c) & (c > 0)
                if ok.any():
                    return [int(d) for d in panel.dates[ok]], c[ok]
        s = store.series(t)
        if s and s.get("dates"):
            return list(s["dates"]), np.asarray(s["c"], dtype=float)
        return None
    return get


def job_grade_learn(settings, state: dict, panel, store, heavy_dir: Path, budget=None) -> tuple[int, str]:
    L = fledger.load(heavy_dir)
    get = _series_fn(panel, store)
    mlast = int(panel.dates[-1]) if panel is not None else ((store.series(BENCH) or {}).get("dates") or [None])[-1]
    n = fledger.grade(L, get, market_last=mlast)
    cfg = settings.c("forecast", default={}) or {}
    calib = learn.calibration(L, min_n=int(cfg.get("calib_min_n", 200)), min_dates=int(cfg.get("calib_min_dates", 20)))
    lr = state.get("learn") or {}
    rl = lr.get("range")
    try:                                              # the range learner: re-tested with every calibration
        rl = ml.range_learner(L, calib, prev=lr.get("range"))
    except Exception as err:  # noqa: BLE001 – keep the last decision if the test itself fails
        log.warning("range learner: %s", err)
    tilt = lr.get("tilt")
    left = budget.left() - 120 if budget is not None else 600.0
    tilt = learn.tilt_in_force(lr, now_utc())
    due = (not tilt or not lr.get("tilt_at") or tilt.get("fv") != ml.FEAT_V
           or (now_utc() - dt.datetime.fromisoformat(lr["tilt_at"].replace("Z", "+00:00"))).days >= 7)
    if due and left > 60:
        try:
            res = ml.compete(L, seconds=min(300.0, left))   # the weekly algorithm competition (time-boxed)
            if not res.get("postponed"):
                tilt = learn.tilt_gate(tilt if (tilt or {}).get("fv") == ml.FEAT_V else None, res)
                lr["tilt_at"] = iso(now_utc())      # on only after two weekly passes in a row
        except Exception as err:  # noqa: BLE001
            log.warning("signal learner: %s", err)
    m = fledger.metrics(L)
    state["learn"] = {**lr, "calib": calib, "tilt": tilt, "range": rl, "blend": learn.blend_weights(L),
                      "updated": iso(now_utc())}
    state["fc_metrics"] = m
    write_json(settings.cache_dir / FEEDBACK, claude_feedback(L, m, get))
    fledger.save(heavy_dir, L)
    settings.mark_heavy_dirty()
    applied = [k for k, v in calib.items() if v.get("status") == "applied"]
    return n, (f"{n} forecasts graded; {m['pending']} waiting; calibration applied at "
               + (", ".join(applied) if applied else "no horizon yet (needs 200+ graded from 20+ days)")
               + f"; signal learner {'on (' + ml.NAMES.get((tilt or {}).get('model'), '?') + ')' if (tilt or {}).get('on') else 'off'}"
               + f"; range learner on at {', '.join(k for k, v in (rl or {}).items() if v.get('status') == 'applied') or 'no horizon'}")


def claude_feedback(L: dict, m: dict, get) -> dict:
    """Claude's own record and its biggest recent misses, for the analyst workflow (which can't see the ledger)."""
    track = {}
    for name, h in forecast.HORIZONS:
        c = ((m.get("by") or {}).get("claude") or {}).get(name)
        q = ((m.get("by") or {}).get("quant") or {}).get(name)
        if not c or not c["all"].get("n"):
            continue
        a = c["all"]
        track[name] = {"graded": a["n"], "inside_80pct_range": a.get("cov80"), "inside_50pct_range": a.get("cov50"),
                       "direction_right": a.get("hit"), "skill_vs_naive": a.get("skill"),
                       "app_model_skill_vs_naive": (q or {}).get("all", {}).get("skill"),
                       "median_bias_log": (c.get("recent") or {}).get("bias"),
                       "advice": ("ranges too narrow: widen them" if (a.get("cov80") or 0.8) < 0.72 else
                                  "ranges too wide: you can narrow them a little" if (a.get("cov80") or 0.8) > 0.9 else
                                  "ranges about right")}
    misses = []
    for r in fledger.recent_misses(L, "claude"):
        lq = r.get("lq") or []
        if len(lq) != 7 or not r.get("p0"):
            continue
        p0 = r["p0"]
        path = None
        s = get(r["t"])
        if s:
            dates, c = s
            try:
                a = dates.index(dt.date.fromisoformat(r["ref"]).toordinal())
                b = dates.index(dt.date.fromisoformat(r["end"]).toordinal())
                path = [{"date": dt.date.fromordinal(int(dates[i])).isoformat(), "close": round(float(c[i]), 2)}
                        for i in range(a, b + 1, 5)]
            except (ValueError, KeyError):
                path = None
        misses.append({"t": r["t"], "h": r["h"], "made": r["made"], "ref": r["ref"], "end": r.get("end"),
                       "p0": p0, "p1": r.get("p1"), "ret_pct": round(math.expm1(r["r"]) * 100, 1),
                       "p10": round(p0 * math.exp(lq[1]), 2), "p50": round(p0 * math.exp(lq[3]), 2),
                       "p90": round(p0 * math.exp(lq[5]), 2), "pit": r["pit"], "path": path})
    return {"generated_at": iso(now_utc()), "track": track, "misses": misses}


# ---------------------------------------------------------------- conviction

def apply_conviction(settings, state: dict, ctx: dict, led: dict) -> int:
    from . import ledger as binary
    cfg = settings.c("conviction", default={}) or {}
    ai = read_json(settings.cache_dir / "ai" / "analyses.json.gz") or {}
    pf = ai.get("price_forecasts") or {}
    lists = (state.get("ideas") or {}).get("lists") or {}
    items = [it for lst in lists.values() for it in lst.get("items", [])]
    events = state.get("events", [])
    quotes = state.get("quotes") or {}
    if not ctx:                         # the context job failed this run: keep the last ratings and streaks
        return len(state.get("conv_top") or [])
    prev = set(state.get("conv_top") or [])
    top, rows = set(), []
    today = now_et().date().isoformat()
    base_any = ((lists.get("momentum") or {}).get("track") or {}).get("all_stocks")
    for t, x in ctx.items():
        cl = conviction.latest_claude(t, events, items, pf.get(t), base_any=base_any)
        sup = conviction.measured_support(t, events, lists)
        r = conviction.rate(t, quote=quotes.get(t), val=x.get("val"), bub=x.get("bubble"), flags=x.get("flags"),
                            claude=cl, support=sup, cfg=cfg)
        x["conv"] = r
        if t in quotes:
            quotes[t]["conv"] = r["rating"]
        if r["rating"] == "top":
            top.add(t)
            if t not in prev:
                binary.record(led, f"conv:{t}:{today}", ticker=t, made=today, p=r.get("p"), source="conviction",
                              kind="conviction", action="top")
        if r["rating"] in ("top", "positive", "caution", "red"):
            rows.append({"ticker": t, "name": x.get("name"), "rating": r["rating"], "label": r["label"],
                         "passed": r["passed"], "of": r["of"], "why": r["why"], "p": r.get("p"),
                         "val": (x.get("val") or {}).get("label"), "bubble": (x.get("bubble") or {}).get("level")})
    order = {"top": 0, "positive": 1, "caution": 2, "red": 3}
    rows.sort(key=lambda r: (order[r["rating"]], -r["passed"]))
    state["conv_top"] = sorted(top)
    state["conviction"] = {"rows": rows[:80], "top": len(top), "asof": iso(now_utc())}
    _write_ctx(settings, ctx)
    return len(top)


# ---------------------------------------------------------------- per-stock files

def _px(panel, store, t: str, n: int = 504) -> dict | None:
    s = None
    if panel is not None and panel.c(t) is not None:
        c = panel.c(t)
        ok = np.isfinite(c) & (c > 0)
        s = ([int(d) for d in panel.dates[ok]], [float(x) for x in c[ok]])
    elif store.series(t):
        x = store.series(t)
        s = (x["dates"], x["c"])
    if not s or not s[0]:
        return None
    d, c = s[0][-n:], s[1][-n:]
    keep = [i for i in range(len(d)) if i >= len(d) - 260 or i % 2 == 0]       # older half thinned
    return {"d": [dt.date.fromordinal(int(d[i])).isoformat() for i in keep], "c": [round(float(c[i]), 4) for i in keep]}


def _close_on(panel, store, t: str, d: str) -> float | None:
    s = _series_fn(panel, store)(t)
    if not s:
        return None
    dates, c = s
    import bisect
    try:
        o = dt.date.fromisoformat(d[:10]).toordinal()
    except ValueError:
        return None
    i = bisect.bisect_right(dates, o) - 1
    if i < 0 or o - dates[i] > 4:
        return None
    x = float(c[i])
    return x if x > 0 and math.isfinite(x) else None


def _claude_view(cf: dict | None, fc: dict | None, panel, store, t: str) -> dict | None:
    """Claude's forecast as the page should show it: re-based for any split since it was written (the price
    history is split-adjusted), and without horizons whose target date has already passed."""
    if not cf:
        return None
    cf = json.loads(json.dumps(cf))
    last, ld = cf.get("last_close"), cf.get("last_close_date")
    f = 1.0
    if last and ld:
        now_c = _close_on(panel, store, t, ld)
        if now_c:
            r = now_c / float(last)
            if math.isfinite(r) and 0.005 < r < 200 and abs(math.log(r)) > 0.02:
                f = r
    if f != 1.0:
        for row in (cf.get("horizons") or {}).values():
            for k in ("low_10", "median", "high_90"):
                if row.get(k):
                    row[k] = round(row[k] * f, 4)
        for sc in cf.get("scenarios") or []:
            if sc.get("price_3m"):
                sc["price_3m"] = round(sc["price_3m"] * f, 4)
        cf["last_close"] = round(float(last) * f, 4)
        cf["split_adjusted"] = round(f, 6)
    asof = (fc or {}).get("made") or now_et().date().isoformat()
    tds = cf.get("target_dates") or {}
    expired = [k for k in list((cf.get("horizons") or {}).keys()) if tds.get(k) and tds[k] <= asof]
    for k in expired:
        cf["horizons"].pop(k, None)
    if expired:
        cf["expired"] = expired
    return cf


def publish_stocks(settings, state: dict, ctx: dict, panel, store) -> int:
    out_dir = settings.out_dir / "stock"
    out_dir.mkdir(parents=True, exist_ok=True)
    fc_on = (settings.c("forecast", default={}) or {}).get("enabled", True)
    fcache = ((read_json(settings.cache_dir / FC_CACHE) or {}).get("by_ticker") or {}) if fc_on else {}
    ai = read_json(settings.cache_dir / "ai" / "analyses.json.gz") or {}
    pf = ai.get("price_forecasts") or {}
    quotes = state.get("quotes") or {}
    lr = state.get("learn") or {}
    gen = iso(now_utc())
    off_note = "Price forecasts are switched off in this app's settings (forecast.enabled in config.yaml)."
    held_note = ("Forecast paused: the latest prices look like a stock split the price history hasn't been adjusted "
                 "for yet. It returns once the history is re-based.")
    names = set(ctx) | set(fcache)
    written, failed = set(), 0
    index: list[dict] = []
    cut = (now_utc() - dt.timedelta(days=45)).isoformat()
    for t in names:
        if not _plausible(t):
            continue
        try:                                   # one bad record must not cost every other stock its page
            x = ctx.get(t) or {}
            fc = fcache.get(t)
            cf = pf.get(t) if (pf.get(t) or {}).get("made_at", "") >= cut else None
            cf = _claude_view(cf, fc, panel, store, t)
            held = (state.get("fc_held") or {}).get(t)
            doc = {"t": t, "generated_at": gen, "name": x.get("name") or (quotes.get(t) or {}).get("name"),
                   "forecast_note": None if fc else off_note if not fc_on else held_note if held else None,
                   "quote": quotes.get(t), "px": _px(panel, store, t),
                   "forecast": _slim_fc(fc), "claude": cf, "blend": _blend_view(fc, cf, lr.get("blend") or {}),
                   "valuation": x.get("val"), "bubble": x.get("bubble"), "ai": x.get("ai"),
                   "news": {"filings": x.get("filings") or [], "headlines": x.get("headlines") or [],
                            "flags": x.get("flags") or [], "claude": (cf or {}).get("news_digest") or [],
                            "headlines_fetched": x.get("headlines_fetched")},
                   "conviction": x.get("conv"), "industry": x.get("industry"), "first_filed": x.get("first_filed")}
            write_json(out_dir / f"{t}.json", doc)
            written.add(f"{t}.json")
            if fc:
                index.append(_index_row(t, doc, fc))
        except Exception as err:  # noqa: BLE001
            failed += 1
            log.warning("stock page %s: %s", t, err)
            if (out_dir / f"{t}.json").exists():
                written.add(f"{t}.json")       # keep the last good page rather than none (and its line in the list)
                try:
                    old = read_json(out_dir / f"{t}.json") or {}
                    if old.get("forecast"):
                        index.append(_index_row(t, old, old["forecast"]))
                except Exception:  # noqa: BLE001
                    pass
    for f in out_dir.glob("*.json"):
        if f.name not in written:
            f.unlink(missing_ok=True)
    state["stock_pages"] = {"written": len(written), "failed": failed, "names": len(names)}
    index.sort(key=lambda r: r["t"])
    write_json(settings.out_dir / "forecasts.json", {"generated_at": gen, "enabled": bool(fc_on), "n": len(index),
                                                     "stocks": index})
    return len(written)


def _index_row(t: str, doc: dict, fc: dict) -> dict:
    """One line of the list of every stock with a forecast: enough to compare them at a glance."""
    def hz(name):
        h = (fc.get("horizons") or {}).get(name) or {}
        q = h.get("q") or {}
        return {"lo": q.get("10"), "mid": q.get("50"), "hi": q.get("90"), "up": h.get("p_up")} if q else None
    conv = doc.get("conviction") or {}
    return {"t": t, "name": doc.get("name"), "made": fc.get("made"), "close": fc.get("anchor"),
            "m1": hz("1m"), "m3": hz("3m"), "y1": hz("1y"),
            "val": (doc.get("valuation") or {}).get("label"), "bubble": (doc.get("bubble") or {}).get("level"),
            "rating": conv.get("rating"), "claude": bool(doc.get("claude")),
            "nudge": (fc.get("tilt") or {}).get("ret") or None}


def _slim_fc(fc: dict | None) -> dict | None:
    if not fc:
        return None
    d = {k: v for k, v in fc.items() if k not in ("cv",)}
    d["horizons"] = {k: {kk: vv for kk, vv in h.items() if kk not in ("lq", "lq_raw")} for k, h in fc["horizons"].items()}
    return d


def _fan_at(fc: dict, day: str) -> list[float] | None:
    """The model's price quantiles for a given date, interpolated (in log price) along its fan."""
    fan = fc.get("fan") or {}
    ds = fan.get("dates") or []
    if not ds or day <= ds[0] or day > ds[-1]:
        return None
    o = dt.date.fromisoformat(day).toordinal()
    od = [dt.date.fromisoformat(d).toordinal() for d in ds]
    k = next(i for i, x in enumerate(od) if x >= o)
    a, b = od[k - 1], od[k]
    w = 0.0 if b == a else (o - a) / (b - a)
    out = []
    for q in forecast.QK:
        lo, hi = fan["p"][q][k - 1], fan["p"][q][k]
        if not lo or not hi or lo <= 0 or hi <= 0:
            return None
        out.append(math.exp((1 - w) * math.log(lo) + w * math.log(hi)))
    return out


def _blend_view(fc: dict | None, cf: dict | None, bw: dict) -> dict | None:
    """Model + Claude combined for each of Claude's (unexpired) target dates: Claude's split-normal quantiles and
    the model's quantiles for that same date, averaged in log price."""
    if not fc or not cf:
        return None
    out = {}
    last = cf.get("last_close")
    for name, row in (cf.get("horizons") or {}).items():
        day = (cf.get("target_dates") or {}).get(name)
        if not last or not day:
            continue
        lq = forecast.from_p10_p50_p90(row["low_10"], row["median"], row["high_90"], float(last))
        mq = _fan_at(fc, day)
        if not lq or not mq:
            continue
        w = float((bw.get(name) or {}).get("w_claude", 0.5))
        bq = [math.exp(w * (math.log(float(last)) + x) + (1 - w) * math.log(y)) for x, y in zip(lq, mq)]
        out[name] = {"q": {k: round(v, 4) for k, v in zip(forecast.QK, bq)}, "w_claude": w, "date": day,
                     "model_q": {k: round(v, 4) for k, v in zip(forecast.QK, mq)}}
    return out or None


def accuracy_doc(settings, state: dict, led_metrics: dict | None) -> dict:
    ai = read_json(settings.cache_dir / "ai" / "analyses.json.gz") or {}
    pms = sorted((ai.get("postmortems") or {}).values(), key=lambda p: p.get("made") or "", reverse=True)[:12]
    lr = state.get("learn") or {}
    return {
        "generated_at": iso(now_utc()), "metrics": state.get("fc_metrics"),
        "learning": {"calibration": lr.get("calib"), "tilt": lr.get("tilt"), "range": lr.get("range"), "blend": lr.get("blend"),
                     "updated": lr.get("updated")},
        "lessons": [{**L, "lesson": A.sanitize_lesson(L.get("lesson"))} for L in reversed((ai.get("lessons") or [])[-20:])
                    if A.sanitize_lesson(L.get("lesson"))],
        "postmortems": [{**{k: p.get(k) for k in ("t", "h", "forecast_made", "cause", "foreseeable", "lesson",
                                                   "range_advice", "made", "model")},
                         "what_happened": A._URLISH.sub("", str(p.get("what_happened") or ""))[:800]} for p in pms],
        "conviction_record": ((led_metrics or {}).get("by_source") or {}).get("conviction"),
        "claude_beat_index_record": ((led_metrics or {}).get("by_source") or {}).get("claude"),
        "targets": {"cov80": 0.80, "cov80_band": [0.75, 0.85], "cov50": 0.50,
                    "direction_note": ("Published research puts realistic direction accuracy for single stocks a few points "
                                       "above 50%; the app shows its own rate next to the 'always say up' baseline.")},
    }
