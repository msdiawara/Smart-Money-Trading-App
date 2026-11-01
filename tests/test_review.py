"""The Results tab's data (pipeline/review.py): every miss taken apart, the win rates against what the forecasts
expected, and what the learning loop does about each kind of miss. Also the news, filing and financial-statement
signals (ml.context_features), the fused model and the stock's own miss record (pipeline/ml.py), with the problems
the round-nine audits found pinned."""
import datetime as dt
import math

import numpy as np
import pytest

from pipeline import fledger, ml, news, review
from pipeline.prices import BENCH
from tests.test_forecast import END, gbm, trading_dates

DATES = trading_dates(END, 320)


def _iso(i: int) -> str:
    return dt.date.fromordinal(DATES[i]).isoformat()


def _row(c, i0, h, mid=0.0, half=0.06, beta=1.0):
    r = math.log(c[i0 + h] / c[i0])
    f = {"beta": beta} if beta is not None else {}
    return {"t": "X", "made": _iso(i0), "r": r, "err": r - mid, "pw10": half, "pw90": half,
            "in80": int(mid - half <= r <= mid + half), "f": f}


def _quiet(n=320, seed=2, vol=0.004):
    return 50 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, vol, n)))


FLAT = {d: 500.0 for d in DATES}


# ---------------------------------------------------------------- one miss, taken apart

def test_an_earnings_jump_is_blamed_on_earnings_when_the_date_is_known():
    c = _quiet()
    c[250:] *= math.exp(0.15)                                  # +15% on its earnings day
    w = review.diagnose(_row(c, 240, 21), 21, DATES, c, FLAT, None, [_iso(250)], {_iso(250): "pre"})
    assert w["c"] == "earnings" and w["side"] == "above" and w["ed"] == _iso(250) and w["em"] == pytest.approx(0.15, abs=0.02)
    assert "est" not in w
    # after the close: the next session reacts
    c2 = _quiet()
    c2[251:] *= math.exp(0.15)
    w2 = review.diagnose(_row(c2, 240, 21), 21, DATES, c2, FLAT, None, [_iso(250)], {_iso(250): "post"})
    assert w2["c"] == "earnings" and w2["ed"] == _iso(251)
    # timing unknown: whichever of the two days moved
    w3 = review.diagnose(_row(c2, 240, 21), 21, DATES, c2, FLAT, None, [_iso(250)], {})
    assert w3["c"] == "earnings" and w3["ed"] == _iso(251)
    # an earnings day estimated from volume spikes counts when the spikes keep a quarterly rhythm, and says so
    w4 = review.diagnose(_row(c, 240, 21), 21, DATES, c, FLAT, None, [], {}, [], [_iso(187), _iso(250)])
    assert w4["c"] == "earnings" and w4["est"] == 1
    # a one-off spike is more likely news: it's a one-day shock, with the 8-K filed that day
    w5 = review.diagnose(_row(c, 240, 21), 21, DATES, c, FLAT, None, [], {}, [{"date": _iso(250), "items": [
        {"item": "8.01", "label": "Other material event", "sev": "info"}]}], [_iso(250)])
    assert w5["c"] == "news" and w5["jd"] == _iso(250) and w5["f8"]["item"] == "8.01"


def test_a_release_before_the_forecast_or_before_its_first_open_isnt_in_the_window():
    c = _quiet()
    c[241:] *= math.exp(0.15)                                  # the move happens on the window's first session
    row = _row(c, 240, 21)
    pre_anchor = review.diagnose(row, 21, DATES, c, FLAT, None, [_iso(240)], {_iso(240): "pre"})
    assert pre_anchor["c"] != "earnings"                       # released before the anchor's open: in its close
    post_anchor = review.diagnose(row, 21, DATES, c, FLAT, None, [_iso(240)], {_iso(240): "post"})
    assert post_anchor["c"] == "earnings" and post_anchor["ed"] == _iso(241)
    assert review.diagnose(row, 21, DATES, c, FLAT, None, [_iso(200)], {})["c"] != "earnings"


def test_a_market_crash_is_blamed_on_the_market_and_a_steady_drift_on_trend():
    spy_c = np.full(320, 500.0)
    spy_c[245:] *= math.exp(-0.10)                             # the market falls 10% in the window
    spy = dict(zip(DATES, spy_c))
    c = _quiet(seed=4)
    c[245:] *= math.exp(-0.15)                                 # a beta-1.5 stock falls 15% with it
    w = review.diagnose(_row(c, 240, 21, beta=1.5), 21, DATES, c, spy, mkt_err=-0.10)
    assert w["c"] == "market" and w["side"] == "below" and w["parts"]["market"] == pytest.approx(-0.15, abs=0.01)
    d = _quiet(seed=5) * np.exp(np.r_[np.zeros(241), 0.005 * np.arange(1, 80)])     # +0.5% a day, every day
    w2 = review.diagnose(_row(d, 240, 21), 21, DATES, d, FLAT)
    assert w2["c"] == "trend" and w2["side"] == "above"


def test_a_row_without_the_models_signals_gets_its_beta_from_its_prices():
    # Claude's rows carry no beta: a beta-2 stock falling with the market is the market's doing, not a shock
    rng = np.random.default_rng(3)
    m = np.cumsum(rng.normal(0, 0.01, 320))
    m[245:] -= 0.12
    spy = dict(zip(DATES, 500 * np.exp(m)))
    c = 50 * np.exp(2 * m + np.cumsum(rng.normal(0, 0.003, 320)))
    no_beta = _row(c, 240, 21, beta=None)
    w = review.diagnose(no_beta, 21, DATES, c, spy)
    assert w["beta_est"] == 1 and w["beta"] == pytest.approx(2.0, abs=0.2) and w["c"] == "market"


def test_a_stock_far_more_volatile_than_forecast_is_blamed_on_volatility():
    rng = np.random.default_rng(7)
    c = 50 * np.exp(np.cumsum(np.r_[rng.normal(0, 0.004, 240), rng.normal(0.004, 0.03, 80)]))
    row = _row(c, 240, 21, half=0.04)
    w = review.diagnose(row, 21, DATES, c, FLAT)
    if not row["in80"]:
        assert w["c"] in ("volatility", "news") and w["vol"] > 1.5
    w1 = review.diagnose(_row(c, 240, 1, half=0.003), 1, DATES, c, FLAT)
    assert w1["c"] in ("volatility", "news") and w1["c"] != "trend"     # a 1-day forecast has no "trend"


def test_no_price_on_the_forecasts_date_means_unknown():
    c = _quiet()
    row = _row(c, 240, 21)
    row["made"] = "2019-01-02"
    assert review.diagnose(row, 21, DATES, c, FLAT)["c"] == "unknown"
    assert review.diagnose({"t": "X", "made": _iso(240), "r": 0.1}, 21, DATES, c, {})["c"] == "unknown"


# ---------------------------------------------------------------- the page

def _agg(L):
    """The monthly totals, as the ledger keeps them for every graded forecast (the S&P 500's included)."""
    L["agg"] = {}
    for src, by_h in L["graded"].items():
        for hk, rows in by_h.items():
            for g in rows:
                a = L["agg"].setdefault(src, {}).setdefault(hk, {}).setdefault(g["made"][:7], {
                    "n": 0, "in50": 0, "in80": 0, "in90": 0, "in80r": 0, "hits": 0, "nhit": 0, "up": 0, "qs": 0.0,
                    "qsn": 0.0, "nqsn": 0, "pit": [0] * 10, "err": 0.0})
                a["n"] += 1
                a["in80"] += g["in80"]
                a["in80r"] += g["in80r"]
                a["up"] += g["up"]
                a["hits"] += g["hit"]
                a["nhit"] += 1
                a["pexp"] = a.get("pexp", 0.0) + max(g["pup"], 1 - g["pup"])
                a["npexp"] = a.get("npexp", 0) + 1
                a["hitx"] = a.get("hitx", 0) + g["hit"]
                a["qs"] += g["qs"]
                a["qsn"] += g["qsn"]
                a["nqsn"] += 1
    return L


def _ledger_with_misses(n_days=150, per=10, seed=3, horizons=(1, 21)):
    """A small live record: forecasts at a few horizons, some missing, and the S&P 500's own rows."""
    rng = np.random.default_rng(seed)
    prices = {f"S{i}": 40 * np.exp(np.cumsum(rng.normal(0, 0.02, 320))) for i in range(per)}
    spy = gbm(320, 0.01, seed=1, start=500)
    L = {"v": 1, "pending": {}, "graded": {"quant": {str(h): [] for h in horizons}}, "agg": {}, "last": {}}
    for k in range(100, 100 + n_days):
        for t, c in list(prices.items()) + [(BENCH, spy)]:
            for h in horizons:
                if k + h >= 320:
                    continue
                r = math.log(c[k + h] / c[k])
                half = 0.02 * 1.2816 * math.sqrt(h) * 0.8            # a bit too narrow: some misses
                pup = float(rng.uniform(0.45, 0.6))
                g = {"t": t, "made": _iso(k), "r": round(r, 5), "err": round(r, 5), "pw10": half, "pw90": half, "pit": 0.5,
                     "w10": half, "w90": half, "in80": int(-half <= r <= half), "in80r": int(-half <= r <= half),
                     "up": int(r > 0), "pup": pup, "hit": int((pup > 0.5) == (r > 0)), "qs": 0.01, "qsn": 0.011,
                     "f": {"beta": 1.0, "lvol": float(rng.normal()), "liq": float(rng.normal()), "ein": int(rng.random() < 0.2),
                           "v": ml.FEAT_V}, "p0": round(float(c[k]), 4), "end": _iso(k + h)}
                L["graded"]["quant"][str(h)].append(g)
    get = lambda t: (DATES, prices[t] if t in prices else spy, None) if (t in prices or t == BENCH) else None  # noqa: E731
    return _agg(L), get


LR = {"calib": {"1m": {"status": "applied", "scale": 1.18}}, "keeper": {"1d": {"status": "applied", "m": 1.07}},
      "range": {"1m": {"status": "held"}}, "tilt": {"on": False, "t": 1.2, "t_needed": 3.1, "status": "rejected"}}


def test_misses_are_taken_apart_once_and_the_page_adds_up():
    L, get = _ledger_with_misses()
    n = review.fill_why(L, "quant", get)
    rows = [r for rr in L["graded"]["quant"].values() for r in rr if r["t"] != BENCH]
    every = [r for rr in L["graded"]["quant"].values() for r in rr]          # (the totals count the S&P 500's too)
    misses = [r for r in rows if not r["in80"]]
    assert n == len(misses) > 0 and all("why" in r for r in misses) and not any("why" in r for r in rows if r["in80"])
    assert review.fill_why(L, "quant", get) == 0                     # each is worked out once
    doc = review.build(L, get, LR, today=dt.date.fromordinal(DATES[-1]))
    q = doc["sources"]["quant"]
    hd = q["headline"]
    assert hd["n"] == len(every) and hd["win"] == pytest.approx(np.mean([r["in80"] for r in every]), abs=1e-3)
    assert hd["n_miss"] == sum(1 - r["in80"] for r in every)
    assert hd["dir_expected"] == pytest.approx(np.mean([max(r["pup"], 1 - r["pup"]) for r in every]), abs=1e-3)
    assert hd["dir_expected_from"] == "all" and hd["first"] == min(r["made"][:7] for r in every)
    one = q["by_h"]["1m"]
    assert set(q["by_h"]) == {"1d", "1m"} and one["n"] == len(L["graded"]["quant"]["21"]) and isinstance(one["win"], float)
    assert one["kept_miss"] == sum(1 for r in L["graded"]["quant"]["21"] if r["t"] != BENCH and not r["in80"])
    assert sum(q["causes"]["all"].values()) == len(misses) and "pending" not in q["causes"]["all"]
    ends = [m["end"] for m in q["misses"]]
    assert ends == sorted(ends, reverse=True)
    # each miss says what changes: its own text where it differs (an 8-K named), else its horizon's for its cause
    assert all(m.get("fix") or q["by_h"][m["h"]]["fix"][m["why"]["c"]] for m in q["misses"])
    m0 = q["misses"][0]
    assert m0["lo"] < m0["mid"] < m0["hi"] and m0["p1"] == pytest.approx(m0["p0"] * (1 + m0["ret"]), rel=1e-3)
    assert (m0["p1"] > m0["hi"]) or (m0["p1"] < m0["lo"])
    assert any(g["kind"] == "earnings" for g in one["groups"])
    assert [t["m"] for t in q["trend"]] == sorted(t["m"] for t in q["trend"]) and all("miss" in t for t in q["trend"])
    assert [t["m"] for t in one["trend"]] == sorted({r["made"][:7] for r in L["graded"]["quant"]["21"]})
    # what changes is said from the learners' actual state
    cor = doc["corrections"]
    assert "×1.07" in review.fix_text("market", "1d", cor)
    assert "×1.18" in review.fix_text("news", "1m", cor)
    assert "t 1.2; 3.1 needed" in review.fix_text("trend", "1m", cor)
    assert "tested wider" in review.fix_text("earnings", "1m", cor)
    assert {x["cause"] for x in q["lessons"]} == set(q["causes"]["all"]) and all(x["fix"] for x in q["lessons"])


def test_the_totals_match_the_records_tab_and_count_more_than_the_kept_sample():
    L, get = _ledger_with_misses()
    rm = L["graded"]["quant"]["21"]
    L["graded"]["quant"]["21"] = rm[: len(rm) // 3]              # the ledger keeps a sample; the totals keep all
    doc = review.build(L, get, LR, today=dt.date.fromordinal(DATES[-1]))
    m = fledger.metrics(L, today=dt.date.fromordinal(DATES[-1]))["by"]["quant"]["1m"]["all"]
    one = doc["sources"]["quant"]["by_h"]["1m"]
    assert one["n"] == m["n"] == len(rm) and one["win"] == pytest.approx(m["cov80"], abs=1e-4)
    assert one["kept"] == len([r for r in L["graded"]["quant"]["21"] if r["t"] != BENCH])


def test_each_horizon_keeps_its_own_newest_misses_and_undone_ones_are_counted_apart():
    L, get = _ledger_with_misses(horizons=(1, 63), n_days=150)
    review.fill_why(L, "quant", get, limit=30)                   # most aren't taken apart yet
    doc = review.build(L, get, LR, today=dt.date.fromordinal(DATES[-1]), per_h_misses=10)
    q = doc["sources"]["quant"]
    by = {h: [m for m in q["misses"] if m["h"] == h] for h in ("1d", "3m")}
    assert len(by["1d"]) == 10 and len(by["3m"]) == 10          # the slow horizon isn't crowded out by the fast one
    allc = q["causes"]["all"]
    assert allc.get("pending", 0) > 0 and sum(allc.values()) == sum(x["kept_miss"] for x in q["by_h"].values())
    assert all(x["cause"] != "pending" for x in q["lessons"])
    assert abs(sum(x["share"] for x in q["lessons"]) - 1) < 0.01  # shares are of the misses taken apart
    pend = [m for m in q["misses"] if "why" not in m]
    assert pend and all((m.get("fix") or q["by_h"][m["h"]]["fix"]["pending"]).startswith("Not taken apart yet") for m in pend)


def test_what_changes_depends_on_whose_forecast_and_which_horizon():
    cor = review.corrections({**LR, "range": {"1m": {"status": "applied"}}, "keeper": {"1m": {"status": "applied", "m": 1.1}}})
    q1y = review.fix_text("market", "1y", cor)
    assert "doesn't run for 1-year" in q1y and "40 days" not in q1y
    assert "doesn't run for 1-year" in review.fix_text("earnings", "1y", cor)
    assert cor["1y"]["range"]["status"] == "n/a" and cor["1y"]["keeper"]["status"] == "n/a"
    cl = review.fix_text("news", "1m", cor, src="claude")
    assert "Claude's ranges are its own" in cl and "×" not in cl          # the model's stretch doesn't touch Claude's
    bl = review.fix_text("earnings", "1m", cor, src="blend")
    assert bl.startswith("The combined forecast") and "range learner widens" in bl
    assert review.fix_text("pending", "1m", cor).startswith("Not taken apart yet")


def test_a_miss_without_prices_is_tried_again_on_a_later_day_not_marked_for_good():
    L, get = _ledger_with_misses(per=2)
    day = dt.date.fromordinal(DATES[-1])
    gone = lambda t: None if t == "S0" else get(t)                 # noqa: E731 – S0's prices missing this run
    review.fill_why(L, "quant", gone, today=day)
    s0 = [r for rr in L["graded"]["quant"].values() for r in rr if r["t"] == "S0" and not r["in80"]]
    assert s0 and all(r["why"]["c"] == "unknown" and r["why"]["try"] == 1 for r in s0)
    assert review.fill_why(L, "quant", get, today=day) == 0         # not again the same day
    review.fill_why(L, "quant", get, today=day + dt.timedelta(days=1))
    assert all(r["why"]["c"] != "unknown" for r in s0)              # the prices are back: taken apart


def test_the_ledger_keeps_the_anchor_price_end_nudge_and_expected_odds():
    L = {"v": 1, "pending": {}, "graded": {}, "agg": {}, "last": {}}
    lq = [-0.06, -0.04, -0.02, 0.0, 0.02, 0.05, 0.07]
    fledger.record(L, "quant", "ACME", "2026-10-01", 21, lq, lq_raw=lq, pup=0.55, naive=lq, extra={"tl": 0.004})
    fid = next(iter(L["pending"]))
    fledger._finish(L, fid, L["pending"][fid], 0.01, 0.0, False, dt.date(2026, 11, 2), 100.0, 101.0, dt.date(2026, 10, 30))
    g = L["graded"]["quant"]["21"][0]
    assert g["p0"] == 100.0 and g["end"] == "2026-10-30" and g["tl"] == 0.004
    a = L["agg"]["quant"]["21"]["2026-10"]
    assert a["pexp"] == pytest.approx(0.55) and a["npexp"] == 1


def test_backtest_misses_are_taken_apart_with_their_own_prices_and_the_page_is_written(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from pipeline import insight, quant
    from pipeline.util import read_json
    from tests.test_forecast import _Matrix
    monkeypatch.setenv("SMR_FAKE_NOW", "2026-10-31T13:00:00+00:00")
    T = 520
    dates = trading_dates(END, T)
    cols = {"SPY": gbm(T, 0.01, seed=1, start=500)}
    for k in range(10):
        cols[f"S{chr(65 + k)}"] = gbm(T, 0.02, seed=10 + k, start=40, t_df=4)
    panel = quant.Panel(_Matrix(cols, dates))
    heavy, out = tmp_path / "heavy", tmp_path / "out"
    heavy.mkdir()
    out.mkdir()
    cfg = {"forecast": {"backtest_stocks": 10, "backtest_anchors": 16, "backtest_paths": 300, "backtest_seconds_per_run": 600}}
    settings = SimpleNamespace(c=lambda *k, default=None: cfg.get(k[0], default) if len(k) == 1 else default,
                               mark_heavy_dirty=lambda: None, out_dir=out)
    ctx = {t: {"industry": "Software"} for t in cols}
    insight.job_backtest(settings, {}, panel, SimpleNamespace(left=lambda: 1000), heavy, None, ctx)
    L = fledger.load(heavy)
    bt = [r for rr in L["graded"]["quant_bt"].values() for r in rr if r["t"] != BENCH]
    misses = [r for r in bt if not r["in80"]]
    assert misses and all("why" in r for r in misses)                  # taken apart as they were graded
    assert sum(r["why"]["c"] == "unknown" for r in misses) == 0
    assert all("mkt_err" in r["why"] for r in misses)                  # the S&P 500's surprise, from its own row
    # a backtest miss from years the live price history doesn't cover (the deep history's) is left for the backtest
    # job, not marked "no prices" for good
    old = {**misses[0], "made": "2017-03-03", "end": "2017-04-03"}
    old.pop("why")
    L["graded"]["quant_bt"]["21"].append(old)
    fledger.save(heavy, L)
    n, note = insight.job_results(settings, {}, panel, None, heavy, ctx)
    L2 = fledger.load(heavy)
    assert "why" not in [r for r in L2["graded"]["quant_bt"]["21"] if r["made"] == "2017-03-03"][0]
    doc = read_json(out / "results.json")
    s = doc["sources"]["quant_bt"]
    tot = sum(m["n"] for months in L2["agg"]["quant_bt"].values() for m in months.values())
    assert s["headline"]["n"] == tot and s["misses"] and "inside their 80% range" in note
    assert all(m["why"]["c"] != "unknown" for m in s["misses"] if m["made"] != "2017-03-03")


# ---------------------------------------------------------------- news, filings and statements, as known at the close

CUT, LAST = "2026-10-30T20:00:00+00:00", "2026-10-30"


def _ctx(**kw):
    x = {"news_sigs": [["2026-10-29T15:00:00+00:00", 9, 0.4, 0], ["2026-10-30T18:30:00+00:00", 12, 0.25, 0],
                       ["2026-10-30T22:00:00+00:00", 40, -0.5, 1]],             # (after the close: not used)
         "filings_at": "2026-10-28T10:00:00+00:00",
         "filings": [{"date": "2026-10-28", "at": "2026-10-28T12:00:00+00:00", "items": [{"item": "1.01", "sev": "info"}]},
                     {"date": "2026-10-30", "items": [{"item": "4.02", "sev": "red"}]},     # same day, time unknown
                     {"date": "2026-10-29", "at": "2026-10-29T12:00:00+00:00", "items": [{"item": "2.02", "sev": "info"}]},
                     {"date": "2026-09-01", "at": "2026-09-01T12:00:00+00:00", "items": [{"item": "3.01", "sev": "red"}]}],
         "val": {"growth": 0.3, "op_margin": 0.9, "fcf_yield": 0.05, "dilution": None},
         "feat": {"val_date": LAST, "val_filed": "2026-08-01", "stmt_filed": "2026-08-01"}}
    x.update(kw)
    return x


def test_news_filing_and_statement_signals_use_only_what_was_known_at_the_close():
    f = ml.context_features(_ctx(), CUT, LAST)
    assert f["news"] == pytest.approx(math.log1p(12)) and f["tone"] == 0.25     # the last check before the close
    assert f["ev8k"] == 1.0 and f["flag"] == 1.0                    # 1.01 counts, 2.02 doesn't; the old 3.01 flags
    assert f["growth"] == 0.3 and f["margin"] == 0.6 and f["fcfy"] == 0.05 and "dilution" not in f
    # a check more than 36 hours before the close is too old; only checks after it: blank, not zero
    stale = ml.context_features(_ctx(news_sigs=[["2026-10-28T10:00:00+00:00", 5, 0.1, 0],
                                                ["2026-10-31T10:00:00+00:00", 5, 0.1, 0]]), CUT, LAST)
    assert "news" not in stale and "tone" not in stale
    late = ml.context_features(_ctx(feat={"val_date": LAST, "val_filed": LAST, "stmt_filed": LAST}), CUT, LAST)
    assert "growth" not in late                                       # filed on the anchor day: not known at its close
    undated = ml.context_features(_ctx(feat={"val_date": LAST, "val_filed": "2026-08-01"}), CUT, LAST)
    assert "growth" not in undated                                    # the facts' own filing date unknown: blank
    full = ml.context_features(_ctx(news_sigs=[["2026-10-30T18:30:00+00:00", 180, 0.1, 1]]), CUT, LAST)
    assert full["news"] == pytest.approx(math.log1p(250))             # a full answer counts as at least 250
    assert ml.context_features(None, CUT, LAST) == {} and set(ml.SIGNAL_FEATS) >= {"news", "tone", "ev8k", "growth"}


def test_an_8k_from_the_sec_list_is_read_as_new_york_time():
    # the submissions list writes 4:31pm ET as "T16:31:00.000Z": after a 4pm close, so not known at it
    sec = [{"date": LAST, "at": "2026-10-30T16:31:00.000Z", "items": [{"item": "4.02", "sev": "red"}]}]
    f = ml.context_features(_ctx(filings=sec), CUT, LAST)
    assert f["ev8k"] == 0.0 and f["flag"] == 0.0
    early = [{"date": LAST, "at": "2026-10-30T09:05:00.000Z", "items": [{"item": "4.02", "sev": "red"}]}]
    assert ml.context_features(_ctx(filings=early), CUT, LAST)["flag"] == 1.0
    assert ml._filed_ts("2026-01-15T16:31:05.000Z").isoformat() == "2026-01-15T21:31:05+00:00"   # EST in winter
    assert ml._filed_ts("2026-10-30T16:05:12-04:00").isoformat() == "2026-10-30T16:05:12-04:00"   # a real offset


def test_filing_signals_are_blank_without_a_recent_filing_record():
    assert "ev8k" not in ml.context_features(_ctx(filings_at=None), CUT, LAST)              # no CIK, say
    assert "flag" not in ml.context_features(_ctx(filings_at="2026-10-01T10:00:00+00:00"), CUT, LAST)
    # a frequent filer's record (the SEC list keeps the newest 30) that stops inside the 90 days: no red flag found
    # means "unknown", not "none"; one found still counts
    quiet = [{"date": "2026-10-26", "at": "2026-10-26T12:00:00+00:00", "items": [{"item": "8.01", "sev": "info"}]}]
    cut_short = ml.context_features(_ctx(filings_feat=quiet, filings_from="2026-09-15"), CUT, LAST)
    assert "flag" not in cut_short and cut_short["ev8k"] == 1.0
    red = quiet + [{"date": "2026-09-20", "at": "2026-09-20T12:00:00+00:00", "items": [{"item": "1.03", "sev": "red"}]}]
    assert ml.context_features(_ctx(filings_feat=red, filings_from="2026-09-15"), CUT, LAST)["flag"] == 1.0
    assert ml.context_features(_ctx(filings_feat=quiet, filings_from="2026-06-01"), CUT, LAST)["flag"] == 0.0


def test_statement_signals_are_dated_by_the_newest_filing_they_include():
    from pipeline.valuation import _latest_filed
    js = {"facts": {"us-gaap": {"Revenues": {"units": {"USD": [{"filed": "2026-05-01"}, {"filed": "2026-10-30"}]}},
                                "NetIncomeLoss": {"units": {"USD": [{"filed": "2026-08-01"}]}}},
                    "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [{"filed": "2026-07-01"}]}}}}}
    assert _latest_filed(js) == "2026-10-30" and _latest_filed({}) is None


def test_headline_tone_and_each_news_check_summary():
    assert news.headline_tone("Nvidia beats estimates and raises guidance") == 1.0
    assert news.headline_tone("Shares plunge after lawsuit, analysts downgrade") == -1.0
    assert news.headline_tone("Company holds annual meeting") == 0.0
    text = ('{"articles": [{"title": "Acme surges on record sales", "seendate": "20261030T143000Z", "url": "https://a.com/1"},'
            '{"title": "Acme surges on record sales", "seendate": "20261030T150000Z", "url": "https://b.com/1"},'
            '{"title": "Acme misses estimates", "seendate": "20261029T090000Z", "url": "https://c.com/1"}]}')
    assert news.gdelt_signal(text) == [2, 0.0, 0]                   # two distinct headlines, +1 and -1, not full
    assert news.gdelt_signal(text, max_records=3) == [2, 0.0, 1]    # GDELT's answer held as many as it can
    assert news.gdelt_signal("") is None and news.gdelt_signal("rate limit exceeded") is None
    assert news.gdelt_signal("{}") == [0, 0.0, 0]


# ---------------------------------------------------------------- the fused model

def test_the_fused_model_averages_the_best_three_and_waits_for_its_months():
    from tests.test_forecast import _tilt_rows
    L = {"graded": {"quant": {"21": _tilt_rows(4000, 0.03, seed=5, n_dates=380)}}}
    res = ml.compete(L)
    assert "fusion" in res["candidates"] and len(res["fusion"]) == ml.FUSE_K and set(res["fusion"]) <= set(ml.AVAILABLE)
    assert res["fusion_months"] <= res["test_months"]
    if res["fusion_months"] < ml.FUSION_MIN_MONTHS:
        assert res["model"] != "fusion" and "fusion" not in res["picks"]
    tw = {**res, "on": True, "accepted": True, "model": "fusion"}
    sm = ml.SignalModel.build(tw, L)
    assert sm is not None and sm.fitted.parts["bases"] == [b for b in ml.AVAILABLE if b in res["fusion"]]
    assert not ml.current({**tw, "fusion": ["ridge", "xgboost"]}) and not ml.current({**tw, "fusion": []})
    assert not ml.current({**tw, "feats": ml.FEATS[:-1]})           # a decision on other signals isn't applied


def test_the_fused_choice_uses_only_months_whose_outcomes_were_in():
    gains = {c: [0.0] * 6 for c in ml.AVAILABLE}
    gains["catboost"][:3] = [0.05, 0.05, 0.05]
    gains["ridge"] = [-0.01, -0.01, -0.01, 0.09, 0.09, 0.09]       # poor at first, strong once later months are in
    assert "catboost" in ml._fuse_set(gains, 3) and "ridge" not in ml._fuse_set(gains, 3)
    assert ml._fuse_set(gains, 2) == ml.AVAILABLE                    # under three months known: all of them
    assert "ridge" in ml._fuse_set(gains, 6)


def test_the_tested_choices_follow_the_same_nine_month_rule_as_putting_it_in_use():
    k = 16
    tested = [f"2025-{m:02d}" if m <= 12 else f"2026-{m - 12:02d}" for m in range(1, k + 1)]
    known_by = [f"{t}-28" for t in tested]                           # each month's outcomes in by its end
    gains = {c: [0.001] * k for c in ml.CANDIDATES}
    gains["fusion"] = [0.02] * k                                     # the fused model leads from the start
    fused = [i >= 3 for i in range(k)]                               # a real fusion from the fourth month
    picks = ml._policy(gains, tested, known_by, fused)
    first = picks.index("fusion")
    assert sum(fused[:first]) >= ml.FUSION_MIN_MONTHS and "fusion" not in picks[:first]
    assert picks[:3] == ["average"] * 3


def test_the_algorithm_put_in_use_must_pass_on_its_own_record(monkeypatch):
    # the month-by-month choices can pass while the one that would be put in use doesn't: then nothing is applied
    from tests.test_forecast import _tilt_rows
    L = {"graded": {"quant": {"21": _tilt_rows(4000, 0.03, seed=5, n_dates=380)}}}
    real = ml._mean_t
    calls = {"n": 0}

    def weak_own(g):
        calls["n"] += 1
        m, t = real(g)
        return (m, 0.5) if calls["n"] == len(ml.CANDIDATES) + 2 else (m, t)   # the call for the installed one
    monkeypatch.setattr(ml, "_mean_t", weak_own)
    res = ml.compete(L)
    assert not res.get("accepted") and "on its own record" in res["status"]


def test_a_fused_model_missing_a_member_is_a_failed_build(monkeypatch):
    import catboost

    class Broken:
        def __init__(self, **kw):
            pass

        def fit(self, *a, **k):
            raise RuntimeError("no CPU support")
    monkeypatch.setattr(catboost, "CatBoostRegressor", Broken)
    rng = np.random.default_rng(1)
    X = rng.normal(size=(600, len(ml.FEATS)))
    y = rng.normal(0, 0.05, 600)
    with pytest.raises(RuntimeError):
        ml.fit("fusion", X, y, periods=12, bases=["ridge", "trees", "catboost"])
    assert ml.fit("fusion", X, y, periods=12, bases=["ridge", "trees", "huber"]).parts["bases"] == ["ridge", "huber", "trees"]
    from tests.test_forecast import _tilt_rows
    with pytest.raises(RuntimeError):                                 # the weekly test stops; it's retried next run
        ml.compete({"graded": {"quant": {"21": _tilt_rows(4000, 0.03, seed=5, n_dates=380)}}})


def test_cached_forecasts_ignore_the_would_be_trio_unless_the_fused_model_is_in_use():
    tw = {"on": True, "accepted": True, "model": "ridge", "fv": ml.FEAT_V, "hp": [ml.HP_V, ml.CANDIDATES], "feats": ml.FEATS,
          "fusion": ["ridge", "huber", "trees"]}
    assert ml.version(tw, None) == ml.version({**tw, "fusion": ["ridge", "huber", "spline"]}, None)
    fz = {**tw, "model": "fusion"}
    assert ml.version(fz, None) != ml.version({**fz, "fusion": ["ridge", "huber", "spline"]}, None)


# ---------------------------------------------------------------- the stock's own record of misses

def test_a_stocks_own_record_of_misses_counts_only_once_graded_and_is_shrunk():
    h = 21
    hist = [{"t": "A", "made": "2026-01-05", "in80r": 0}, {"t": "A", "made": "2026-01-06", "in80r": 0},
            {"t": "B", "made": "2026-01-05", "in80r": 1}]
    early = ml.miss_rates([{"t": "A", "made": "2026-01-20"}], hist, h)[0]          # outcomes not in yet
    later = ml.miss_rates([{"t": "A", "made": "2026-03-02"}, {"t": "B", "made": "2026-03-02"},
                           {"t": "Z", "made": "2026-03-02"}], hist, h)
    assert early == ml.MISS_PRIOR
    assert later[0] == pytest.approx((2 + 0.2 * 10) / 12) and later[1] == pytest.approx((0 + 2) / 11)
    assert later[2] == ml.MISS_PRIOR
    assert ml.miss_now(hist, h, dt.date(2026, 3, 2))["miss"]["A"] == pytest.approx(later[0])
    # a stock whose prices skipped sessions is graded later than the calendar says: known only from its recorded end
    late = [{"t": "A", "made": "2026-01-05", "in80r": 0, "end": "2026-03-10"}]
    assert ml.miss_rates([{"t": "A", "made": "2026-03-02"}], late, h)[0] == ml.MISS_PRIOR


def test_misses_caused_by_earnings_are_counted_per_stock():
    hist = [{"t": "A", "made": "2026-01-05", "in80r": 0, "in80": 0, "why": {"c": "earnings"}},
            {"t": "A", "made": "2026-01-06", "in80r": 0, "in80": 0, "why": {"c": "market"}},
            {"t": "B", "made": "2026-01-05", "in80r": 1, "in80": 1}]
    e = ml.miss_rates([{"t": "A", "made": "2026-03-02"}, {"t": "B", "made": "2026-03-02"}], hist, 21, ml._earn_miss, 0.0)
    assert e[0] == pytest.approx(1 / 12) and e[1] == 0.0
    # shrunk toward zero: a stock with none reads 0 however long its record (it can't stand in for time)
    many = [{"t": "C", "made": f"2025-{m:02d}-05", "in80r": 1, "in80": 1} for m in range(1, 13)]
    assert ml.miss_rates([{"t": "C", "made": "2026-03-02"}], many, 21, ml._earn_miss, 0.0)[0] == 0.0
    assert ml.miss_now(hist, 21, dt.date(2026, 3, 2))["emiss"]["A"] == pytest.approx(1 / 12)


def test_the_range_learner_widens_the_stocks_that_keep_missing():
    rng = np.random.default_rng(11)
    base = dt.date(2024, 1, 2)
    rows = []
    for d in range(420):
        made = (base + dt.timedelta(days=d * 7 // 5)).isoformat()
        for i in range(40):
            f = {k: float(rng.normal()) for k in ml.PRICE_FEATS}
            f.update(v=ml.FEAT_V, ein=0)
            k = 1.6 if i < 8 else 0.85                         # eight stocks are much wilder than their ranges say
            er = float(rng.normal(0, 0.06 / 1.2816 * k))
            rows.append({"t": f"S{i}", "made": made, "f": f, "er": er, "w10": 0.06, "w90": 0.06,
                         "in80r": int(abs(er) <= 0.06), "in80": int(abs(er) <= 0.06)})
    L = {"graded": {"quant": {"21": rows}}}
    from pipeline import learn
    cal = learn.calibration(L)
    rl = ml.range_learner(L, cal)
    assert rl["1m"]["status"] == "applied" and rl["1m"]["v"] == ml.RANGE_V
    rm = ml.RangeModel.build(rl, cal, L, today=dt.date(2026, 1, 1))
    fc = {"horizons": {"1m": {"date": "2026-01-30"}}, "events": []}
    wild = rm.calib_for({"v": ml.FEAT_V}, fc, cal, t="S1")["1m"]["scale"]
    tame = rm.calib_for({"v": ml.FEAT_V}, fc, cal, t="S30")["1m"]["scale"]
    assert wild > tame * 1.3
    # a decision made by an older version of the range learner isn't applied, and doesn't count as "already on"
    old = {k: {**v, "v": ml.RANGE_V - 1} for k, v in rl.items()}
    assert ml.RangeModel.build(old, cal, L) is None
    assert ml._range_status(0.002, {"status": "applied"}, 0.005, 0.0) == "held"
    assert ml._range_status(0.002, {"status": "applied", "v": ml.RANGE_V}, 0.005, 0.0) == "applied"
