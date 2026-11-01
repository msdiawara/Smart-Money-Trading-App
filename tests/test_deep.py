"""The deep price history (pipeline/deep.py): Alpaca's free plan back to 2016, built once in resumable stages, and
the backtest reading the weeks before the Massive matrix from it."""
import datetime as dt
import math
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest

from pipeline import deep as D
from pipeline import fledger, insight, quant
from pipeline.http import HttpError
from tests.test_forecast import END, _Matrix, gbm, trading_dates


class FakeAlpaca:
    """Answers the asset lists and the multi-symbol bars endpoint from made-up prices."""

    def __init__(self, days, prices, volumes, active, inactive, paper_ok=True, bad=(), delisted=None, asof_only=None):
        self.days, self.prices, self.volumes = days, prices, volumes
        self.active, self.inactive, self.paper_ok, self.bad = active, inactive, paper_ok, set(bad)
        self.delisted = delisted or {}          # Massive's reference list: type → rows, newest delisting first
        self.asof_only = asof_only or {}        # tickers whose company Alpaca finds only as of that date
        self.calls = []

    def _massive(self, url, q):
        assert q["apiKey"] == "mk" and q["active"] == "false" and q["sort"] == "delisted_utc" and q["order"] == "desc"
        rows = self.delisted.get(q["type"], [])
        start = int(q.get("cursor") or 0)
        page = rows[start:start + 2]                      # two a page, so the test pages
        nxt = start + len(page)
        return {"results": page, "next_url": (f"https://api.massive.com/v3/reference/tickers?cursor={nxt}&type={q['type']}"
                                              f"&market=stocks&active=false&sort=delisted_utc&order=desc&limit=1000")
                if nxt < len(rows) else None}

    def get_json(self, url, headers=None, **kw):
        self.calls.append(url)
        u = urlparse(url)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if "massive" in u.netloc:
            return self._massive(url, q)
        assert headers["APCA-API-KEY-ID"] == "k" and headers["APCA-API-SECRET-KEY"] == "s"
        if u.path.endswith("/assets"):
            if u.netloc.startswith("paper") and not self.paper_ok:
                raise HttpError(401, url)
            return self.active if q["status"] == "active" else self.inactive
        syms = q["symbols"].split(",")
        assert q["feed"] == "sip" and q["adjustment"] == "split"
        if self.bad & set(syms):
            raise HttpError(422, url, "invalid symbol")
        lo, hi = q["start"][:10], q["end"][:10]
        rows = []
        for s in syms:
            if s in self.asof_only and q.get("asof") != self.asof_only[s]:
                continue                                    # asked as of the wrong date: another company, or none
            for d, c, v in zip(self.days, self.prices.get(s, []), self.volumes.get(s, [])):
                if lo <= d.isoformat() <= hi and np.isfinite(c):
                    rows.append((s, d, float(c), float(v)))
        if q["timeframe"] == "1Month":                      # last close and summed volume of each month
            agg = {}
            for s, d, c, v in rows:
                k = (s, d.year, d.month)
                agg[k] = (d.replace(day=1), c, agg.get(k, (None, 0, 0))[2] + v)
            rows = [(s, d0, c, v) for (s, _, _), (d0, c, v) in sorted(agg.items())]
        start = int(q.get("page_token") or 0)
        page = rows[start:start + int(q["limit"])]
        out = {}
        for s, d, c, v in page:
            out.setdefault(s, []).append({"t": f"{d.isoformat()}T04:00:00Z", "c": c, "v": v, "o": c, "h": c, "l": c})
        nxt = start + len(page)
        return {"bars": out, "next_page_token": str(nxt) if nxt < len(rows) else None}


def _market(n_days=700, n=8):
    days = [dt.date.fromordinal(d) for d in trading_dates(END, n_days)]
    prices = {"SPY": gbm(n_days, 0.01, seed=1, start=400)}
    vols = {"SPY": [8e7] * n_days}
    for k in range(n):
        s = f"S{chr(65 + k)}"
        prices[s] = gbm(n_days, 0.02, seed=10 + k, start=30 + 5 * k)
        vols[s] = [1e6 * (k + 1)] * n_days                 # SH trades the most, SA the least
    prices["GONE"] = np.r_[gbm(300, 0.03, seed=99, start=20), [np.nan] * (n_days - 300)]   # delisted
    vols["GONE"] = [5e7] * n_days
    active = [{"symbol": s, "exchange": "NYSE"} for s in prices if s != "GONE"] + [
        {"symbol": "OTCX", "exchange": "OTC"}, {"symbol": "BRK.B", "exchange": "NYSE"}, {"symbol": "WS/U", "exchange": "NYSE"}]
    inactive = [{"symbol": "GONE", "exchange": "NASDAQ"}]
    return days, prices, vols, active, inactive


MIN_DV = 2.5e9          # in _market(), SH and up trade more than this in their busiest month; SA never does


def _deep(tmp_path, start, min_dv=MIN_DV):
    return D.DeepHistory(tmp_path, start=start, min_month_dv=min_dv, min_stocks=3)


def _build(tmp_path, fake, min_dv=MIN_DV, seconds=600.0, start="2024-01-01", **kw):
    h = _deep(tmp_path, start, min_dv)
    n, note = h.step(fake, "k", "s", seconds=seconds, **kw)
    return h, note


def _chosen(tmp_path, start, min_dv=MIN_DV):
    return {s for s, _ in _deep(tmp_path, start, min_dv).meta.get("chosen") or []}


def test_it_lists_listed_and_delisted_stocks_ranks_them_and_fetches_the_liquid_ones(tmp_path):
    days, prices, vols, active, inactive = _market()
    fake = FakeAlpaca(days, prices, vols, active, inactive)
    h, note = _build(tmp_path, fake, start=days[0].isoformat())
    assert h.done and note.startswith("complete")
    uni = _chosen(tmp_path, days[0].isoformat())
    assert "GONE" in uni and "SPY" in uni and "SH" in uni and "SA" not in uni     # the busy ones, delisted too
    assert "OTCX" not in uni and "WS/U" not in uni
    again = _deep(tmp_path, days[0].isoformat())       # read back from disk
    assert again.done and not again._loaded                    # checked without reading the arrays
    assert again.dates == h.dates and again.tickers == h.tickers
    j = again.col["SH"]
    assert np.allclose(again.close[:, j], np.asarray(prices["SH"], dtype=np.float32), equal_nan=True)
    g = again.close[:, again.col["GONE"]]
    assert np.isfinite(g[:300]).all() and np.isnan(g[300:]).all()
    P = quant.Panel(again)
    assert P.ok() and P.T == len(days)


def test_it_resumes_where_it_stopped_and_tries_live_keys_when_paper_ones_are_refused(tmp_path):
    days, prices, vols, active, inactive = _market()
    fake = FakeAlpaca(days, prices, vols, active, inactive, paper_ok=False)
    h, note = _build(tmp_path, fake, seconds=0.0, start=days[0].isoformat())
    assert not h.done and h.meta["stage"] == "rank"
    assert any("//api.alpaca.markets" in u for u in fake.calls)
    for _ in range(40):
        h = _deep(tmp_path, days[0].isoformat())
        if h.done:
            break
        h.step(fake, "k", "s", seconds=0.0)                 # one batch a run
    assert h.done and len(h.tickers) >= 5


def test_a_symbol_alpaca_rejects_costs_only_itself_and_bad_keys_pause_it(tmp_path):
    days, prices, vols, active, inactive = _market()
    fake = FakeAlpaca(days, prices, vols, active, inactive, bad={"SC"})
    h, _ = _build(tmp_path, fake, min_dv=0, start=days[0].isoformat())
    assert h.done and "SC" not in [t for t in h.tickers if np.isfinite(h.close[:, h.col[t]]).any()]
    assert np.isfinite(h.close[:, h.col["SD"]]).all()
    t2 = tmp_path / "other"
    t2.mkdir()
    fake2 = FakeAlpaca(days, prices, vols, active, inactive, paper_ok=False)
    fake2.get_json = lambda url, headers=None, **kw: (_ for _ in ()).throw(HttpError(403, url))
    with pytest.raises(RuntimeError, match="Alpaca rejected the keys"):        # shown as a failure on the Data tab
        _build(t2, fake2, start=days[0].isoformat())
    h2 = _deep(t2, days[0].isoformat())
    assert not h2.done and h2.note().startswith("paused: Alpaca rejected the keys")
    # an error that isn't about one symbol stops the run (it resumes next time) instead of splitting the batch
    t3 = tmp_path / "third"
    t3.mkdir()
    fake3 = FakeAlpaca(days, prices, vols, active, inactive)
    real = fake3.get_json

    def flaky(url, headers=None, **kw):
        if "/bars" in url and "1Month" in url:
            raise HttpError(400, url, '{"message":"invalid page token"}')
        return real(url, headers, **kw)
    fake3.get_json = flaky
    with pytest.raises(RuntimeError, match="HTTP 400"):
        _build(t3, fake3, start=days[0].isoformat())
    assert sum("1Month" in u for u in fake3.calls) == 0      # the failing request isn't counted: no splitting
    h3 = _deep(t3, days[0].isoformat())
    assert h3.meta["stage"] == "rank" and h3.meta["rank_i"] == 0


def test_delisted_stocks_come_from_massive_and_are_asked_for_as_of_before_they_went(tmp_path):
    days, prices, vols, active, inactive = _market()
    prices["OLDCO"] = np.r_[gbm(400, 0.02, seed=7, start=40), [np.nan] * (len(days) - 400)]
    vols["OLDCO"] = [9e7] * len(days)                      # the most traded of all, while it traded
    gone = days[400].isoformat()
    delisted = {"CS": [{"ticker": "OLDCO", "delisted_utc": f"{gone}T00:00:00Z", "cik": "123"},
                       {"ticker": "SB", "delisted_utc": f"{gone}T00:00:00Z", "cik": "9"},          # ticker reused now
                       {"ticker": "RENM", "delisted_utc": f"{gone}T00:00:00Z", "cik": "555"},      # renamed, still listed
                       {"ticker": "ANCIENT", "delisted_utc": "2001-05-01T00:00:00Z", "cik": "1"}],
                "ETF": [{"ticker": "OLDETF", "delisted_utc": f"{gone}T00:00:00Z"}]}
    asof = D._asof_before(gone)
    fake = FakeAlpaca(days, prices, vols, active, inactive, delisted=delisted, asof_only={"OLDCO": asof})
    start = days[0].isoformat()
    h, note = _build(tmp_path, fake, min_dv=0, start=start, massive_key="mk", listed_ciks={"0000000555"})
    assert h.done
    chosen = dict(_deep(tmp_path, start, 0).meta["chosen"])
    assert chosen.get("OLDCO") == asof and chosen.get("SB", "") == ""
    assert "RENM" not in chosen and "ANCIENT" not in chosen
    j = h.col["OLDCO"]
    assert np.isfinite(h.close[:400, j]).all() and np.isnan(h.close[400:, j]).all()
    assert any("asof=" + asof in u for u in fake.calls)
    raw = (tmp_path / "deep.json.gz").read_bytes()
    import gzip
    assert b"mk" not in gzip.decompress(raw)                 # the Massive key is never stored


def test_a_missing_or_foreign_matrix_is_fetched_again_and_an_empty_build_is_not_finished(tmp_path):
    days, prices, vols, active, inactive = _market()
    start = days[0].isoformat()
    h, _ = _build(tmp_path, FakeAlpaca(days, prices, vols, active, inactive), start=start)
    assert h.done
    (tmp_path / "deep.npz").unlink()
    h2 = _deep(tmp_path, start)
    assert not h2.done and h2.meta["stage"] == "daily" and h2.meta["daily_i"] == 0
    h2.step(FakeAlpaca(days, prices, vols, active, inactive), "k", "s", seconds=600.0)
    assert _deep(tmp_path, start).done
    # a matrix from another build (an older run, say) isn't used
    from pipeline.util import read_json, write_json
    write_json(tmp_path / "deep.json.gz", {**read_json(tmp_path / "deep.json.gz"), "build": "other"})
    assert not _deep(tmp_path, start).done
    # nearly nothing came back: not marked complete, tried again later
    t2 = tmp_path / "empty"
    t2.mkdir()
    with pytest.raises(RuntimeError, match="stocks came back"):
        _deep(t2, start).step(
            _ThinDaily(days, prices, vols, active, inactive), "k", "s", seconds=600.0)
    h3 = _deep(t2, start)
    assert not h3.done and h3.meta["daily_i"] == 0 and h3.meta.get("retry_after")


class _ThinDaily(FakeAlpaca):
    """Monthly bars as usual, but daily bars only for the S&P 500 ETF."""

    def get_json(self, url, headers=None, **kw):
        out = super().get_json(url, headers, **kw)
        if "timeframe=1Day" in url and "symbols=SPY&" not in url:
            return {"bars": {}, "next_page_token": None}
        return out


def test_the_backtest_reaches_back_through_the_deep_history(tmp_path, monkeypatch):
    monkeypatch.setenv("SMR_FAKE_NOW", "2026-10-31T13:00:00+00:00")
    days, prices, vols, active, inactive = _market(n_days=900, n=6)
    fake = FakeAlpaca(days, prices, vols, active, inactive)
    h, _ = _build(tmp_path, fake, min_dv=0, start=days[0].isoformat())
    deep = quant.Panel(h)
    # the Massive matrix: only the last 520 days
    recent = [d.toordinal() for d in days[-520:]]
    massive = quant.Panel(_Matrix({s: np.asarray(p)[-520:] for s, p in prices.items() if s != "GONE"}, recent))
    heavy = tmp_path / "heavy"
    heavy.mkdir()
    cfg = {"forecast": {"backtest_stocks": 6, "backtest_anchors": 30, "backtest_deep_anchors": 200,
                        "backtest_paths": 200, "backtest_seconds_per_run": 900}}
    settings = SimpleNamespace(c=lambda *k, default=None: cfg.get(k[0], default) if len(k) == 1 else default,
                               mark_heavy_dirty=lambda: None)
    for _ in range(4):
        insight.job_backtest(settings, {}, massive, SimpleNamespace(left=lambda: 2000), heavy, None, {}, deep)
    L = fledger.load(heavy)
    made = sorted({r["made"] for r in L["graded"]["quant_bt"]["21"]})
    first_m = dt.date.fromordinal(recent[260]).isoformat()
    assert made[0] < first_m <= made[-1]                  # weeks before the Massive matrix's first year, and after
    old = [r for r in L["graded"]["quant_bt"]["21"] if r["made"] < first_m]
    assert old and all(r["f"]["v"] == insight.ml.FEAT_V for r in old if r["t"] != "SPY")
    assert any(r["t"] == "GONE" for r in L["graded"]["quant_bt"]["21"])    # a stock that later delisted
    weeks = [dt.date.fromisoformat(m).isocalendar()[:2] for m in made]
    assert len(set(weeks)) == len(weeks)                                   # one anchor a week
    # no hole between the two histories: the deep weeks run up to the oldest week the Massive matrix gives
    gaps = [(dt.date.fromisoformat(b) - dt.date.fromisoformat(a)).days for a, b in zip(made, made[1:])]
    assert max(gaps) <= 10
    assert L["bt"]["panels"] == ["d", "m"] and not L["bt"]["todo"]


def _old_earnings_proxy(c, v, dates):
    """The loop version this replaced, kept to check the fast one gives the same days."""
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
    picked = []
    for _, k in sorted(cands, reverse=True):
        if all(abs(k - j) >= 40 for j in picked):
            picked.append(k)
    return sorted(dt.date.fromordinal(int(dates[idx[k]])).isoformat() for k in picked)


def test_the_fast_earnings_estimate_matches_the_old_one_and_never_looks_ahead():
    n = 900
    days = np.asarray(trading_dates(END, n), dtype=np.int64)
    for seed in range(6):
        rng = np.random.default_rng(seed)
        c = gbm(n, 0.02, seed=seed, start=30).astype(float)
        v = rng.lognormal(13, 0.3, n)
        for k in rng.choice(np.arange(70, n - 2), 14, replace=False):     # earnings-like days: big move, big volume
            v[k] *= rng.uniform(3, 9)
            c[k:] *= math.exp(rng.choice([-1, 1]) * 0.08)
        c[rng.choice(n, 20, replace=False)] = np.nan                     # missing days
        assert insight.earnings_proxy(c, v, days) == _old_earnings_proxy(c, v, days)
    # a bigger spike after the forecast date doesn't push out a smaller one before it
    c = np.full(n, 50.0)
    v = np.full(n, 1e6)
    c *= np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.01, n)))
    for k, size in ((500, 4.0), (520, 9.0)):
        v[k] *= size
        c[k:] *= 1.1
    d500, d520 = (dt.date.fromordinal(int(days[k])).isoformat() for k in (500, 520))
    assert d500 not in insight.earnings_proxy(c, v, days)                       # seen from the end: one per 40 days
    assert d500 in insight.earnings_proxy(c, v, days, upto=dt.date.fromordinal(int(days[510])).isoformat())


def test_a_ticker_with_a_long_gap_uses_only_the_run_ending_at_the_anchor():
    h = np.r_[np.full(300, 5.0), [np.nan] * 40, np.full(280, 50.0)]
    ok = insight._segment(h)
    assert not ok[:340].any() and ok[340:].all()
    h2 = np.r_[np.full(300, 5.0), [np.nan] * 5, np.full(280, 50.0)]       # a short halt: kept whole
    assert insight._segment(h2).sum() == 580


def test_which_stocks_are_kept_never_depends_on_how_they_did_later(tmp_path):
    # WIN and LOSE trade the same until day 400; then WIN triples and LOSE loses 70%, on the same shares traded.
    # Ranking by later trading would keep WIN and drop LOSE; the dollar-volume net keeps both, and on a date before
    # day 400 the backtest's own (point-in-time) filter sees them alike.
    days, prices, vols, active, inactive = _market()
    n = len(days)
    base = gbm(n, 0.01, seed=3, start=20)
    prices["WIN"] = np.r_[base[:400], base[400:] * np.linspace(1, 3, n - 400)]
    prices["LOSE"] = np.r_[base[:400], base[400:] * np.linspace(1, 0.3, n - 400)]
    vols["WIN"] = vols["LOSE"] = [3e6] * n                        # about $60M a day before day 400
    active += [{"symbol": "WIN", "exchange": "NYSE"}, {"symbol": "LOSE", "exchange": "NYSE"}]
    h, _ = _build(tmp_path, FakeAlpaca(days, prices, vols, active, inactive), min_dv=D.MIN_MONTH_DV,
                  start=days[0].isoformat())
    assert {"WIN", "LOSE"} <= set(h.tickers)
    P = quant.Panel(h)
    uni = set(quant.liquid_universe(P, 380, None, min_price=1.0))
    assert {"WIN", "LOSE"} <= uni
    # the net is wide enough for the backtest's own floor: 20 sessions at $10M a day clear it in at least one month
    import inspect
    floor = inspect.signature(quant.liquid_universe).parameters["min_adv"].default
    assert D.MIN_MONTH_DV <= floor * 20 / 2


def test_a_company_that_went_over_the_counter_is_gone_not_renamed():
    tick = {"by_cik": {"0000000777": {"ticker": "BUSTQ", "exchange": "OTC"},
                       "0000000555": {"ticker": "NEWN", "exchange": "Nasdaq"},
                       "0000000556": {"ticker": "NOEX", "exchange": None}}}
    assert D.listed_ciks(tick) == {"0000000555"}
    assert D.listed_ciks(None) == set()


def test_the_delisted_list_resumes_after_an_error_and_a_stale_page_link(tmp_path):
    days, prices, vols, active, inactive = _market()
    rows = [{"ticker": f"D{chr(65 + k)}", "delisted_utc": f"{days[500 - 20 * k].isoformat()}T00:00:00Z"} for k in range(6)]
    fake = FakeAlpaca(days, prices, vols, active, inactive, delisted={"CS": rows, "ETF": []})
    real = fake._massive
    state = {"n": 0}

    def flaky(url, q):
        state["n"] += 1
        if state["n"] == 2:
            raise HttpError(502, url, "bad gateway")                 # a passing failure: stops this run only
        if state["n"] == 4:
            raise HttpError(400, url, "invalid cursor")              # a stale page link: that type starts again
        return real(url, q)
    fake._massive = flaky
    start = days[0].isoformat()
    with pytest.raises(RuntimeError, match="Massive answered HTTP 502"):
        _build(tmp_path, fake, start=start, massive_key="mk", listed_ciks={"0000000001"})
    h = _deep(tmp_path, start)
    assert h.meta["stage"] == "universe" and h.meta["delisted_q"]["next"]       # it kept its place
    h.step(fake, "k", "s", seconds=600.0, massive_key="mk", listed_ciks={"0000000001"})
    h = _deep(tmp_path, start, 0)
    m = _deep(tmp_path, start).meta
    assert m["stage"] != "universe" and not m.get("delisted_note")
    assert m["n_delisted"] == 6                                     # every one of them, despite both errors


def test_a_network_error_never_stores_the_massive_key(tmp_path):
    import gzip

    import requests
    from pipeline import util
    util.register_secret("mk-secret-123")
    days, prices, vols, active, inactive = _market()
    fake = FakeAlpaca(days, prices, vols, active, inactive)

    def down(url, q):
        raise requests.ConnectionError(f"Max retries exceeded with url: {url}&apiKey=mk-secret-123")
    fake._massive = down
    with pytest.raises(RuntimeError) as e:
        _build(tmp_path, fake, start=days[0].isoformat(), massive_key="mk-secret-123", listed_ciks={"0000000001"})
    assert "mk-secret-123" not in str(e.value)
    assert b"mk-secret-123" not in gzip.decompress((tmp_path / "deep.json.gz").read_bytes())


def test_the_universe_waits_for_the_sec_list_then_leaves_delisted_stocks_out(tmp_path, monkeypatch):
    days, prices, vols, active, inactive = _market()
    rows = [{"ticker": "OLDS", "delisted_utc": f"{days[300].isoformat()}T00:00:00Z", "cik": "42"}]
    etfs = [{"ticker": "OLDE", "delisted_utc": f"{days[300].isoformat()}T00:00:00Z"}]
    fake = FakeAlpaca(days, prices, vols, active, inactive, delisted={"CS": rows, "ETF": etfs})
    start = days[0].isoformat()
    h, note = _build(tmp_path, fake, start=start, massive_key="mk", listed_ciks=set())
    assert h.meta["stage"] == "universe" and "waiting for the SEC ticker list" in note
    later = dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=4)
    monkeypatch.setattr(D, "now_utc", lambda: later)
    h, note = _build(tmp_path, fake, min_dv=MIN_DV, start=start, massive_key="mk", listed_ciks=set())
    m = _deep(tmp_path, start).meta
    assert m["stage"] != "universe" and "left out" in m.get("delisted_note", "")
    assert m["n_delisted"] == 1                                     # the ETF stays (no rename test for those)


def test_a_damaged_matrix_is_marked_for_fetching_again_on_disk(tmp_path):
    days, prices, vols, active, inactive = _market()
    start = days[0].isoformat()
    h, _ = _build(tmp_path, FakeAlpaca(days, prices, vols, active, inactive), start=start)
    meta = _deep(tmp_path, start).meta
    np.savez_compressed(tmp_path / "deep.npz", build=np.asarray(meta["build"]))     # the build id reads, the rest doesn't
    again = _deep(tmp_path, start)
    assert again.done and again.dates == []                  # discovered only when the arrays are read
    assert _deep(tmp_path, start).meta["stage"] == "daily"    # …and saved, so the next run fetches them again


def test_delisted_adrs_and_etns_are_asked_for_and_a_failed_sweep_waits_before_trying_again(tmp_path):
    days, prices, vols, active, inactive = _market()
    fake = FakeAlpaca(days, prices, vols, active, inactive, delisted={})
    start = days[0].isoformat()
    _build(tmp_path, fake, start=start, massive_key="mk", listed_ciks={"0000000001"})
    asked = {parse_qs(urlparse(u).query)["type"][0] for u in fake.calls if "massive" in u}
    assert asked == {"CS", "ADRC", "ETF", "ETN"}
    # a sweep that finds (almost) nothing isn't retried every hour: it waits a day, then longer
    t2 = tmp_path / "empty"
    t2.mkdir()
    blank = FakeAlpaca(days, {"SPY": prices["SPY"]}, {"SPY": vols["SPY"]}, active, inactive)
    with pytest.raises(RuntimeError, match="monthly sweep found prices for only"):
        D.DeepHistory(t2, start=start, min_month_dv=MIN_DV, min_stocks=200).step(blank, "k", "s", seconds=600.0)
    again = D.DeepHistory(t2, start=start, min_month_dv=MIN_DV, min_stocks=200)
    assert again.meta.get("retry_after") and again.meta["stage"] == "rank"
    n_calls = len(blank.calls)
    assert again.step(blank, "k", "s", seconds=600.0)[0] == 0 and len(blank.calls) == n_calls   # not yet
