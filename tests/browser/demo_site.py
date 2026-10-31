"""Build a demo copy of the app with sample data from the fake internet + fake Claude API (no network), for the
browser test and for screenshots. Usage: python tests/browser/demo_site.py /tmp/site"""
import os
import pathlib
import shutil
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.update(SMR_FAKE_NOW="2026-10-30T20:30:00+00:00", SEC_CONTACT_EMAIL="demo@example.com", NTFY_TOPIC="demo-topic",
                  MASSIVE_API_KEY="demo", ALPACA_KEY_ID="k", ALPACA_SECRET_KEY="s", ALPHAVANTAGE_API_KEY="av",
                  ANTHROPIC_API_KEY="sk-ant-demo")
time.sleep = lambda s: None
from pipeline import analyst_main, main  # noqa: E402
from pipeline.settings import Settings  # noqa: E402
import tests.fake_web as fw  # noqa: E402
from tests.fake_web import FakeWeb  # noqa: E402


def demo_chart(sym, start_price, drift, vol, n=560):
    """Demo-only price history: two-plus years of a fat-tailed random walk with volatility clustering (the unit
    tests keep the shorter deterministic series), so the demo's forecasts and backtest look like real markets."""
    import datetime as _dt
    import math as _m
    import random as _r
    rnd = _r.Random(sum(ord(ch) for ch in sym) * 7919)
    base = _dt.datetime(2026, 10, 30, 20, 0, tzinfo=_dt.timezone.utc)
    days, d = [], base
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d -= _dt.timedelta(days=1)
    days.reverse()
    p, v, ts, closes = start_price, vol, [], []
    lr = vol * 1.1
    for day in days:
        z = rnd.gauss(0, 1) * (1.0 if rnd.random() > 0.04 else 2.6)
        v = _m.sqrt(0.86 * v * v + 0.07 * (z * v) ** 2 + 0.054 * lr * lr)
        p *= _m.exp(drift * 0.6 + v * z)
        ts.append(int(day.timestamp()))
        closes.append(round(p, 4))
    return {"chart": {"result": [{"meta": {"regularMarketPrice": closes[-1], "regularMarketTime": ts[-1]},
                                  "timestamp": ts, "indicators": {"quote": [{"close": closes, "volume": [2_000_000] * n}],
                                                                   "adjclose": [{"adjclose": closes}]}}], "error": None}}


fw.chart = demo_chart
fw._CLOSES.clear()
from tests.test_analyst import FakeClaude  # noqa: E402

site = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/site")
shutil.rmtree(site, ignore_errors=True)
shutil.copytree(ROOT / "web", site)
cache = site.parent / (site.name + "-cache")
shutil.rmtree(cache, ignore_errors=True)
s = Settings.load(cache_dir=cache, out_dir=site / "data")
s.config["refresh"]["run_budget_minutes"] = 60
s.config["funds"]["list"] = [{"cik": "0001067983", "name": "Berkshire Hathaway", "manager": "Greg Abel", "style": "concentrated",
                              "expect": "BERKSHIRE HATHAWAY"}]
s.config["analyst"] = {"deep_min_score": 60, "standard_min_score": 30}
s.config["alerts"]["watch_tickers"] = ["NVDA"]          # so the demo has a Claude price forecast to show
s.config.setdefault("bubble", {})["core"] = ["NVDA", "NEWC", "GOOG", "MSFT", "AAPL", "BTC1"]
fake = FakeWeb()
main.run(force={"all"}, transport=fake, settings=s)
analyst_main.run(s, transport=FakeClaude())
fake.ark_version = 2
os.environ["SMR_FAKE_NOW"] = "2026-11-02T13:30:00+00:00"
main.run(force={"etf_daily"}, transport=fake, settings=s)
# the browser test talks to a local mock of the Claude API; allow it in this copy's security policy only
idx = site / "index.html"
idx.write_text(idx.read_text().replace("connect-src 'self' https://api.anthropic.com", "connect-src 'self' https://api.anthropic.com http://127.0.0.1:8899"))
print("demo site written to", site)
