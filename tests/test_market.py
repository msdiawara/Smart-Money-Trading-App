import datetime as dt

from pipeline.prices import MarketMatrix, PriceStore
from pipeline import risk


def test_matrix_roundtrip(tmp_path):
    m = MarketMatrix(tmp_path / "m.npz")
    m.add_day(dt.date(2026, 10, 28), [("AAA", 10.0, 100), ("SPY", 500.0, 1e6)])
    m.add_day(dt.date(2026, 10, 26), [("AAA", 9.0, 90)])           # earlier day inserted in order
    m.add_day(dt.date(2026, 10, 27), [])                            # holiday-like empty day
    m.add_day(dt.date(2026, 10, 29), [("BBB", 3.0, 5), ("AAA", 11.0, 110), ("SPY", 505.0, 1e6)])
    s = m.series("AAA")
    assert [dt.date.fromordinal(d).day for d in s["dates"]] == [26, 28, 29] and s["c"] == [9.0, 10.0, 11.0]
    assert m.series("BBB")["c"] == [3.0] and m.series("ZZZ") is None
    assert dt.date(2026, 10, 27).toordinal() in m.checked
    m.save()
    m2 = MarketMatrix(tmp_path / "m.npz")
    assert m2.series("AAA")["c"] == [9.0, 10.0, 11.0] and m2.close.shape == (3, 3)
    store = PriceStore(tmp_path)
    assert m2.export(["AAA", "BBB", "NOPE"], store) == 2 and store.series("AAA")["src"] == "massive"


def test_risk_levels_from_prices(tmp_path):
    import math
    store = PriceStore(tmp_path)
    base = dt.date(2025, 10, 1)
    days = [base + dt.timedelta(days=i) for i in range(400) if (base + dt.timedelta(days=i)).weekday() < 5][:260]
    calm = [(d, 100 * math.exp(0.0003 * i + 0.004 * math.sin(i)), 5e6) for i, d in enumerate(days)]
    wild = [(d, 20 * math.exp(-0.002 * i + 0.06 * math.sin(i * 1.3)), 2e5) for i, d in enumerate(days)]
    spy = [(d, 500 * math.exp(0.0003 * i + 0.003 * math.sin(i)), 1e8) for i, d in enumerate(days)]
    store.put("CALM", calm, "t"); store.put("WILD", wild, "t"); store.put("SPY", spy, "t")
    mc, mw = risk.metrics(store, "CALM"), risk.metrics(store, "WILD")
    lc = risk.risk_level(mc, 500e9, True)
    lw = risk.risk_level(mw, 150e6, False)
    assert lc["level"] <= 2 and lw["level"] >= 4
    assert any("Lost money" in f for f in lw["factors"]) and any("Thinly traded" in f for f in lw["factors"])
