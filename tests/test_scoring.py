from pipeline.scoring import Ctx, score_event


def ins(action="buy", rank=5, value=500_000, chg=40.0, plan=False, role="CEO"):
    return {"id": "x", "kind": "insider", "ticker": "AAA", "company": "Aaa Inc", "issuer_cik": "1",
            "actor": {"name": "Doe Jane", "id": "cik:1", "role": role, "rank": rank}, "action": action,
            "action_label": "Open-market buy" if action == "buy" else "Open-market sale", "trade_date": "2026-10-20",
            "value": value, "price": 10.0, "shares": value / 10, "holdings_change_pct": chg,
            "details": {"plan_10b5_1": plan, "late": False, "indirect": False}}


def test_buy_beats_sale_and_plan_penalty():
    b, s, p = score_event(ins(), Ctx()), score_event(ins("sell", chg=-3), Ctx()), score_event(ins(plan=True), Ctx())
    assert b["direction"] == "bullish" and s["direction"] == "bearish"
    assert b["score"] > s["score"] and b["score"] - p["score"] == 15
    assert any("10b5-1" in c for c in p["read"]["cons"])


def test_cluster_small_cap_and_track_record_raise_score():
    base = score_event(ins(), Ctx())["score"]
    rich = Ctx(quote=lambda t: {"cap_tier": "small", "price": 9.0, "lo_52w": 8.8, "hi_52w": 20, "ret_3m": -30},
               insider_cluster=lambda e: {"buyers": 3, "sellers": 0},
               insider_record=lambda a: {"n": 6, "win_rate": 0.8},
               insider_routine=lambda e: False)
    e = score_event(ins(), rich)
    assert e["score"] >= base + 25 and e["signal"] == "strong"
    assert any("Cluster buy" in x for x in e["read"]["pros"]) and any("Track record" in x for x in e["read"]["pros"])


def test_routine_trader_and_noise():
    r = score_event(ins(), Ctx(insider_routine=lambda e: True))
    assert any("Routine" in c for c in r["read"]["cons"])
    n = score_event({**ins(), "action": "acquire", "action_label": "Grant/award"}, Ctx())
    assert n["signal"] == "noise" and n["score"] <= 5


def test_congress_and_fund_scores_are_bounded_and_explained():
    c = {"id": "c", "kind": "congress", "ticker": "NVDA", "company": "NVIDIA", "action": "buy",
         "actor": {"name": "Sen. A B", "id": "congress:b-a", "role": "Senate"}, "trade_date": "2026-10-01",
         "lag_days": 5, "value_range": [1_000_001, 5_000_000], "details": {"owner": "Spouse", "asset_type": "Stock"}}
    e = score_event(c, Ctx(is_leader=lambda actor: "Speaker of the House"))
    assert 1 <= e["score"] <= 99 and any("leadership" in p for p in e["read"]["pros"])
    f = {"id": "f", "kind": "fund", "ticker": "XYZ", "company": "Xyz", "action": "new", "value": 4e9,
         "actor": {"name": "Fund", "id": "cik:9", "role": "PM"}, "filed_at": "2026-08-14",
         "details": {"change": "new", "weight": 6.0, "prev_weight": 0, "fund_cik": "9", "period": "2026-06-30",
                     "days_since_quarter_end": 120}}
    e = score_event(f, Ctx(fund_meta=lambda c: {"style": "concentrated", "name": "Fund"}))
    assert e["direction"] == "bullish" and any("High-conviction" in p for p in e["read"]["pros"])
    assert any("days old" in c for c in e["read"]["cons"])
