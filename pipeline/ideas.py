"""Ideas: setups worth watching that aren't a single headline trade.

Three transparent screens, each with its measured track record where the data allows:
1. Smart-money convergence: several independent informed buyers (insiders, funds, activists, Congress) in the
   same stock within ~45 days.
2. Insiders buying into weakness: two or more insiders buying after a sharp fall.
3. Momentum near highs: strong 6-month momentum, still within 5% of the 52-week high, not wildly volatile
   (Jegadeesh & Titman 1993; George & Hwang 2004). Backtested month by month on the stored history.

Claude's daily brief then reads these lists (plus the market backdrop) and writes up the most interesting ones,
with what would trigger them and what would prove them wrong.
"""
from __future__ import annotations

import datetime as dt

from .quant import BENCH, Panel, liquid_universe, momentum_screen, price_features, projection
from .util import now_et, now_utc, parse_date

BULLISH = {("insider", "buy"), ("fund", "new"), ("fund", "add"), ("activist", "stake"), ("congress", "buy"),
           ("etoro", "open"), ("etf", "new"), ("etf", "add")}


def _recent(events: list[dict], days: int) -> list[dict]:
    cutoff = (now_utc() - dt.timedelta(days=days)).date().isoformat()
    return [e for e in events if (e.get("filed_at") or e.get("trade_date") or "")[:10] >= cutoff]


def convergence(events: list[dict], quotes: dict, days: int = 45, min_score: int = 45) -> list[dict]:
    by: dict[str, list] = {}
    for e in _recent(events, days):
        if not e.get("ticker") or (e.get("kind"), e.get("action")) not in BULLISH:
            continue
        if (e.get("score") or 0) < min_score or e.get("direction") == "bearish":
            continue
        if e.get("kind") == "activist" and (e.get("details") or {}).get("schedule") != "13D":
            continue
        by.setdefault(e["ticker"], []).append(e)
    out = []
    for t, evs in by.items():
        actors = {e["actor"].get("id") for e in evs}
        kinds = {e["kind"] for e in evs}
        if len(actors) < 2 or (len(kinds) < 2 and len(actors) < 3):
            continue
        evs.sort(key=lambda e: -(e.get("score") or 0))
        q = quotes.get(t) or {}
        out.append({
            "id": f"conv:{t}", "ticker": t, "screen": "convergence", "company": evs[0].get("company"),
            "title": f"{len(actors)} informed buyers across {', '.join(sorted(kinds))}",
            "why": [f"{e['actor']['name']} ({e.get('action_label')}, signal {e.get('score')}/100)" for e in evs[:5]],
            "signals": [e["id"] for e in evs[:8]], "strength": sum(e.get("score") or 0 for e in evs[:4]),
            "risk_level": q.get("risk_level"), "projection": projection(q.get("price"), q.get("vol_1y")),
        })
    out.sort(key=lambda d: -d["strength"])
    return out[:15]


def insider_weakness(events: list[dict], quotes: dict, panel: Panel | None = None, days: int = 30) -> list[dict]:
    """Two or more insiders bought within 30 days of the latest purchase, and the stock was down 15%+ over the
    3 months before that latest filing. Same rule as the track record shown with it (base rate
    'insider_buy_cluster_weakness'), measured on the filing-day close; without the market history it falls back to
    today's 3-month move and says so."""
    by: dict[str, list] = {}
    for e in _recent(events, days):
        if (e.get("kind") == "insider" and e.get("action") == "buy" and e.get("ticker")
                and (e.get("value") or 0) >= 1000 and parse_date(e.get("filed_at"))):
            by.setdefault(e["ticker"], []).append(e)
    out = []
    for t, evs in by.items():
        q = quotes.get(t) or {}
        f_last = max(parse_date(e["filed_at"]) for e in evs)
        window = [e for e in evs if (f_last - parse_date(e["filed_at"])).days <= 30]
        buyers = {e["actor"].get("id") for e in window}
        if len(buyers) < 2:
            continue
        r3, at = None, "today"
        if panel is not None and panel.ok():
            i = panel.ix_on_or_before(f_last)
            pf = price_features(panel, t, i) if i is not None else None
            if pf and pf.get("r3m") is not None:
                r3, at = pf["r3m"], f_last.isoformat()
        if r3 is None and q.get("ret_3m") is not None:
            r3 = q["ret_3m"] / 100
        if r3 is None or r3 > -0.15:
            continue
        value = sum(e.get("value") or 0 for e in window)
        when = "at the latest filing" if at != "today" else "as of today (no filing-day price history)"
        out.append({
            "id": f"weak:{t}", "ticker": t, "screen": "insider_weakness", "company": window[0].get("company"),
            "title": f"{len(buyers)} insiders bought after a {abs(r3):.0%} 3-month fall",
            "why": [f"3-month move {r3:+.0%} {when}"] +
                   [f"{e['actor']['name']} ({e['actor'].get('role')}) bought ${(e.get('value') or 0):,.0f}" for e in window[:5]],
            "signals": [e["id"] for e in window[:8]], "strength": len(buyers) * 10 + min(30, value / 1e5),
            "risk_level": q.get("risk_level"), "projection": projection(q.get("price"), q.get("vol_1y")),
            "measured_at": at,
        })
    out.sort(key=lambda d: -d["strength"])
    return out[:15]


def momentum_ideas(panel: Panel | None, allowed: set[str] | None, quotes: dict, limit: int = 15) -> list[dict]:
    if panel is None or not panel.ok():
        return []
    i = panel.T - 1
    picks = momentum_screen(panel, i, liquid_universe(panel, i, allowed))
    out = []
    for d in picks[:limit]:
        t = d["ticker"]
        if t == BENCH:
            continue
        q = quotes.get(t) or {}
        out.append({
            "id": f"mom:{t}", "ticker": t, "screen": "momentum", "company": q.get("name"),
            "title": f"Up {d['mom6_1']:.0%} over 6 months, {abs(d['dist_hi']):.1%} from its 52-week high",
            "why": [f"6-month momentum (skipping last month): {d['mom6_1']:+.0%}",
                    f"Last month: {d['r1m']:+.1%}" if d.get("r1m") is not None else "",
                    f"Volatility: {d['vol3m']:.0%} a year", f"Trades ~${(d.get('adv') or 0) / 1e6:,.0f}M a day"],
            "signals": [], "strength": d["mom6_1"] * 100,
            "risk_level": q.get("risk_level"), "projection": projection(d["price"], d["vol3m"]),
        })
    for o in out:
        o["why"] = [w for w in o["why"] if w]
    return out


def build(events: list[dict], quotes: dict, panel: Panel | None, allowed: set[str] | None, quant: dict) -> dict:
    rates = (quant or {}).get("rates") or {}
    screen_bt = (quant or {}).get("screen") or {}
    lists = {
        # No long history exists for this exact mix of funds, activists, Congress and insiders, so it gets no
        # probability until the app's own graded record for it reaches 30; the closest measured pattern is shown
        # separately and labelled as such.
        "convergence": {"title": "Smart-money convergence",
                        "about": "Several independent informed buyers in the same stock within ~45 days.",
                        "track": None,
                        "related": ({**rates["insider_buy_cluster"],
                                     "label": "Closest measured pattern: 3+ insiders buying within 30 days"}
                                    if rates.get("insider_buy_cluster") else None),
                        "items": convergence(events, quotes)},
        "insider_weakness": {"title": "Insiders buying into weakness",
                             "about": "Two or more insiders buying within 30 days after a 15%+ fall over 3 months.",
                             "track": rates.get("insider_buy_cluster_weakness"),
                             "items": insider_weakness(events, quotes, panel)},
        "momentum": {"title": "Momentum near highs",
                     "about": "Top-decile 6-month momentum, within 5% of the 52-week high, volatility under 70%.",
                     "track": ({"n": screen_bt.get("cohorts"), "p": screen_bt.get("beat_rate"),
                                "lo": (screen_bt.get("beat_rate_ci") or [None, None])[0],
                                "hi": (screen_bt.get("beat_rate_ci") or [None, None])[1],
                                "mean_excess": screen_bt.get("mean_excess"),
                                "ex_lo": (screen_bt.get("mean_excess_ci") or [None, None])[0],
                                "ex_hi": (screen_bt.get("mean_excess_ci") or [None, None])[1],
                                "all_stocks": screen_bt.get("all_stocks_beat_rate"), "unit": "months"}
                               if screen_bt.get("status") == "ok" else None),
                     "items": momentum_ideas(panel, allowed, quotes)},
    }
    return {"built": now_utc().isoformat(), "lists": lists}


def apply_registry(new: dict, registry: dict, today: dt.date | None = None, grace_days: int = 30) -> dict:
    """Give every idea the date it first appeared. An idea that drops off a list and comes back within
    `grace_days` keeps its original date, so it is graded once, from then (no fresh start after a bad stretch).
    `registry` ({id: {"first_seen", "last_seen"}}) is kept in the app's state and pruned here."""
    today = today or now_et().date()
    t = today.isoformat()
    for lst in new["lists"].values():
        for it in lst["items"]:
            r = registry.get(it["id"])
            last = parse_date((r or {}).get("last_seen"))
            if not r or not last or (today - last).days > grace_days:
                r = {"first_seen": t}
            r["last_seen"] = t
            registry[it["id"]] = r
            it["first_seen"] = r["first_seen"]
    for k in [k for k, r in registry.items() if (today - (parse_date(r.get("last_seen")) or today)).days > grace_days]:
        registry.pop(k, None)
    return new


def as_of(d: str | None) -> dt.date | None:
    return parse_date(d)
