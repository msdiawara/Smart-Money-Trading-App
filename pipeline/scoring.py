"""Signal score (0–100) and "Claude's read" for every event.

The score rates how *informative* a disclosed trade has historically tended to be, using findings from
academic research (see docs/DATA_SOURCES.md §8). It is not a price prediction and not a recommendation.
Each point added or removed is explained in the pros/cons shown in the app."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .util import clamp, fmt_money

STRONG, NOTABLE, MILD = 70, 50, 30


@dataclass
class Ctx:
    quote: Callable[[str | None], dict | None] = lambda t: None
    insider_cluster: Callable[[dict], dict] = lambda e: {"buyers": 0, "sellers": 0, "names": []}
    insider_record: Callable[[str], dict | None] = lambda actor_id: None
    insider_routine: Callable[[dict], bool | None] = lambda e: None
    fund_meta: Callable[[str], dict | None] = lambda cik: None
    fund_consensus: Callable[[dict], int] = lambda e: 0
    etf_consensus: Callable[[dict], int] = lambda e: 0        # other tracked ETFs trading the same way that day
    member_record: Callable[[str], dict | None] = lambda actor_id: None
    is_leader: Callable[[dict], str | None] = lambda actor: None
    extra: dict = field(default_factory=dict)


def _label(score: float, noise: bool = False) -> str:
    if noise:
        return "noise"
    return "strong" if score >= STRONG else "notable" if score >= NOTABLE else "mild" if score >= MILD else "weak"


def frozen_quote(e: dict, q: dict | None) -> dict | None:
    """Price context as of when the app first saw the filing, so a later rally or slump can't rewrite why a
    trade looked interesting (e.g. 'bought near the 52-week low' stays true to the moment it was disclosed)."""
    snap = e.get("quote_at_filing")
    if snap is None and q and q.get("price"):
        snap = {k: q.get(k) for k in ("price", "lo_52w", "hi_52w", "ret_3m", "as_of")}
        e["quote_at_filing"] = snap
    if not snap:
        return q
    return {**(q or {}), **{k: v for k, v in snap.items() if k != "as_of"}}


def _price_context(q: dict | None, pros: list, cons: list, bullish: bool) -> float:
    pts = 0.0
    if not q:
        return pts
    price, lo, hi = q.get("price"), q.get("lo_52w"), q.get("hi_52w")
    if bullish and price and lo and hi and hi > lo:
        pos = (price - lo) / (hi - lo)
        if pos <= 0.15:
            pts += 4
            pros.append("Bought near the 52-week low: buying into weakness, not chasing")
        elif pos >= 0.9:
            cons.append("Stock is near its 52-week high")
    r3 = q.get("ret_3m")
    if bullish and r3 is not None and r3 <= -20:
        pts += 3
        pros.append(f"Stock fell {abs(r3):.0f}% over 3 months before this buy")
    lvl = q.get("risk_level")
    if lvl and lvl >= 4:
        cons.append(f"Risk level {lvl}/5 ({q.get('risk_label', '').lower()}): big swings are normal here")
    return pts


def _size_tier_pts(q: dict | None, pros: list, cons: list) -> float:
    tier = (q or {}).get("cap_tier")
    if tier in ("micro", "small"):
        pros.append("Smaller company: insider buys have historically been most informative here")
        return 6
    if tier == "mid":
        return 3
    if tier == "mega":
        cons.append("Mega-cap: one insider's buy rarely moves a company this size")
        return -3
    return 0


def score_insider(e: dict, ctx: Ctx) -> None:
    d = e.get("details", {})
    a = e.get("action")
    pros: list[str] = []
    cons: list[str] = []
    checks: list[str] = []
    name, role, rank = e["actor"]["name"], e["actor"].get("role", "Insider"), e["actor"].get("rank", 1)
    v = e.get("value") or 0
    q = ctx.quote(e.get("ticker"))
    if d.get("notice"):
        return score_sale_notice(e, ctx)
    noise = a not in ("buy", "sell")
    if noise:
        score = 3.0
        summary = f"{name} ({role}): {e.get('action_label', 'non-market transaction').lower()}. Not a buy/sell decision."
        cons.append("Grants, option exercises, tax withholding and gifts are routine, so they carry little information")
        e.update(score=int(score), signal="noise", direction="neutral",
                 read={"summary": summary, "pros": pros, "cons": cons, "check": checks})
        return
    bullish = a == "buy"
    if bullish:
        score = 40.0
        score += {5: 15, 4: 12, 3: 7, 2: 3}.get(rank, 0)
        if rank >= 4:
            pros.append(f"Senior executive ({role}): they know the business best")
        elif "Director" in role:
            pros.append("Board director buying with their own money")
        if v >= 1e6:
            score += 12
            pros.append(f"Large purchase ({fmt_money(v)})")
        elif v >= 250e3:
            score += 9
            pros.append(f"Sizable purchase ({fmt_money(v)})")
        elif v >= 100e3:
            score += 6
        elif v >= 25e3:
            score += 3
        elif v and v < 10e3:
            score -= 5
            cons.append(f"Small purchase ({fmt_money(v)}), possibly symbolic")
        chg = e.get("holdings_change_pct")
        if chg is not None:
            if chg >= 50:
                score += 8
                pros.append(f"Raised their stake {chg:.0f}%")
            elif chg >= 20:
                score += 5
                pros.append(f"Raised their stake {chg:.0f}%")
            elif chg >= 5:
                score += 2
            elif chg < 2:
                cons.append("Tiny vs. what they already own (+" + (f"{chg:.1f}%" if chg >= 0.1 else "under 0.1%") + ")")
        elif e.get("holdings_after") and e.get("shares") and abs(e["holdings_after"] - e["shares"]) < 1:
            score += 5
            pros.append("A brand-new position for this insider")
        cl = ctx.insider_cluster(e)
        others = max(0, cl.get("buyers", 0) - 1)
        if others >= 2:
            score += 14
            pros.append(f"Cluster buy: {others} other insiders bought within 30 days. Clusters have historically "
                        "earned about twice the excess return of a lone buy")
        elif others == 1:
            score += 8
            pros.append("Another insider also bought within 30 days")
        score += _size_tier_pts(q, pros, cons)
        score += _price_context(frozen_quote(e, q), pros, cons, True)
    else:
        score = 15.0
        if rank >= 4:
            score += 5
        if v >= 5e6:
            score += 5
            pros.append(f"Large sale ({fmt_money(v)})")
        chg = e.get("holdings_change_pct")
        if chg is not None and chg <= -50:
            score += 8
            pros.append(f"Sold {abs(chg):.0f}% of their stake")
        elif chg is not None and chg <= -20:
            score += 4
            pros.append(f"Sold {abs(chg):.0f}% of their stake")
        elif chg is not None and chg > -5:
            share = f"{abs(chg):.1f}%" if abs(chg) >= 0.1 else "under 0.1%"
            cons.append(f"Only {share} of their holdings")
        cl = ctx.insider_cluster(e)
        if cl.get("sellers", 0) >= 3:
            score += 6
            pros.append(f"{cl['sellers']} insiders sold or filed to sell within 30 days")
        if d.get("exercise_linked"):
            score -= 5
            cons.append("Sale paired with an option exercise, often just cashing compensation")
        cons.append("Insiders sell for many reasons (taxes, diversification, a house). Sales predict much less "
                    "than purchases")
    if d.get("plan_10b5_1"):
        score -= 15 if bullish else 10
        cons.append("Made under a pre-scheduled 10b5-1 plan, so it wasn't a fresh decision")
    routine = ctx.insider_routine(e)
    if routine is True:
        score -= 10
        cons.append("Routine trader: trades in the same month every year. Research found such trades carry ~no "
                    "information")
    elif routine is False:
        score += 4
        pros.append("Opportunistic timing (not their usual calendar pattern)")
    rec = ctx.insider_record(e["actor"].get("id", ""))
    if rec and rec.get("n", 0) >= 4 and bullish:
        wr = rec.get("win_rate")
        if wr is not None and wr >= 0.65:
            score += 8
            pros.append(f"Track record: {wr*100:.0f}% of their {rec['n']} past buys were up 3 months later")
        elif wr is not None and wr <= 0.35:
            score -= 6
            cons.append(f"Weak track record: only {wr*100:.0f}% of {rec['n']} past buys were up 3 months later")
    if d.get("indirect"):
        checks.append(f"Held indirectly ({d.get('ownership_nature') or 'trust/family/fund'})")
    if d.get("late"):
        score -= 2
        cons.append(f"Filed late ({e.get('lag_days')} business days after the trade; the rule is 2)")
    if d.get("amendment"):
        score -= 3
        checks.append("This is an amended filing (4/A); compare with the original")
    score = clamp(score, 1, 99)
    verb = "bought" if bullish else "sold"
    money = fmt_money(v) if v else f"{(e.get('shares') or 0):,.0f} shares"
    price = f" at ~${e['price']:,.2f}" if e.get("price") else ""
    summary = f"{name} ({role}) {verb} {money} of {e.get('ticker') or e.get('company')}{price}."
    cl = ctx.insider_cluster(e)
    if bullish and cl.get("buyers", 0) >= 2:
        summary += f" {cl['buyers']} insiders bought in the last 30 days."
    checks.append("Open the filing's footnotes. They explain weighted-average prices, trusts and plans")
    if bullish:
        checks.append("Look for news or earnings in the next few weeks; insiders often buy ahead of good updates, "
                      "but not always")
    e.update(score=int(round(score)), signal=_label(score), direction="bullish" if bullish else "bearish",
             read={"summary": summary, "pros": pros, "cons": cons, "check": checks})


def score_sale_notice(e: dict, ctx: Ctx) -> None:
    """Form 144: notice of an intended sale. Earlier than the Form 4, but weaker: most are compensation shares
    or scheduled plans, and the sale may not happen at all."""
    d = e.get("details", {})
    name, role, rank = e["actor"]["name"], e["actor"].get("role", "Insider"), e["actor"].get("rank", 1)
    v = e.get("value") or 0
    pros, cons, checks = [], [], []
    score = 12.0
    if v >= 10e6:
        score += 10
        pros.append(f"Large intended sale ({fmt_money(v)})")
    elif v >= 2e6:
        score += 5
        pros.append(f"Sizable intended sale ({fmt_money(v)})")
    pct = d.get("pct_outstanding")
    if pct is not None and pct >= 0.5:
        score += 8
        pros.append(f"{pct:.2f}% of all shares outstanding")
    elif pct is not None and pct >= 0.1:
        score += 3
    if rank >= 4:
        score += 4
    if d.get("plan_10b5_1"):
        score -= 6
        cons.append("Under a pre-scheduled 10b5-1 plan, so it wasn't a fresh decision")
    acq = d.get("acquisitions") or []
    comp = [a for a in acq if any(k in (a.get("nature") or "").lower()
                                   for k in ("rsu", "psu", "restricted", "option", "award", "grant", "compensation", "services"))]
    if acq and len(comp) == len(acq):
        score -= 4
        cons.append("Shares came from compensation (RSUs, options, awards): usually routine selling")
    past = d.get("sold_past_3m") or []
    if len(past) >= 3:
        cons.append(f"Part of an ongoing selling program ({len(past)} sales in the past 3 months)")
    cl = ctx.insider_cluster(e)
    if cl.get("sellers", 0) >= 2:
        score += 5
        pros.append(f"{cl['sellers']} insiders sold or filed to sell within 30 days")
    cons.append("A notice of intent, not a trade: the Form 4 confirms whether and at what price it happened")
    cons.append("Insider sales predict much less than purchases")
    checks.append("Watch for the Form 4 within 2 business days of the sale")
    score = clamp(score, 1, 99)
    when = f" around {d['approx_sale_date']}" if d.get("approx_sale_date") else ""
    summary = (f"{name} ({role}) filed notice to sell {fmt_money(v) if v else f'{(e.get('shares') or 0):,.0f} shares'} of "
               f"{e.get('ticker') or e.get('company')}{when}.")
    e.update(score=int(round(score)), signal=_label(score), direction="bearish",
             read={"summary": summary, "pros": pros, "cons": cons, "check": checks})


def score_fund(e: dict, ctx: Ctx) -> None:
    d = e.get("details", {})
    kind = d.get("change")
    pros, cons, checks = [], [], []
    w, pw = d.get("weight") or 0, d.get("prev_weight") or 0
    meta = ctx.fund_meta(d.get("fund_cik", "")) or {}
    style = meta.get("style", "")
    bullish = kind in ("new", "add")
    score = {"new": 35, "add": 25, "trim": 15, "exit": 25}.get(kind, 10)
    if bullish:
        if w >= 5:
            score += 20
            pros.append(f"High-conviction size: {w:.1f}% of the fund")
        elif w >= 2:
            score += 12
            pros.append(f"Meaningful size: {w:.1f}% of the fund")
        elif w >= 1:
            score += 6
        else:
            cons.append(f"Small position ({w:.2f}% of the fund)")
        if kind == "add" and (d.get("pct_change") or 0) >= 50:
            score += 6
            pros.append(f"Share count up {d['pct_change']:.0f}% vs. last quarter")
        n = ctx.fund_consensus(e)
        if n >= 2:
            score += min(16, 8 * (n - 1))
            pros.append(f"{n} tracked funds bought or added this quarter")
    else:
        if pw >= 5:
            score += 15
            pros.append(f"Was a big position ({pw:.1f}% of the fund)")
        elif pw >= 2:
            score += 8
    if style in ("concentrated", "value"):
        score += 8
        pros.append("Concentrated, long-term manager. These are the 13Fs that have copied best")
    elif style in ("high-turnover", "macro", "diversified"):
        score -= 8
        cons.append("High-turnover or very diversified fund: the position may already be gone")
    clone = meta.get("clone") or {}
    if clone.get("quarters", 0) >= 4 and clone.get("beat_rate") is not None:
        if clone["beat_rate"] >= 0.6:
            score += 8
            pros.append(f"Copying this fund's 13F beat the S&P 500 in {clone['beat_rate']*100:.0f}% of the last "
                        f"{clone['quarters']} quarters")
        elif clone["beat_rate"] <= 0.35:
            score -= 5
            cons.append(f"Copying this fund's 13F beat the S&P 500 in only {clone['beat_rate']*100:.0f}% of quarters")
    age = d.get("days_since_quarter_end")
    if age is not None and age > 60:
        score -= 5
        cons.append(f"Holdings are {age} days old: the fund may have changed them since")
    cons.append("13Fs show long US stock positions only. Hedges, shorts and bonds are invisible")
    q = ctx.quote(e.get("ticker"))
    score += _price_context(frozen_quote(e, q), pros, cons, bullish) * 0.5
    score = clamp(score, 1, 99)
    verbs = {"new": "opened a new position in", "add": "added to", "trim": "trimmed", "exit": "sold out of"}
    summary = (f"{e['actor']['name']} {verbs.get(kind, 'changed')} {e.get('ticker') or e.get('company')}"
               f" ({fmt_money(e.get('value'))} at quarter end" + (f", {w:.1f}% of the fund)." if w else ")."))
    checks.append(f"Quarter ended {d.get('period')}; filed {e.get('filed_at', '')[:10]}")
    e.update(score=int(round(score)), signal=_label(score), direction="bullish" if bullish else "bearish",
             read={"summary": summary, "pros": pros, "cons": cons, "check": checks})


def score_activist(e: dict, ctx: Ctx) -> None:
    d = e.get("details", {})
    pros, cons, checks = [], [], []
    pct = d.get("percent")
    is13d = d.get("schedule") == "13D"
    amend = d.get("amendment")
    score = (30 if amend else 55) if is13d else (18 if amend else 25)
    if is13d and not amend:
        pros.append("New activist (13D) stake: activist campaigns have historically lifted prices around the filing")
    if not is13d:
        cons.append("13G = passive holder (index funds, institutions); usually not a signal")
    if pct is not None:
        if pct >= 10:
            score += 5
            pros.append(f"Owns {pct:.1f}% of the company")
        else:
            checks.append(f"Owns {pct:.1f}% of the company")
    meta = ctx.fund_meta(d.get("filer_cik", "")) or {}
    if meta.get("style") == "activist":
        score += 10
        pros.append(f"Known activist ({meta.get('name')})")
    if d.get("purpose"):
        checks.append("Read 'Purpose of transaction' below for what they want to change")
    q = ctx.quote(e.get("ticker"))
    score += _price_context(frozen_quote(e, q), pros, cons, True) * 0.5
    score = clamp(score, 1, 99)
    summary = (f"{e['actor']['name']} reported a {'new ' if not amend else ''}{'activist' if is13d else 'passive'} "
               f"stake in {e.get('ticker') or e.get('company')}" + (f" ({pct:.1f}%)." if pct is not None else "."))
    e.update(score=int(round(score)), signal=_label(score), direction="bullish" if is13d else "neutral",
             read={"summary": summary, "pros": pros, "cons": cons, "check": checks})


def score_congress(e: dict, ctx: Ctx) -> None:
    d = e.get("details", {})
    pros, cons, checks = [], [], []
    a = e.get("action")
    lo, hi = (e.get("value_range") or [None, None]) + [None] * 0
    mid = ((lo or 0) + (hi or lo or 0)) / 2 if lo else 0
    bullish = a == "buy"
    score = 30.0 if bullish else 20.0 if a == "sell" else 8.0
    if mid >= 1e6:
        score += 15
        pros.append(f"Large trade (range {fmt_money(lo)}–{fmt_money(hi)})")
    elif mid >= 250e3:
        score += 10
        pros.append(f"Sizable trade ({fmt_money(lo)}–{fmt_money(hi)})")
    elif mid >= 50e3:
        score += 5
    elif lo and hi and hi <= 15000:
        score -= 3
        cons.append("Smallest disclosure bracket ($1K–$15K)")
    title = ctx.is_leader(e["actor"])
    if title:
        score += 10
        pros.append("Congressional leadership" + (f" ({title})" if isinstance(title, str) else "")
                    + ": research found party leaders' trades beat their peers'")
    else:
        cons.append("Research finds rank-and-file members don't beat the market on average")
    rec = ctx.member_record(e["actor"].get("id", ""))
    if rec and rec.get("n", 0) >= 8 and bullish:
        wr = rec.get("win_rate")
        if wr is not None and wr >= 0.6:
            score += 8
            pros.append(f"Their past buys: {wr*100:.0f}% up 3 months after disclosure (n={rec['n']})")
        elif wr is not None and wr <= 0.4:
            score -= 5
            cons.append(f"Their past buys: only {wr*100:.0f}% up 3 months after disclosure (n={rec['n']})")
    if (d.get("asset_type") or "").lower().startswith(("stock option", "option")) or "option" in (d.get("description") or "").lower():
        score += 4
        pros.append("Options trade: a more deliberate, leveraged bet")
    lag = e.get("lag_days")
    if lag is not None:
        if lag > 45:
            score -= 3
            cons.append(f"Disclosed late ({lag} days; the limit is 45)")
        elif lag > 30:
            score -= 5
            cons.append(f"Disclosed {lag} days after the trade, so much of any edge may be gone")
        elif lag <= 7:
            score += 3
            pros.append(f"Disclosed quickly ({lag} days after the trade)")
    if d.get("owner") and d["owner"].lower() not in ("self", ""):
        checks.append(f"Owner: {d['owner']}")
    checks.append("Amounts are disclosed only as ranges; exact size is unknown")
    score = clamp(score, 1, 99)
    verb = {"buy": "bought", "sell": "sold", "exchange": "exchanged"}.get(a, "traded")
    rng = f"{fmt_money(lo)}–{fmt_money(hi)}" if lo and hi else (f"over {fmt_money(lo)}" if lo else "an undisclosed amount")
    summary = f"{e['actor']['name']} {verb} {rng} of {e.get('ticker') or e.get('company')} on {e.get('trade_date')}."
    e.update(score=int(round(score)), signal=_label(score), direction="bullish" if bullish else "bearish" if a == "sell" else "neutral",
             read={"summary": summary, "pros": pros, "cons": cons, "check": checks})


def score_etoro(e: dict, ctx: Ctx) -> None:
    d = e.get("details", {})
    pros, cons, checks = [], [], []
    opened = e.get("action") in ("open", "add")
    long = d.get("is_buy", True)
    score = 30.0 if opened else 15.0
    wr = d.get("win_ratio")
    if wr is not None:
        if wr >= 70:
            score += 10
            pros.append(f"Win rate {wr:.0f}% on closed trades")
        elif wr >= 60:
            score += 6
            pros.append(f"Win rate {wr:.0f}% on closed trades")
        elif wr < 50:
            cons.append(f"Win rate only {wr:.0f}%")
    g = d.get("gain_pct")
    if g is not None and g >= 30:
        score += 6
        pros.append(f"Up {g:.0f}% over the ranking period")
    rs = d.get("risk_score")
    if rs is not None:
        if rs <= 4:
            score += 5
            pros.append(f"eToro risk score {rs}/10 (disciplined)")
        elif rs >= 7:
            score -= 6
            cons.append(f"eToro risk score {rs}/10 (aggressive)")
    ip = d.get("investment_pct")
    if opened and ip is not None:
        if ip >= 10:
            score += 10
            pros.append(f"Big bet: {ip:.1f}% of their portfolio")
        elif ip >= 5:
            score += 6
            pros.append(f"{ip:.1f}% of their portfolio")
        elif ip < 1:
            score -= 5
            cons.append(f"Small position ({ip:.2f}% of their portfolio)")
    lev = d.get("leverage") or 1
    if lev > 1:
        score -= 6
        cons.append(f"Leveraged x{lev}: a short-term trade, not an investment")
    if (d.get("copiers") or 0) >= 1000:
        score += 3
    cons.append("Win rate counts winning trades, not how much was won or lost; check their drawdown too")
    score = clamp(score, 1, 99)
    dirn = "long" if long else "short"
    verb = "opened a" if opened else "closed a"
    summary = f"eToro investor {e['actor']['name']} {verb} {dirn} position in {e.get('ticker') or e.get('company')}"
    summary += f" at {e['price']:,.4g}." if e.get("price") else "."
    e.update(score=int(round(score)), signal=_label(score),
             direction=("bullish" if long else "bearish") if opened else "neutral",
             read={"summary": summary, "pros": pros, "cons": cons, "check": checks})


def score_etf(e: dict, ctx: Ctx) -> None:
    """A transparent ETF's trade, from the fund's own daily holdings file (1-day lag). Daily and official, but
    these funds trade a lot, so a single add or trim is weaker evidence than a 13F change or an insider buy."""
    d = e.get("details", {})
    a = e.get("action")
    bullish = a in ("new", "add")
    pros, cons, checks = [], [], []
    w, pw, v = d.get("weight") or 0, d.get("prev_weight") or 0, e.get("value") or 0
    score = {"new": 36.0, "add": 22.0, "trim": 14.0, "exit": 22.0}.get(a, 10.0)
    if bullish:
        if w >= 4:
            score += 14
            pros.append(f"High-conviction size: {w:.1f}% of the fund")
        elif w >= 2:
            score += 8
            pros.append(f"Meaningful size: {w:.1f}% of the fund")
        elif w >= 1:
            score += 4
        else:
            cons.append(f"Small position ({w:.2f}% of the fund)")
        if a == "add" and (d.get("pct_change") or 0) >= 25:
            score += 6
            pros.append(f"Share count up {d['pct_change']:.0f}% beyond what fund inflows explain")
        if v >= 20e6:
            score += 6
            pros.append(f"Large trade ({fmt_money(v)})")
        elif v >= 5e6:
            score += 3
        n = ctx.etf_consensus(e)
        if n >= 1:
            score += min(6, 3 * n)
            pros.append(f"{n} other fund{'s' if n > 1 else ''} in the same family bought the same day (often one "
                        "decision applied across funds, so this counts only a little)")
    else:
        if pw >= 4:
            score += 10
            pros.append(f"Was a big position ({pw:.1f}% of the fund)")
        elif pw >= 2:
            score += 5
        if a == "trim" and abs(d.get("pct_change") or 0) < 5:
            score -= 4
            cons.append("A small trim; active ETFs rebalance constantly")
        n = ctx.etf_consensus(e)
        if n >= 1:
            score += min(6, 3 * n)
            pros.append(f"{n} other fund{'s' if n > 1 else ''} in the same family sold the same day")
    pros.append("Published by the fund itself the next morning: a one-day lag, not the 45+ days of a 13F")
    cons.append("The file shows what changed, not why; a trade can be a rebalance, a redemption or a view")
    q = ctx.quote(e.get("ticker"))
    score += _price_context(frozen_quote(e, q), pros, cons, bullish) * 0.5
    score = clamp(score, 1, 99)
    verbs = {"new": "opened a new position in", "add": "added to", "trim": "trimmed", "exit": "sold out of"}
    delta = abs(d.get("delta_shares") or 0)
    summary = (f"{e['actor']['name']} {verbs.get(a, 'changed')} {e.get('ticker') or e.get('company')}"
               f" ({delta:,.0f} shares, ~{fmt_money(v)})" + (f", now {w:.1f}% of the fund." if w else "."))
    checks.append(f"From the fund's holdings file dated {d.get('file_date')} (positions at the start of that day)")
    e.update(score=int(round(score)), signal=_label(score), direction="bullish" if bullish else "bearish",
             read={"summary": summary, "pros": pros, "cons": cons, "check": checks})


SCORERS = {"insider": score_insider, "fund": score_fund, "activist": score_activist, "etf": score_etf,
           "congress": score_congress, "etoro": score_etoro}


def score_event(e: dict, ctx: Ctx) -> dict:
    fn = SCORERS.get(e.get("kind"))
    if fn:
        fn(e, ctx)
    return e
