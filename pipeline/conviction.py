"""The app's overall rating for a stock, with "Top conviction" as its strongest call.

"Top conviction" appears only when EVERY check passes; each check is shown so nothing is hidden:
  1. Claude's latest research view is favorable, with at least a 58% chance (and 8 points above the measured base
     rate) of beating the S&P 500 over ~3 months, at medium or high confidence, made in the last 30 days;
  2. measured evidence agrees: a recent bullish signal from a model that passed its out-of-sample gate and
     predicts a real lift, a kind of signal whose beat rate is above 50% with its whole 95% interval, or a screen
     whose own track record beats the index with confidence;
  3. not "Very expensive" relative to its fundamentals;
  4. bubble risk is not High;
  5. risk level 4 of 5 or lower, and it trades at least $5M a day;
  6. no red-flag company filings in 90 days (restatement, delisting notice, bankruptcy, debt acceleration).
Every Top conviction call is written down and graded against the S&P 500 after 63 trading days, like every
other probability in the app, and its record is shown on the Records tab. Research, not personal advice.

Below it: Positive · Neutral · Caution · Red flags.
"""
from __future__ import annotations

import datetime as dt

from .util import now_utc, parse_iso

RATINGS = {"top": "Top conviction", "positive": "Positive", "neutral": "Neutral", "caution": "Caution",
           "red": "Red flags"}


def latest_claude(t: str, events: list[dict], idea_items: list[dict], cfc: dict | None, days: int = 30,
                  base_any: float | None = None) -> dict | None:
    """Most recent Claude view on this ticker (trade view, idea write-up or price forecast) within `days`."""
    cut = now_utc() - dt.timedelta(days=days)
    cands = []
    for e in events:
        a = e.get("analysis")
        if e.get("ticker") == t and a and a.get("stance"):
            cands.append({**a, "from": "trade", "base": a.get("base_p")})
    for it in idea_items:
        a = it.get("analysis")
        if it.get("ticker") == t and a and a.get("stance"):
            cands.append({**a, "from": "idea", "base": a.get("base_p")})
    if cfc and cfc.get("stance"):
        cands.append({"stance": cfc["stance"], "probability": cfc.get("probability"),
                      "confidence": cfc.get("confidence"), "made": cfc.get("made_at"), "from": "forecast",
                      # a price forecast isn't tied to a signal: its base rate is how often any stock beats the index
                      "base": cfc.get("base_p") if cfc.get("base_p") is not None else base_any})
    ok = [c for c in cands if parse_iso(c.get("made")) and parse_iso(c["made"]) > cut]
    if not ok:
        return None
    return max(ok, key=lambda c: parse_iso(c["made"]))


def measured_support(t: str, events: list[dict], idea_lists: dict, days: int = 60) -> dict | None:
    """The strongest measured evidence that agrees with a bullish call, or None. Thin evidence doesn't count:
    a model only after passing its gate and with a lift of 3+ points; a base rate only when its whole 95%
    interval is above 50%; a screen only when its track record beats the index with confidence."""
    cut = (now_utc() - dt.timedelta(days=days)).date().isoformat()
    best = None

    def offer(cand):
        nonlocal best
        if best is None or cand["edge"] > best["edge"]:
            best = cand
    for e in events:
        if e.get("ticker") != t or e.get("direction") != "bullish" or (e.get("filed_at") or "")[:10] < cut:
            continue
        q = e.get("quant") or {}
        br = (q.get("base_rate") or {})
        p = q.get("p")
        what = f"{e.get('action_label') or e.get('action')} by {e['actor']['name']}"
        if p is not None and br.get("p") is not None and q.get("source") == "model" and p - br["p"] >= 0.03:
            offer({"what": f"{what}: the tested model gives {p*100:.0f}% vs a {br['p']*100:.0f}% base rate",
                   "edge": p - br["p"], "kind": "signal", "id": e["id"]})
        if (br.get("n") or 0) >= 30 and br.get("lo") is not None and br["lo"] > 0.5:
            offer({"what": f"{what}: signals like it beat the index {br['p']*100:.0f}% of the time "
                           f"(95% range {br['lo']*100:.0f}–{br['hi']*100:.0f}%, {br['n']} cases)",
                   "edge": br["p"] - 0.5, "kind": "signal", "id": e["id"]})
    for key, lst in (idea_lists or {}).items():
        trk = lst.get("track") or {}
        enough = (trk.get("n") or 0) >= (10 if trk.get("unit") == "months" else 30)
        sure = (trk.get("lo") is not None and trk["lo"] > 0.5) or (trk.get("ex_lo") is not None and trk["ex_lo"] > 0)
        if not enough or not sure or (trk.get("p") or 0) <= 0.5:
            continue
        for it in lst.get("items", []):
            if it.get("ticker") == t:
                offer({"what": f"On the '{lst.get('title') or key}' screen (beat the index {trk['p']*100:.0f}% of the time)",
                       "edge": trk["p"] - 0.5, "kind": "screen"})
    return best


def rate(t: str, *, quote: dict | None, val: dict | None, bub: dict | None, flags: list[dict] | None,
         claude: dict | None, support: dict | None, cfg: dict | None = None) -> dict:
    cfg = cfg or {}
    min_p = float(cfg.get("min_claude_p", 0.58))
    min_edge = float(cfg.get("min_edge_vs_base", 0.08))
    min_adv = float(cfg.get("min_adv_usd", 5e6))
    q = quote or {}
    checks = []

    def check(ok: bool | None, label: str, detail: str):
        checks.append({"ok": ok, "label": label, "detail": detail})
    p = (claude or {}).get("probability")
    base = (claude or {}).get("base")
    conf = (claude or {}).get("confidence")
    if claude is None:
        check(None, "Claude's research view", "No Claude view in the last 30 days")
    else:
        need = max(min_p, (base or 0) + min_edge) if base is not None else min_p
        ok = (claude.get("stance") == "favorable" and p is not None and p >= need and conf in ("medium", "high"))
        check(ok, "Claude's research view",
              f"{claude.get('stance', '?').capitalize()}, {p*100:.0f}% to beat the S&P 500 (needs ≥{need*100:.0f}%), "
              f"{conf or '?'} confidence" if p is not None else f"{claude.get('stance', '?').capitalize()}, no probability")
    check(support is not None, "Measured evidence agrees",
          support["what"] if support else "No recent bullish signal with a measured edge")
    lab = (val or {}).get("label")
    check(None if lab is None else lab != "Very expensive", "Not very expensive",
          f"Valuation: {lab}" if lab else "Valuation not available")
    lvl = (bub or {}).get("level")
    check(None if lvl is None else lvl != "High", "Bubble risk not High", f"Bubble risk: {lvl}" if lvl else "Not rated")
    rl, adv = q.get("risk_level"), q.get("adv_usd")
    check(None if rl is None else (rl <= 4 and (adv or 0) >= min_adv), "Risk & liquidity",
          f"Risk level {rl}/5, trades ${(adv or 0)/1e6:,.0f}M a day" if rl else "No risk data yet")
    fl = flags or []
    check(not fl, "No red-flag filings (90 days)",
          "None" if not fl else "; ".join(f"{f['label']} ({f['date']})" for f in fl[:2]))
    passed = all(c["ok"] is True for c in checks)
    why = []
    if passed:
        rating = "top"
    elif fl and any(f["item"] in ("1.03", "4.02", "3.01", "2.04") for f in fl):
        rating = "red"
        why.append(fl[0]["label"])
    elif (lvl == "High" and lab == "Very expensive") or (claude and claude.get("stance") == "unfavorable"):
        rating = "caution"
        why.append("High bubble risk and very expensive" if lvl == "High" and lab == "Very expensive"
                   else "Claude's view is unfavorable")
    elif lvl == "High" or lab == "Very expensive":
        rating = "caution"
        why.append("High bubble risk" if lvl == "High" else "Very expensive vs its fundamentals")
    elif (claude and claude.get("stance") == "favorable") or support is not None:
        rating = "positive"
        why.append("Claude favorable" if claude and claude.get("stance") == "favorable" else support["what"])
    else:
        rating = "neutral"
    return {"rating": rating, "label": RATINGS[rating], "checks": checks, "passed": sum(c["ok"] is True for c in checks),
            "of": len(checks), "why": why, "p": p if rating == "top" else None}
