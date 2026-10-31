"""Push alerts to your phone via ntfy (free, no account) and/or a Telegram bot. Alerts are bundled per run
and capped per day, with quiet hours (queued and delivered in the morning)."""
from __future__ import annotations

from .http import HttpClient, HttpError
from .settings import Settings
import datetime as dt

from .util import fmt_money, log, now_et, parse_date

ICON = {"insider": "👔", "fund": "🏦", "activist": "📣", "congress": "🏛️", "etoro": "📈"}


def matches_watch(e: dict, tickers: list[str], people: list[str]) -> bool:
    t = (e.get("ticker") or "").upper()
    if t and t in {x.upper() for x in tickers}:
        return True
    hay = " ".join([e.get("actor", {}).get("name", ""), e.get("company", "") or ""]).lower()
    return any(p.lower() in hay for p in people if p)


def line(e: dict) -> str:
    verb = {"buy": "BUY", "sell": "SELL", "new": "NEW", "add": "ADD", "trim": "TRIM", "exit": "EXIT",
            "stake": "STAKE", "open": "OPEN", "close": "CLOSE"}.get(e.get("action"), (e.get("action") or "").upper())
    amt = fmt_money(e["value"]) if e.get("value") else ""
    who = e.get("actor", {}).get("name", "")
    return f"{ICON.get(e['kind'], '•')} {verb} {e.get('ticker') or e.get('company', '')[:24]} · {who[:38]} {amt} · {e.get('score', 0)}/100"


def select(settings: Settings, state: dict, new_events: list[dict]) -> list[dict]:
    a = settings.c("alerts")
    sent = state["alerts"].setdefault("sent", {})
    out = []
    for e in new_events:
        if e["id"] in sent:
            continue
        if e.get("signal") == "noise":
            continue
        filed = parse_date((e.get("filed_at") or "")[:10])
        if filed and (now_et().date() - filed) > dt.timedelta(days=3):
            continue  # old news (first run / backfill) — it's in the app, but not worth a push
        if (e.get("score") or 0) >= a.get("min_score", 70) or matches_watch(e, a.get("watch_tickers", []),
                                                                            a.get("watch_people", [])):
            out.append(e)
    out.sort(key=lambda e: e.get("score", 0), reverse=True)
    return out


def deliver(settings: Settings, http: HttpClient, state: dict, events: list[dict]) -> dict:
    sec = settings.secrets
    a = settings.c("alerts")
    al = state["alerts"]
    today = now_et().date().isoformat()
    if al.get("day") != today:
        al["day"], al["count"] = today, 0
    queue = al.setdefault("queue", [])
    known = {q["id"] for q in queue}
    for e in events:
        if e["id"] not in known:
            queue.append({"id": e["id"], "line": line(e), "score": e.get("score", 0)})
    qh = a.get("quiet_hours_et") or [23, 6]
    h = now_et().hour
    quiet = (h >= qh[0] or h < qh[1]) if qh[0] > qh[1] else (qh[0] <= h < qh[1])
    result = {"queued": len(queue), "sent": 0, "channels": []}
    if quiet or not queue or not (sec.ntfy_topic or (sec.telegram_token and sec.telegram_chat_id)):
        if len(queue) > 200:
            del queue[:-200]
        return result
    room = max(0, a.get("max_per_day", 60) - al.get("count", 0))
    if room <= 0:
        return result
    queue.sort(key=lambda q: q["score"], reverse=True)
    batch = queue[: min(a.get("max_per_run", 6), room)]
    more = len(queue) - len(batch)
    title = f"{len(batch)} new smart-money signal{'s' if len(batch) != 1 else ''}"
    body = "\n".join(q["line"] for q in batch) + (f"\n…and {more} more in the app" if more > 0 else "")
    click = (sec.site_url or "") + (f"#/event/{batch[0]['id']}" if len(batch) == 1 else "#/feed")
    ok = False
    if sec.ntfy_topic:
        try:
            hdr = {"Title": title, "Tags": "chart_with_upwards_trend", "Priority": "high" if batch[0]["score"] >= 80 else "default"}
            if click.startswith("http"):
                hdr["Click"] = click
            if sec.ntfy_token:
                hdr["Authorization"] = f"Bearer {sec.ntfy_token}"
            http.post(f"{sec.ntfy_server}/{sec.ntfy_topic}", data=body.encode("utf-8"), headers=hdr)
            ok = True
            result["channels"].append("ntfy")
        except HttpError as e:
            log.warning("ntfy failed: %s", e)
    if sec.telegram_token and sec.telegram_chat_id:
        try:
            text = f"*{title}*\n{body}" + (f"\n{click}" if click.startswith("http") else "")
            http.post(f"https://api.telegram.org/bot{sec.telegram_token}/sendMessage",
                      data={"chat_id": sec.telegram_chat_id, "text": text, "disable_web_page_preview": True},
                      headers={"Content-Type": "application/json"})
            ok = True
            result["channels"].append("telegram")
        except HttpError as e:
            log.warning("telegram failed: %s", e)
    if ok:
        sent = al.setdefault("sent", {})
        for q in batch:
            sent[q["id"]] = today
        al["count"] = al.get("count", 0) + len(batch)
        del queue[:len(batch)]
        result["sent"] = len(batch)
        if len(sent) > 5000:
            for k in sorted(sent, key=sent.get)[:1000]:
                sent.pop(k, None)
    return result


def in_quiet_hours(settings: Settings) -> bool:
    a = settings.c("alerts", default={}) or {}
    qh = a.get("quiet_hours_et") or [23, 6]
    h = now_et().hour
    return (h >= qh[0] or h < qh[1]) if qh[0] > qh[1] else (qh[0] <= h < qh[1])


def send_system(settings: Settings, http: HttpClient, title: str, body: str, path: str = "") -> bool:
    """A one-off notice about the app itself (e.g. a failed self-check). Returns False when it wasn't sent (no
    channel, quiet hours, or every channel failed), so the caller keeps it pending and tries again later."""
    sec = settings.secrets
    if not (sec.ntfy_topic or (sec.telegram_token and sec.telegram_chat_id)):
        return False
    if in_quiet_hours(settings):
        return False
    click = (sec.site_url or "") + path
    ok = False
    if sec.ntfy_topic:
        try:
            hdr = {"Title": title, "Tags": "warning", "Priority": "default"}
            if click.startswith("http"):
                hdr["Click"] = click
            if sec.ntfy_token:
                hdr["Authorization"] = f"Bearer {sec.ntfy_token}"
            http.post(f"{sec.ntfy_server}/{sec.ntfy_topic}", data=body.encode("utf-8"), headers=hdr)
            ok = True
        except HttpError as e:
            log.warning("ntfy failed: %s", e)
    if sec.telegram_token and sec.telegram_chat_id:
        try:
            http.post(f"https://api.telegram.org/bot{sec.telegram_token}/sendMessage",
                      data={"chat_id": sec.telegram_chat_id, "text": f"*{title}*\n{body}", "disable_web_page_preview": True},
                      headers={"Content-Type": "application/json"})
            ok = True
        except HttpError as e:
            log.warning("telegram failed: %s", e)
    return ok
