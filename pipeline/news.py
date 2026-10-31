"""Recent news for each stock.

1. Official company announcements: every 8-K the company files (earnings, deals, executive changes, delisting
   notices, restatements…), with the SEC's own item numbers translated to plain English and red flags marked.
   From the submissions API (cached with the company's financials) plus EDGAR's live feed for minutes-old ones.
2. Headlines from the GDELT Project's DOC 2.0 API: free, no key, open for any use with a citation and link
   (gdeltproject.org/about.html#termsofuse). Searched by company name, English-language sources, reputable
   financial outlets first. GDELT asks for at most one request every 5 seconds, so headlines refresh in batches.
3. Optional: Massive (ex-Polygon) news — OFF by default, because Massive's free/individual plans are a personal
   licence and this app's website is public.
4. Claude's news read (from the analyst workflow's forecast task, with sources) is merged in elsewhere.
"""
from __future__ import annotations

import datetime as dt
import html
import json
import re
import time
from pathlib import Path
from urllib.parse import quote, urlparse

from .http import HttpClient, HttpError
from .util import iso, now_utc, parse_iso, read_json, write_json

GDELT = "https://api.gdeltproject.org/api/v2/doc/doc"
MASSIVE_NEWS = "https://api.massive.com/v2/reference/news"
CACHE = "news.json.gz"

ITEMS = {
    "1.01": ("Signed a major agreement", "info"), "1.02": ("Ended a major agreement", "warn"),
    "1.03": ("Bankruptcy or receivership", "red"), "1.04": ("Mine safety notice", "info"),
    "1.05": ("Material cybersecurity incident", "warn"),
    "2.01": ("Completed an acquisition or sale of assets", "info"), "2.02": ("Quarterly results", "info"),
    "2.03": ("Took on new debt or obligations", "info"),
    "2.04": ("Debt payment accelerated (triggering event)", "red"),
    "2.05": ("Restructuring or layoffs (exit costs)", "warn"), "2.06": ("Material impairment (write-down)", "warn"),
    "3.01": ("Delisting notice / failed a listing rule", "red"), "3.02": ("Sold unregistered shares (dilution)", "warn"),
    "3.03": ("Changed shareholders' rights", "warn"), "4.01": ("Changed auditor", "warn"),
    "4.02": ("Past financial statements can't be relied on (restatement)", "red"),
    "5.01": ("Change in control", "info"), "5.02": ("Director or officer change / pay arrangement", "info"),
    "5.03": ("Bylaws or fiscal-year change", "info"), "5.04": ("Benefit-plan trading suspension", "info"),
    "5.05": ("Code of ethics change or waiver", "info"), "5.06": ("Shell-company status change", "warn"),
    "5.07": ("Shareholder vote results", "info"), "5.08": ("Shareholder director nominations", "info"),
    "7.01": ("Regulation FD disclosure (investor presentation, update)", "info"), "8.01": ("Other material event", "info"),
    "9.01": ("Financial statements and exhibits", "skip"),
}
RED_ITEMS = {k for k, (_, s) in ITEMS.items() if s == "red"}

RATE_LIMIT = re.compile(r"rate limit|too many requests|limit requests|please limit|slow down", re.I)

# reputable outlets shown first (GDELT indexes tens of thousands of sites). Press-release wires are not on the list:
# they carry what companies (or anyone paying) want said, so they rank with everything else.
GOOD_DOMAINS = {
    "reuters.com", "apnews.com", "bloomberg.com", "wsj.com", "ft.com", "cnbc.com", "marketwatch.com", "barrons.com",
    "finance.yahoo.com", "nytimes.com", "washingtonpost.com", "economist.com", "axios.com", "fortune.com",
    "businessinsider.com", "forbes.com", "techcrunch.com", "theverge.com", "investors.com", "morningstar.com",
    "fool.com", "seekingalpha.com", "zacks.com", "investing.com", "thestreet.com", "kiplinger.com", "bbc.com",
    "bbc.co.uk", "theguardian.com", "cnn.com", "foxbusiness.com", "nasdaq.com", "benzinga.com", "fiercebiotech.com",
    "statnews.com", "wired.com", "arstechnica.com", "semafor.com", "theinformation.com", "latimes.com",
    "sec.gov", "nikkei.com", "asia.nikkei.com",
    "scmp.com", "electrek.co", "tomshardware.com", "anandtech.com", "digitimes.com",
}
_SUFFIX = re.compile(r"[\s,]+(inc|incorporated|corp|corporation|co|company|ltd|limited|plc|holdings?|group|"
                     r"s\.?a|n\.?v|a\.?g|se|l\.?p|llc|trust|the|class [a-c]|common stock|/[a-z]{2}/?)\.?$", re.I)
_GENERIC = {"target", "block", "apple", "visa", "ford", "oracle", "gap", "match", "snap", "square", "shell", "unity",
            "zoom", "dell", "intel", "arm", "box", "coin", "fox", "news", "best", "general", "first", "united",
            "american", "international", "national", "global", "energy", "capital", "financial", "royal", "southern"}


def clean_name(name: str | None) -> str | None:
    if not name:
        return None
    s = html.unescape(str(name)).strip()
    s = re.sub(r"\s+", " ", s)
    for _ in range(4):
        s2 = _SUFFIX.sub("", s).strip(" ,.&")
        if s2 == s:
            break
        s = s2
    if not s:
        return None
    if s.isupper() and len(s) > 4:                       # "NVIDIA CORP" → "Nvidia"; keep short acronyms (IBM, AMD)
        s = " ".join(w if len(w) <= 3 else w.capitalize() for w in s.split())
    return s


def gdelt_query(name: str | None, ticker: str | None) -> str | None:
    n = clean_name(name)
    if not n or len(n) < 3:
        return None
    n = n.replace('"', "")
    terms = [f'"{n}"']
    if ticker and len(ticker) >= 3 and ticker.isalpha() and ticker.lower() != n.lower():
        terms.append(f'"{ticker}"')
    core = terms[0] if len(terms) == 1 else "(" + " OR ".join(terms) + ")"
    context = "(stock OR shares OR earnings OR investors OR analysts OR revenue)"
    if n.lower() in _GENERIC or len(n.split()) == 1:
        return f"{core} {context} sourcelang:english"
    return f"{core} sourcelang:english"


def gdelt_url(query: str, days: int = 7, max_records: int = 40) -> str:
    return (f"{GDELT}?query={quote(query)}&mode=ArtList&format=json&maxrecords={max_records}"
            f"&timespan={days}d&sort=DateDesc")


def _seen(s: str | None) -> str | None:
    """GDELT's 'seendate' (20261030T143000Z) → ISO."""
    m = re.match(r"^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z$", s or "")
    if not m:
        return None
    y, mo, d, h, mi, se = (int(x) for x in m.groups())
    return iso(dt.datetime(y, mo, d, h, mi, se, tzinfo=dt.timezone.utc))


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", (t or "").lower())[:90]


def parse_gdelt(text: str, limit: int = 8) -> tuple[list[dict], str | None]:
    """(articles, error). GDELT answers errors and rate limits as plain text with HTTP 200."""
    t = (text or "").strip()
    if not t:
        return [], None
    if not t.startswith("{"):
        return [], re.sub(r"\s+", " ", t)[:200]
    try:
        js = json.loads(t)
    except ValueError:
        return [], "unreadable response"
    arts = js.get("articles") or []
    seen_titles: set[str] = set()
    rows = []
    for a in arts:
        url, title = a.get("url"), html.unescape(a.get("title") or "").strip()
        if not url or not title or not str(url).startswith("https://"):
            continue
        key = _norm_title(title)
        if key in seen_titles:
            continue
        seen_titles.add(key)
        dom = (a.get("domain") or urlparse(url).hostname or "").lower().removeprefix("www.")
        rows.append({"title": title[:220], "url": url, "domain": dom, "seen": _seen(a.get("seendate")),
                     "country": a.get("sourcecountry"), "good": _is_good(dom)})
    rows.sort(key=lambda r: (not r["good"], -(_ts(r["seen"]))))
    good = [r for r in rows if r["good"]]
    pick = good[:limit] if len(good) >= 3 else (good + [r for r in rows if not r["good"]])[:limit]
    pick.sort(key=lambda r: -_ts(r["seen"]))
    return pick, None


def _ts(s: str | None) -> float:
    d = parse_iso(s) if s else None
    return d.timestamp() if d else 0.0


def _is_good(dom: str) -> bool:
    return dom in GOOD_DOMAINS or any(dom.endswith("." + g) for g in GOOD_DOMAINS)


def load(cache_dir: Path) -> dict:
    c = read_json(cache_dir / CACHE)
    if not isinstance(c, dict):
        c = {}
    c.setdefault("by_ticker", {})
    c.setdefault("live_8k", [])
    return c


def save(cache_dir: Path, c: dict) -> None:
    write_json(cache_dir / CACHE, c)


def refresh_headlines(http: HttpClient, cache_dir: Path, tickers: list[str], names: dict[str, str],
                      budget_seconds: float = 60, per_run: int = 12, max_age_hours: float = 12,
                      massive_key: str | None = None) -> tuple[int, str]:
    """Headlines for the most important tickers whose headlines are oldest, from GDELT. (Massive's news feed isn't
    fetched: its individual plans are a personal, non-display licence and this app's site is public.)"""
    t0 = time.monotonic()
    c = load(cache_dir)
    bt = c["by_ticker"]
    now = now_utc()
    due = []
    for rank, t in enumerate(tickers):
        rec = bt.get(t) or {}
        retry = parse_iso(rec.get("retry_after"))
        if retry and retry > now:                  # a failing ticker backs off instead of blocking the queue
            continue
        f = parse_iso(rec.get("fetched"))
        age = (now - f).total_seconds() / 3600 if f else 1e9
        if age >= max_age_hours * (1 if rank < 60 else 2):
            due.append((0 if f is None else 1, rank, t))
    due.sort()

    def backoff(rec: dict) -> None:
        n = int(rec.get("fails") or 0) + 1
        rec.update(fails=n, retry_after=iso(now + dt.timedelta(hours=min(48, 2 ** n))))
    done, errs, note = 0, 0, ""
    for _, _, t in due[:per_run]:
        if time.monotonic() - t0 > budget_seconds:
            break
        rec = bt.setdefault(t, {})
        q = gdelt_query(names.get(t), t)
        if not q:
            rec.update(fetched=iso(now), items=[], error="no usable company name to search for")
            continue
        left = budget_seconds - (time.monotonic() - t0)
        if left < 8:
            break
        try:
            text = http.get(gdelt_url(q), timeout=min(30.0, max(5.0, left - 2)), retries=0).text()
            items, err = parse_gdelt(text)
            if err and RATE_LIMIT.search(err):
                note = "GDELT asked the app to slow down; continuing next run"
                break
            rec.update(fetched=iso(now), q=q, items=items, error=err, source="GDELT")
            rec.pop("fails", None)
            rec.pop("retry_after", None)
            done += 1
        except HttpError as e:
            errs += 1
            rec.update(error=f"HTTP {e.status}")
            if e.status == 429:
                note = "GDELT rate limit; continuing next run"
                break
            backoff(rec)
        except Exception as e:  # noqa: BLE001 – network trouble: back off this ticker, keep going
            errs += 1
            rec.update(error=f"{type(e).__name__}")
            backoff(rec)
    for t in [t for t in bt if t not in set(tickers)]:          # forget tickers that left the app
        f = parse_iso(bt[t].get("fetched"))
        if not f or (now - f).days > 14:
            bt.pop(t, None)
    save(cache_dir, c)
    have = sum(1 for r in bt.values() if r.get("items"))
    return done, f"{have} stocks with headlines" + (f"; {errs} errors" if errs else "") + (f"; {note}" if note else "")


def massive_news(http: HttpClient, key: str, t: str, limit: int = 8) -> list[dict]:
    d = http.get_json(f"{MASSIVE_NEWS}?ticker={quote(t)}&limit={limit}&order=desc&sort=published_utc&apiKey={key}")
    out = []
    for r in d.get("results") or []:
        u = r.get("article_url")
        if not u or not str(u).startswith("https://"):
            continue
        sent = next((i.get("sentiment") for i in r.get("insights") or [] if i.get("ticker") == t), None)
        out.append({"title": (r.get("title") or "")[:220], "url": u, "domain": (r.get("publisher") or {}).get("name"),
                    "seen": r.get("published_utc"), "sentiment": sent, "good": True})
    return out


# ---------------------------------------------------------------- 8-K filings

_ITEM_RE = re.compile(r"Item\s+(\d\.\d{2})", re.I)


def items_from_summary(summary: str | None) -> list[str]:
    """Item numbers listed in an EDGAR live-feed entry's summary, when EDGAR includes them."""
    return sorted(set(_ITEM_RE.findall(html.unescape(summary or ""))))


def filing_view(k: dict, cik: str) -> dict:
    """One 8-K as the app shows it: plain-English items, severity, link to the filing index."""
    items = [i for i in (k.get("items") or []) if i]
    labels, sev = [], "info"
    order = {"skip": 0, "info": 1, "warn": 2, "red": 3}
    for i in items:
        lab, s = ITEMS.get(i, (f"Item {i}", "info"))
        if s == "skip":
            continue
        labels.append({"item": i, "label": lab, "sev": s})
        if order[s] > order[sev]:
            sev = s
    acc = k.get("acc") or ""
    url = (f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}/{acc}-index.htm"
           if acc and str(cik).isdigit() else None)
    title = labels[0]["label"] if labels else ("Current report (8-K)" if not items else "Financial exhibits")
    if any(x["item"] == "2.02" for x in labels):
        title = "Quarterly results"
    return {"date": k.get("d"), "at": k.get("at"), "form": k.get("form") or "8-K", "items": labels, "sev": sev,
            "title": title, "url": url, "acc": acc}


def filings_for(rec: dict | None, live: list[dict], cik: str | None, limit: int = 12) -> list[dict]:
    """Newest-first 8-Ks for one company: the cached submissions list plus anything newer from the live feed."""
    out, seen = [], set()
    for k in sorted([x for x in live if x.get("cik") == cik], key=lambda x: x.get("at") or "", reverse=True):
        if k.get("acc") in seen:
            continue
        seen.add(k.get("acc"))
        out.append(filing_view({"d": (k.get("at") or "")[:10], **k}, cik))
    for k in ((rec or {}).get("sub") or {}).get("k8") or []:
        if k.get("acc") in seen:
            continue
        seen.add(k.get("acc"))
        out.append(filing_view(k, cik or (rec or {}).get("cik") or "0"))
    out.sort(key=lambda f: f.get("date") or "", reverse=True)
    return out[:limit]


def red_flags(filings: list[dict], days: int = 90, today: dt.date | None = None) -> list[dict]:
    today = today or now_utc().date()
    out = []
    for f in filings:
        d = None
        try:
            d = dt.date.fromisoformat((f.get("date") or "")[:10])
        except ValueError:
            pass
        if d is None or (today - d).days > days:
            continue
        for it in f.get("items") or []:
            if it["sev"] == "red":
                out.append({"date": f["date"], "item": it["item"], "label": it["label"], "url": f.get("url")})
    return out


def earnings_dates(filings_raw: list[dict]) -> list[str]:
    """Past earnings-release dates (8-K item 2.02), oldest first: used to measure how the stock reacts."""
    ds = sorted({k.get("d") for k in filings_raw if "2.02" in (k.get("items") or []) and k.get("d")})
    return ds


_ACCEPTED = re.compile(r"^(\d{4}-\d{2}-\d{2})T(\d{2}):(\d{2})")


def earnings_when(filings_raw: list[dict]) -> dict[str, str]:
    """{filing date: "pre" | "post"} for earnings 8-Ks (item 2.02): "pre" when the release came out before that
    day's close, so that session's close already reflects it; "post" when it came after the close, so the next
    session reacts. The SEC's acceptanceDateTime is Eastern clock time despite its "Z" suffix (Apple's 4:30pm ET
    releases read "T16:30:..Z"), and an 8-K accepted after 5:30pm ET is dated the next business day, so an
    acceptance on an earlier calendar day than the filing date also counts as "pre"."""
    from .forecast import close_hm
    from .util import ET
    out = {}
    for k in filings_raw:
        if "2.02" not in (k.get("items") or []) or not k.get("d"):
            continue
        at, d = str(k.get("at") or ""), str(k["d"])[:10]
        if re.search(r"[+-]\d{2}:?\d{2}$", at):             # a real UTC offset: convert to New York time
            p = parse_iso(at)
            if p is None:
                continue
            loc = p.astimezone(ET)
            day, hm = loc.date().isoformat(), (loc.hour, loc.minute)
        else:
            m = _ACCEPTED.match(at)
            if not m:
                continue
            day, hm = m.group(1), (int(m.group(2)), int(m.group(3)))
        try:
            close = close_hm(dt.date.fromisoformat(d))
        except ValueError:
            continue
        out[d] = "pre" if day < d or (day == d and hm < close) else "post"
    return out


def poll_live_8k(http: HttpClient, cache_dir: Path, ciks: set[str], pages: int = 3) -> tuple[list[dict], int]:
    """New 8-Ks for companies the app tracks, minutes after filing. Returns (new filings, entries scanned)."""
    from .sec import ATOM, SEC
    import xml.etree.ElementTree as ET
    c = load(cache_dir)
    have = {k.get("acc") for k in c["live_8k"]}
    seen_all = list(c.get("seen_8k") or [])
    seen_set = set(seen_all)
    new, scanned = [], 0
    for page in range(pages):
        url = (f"{SEC}/cgi-bin/browse-edgar?action=getcurrent&CIK=&type=8-K&company=&dateb=&owner=include"
               f"&start={page * 100}&count=100&output=atom")
        root = ET.fromstring(http.get(url).content)
        entries = root.findall(f"{ATOM}entry")
        if not entries:
            break
        all_known = True
        for e in entries:
            scanned += 1
            title = (e.findtext(f"{ATOM}title") or "").strip()
            m = re.match(r"^(?P<form>8-K(?:/A)?) - (?P<name>.*) \((?P<cik>\d{4,10})\) \((?P<role>[^)]+)\)\s*$", title)
            if not m:
                continue
            am = re.search(r"(\d{10}-\d{2}-\d{6})", e.findtext(f"{ATOM}id") or "")
            if not am:
                continue
            acc = am.group(1)
            if acc in seen_set:
                continue
            all_known = False
            seen_set.add(acc)
            seen_all.append(acc)
            cik = m.group("cik").zfill(10)
            if cik not in ciks or acc in have:
                continue
            have.add(acc)
            new.append({"cik": cik, "acc": acc, "form": m.group("form"), "name": html.unescape(m.group("name")),
                        "at": e.findtext(f"{ATOM}updated"), "items": items_from_summary(e.findtext(f"{ATOM}summary"))})
        if all_known or len(entries) < 100:
            break
    cutoff = (now_utc() - dt.timedelta(days=5)).date().isoformat()
    c["live_8k"] = sorted([k for k in c["live_8k"] if (k.get("at") or "")[:10] >= cutoff] + new,
                          key=lambda k: k.get("at") or "", reverse=True)[:600]
    c["seen_8k"] = seen_all[-1500:]
    save(cache_dir, c)
    return new, scanned
