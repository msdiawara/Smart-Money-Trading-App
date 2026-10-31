"""HTTP client with per-host rate limits, retries/backoff, Retry-After support and per-host stats.

Every outbound request in the pipeline goes through here so we can (a) stay inside each provider's
published limits (SEC: 10 req/s; OpenFIGI: 25/min anonymous; eToro: 60/min ...), and (b) report
per-source health in the app.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import urlparse

import requests

from .util import log, redact


@dataclass
class Response:
    status: int
    headers: dict
    content: bytes
    url: str

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def text(self, encoding: str = "utf-8") -> str:
        return self.content.decode(encoding, errors="replace")

    def json(self) -> Any:
        import json
        return json.loads(self.content.decode("utf-8", errors="replace"))


class HttpError(RuntimeError):
    def __init__(self, status: int, url: str, body: str = "", headers: dict | None = None):
        super().__init__(redact(f"HTTP {status} for {url}: {body[:200]}"))
        self.status = status
        self.url = redact(url)
        self.body = redact(body[:600])
        # only the few headers a caller may act on (never auth headers)
        keep = ("retry-after", "x-should-retry", "request-id", "anthropic-ratelimit-requests-remaining",
                "anthropic-ratelimit-tokens-remaining")
        self.headers = {k.lower(): v for k, v in (headers or {}).items() if k.lower() in keep}


# requests per second allowed per host (conservative vs. published limits)
DEFAULT_RATES = {
    "www.sec.gov": 8.0,                 # SEC fair access: max 10/s for all SEC hosts combined
    "data.sec.gov": 8.0,
    "efts.sec.gov": 4.0,
    "api.openfigi.com": 25 / 60.0,      # 25 per minute without key (raised when a key is set)
    "public-api.etoro.com": 0.9,        # 60/min
    "data.alpaca.markets": 3.0,         # 200/min
    "query1.finance.yahoo.com": 1.0,
    "query2.finance.yahoo.com": 1.0,
    "disclosures-clerk.house.gov": 2.0,
    "efdsearch.senate.gov": 1.0,
    "financialmodelingprep.com": 2.0,
    "api.quiverquant.com": 1.0,
    "api.anthropic.com": 2.0,
    "api.stlouisfed.org": 1.5,          # FRED: 120 requests/minute
    "cdn.finra.org": 1.0,
    "assets.ark-funds.com": 1.0,
    "www.alphavantage.co": 0.2,         # free key: 25 requests/day (the app uses 1)
    "api.massive.com": 5 / 63.0,       # free Basic plan: 5 requests per minute
    "api.polygon.io": 5 / 63.0,
    "ntfy.sh": 0.5,
    "api.telegram.org": 1.0,
    "api.gdeltproject.org": 1 / 5.5,     # GDELT asks for at most one request every 5 seconds
    "api.github.com": 1.0,
}
# hosts sharing one budget (SEC counts all of its hosts together)
HOST_GROUPS = {"www.sec.gov": "sec", "data.sec.gov": "sec", "efts.sec.gov": "sec"}
GROUP_RATES = {"sec": 8.0}


@dataclass
class HostStats:
    requests: int = 0
    errors: int = 0
    retries: int = 0
    last_status: int | None = None
    last_error: str | None = None
    seconds: float = 0.0


class _Bucket:
    def __init__(self, rate: float):
        self.interval = 1.0 / rate if rate > 0 else 0.0
        self.next_at = 0.0
        self.lock = threading.Lock()

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            if now < self.next_at:
                time.sleep(self.next_at - now)
                now = self.next_at
            self.next_at = now + self.interval


Transport = Callable[[str, str, dict, Any, float], Response]


def _requests_transport(session: requests.Session) -> Transport:
    def send(method: str, url: str, headers: dict, data: Any, timeout: float) -> Response:
        kw: dict = {"headers": headers, "timeout": timeout}
        if data is not None:
            if isinstance(data, (dict, list)) and headers.get("Content-Type", "").startswith("application/json"):
                kw["json"] = data
            else:
                kw["data"] = data
        r = session.request(method, url, **kw)
        return Response(r.status_code, dict(r.headers), r.content, r.url)
    return send


@dataclass
class HttpClient:
    sec_user_agent: str
    transport: Transport | None = None
    rates: dict = field(default_factory=lambda: dict(DEFAULT_RATES))
    max_retries: int = 4
    timeout: float = 30.0
    stats: dict = field(default_factory=dict)
    deadline: float | None = None   # time.monotonic() after which we stop sleeping for retries

    def __post_init__(self) -> None:
        self._session = requests.Session()
        if self.transport is None:
            self.transport = _requests_transport(self._session)
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def set_rate(self, host: str, per_second: float) -> None:
        self.rates[host] = per_second
        self._buckets.pop(host, None)

    def _bucket(self, host: str) -> _Bucket:
        key = HOST_GROUPS.get(host, host)
        with self._lock:
            if key not in self._buckets:
                rate = GROUP_RATES.get(key) or self.rates.get(host, 2.0)
                self._buckets[key] = _Bucket(rate)
            return self._buckets[key]

    def _headers_for(self, host: str, extra: dict | None) -> dict:
        h = {"Accept-Encoding": "gzip, deflate"}
        if host.endswith("sec.gov"):
            # SEC requires a declared User-Agent with contact info.
            h["User-Agent"] = self.sec_user_agent
            h["Host"] = host
        else:
            h["User-Agent"] = "SmartMoneyRadar/1.0 (personal research tool)"
        if host == "public-api.etoro.com":
            h["x-request-id"] = str(uuid.uuid4())
        if extra:
            h.update(extra)
        return h

    def request(self, method: str, url: str, headers: dict | None = None, data: Any = None,
                ok_statuses: tuple = (), timeout: float | None = None, retries: int | None = None) -> Response:
        """`retries=0` for requests that must never be re-sent automatically (a billed API call whose response
        was lost would be billed twice); the caller then decides."""
        max_retries = self.max_retries if retries is None else retries
        host = urlparse(url).hostname or ""
        st = self.stats.setdefault(host, HostStats())
        bucket = self._bucket(host)
        attempt = 0
        backoff = 2.0
        while True:
            bucket.wait()
            hdrs = self._headers_for(host, headers)
            t0 = time.monotonic()
            st.requests += 1
            try:
                resp = self.transport(method, url, hdrs, data, timeout or self.timeout)
            except requests.RequestException as e:  # network error
                st.seconds += time.monotonic() - t0
                st.errors += 1
                st.last_error = redact(f"{type(e).__name__}: {e}")[:300]
                if attempt >= max_retries or not self._can_wait(backoff):
                    raise
                attempt += 1
                st.retries += 1
                time.sleep(backoff)
                backoff *= 2
                continue
            st.seconds += time.monotonic() - t0
            st.last_status = resp.status
            if resp.ok or resp.status in ok_statuses:
                return resp
            retryable = resp.status in (429, 500, 502, 503, 504)
            if retryable and attempt < max_retries:
                attempt += 1
                st.retries += 1
                wait = backoff
                ra = resp.headers.get("Retry-After") or resp.headers.get("retry-after")
                if ra:
                    try:
                        wait = min(120.0, max(1.0, float(ra)))
                    except ValueError:
                        pass
                if host.endswith("sec.gov") and resp.status == 429:
                    wait = max(wait, 10.0)  # SEC throttles by IP; back off hard
                if not self._can_wait(wait):
                    st.errors += 1
                    st.last_error = f"HTTP {resp.status} (out of time to retry)"
                    raise HttpError(resp.status, url, "run time budget exhausted while retrying", resp.headers)
                log.info("retry %s in %.0fs (HTTP %s)", redact(url)[:120], wait, resp.status)
                time.sleep(wait)
                backoff *= 2
                continue
            st.errors += 1
            st.last_error = f"HTTP {resp.status}"
            raise HttpError(resp.status, url, resp.text()[:600], resp.headers)

    def _can_wait(self, seconds: float) -> bool:
        return self.deadline is None or time.monotonic() + seconds + 20 < self.deadline

    def get(self, url: str, **kw) -> Response:
        return self.request("GET", url, **kw)

    def post(self, url: str, data: Any = None, **kw) -> Response:
        return self.request("POST", url, data=data, **kw)

    def get_json(self, url: str, **kw) -> Any:
        return self.get(url, **kw).json()

    def stats_summary(self) -> dict:
        return {h: vars(s) for h, s in self.stats.items()}
