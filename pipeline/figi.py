"""CUSIP → ticker mapping via OpenFIGI (Bloomberg's free API), cached permanently in .cache/figi.json.

Limits: without a key 25 requests/min and 10 jobs per request; with a free key 25 per 6 s and 100 jobs."""
from __future__ import annotations

from pathlib import Path

from .http import HttpClient
from .util import chunks, log, read_json, write_json

URL = "https://api.openfigi.com/v3/mapping"
_PREFERRED_TYPES = ("Common Stock", "ADR", "ETP", "REIT", "Ltd Part", "MLP", "Closed-End Fund", "Open-End Fund")


def normalize_ticker(t: str | None) -> str | None:
    if not t:
        return None
    t = t.strip().upper().replace("/", ".").replace(" ", ".")
    return t or None


class FigiMapper:
    def __init__(self, http: HttpClient, cache_dir: Path, api_key: str | None = None):
        self.http = http
        self.path = cache_dir / "figi.json"
        self.cache: dict[str, dict | None] = read_json(self.path, {}) or {}
        self.key = api_key
        self.batch = 100 if api_key else 10
        if api_key:
            http.set_rate("api.openfigi.com", 25 / 6.5)
        self.dirty = False

    def lookup(self, cusip: str) -> dict | None:
        return self.cache.get(cusip.upper())

    def map(self, cusips: list[str], max_requests: int = 60) -> int:
        todo = [c.upper() for c in dict.fromkeys(cusips) if c and c.upper() not in self.cache]
        done = 0
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["X-OPENFIGI-APIKEY"] = self.key
        for batch in chunks(todo, self.batch):
            if done >= max_requests:
                break
            jobs = [{"idType": "ID_CUSIP", "idValue": c, "exchCode": "US"} for c in batch]
            try:
                res = self.http.post(URL, data=jobs, headers=headers).json()
                if not isinstance(res, list) or len(res) != len(batch):
                    raise ValueError("unexpected OpenFIGI response shape")
            except Exception as e:  # noqa: BLE001 – unmapped CUSIPs stay uncached and are retried next run
                log.warning("OpenFIGI failed (%s); will retry next run", e)
                break
            done += 1
            for cusip, item in zip(batch, res):
                if isinstance(item, dict) and item.get("error"):   # per-item error (not "no match"): retry later
                    continue
                data = (item or {}).get("data") or []
                pick = None
                for want in _PREFERRED_TYPES:
                    pick = next((d for d in data if d.get("securityType") == want or d.get("securityType2") == want), None)
                    if pick:
                        break
                pick = pick or (data[0] if data else None)
                self.cache[cusip] = ({"ticker": normalize_ticker(pick.get("ticker")), "name": pick.get("name"),
                                      "type": pick.get("securityType2") or pick.get("securityType"),
                                      "figi": pick.get("figi")} if pick else None)
                self.dirty = True
        return len(todo)

    def save(self) -> None:
        if self.dirty:
            write_json(self.path, self.cache)
            self.dirty = False
