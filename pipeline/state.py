"""Persistent pipeline state.

Stored in .cache/ (restored/saved by actions/cache between GitHub Actions runs) so that each 5-minute run
knows which filings it has already processed, what the previous eToro snapshots looked like, which alerts
went out, etc. Large caches (price history, CUSIP→ticker map, parsed 13F tables) are separate files.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

from .util import iso, log, now_utc, parse_iso, read_json, write_json

STATE_VERSION = 1


def empty_state() -> dict:
    return {
        "version": STATE_VERSION,
        "created": iso(now_utc()),
        "jobs": {},          # job name -> {last_run, last_ok, error, items, note}
        "seen": {},          # namespace -> {key: iso_time_seen}
        "events": [],        # canonical events, newest first
        "etoro": {},
        "funds": {},
        "alerts": {"sent": {}, "day": None, "count": 0, "queue": []},
        "quotes": {},        # ticker -> price & risk snapshot
        "track": {},         # leaderboards
        "runs": [],          # last N run summaries
    }


def load_state(cache_dir: Path) -> dict:
    st = read_json(cache_dir / "state.json.gz")
    if not isinstance(st, dict) or st.get("version") != STATE_VERSION:
        if st:
            log.warning("state version mismatch; starting fresh")
        st = empty_state()
    base = empty_state()
    for k, v in base.items():
        st.setdefault(k, v)
    return st


def save_state(cache_dir: Path, state: dict) -> int:
    prune_seen(state)
    return write_json(cache_dir / "state.json.gz", state)


def mark_seen(state: dict, ns: str, key: str) -> None:
    state["seen"].setdefault(ns, {})[key] = iso(now_utc())


def is_seen(state: dict, ns: str, key: str) -> bool:
    return key in state["seen"].get(ns, {})


def prune_seen(state: dict, keep_days: int = 400) -> None:
    cutoff = now_utc() - dt.timedelta(days=keep_days)
    for ns, d in state["seen"].items():
        if len(d) < 5000:
            continue
        for k in [k for k, t in d.items() if (parse_iso(t) or now_utc()) < cutoff]:
            d.pop(k, None)


# ---------------------------------------------------------------- job scheduling

def job_due(state: dict, name: str, every_minutes: float, force: bool = False) -> bool:
    if force:
        return True
    j = state["jobs"].get(name)
    if not j or not j.get("last_run"):
        return True
    last = parse_iso(j["last_run"])
    if last is None:
        return True
    every = every_minutes
    fails = int(j.get("consecutive_failures") or 0)
    if fails:  # a failed slow job (daily/weekly) retries in 15, 30, 60… minutes instead of waiting a full cycle
        every = min(every_minutes, 15 * 2 ** min(fails - 1, 5))
    # allow 30s of slack so a 5-minute job isn't skipped because cron fired at 4m55s
    return (now_utc() - last).total_seconds() >= every * 60 - 30


def job_record(state: dict, name: str, ok: bool, items: int = 0, error: str | None = None,
               note: str | None = None, extra: dict | None = None) -> None:
    j = state["jobs"].setdefault(name, {})
    t = iso(now_utc())
    j["last_run"] = t
    if ok:
        j["last_ok"] = t
        j["error"] = None
        j["consecutive_failures"] = 0
    else:
        j["error"] = (error or "failed")[:500]
        j["consecutive_failures"] = int(j.get("consecutive_failures") or 0) + 1
    j["items"] = items
    if note is not None:
        j["note"] = note
    if extra:
        j.update(extra)


def get_ns(state: dict, ns: str) -> dict[str, Any]:
    return state.setdefault(ns, {})
