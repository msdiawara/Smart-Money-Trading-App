"""Loads config/config.yaml plus secrets from environment variables (GitHub Actions secrets)."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str) -> str | None:
    v = os.environ.get(name, "").strip()
    return v or None


@dataclass
class Secrets:
    sec_contact_email: str | None = None
    alpaca_key_id: str | None = None
    alpaca_secret: str | None = None
    massive_key: str | None = None
    openfigi_key: str | None = None
    etoro_api_key: str | None = None
    etoro_user_key: str | None = None
    fmp_key: str | None = None
    quiver_key: str | None = None
    anthropic_key: str | None = None
    fred_key: str | None = None
    alphavantage_key: str | None = None
    ntfy_topic: str | None = None
    ntfy_server: str = "https://ntfy.sh"
    ntfy_token: str | None = None
    telegram_token: str | None = None
    telegram_chat_id: str | None = None
    site_url: str | None = None
    heartbeat_url: str | None = None

    @classmethod
    def from_env(cls) -> "Secrets":
        site = _env("SITE_URL")
        repo = _env("GITHUB_REPOSITORY")  # owner/name, set automatically inside GitHub Actions
        if not site and repo and "/" in repo:
            owner, name = repo.split("/", 1)
            site = f"https://{owner.lower()}.github.io/{name}/"
        return cls(
            sec_contact_email=_env("SEC_CONTACT_EMAIL"),
            alpaca_key_id=_env("ALPACA_KEY_ID"),
            alpaca_secret=_env("ALPACA_SECRET_KEY"),
            massive_key=_env("MASSIVE_API_KEY") or _env("POLYGON_API_KEY"),
            openfigi_key=_env("OPENFIGI_API_KEY"),
            etoro_api_key=_env("ETORO_API_KEY"),
            etoro_user_key=_env("ETORO_USER_KEY"),
            fmp_key=_env("FMP_API_KEY"),
            quiver_key=_env("QUIVER_API_KEY"),
            anthropic_key=_env("ANTHROPIC_API_KEY"),
            fred_key=_env("FRED_API_KEY"),
            alphavantage_key=_env("ALPHAVANTAGE_API_KEY"),
            ntfy_topic=_env("NTFY_TOPIC"),
            ntfy_server=(_env("NTFY_SERVER") or "https://ntfy.sh").rstrip("/"),
            ntfy_token=_env("NTFY_TOKEN"),
            telegram_token=_env("TELEGRAM_BOT_TOKEN"),
            telegram_chat_id=_env("TELEGRAM_CHAT_ID"),
            site_url=site,
            heartbeat_url=_env("HEARTBEAT_URL"),
        )

    def register_all(self) -> None:
        from .util import register_secret
        for v in (self.alpaca_key_id, self.alpaca_secret, self.massive_key, self.openfigi_key, self.etoro_api_key,
                  self.etoro_user_key, self.fmp_key, self.quiver_key, self.anthropic_key, self.fred_key, self.alphavantage_key, self.ntfy_topic,
                  self.ntfy_token, self.telegram_token, self.telegram_chat_id, self.heartbeat_url):
            register_secret(v)
        if self.has_sec_contact():
            register_secret(self.sec_contact_email)     # personal: never in a log, note or published file

    def has_sec_contact(self) -> bool:
        return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", self.sec_contact_email or ""))

    def sec_user_agent(self) -> str:
        email = self.sec_contact_email or ""
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
            # SEC rejects undeclared tools; run anyway but health will flag it.
            email = "missing-contact@example.com"
        return f"SmartMoneyRadar personal-research {email}"

    def present(self) -> dict[str, bool]:
        """Which optional integrations are configured (shown in Data Health, never the values)."""
        return {
            "sec_contact_email": bool(self.sec_contact_email),
            "massive": bool(self.massive_key),
            "alpaca": bool(self.alpaca_key_id and self.alpaca_secret),
            "openfigi": bool(self.openfigi_key),
            "etoro": bool(self.etoro_api_key and self.etoro_user_key),
            "fmp": bool(self.fmp_key),
            "quiver": bool(self.quiver_key),
            "claude_api": bool(self.anthropic_key) or os.environ.get("HAS_ANTHROPIC_KEY", "").lower() == "true",
            "fred": bool(self.fred_key),
            "alphavantage": bool(self.alphavantage_key),
            "ntfy": bool(self.ntfy_topic),
            "telegram": bool(self.telegram_token and self.telegram_chat_id),
            "heartbeat": bool(self.heartbeat_url),
        }


DEFAULTS: dict[str, Any] = {
    "app": {"name": "Smart Money Radar", "max_events": 2500, "feed_days": 120},
    "alerts": {"min_score": 70, "watch_tickers": [], "watch_people": [], "max_per_run": 6,
               "max_per_day": 60, "quiet_hours_et": [23, 6]},
    "insiders": {"min_value_usd": 0, "show_non_discretionary": True},
    "congress": {"house": True, "senate": True, "leadership_extra": []},
    "etoro": {"enabled": False, "period": "AbsOneYear", "top_n": 20, "max_risk_score": 6,
              "min_trades": 30, "min_copiers": 100},
    "funds": {"top_positions_for_backtest": 50, "list": []},
    "refresh": {"insiders_live": 5, "activists_live": 5, "etoro": 15, "congress_house": 30,
                "congress_senate": 60, "funds": 360, "funds_deadline_window": 30,
                "prices_market_hours": 15, "prices_off_hours": 120, "daily_backstop": 1440,
                "splits": 720, "track_records": 1440, "fundamentals": 1440, "run_budget_minutes": 9,
                "predictability": 1440, "market_context": 60, "macro": 720, "ideas": 60, "ledger": 360,
                "short_interest": 720, "earnings": 1440, "sale_notices": 10, "etf_daily": 120,
                "short_volume": 180, "companies": 30, "filings_live": 10, "headlines": 60, "forecasts": 30,
                "forecast_grading": 360, "backtest": 1440},
    "analyst": {},
}


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        elif v is not None:
            out[k] = v
    return out


@dataclass
class Settings:
    config: dict
    secrets: Secrets
    root: Path = ROOT
    cache_dir: Path = field(default_factory=lambda: ROOT / ".cache")
    out_dir: Path = field(default_factory=lambda: ROOT / "web" / "data")

    @classmethod
    def load(cls, config_path: Path | None = None, cache_dir: Path | None = None,
             out_dir: Path | None = None) -> "Settings":
        path = config_path or (ROOT / "config" / "config.yaml")
        raw = {}
        if path.exists():
            with open(path, encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        cfg = _merge(DEFAULTS, raw)
        s = cls(config=cfg, secrets=Secrets.from_env())
        s.secrets.register_all()
        if cache_dir:
            s.cache_dir = cache_dir
        if out_dir:
            s.out_dir = out_dir
        s.cache_dir.mkdir(parents=True, exist_ok=True)
        s.out_dir.mkdir(parents=True, exist_ok=True)
        return s

    @property
    def heavy_dir(self) -> Path:
        """Big, slowly-changing caches (13F tables, SEC insider data sets, long price history).
        GitHub Actions only re-saves this folder when something in it changed (see .dirty)."""
        d = self.cache_dir / "heavy"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def mark_heavy_dirty(self) -> None:
        (self.heavy_dir / ".dirty").write_text("1")

    def c(self, *keys, default=None):
        node: Any = self.config
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node
