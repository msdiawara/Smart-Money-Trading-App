"""Small shared helpers: time, parsing, JSON IO, logging."""
from __future__ import annotations

import datetime as dt
import gzip
import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
UTC = dt.timezone.utc

log = logging.getLogger("smr")


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------- time

def now_utc() -> dt.datetime:
    fake = os.environ.get("SMR_FAKE_NOW")  # used by tests / offline demo runs
    if fake:
        return dt.datetime.fromisoformat(fake).astimezone(UTC)
    return dt.datetime.now(UTC)


def now_et() -> dt.datetime:
    return now_utc().astimezone(ET)


def iso(ts: dt.datetime | None) -> str | None:
    if ts is None:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_iso(s: str | None) -> dt.datetime | None:
    if not s:
        return None
    try:
        s = s.strip().replace("Z", "+00:00")
        d = dt.datetime.fromisoformat(s)
        if d.tzinfo is None:
            d = d.replace(tzinfo=ET)
        return d
    except ValueError:
        return None


_DATE_PATTERNS = [
    (re.compile(r"^(\d{4})-(\d{2})-(\d{2})"), ("y", "m", "d")),
    (re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})"), ("m", "d", "y")),
    (re.compile(r"^(\d{1,2})-(\d{1,2})-(\d{4})"), ("m", "d", "y")),
    (re.compile(r"^(\d{4})(\d{2})(\d{2})$"), ("y", "m", "d")),
]


_MON_PATTERN = re.compile(r"^(\d{1,2})-([A-Za-z]{3,9})-(\d{4})")
_MONTHS = {m: i for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT",
                                        "NOV", "DEC"], 1)}


def parse_date(s: Any) -> dt.date | None:
    """Accepts 2026-10-28, 10/28/2026, 10-28-2026, 20261028, datetime/date objects."""
    if s is None:
        return None
    if isinstance(s, dt.datetime):
        return s.date()
    if isinstance(s, dt.date):
        return s
    s = str(s).strip()
    m = _MON_PATTERN.match(s)  # 03-JAN-2023 (SEC DERA data sets)
    if m:
        mon = _MONTHS.get(m.group(2).upper()[:3])
        if mon:
            try:
                return dt.date(int(m.group(3)), mon, int(m.group(1)))
            except ValueError:
                return None
    for rx, order in _DATE_PATTERNS:
        m = rx.match(s)
        if m:
            parts = dict(zip(order, (int(g) for g in m.groups())))
            try:
                return dt.date(parts["y"], parts["m"], parts["d"])
            except ValueError:
                return None
    return None


def date_str(d: dt.date | None) -> str | None:
    return d.isoformat() if d else None


def business_days_between(a: dt.date, b: dt.date) -> int:
    """Number of weekdays after a up to and including b (0 if same day). Holidays ignored."""
    if a is None or b is None:
        return 0
    sign = 1
    if b < a:
        a, b, sign = b, a, -1
    days = 0
    cur = a
    while cur < b:
        cur += dt.timedelta(days=1)
        if cur.weekday() < 5:
            days += 1
    return sign * days


def is_market_hours(ts: dt.datetime | None = None) -> bool:
    """Regular NYSE session (plus 15 minutes for the closing prints), holidays and 1pm early closes included."""
    from .forecast import close_hm, is_trading_day     # lazy: forecast imports this module
    t = (ts or now_utc()).astimezone(ET)
    if not is_trading_day(t.date()):
        return False
    minutes = t.hour * 60 + t.minute
    ch, cm = close_hm(t.date())
    return 9 * 60 + 30 <= minutes <= ch * 60 + cm + 15


# ---------------------------------------------------------------- numbers

def to_float(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return None if (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else float(v)
    s = str(v).strip().replace(",", "").replace("$", "")
    if not s or s.lower() in {"nan", "none", "null", "n/a", "--", "-"}:
        return None
    try:
        f = float(s)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except ValueError:
        return None


def to_int(v: Any) -> int | None:
    f = to_float(v)
    return int(round(f)) if f is not None else None


def to_bool(v: Any) -> bool | None:
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "y", "yes", "x"}:
        return True
    if s in {"0", "false", "f", "n", "no", ""}:
        return False
    return None


def round_sig(x: float | None, nd: int = 4) -> float | None:
    if x is None:
        return None
    try:
        return float(f"{x:.{nd}g}")
    except (TypeError, ValueError):
        return None


def pct(a: float | None, b: float | None) -> float | None:
    """Percent change from b to a."""
    if a is None or b is None or b == 0:
        return None
    return (a / b - 1.0) * 100.0


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def fmt_money(v: float | None) -> str:
    if v is None:
        return "?"
    a = abs(v)
    sign = "-" if v < 0 else ""
    if a >= 1e9:
        return f"{sign}${a/1e9:.2f}B"
    if a >= 1e6:
        return f"{sign}${a/1e6:.2f}M"
    if a >= 1e3:
        return f"{sign}${a/1e3:.0f}K"
    return f"{sign}${a:,.0f}"


# ---------------------------------------------------------------- text

_WS = re.compile(r"\s+")


def clean(s: Any) -> str:
    if s is None:
        return ""
    return _WS.sub(" ", str(s)).strip()


def title_name(raw: str) -> str:
    """SEC owner names are 'LAST FIRST MIDDLE'. Make them readable but keep entities as-is."""
    s = clean(raw)
    if not s:
        return s
    if any(tok in s.upper() for tok in (" LLC", " LP", " L.P.", " INC", " CORP", " TRUST", " FUND", " CAPITAL",
                                         " PARTNERS", " HOLDINGS", " MANAGEMENT", " GROUP", " CO.", " LTD", " PLC",
                                         " ADVISORS", " FOUNDATION", " BANK", " N.A.")):
        return s.title().replace("Llc", "LLC").replace("L.P.", "L.P.").replace(" Lp", " LP").replace("Inc", "Inc")
    return s.title()


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


# ---------------------------------------------------------------- files

def read_json(path: Path, default: Any = None) -> Any:
    try:
        if str(path).endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return json.load(f)
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except (OSError, ValueError) as e:
        log.warning("could not read %s: %s", path, e)
        return default


def finite(o: Any) -> Any:
    """Copy of `o` with NaN/Infinity replaced by None (they aren't valid JSON; browsers refuse the whole file)."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [finite(v) for v in o]
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):          # numpy scalar
        try:
            return finite(o.item())
        except (TypeError, ValueError):
            return o
    return o


def write_json(path: Path, data: Any, compact: bool = True) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    kw = dict(separators=(",", ":"), ensure_ascii=False) if compact else dict(indent=2, ensure_ascii=False)
    try:
        text = json.dumps(data, **kw, default=_json_default, allow_nan=False)
    except ValueError:                       # a NaN/inf slipped in somewhere: clean it rather than publish bad JSON
        text = json.dumps(finite(data), **kw, default=_json_default, allow_nan=False)
    if str(path).endswith(".gz"):
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            f.write(text)
    else:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
    os.replace(tmp, path)
    return path.stat().st_size


def _json_default(o: Any):
    if isinstance(o, (dt.datetime,)):
        return iso(o)
    if isinstance(o, dt.date):
        return o.isoformat()
    if isinstance(o, set):
        return sorted(o)
    if hasattr(o, "item"):  # numpy scalars
        return o.item()
    raise TypeError(f"not JSON serialisable: {type(o)}")


_SECRET_PARAM = re.compile(r"(?i)\b(apikey|api_key|apiKey|token|key|access_token)=([^&\s'\"]+)")
_TG_TOKEN = re.compile(r"bot\d{5,}:[A-Za-z0-9_-]{20,}")


_SECRETS: set[str] = set()


def register_secret(value: str | None) -> None:
    """Any registered value (API keys, ntfy topic, tokens) is masked by redact()."""
    if value and len(value) >= 6:
        _SECRETS.add(value)


def redact(s: object) -> str:
    """Remove API keys and bot tokens from URLs/messages before they reach logs or published JSON."""
    txt = str(s)
    for sec in sorted(_SECRETS, key=len, reverse=True):
        txt = txt.replace(sec, "REDACTED")
    txt = _SECRET_PARAM.sub(lambda m: f"{m.group(1)}=REDACTED", txt)
    return _TG_TOKEN.sub("botREDACTED", txt)


_DECL = re.compile(r"^\s*<\?xml[^>]*\?>", re.S)


def xml_bytes(x: str | bytes) -> bytes:
    """Tolerant XML input: strips BOM/leading blank lines and the declaration (some SEC filings put a newline
    before '<?xml', which ElementTree rejects)."""
    if isinstance(x, bytes):
        x = x.decode("utf-8", errors="replace")
    x = x.lstrip("\ufeff\r\n\t ")
    return _DECL.sub("", x, count=1).lstrip().encode("utf-8")


_ACRONYMS = {"SVP", "EVP", "VP", "GC", "CEO", "CFO", "COO", "CTO", "CAO", "CIO", "CMO", "CLO", "CHRO", "CPO", "CRO",
             "CSO", "CCO", "II", "III", "IV", "US", "USA", "R&D", "AI", "IT", "HR", "LLC", "LP", "PLC", "NA", "N.A."}


def nice_title(s: str) -> str:
    words = clean(s).title().split(" ")
    out = []
    for w in words:
        core = w.strip(",.;:()").upper()
        out.append(w.upper() if core in _ACRONYMS else w)
    return " ".join(out).replace(" And ", " and ").replace(" Of ", " of ").replace(" The ", " the ")


def chunks(seq: list, n: int) -> Iterable[list]:
    for i in range(0, len(seq), n):
        yield seq[i:i + n]
