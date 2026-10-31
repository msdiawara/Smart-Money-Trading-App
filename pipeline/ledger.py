"""The forecast ledger: the app's report card.

Every probability the app shows is written down the first time it's shown (never edited afterwards), then graded
63 trading days later against what actually happened: did the stock beat the S&P 500 over that window?
Trades shown without a probability are tracked too, so base rates for every kind of signal build up over time.

From the graded entries the app reports, per source (the statistical model, base rates, Claude's own calls,
the idea screens):
* hit rate: how often the call's direction was right (probability above/below 50%),
* Brier score: mean squared error of the probabilities (0 is perfect; always saying the base rate is the bar
  to beat), and the skill versus that bar,
* calibration: when it said ~60%, did ~60% happen?
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np

from .quant import HOLD, Panel, block_of, clustered_interval, forward_excess
from .util import now_et, parse_date, read_json, write_json

BINS = [0.0, 0.40, 0.45, 0.50, 0.55, 0.60, 1.0001]


def load(cache_dir: Path) -> dict:
    led = read_json(cache_dir / "ledger.json.gz")
    if not isinstance(led, dict) or "entries" not in led:
        led = {"entries": {}}
    return led


def save(cache_dir: Path, ledger: dict) -> None:
    cutoff = (now_et().date() - dt.timedelta(days=3 * 365)).isoformat()
    E = ledger["entries"]
    for k in [k for k, e in E.items() if (e.get("made") or "") < cutoff]:
        E.pop(k, None)
    write_json(cache_dir / "ledger.json.gz", ledger)


def record(ledger: dict, fid: str, *, ticker: str | None, made: str | None, p: float | None, source: str,
           kind: str | None = None, action: str | None = None, stance: str | None = None, horizon: int = HOLD,
           **meta) -> bool:
    """First write wins: a forecast can never be revised after the fact."""
    E = ledger.setdefault("entries", {})
    if fid in E or not ticker or not made:
        return False
    rec = {"t": ticker, "made": made[:10], "p": None if p is None else round(float(p), 4), "src": source,
           "kind": kind, "act": action, "stance": stance, "h": horizon}
    rec.update({k: v for k, v in meta.items() if v is not None})
    E[fid] = rec
    return True


def grade(ledger: dict, panel: Panel | None, store=None, today: dt.date | None = None) -> int:
    """Grade every entry whose 63-trading-day window has closed. Uses the whole-market history when available,
    otherwise the per-ticker price cache (same SPY-calendar method either way)."""
    from .track_records import forward_return

    today = today or now_et().date()
    n = 0
    for e in ledger.get("entries", {}).values():
        if "out" in e:
            continue
        made = parse_date(e.get("made"))
        h = int(e.get("h") or HOLD)
        if not made or (today - made).days < int(h * 1.35):     # can't be complete before h trading days
            continue
        fx = None
        info: dict = {}
        if panel is not None:
            j = panel.ix_after(made)
            r = forward_excess(panel, e["t"], j, h, info) if j is not None else None
            fx = (r[0], r[1]) if r else None
        elif store is not None:
            r = forward_return(store, e["t"], made, h, info)
            fx = (r[0], r[0] - r[1]) if r else None
        if info.get("delisted"):
            e["delisted"] = True            # graded to its last trade (see quant.forward_excess)
        if fx is None:
            if (today - made).days > int(h * 1.35) + 115:   # never priced at all: excluded (delistings are graded)
                e["out"], e["void"] = None, True
                n += 1
            continue
        e["out"] = 1 if fx[1] > 0 else 0
        e["ret"], e["ex"] = round(fx[0], 5), round(fx[1], 5)
        e["graded"] = today.isoformat()
        n += 1
    return n


def _interval(rows: list[dict]) -> tuple[float, float, int]:
    """Forecasts made in the same quarter share the market's moves, so intervals are clustered by 13-week block
    (see quant.clustered_interval): far wider, and honest, compared with treating each forecast as independent."""
    return clustered_interval([r["out"] for r in rows], [block_of(r.get("made")) for r in rows])


def _score(rows: list[dict]) -> dict:
    with_p = [r for r in rows if r.get("p") is not None]
    out = {"n": len(rows), "beat_rate": float(np.mean([r["out"] for r in rows])) if rows else None}
    if rows:
        lo, hi, g = _interval(rows)
        out.update(beat_lo=lo, beat_hi=hi, periods=g, mean_excess=float(np.mean([r.get("ex") or 0 for r in rows])))
    if len(with_p) >= 5:
        p = np.asarray([r["p"] for r in with_p], dtype=float)
        y = np.asarray([r["out"] for r in with_p], dtype=float)
        brier = float(np.mean((p - y) ** 2))
        clim = float(y.mean())
        ref = float(np.mean((clim - y) ** 2))
        # a hit rate only means something when the probabilities actually vary (one constant base rate would
        # make it "how often the base rate's side came true")
        varies = float(np.ptp(p)) >= 0.02
        out.update(n_prob=len(with_p), brier=brier, brier_ref=ref, skill=(1 - brier / ref) if ref > 0 else None,
                   hit_rate=float(np.mean((p >= 0.5) == (y == 1))) if varies else None)
        cal = []
        for a, b in zip(BINS[:-1], BINS[1:]):
            m = (p >= a) & (p < b)
            if m.sum():
                k = int(y[m].sum())
                lo, hi, _ = _interval([r for r, keep in zip(with_p, m) if keep])
                cal.append({"from": a, "to": min(b, 1.0), "pred": float(p[m].mean()), "obs": k / int(m.sum()),
                            "lo": lo, "hi": hi, "n": int(m.sum())})
        out["calibration"] = cal
    return out


def metrics(ledger: dict) -> dict:
    E = list(ledger.get("entries", {}).values())
    graded = [e for e in E if e.get("out") is not None]
    pending = [e for e in E if "out" not in e]
    by_src: dict[str, list] = {}
    by_kind: dict[str, list] = {}
    by_stance: dict[str, list] = {}
    for e in graded:
        by_src.setdefault(e.get("src") or "?", []).append(e)
        by_kind.setdefault(f"{e.get('kind')}:{e.get('act')}", []).append(e)
        if e.get("src") == "claude" and e.get("stance"):
            by_stance.setdefault(e["stance"], []).append(e)
    return {
        "graded": len(graded), "pending": len(pending),
        "pending_by_src": {s: sum(1 for e in pending if e.get("src") == s) for s in {e.get("src") for e in pending}},
        "oldest_pending": (sorted({e["made"] for e in pending}) or [None])[0],
        "by_source": {k: _score(v) for k, v in by_src.items()},
        "by_kind": {k: _score(v) for k, v in by_kind.items() if len(v) >= 10},
        "claude_by_stance": {k: _score(v) for k, v in by_stance.items()},
    }


def kind_rates(ledger: dict, min_n: int = 30) -> dict:
    """Base rates learned from the app's own graded history ('kind:action' → rate), for signal types that have no
    long official history to backtest (fund moves, activist stakes…)."""
    by: dict[str, list] = {}
    for e in ledger.get("entries", {}).values():
        if e.get("out") is None or e.get("src") == "claude" or e.get("src") == "idea":
            continue
        by.setdefault(f"{e.get('kind')}:{e.get('act')}", []).append(e)
    out = {}
    for k, rows in by.items():
        if len(rows) >= min_n:
            kk = int(sum(r["out"] for r in rows))
            lo, hi, _ = _interval(rows)
            out[k] = {"n": len(rows), "p": kk / len(rows), "lo": lo, "hi": hi,
                      "mean_excess": float(np.mean([r.get("ex") or 0 for r in rows])), "from": "app history"}
    return out
