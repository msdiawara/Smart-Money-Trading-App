"""Summarise tests/sim_learning.py results: accuracy with learning vs without, quarter by quarter."""
import json
import sys

import numpy as np

for path in sys.argv[1:]:
    d = json.load(open(path))
    ev = d["evals"]
    print(f"\n=== {path}  K={d['K']}  stocks={d['N']}  days={d['DAYS']}  ({d['secs']:.0f}s)")
    months = sorted({e["m"] for e in ev})
    blocks = [(0, 2), (3, 5), (6, 8), (9, 11), (12, 14), (15, 19)]
    for h in (5, 21, 63):
        print(f"  horizon {h} sessions: 80%-range coverage and skill vs naive (learn | no learning)")
        for a, b in blocks:
            rows = [e for e in ev if e["h"] == h and a <= e["m"] <= b]
            if not rows:
                continue
            out = []
            for v in ("learn", "base"):
                rr = [e for e in rows if e["v"] == v]
                cov = np.mean([e["in80"] for e in rr])
                sk = 1 - sum(e["qs"] for e in rr) / sum(e["qsn"] for e in rr)
                qs = np.mean([e["qs"] for e in rr])
                out.append((cov, sk, qs))
            (cl, sl, ql), (cb, sb, qb) = out
            print(f"    months {a:2d}-{b:2d}: cover {cl:5.1%} | {cb:5.1%}   skill {sl:+6.2%} | {sb:+6.2%}"
                  f"   score better by {(qb - ql) / qb:+6.2%}  (n={len(rows) // 2})")
    on = [t for t in d["timeline"] if (t["tilt"] or {}).get("on")]
    first = on[0]["day"] if on else None
    print(f"  signal tilt first on at day {first}; final status: {d['timeline'][-1]['tilt'].get('status')}")
    last = d["timeline"][-1]
    print("  final calibration:", {k: (v["status"], v["from"], v["scale"], v["shift"]) for k, v in last["calib"].items()})
    print("  final tilt:", {k: last["tilt"].get(k) for k in ("t", "test_gain", "recent_gain", "test_months", "weights")})
