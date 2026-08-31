#!/usr/bin/env python
"""
Are SPIKES repeatable? (level-relative, not "heavy buyers")

The proposal: users who spiked in Feb-May 2025 -- spent far more than is normal FOR THEM --
may spike again in Feb-March 2026, which is the test window.

The exact hypothesis cannot be tested offline: we have only one spring, and the 2026
Feb-March outcome is the hidden test set. That is what makes an LB probe tempting. But two
free offline tests bracket it, and both must be run before spending a submission:

  A. SAME-CALENDAR-WINDOW YEAR-OVER-YEAR. One window exists in both years of our history:
     15 Jan - 13 Feb. Measuring each user's spike in that window in 2025 and again in 2026 --
     against a COMMON reference period, so level cancels -- gives the only direct measurement
     of "is a calendar-positioned bump repeatable for the same user" that this data allows.

  B. SPRING-SPIKER FOLLOW-THROUGH. Take the users who actually spiked in Feb-May 2025 and ask
     what they did in the most recent observable window (Jan-Feb 2026). This is the closest
     free stand-in for the proposed LB probe: same subset, observable outcome.

If A is ~0 and B shows no elevation, an LB probe would spend a submission to measure a
quantity we already know is ~0, and the answer would be confounded anyway.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path("/path/to/ecup")
sys.path.insert(0, str(ROOT / "src"))
from data import Panel                      # noqa: E402

OUT = ROOT / "reports" / "eda"
R: dict = {}


def hdr(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}", flush=True)


p = Panel()


def rate(a: date, b: date):
    """GMV per day over [a,b]"""
    n = (b - a).days + 1
    return p.wsum("gmv", p.idx(a), p.idx(b)) / n


# common reference period, disjoint from both test windows and from spring
REF = (date(2025, 6, 1), date(2025, 12, 15))
ref = rate(*REF)

W25 = (date(2025, 1, 15), date(2025, 2, 13))
W26 = (date(2026, 1, 15), date(2026, 2, 13))
SPRING = (date(2025, 2, 14), date(2025, 5, 31))

hdr("A -- IS A CALENDAR-POSITIONED SPIKE REPEATABLE, SAME USER, SAME WINDOW?")
print(f"  window   : 15 Jan - 13 Feb, measured in 2025 AND 2026")
print(f"  reference: {REF[0]} .. {REF[1]} (common to both, so the user's LEVEL cancels)\n")
s25 = np.log1p(rate(*W25)) - np.log1p(ref)
s26 = np.log1p(rate(*W26)) - np.log1p(ref)
m = ref > 0                                     # reference must exist for the spike to mean anything
print(f"  users with a non-zero reference period: {int(m.sum()):,}")
print(f"  mean spike 2025 = {s25[m].mean():+.4f}   sd {s25[m].std():.4f}")
print(f"  mean spike 2026 = {s26[m].mean():+.4f}   sd {s26[m].std():.4f}")
r = float(np.corrcoef(s25[m], s26[m])[0, 1])
rk = lambda v: np.argsort(np.argsort(v))
rs = float(np.corrcoef(rk(s25[m]), rk(s26[m]))[0, 1])
print(f"\n  corr(spike 2025, spike 2026) = {r:+.4f}   spearman {rs:+.4f}   <- INFLATED, see below")

# Both spikes subtract the SAME reference, so corr(A-C, B-C) carries a spurious +Var(C).
# Partial the reference out of both to get the honest number.
lref = np.log1p(ref)[m]
def _res(v):
    A = np.column_stack([np.ones(v.size), lref])
    c, *_ = np.linalg.lstsq(A, v, rcond=None)
    return v - A @ c
rp = float(np.corrcoef(_res(s25[m]), _res(s26[m]))[0, 1])
print(f"  PARTIALLING OUT the shared reference: corr = {rp:+.4f}")
print(f"  (the drop from {r:+.4f} to {rp:+.4f} is the shared-denominator artefact)")
R["A_partial"] = rp
lv = float(np.corrcoef(np.log1p(rate(*W25))[m], np.log1p(rate(*W26))[m])[0, 1])
print(f"  control -- corr of the raw LEVELS in the same two windows = {lv:+.4f}")
print("\n  -> the control shows how much signal is there when level is NOT removed.")
print("     The spike correlation is what survives once it is.")
R["A"] = {"spike_corr": r, "spearman": rs, "level_control": lv}

hdr("B -- SPRING-2025 SPIKERS: WHAT DID THEY DO IN JAN-FEB 2026?")
print(f"  spring window: {SPRING[0]} .. {SPRING[1]} (contains 23 Feb, 8 Mar, 9 May)")
spring_spike = np.log1p(rate(*SPRING)) - np.log1p(ref)
for pct in [90, 95, 99]:
    thr = np.quantile(spring_spike[m], pct / 100)
    sel = m & (spring_spike >= thr)
    rest = m & (spring_spike < thr)
    print(f"\n  top {100 - pct}% spring spikers (n={int(sel.sum()):,}):")
    print(f"    their mean Jan-Feb 2026 spike  = {s26[sel].mean():+.4f}")
    print(f"    everyone else                  = {s26[rest].mean():+.4f}")
    print(f"    raw difference                 = {s26[sel].mean() - s26[rest].mean():+.4f}"
          f"   <- shares the reference, so inflated")
    # stratify on the shared reference level to remove the artefact
    lr = np.log1p(ref)
    ed = np.quantile(lr[m], np.linspace(0, 1, 11)[1:-1])
    dd = np.digitize(lr, ed)
    ds, ws = [], []
    for g in range(10):
        a, b = sel & (dd == g), rest & (dd == g)
        if a.sum() >= 100 and b.sum() >= 100:
            ds.append(s26[a].mean() - s26[b].mean()); ws.append(a.sum())
    md = float(np.average(ds, weights=np.array(ws, float))) if ds else float("nan")
    print(f"    REFERENCE-MATCHED difference   = {md:+.4f}")
    R.setdefault("B", {})[f"top{100 - pct}pct_raw"] = float(s26[sel].mean() - s26[rest].mean())
    R.setdefault("B", {})[f"top{100 - pct}pct_matched"] = md

hdr("C -- WHAT WOULD AN LB PROBE HAVE COST AND TOLD US?")
print("  A subset probe (predict a constant c on subset S, 0 elsewhere) recovers")
print("      sum_{i in S} log1p(y_i) = (n*E[L^2] + |S|*log1p(c)^2 - n*RMSLE^2) / (2*log1p(c))")
print("  using E[L^2]=10.7584 from the all-zeros probe, so ONE submission gives E[L | S]")
print("  for any chosen S. That machinery works and is already validated.\n")
thr = np.quantile(spring_spike[m], 0.9)
sel = m & (spring_spike >= thr)
print(f"  If we probed S = top-10% spring-2025 spikers (n={int(sel.sum()):,}), the answer would")
print(f"  be a single number: E[log1p(y_2026)|S]. Whether that number is high or low, it is")
print(f"  NOT decomposable into 'seasonal repeat' versus 'these users are simply more active")
print(f"  lately' -- and section B already measures the same contrast on observable data,")
print(f"  for free, with the level controlled by construction.")

(OUT).mkdir(parents=True, exist_ok=True)
(OUT / "spikes.json").write_text(json.dumps(R, indent=2, default=str))
print("\n  wrote reports/eda/spikes.json")
