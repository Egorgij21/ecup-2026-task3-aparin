#!/usr/bin/env python
"""
Does "same calendar window, one year earlier" predict spending beyond recent level?

The proposal: users who bought a lot in Feb-March 2025 may buy a lot again in Feb-March 2026,
so a seasonal year-lag feature could help the test window. This cannot be validated on our
Jun-Oct CV folds (they would need Jun-Oct 2024, which does not exist) -- but it does NOT need
an LB probe either, because one calendar window is present in BOTH years of our history:

    2025-01-15 .. 2025-02-13     and     2026-01-15 .. 2026-02-13

So we can run the exact experiment offline:

    target  y = GMV in 2026-01-15 .. 2026-02-13          (the "this year" window)
    lag     s = GMV in 2025-01-15 .. 2025-02-13          (same calendar window, 1 year back)
    control p = GMV in 2025-12-16 .. 2026-01-14          (the 30 days immediately before y)

The question is NOT whether s correlates with y -- it will, because spending level is
persistent (split-half 0.65, reports/gifters.log). The question is whether s adds anything
ONCE p AND a longer recent history are already known. That is a partial correlation, and it
is the only version of the question a GBDT cares about: our model already sees 30/60/90/180/
365-day windows, so a year-lag feature must beat all of them to earn its place.

If the partial correlation is ~0, the seasonal year-lag is redundant and no LB submission
should be spent on it.
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


def resid(y, X):
    """residual of y after regressing out the columns of X (with intercept)"""
    A = np.column_stack([np.ones(len(y))] + list(X))
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ coef


p = Panel()
W = lambda a, b: p.wsum("gmv", p.idx(a), p.idx(b))
O = lambda a, b: p.wsum("ord", p.idx(a), p.idx(b))

y = W(date(2026, 1, 15), date(2026, 2, 13))       # "this year" window
s = W(date(2025, 1, 15), date(2025, 2, 13))       # same window, one year earlier
p30 = W(date(2025, 12, 16), date(2026, 1, 14))    # immediately-preceding 30 days
p90 = W(date(2025, 10, 17), date(2026, 1, 14))    # preceding 90 days
p365 = W(date(2025, 1, 15), date(2026, 1, 14))    # preceding 365 days (CONTAINS s)
o90 = O(date(2025, 10, 17), date(2026, 1, 14))

L, S = np.log1p(y), np.log1p(s)
P30, P90, P365, O90 = np.log1p(p30), np.log1p(p90), np.log1p(p365), np.log1p(o90)

hdr("1 -- RAW CORRELATIONS (everything correlates; this is not the question)")
for nm, v in [("year-lag s", S), ("prev 30d", P30), ("prev 90d", P90),
              ("prev 365d", P365), ("orders 90d", O90)]:
    print(f"  corr(log1p(y), {nm:12s}) = {np.corrcoef(L, v)[0, 1]:+.4f}")

hdr("2 -- PARTIAL CORRELATION: does the year-lag survive the controls?")
sets = [("p30", [P30]), ("p30+p90", [P30, P90]), ("p30+p90+ord90", [P30, P90, O90]),
        ("p30+p90+ord90+p365", [P30, P90, O90, P365])]
for nm, ctrl in sets:
    rl, rs = resid(L, ctrl), resid(S, ctrl)
    pc = float(np.corrcoef(rl, rs)[0, 1])
    print(f"  controlling for {nm:22s} partial corr(y, year-lag) = {pc:+.4f}")
    R[f"partial_{nm}"] = pc
print("\n  Note the last row is the honest one: our model already has a 365-day window that")
print("  CONTAINS the year-lag period, so the feature must add signal beyond even that.")

hdr("3 -- INCREMENTAL R^2 IN LOG SPACE (what a feature could actually buy)")
base = [P30, P90, O90, P365]
r_base = resid(L, base)
r_full = resid(L, base + [S])
ss0, ss1 = float((r_base ** 2).sum()), float((r_full ** 2).sum())
print(f"  residual SS without year-lag = {ss0:,.1f}")
print(f"  residual SS with    year-lag = {ss1:,.1f}")
print(f"  incremental R^2 = {1 - ss1 / ss0:.6f}")
rmse0, rmse1 = np.sqrt(ss0 / len(L)), np.sqrt(ss1 / len(L))
print(f"  implied RMSLE-scale change (linear model) = {rmse1 - rmse0:+.5f}")
print(f"  (for reference sigma_noise = 0.00009 and our best CV gain from a feature block "
      f"was -0.00096)")
R["incremental_r2"] = float(1 - ss1 / ss0)
R["implied_rmsle_delta"] = float(rmse1 - rmse0)

hdr("4 -- THE PROPOSED SUBSET TEST, DONE OFFLINE")
print("  'Take users who bought a lot in the same window last year -- do they buy more now?'")
print("  Answered by comparing them to users MATCHED on recent spend.\n")
q = np.quantile(s[s > 0], 0.9)
heavy = s >= q
print(f"  users in the top decile of last-year spend: {int(heavy.sum()):,}")
print(f"  their mean log1p(y)      = {L[heavy].mean():.4f}")
print(f"  everyone else            = {L[~heavy].mean():.4f}")
print(f"  raw difference           = {L[heavy].mean() - L[~heavy].mean():+.4f}  "
      f"<- this is what an LB probe would have measured")
# now match on recent level: compare within deciles of p90
dec = np.digitize(P90, np.quantile(P90, np.linspace(0, 1, 11)[1:-1]))
diffs, wts = [], []
print(f"\n  {'p90 decile':11s} {'n heavy':>9s} {'n other':>9s} {'mean L heavy':>13s} "
      f"{'mean L other':>13s} {'diff':>8s}")
for d in range(10):
    m = dec == d
    a, b = m & heavy, m & ~heavy
    if a.sum() < 200 or b.sum() < 200:
        continue
    df = L[a].mean() - L[b].mean()
    diffs.append(df); wts.append(a.sum())
    print(f"  {d:<11d} {int(a.sum()):>9,} {int(b.sum()):>9,} {L[a].mean():>13.4f} "
          f"{L[b].mean():>13.4f} {df:>+8.4f}")
w = np.array(wts, float)
print(f"\n  LEVEL-MATCHED difference = {np.average(diffs, weights=w):+.4f}   "
      f"(raw was {L[heavy].mean() - L[~heavy].mean():+.4f})")
print("  -> the gap between these two numbers is exactly the confound an LB probe could not")
print("     have separated: 'heavy last year' mostly means 'heavy, full stop'.")
R["raw_diff"] = float(L[heavy].mean() - L[~heavy].mean())
R["matched_diff"] = float(np.average(diffs, weights=w))

(OUT).mkdir(parents=True, exist_ok=True)
(OUT / "yearlag.json").write_text(json.dumps(R, indent=2, default=str))
print("\n  wrote reports/eda/yearlag.json")
