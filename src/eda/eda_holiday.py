#!/usr/bin/env python
"""
Gift-holiday seasonality: does it need a per-USER feature, or just a global multiplier?

The test window 2026-02-14 .. 2026-03-15 contains BOTH Russian gift holidays:
  * 23 February -- Defender of the Fatherland Day ("men's day")
  * 8 March     -- International Women's Day
DATA.md 5.4 already measured the aggregate lift on the 2025 analogue: the window runs
1.1628x the preceding 30 days, and the spending happens in the RUN-UP, not on the day
(23 Feb itself was 94.8% of baseline, 8 March 93.8%).

Our 5 CV folds are all anchored Jun-Oct 2025, so their target windows contain no gift
holiday at all. CV therefore cannot measure any of this -- it is a structural train/test
mismatch, and this is one of the rare places where we must reason from the calendar analogue
instead. That makes it worth measuring carefully before acting.

The decision this script informs:
  (a) if the lift is roughly UNIFORM across users -> a single global multiplier is the whole
      story, and no feature can help;
  (b) if it is CONCENTRATED in identifiable "gift buyers" -> a per-user responsiveness
      feature is justified, and we need a proxy to build it from.

The persistence test is the crux. We cannot validate a Feb/March feature against a Feb/March
target (no 2024 data, and the 2026 target is hidden). But BOTH the Feb-March gifting window
and the New Year gifting window sit inside our history, so we can ask whether "responds to
gift holidays" is a stable user trait at all.
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path("/path/to/ecup")
sys.path.insert(0, str(ROOT / "src"))
from data import Panel                      # noqa: E402

OUT = ROOT / "reports" / "eda"
R: dict = {}


def hdr(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}", flush=True)


def gini(y):
    y = np.sort(np.asarray(y, np.float64)); n = y.size; s = y.sum()
    return 0.0 if s <= 0 else float(2 * (np.arange(1, n + 1) * y).sum() / (n * s) - (n + 1) / n)


p = Panel()
D = {i: p.dmin + timedelta(days=i) for i in range(p.n_days)}
IDX = {v: k for k, v in D.items()}


def w(col, a: date, b: date):
    return p.wsum(col, p.idx(a), p.idx(b))


# ============================================================================
hdr("1 -- DAILY SHAPE AROUND EACH GIFT HOLIDAY (2025, whole panel)")
daily = np.array([p.wsum("gmv", i, i).sum() for i in range(p.n_days)])
buyers = np.array([(p.wsum("gmv", i, i) > 0).sum() for i in range(p.n_days)])


def show(centre: date, lo: int, hi: int, label: str):
    ci = p.idx(centre)
    base = np.median(daily[max(ci - 45, 0):ci - 20])           # local pre-window baseline
    print(f"\n  {label}  (baseline = median daily GMV 45..20 days before = {base:,.0f})")
    for off in range(lo, hi + 1):
        i = ci + off
        if not (0 <= i < p.n_days):
            continue
        bar = "#" * int(max(0, 40 * (daily[i] / base - 0.8) / 0.6))
        print(f"    {D[i]} {D[i].strftime('%a')} d{off:+3d}  {daily[i]:>12,.0f}  "
              f"{100 * daily[i] / base:6.1f}%  {bar}")


show(date(2025, 2, 23), -14, 4, "23 FEBRUARY 2025 (Defender of the Fatherland)")
show(date(2025, 3, 8), -14, 4, "8 MARCH 2025 (International Women's Day)")
show(date(2025, 12, 31), -21, 3, "NEW YEAR 2025/26")

# ============================================================================
hdr("2 -- HOW MUCH OF THE WINDOW'S GMV IS 'EXCESS' OVER BASELINE?")
H1 = (date(2025, 2, 14), date(2025, 3, 15))       # the exact calendar analogue of the test
B1 = (date(2025, 1, 15), date(2025, 2, 13))       # the 30 days before it
g_h1, g_b1 = w("gmv", *H1), w("gmv", *B1)
print(f"  analogue window {H1[0]}..{H1[1]}: total {g_h1.sum():,.0f}")
print(f"  baseline window {B1[0]}..{B1[1]}: total {g_b1.sum():,.0f}")
print(f"  aggregate lift = {g_h1.sum() / g_b1.sum():.4f}  (DATA.md 5.4 said 1.1628)")
print(f"  excess GMV attributable to the holidays = {g_h1.sum() - g_b1.sum():,.0f} "
      f"({100 * (g_h1.sum() - g_b1.sum()) / g_h1.sum():.1f}% of the window)")

# ============================================================================
hdr("3 -- IS THE LIFT UNIFORM OR CONCENTRATED?")
act = (p.wdays(p.idx(B1[0]), p.idx(H1[1])) > 0)
print(f"  users active anywhere in baseline+holiday span: {int(act.sum()):,}")
lift = np.log1p(g_h1[act]) - np.log1p(g_b1[act])
print(f"  per-user log-lift: mean={lift.mean():+.4f}  sd={lift.std():.4f}")
for q in [1, 5, 25, 50, 75, 95, 99]:
    print(f"    p{q:<3d} {np.quantile(lift, q / 100):+.4f}")
up = g_h1[act] - g_b1[act]
pos = up > 0
print(f"  users who spent MORE in the holiday window: {100 * pos.mean():.1f}%")
print(f"  Gini of the positive excess: {gini(np.maximum(up, 0)):.4f}")
srt = np.sort(np.maximum(up, 0))[::-1]
tot = srt.sum()
for f in [0.01, 0.05, 0.10, 0.25]:
    print(f"    top {100 * f:>4.0f}% of users hold {100 * srt[:int(f * srt.size)].sum() / tot:5.1f}% "
          f"of the excess GMV")
R["lift"] = {"aggregate": float(g_h1.sum() / g_b1.sum()), "mean_log_lift": float(lift.mean()),
             "sd_log_lift": float(lift.std()), "share_up": float(pos.mean())}

# ============================================================================
hdr("4 -- IS 'GIFT BUYER' A PERSISTENT USER TRAIT?  (the crux)")
print("  We cannot validate a Feb/March feature against a Feb/March target -- there is no")
print("  2024 data and the 2026 target is hidden. But both gifting seasons are inside our")
print("  history, so we can ask whether responsiveness is stable at all.\n")
H2 = (date(2025, 11, 25), date(2025, 12, 31))     # New Year gifting run-up
B2 = (date(2025, 10, 15), date(2025, 11, 24))
g_h2, g_b2 = w("gmv", *H2), w("gmv", *B2)
sc = (H2[1] - H2[0]).days + 1, (B2[1] - B2[0]).days + 1
lift2_all = np.log1p(g_h2 / sc[0] * 30) - np.log1p(g_b2 / sc[1] * 30)
lift1_all = np.log1p(g_h1) - np.log1p(g_b1)
both = act & (p.wdays(p.idx(B2[0]), p.idx(H2[1])) > 0)
a, b = lift1_all[both], lift2_all[both]
print(f"  users present in both spans: {int(both.sum()):,}")
print(f"  corr(FebMar lift, NewYear lift)          = {np.corrcoef(a, b)[0, 1]:+.4f}")
rk = lambda v: np.argsort(np.argsort(v))
print(f"  spearman                                  = {np.corrcoef(rk(a), rk(b))[0, 1]:+.4f}")
# control: is any correlation just 'active users move together'?
base_corr = np.corrcoef(np.log1p(g_b1[both]), np.log1p(g_b2[both]))[0, 1]
print(f"  corr of the two BASELINE windows (control) = {base_corr:+.4f}")
print("\n  -> if the lift correlation is near zero while the baseline correlation is high,")
print("     'gift responsiveness' is NOT a stable trait and no per-user feature can capture it.")
R["persistence"] = {"pearson": float(np.corrcoef(a, b)[0, 1]),
                    "spearman": float(np.corrcoef(rk(a), rk(b))[0, 1]),
                    "baseline_control": float(base_corr)}

# ============================================================================
hdr("5 -- DOES LIFT TRACK ANY OBSERVABLE WE ALREADY HAVE?")
ai = p.idx(B1[1])
feats = {
    "log1p(gmv_90)": np.log1p(p.wsum("gmv", ai - 89, ai)),
    "log1p(ord_90)": np.log1p(p.wsum("ord", ai - 89, ai)),
    "active_days_90": p.wdays(ai - 89, ai),
    "aov_90": p.wsum("gmv", ai - 89, ai) / np.maximum(p.wsum("ord", ai - 89, ai), 1),
    "cat_share_90": p.wsum("gmvc", ai - 89, ai) / np.maximum(p.wsum("gmv", ai - 89, ai), 1e-9),
    "recency": -p.recency(ai),
}
print(f"  {'feature':18s} {'corr with FebMar log-lift':>26s}")
for nm, v in feats.items():
    print(f"  {nm:18s} {np.corrcoef(v[act], lift)[0, 1]:>26.4f}")

hdr("6 -- WHAT A GLOBAL MULTIPLIER WOULD DO TO RMSLE")
print("  Applying a constant k to every prediction shifts log1p(pred) by ~log(k) for large")
print("  predictions but leaves the many near-zero predictions almost untouched, so the")
print("  effect on RMSLE is NOT symmetric and must be fitted on OOF, not assumed from 1.1628.")
print("  Recorded here as a decision input; the actual grid search belongs in a calibration")
print("  experiment on OOF predictions (BACKLOG C1/C3).")

(OUT).mkdir(parents=True, exist_ok=True)
(OUT / "holiday.json").write_text(json.dumps(R, indent=2, default=str))
print("\n  wrote reports/eda/holiday.json")
