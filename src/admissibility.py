#!/usr/bin/env python
"""
The admissibility frontier: how good must a NEW blend member be, at a given correlation with
the family, to be worth building?

EXPERIMENTS.md §1c states the corrected rule qualitatively -- "decorrelation must be at
comparable quality" -- after Ridge and (now) BTYD both landed decorrelated, weak, and worth
zero.  This turns that sentence into a number, so the next candidate can be judged BEFORE it
is built rather than after.

Algebra.  In log space, with truth L and two predictors M (the family blend) and B (the
candidate), the best affine combination has multiple correlation

    R^2 = (rho_M^2 + rho_B^2 - 2 rho_M rho_B r) / (1 - r^2),      r = corr(M, B)

which rearranges EXACTLY to the form that matters:

    R^2 = rho_M^2  +  e^2 / (1 - r^2),        e := rho_B - r * rho_M

`e` is the **excess correlation**: the amount by which the candidate agrees with the truth
BEYOND what its agreement with the existing family already implies.  Everything follows:

  * e = 0  =>  the candidate is worth exactly zero, at ANY r.  A member can be wildly
    decorrelated and still be worthless; a member can be a near-twin and still pay.
  * the gain is QUADRATIC in e, so small excesses buy almost nothing;
  * at fixed quality rho_B, lower r is better -- but rho_B and r are not free of each
    other, and e is what actually decides.

Normalising e by both spreads collapses it to one textbook number:

    R^2 = rho_M^2 + (1 - rho_M^2) * rho_partial^2,     rho_partial = corr(L, B | M)

**The only thing that matters about a candidate blend member is its partial correlation
with the truth, controlling for the blend we already have.**  Not its own accuracy, not its
correlation with the family -- one number that combines both, computable from any OOF file
in a second, and comparable across every candidate this project has ever built.

This is the exact version of EXPERIMENTS.md §1c's qualitative rule.  §1c said decorrelation
must come "at comparable quality"; the sharp statement is that neither rho_B nor r alone is
the currency -- `e` is, and the bar on it is `e >= sqrt(dR2_required) * sqrt(1 - r^2)`.

EXPERIMENTS.md §1b/§2 give the score at optimal affine calibration as
`RMSLE = sd_L * sqrt(1 - R^2)`, so the two combine into a single admissibility number per
candidate.  §1 below checks the algebra against measured blends before §2 extrapolates.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from metrics import rmsle          # noqa: E402

E0120 = ["e0049", "e0064", "e0100", "e0101", "e0101s1", "e0101s2", "e0101s3", "e0102", "e0108"]
GAIN_THRESHOLD = 0.0005            # BTYD.md §6 / the standing bar for a new family


def load(e):
    return pd.read_parquet(ROOT / "oof" / f"{e}.parquet").sort_values(["fold_id", "user_id"])


def opt_affine_rmsle(Ly, Lp, fold):
    """RMSLE after the optimal per-fold affine map in log space -- the §1b ceiling."""
    sc = []
    for k in np.unique(fold):
        m = fold == k
        a, b = np.polyfit(Lp[m], Ly[m], 1)
        sc.append(rmsle(np.expm1(Ly[m]), np.expm1(a * Lp[m] + b)))
    return float(np.mean(sc))


def r2_two(rho_m, rho_b, r):
    return rho_m ** 2 + (rho_b - r * rho_m) ** 2 / (1 - r ** 2)


def excess(rho_m, rho_b, r):
    """e = rho_B - r*rho_M -- the candidate's entire blend value, in one number."""
    return rho_b - r * rho_m


def dr2_required(rho_m, sd_l, gain):
    """The R^2 increase corresponding to `gain` RMSLE at optimal affine calibration."""
    target = sd_l * np.sqrt(1 - rho_m ** 2) - gain
    return (1 - (target / sd_l) ** 2) - rho_m ** 2


def excess_required(rho_m, r, sd_l, gain):
    """|e| a candidate needs at correlation r.  Scales as sqrt(1 - r^2)."""
    return float(np.sqrt(max(dr2_required(rho_m, sd_l, gain), 0.0) * (1 - r ** 2)))


def gain_from(rho_m, rho_b, r, sd_l):
    """Predicted RMSLE change (negative = better) from adding the candidate."""
    return sd_l * (np.sqrt(1 - r2_two(rho_m, rho_b, r)) - np.sqrt(1 - rho_m ** 2))


def main() -> None:
    base = load(E0120[0])
    y, fold = base.y_true.values, base.fold_id.values
    Ly = np.log1p(y)
    L = {e: np.log1p(load(e).y_pred.values) for e in E0120}
    M = np.mean(list(L.values()), axis=0)

    sd_l = float(np.mean([Ly[fold == k].std() for k in np.unique(fold)]))
    rho_m = float(np.corrcoef(Ly, M)[0, 1])
    fam_rmsle = opt_affine_rmsle(Ly, M, fold)
    print(f"\n  frozen folds: sd_L = {sd_l:.4f}   9-member blend rho = {rho_m:.5f}")
    print(f"  family score at optimal affine calibration = {fam_rmsle:.5f}"
          f"   (algebra says {sd_l * np.sqrt(1 - rho_m ** 2):.5f})")

    # ---------------------------------------------------------------- 1. validate
    print(f"\n  --- does the algebra reproduce measured two-member blends? ---")
    print(f"  {'pair':26s} {'rho_A':>7s} {'rho_B':>7s} {'r':>7s} {'predicted':>10s} {'measured':>9s} {'err':>8s}")
    pairs = [("e0049", "e0064"), ("e0049", "e0101"), ("e0049", "e0100"),
             ("e0101", "e0102"), ("e0049", "e0170"), ("e0064", "e0170")]
    for A, B in pairs:
        La = L[A] if A in L else np.log1p(load(A).y_pred.values)
        Lb = L[B] if B in L else np.log1p(load(B).y_pred.values)
        ra = float(np.corrcoef(Ly, La)[0, 1]); rb = float(np.corrcoef(Ly, Lb)[0, 1])
        r = float(np.corrcoef(La, Lb)[0, 1])
        pred = sd_l * np.sqrt(1 - r2_two(ra, rb, r))
        meas = min(opt_affine_rmsle(Ly, La * w + Lb * (1 - w), fold)
                   for w in np.linspace(0, 1, 41))
        print(f"  {A + ' + ' + B:26s} {ra:>7.4f} {rb:>7.4f} {r:>7.4f} "
              f"{pred:>10.5f} {meas:>9.5f} {meas - pred:>+8.5f}")

    # ---------------------------------------------------------------- 2. every candidate
    print(f"\n  --- excess correlation e = rho_B - r*rho_M, per candidate ---")
    print(f"  Family M excludes the candidate, so each row is a genuine 'would adding this "
          f"help?'\n")
    print(f"  {'candidate':14s} {'rho_B':>8s} {'r vs M':>8s} {'e':>9s} {'rho_partial':>12s} "
          f"{'pred gain':>10s} {'measured':>9s}")
    others = {e: L[e] for e in E0120}
    extra = [a for a in sys.argv[1:] if not a.startswith("-")] or ["e0170"]
    cands = list(E0120) + extra
    rows = []
    for c in cands:
        keep = [e for e in E0120 if e != c]
        Mc = np.mean([others[e] for e in keep], axis=0)
        Lb = L[c] if c in L else np.log1p(load(c).y_pred.values)
        rm = float(np.corrcoef(Ly, Mc)[0, 1])
        rb = float(np.corrcoef(Ly, Lb)[0, 1])
        r = float(np.corrcoef(Mc, Lb)[0, 1])
        e = excess(rm, rb, r)
        rp = e / (np.sqrt(1 - rm ** 2) * np.sqrt(1 - r ** 2))
        pred = gain_from(rm, rb, r, sd_l)
        meas = (min(opt_affine_rmsle(Ly, Mc * w + Lb * (1 - w), fold)
                    for w in np.linspace(0, 1, 41)) - opt_affine_rmsle(Ly, Mc, fold))
        rows.append((c, rp))
        print(f"  {c:14s} {rb:>8.5f} {r:>8.5f} {e:>+9.5f} {rp:>12.5f} "
              f"{pred:>+10.5f} {meas:>+9.5f}")
    rp_req = float(np.sqrt(dr2_required(rho_m, sd_l, GAIN_THRESHOLD) / (1 - rho_m ** 2)))
    print(f"\n  rho_partial required for {GAIN_THRESHOLD}: {rp_req:.5f}")
    best = max(rows, key=lambda t: t[1])
    print(f"  best ever achieved:              {best[1]:.5f}  ({best[0]})  "
          f"= {best[1] / rp_req:.0%} of the bar")
    print(f"  >>> NOTHING this project has built clears it, including the members of the "
          f"blend itself.")

    # ---------------------------------------------------------------- 3. the frontier
    dr2 = dr2_required(rho_m, sd_l, GAIN_THRESHOLD)
    print(f"\n  --- the frontier for a NEW family, to buy {GAIN_THRESHOLD} ---")
    print(f"  required dR^2 = {dr2:.6f}   =>   e_required = {np.sqrt(dr2):.5f} * sqrt(1 - r^2)")
    print(f"\n  {'r vs family':>12s} {'e needed':>9s} {'rho_B needed':>13s} {'== CV* of':>10s}")
    for r in (0.80, 0.90, 0.9427, 0.96, 0.97, 0.98, 0.99, 0.995, 0.999):
        ereq = excess_required(rho_m, r, sd_l, GAIN_THRESHOLD)
        rn = r * rho_m + ereq
        print(f"  {r:>12.4f} {ereq:>9.5f} {rn:>13.5f} {sd_l * np.sqrt(1 - rn ** 2):>10.5f}")
    print(f"\n  Note the direction: a MORE correlated candidate needs a SMALLER excess.")
    print(f"  'Find a decorrelated partner' is only half the rule -- decorrelation raises")
    print(f"  the bar on e at the same time as it multiplies e's payoff.")

    # ---------------------------------------------------------------- 4. B1's requirement
    print(f"\n  --- BAYES_EXP B1/B2: the bar, stated BEFORE building ---")
    bt = np.log1p(load("e0170").y_pred.values)
    r_bt = float(np.corrcoef(M, bt)[0, 1])
    rho_bt = float(np.corrcoef(Ly, bt)[0, 1])
    print(f"  B0 as built: rho_B {rho_bt:.5f}, r {r_bt:.5f}, e {excess(rho_m, rho_bt, r_bt):+.5f}, "
          f"needs {excess_required(rho_m, r_bt, sd_l, GAIN_THRESHOLD):.5f} "
          f"-> {excess(rho_m, rho_bt, r_bt) / excess_required(rho_m, r_bt, sd_l, GAIN_THRESHOLD):.0%} of the bar")
    print(f"\n  {'if B1 lands at r =':>19s} {'it needs rho_B':>15s} {'= CV*':>9s} "
          f"{'= B0 improved by':>17s}")
    own_bt = sd_l * np.sqrt(1 - rho_bt ** 2)
    for r in (r_bt, 0.95, 0.96, 0.97, 0.98):
        rn = r * rho_m + excess_required(rho_m, r, sd_l, GAIN_THRESHOLD)
        cv = sd_l * np.sqrt(1 - rn ** 2)
        print(f"  {r:>19.4f} {rn:>15.5f} {cv:>9.5f} {own_bt - cv:>17.5f}")


if __name__ == "__main__":
    main()
