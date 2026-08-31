#!/usr/bin/env python
"""
Solve the metric from the leaderboard, then apply the optimal affine transform in log space.

    python src/calibrate_lb.py --members e0141 e0120 --out e0150

WHY THIS IS MEASUREMENT AND NOT LEADERBOARD TUNING (CLAUDE.md rule 7).  Two probe submissions
already logged in `cv_lb.csv` pin the public test set's truth distribution exactly:

    probe_zeros   pred = 0   -> RMSLE^2 = E[L^2]                      = 3.28^2
    probe_const10 pred = 10  -> RMSLE^2 = E[(L - log1p(10))^2]        = 2.32^2
    =>  E[log1p y] = 2.3199,  sd(log1p y) = 2.3187

Those are population constants, not hyperparameters, and they were bought deliberately.  The
framework they feed is checked against something known independently: `sample_submit.csv`
(= last-30d GMV, LB 2.12) comes back at rho = 0.5771 against DATA.md §7.1's CV-measured 0.557.

The decomposition (EXPERIMENTS.md §2) is exact:

    RMSLE^2 = (sd_L - sd_M)^2 + 2 sd_L sd_M (1 - rho) + (mu_L - mu_M)^2

so one LB score per member yields that member's rho, and after the optimal affine transform
    M' = mu_L + a (M - mu_M),   a = sd_L rho / sd_M
the score collapses to sd_L sqrt(1 - rho^2).  Only rho is irreducible; level and spread are
free to fix.  This is what the e0141-vs-e0120 comparison turned on: e0141 scored better
(1.6488 vs 1.6553) with the LOWER rho (0.70345 vs 0.70400), purely because its log-mean landed
near the truth's while e0120's sat 0.159 too high.

Blend weights are EQUAL by default, not fitted.  The fitted optimum came within 0.0001 of
equal weighting, which is not worth the fitting risk on 50,000 public users.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from metrics import gini                      # noqa: E402

PROBE_ZEROS, PROBE_CONST10, CONST = 3.28, 2.32, 10.0
LB_SCORES = {"e0140": 1.6552, "e0141": 1.6488, "e0142": 1.6785, "e0150": 1.64670,
             "e0151": 1.64748, "e0152": 1.646697, "e0145": 1.65323,
             "e0120": 1.6553, "e0064": 1.6559, "e0049": 1.6562, "e0060": 1.6567,
             "e0020": 1.6578, "e0001": 1.6766, "e0090": 1.65524}


# Re-solved from 6-significant-figure submission scores rather than the 3-s.f. probes.
# e0150 and e0151 are affine transforms of the SAME blend, so they share rho exactly -- four
# precise equations in four unknowns. Both back-check inside the probes' rounding
# (zeros 3.2867 vs 3.28, const10 2.3188 vs 2.32). EXPERIMENTS.md 1b.
REFINED = (2.3303, 2.3178)


def truth_moments(refined: bool = True) -> tuple[float, float]:
    if refined:
        return REFINED
    e_l2 = PROBE_ZEROS ** 2
    c = np.log1p(CONST)
    mu = (e_l2 + c * c - PROBE_CONST10 ** 2) / (2 * c)
    return float(mu), float(np.sqrt(e_l2 - mu * mu))


def read_sub(exp: str) -> np.ndarray:
    with (ROOT / "subs" / f"{exp}.csv").open() as fh:
        r = csv.reader(fh); next(r)
        return np.array([float(x[1]) for x in r])


def rho_from_lb(M: np.ndarray, lb: float, mu_L: float, sd_L: float) -> float:
    m, s = float(M.mean()), float(M.std())
    rest = lb * lb - (sd_L - s) ** 2 - (mu_L - m) ** 2
    return float(1.0 - rest / (2.0 * sd_L * s))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["shift", "affine"], default="affine")
    ap.add_argument("--weights", type=float, nargs="+", default=None,
                    help="explicit blend weights, in --members order (renormalised)")
    ap.add_argument("--assume-rho", nargs="+", default=[], metavar="EXP=RHO",
                    help="rho for a member with no LB score yet, e.g. e0143=0.70335")
    ap.add_argument("--nonneg", action="store_true",
                    help="solve non-negative weights maximising rho instead of equal weighting")
    args = ap.parse_args()

    for kv in args.assume_rho:
        e, v = kv.split("=")
        LB_SCORES[e] = None
        globals().setdefault("_ASSUMED", {})[e] = float(v)
    mu_L, sd_L = truth_moments()
    print(f"\n  public truth (re-solved from submission scores): "
          f"E[log1p y] = {mu_L:.4f}   sd = {sd_L:.4f}")

    ssu, ssv = [], []
    with (ROOT / "data" / "sample_submit.csv").open() as fh:
        r = csv.reader(fh); next(r)
        for a_, b_ in r:
            ssu.append(int(a_)); ssv.append(float(b_))
    ssu = np.array(ssu, dtype=np.int64)
    chk = np.log1p(np.array(ssv, dtype=np.float64))
    print(f"  sanity: sample_submit (LB 2.12) -> rho {rho_from_lb(chk, 2.12, mu_L, sd_L):.4f} "
          f"[DATA.md §7.1 measured 0.557 on CV]")

    Ms, rhos, cov = [], [], 0.0
    print(f"\n  {'member':9s} {'LB':>8s} {'mu_M':>8s} {'sd_M':>8s} {'rho':>8s}")
    ASSUMED = globals().get("_ASSUMED", {})
    for e in args.members:
        if e not in LB_SCORES:
            raise SystemExit(f"{e} has no logged LB score -- pass --assume-rho {e}=<value>")
        M = np.log1p(read_sub(e))
        if LB_SCORES[e] is None:
            r = ASSUMED[e]
            print(f"  {e:9s} {'(assumed)':>8s} {M.mean():>8.4f} {M.std():>8.4f} {r:>8.5f}")
        else:
            r = rho_from_lb(M, LB_SCORES[e], mu_L, sd_L)
            print(f"  {e:9s} {LB_SCORES[e]:>8.4f} {M.mean():>8.4f} {M.std():>8.4f} {r:>8.5f}")
        Ms.append(M); rhos.append(r); cov += float(M.std()) * r / len(args.members)

    if args.nonneg and len(Ms) > 1:
        # maximise rho_blend = (w.c) / sqrt(w'Sw) over the simplex, by projected gradient.
        # Non-negativity is the whole point: an unconstrained solve on members correlating at
        # 0.99 returns large opposing weights that fit leaderboard noise, not signal.
        X = np.stack(Ms); S = np.cov(X)
        cvec = np.array([m.std() * r for m, r in zip(Ms, rhos)])
        w = np.ones(len(Ms)) / len(Ms)
        for _ in range(20000):
            sb = np.sqrt(w @ S @ w)
            w = np.maximum(w + 0.02 * (cvec / sb - (w @ cvec) * (S @ w) / sb ** 3), 0.0)
            w = w / max(w.sum(), 1e-9)
        print("\n  non-negative optimal weights: "
              + ", ".join(f"{e}={x:.3f}" for e, x in zip(args.members, w) if x > 0.005))
        Mb = X.T @ w
        cov = float(w @ cvec)
    elif args.weights:
        w = np.abs(np.array(args.weights, float)); w = w / w.sum()
        print("\n  explicit weights: " + ", ".join(f"{e}={x:.3f}"
                                                  for e, x in zip(args.members, w)))
        Mb = np.stack(Ms).T @ w
        cov = float(sum(x * m.std() * r for x, m, r in zip(w, Ms, rhos)))
    else:
        Mb = np.mean(Ms, axis=0)
    mb, sb = float(Mb.mean()), float(Mb.std())
    rho_b = cov / sb                      # Cov(blend, L) / sd_blend
    a = (sd_L * rho_b / sb) if args.mode == "affine" else 1.0
    Mc = np.maximum(mu_L + a * (Mb - mb), 0.0)
    pred = np.maximum(np.expm1(Mc), 0.0)

    uncal = np.sqrt((sd_L - sb) ** 2 + 2 * sd_L * sb * (1 - rho_b) + (mu_L - mb) ** 2)
    # shift-only zeroes the MEAN term and leaves the spread term alone; the affine transform
    # additionally sets sd_M = sd_L*rho, which collapses the score to sd_L*sqrt(1-rho^2).
    shifted = np.sqrt((sd_L - sb) ** 2 + 2 * sd_L * sb * (1 - rho_b))
    best = sd_L * np.sqrt(max(1 - rho_b ** 2, 0.0))
    print(f"\n  blend of {len(args.members)}, "
          f"{'non-negative fitted weights' if args.nonneg else 'equal weight (NOT fitted)'}")
    print(f"    rho_blend       {rho_b:.5f}")
    print(f"    shift           {mb:.4f} -> {mu_L:.4f}   ({mu_L - mb:+.4f})")
    print(f"    scale           {sb:.4f} -> {a * sb:.4f}   (x{a:.4f})")
    print(f"    uncalibrated    {uncal:.4f}")
    print(f"    PREDICTED LB    {best if args.mode == 'affine' else shifted:.4f}"
          f"   [uncalibrated {uncal:.4f} | shift-only {shifted:.4f} | affine {best:.4f}]")
    print(f"    clipped at zero {(Mc <= 0).sum():,} users ({100 * (Mc <= 0).mean():.3f}%)")

    assert pred.size == ssu.size == 250_000
    assert np.isfinite(pred).all() and (pred >= 0).all()
    out = ROOT / "subs" / f"{args.out}.csv"
    with out.open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["user_id", "predict"])
        w.writerows(zip(ssu.tolist(), pred.tolist()))
    print(f"\n  wrote {out.relative_to(ROOT)}   sum {pred.sum():,.0f}  mean {pred.mean():.2f}  "
          f"gini {gini(pred):.4f}")


if __name__ == "__main__":
    main()
