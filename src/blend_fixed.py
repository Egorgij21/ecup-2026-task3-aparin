#!/usr/bin/env python
"""
Score a FIXED-weight log-space blend on the frozen folds.

    python src/blend_fixed.py --group gbdt e0049 e0064 --group seq e0100 e0101 --alpha 0.5

src/blend.py fits weights and reports a leave-one-fold-out number, which is the honest way to
score a *fitting procedure*.  This script scores a *stated* weighting instead: no parameters
are estimated from the folds, so the per-fold numbers are directly comparable to any logged
experiment with no LOFO machinery and no fitting risk at all.

It exists for one decision.  The equal-weight blend over all members puts ~78% of its mass on
the `seq` family, which has never been scored on the leaderboard -- CV says it is the better
family, but CV has been wrong about transfer before (e0060 won CV by 0.4 sigma and lost 0.0005
on the LB).  `--alpha` sweeps the split between two named groups, members equal-weighted
inside each group, so the cost of hedging toward the LB-validated family is a measured number
rather than a feeling.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from metrics import gini, rmsle, total_gmv_rel_err        # noqa: E402


def load(members: list[str]):
    d = {m: pl.read_parquet(ROOT / "oof" / f"{m}.parquet").sort(["fold_id", "user_id"])
         for m in members}
    base = d[members[0]]
    for m in members:
        assert np.array_equal(d[m]["user_id"].to_numpy(), base["user_id"].to_numpy()), \
            f"{m}: OOF user order differs"
        assert np.array_equal(d[m]["fold_id"].to_numpy(), base["fold_id"].to_numpy()), \
            f"{m}: fold assignment differs"
    return (base["y_true"].to_numpy(), base["fold_id"].to_numpy(),
            {m: np.log1p(d[m]["y_pred"].to_numpy()) for m in members})


def score(y, fold, L) -> tuple[float, list[float]]:
    per = [rmsle(y[fold == k], np.expm1(L[fold == k])) for k in np.unique(fold)]
    return float(np.mean(per)), per


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", nargs="+", action="append", required=True,
                    metavar="NAME MEMBER...", help="repeatable: a group name then its members")
    ap.add_argument("--alpha", type=float, nargs="+",
                    default=[0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
                    help="weight on the SECOND group; swept")
    args = ap.parse_args()
    assert len(args.group) == 2, "exactly two groups (the sweep is one-dimensional)"

    (n1, *m1), (n2, *m2) = args.group
    y, fold, Ls = load(m1 + m2)
    G1 = np.mean([Ls[m] for m in m1], axis=0)
    G2 = np.mean([Ls[m] for m in m2], axis=0)

    print(f"\n  group {n1}: {m1}")
    print(f"  group {n2}: {m2}")
    print(f"  corr(log {n1}, log {n2}) = {np.corrcoef(G1, G2)[0, 1]:.4f}")
    print(f"\n  {'w(' + n2 + ')':>9s} {'cv_mean':>9s}  {'folds':<46s} {'gini':>7s} {'tot_err':>8s}")
    best = None
    for a in args.alpha:
        L = (1 - a) * G1 + a * G2
        m, per = score(y, fold, L)
        p = np.expm1(L)
        print(f"  {a:>9.2f} {m:>9.5f}  {str([round(x, 5) for x in per]):<46s} "
              f"{gini(p):>7.4f} {total_gmv_rel_err(y, p):>+8.4f}")
        if best is None or m < best[1]:
            best = (a, m)
    print(f"\n  best on CV: w({n2}) = {best[0]:.2f} at {best[1]:.5f}")
    print(f"  NOTE: these are stated weights, not fitted -- every row above is directly")
    print(f"        comparable to a logged cv_mean, with no leave-one-fold-out correction.")


if __name__ == "__main__":
    main()
