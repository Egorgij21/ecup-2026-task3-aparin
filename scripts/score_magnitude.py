"""Score a magnitude model the way IDEAS.md §I13 prescribes: corr(L, ·|Z=1).

The whole point of I13 is that the magnitude term was invisible because everything was
scored END-TO-END. This scores it on the buyer subset only, against the measured ceiling.

    corr(L, M)  among fold users with y > 0 in the SCORED window
    ceiling 0.6001   e0049 achieves 0.4814 (80.2%)   buy flag achieves 89.6%

Conditioning note (I13 got this wrong once and it moved the number 0.015): condition on the
window being scored, not on any earlier window -- otherwise zeros stay in the target and the
buy/no-buy decision leaks back into a supposedly magnitude-only number.

    python3 scripts/score_magnitude.py oof/e0240.parquet oof/e0241.parquet
"""
import sys
import numpy as np
import polars as pl

CEILING = 0.6001
E0049 = 0.4814


def magnitude_corr(path):
    o = pl.read_parquet(path)
    per, Ls, Ms = [], [], []
    for f in sorted(o["fold_id"].unique().to_list()):
        d = o.filter(pl.col("fold_id") == f)
        y = d["y_true"].to_numpy()
        m = y > 0
        L = np.log1p(y[m])
        M = np.log1p(np.maximum(d["y_pred"].to_numpy()[m], 0.0))
        per.append((f, int(m.sum()), float(np.corrcoef(L, M)[0, 1])))
        Ls.append(L); Ms.append(M)
    L, M = np.concatenate(Ls), np.concatenate(Ms)
    return per, float(np.corrcoef(L, M)[0, 1]), len(L)


def main(paths):
    print(f"{'model':32}{'pooled':>9}{'vs e0049':>10}{'% of ceiling':>14}  per-fold")
    base = None
    for p in paths:
        per, pooled, n = magnitude_corr(p)
        if base is None:
            base = pooled
        folds = " ".join(f"{c:.4f}" for _, _, c in per)
        print(f"{p.split('/')[-1]:32}{pooled:9.4f}{pooled - E0049:+10.4f}"
              f"{pooled / CEILING * 100:13.1f}%  [{folds}]")
    print(f"\nceiling {CEILING}  |  e0049 {E0049} ({E0049 / CEILING * 100:.1f}% captured)")
    print("kill condition (pre-registered): treatment must beat the same-harness control by >0.002")


if __name__ == "__main__":
    main(sys.argv[1:] or ["oof/e0049.parquet"])
