#!/usr/bin/env python
"""
Blend several members' TEST-anchor submissions into one, in log space.

    python src/blend_submit.py --members e0049 e0064 e0101 --weights equal --out e0120

RMSLE's optimal point prediction is E[log1p(y)|x], so members are averaged in LOG space --
`expm1(sum w_i * log1p(p_i))`, a weighted geometric mean in linear space.  Averaging the raw
predictions would target E[y|x] instead.  This mirrors src/blend.py exactly, so the CV number
that script reports is a prediction about the file this script writes.

`--weights equal` is the default on purpose.  src/blend.py's leave-one-fold-out fit and its
equal-weight fallback landed within 0.00005 of each other on every combination measured, so
fitted weights buy nothing here and carry fitting risk that the equal-weight blend does not.
`--weights oof` fits on the pooled OOF if you want it anyway; the honest CV estimate of THAT
procedure is blend.py's leave-one-fold-out number, never the pooled in-sample fit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from metrics import gini, rmsle       # noqa: E402


def fit_oof_weights(members: list[str]) -> np.ndarray:
    from scipy.optimize import minimize
    d = {m: pl.read_parquet(ROOT / "oof" / f"{m}.parquet").sort(["fold_id", "user_id"])
         for m in members}
    base = d[members[0]]
    for m in members:
        assert np.array_equal(d[m]["user_id"].to_numpy(), base["user_id"].to_numpy()), \
            f"{m}: OOF user order differs"
    y = base["y_true"].to_numpy()
    L = np.column_stack([np.log1p(d[m]["y_pred"].to_numpy()) for m in members])
    f = lambda w: rmsle(y, np.expm1(L @ (np.abs(w) / np.abs(w).sum())))       # noqa: E731
    r = minimize(f, np.ones(len(members)) / len(members), method="Nelder-Mead",
                 options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-7})
    w = np.abs(r.x)
    return w / w.sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--weights", default="equal", choices=["equal", "oof"])
    ap.add_argument("--out", required=True, help="exp_id for the blended submission")
    args = ap.parse_args()

    ss = pl.read_csv(ROOT / "data" / "sample_submit.csv")
    uid = ss["user_id"].to_numpy()

    P = []
    for m in args.members:
        f = ROOT / "subs" / f"{m}.csv"
        if not f.exists():
            raise SystemExit(f"missing {f.relative_to(ROOT)} -- run that member in submit mode first")
        s = pl.read_csv(f)
        assert s.height == 250_000, f"{m}: {s.height} rows"
        assert np.array_equal(s["user_id"].to_numpy(), uid), f"{m}: user order differs from sample"
        v = np.maximum(s["predict"].to_numpy().astype(np.float64), 0.0)
        assert np.isfinite(v).all(), f"{m}: non-finite predictions"
        P.append(v)
    L = np.column_stack([np.log1p(v) for v in P])

    w = (np.ones(len(args.members)) / len(args.members) if args.weights == "equal"
         else fit_oof_weights(args.members))
    pred = np.maximum(np.expm1(L @ w), 0.0)

    C = np.corrcoef(L, rowvar=False)
    print(f"\n  members: {args.members}   weights ({args.weights}): "
          + ", ".join(f"{m}={x:.3f}" for m, x in zip(args.members, w)))
    print(f"\n  {'member':10s} {'weight':>7s} {'sum':>16s} {'mean':>9s} {'gini':>7s}   "
          f"pairwise corr of log-preds")
    for i, m in enumerate(args.members):
        print(f"  {m:10s} {w[i]:>7.3f} {P[i].sum():>16,.0f} {P[i].mean():>9.2f} "
              f"{gini(P[i]):>7.4f}   " + "  ".join(f"{C[i, j]:.4f}" for j in range(len(P))))
    print(f"  {args.out:10s} {'':>7s} {pred.sum():>16,.0f} {pred.mean():>9.2f} "
          f"{gini(pred):>7.4f}   <- blend")

    out = ROOT / "subs" / f"{args.out}.csv"
    out.parent.mkdir(exist_ok=True)
    pl.DataFrame({"user_id": uid, "predict": pred}).write_csv(out)
    print(f"\n  wrote {out.relative_to(ROOT)}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
