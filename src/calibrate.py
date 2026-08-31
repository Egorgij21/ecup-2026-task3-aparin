#!/usr/bin/env python
"""
Fit post-hoc calibration on OOF predictions and measure the RMSLE <-> tie-breaker trade-off.

    python src/calibrate.py --exp e0020

Why this matters here: RMSLE is nearly indifferent to the overall level (DATA.md §9.4 --
the level term is 0.006 of a ~2.1 total, the correlation term is 4.49), but the jury's
RMSPE-on-total-GMV tie-breaker is *entirely* about level, and we currently under-shoot the
aggregate badly. So there is a real trade-off and it should be measured, not guessed.

Everything is fitted LEAVE-ONE-FOLD-OUT: the calibration for fold j is fitted on the other
four folds only. With 1 M OOF rows and 1-2 parameters the overfitting risk is negligible,
but the protocol is cheap to honour and makes the reported gain honest.

Candidate forms
    A  linear multiplier      p' = k * p
    B  log-space power        p' = expm1(a * log1p(p))
    C  log-space affine       p' = expm1(a * log1p(p) + b)
    D  per-decile multiplier  k_d within deciles of the prediction
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from metrics import rmsle, gini, total_gmv_rel_err, rmspe   # noqa: E402


def fit_k(y, p):
    ks = np.linspace(0.5, 4.0, 351)
    return float(ks[np.argmin([rmsle(y, k * p) for k in ks])])


def fit_power(y, p):
    """p' = expm1(a*log1p(p)). Minimising RMSLE over `a` IS least squares of L on M
    through the origin, so it is closed form -- no grid needed, and exactly optimal."""
    L, M = np.log1p(y), np.log1p(p)
    return float((L * M).sum() / max((M * M).sum(), 1e-12))


def fit_affine(y, p):
    """p' = expm1(a*log1p(p) + b). RMSLE^2 = mean((L - (aM+b))^2) -- plain OLS of L on M."""
    L, M = np.log1p(y), np.log1p(p)
    a, b = np.polyfit(M, L, 1)
    return float(a), float(b)


def fit_decile(y, p, edges):
    d = np.digitize(p, edges)
    ks = np.ones(len(edges) + 1)
    grid = np.linspace(0.5, 4.0, 141)
    for j in range(len(ks)):
        m = d == j
        if m.sum() > 500:
            ks[j] = grid[np.argmin([rmsle(y[m], g * p[m]) for g in grid])]
    return ks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="e0020")
    args = ap.parse_args()
    oof = pd.read_parquet(ROOT / "oof" / f"{args.exp}.parquet")
    folds = sorted(oof["fold_id"].unique().tolist())
    print(f"\n  {args.exp}: {len(oof):,} OOF rows over {len(folds)} folds")

    y_all = oof["y_true"].to_numpy(); p_all = oof["y_pred"].to_numpy()
    f_all = oof["fold_id"].to_numpy()
    base = np.mean([rmsle(y_all[f_all == k], p_all[f_all == k]) for k in folds])
    print(f"  uncalibrated CV = {base:.5f}   total_rel_err = {total_gmv_rel_err(y_all, p_all):+.4f}"
          f"   gini = {gini(p_all):.4f} (true {gini(y_all):.4f})")

    edges = np.quantile(p_all[p_all > 0], np.linspace(0, 1, 11)[1:-1])
    res = {n: [] for n in ["A linear k", "B log power", "C log affine", "D per-decile k"]}
    params = {n: [] for n in res}
    for k in folds:
        tr, te = f_all != k, f_all == k
        yt, pt, yv, pv = y_all[tr], p_all[tr], y_all[te], p_all[te]
        kk = fit_k(yt, pt);            res["A linear k"].append(rmsle(yv, kk * pv)); params["A linear k"].append(kk)
        aa = fit_power(yt, pt);        res["B log power"].append(rmsle(yv, np.expm1(aa * np.log1p(pv)))); params["B log power"].append(aa)
        a2, b2 = fit_affine(yt, pt);   res["C log affine"].append(rmsle(yv, np.expm1(a2 * np.log1p(pv) + b2))); params["C log affine"].append((a2, b2))
        kd = fit_decile(yt, pt, edges)
        res["D per-decile k"].append(rmsle(yv, kd[np.digitize(pv, edges)] * pv)); params["D per-decile k"].append(kd.round(2).tolist())

    print(f"\n  LEAVE-ONE-FOLD-OUT calibrated CV")
    print(f"  {'form':16s} {'CV':>9s} {'delta':>9s} {'fitted params (per fold)':>34s}")
    print(f"  {'uncalibrated':16s} {base:>9.5f} {0.0:>+9.5f}")
    for n in res:
        v = np.mean(res[n])
        pr = params[n][0] if n != "D per-decile k" else "10 multipliers"
        print(f"  {n:16s} {v:>9.5f} {v - base:>+9.5f} {str(pr):>34s}")

    # ---- the trade-off: what does forcing the aggregate to be right cost in RMSLE? ----
    print(f"\n  RMSLE <-> AGGREGATE TRADE-OFF (fitted on all OOF, reported on all OOF)")
    k_rmsle = fit_k(y_all, p_all)
    k_agg = float(y_all.sum() / p_all.sum())
    print(f"  {'k':>6s} {'what':26s} {'RMSLE':>9s} {'total_rel_err':>14s} {'RMSPE':>9s} {'gini':>8s}")
    for k, lab in [(1.0, "as predicted"), (k_rmsle, "RMSLE-optimal"), (k_agg, "aggregate-exact"),
                   (k_agg * 1.1628, "aggregate x seasonal 1.16")]:
        q = k * p_all
        print(f"  {k:>6.3f} {lab:26s} {rmsle(y_all, q):>9.5f} "
              f"{total_gmv_rel_err(y_all, q):>+14.4f} {rmspe(y_all, q):>9.2f} {gini(q):>8.4f}")
    print(f"\n  cost of an exactly-calibrated aggregate: "
          f"{rmsle(y_all, k_agg * p_all) - rmsle(y_all, k_rmsle * p_all):+.5f} RMSLE")
    print(f"  NOTE gini is scale-invariant, so a multiplier cannot fix the Gini gap "
          f"({gini(p_all):.4f} vs true {gini(y_all):.4f}) -- only re-ranking can.")


if __name__ == "__main__":
    main()
