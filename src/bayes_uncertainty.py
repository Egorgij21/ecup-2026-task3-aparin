#!/usr/bin/env python
"""
BAYES_EXP.md §10 and §5.3: are the posterior UNCERTAINTY columns worth anything?

§10's claim, and it is the one thing in that document no experiment here has tested:

  > The posterior standard deviations are the interesting ones. A GBDT can reconstruct
  > `e_orders` from RFM features given enough splits; it cannot manufacture "how much does
  > the history actually pin this user down", and that is exactly the quantity RMSLE cares
  > about when deciding how hard to shrink toward zero.

The claim is correct that a GBDT cannot build the column.  Whether the column carries signal
the blend has not already extracted is a separate question, and that is what this measures.

e0171 already tested BTYD's POINT estimates (P(alive), E[X30], E[M]) and found -0.00008
against the no-op control.  This tests the two columns e0171 did not have:

    sd_log1p   posterior predictive sd of log1p(Y)  -- BAYES_EXP §5.3's `s_u`, §10's headline
    p_zero     posterior P(Y = 0)                   -- not a monotone function of P(alive)

Both come free from the same simulation that produces the point estimate.

Every arm carries the no-op control that EXPERIMENTS.md §1b made a standing rule: refitting
on the blend alone already costs something, and without subtracting it a null reads as a
small loss and the reason is invisible.
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


def load(e):
    return pd.read_parquet(ROOT / "oof" / f"{e}.parquet").sort_values(["fold_id", "user_id"])


def lofo(X, y, fold):
    """OLS in log space, intercept added, fitted leave-one-fold-out."""
    Ly = np.log1p(y)
    X = np.column_stack([np.ones(len(y))] + [np.asarray(c, np.float64) for c in X])
    sc = []
    for k in np.unique(fold):
        tr, te = fold != k, fold == k
        beta, *_ = np.linalg.lstsq(X[tr], Ly[tr], rcond=None)
        sc.append(rmsle(y[te], np.expm1(X[te] @ beta)))
    return float(np.mean(sc))


def main() -> None:
    bt = load("e0170")
    y, fold = bt.y_true.values, bt.fold_id.values
    M = np.mean([np.log1p(load(e).y_pred.values) for e in E0120], axis=0)
    Ly = np.log1p(y)

    sd_u = bt.sd_log1p.values
    p0 = bt.p_zero.values
    pa = bt.p_alive.values
    print(f"\n  posterior uncertainty columns from the B0 fit ({len(y):,} rows)")
    for nm, v in (("sd_log1p", sd_u), ("p_zero", p0), ("p_alive", pa)):
        print(f"    {nm:10s} min {v.min():.4f}  p50 {np.median(v):.4f}  max {v.max():.4f}  "
              f"corr with |resid| {np.corrcoef(v, np.abs(Ly - M))[0, 1]:+.4f}")
    print(f"    corr(sd_log1p, p_zero) {np.corrcoef(sd_u, p0)[0, 1]:+.4f}   "
          f"corr(sd_log1p, p_alive) {np.corrcoef(sd_u, pa)[0, 1]:+.4f}")

    # ------------------------------------------------------------------ §10: as features
    print(f"\n  --- §10: uncertainty columns stacked on the blend ---")
    ctrl = lofo([M], y, fold)
    print(f"  CONTROL: blend alone                        {ctrl:.5f}")
    arms = {
        "+ sd_log1p": [M, sd_u],
        "+ p_zero": [M, p0],
        "+ sd_log1p, p_zero": [M, sd_u, p0],
        "+ sd_log1p x blend (interaction)": [M, sd_u, M * sd_u],
        "+ all uncertainty (sd, p0, p_alive)": [M, sd_u, p0, pa],
        "+ e0171 point estimates too": [M, sd_u, p0, pa, np.log1p(bt.e_x30.values),
                                        np.log1p(bt.e_m.values), np.log1p(bt.y_pred.values)],
    }
    for nm, X in arms.items():
        s = lofo(X, y, fold)
        print(f"  {nm:-<43s} {s:.5f}   marginal {s - ctrl:+.5f}")

    # ------------------------------------------------------------------ §5.3: calibration
    print(f"\n  --- §5.3: the gamma*s_u calibration term, on the blend ---")
    print(f"  log1p(y_hat) = a*log1p(y_pred) + b + gamma*s_u, fitted leave-one-fold-out")
    a_only = lofo([M], y, fold)
    a_su = lofo([M, sd_u], y, fold)
    print(f"  affine only (a, b)                          {a_only:.5f}")
    print(f"  affine + gamma*s_u                          {a_su:.5f}   "
          f"marginal {a_su - a_only:+.5f}")
    # per-cohort affine, as §5.3 specifies (cohort = buyer / browser)
    x0 = np.isclose(pa, 1.0)                      # BG/NBD sets P(alive)=1 exactly iff x=0
    print(f"  cohorts: x=0 {100 * x0.mean():.1f}%   x>=1 {100 * (~x0).mean():.1f}%")
    coh = lofo([M, x0.astype(float), M * x0.astype(float)], y, fold)
    print(f"  per-cohort affine (a_c, b_c)                {coh:.5f}   "
          f"marginal {coh - a_only:+.5f}")
    full = lofo([M, x0.astype(float), M * x0.astype(float), sd_u, sd_u * x0.astype(float)], y, fold)
    print(f"  full §5.3 (per-cohort affine + gamma*s_u)   {full:.5f}   "
          f"marginal {full - a_only:+.5f}")

    # ------------------------------------------------------------------ the blocking rule
    print(f"\n  --- §5.3's blocking rule: is a_c in [0.9, 1.1]? ---")
    for k in np.unique(fold):
        tr = fold != k
        X = np.column_stack([np.ones(tr.sum()), M[tr]])
        beta, *_ = np.linalg.lstsq(X, Ly[tr], rcond=None)
        print(f"    fold {k}: a = {beta[1]:.4f}  b = {beta[0]:+.4f}", end="")
        print("   <- OUTSIDE [0.9, 1.1]" if not 0.9 <= beta[1] <= 1.1 else "")


if __name__ == "__main__":
    main()
