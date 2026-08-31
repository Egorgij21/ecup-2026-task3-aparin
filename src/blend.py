#!/usr/bin/env python
"""
Blend OOF predictions from several models, fitted leave-one-fold-out.

    python src/blend.py e0023 e0027 e0028

RMSLE's optimal point prediction is E[log1p(y)|x], so members are averaged in LOG space --
`expm1(sum w_i * log1p(p_i))`, a weighted geometric mean in linear space. Averaging the raw
predictions would target E[y|x] instead, the same functional error flagged for simulation
averaging in PAPERS_FEATURES_AND_IDEAS.md §6.

Weights are non-negative and sum to 1, fitted on the four training folds and applied to the
held-out fold, so the reported CV is honest rather than the in-sample optimum.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from metrics import rmsle          # noqa: E402

exps = sys.argv[1:]
if not exps:
    print(__doc__); sys.exit(1)
d = {e: pd.read_parquet(ROOT / "oof" / f"{e}.parquet").sort_values(["fold_id", "user_id"])
     for e in exps}
base = d[exps[0]]
y, fold = base.y_true.values, base.fold_id.values
L = np.column_stack([np.log1p(d[e].y_pred.values) for e in exps])
for e in exps:
    assert np.array_equal(d[e].user_id.values, base.user_id.values), f"{e}: user order differs"

print(f"\n  members: {exps}   rows {len(y):,}\n")
print(f"  {'member':10s} {'own CV':>9s}   pairwise corr of log-preds")
C = np.corrcoef(L, rowvar=False)
for i, e in enumerate(exps):
    own = np.mean([rmsle(y[fold == k], np.expm1(L[fold == k, i])) for k in np.unique(fold)])
    print(f"  {e:10s} {own:>9.5f}   " + "  ".join(f"{C[i, j]:.4f}" for j in range(len(exps))))

def fit_w(Ltr, ytr):
    n = Ltr.shape[1]
    f = lambda w: rmsle(ytr, np.expm1(Ltr @ (np.abs(w) / np.abs(w).sum())))
    r = minimize(f, np.ones(n) / n, method="Nelder-Mead",
                 options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-7})
    w = np.abs(r.x); return w / w.sum()

sc, ws = [], []
for k in np.unique(fold):
    tr, te = fold != k, fold == k
    w = fit_w(L[tr], y[tr]); ws.append(w)
    sc.append(rmsle(y[te], np.expm1(L[te] @ w)))
print(f"\n  LEAVE-ONE-FOLD-OUT blend CV = {np.mean(sc):.5f}   folds {np.round(sc, 5).tolist()}")
print(f"  mean fitted weights: " + ", ".join(f"{e}={w:.3f}" for e, w in zip(exps, np.mean(ws, 0))))
best = min((np.mean([rmsle(y[fold == k], np.expm1(L[fold == k, i])) for k in np.unique(fold)]), e)
           for i, e in enumerate(exps))
print(f"  best single member  = {best[0]:.5f} ({best[1]})")
print(f"  blend gain          = {np.mean(sc) - best[0]:+.5f}   (sigma_noise 0.00009)")
eq = np.mean([rmsle(y[fold == k], np.expm1(L[fold == k].mean(1))) for k in np.unique(fold)])
print(f"  equal-weight blend  = {eq:.5f}   (robust fallback, no fitting)")
