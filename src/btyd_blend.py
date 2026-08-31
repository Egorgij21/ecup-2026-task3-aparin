#!/usr/bin/env python
"""
The pre-registered decision rule for BTYD (BTYD.md §6), plus the cheapest honest version of
BACKLOG e0033's "P(alive) as an extra column" route.

    python src/btyd_blend.py [--btyd e0170]

Three measurements, all leave-one-fold-out on the frozen folds:

  1. correlations of log-predictions against e0049 (gbdt) and e0101 (seq) -- the numbers
     BTYD.md §6 nominates as the decision inputs;
  2. fitted-weight blend WITH vs WITHOUT BTYD.  Weights live on the simplex (w >= 0,
     sum 1), which makes the objective a convex QP in log space, so SLSQP returns the
     global optimum and the with/without delta is not optimiser noise;
  3. a stack of the blend prediction against BTYD's latent columns (P(alive), E[X(30)],
     E[M]) -- WITH the no-op control that EXPERIMENTS.md §1b made a standing rule, i.e.
     refitting on the blend alone, so the extra columns' marginal value is separated from
     the refit itself.

Decision rule, fixed in advance: keep BTYD only if (2) beats the same blend without it by
more than 0.0005.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from metrics import rmsle          # noqa: E402

# the nine members of e0120: two gbdt + the seq family
E0120 = ["e0049", "e0064", "e0100", "e0101", "e0101s1", "e0101s2", "e0101s3", "e0102", "e0108"]


def load(exp: str) -> pd.DataFrame:
    return pd.read_parquet(ROOT / "oof" / f"{exp}.parquet").sort_values(["fold_id", "user_id"])


def fit_simplex(L: np.ndarray, y: np.ndarray) -> np.ndarray:
    """argmin_w RMSLE(y, expm1(L @ w))  s.t.  w >= 0, sum w = 1.  Convex in log space."""
    Ly = np.log1p(y)
    n = L.shape[1]
    f = lambda w: float(np.mean((Ly - L @ w) ** 2))
    g = lambda w: -2.0 * L.T @ (Ly - L @ w) / L.shape[0]
    r = minimize(f, np.full(n, 1.0 / n), jac=g, method="SLSQP",
                 bounds=[(0.0, 1.0)] * n,
                 constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0,
                               "jac": lambda w: np.ones(n)}],
                 options={"maxiter": 500, "ftol": 1e-14})
    w = np.clip(r.x, 0.0, None)
    return w / w.sum()


def lofo_blend(L: np.ndarray, y: np.ndarray, fold: np.ndarray) -> tuple[float, np.ndarray, list]:
    sc, ws = [], []
    for k in np.unique(fold):
        tr, te = fold != k, fold == k
        w = fit_simplex(L[tr], y[tr]); ws.append(w)
        sc.append(rmsle(y[te], np.expm1(L[te] @ w)))
    return float(np.mean(sc)), np.mean(ws, 0), [round(s, 5) for s in sc]


def lofo_stack(X: np.ndarray, y: np.ndarray, fold: np.ndarray) -> tuple[float, list]:
    """OLS in log space on the given design (intercept added), fitted leave-one-fold-out."""
    Ly = np.log1p(y)
    X = np.column_stack([np.ones(len(y)), X])
    sc = []
    for k in np.unique(fold):
        tr, te = fold != k, fold == k
        beta, *_ = np.linalg.lstsq(X[tr], Ly[tr], rcond=None)
        sc.append(rmsle(y[te], np.expm1(X[te] @ beta)))
    return float(np.mean(sc)), [round(s, 5) for s in sc]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--btyd", default="e0170")
    ap.add_argument("--members", nargs="*", default=E0120)
    args = ap.parse_args()

    base = load(args.members[0])
    y, fold = base.y_true.values, base.fold_id.values
    d = {e: load(e) for e in args.members}
    for e, df in d.items():
        assert np.array_equal(df.user_id.values, base.user_id.values), f"{e}: user order differs"
    bt = load(args.btyd)
    assert np.array_equal(bt.user_id.values, base.user_id.values), "btyd: user order differs"

    L_base = np.column_stack([np.log1p(d[e].y_pred.values) for e in args.members])
    L_bt = np.log1p(bt.y_pred.values)
    Ly = np.log1p(y)

    # ---------------------------------------------------------------- 1. correlations
    def cv(p):
        return float(np.mean([rmsle(y[fold == k], p[fold == k]) for k in np.unique(fold)]))

    print(f"\n  BTYD = {args.btyd}   own CV {cv(bt.y_pred.values):.5f}")
    print(f"  rho against truth (log space)        {np.corrcoef(Ly, L_bt)[0, 1]:.4f}")
    for e in ("e0049", "e0101"):
        if e in d:
            r = np.corrcoef(np.log1p(d[e].y_pred.values), L_bt)[0, 1]
            print(f"  corr(log BTYD, log {e})            {r:.4f}   "
                  f"[own CV {cv(d[e].y_pred.values):.5f}]")
    L_blend_eq = L_base.mean(1)
    print(f"  corr(log BTYD, log 9-member blend)   {np.corrcoef(L_blend_eq, L_bt)[0, 1]:.4f}")
    print(f"\n  reference thresholds (EXPERIMENTS.md §1c)")
    print(f"    e0049 <-> e0064   0.9983  -> blend gain ~0     (twins)")
    print(f"    gbdt  <-> e0101   0.9951  -> +0.00048 rho      (paid)")
    print(f"    usercv_ridge      0.9433  -> 0.00000           (decorrelated but too weak)")

    # ---------------------------------------------------------------- 2. the decision
    print(f"\n  --- leave-one-fold-out fitted blend (simplex weights) ---")
    s0, w0, f0 = lofo_blend(L_base, y, fold)
    s1, w1, f1 = lofo_blend(np.column_stack([L_base, L_bt]), y, fold)
    print(f"  without BTYD ({len(args.members)} members)   {s0:.5f}   folds {f0}")
    print(f"  with    BTYD ({len(args.members) + 1} members)   {s1:.5f}   folds {f1}")
    print(f"  DELTA = {s1 - s0:+.5f}   (threshold: better than -0.00050)")
    print(f"  mean fitted weight on BTYD = {w1[-1]:.5f}")
    print(f"  weights with BTYD: " +
          ", ".join(f"{e}={w:.3f}" for e, w in zip(args.members + [args.btyd], w1)))
    verdict = "KEEP" if (s1 - s0) < -0.0005 else "KILL"
    print(f"  >>> pre-registered verdict: {verdict}")

    # ---------------------------------------------------------------- 3. as columns
    print(f"\n  --- BTYD latent columns on top of the blend (BACKLOG e0033) ---")
    M = L_blend_eq
    cols = {"p_alive": bt.p_alive.values,
            "log_e_x30": np.log1p(bt.e_x30.values),
            "log_e_m": np.log1p(bt.e_m.values),
            "log_btyd": L_bt}
    ctrl, cf = lofo_stack(M[:, None], y, fold)
    print(f"  CONTROL: stack on the blend alone            {ctrl:.5f}   folds {cf}")
    for nm, v in cols.items():
        s, sf = lofo_stack(np.column_stack([M, v]), y, fold)
        print(f"  + {nm:-<36s} {s:.5f}   marginal {s - ctrl:+.5f}")
    s_all, _ = lofo_stack(np.column_stack([M] + list(cols.values())), y, fold)
    print(f"  + all four {'':-<27s} {s_all:.5f}   marginal {s_all - ctrl:+.5f}")
    print(f"  (equal-weight blend, unstacked, for reference: {cv(np.expm1(M)):.5f})")


if __name__ == "__main__":
    main()
