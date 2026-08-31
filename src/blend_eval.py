#!/usr/bin/env python
"""
What does a new member actually add to the blend?

Standalone score is the wrong question. §1c established the rule the hard way: three
genuinely decorrelated families (CatBoost 0.974, XGBoost 0.973, Ridge 0.943 against the
existing family) moved the honestly-fitted blend by +0.00001. Decorrelation is necessary but
not sufficient -- it must be decorrelation AT COMPARABLE QUALITY.

So for each candidate this reports three things together, because any one alone misleads:
  * its standalone CV
  * its log-space correlation with every existing member
  * the blend CV with it added, weights refitted leave-one-fold-out

and, critically, the NO-OP CONTROL: the same base set refitted without the candidate. §1b was
only interpretable because the control was run -- refitting alone cost +0.00024, so the
classifier's true marginal contribution was +0.00007 rather than the +0.00031 it appeared to
be. Without the control a null reads as a small loss and the reason is invisible.

Weights are fitted on four folds and applied to the fifth, so every number here is honest.
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from metrics import rmsle                       # noqa: E402


def load(exps):
    """Align every member on the INTERSECTION of (fold_id, user_id).

    The usercv_* members keep ~8% more users per anchor than the frozen-fold protocol (it
    applies a different activity restriction), so their key sets overlap at 99.2% rather than
    matching exactly. That is a population difference, not a semantic one: y_true agrees on
    100.0000% of the 1,062,003 shared keys, which could not happen if fold_id meant different
    things in the two files.

    So the join is sound -- but EVERY member, including the base, must then be evaluated on
    the same intersected rows. Comparing a base blend scored on 1,071,040 rows against a
    candidate blend scored on 1,062,003 would attribute a population change to the candidate.
    """
    frames = {}
    for e in exps:
        f = ROOT / "oof" / f"{e}.parquet"
        if not f.exists():
            print(f"  SKIP {e}: no OOF file"); continue
        frames[e] = pd.read_parquet(f)[["fold_id", "user_id", "y_true", "y_pred"]]

    inter = None
    for e, d in frames.items():
        k = set(zip(d.fold_id.to_numpy().tolist(), d.user_id.to_numpy().tolist()))
        inter = k if inter is None else (inter & k)
    print(f"  aligned on {len(inter):,} shared (fold_id, user_id) keys across "
          f"{len(frames)} members")
    for e, d in frames.items():
        n = len(d)
        print(f"    {e:24s} {n:>10,} rows -> keeps {len(inter)/n:6.2%}")

    idx = pd.MultiIndex.from_tuples(sorted(inter), names=["fold_id", "user_id"])
    out, ref = {}, None
    for e, d in frames.items():
        g = d.set_index(["fold_id", "user_id"]).reindex(idx).reset_index()
        assert g.y_pred.notna().all(), f"{e}: reindex produced NaN"
        if ref is None:
            ref = g
        else:
            assert np.allclose(ref.y_true.to_numpy(), g.y_true.to_numpy()), \
                f"{e}: y_true disagrees with the first member on shared keys"
        out[e] = g
    return out, (ref.fold_id.to_numpy(), ref.user_id.to_numpy())


def fit_loo(M, y, folds):
    """Non-negative weights summing to 1, fitted on the other folds, applied to the held-out
    one. M is (n_rows, n_members) in LOG space."""
    pred = np.zeros(len(y))
    W = []
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        n = M.shape[1]

        def obj(w):
            w = np.abs(w); w = w / max(w.sum(), 1e-12)
            return float(np.sqrt(np.mean((np.log1p(y[tr]) - M[tr] @ w) ** 2)))

        r = minimize(obj, np.full(n, 1.0 / n), method="Nelder-Mead",
                     options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-9})
        w = np.abs(r.x); w = w / max(w.sum(), 1e-12)
        W.append(w)
        pred[te] = M[te] @ w
    return float(rmsle(y, np.maximum(np.expm1(pred), 0.0))), np.mean(W, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", nargs="+", required=True, help="members already in the blend")
    ap.add_argument("--cand", nargs="*", default=[], help="candidates to test one at a time")
    args = ap.parse_args()

    names = list(dict.fromkeys(args.base + args.cand))
    d, keys = load(names)
    base = [e for e in args.base if e in d]
    cand = [e for e in args.cand if e in d]
    any_d = next(iter(d.values()))
    y = any_d.y_true.to_numpy()
    folds = any_d.fold_id.to_numpy()
    L = {e: np.log1p(np.maximum(v.y_pred.to_numpy(), 0.0)) for e, v in d.items()}

    print(f"\n{len(y):,} rows, {len(np.unique(folds))} folds\n")
    print("=== STANDALONE ===")
    for e in names:
        if e in d:
            print(f"  {e:24s} {rmsle(y, np.maximum(d[e].y_pred.to_numpy(), 0)):.5f}")

    print("\n=== LOG-SPACE CORRELATION ===")
    ks = [e for e in names if e in d]
    print("  " + " " * 24 + "".join(f"{e[:10]:>11s}" for e in ks))
    for a in ks:
        print(f"  {a:24s}" + "".join(f"{np.corrcoef(L[a], L[b])[0,1]:>11.4f}" for b in ks))

    M = np.column_stack([L[e] for e in base])
    b_cv, b_w = fit_loo(M, y, folds)
    print(f"\n=== BASE BLEND ({len(base)} members) ===")
    print(f"  CV {b_cv:.5f}")
    for e, w in sorted(zip(base, b_w), key=lambda t: -t[1]):
        print(f"    {e:24s} weight {w:.4f}")

    if cand:
        print(f"\n=== ADDING EACH CANDIDATE (weights refitted, LOO) ===")
        print(f"  {'candidate':24s} {'blend CV':>10s} {'delta':>10s} {'its weight':>11s}")
        print(f"  {'(base, no-op control)':24s} {b_cv:>10.5f} {0.0:>+10.5f} {'-':>11s}")
        for c in cand:
            mm = np.column_stack([L[e] for e in base + [c]])
            cv, w = fit_loo(mm, y, folds)
            print(f"  {c:24s} {cv:>10.5f} {cv-b_cv:>+10.5f} {w[-1]:>11.4f}")
        mm = np.column_stack([L[e] for e in base + cand])
        cv, w = fit_loo(mm, y, folds)
        print(f"  {'ALL candidates':24s} {cv:>10.5f} {cv-b_cv:>+10.5f}")
        for e, ww in sorted(zip(base + cand, w), key=lambda t: -t[1]):
            print(f"    {e:24s} weight {ww:.4f}")


if __name__ == "__main__":
    main()
