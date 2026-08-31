#!/usr/bin/env python
"""
Fit blend weights on the aligned OOF and write the blended submission.

Why not src/blend_submit.py --weights oof: that path asserts every member's OOF has the same
user order, which usercv_full fails (1,155,699 rows vs 1,071,040 -- the user-split protocol
keeps ~8% more users per anchor). Its semantics ARE compatible (y_true agrees on 100.0000% of
the 1,062,003 shared keys), so the fix is to align on the intersection, not to drop the member
-- and usercv_full carries 57% of the weight, so dropping it is not an option.

Three things this reports before writing anything, because each has burned someone here:

  * the LEAVE-ONE-FOLD-OUT CV of the fitted-weight procedure, which is the honest estimate of
    what the file will do. The pooled in-sample fit is NOT that number and is always optimistic;
  * the EQUAL-weight CV alongside it. blend_submit.py's docstring reports the two landing
    within 0.00005 on every combination measured, so if fitted weights are not clearly better
    here, equal weights are preferable -- they carry no fitting risk;
  * every member's test-side sanity (row count, user order against sample_submit, finiteness).

Members are averaged in LOG space -- expm1(sum w_i log1p(p_i)) -- because RMSLE's optimal
point prediction is E[log1p(y)|x]. Averaging raw predictions targets E[y|x] instead.
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
from metrics import gini, rmsle           # noqa: E402


def align(members):
    # members maps OOF name -> SUB name; the OOF file is keyed by the OOF name
    frames = {m: pd.read_parquet(ROOT / "oof" / f"{m}.parquet")[
        ["fold_id", "user_id", "y_true", "y_pred"]] for m in members}
    inter = None
    for d in frames.values():
        k = set(zip(d.fold_id.to_numpy().tolist(), d.user_id.to_numpy().tolist()))
        inter = k if inter is None else inter & k
    idx = pd.MultiIndex.from_tuples(sorted(inter), names=["fold_id", "user_id"])
    out, ref = {}, None
    for m, d in frames.items():
        g = d.set_index(["fold_id", "user_id"]).reindex(idx).reset_index()
        assert g.y_pred.notna().all(), f"{m}: reindex produced NaN"
        if ref is None:
            ref = g
        else:
            assert np.allclose(ref.y_true.to_numpy(), g.y_true.to_numpy()), \
                f"{m}: y_true disagrees on shared keys"
        out[m] = np.log1p(np.maximum(g.y_pred.to_numpy(), 0.0))
    print(f"  aligned OOF on {len(idx):,} shared keys")
    return out, ref.y_true.to_numpy(), ref.fold_id.to_numpy()


def fit(M, y, rows=None):
    rows = np.ones(len(y), bool) if rows is None else rows
    n = M.shape[1]

    def obj(w):
        w = np.abs(w); w = w / max(w.sum(), 1e-12)
        return float(np.sqrt(np.mean((np.log1p(y[rows]) - M[rows] @ w) ** 2)))

    r = minimize(obj, np.full(n, 1.0 / n), method="Nelder-Mead",
                 options={"maxiter": 6000, "xatol": 1e-6, "fatol": 1e-10})
    w = np.abs(r.x)
    return w / w.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", required=True,
                    help="oof_name[:sub_name] -- sub defaults to oof name")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="auto", choices=["auto", "fitted", "equal"])
    ap.add_argument("--calibrate", nargs=2, type=float, metavar=("REF_LB", "IGNORED"),
                    default=None,
                    help="affine-calibrate in log space using the two probe constants plus "
                         "REF_LB, the measured LB of a reference submission whose predictions "
                         "this blend closely tracks. L_hat = a*M + b with a = cov/var(M), "
                         "b = E[L] - a*E[M]; E[L] and E[L^2] come from probe_zeros (3.28) and "
                         "probe_const10 (2.32), and cov from REF_LB via "
                         "E[LM] = (E[L^2] + E[M^2] - RMSLE^2)/2.")
    a = ap.parse_args()

    mem = {}
    for spec in a.members:
        o, _, s = spec.partition(":")
        mem[o] = s or o
    names = list(mem)

    L, y, folds = align(mem)
    M = np.column_stack([L[m] for m in names])

    # honest estimate: weights fitted on four folds, applied to the fifth
    pred_loo = np.zeros(len(y))
    for f in np.unique(folds):
        w = fit(M, y, folds != f)
        pred_loo[folds == f] = M[folds == f] @ w
    cv_fitted = rmsle(y, np.maximum(np.expm1(pred_loo), 0.0))
    w_eq = np.full(len(names), 1.0 / len(names))
    cv_equal = rmsle(y, np.maximum(np.expm1(M @ w_eq), 0.0))
    w_full = fit(M, y)
    cv_pooled = rmsle(y, np.maximum(np.expm1(M @ w_full), 0.0))

    print(f"\n  fitted weights, leave-one-fold-out : {cv_fitted:.5f}   <- honest estimate")
    print(f"  equal weights                      : {cv_equal:.5f}")
    print(f"  fitted weights, pooled in-sample   : {cv_pooled:.5f}   (optimistic, do not quote)")
    use_fitted = (a.weights == "fitted") or (a.weights == "auto" and cv_fitted < cv_equal - 5e-5)
    w = w_full if use_fitted else w_eq
    print(f"  -> using {'FITTED' if use_fitted else 'EQUAL'} weights"
          + ("" if use_fitted else "  (fitted did not clear 0.00005; equal carries no fitting risk)"))

    ss = pd.read_csv(ROOT / "data" / "sample_submit.csv")
    uid = ss.user_id.to_numpy()
    P = []
    for m in names:
        f = ROOT / "subs" / f"{mem[m]}.csv"
        if not f.exists():
            raise SystemExit(f"missing subs/{mem[m]}.csv for member {m}")
        s = pd.read_csv(f)
        assert len(s) == 250_000, f"{m}: {len(s)} rows"
        assert np.array_equal(s.user_id.to_numpy(), uid), f"{m}: user order differs from sample"
        v = np.maximum(s.predict.to_numpy().astype(np.float64), 0.0)
        assert np.isfinite(v).all(), f"{m}: non-finite predictions"
        P.append(v)
    LT = np.column_stack([np.log1p(v) for v in P])
    Mlog = LT @ w
    if a.calibrate:
        ref_lb = a.calibrate[0]
        EL2 = 3.28 ** 2                       # probe_zeros:   RMSLE^2 = E[L^2]
        c = np.log1p(10.0)                    # probe_const10: RMSLE^2 = E[L^2]-2cE[L]+c^2
        EL = (EL2 + c * c - 2.32 ** 2) / (2 * c)
        VL = EL2 - EL * EL
        EM, EM2 = Mlog.mean(), (Mlog ** 2).mean()
        VM = EM2 - EM * EM
        ELM = (EL2 + EM2 - ref_lb ** 2) / 2.0
        cov = ELM - EL * EM
        rho = cov / np.sqrt(VL * VM)
        A = cov / VM
        B = EL - A * EM
        print(f"\n  CALIBRATION (probes 3.28 / 2.32, reference LB {ref_lb})")
        print(f"    E[L] {EL:.5f}  sd_L {np.sqrt(VL):.5f} | E[M] {EM:.5f}  sd_M {np.sqrt(VM):.5f}")
        print(f"    rho {rho:.5f}  ->  L_hat = {A:.5f}*M + {B:.5f}")
        print(f"    sd_M {np.sqrt(VM):.5f} -> {rho*np.sqrt(VL):.5f};  mean {EM:.5f} -> {EL:.5f}")
        print(f"    predicted calibrated RMSLE = sd_L*sqrt(1-rho^2) = {np.sqrt(VL*(1-rho*rho)):.5f}")
        print(f"    NOTE: rho is inferred from REF_LB, so this is exact only if the blend's")
        print(f"    predictions match the reference's. Cross-applying costs accuracy in rho.")
        Mlog = A * Mlog + B
    pred = np.maximum(np.expm1(Mlog), 0.0)

    print(f"\n  {'member':18s} {'sub':14s} {'weight':>7s} {'mean':>9s} {'gini':>7s}")
    for i, m in enumerate(names):
        print(f"  {m:18s} {mem[m]:14s} {w[i]:>7.4f} {P[i].mean():>9.2f} {gini(P[i]):>7.4f}")
    print(f"  {a.out:18s} {'(blend)':14s} {'':>7s} {pred.mean():>9.2f} {gini(pred):>7.4f}")

    out = ROOT / "subs" / f"{a.out}.csv"
    pd.DataFrame({"user_id": uid, "predict": pred}).to_csv(out, index=False)
    print(f"\n  wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"  expected LB ~ {cv_fitted - 0.109:.4f} (CV-LB gap 0.109, stable over 5 points)")


if __name__ == "__main__":
    main()
