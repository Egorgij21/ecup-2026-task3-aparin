#!/usr/bin/env python
"""
Turn the two LB probe scores into the hidden test target's log-moments, and say which
CV population hypothesis survives.

    python3 src/probe_solve.py <rmsle_zeros> [<rmsle_const> [c]]

With only <rmsle_zeros> it reports E[log1p(y)^2] and ranks the hypotheses.
With both it additionally solves E[log1p(y)] and Var[log1p(y)] exactly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# measured on our folds; see reports/eda_lbgap.log
HYP = {
    "clean anchors / re-selected  (protocol DATA.md 4.4)": {"rmsle0": 3.3053, "EL": 2.3585},
    "contaminated anchor 2026-01-14":                      {"rmsle0": 3.2036, "EL": 2.2421},
    "clean anchors / all 250k users":                      {"rmsle0": 3.0669, "EL": 1.9877},
}
LB_ROUNDING = 0.005     # LB appears to be reported to 2 decimals
LB_NOISE = 0.0059       # 50k-user sampling sd, reports/eda_selection.log section E


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    r0 = float(sys.argv[1].replace(",", "."))
    E2 = r0 ** 2
    # verdicts are compared in RMSLE units; E[L^2] uncertainty is that scaled by dE2/dr = 2r
    tol = LB_ROUNDING + 2 * LB_NOISE
    tol_E2 = 2 * r0 * tol

    print(f"\n  probe A (all zeros) RMSLE = {r0}")
    print(f"    => E[log1p(y)^2] = {E2:.4f}  (+/- {tol_E2:.4f} from LB rounding + 2sd noise)")
    print(f"    verdict tolerance on RMSLE = +/-{tol:.4f}\n")
    print(f"  {'hypothesis':54s} {'predicted':>10s} {'observed':>9s} {'diff':>8s}  verdict")
    best, bestd = None, 1e9
    for k, v in HYP.items():
        d = r0 - v["rmsle0"]
        verdict = "CONSISTENT" if abs(d) <= tol else "rejected"
        if abs(d) < bestd:
            best, bestd = k, abs(d)
        print(f"  {k:54s} {v['rmsle0']:>10.4f} {r0:>9.4f} {d:>+8.4f}  {verdict}")
    print(f"\n  closest: {best}  (|diff| = {bestd:.4f})")
    if bestd > tol:
        print("  NONE of the hypotheses fits -- the test target differs from every fold we")
        print("  can build. Re-derive the population rule before trusting any CV level.")

    if len(sys.argv) >= 3:
        rc = float(sys.argv[2].replace(",", "."))
        c = float(sys.argv[3]) if len(sys.argv) >= 4 else 10.0
        lc = np.log1p(c)
        EL = (E2 + lc ** 2 - rc ** 2) / (2 * lc)
        VarL = E2 - EL ** 2
        print(f"\n  probe B (constant c={c:g}, log1p(c)={lc:.4f}) RMSLE = {rc}")
        print(f"    => E[log1p(y)]   = {EL:.4f}")
        print(f"    => Var[log1p(y)] = {VarL:.4f}   sd = {np.sqrt(max(VarL, 0)):.4f}")
        print(f"\n  Best achievable by ANY constant prediction: RMSLE = sd = "
              f"{np.sqrt(max(VarL, 0)):.4f}  (at c = expm1(E[L]) = {np.expm1(EL):.3f})")
        print(f"\n  {'hypothesis':54s} {'E[L] pred':>10s} {'E[L] obs':>9s} {'diff':>8s}")
        for k, v in HYP.items():
            print(f"  {k:54s} {v['EL']:>10.4f} {EL:>9.4f} {EL - v['EL']:>+8.4f}")
        out = {"rmsle_zeros": r0, "rmsle_const": rc, "c": c,
               "E_L2": E2, "E_L": EL, "Var_L": VarL}
        p = ROOT / "reports" / "eda" / "probe_result.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2))
        print(f"\n  written {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
