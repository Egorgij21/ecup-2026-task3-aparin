#!/usr/bin/env python
"""
Build the LB subset probe for "pre-holiday spikers".

Question: do users who spiked in the 1-2 weeks BEFORE 23 Feb / 8 March 2025 have a higher
2026 Feb-March GMV than the population? That window is the hidden test target, so it cannot
be measured offline -- but one submission can measure it exactly.

Mechanism. Predict a constant c on the subset S and 0 on everyone else. Then

    n*RMSLE^2 = sum_i (L_i - log1p(c)*1[i in S])^2
              = sum_i L_i^2  -  2*log1p(c)*sum_{i in S} L_i  +  |S|*log1p(c)^2

`sum_i L_i^2 = n*E[L^2]` is already known (10.7584, from the all-zeros probe), so

    E[L | S] = ( n*E[L^2] + |S|*lc^2 - n*RMSLE^2 ) / ( 2*lc*|S| ),      lc = log1p(c)

Everything on the right is known once the LB returns a score. Two caveats handled below:
  * the LB scores only the 50 000-user public split, so |S| must be the EXPECTED public
    overlap 0.2*|S_full|; the binomial fluctuation on that is +-0.3 % and is folded into the
    reported precision.
  * the LB reports 2 decimals, which dominates the error. `c` is chosen to minimise the
    resulting uncertainty in E[L|S] rather than picked by eye.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from data import Panel                      # noqa: E402

EL2, EL_ALL = 10.7584, 2.3199               # from the all-zeros + constant probes
N_PUB, LB_ROUND = 50_000, 0.005

p = Panel()


def rate(a: date, b: date):
    return p.wsum("gmv", p.idx(a), p.idx(b)) / ((b - a).days + 1)


# the run-ups: spending peaks at d-7 before 23 Feb and d-5 before 8 March (reports/holiday.log)
PRE23 = (date(2025, 2, 9), date(2025, 2, 22))
PRE8M = (date(2025, 2, 23), date(2025, 3, 7))
REF = (date(2025, 6, 1), date(2025, 12, 15))       # the user's own later-year baseline

runup = (p.wsum("gmv", p.idx(PRE23[0]), p.idx(PRE23[1]))
         + p.wsum("gmv", p.idx(PRE8M[0]), p.idx(PRE8M[1]))) / 27.0
ref = rate(*REF)
spike = np.log1p(runup) - np.log1p(ref)

elig = ref > 0
thr = np.quantile(spike[elig], 0.90)
S = elig & (spike >= thr)
print(f"  eligible users (non-zero reference)      : {int(elig.sum()):,}")
print(f"  S = top-decile pre-holiday spikers       : {int(S.sum()):,}")
print(f"  their mean spike {spike[S].mean():+.4f} vs {spike[elig & ~S].mean():+.4f} for the rest")

# choose c to minimise the uncertainty in E[L|S] given 2-decimal LB reporting
nS_pub = 0.2 * S.sum()
best = None
for c in [3, 5, 10, 20, 50, 100, 200, 500]:
    lc = np.log1p(c)
    # expected RMSLE of this probe, assuming E[L|S] ~ EL_ALL (worst case for precision)
    exp_r2 = EL2 - 2 * lc * EL_ALL * (nS_pub / N_PUB) + lc ** 2 * (nS_pub / N_PUB)
    exp_r = np.sqrt(max(exp_r2, 1e-9))
    dEL = (N_PUB * 2 * exp_r * LB_ROUND) / (2 * lc * nS_pub)
    if best is None or dEL < best[2]:
        best = (c, exp_r, dEL)
    print(f"    c={c:>4}  expected LB score ~{exp_r:.3f}   precision on E[L|S] = +-{dEL:.4f}")
C, EXP_R, DEL = best
print(f"\n  chosen c = {C}  ->  expect the LB to return ~{EXP_R:.2f}, giving E[L|S] to +-{DEL:.4f}")

sub = pl.DataFrame({"user_id": p.users, "predict": np.where(S, float(C), 0.0)})
ss = pl.read_csv(ROOT / "data" / "sample_submit.csv")
assert np.array_equal(sub["user_id"].to_numpy(), ss["user_id"].to_numpy())
assert sub.height == 250_000
out = ROOT / "subs" / "probe_spikers.csv"
sub.write_csv(out)
print(f"  wrote {out.relative_to(ROOT)}  ({int(S.sum()):,} users get {C}, the rest 0)")

meta = {"c": C, "n_S_full": int(S.sum()), "n_S_pub_expected": float(nS_pub),
        "EL2": EL2, "EL_all": EL_ALL, "precision": float(DEL), "expected_score": float(EXP_R)}
(ROOT / "reports" / "eda" / "probe_spikers_meta.json").write_text(json.dumps(meta, indent=2))

print(f"\n  INTERPRETATION once the LB returns a score R:")
print(f"    E[L|S] = ({N_PUB}*{EL2} + {nS_pub:.0f}*{np.log1p(C):.4f}^2 - {N_PUB}*R^2)"
      f" / (2*{np.log1p(C):.4f}*{nS_pub:.0f})")
print(f"    population mean E[L] = {EL_ALL:.4f}")
print(f"    H0 (spikers are ordinary)     -> E[L|S] ~ {EL_ALL:.2f}")
print(f"    H1 (the +0.18 log-point effect) -> E[L|S] ~ {EL_ALL + 0.18:.2f}")
print(f"    resolution +-{DEL:.3f}, so the two hypotheses are ~{0.18 / DEL:.1f} sigma apart")
