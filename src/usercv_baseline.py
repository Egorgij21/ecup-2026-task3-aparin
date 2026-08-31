#!/usr/bin/env python
"""
CAUSAL_EXP.md §6's missing row: the naive baseline, on this exact protocol.

    python src/usercv_baseline.py

"a baseline row: predict rolling_mean(gmv, 30) at every timestamp. If a variant doesn't beat
this, say so plainly."  Scored under the identical tmask, target and user folds as
src/run_usercv.py, so it is directly comparable to the three variants' unseen-user numbers.

Two naive predictors, because the doc names one and this repo's measurements name another:
  p30    the trailing 30-day GMV sum -- the doc's `rolling_mean(gmv,30)` rescaled to the sum
         target, and exactly what `sample_submit.csv` contains (DATA.md §9.3).
  geo3   expm1(mean log1p of the last three 30-day blocks) -- the strongest naive predictor
         this project has found (DATA.md §9), and the reference every date-split experiment
         is scored against.  Included so `delta vs naive` means the same thing in both
         protocols.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from run_usercv import hash_fold                                              # noqa: E402
from usercv_features import HORIZON, Raw, build_target, build_tmask, max_anchor  # noqa: E402


def main() -> None:
    raw = Raw()
    last_anchor = max_anchor(raw)
    Y = build_target(raw, "sum")
    M = build_tmask(raw, last_anchor, burn_in=14, trim_to_first_seen=True)

    cs = np.concatenate([np.zeros((raw.n, 1)),
                         np.cumsum(raw.col["gmv"], 1, dtype=np.float64)], 1)

    def block(j: int) -> np.ndarray:
        """sum of gmv over [t-29-30j, t-30j], causal, zero where it runs off the start."""
        t = np.arange(raw.T)
        hi, lo = t - 30 * j, t - 29 - 30 * j
        return (cs[:, np.maximum(hi, 0) + 1] - cs[:, np.maximum(lo, 0)]) * (hi >= 0)

    p30 = np.log1p(block(0)).astype(np.float32)
    geo3 = (np.log1p(block(0)) + np.log1p(block(1)) + np.log1p(block(2))).astype(np.float32) / 3.0

    csa = np.concatenate([np.zeros((raw.n, 1), np.int32),
                          np.cumsum(raw.active > 0, 1, dtype=np.int32)], 1)
    lo = np.maximum(np.arange(raw.T) - 29, 0)
    POP = (csa[:, 1:] - csa[:, lo]) > 0

    fold_of = hash_fold(raw.users)
    out = {}
    print(f"\n  {'predictor':10s} {'UNSEEN':>9s} {'in-pop':>9s}   per-fold (unseen)")
    for nm, P in (("p30", p30), ("geo3", geo3)):
        per, perp = [], []
        for k in range(5):
            u = fold_of == k
            m = M[u]
            e2 = (P[u] - Y[u]) ** 2
            per.append(float(np.sqrt((e2 * m).sum() / m.sum())))
            mp = m & POP[u]
            perp.append(float(np.sqrt((e2 * mp).sum() / mp.sum())))
        out[nm] = {"per_fold": per, "mean": float(np.mean(per)),
                   "in_population_per_fold": perp, "in_population_mean": float(np.mean(perp))}
        print(f"  {nm:10s} {np.mean(per):>9.5f} {np.mean(perp):>9.5f}   "
              f"{[round(x, 5) for x in per]}")

    print(f"\n  variants vs the naive floor (paired, per fold):")
    for v in ("gmv_only", "full", "extra"):
        f = ROOT / "reports" / "eda" / f"usercv_{v}.json"
        if not f.exists():
            continue
        pf = np.array(json.loads(f.read_text())["per_fold"])
        for nm in ("p30", "geo3"):
            d = pf - np.array(out[nm]["per_fold"])
            print(f"    {v:9s} vs {nm:5s}  {d.mean():+.5f}   wins {int((d < 0).sum())}/5")

    dst = ROOT / "reports" / "eda" / "usercv_baseline.json"
    dst.write_text(json.dumps(out, indent=2))
    print(f"\n  wrote reports/eda/usercv_baseline.json")


if __name__ == "__main__":
    main()
