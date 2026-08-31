#!/usr/bin/env python
"""
Does the model systematically UNDER-predict spikers? -- the CV-testable version.

The holiday-specific question ("do pre-8-March spikers spike again next March?") cannot be
validated: the feature only exists at the test anchor and the payoff is below LB noise
(best case -0.00078 vs 2-sigma 0.0118). But the mechanism underneath is general and free to
test on our frozen folds:

    if the model under-predicts users who recently spiked, then a spiker correction helps
    EVERYWHERE -- including the test window -- and it is CV-validatable.
    if their mean residual is ~0, the model already prices spikes correctly and the
    holiday-specific version has no reason to be special.

For each of the 5 frozen folds we recompute, from PRE-ANCHOR data only, each user's spike
(recent activity relative to their own longer-run baseline) and read off the mean OOF
residual by spike decile. No new model is trained; this reuses oof/e0020.parquet.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path("/path/to/ecup")
sys.path.insert(0, str(ROOT / "src"))
from data import Panel                      # noqa: E402
from metrics import rmsle                   # noqa: E402

OUT = ROOT / "reports" / "eda"
R: dict = {}


def hdr(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}", flush=True)


p = Panel()
oof = pl.read_parquet(ROOT / "oof" / "e0020.parquet")
spec = json.loads((ROOT / "data" / "fold_spec.json").read_text())

hdr("MEAN OOF RESIDUAL BY PRE-ANCHOR SPIKE DECILE")
print("  spike = log1p(GMV/day over the 30d before the anchor)")
print("        - log1p(GMV/day over days 31..210 before the anchor)   [own baseline]\n")
allres, allspk = [], []
for k in range(len(spec["folds"])):
    A = date.fromisoformat(spec["folds"][k]["valid_anchor"])
    ai = p.idx(A)
    keep = p.active_in(ai - 29, ai)
    recent = p.wsum("gmv", ai - 29, ai) / 30.0
    base = p.wsum("gmv", ai - 209, ai - 30) / 180.0
    spike = (np.log1p(recent) - np.log1p(base))[keep]

    d = oof.filter(pl.col("fold_id") == k).sort("user_id")
    y, pr = d["y_true"].to_numpy(), d["y_pred"].to_numpy()
    res = np.log1p(y) - np.log1p(pr)
    assert res.size == spike.size, (res.size, spike.size)
    allres.append(res); allspk.append(spike)

    ed = np.quantile(spike, np.linspace(0, 1, 11)[1:-1])
    dec = np.digitize(spike, ed)
    means = [res[dec == g].mean() for g in range(10)]
    print(f"  fold {k} ({A}) mean residual by spike decile (low -> high):")
    print("    " + "  ".join(f"{m:+.3f}" for m in means))

res = np.concatenate(allres); spk = np.concatenate(allspk)
ed = np.quantile(spk, np.linspace(0, 1, 11)[1:-1])
dec = np.digitize(spk, ed)
hdr("POOLED OVER ALL FOLDS")
print(f"  {'decile':8s} {'n':>9s} {'mean spike':>11s} {'mean residual':>14s}")
for g in range(10):
    m = dec == g
    print(f"  {g:<8d} {int(m.sum()):>9,} {spk[m].mean():>11.3f} {res[m].mean():>14.4f}")
top = dec == 9
print(f"\n  top-decile spikers   mean residual = {res[top].mean():+.4f}")
print(f"  everyone else        mean residual = {res[~top].mean():+.4f}")
print(f"  difference                          = {res[top].mean() - res[~top].mean():+.4f}")
print(f"  corr(spike, residual)               = {np.corrcoef(spk, res)[0, 1]:+.4f}")
R["top_decile_residual"] = float(res[top].mean())
R["diff"] = float(res[top].mean() - res[~top].mean())
R["corr"] = float(np.corrcoef(spk, res)[0, 1])

hdr("WHAT A SPIKER MULTIPLIER WOULD BUY ON CV")
y = oof["y_true"].to_numpy(); pr = oof["y_pred"].to_numpy()
base_s = rmsle(y, pr)
print(f"  {'k on top-decile spikers':28s} {'CV RMSLE':>10s} {'delta':>10s}")
print(f"  {'1.00 (unchanged)':28s} {base_s:>10.5f} {0.0:>+10.5f}")
best = (1.0, base_s)
for kk in [1.05, 1.1, 1.15, 1.2, 1.3, 1.5]:
    q = pr.copy(); q[top] *= kk
    s = rmsle(y, q)
    if s < best[1]:
        best = (kk, s)
    print(f"  {kk:<28.2f} {s:>10.5f} {s - base_s:>+10.5f}")
print(f"\n  best k = {best[0]:.2f} -> {best[1]:.5f} ({best[1] - base_s:+.5f}, "
      f"sigma_noise 0.00009)")
print("  NOTE this k is fitted ON the OOF it is evaluated on, so it is an UPPER BOUND;")
print("  a leave-one-fold-out fit would be the honest number if this looks worth pursuing.")
R["best_k"] = best[0]; R["best_rmsle"] = float(best[1]); R["base_rmsle"] = float(base_s)

(OUT).mkdir(parents=True, exist_ok=True)
(OUT / "spiker_residual.json").write_text(json.dumps(R, indent=2, default=str))
print("\n  wrote reports/eda/spiker_residual.json")
