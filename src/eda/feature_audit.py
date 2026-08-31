#!/usr/bin/env python
"""
Sanity-audit every feature and feature family in the current set.

The CV score tells us whether a block helps on average; it says nothing about whether an
individual feature is degenerate, duplicated, silently NaN, or quietly reading the future.
With ~830 features -- most of them auto-generated across 7 windows and 2 columns -- those
failures are easy to ship and hard to notice.

Seven checks, in order of how much damage the failure would do:

  1. LOOK-AHEAD, per feature. Rebuild the whole set on a panel with everything after the
     anchor erased -- prefix sums frozen, EWMs frozen, RAW DAILY MATRICES ZEROED -- and name
     every column that changes. The raw matrices are the important part: block_sbc and
     block_tsfeat read them directly, so until now the guard proved nothing about them.
  2. Target correlation. Anything above ~0.9 in log space is a leak, not a feature.
  3. NaN / infinity, which LightGBM tolerates silently and which can mask a broken formula.
  4. Constant / near-constant columns -- dead weight that dilutes feature_fraction.
  5. Duplicate pairs (|r| > 0.9999) -- the 7-window grid generates these mechanically when a
     window exceeds the available history.
  6. Degenerate-by-construction columns: all-zero, or identical to another window of the
     same statistic.
  7. Per-family rollup, so a whole block can be judged rather than 644 columns read one by one.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import yaml

ROOT = Path("/path/to/ecup")
sys.path.insert(0, str(ROOT / "src"))
from data import Panel                      # noqa: E402
from features import build                  # noqa: E402

OUT = ROOT / "reports" / "eda"
CFG = ROOT / "configs" / "e0040_sbc_tsfeat.yaml"


def hdr(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}", flush=True)


cfg = yaml.safe_load(CFG.read_text())
spec = json.loads((ROOT / "data" / "fold_spec.json").read_text())
p = Panel()
A = date.fromisoformat(spec["folds"][-1]["valid_anchor"])
ai = p.idx(A)
keep = p.active_in(ai - 29, ai)
X, names = build(p, ai, keep, cfg["feature_blocks"])
y = pl.read_parquet(ROOT / "data" / "folds.parquet").filter(
    pl.col("fold_id") == len(spec["folds"]) - 1).sort("user_id")["target"].to_numpy()
L = np.log1p(y)
print(f"  anchor {A}   users {X.shape[0]:,}   features {X.shape[1]}")

# ============================================================================
hdr("1 -- LOOK-AHEAD, PER FEATURE (the check that matters most)")
arrays = p.prefix_arrays() + p.ewm_arrays() + p.raw_arrays()
saved = [a.copy() for a in arrays]
try:
    for a in arrays:
        if a.shape[1] > p.n_days:
            a[:, ai + 2:] = a[:, ai + 1][:, None]
        else:
            a[:, ai + 1:] = 0.0
    X2, _ = build(p, ai, keep, cfg["feature_blocks"])
finally:
    for a, s_ in zip(arrays, saved):
        a[...] = s_
    del saved
bad = ~np.isclose(np.nan_to_num(X), np.nan_to_num(X2), rtol=1e-9, atol=1e-9)
cols = np.where(bad.any(axis=0))[0]
if len(cols) == 0:
    print(f"  PASS -- all {X.shape[1]} features identical when the future is erased")
else:
    print(f"  FAIL -- {len(cols)} features change when the future is erased:")
    for c in cols[:40]:
        print(f"    {names[c]}   rows affected {int(bad[:, c].sum()):,}")
del X2

# ============================================================================
hdr("2 -- TARGET CORRELATION (a leak looks like a very high one)")
Xf = np.nan_to_num(X.astype(np.float64), posinf=0, neginf=0)
sd = Xf.std(0)
ok = sd > 0
c = np.zeros(X.shape[1])
c[ok] = ((Xf[:, ok] - Xf[:, ok].mean(0)) * (L - L.mean())[:, None]).sum(0) / (
    Xf.shape[0] * sd[ok] * L.std())
o = np.argsort(-np.abs(c))
print(f"  {'top 15 by |corr| with log1p(target)':44s} {'corr':>8s}")
for i in o[:15]:
    print(f"    {names[i]:42s} {c[i]:>8.4f}")
sus = [names[i] for i in range(len(names)) if abs(c[i]) > 0.90]
print(f"\n  features with |corr| > 0.90 (leak suspects): {sus if sus else 'NONE'}")

# ============================================================================
hdr("3 -- NaN / INF")
nan = np.isnan(X).sum(0); inf = np.isinf(X).sum(0)
bad_n = [(names[i], int(nan[i]), int(inf[i])) for i in range(len(names)) if nan[i] or inf[i]]
print(f"  columns containing NaN or inf: {len(bad_n)}")
for n_, a_, b_ in bad_n[:25]:
    print(f"    {n_:42s} nan={a_:>8,} inf={b_:>8,}")

# ============================================================================
hdr("4 -- CONSTANT / NEAR-CONSTANT")
const = [names[i] for i in range(len(names)) if sd[i] == 0]
near = [names[i] for i in range(len(names)) if 0 < sd[i] < 1e-8]
print(f"  exactly constant : {len(const)}")
for n_ in const[:25]:
    print(f"    {n_}")
print(f"  near-constant (sd < 1e-8): {len(near)}")

# ============================================================================
hdr("5 -- DUPLICATE PAIRS (|r| > 0.9999), on a 40k-user subsample")
rng = np.random.default_rng(0)
idx = rng.choice(Xf.shape[0], size=min(40000, Xf.shape[0]), replace=False)
Z = Xf[idx][:, ok]
Z = (Z - Z.mean(0)) / np.maximum(Z.std(0), 1e-12)
nm_ok = [names[i] for i in range(len(names)) if ok[i]]
C = (Z.T @ Z) / Z.shape[0]
iu = np.triu_indices(C.shape[0], k=1)
dup = np.where(np.abs(C[iu]) > 0.9999)[0]
print(f"  duplicate pairs: {len(dup)}")
seen = defaultdict(list)
for d_ in dup[:400]:
    a_, b_ = iu[0][d_], iu[1][d_]
    seen[nm_ok[a_]].append(nm_ok[b_])
for k_, v_ in list(seen.items())[:20]:
    print(f"    {k_:42s} == {', '.join(v_[:3])}")
print(f"  -> these are mechanical: a window longer than the available history collapses onto")
print(f"     the longest one that fits. Harmless for a tree, but they dilute feature_fraction.")

# ============================================================================
hdr("6 -- ALL-ZERO COLUMNS")
allz = [names[i] for i in range(len(names)) if np.all(X[:, i] == 0)]
print(f"  all-zero columns: {len(allz)}")
for n_ in allz[:25]:
    print(f"    {n_}")

# ============================================================================
hdr("7 -- PER-FAMILY ROLLUP")
fam = defaultdict(list)
for i, n_ in enumerate(names):
    f_ = "sbc" if n_.startswith("sbc_") else ("ts" if any(
        n_.startswith(t) for t in ("skew_", "kurt_", "trendslope_", "autocorr", "weeklypower_",
                                   "specentropy_", "mad_", "maxshare_", "dutycycle_",
                                   "switches_", "maxzerorun_", "distshift_")) else "core")
    fam[f_].append(i)
print(f"  {'family':8s} {'n':>6s} {'constant':>9s} {'nan/inf':>8s} {'max|corr|':>10s} {'mean|corr|':>11s}")
for f_, ii in fam.items():
    ii = np.array(ii)
    print(f"  {f_:8s} {len(ii):>6d} {int((sd[ii] == 0).sum()):>9d} "
          f"{int(((nan[ii] + inf[ii]) > 0).sum()):>8d} {np.abs(c[ii]).max():>10.4f} "
          f"{np.abs(c[ii]).mean():>11.4f}")

json.dump({"n_features": int(X.shape[1]), "lookahead_fail": [names[i] for i in cols],
           "constant": const, "allzero": allz, "nan_inf": [b[0] for b in bad_n],
           "leak_suspects": sus},
          open(OUT / "feature_audit.json", "w"), indent=2)
print("\n  wrote reports/eda/feature_audit.json")
