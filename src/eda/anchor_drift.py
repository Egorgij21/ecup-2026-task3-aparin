#!/usr/bin/env python
"""
Do our rolling-origin cut-offs actually transfer? (PAPERS_new.md §8.5)

Our whole CV design rests on an assumption we have never tested: that a model trained at
cut-off t1 learns something that transfers to t2, and ultimately to the real cut-off of
2026-02-13. §8.5 gives a way to test it -- treat each cut-off's feature matrix as an
empirical multivariate normal and measure the Wasserstein distance between them. The paper
reports correlations above 0.60 between that distance and actual transfer loss.

This matters concretely rather than academically: we measured a CV-LB gap of +0.11 and
attributed it to fold-period predictability being lower than the test period's. If the test
cut-off sits far from every training cut-off in feature space, that is an independent
confirmation -- and §8.5 says it would be advance warning that CV overstates the LB.

The MVN assumption is crude for our heavy-tailed features, so as the paper suggests
everything is fitted on log1p-transformed features where normality is far more plausible.

Wasserstein-2 between N(m1,S1) and N(m2,S2):
    ||m1-m2||^2 + tr(S1 + S2 - 2 (S2^1/2 S1 S2^1/2)^1/2)
computed on a diagonal approximation -- with ~830 features the full matrix square root is
both expensive and badly conditioned, and the diagonal form already answers "how far apart".
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import yaml

ROOT = Path("/path/to/ecup")
sys.path.insert(0, str(ROOT / "src"))
from data import Panel                      # noqa: E402
from features import build                  # noqa: E402


def hdr(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}", flush=True)


import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--config", default="configs/e0020_fixedrounds.yaml")
_args = _ap.parse_args()
cfg = yaml.safe_load((ROOT / _args.config).read_text())
print(f"  feature set from {_args.config}")
spec = json.loads((ROOT / "data" / "fold_spec.json").read_text())
p = Panel()

anchors = [date.fromisoformat(f["valid_anchor"]) for f in spec["folds"]]
train_pool = sorted({date.fromisoformat(x) for f in spec["folds"] for x in f["train_anchors"]})
TEST = p.dmax
pts = [("train", a) for a in train_pool[::4]] + [("valid", a) for a in anchors] + [("TEST", TEST)]

hdr("1 -- FIT A DIAGONAL MVN TO EACH CUT-OFF'S FEATURE MATRIX (log1p space)")
stats = []
for kind, a in pts:
    ai = p.idx(a)
    keep = p.active_in(ai - 29, ai)
    X, names = build(p, ai, keep, cfg["feature_blocks"])
    if cfg.get("feature_exclude_patterns"):
        import re as _re
        pats = [_re.compile(x) for x in cfg["feature_exclude_patterns"]]
        sel = [i for i, n in enumerate(names) if not any(q.search(n) for q in pats)]
        X = X[:, sel]; names = [names[i] for i in sel]
    L = np.log1p(np.abs(np.nan_to_num(X, posinf=0, neginf=0)))
    stats.append((kind, a, L.mean(0), L.var(0) + 1e-9, int(keep.sum())))
    print(f"  {kind:6s} {a}  n={int(keep.sum()):>7,}  features={X.shape[1]}")
    del X, L


def w2(s1, s2):
    m1, v1 = s1[2], s1[3]
    m2, v2 = s2[2], s2[3]
    return float(np.sum((m1 - m2) ** 2) + np.sum(v1 + v2 - 2 * np.sqrt(v1 * v2)))


hdr("2 -- WASSERSTEIN-2 DISTANCE BETWEEN CUT-OFFS")
lab = [f"{k[:3]}:{a.strftime('%m-%d')}" for k, a, *_ in stats]
D = np.zeros((len(stats), len(stats)))
for i in range(len(stats)):
    for j in range(len(stats)):
        D[i, j] = w2(stats[i], stats[j])
print("        " + "".join(f"{l:>11s}" for l in lab[-6:]))
for i, l in enumerate(lab):
    print(f"  {l:9s} " + "".join(f"{D[i, j]:>11.3f}" for j in range(len(stats) - 6, len(stats))))

hdr("3 -- IS THE TEST CUT-OFF AN OUTLIER?")
ti = len(stats) - 1
d_to_test = np.array([D[ti, j] for j in range(ti)])
tr_idx = [j for j in range(ti) if stats[j][0] == "train"]
va_idx = [j for j in range(ti) if stats[j][0] == "valid"]
inter = [D[i, j] for i in range(ti) for j in range(i + 1, ti)]
print(f"  mean distance among training/validation cut-offs : {np.mean(inter):.4f}")
print(f"  mean distance from TEST to training cut-offs     : {np.mean(D[ti, tr_idx]):.4f}")
print(f"  mean distance from TEST to validation cut-offs   : {np.mean(D[ti, va_idx]):.4f}")
print(f"  nearest cut-off to TEST                          : {lab[int(np.argmin(d_to_test))]} "
      f"({d_to_test.min():.4f})")
r = np.mean(D[ti, tr_idx]) / max(np.mean(inter), 1e-9)
print(f"\n  ratio (TEST-to-train) / (train-to-train) = {r:.2f}")
if r > 1.5:
    print("  -> the test cut-off IS an outlier in feature space. §8.5 says this is advance")
    print("     warning that CV overstates the LB -- which is exactly the +0.11 gap we measured.")
else:
    print("  -> the test cut-off sits within the spread of the training cut-offs, so the")
    print("     CV-LB gap is NOT explained by feature-space drift and must come from elsewhere.")

hdr("4 -- WHICH FEATURES DRIFT MOST BETWEEN THE LAST FOLD AND TEST (§8.6 attribution)")
last = [s for s in stats if s[0] == "valid"][-1]
tst = stats[-1]
per = (last[2] - tst[2]) ** 2 + (last[3] + tst[3] - 2 * np.sqrt(last[3] * tst[3]))
o = np.argsort(-per)
print(f"  {'feature':44s} {'contribution':>13s} {'mean last':>10s} {'mean test':>10s}")
for i in o[:20]:
    print(f"  {names[i]:44s} {per[i]:>13.4f} {last[2][i]:>10.3f} {tst[2][i]:>10.3f}")
print(f"\n  top-20 features carry {100 * per[o[:20]].sum() / per.sum():.1f}% of the total drift")

(ROOT / "reports" / "eda").mkdir(parents=True, exist_ok=True)
json.dump({"labels": lab, "W2": D.tolist(), "ratio_test_vs_train": float(r)},
          open(ROOT / "reports" / "eda" / "anchor_drift.json", "w"), indent=2)
print("\n  wrote reports/eda/anchor_drift.json")
