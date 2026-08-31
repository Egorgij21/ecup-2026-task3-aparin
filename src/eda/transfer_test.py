#!/usr/bin/env python
"""
Does feature drift actually PREDICT transfer loss? -- the link I assumed and never tested.

anchor_drift.py measured that the test cut-off sits 3.92x further from the training cut-offs
than they sit from each other, and I proposed dropping the features responsible. That skips a
step: a distance is not a cost. A feature can drift and still be perfectly useful if its
RELATIONSHIP to the target is stable, and the evidence we already have points that way --
93 % delta transfer from CV to LB, and a CV-LB gap that is stable rather than widening.

This tests the link directly and for free. Train one model per fold's anchor set, then
evaluate every model on every fold's validation anchor. That gives a 5x5 transfer matrix.
Correlate the off-diagonal degradation against the Wasserstein distance between the same
pairs of cut-offs:

  * strong positive correlation -> distance predicts transfer loss, and de-drifting the
    features is justified even though it costs CV;
  * weak or no correlation -> the drift is real but harmless, dropping features is pure
    cost, and the CV-LB gap has to be explained by something else.

Both models are trained on the SAME anchors they always are, so nothing here changes the
frozen fold definition; this is a diagnostic over the existing scheme.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import yaml

ROOT = Path("/path/to/ecup")
sys.path.insert(0, str(ROOT / "src"))
from data import Panel                      # noqa: E402
from features import build                  # noqa: E402
from metrics import rmsle                   # noqa: E402


def hdr(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}", flush=True)


cfg = yaml.safe_load((ROOT / "configs" / "e0020_fixedrounds.yaml").read_text())
spec = json.loads((ROOT / "data" / "fold_spec.json").read_text())
folds = pl.read_parquet(ROOT / "data" / "folds.parquet")
p = Panel()
import lightgbm as lgb

params = dict(cfg["lgb_params"]); params["seed"] = cfg["seed"]
ROUNDS = int(cfg["fixed_rounds"])
K = len(spec["folds"])

hdr("1 -- TRAIN ONE MODEL PER FOLD'S ANCHOR SET")
models, mstats = [], []
for k in range(K):
    tr = [date.fromisoformat(x) for x in spec["folds"][k]["train_anchors"]]
    X, y, S = [], [], []
    for a in tr:
        ai = p.idx(a); keep = p.active_in(ai - 29, ai)
        Xb, names = build(p, ai, keep, cfg["feature_blocks"])
        X.append(Xb); y.append(np.log1p(p.target(ai)[keep]))
        S.append(np.log1p(np.abs(np.nan_to_num(Xb, posinf=0, neginf=0))))
    X = np.concatenate(X); y = np.concatenate(y); S = np.concatenate(S)
    mstats.append((S.mean(0), S.var(0) + 1e-9))
    models.append(lgb.train(params, lgb.Dataset(X, y, feature_name=names),
                            num_boost_round=ROUNDS, callbacks=[lgb.log_evaluation(0)]))
    print(f"  model {k}: trained on {len(tr)} anchors, {X.shape[0]:,} rows")
    del X, y, S

hdr("2 -- TRANSFER MATRIX: model trained on fold i, scored on fold j's validation anchor")
T = np.full((K, K), np.nan)
for j in range(K):
    va = date.fromisoformat(spec["folds"][j]["valid_anchor"]); vai = p.idx(va)
    vk = p.active_in(vai - 29, vai)
    Xva, _ = build(p, vai, vk, cfg["feature_blocks"])
    yva = folds.filter(pl.col("fold_id") == j).sort("user_id")["target"].to_numpy()
    for i in range(K):
        T[i, j] = rmsle(yva, np.maximum(np.expm1(models[i].predict(Xva)), 0.0))
    del Xva
print("      " + "".join(f"{'eval f'+str(j):>10s}" for j in range(K)))
for i in range(K):
    star = "  <- its own fold" if False else ""
    print(f"  tr{i} " + "".join(f"{T[i, j]:>10.5f}" for j in range(K)))

hdr("3 -- DOES DISTANCE PREDICT DEGRADATION?")
# degradation = how much worse model i is on fold j than the model actually trained for j
best = np.array([T[j, j] for j in range(K)])
deg, dist = [], []
for i in range(K):
    for j in range(K):
        if i == j:
            continue
        m1, v1 = mstats[i]; m2, v2 = mstats[j]
        d = float(np.sum((m1 - m2) ** 2) + np.sum(v1 + v2 - 2 * np.sqrt(v1 * v2)))
        deg.append(T[i, j] - best[j]); dist.append(d)
deg, dist = np.array(deg), np.array(dist)
r = float(np.corrcoef(dist, deg)[0, 1])
print(f"  {'pair':10s} {'W2 distance':>13s} {'degradation':>13s}")
o = np.argsort(-dist)
for t in o[:10]:
    i, j = divmod(t if t < K * K else 0, 1)
    print(f"  {'':10s} {dist[t]:>13.3f} {deg[t]:>+13.5f}")
print(f"\n  corr(W2 distance, transfer degradation) = {r:+.4f}   over {len(deg)} ordered pairs")
print(f"  mean degradation when transferring across folds = {deg.mean():+.5f}")
print(f"  worst single transfer                           = {deg.max():+.5f}")
if r > 0.5:
    print("\n  -> distance PREDICTS transfer loss. De-drifting is justified despite its CV cost.")
elif r > 0.2:
    print("\n  -> weak positive link. De-drifting is defensible but not clearly worth its CV cost.")
else:
    print("\n  -> distance does NOT predict transfer loss here. The drift is real but harmless,")
    print("     dropping the features is pure cost, and the CV-LB gap needs another explanation.")

json.dump({"transfer": T.tolist(), "corr_dist_degradation": r},
          open(ROOT / "reports" / "eda" / "transfer_test.json", "w"), indent=2)
print("\n  wrote reports/eda/transfer_test.json")
