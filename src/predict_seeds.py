#!/usr/bin/env python
"""
Test predictions for a MULTI-SEED averaged model (e0260 / e0266).

    python src/predict_seeds.py --config configs/e0260_regime.yaml --arm e0266 --out e0266

`predict_tuned.py` trains ONE model from a `*_params.json`. e0266 is e0093's tuned parameter
set averaged over 3 seeds, and e0260 is e0049's default set over 5 -- the averaging IS the
experiment, so shipping a single seed would ship a different model from the one confirmed on
the frozen folds.

Everything else is taken from `predict.py` / `predict_tuned.py` unchanged so the two paths
cannot drift: all clean anchors on the frozen 7-day grid ending 31 days before the guard zone,
features built at the test anchor from the same blocks, the same population assertion.

⚠ The seed average is taken in LOG space -- `expm1(mean_s(model_s(x)))`, not
`mean_s(expm1(...))`. RMSLE's optimal point prediction is `E[log1p y | x]`, so the log-space
mean estimates the quantity being modelled; averaging in linear space would estimate
`log1p(E[y])`, which EXPERIMENTS.md §1e prices at +0.5626. `run_regime.py` averages the same
way, so the submission matches its CV.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from data import Panel                    # noqa: E402
from features import build                # noqa: E402

GUARD_START, MIN_HISTORY, TRAIN_STRIDE = date(2025, 11, 16), 90, 7

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--arm", required=True, help="exp_id of the arm inside the config's `arms`")
ap.add_argument("--out", required=True)
a = ap.parse_args()

cfg = yaml.safe_load((ROOT / a.config).read_text())
arm = next((x for x in cfg["arms"] if x["exp_id"] == a.arm), None)
if arm is None:
    raise SystemExit(f"arm {a.arm} not in {a.config}")
seeds = [int(s) for s in arm.get("seeds", [cfg["seed"]])]
rounds = int(arm.get("rounds", cfg["fixed_rounds"]))
params = dict(cfg["lgb_params"]); params.update(arm.get("params", {}))
print(f"  arm {a.arm}: {len(seeds)} seeds {seeds}, {rounds} rounds", flush=True)
print(f"  change: {arm['change']}", flush=True)

p = Panel()
test_anchor = p.dmax
tai = p.idx(test_anchor)
latest, earliest = GUARD_START - timedelta(days=31), p.dmin + timedelta(days=MIN_HISTORY - 1)
anchors, d_ = [], latest
while d_ >= earliest:
    anchors.append(d_); d_ -= timedelta(days=TRAIN_STRIDE)
anchors = sorted(anchors)
print(f"  {len(anchors)} training anchors {anchors[0]} .. {anchors[-1]}", flush=True)

Xs, ys, names = [], [], None
for an in anchors:
    ai = p.idx(an)
    keep = p.active_in(ai - 29, ai)
    Xb, names = build(p, ai, keep, cfg["feature_blocks"])
    Xs.append(Xb); ys.append(np.log1p(p.target(ai)[keep]))
X = np.concatenate(Xs); y = np.concatenate(ys)
del Xs, ys

keep_test = p.active_in(tai - 29, tai)
assert keep_test.all(), "some test users fail the population rule"
Xte, nt = build(p, tai, keep_test, cfg["feature_blocks"])
assert nt == names, "test feature names differ from train"
print(f"  train {X.shape[0]:,} x {X.shape[1]}   test {Xte.shape[0]:,}", flush=True)

import lightgbm as lgb
preds = []
for sd in seeds:
    pr = dict(params)
    pr["seed"] = sd
    # set these explicitly: LightGBM only derives them from `seed` when they are unset, and
    # run_regime.py sets them the same way -- otherwise a "seed change" is a partial no-op
    pr["bagging_seed"] = sd + 1000
    pr["feature_fraction_seed"] = sd + 2000
    m = lgb.train(pr, lgb.Dataset(X, y, feature_name=names), num_boost_round=rounds,
                  callbacks=[lgb.log_evaluation(0)])
    v = m.predict(Xte)
    preds.append(v)
    print(f"    seed {sd}: test log-mean {v.mean():.4f} sd {v.std():.4f}", flush=True)

Lavg = np.mean(preds, axis=0)                      # log space -- see the module docstring
spread = float(np.mean(np.std(preds, axis=0)))
print(f"  per-user seed sd (mean) {spread:.4f}  <- what the averaging removes", flush=True)
pred = np.maximum(np.expm1(Lavg), 0.0)

out = ROOT / "subs" / f"{a.out}.csv"
pl.DataFrame({"user_id": p.users, "predict": pred}).write_csv(out)
print(f"  wrote {out}  ({len(pred):,} rows, mean {pred.mean():,.2f}, "
      f"zeros {100*(pred <= 0).mean():.2f}%)", flush=True)
