#!/usr/bin/env python
"""
Test-set predictions for a model tuned by src/tune.py, LightGBM or CatBoost.

`src/predict.py` only knows LightGBM and takes its parameters from a YAML config. The tuned
models are described by the `*_params.json` that tune.py's confirm step writes, and one of
them is CatBoost -- so neither the parameter source nor the learner matches. This reads that
json directly, so the model that ships is byte-for-byte the one that was confirmed on the
frozen folds rather than a hand-copied approximation of it.

Training set and column selection are taken from predict.py so the two paths cannot drift:
all clean anchors on the frozen 7-day grid, features built at the test anchor, the config's
whitelist/exclude applied to both.
"""
from __future__ import annotations
import argparse, json, sys
from datetime import date, timedelta
from pathlib import Path
import numpy as np, polars as pl, yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from data import Panel                    # noqa: E402
from features import build                # noqa: E402

GUARD_START, MIN_HISTORY, TRAIN_STRIDE = date(2025, 11, 16), 90, 7

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True, help="for feature_blocks + whitelist/exclude")
ap.add_argument("--params", required=True, help="the *_params.json from tune.py")
ap.add_argument("--out", required=True, help="exp_id -> subs/<id>.csv")
ap.add_argument("--cache", action="store_true")
a = ap.parse_args()

cfg = yaml.safe_load((ROOT / a.config).read_text())
spec = json.loads(Path(a.params).read_text() if Path(a.params).exists()
                  else (ROOT / a.params).read_text())
kind, params, rounds = spec["model"], dict(spec["params"]), int(spec["rounds"])
print(f"  {kind}, {rounds} rounds, confirmed CV {spec.get('cv_mean')}", flush=True)

if a.cache:
    import features as _f
    _f.enable_cache(True)
p = Panel()

def select(names, mats):
    if cfg.get("feature_exclude_patterns"):
        import re as _re
        pats = [_re.compile(x) for x in cfg["feature_exclude_patterns"]]
        s = [i for i, n in enumerate(names) if not any(q.search(n) for q in pats)]
        mats = [m[:, s] for m in mats]; names = [names[i] for i in s]
    if cfg.get("feature_whitelist"):
        WL = set(json.loads((ROOT / cfg["feature_whitelist"]).read_text()))
        s = [i for i, n in enumerate(names) if n in WL]
        mats = [m[:, s] for m in mats]; names = [names[i] for i in s]
    return names, mats

test_anchor = p.dmax; tai = p.idx(test_anchor)
latest, earliest = GUARD_START - timedelta(days=31), p.dmin + timedelta(days=MIN_HISTORY - 1)
anchors, d_ = [], latest
while d_ >= earliest:
    anchors.append(d_); d_ -= timedelta(days=TRAIN_STRIDE)
anchors = sorted(anchors)
print(f"  {len(anchors)} training anchors {anchors[0]} .. {anchors[-1]}", flush=True)

Xs, ys, names = [], [], None
for an in anchors:
    ai = p.idx(an); keep = p.active_in(ai - 29, ai)
    Xb, names = build(p, ai, keep, cfg["feature_blocks"])
    Xs.append(Xb); ys.append(np.log1p(p.target(ai)[keep]))
X = np.concatenate(Xs); y = np.concatenate(ys); del Xs, ys
keep_test = p.active_in(tai - 29, tai)
assert keep_test.all(), "some test users fail the population rule"
Xte, nt = build(p, tai, keep_test, cfg["feature_blocks"])
assert nt == names
names, (X, Xte) = select(names, (X, Xte))
print(f"  train {X.shape[0]:,} x {X.shape[1]}   test {Xte.shape[0]:,}", flush=True)

if kind == "lgb":
    import lightgbm as lgb
    pr = dict(objective="regression", metric="rmse", verbosity=-1, seed=cfg.get("seed", 0))
    pr.update(params)
    m = lgb.train(pr, lgb.Dataset(X, y, feature_name=names), num_boost_round=rounds,
                  callbacks=[lgb.log_evaluation(0)])
    pred = m.predict(Xte)
else:
    from catboost import CatBoostRegressor
    params.pop("num_threads", None)
    m = CatBoostRegressor(iterations=rounds, loss_function="RMSE", task_type="GPU",
                          devices="0", random_seed=cfg.get("seed", 0), verbose=0,
                          allow_writing_files=False, **params)
    m.fit(X, y)
    pred = m.predict(Xte)

pred = np.maximum(np.expm1(pred), 0.0)
out = ROOT / "subs" / f"{a.out}.csv"
pl.DataFrame({"user_id": p.users, "predict": pred}).write_csv(out)
print(f"  wrote {out}  ({len(pred):,} rows, mean {pred.mean():,.2f}, "
      f"zeros {100*(pred<=0).mean():.2f}%)", flush=True)
