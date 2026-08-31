#!/usr/bin/env python
"""
Diagnostic: is the early-stopping anchor picking the right iteration?

e0017 fixed a contaminated ES set and gained -0.00384 overall, but fold 4 -- the most
test-like anchor -- got WORSE by +0.00407 while folds 0-3 improved. Either
  (a) ES stops at roughly the validation optimum and fold 4's regression is something else, or
  (b) ES is noisy and stops well away from the optimum, and fold 4 is where it bit.

This traces validation RMSLE against boosting iteration for every fold and marks where ES
stopped. It is ANALYSIS ONLY -- the validation curve is never used to pick a model, and
nothing here writes to experiments.csv. Reporting the curve tells us how much is on the
table and whether the ES signal needs strengthening (e.g. several ES anchors averaged).
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data import Panel                # noqa: E402
from features import build            # noqa: E402
from metrics import rmsle             # noqa: E402

GRID = list(range(20, 1001, 20)) + list(range(1100, 3001, 100))


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "e0017_esgap.yaml").read_text())
    spec = json.loads((ROOT / "data" / "fold_spec.json").read_text())
    folds = pl.read_parquet(ROOT / "data" / "folds.parquet")
    p = Panel()
    import lightgbm as lgb

    out = {}
    for k in range(len(spec["folds"])):
        fs = spec["folds"][k]
        va = date.fromisoformat(fs["valid_anchor"]); vai = p.idx(va)
        tr = [date.fromisoformat(x) for x in fs["train_anchors"]]
        es_anchor = tr[-1]
        gap = int(cfg.get("es_gap_days", 0))
        fit = [a for a in tr[:-1] if (es_anchor - a).days >= gap]

        Xtr, ytr = [], []
        for a in fit:
            ai = p.idx(a); keep = p.active_in(ai - 29, ai)
            X, names = build(p, ai, keep, cfg["feature_blocks"])
            Xtr.append(X); ytr.append(np.log1p(p.target(ai)[keep]))
        Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr)

        ei = p.idx(es_anchor); ek = p.active_in(ei - 29, ei)
        Xes, _ = build(p, ei, ek, cfg["feature_blocks"])
        yes = np.log1p(p.target(ei)[ek])

        vk = p.active_in(vai - 29, vai)
        Xva, _ = build(p, vai, vk, cfg["feature_blocks"])
        yva = folds.filter(pl.col("fold_id") == k).sort("user_id")["target"].to_numpy()

        params = dict(cfg["lgb_params"]); params["seed"] = cfg["seed"]
        model = lgb.train(params, lgb.Dataset(Xtr, ytr, feature_name=names),
                          num_boost_round=3000,
                          valid_sets=[lgb.Dataset(Xes, yes, feature_name=names)],
                          callbacks=[lgb.early_stopping(100, verbose=False),
                                     lgb.log_evaluation(0)])
        es_stop = model.best_iteration

        # retrain WITHOUT early stopping so the full curve is available
        full = lgb.train(params, lgb.Dataset(Xtr, ytr, feature_name=names),
                         num_boost_round=3000, callbacks=[lgb.log_evaluation(0)])
        curve = []
        for it in GRID:
            pr = np.maximum(np.expm1(full.predict(Xva, num_iteration=it)), 0.0)
            curve.append((it, rmsle(yva, pr)))
        pr_es = np.maximum(np.expm1(full.predict(Xva, num_iteration=es_stop)), 0.0)
        s_es = rmsle(yva, pr_es)
        best_it, best_s = min(curve, key=lambda z: z[1])
        out[k] = {"anchor": str(va), "es_stop": es_stop, "rmsle_at_es": s_es,
                  "best_it": best_it, "best_rmsle": best_s, "regret": s_es - best_s,
                  "curve": curve}
        print(f"\n  fold {k} ({va}):  ES stopped at {es_stop:>5d} -> val {s_es:.5f}")
        print(f"      validation optimum at {best_it:>5d} -> val {best_s:.5f}   "
              f"REGRET = {s_es - best_s:+.5f}")
        near = [c for c in curve if c[1] <= best_s + 0.001]
        print(f"      iterations within 0.001 of optimum: {near[0][0]} .. {near[-1][0]}")
        step = max(1, len(curve) // 18)
        print("      curve: " + "  ".join(f"{it}:{s:.4f}" for it, s in curve[::step]))

    tot_es = np.mean([v["rmsle_at_es"] for v in out.values()])
    tot_best = np.mean([v["best_rmsle"] for v in out.values()])
    print(f"\n  mean RMSLE at the ES-chosen iteration : {tot_es:.5f}")
    print(f"  mean RMSLE at the per-fold optimum    : {tot_best:.5f}  (unreachable oracle)")
    print(f"  total regret from ES noise            : {tot_es - tot_best:+.5f}")
    med = int(np.median([v["es_stop"] for v in out.values()]))
    fixed = np.mean([dict(v["curve"])[min(GRID, key=lambda g: abs(g - med))]
                     for v in out.values()])
    print(f"  RMSLE if every fold used the MEDIAN stop ({med}): {fixed:.5f}")
    (ROOT / "reports" / "eda").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports" / "eda" / "diag_es.json").write_text(json.dumps(out, indent=2, default=str))
    print("\n  wrote reports/eda/diag_es.json")


if __name__ == "__main__":
    main()
