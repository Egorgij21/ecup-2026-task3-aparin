#!/usr/bin/env python
"""
NULL-IMPORTANCE feature selection (CLAUDE.md §5.3), computed per fold on training data only.

Why not gain importance: §5.3 is explicit that raw gain is biased toward high-cardinality
continuous features. Our feature set is now ~830 columns, most of them auto-generated
continuous statistics across 7 windows -- exactly the case where gain lies. A feature that
looks important because it offers many split points is indistinguishable, by gain alone,
from one that carries signal.

Null importance fixes that by asking a different question: how much gain does this feature
earn when the target is SHUFFLED and there is nothing to learn? A real feature beats its own
null distribution; a feature that merely offers convenient splits does not.

    score(f) = log( (1 + actual_gain(f)) / (1 + percentile_75(null_gains(f))) )

Protocol points that matter:
  * importances are computed per fold on that fold's TRAINING anchors only -- never on the
    validation fold, which would select features using the data we score on;
  * features are built ONCE per fold and reused across the real fit and all null fits, so the
    cost is one feature build plus N+1 cheap fits rather than N+1 full rebuilds;
  * the output is a RANKING plus candidate cut-offs. §5.4 is emphatic that filtering must
    then be re-validated and can lose, so nothing here is applied automatically.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from data import Panel                    # noqa: E402
from features import build                # noqa: E402

N_NULL = 5
OUT = ROOT / "reports"


def log(m):
    print(m, flush=True)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/e0039_sbc.yaml")
    ap.add_argument("--tag", default="", help="suffix for the whitelist filenames")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text())
    print(f"  scoring the feature set from {args.config}")
    spec = json.loads((ROOT / "data" / "fold_spec.json").read_text())
    p = Panel()
    import lightgbm as lgb

    params = dict(cfg["lgb_params"]); params["seed"] = cfg["seed"]
    ROUNDS = int(cfg["fixed_rounds"])
    t0 = time.time()

    actual, nulls, names = [], [], None
    for k in range(len(spec["folds"])):
        fs = spec["folds"][k]
        va = date.fromisoformat(fs["valid_anchor"])
        tr = [date.fromisoformat(x) for x in fs["train_anchors"]]
        X, y = [], []
        for a in tr:
            ai = p.idx(a); keep = p.active_in(ai - 29, ai)
            Xb, names = build(p, ai, keep, cfg["feature_blocks"])
            X.append(Xb); y.append(np.log1p(p.target(ai)[keep]))
        X = np.concatenate(X); y = np.concatenate(y)
        log(f"  fold {k} ({va}): {X.shape[0]:,} rows x {X.shape[1]} features  "
            f"[t+{(time.time()-t0)/60:.1f}m]")

        m = lgb.train(params, lgb.Dataset(X, y, feature_name=names),
                      num_boost_round=ROUNDS, callbacks=[lgb.log_evaluation(0)])
        actual.append(m.feature_importance("gain"))

        rng = np.random.default_rng(100 + k)
        for j in range(N_NULL):
            ys = rng.permutation(y)          # target shuffled -> nothing to learn
            mn = lgb.train(params, lgb.Dataset(X, ys, feature_name=names),
                           num_boost_round=ROUNDS, callbacks=[lgb.log_evaluation(0)])
            nulls.append(mn.feature_importance("gain"))
        log(f"    fitted 1 real + {N_NULL} null models  [t+{(time.time()-t0)/60:.1f}m]")
        del X, y

    A = np.mean(actual, axis=0)
    N = np.percentile(np.array(nulls), 75, axis=0)
    score = np.log((1.0 + A) / (1.0 + N))
    order = np.argsort(-score)

    log(f"\n  {'rank':>5s} {'feature':44s} {'actual gain':>13s} {'null p75':>12s} {'score':>8s}")
    for i in order[:30]:
        log(f"  {list(order).index(i)+1:>5d} {names[i]:44s} {A[i]:>13,.0f} {N[i]:>12,.0f} {score[i]:>8.3f}")
    log("\n  ... worst 10:")
    for i in order[-10:]:
        log(f"  {list(order).index(i)+1:>5d} {names[i]:44s} {A[i]:>13,.0f} {N[i]:>12,.0f} {score[i]:>8.3f}")

    n_pos = int((score > 0).sum())
    log(f"\n  features beating their own null (score > 0): {n_pos} of {len(names)}")
    log(f"  features scoring <= 0 (no better than noise): {len(names) - n_pos}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"importance{args.tag}.json").write_text(json.dumps(
        {"names": names, "actual": A.tolist(), "null_p75": N.tolist(),
         "score": score.tolist()}, indent=2))

    # candidate cut-offs -- each becomes its own logged experiment (§5.4)
    cuts = {}
    for tag, sel in [
        ("top100", [names[i] for i in order[:100]]),
        ("top200", [names[i] for i in order[:200]]),
        ("top400", [names[i] for i in order[:400]]),
        ("positive", [names[i] for i in order if score[i] > 0]),
        ("positive_strict", [names[i] for i in order if score[i] > 0.10]),
    ]:
        cuts[tag] = sel
        (ROOT / "configs" / f"whitelist{args.tag}_{tag}.json").write_text(json.dumps(sel, indent=1))
        log(f"  wrote configs/whitelist{args.tag}_{tag}.json  ({len(sel)} features)")
    log(f"\n  total runtime {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
