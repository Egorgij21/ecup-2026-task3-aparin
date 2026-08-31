#!/usr/bin/env python
"""
Stacked short-horizon predictions as features for the 30-day model (BACKLOG B4, ODMN §1.4).

Idea: a model trained to predict GMV over the NEXT 7 (or 14) days learns near-term intent
that a 30-day model averages away. Its prediction becomes a feature for the 30-day model.

LEAKAGE IS THE WHOLE DIFFICULTY. The 7-day target [A+1, A+7] is a subset of the 30-day
target [A+1, A+30], so a short model trained on anchor A and applied at anchor A hands over
part of the answer. Derivation of the required buffer. A short model trained at anchor A' has target
[A'+1, A'+h]; the 30-day target at A is [A+1, A+30]. They intersect iff
A'+1 <= A+30 AND A'+h >= A+1, i.e. A-h < A' <= A+29. So a source anchor is clean iff
A' <= A - h (or A' > A + 29). With h <= 14 the buffer is therefore 14 days, not 30 -- the
SHORT target is short, which is the whole point. We use 21 days (h + one 7-day grid step).

A first version of this script used 37 days, reasoning as if both targets were 30 days long.
That was wrong, and it made fold 0 infeasible: its training anchors span only 49 days, so no
source anchor could ever sit 37 days from every anchor of a block.

The condition is also ASYMMETRIC, which matters: a source may sit shortly BEFORE the target
anchor only if it is >= h days before, but it may sit AFTER with no restriction until +29
days. And the source pool is wider than the 30-day model's training anchors -- a short model
may be trained on anchors later than A_val - 30, as long as it stays clean w.r.t. the
validation target itself (A' <= A_val - h). Using the symmetric 21-day rule on the narrow
pool left fold 0's MIDDLE block with no sources at all.

Scheme, per fold:
  * source pool = the 7-day grid from the earliest training anchor up to A_val - h
  * split the fold's training anchors into 3 contiguous time blocks
  * for block i, train the short models on pool anchors clean w.r.t. EVERY anchor in block i,
    then predict for block i
  * for the validation anchor, train on pool anchors clean w.r.t. it
so every short-horizon feature value comes from a model that never saw an overlapping target.
run.py's look-ahead guard cannot detect this class of leak, so it is enforced by construction
and asserted at every use below.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from data import Panel                    # noqa: E402
from features import build                # noqa: E402
from metrics import rmsle, score_all      # noqa: E402

def clean(src: date, tgt: date, h: int) -> bool:
    """Is a short model trained at `src` (horizon h) clean for the 30-day target at `tgt`?"""
    return (src - tgt).days <= -h or (src - tgt).days > 29


def log(m):
    print(m, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--leaky", action="store_true",
                    help="NEGATIVE CONTROL: drop the buffer so short models see "
                         "overlapping targets. Never for a real experiment.")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    exp_id = cfg["exp_id"] + ("_LEAKY" if args.leaky else "")
    t0 = time.time()
    HS = list(cfg.get("stack_horizons", [7, 14]))
    log(f"\n=== {exp_id} : {cfg['change']} ===\n    stack horizons {HS}")

    spec = json.loads((ROOT / "data" / "fold_spec.json").read_text())
    folds = pl.read_parquet(ROOT / "data" / "folds.parquet")
    p = Panel()
    import lightgbm as lgb

    params = dict(cfg["lgb_params"]); params["seed"] = cfg["seed"]
    ROUNDS = int(cfg["fixed_rounds"])
    SHORT_ROUNDS = int(cfg.get("short_rounds", 150))

    def fit_short(train_anchors, h):
        X, y = [], []
        for a in train_anchors:
            ai = p.idx(a); k = p.active_in(ai - 29, ai)
            Xb, nm = build(p, ai, k, cfg["feature_blocks"])
            X.append(Xb); y.append(np.log1p(p.wsum("gmv", ai + 1, ai + h)[k]))
        return lgb.train(params, lgb.Dataset(np.concatenate(X), np.concatenate(y), feature_name=nm),
                         num_boost_round=SHORT_ROUNDS, callbacks=[lgb.log_evaluation(0)])

    oof, per_fold, per_naive = [], [], []
    for kf in range(len(spec["folds"])):
        fs = spec["folds"][kf]
        va = date.fromisoformat(fs["valid_anchor"]); vai = p.idx(va)
        tr = [date.fromisoformat(x) for x in fs["train_anchors"]]
        H = max(HS)
        pool, a_ = [], tr[0]
        while a_ <= va - timedelta(days=H):
            pool.append(a_); a_ += timedelta(days=7)
        blocks = np.array_split(np.array(tr), 3)

        short_feat = {}
        for bi, blk in enumerate(blocks):
            if args.leaky:
                src = list(tr)                       # deliberately overlapping
            else:
                src = [s_ for s_ in pool if all(clean(s_, c, H) for c in blk)]
                assert src, f"fold {kf} block {bi}: no clean source anchors"
                for a in blk:
                    for c in src:
                        assert clean(c, a, H), f"LEAK: source {c} overlaps the 30d target at {a}"
            ms = {h: fit_short(src, h) for h in HS}
            for a in blk:
                ai = p.idx(a); k = p.active_in(ai - 29, ai)
                Xb, _ = build(p, ai, k, cfg["feature_blocks"])
                short_feat[a] = np.column_stack([ms[h].predict(Xb) for h in HS])
            log(f"    fold {kf} block {bi}: {len(blk)} anchors fed by {len(src)} clean sources")

        src_v = list(tr) if args.leaky else [s_ for s_ in pool if clean(s_, va, H)]
        assert src_v, f"fold {kf}: no clean source anchors for validation"
        ms_v = {h: fit_short(src_v, h) for h in HS}

        Xtr, ytr = [], []
        for a in tr:
            ai = p.idx(a); k = p.active_in(ai - 29, ai)
            Xb, names = build(p, ai, k, cfg["feature_blocks"])
            Xtr.append(np.column_stack([Xb, short_feat[a]]))
            ytr.append(np.log1p(p.target(ai)[k]))
        Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr)
        names_s = names + [f"stack_h{h}" for h in HS]

        vk = p.active_in(vai - 29, vai)
        Xva, _ = build(p, vai, vk, cfg["feature_blocks"])
        Xva = np.column_stack([Xva, np.column_stack([ms_v[h].predict(Xva) for h in HS])])
        fv = folds.filter(pl.col("fold_id") == kf).sort("user_id")
        yva = fv["target"].to_numpy()

        model = lgb.train(params, lgb.Dataset(Xtr, ytr, feature_name=names_s),
                          num_boost_round=ROUNDS, callbacks=[lgb.log_evaluation(0)])
        pred = np.maximum(np.expm1(model.predict(Xva)), 0.0)
        naive = np.maximum(Xva[:, names_s.index("geo3")], 0.0)
        s, sn = rmsle(yva, pred), rmsle(yva, naive)
        per_fold.append(s); per_naive.append(sn)
        oof.append(pl.DataFrame({"fold_id": np.full(yva.size, kf, np.int8),
                                 "anchor_date": pl.Series("anchor_date", [va] * yva.size, dtype=pl.Date),
                                 "user_id": fv["user_id"].to_numpy(),
                                 "y_true": yva, "y_pred": pred, "y_naive": naive}))
        gi = [names_s.index(f"stack_h{h}") for h in HS]
        imp = model.feature_importance("gain")
        ctr = [float(np.corrcoef(Xtr[:, j], ytr)[0, 1]) for j in gi]
        cva = [float(np.corrcoef(Xva[:, j], np.log1p(yva))[0, 1]) for j in gi]
        log(f"      stack-vs-target corr  train {np.round(ctr, 4).tolist()}  "
            f"valid {np.round(cva, 4).tolist()}   (a big train>valid gap means leakage)")
        log(f"    fold {kf} {va}  rmsle={s:.5f}  naive={sn:.5f}   "
            f"stack-feature gain share = {100 * imp[gi].sum() / imp.sum():.1f}%")

    oof = pl.concat(oof); (ROOT / "oof").mkdir(exist_ok=True)
    oof.write_parquet(ROOT / "oof" / f"{exp_id}.parquet")
    pf, pn = np.array(per_fold), np.array(per_naive)
    agg = score_all(oof["y_true"].to_numpy(), oof["y_pred"].to_numpy())
    rt = (time.time() - t0) / 60
    log(f"\n  cv_mean = {pf.mean():.5f} +/- {pf.std():.5f}   folds {np.round(pf, 5).tolist()}")
    log(f"  delta vs naive = {pf.mean() - pn.mean():+.5f}   runtime {rt:.1f} min")

    row = {"exp_id": exp_id, "parent_id": cfg["parent_id"],
           "date": datetime.now().isoformat(timespec="seconds"), "approach": cfg["approach"],
           "change": cfg["change"], "tier": "confirm", "n_features": len(names_s),
           "cv_mean": round(float(pf.mean()), 5), "cv_std": round(float(pf.std()), 5),
           "folds": json.dumps([round(float(x), 5) for x in pf]),
           "delta": round(float(pf.mean() - pn.mean()), 5), "significant": "",
           "lb": "", "runtime_min": round(rt, 1), "seed": cfg["seed"], "config": args.config,
           "verdict": "", "gini_pred": round(agg["gini_pred"], 4),
           "total_rel_err": round(agg["total_rel_err"], 4),
           "best_iters": json.dumps([ROUNDS] * len(pf)), "notes": cfg.get("notes", "")}
    (ROOT / "runs").mkdir(exist_ok=True)
    (ROOT / "runs" / f"{exp_id}.json").write_text(json.dumps(row, indent=2))
    import subprocess
    subprocess.run([sys.executable, str(ROOT / "src" / "collect.py")], check=False)
    log(f"  wrote runs/{exp_id}.json")


if __name__ == "__main__":
    main()
