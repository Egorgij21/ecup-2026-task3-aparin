#!/usr/bin/env python
"""
Optuna hyperparameter search for the LightGBM member, on the FROZEN folds.

This is the one large lever never pulled in ~90 experiments. `lr=0.05, num_leaves=63,
min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8, lambda_l2=1.0` have been
fixed since e0001 as deliberate "fast honest defaults" -- CLAUDE.md §5 says do not tune a
losing feature set, and the feature set only settled recently.

Why it matters for the BLEND specifically, given §1c's corrected rule: decorrelation is only
worth something AT COMPARABLE QUALITY. A weaker member contributes its own error, not a
different view of the truth. So the way to make the tabular family pull its weight in the
blend is to make it BETTER, not to make it more exotic.

Two-tier, per CLAUDE.md §4.2:
  * SCREEN  -- `--folds 3 --subsample 0.5`. Cheap and noisy, kills dead regions fast. Screen
               results never enter a decision; they only rank.
  * CONFIRM -- `--confirm '<json>'` re-runs a specific parameter set on all 5 folds, full
               data, through the same path, so the number is comparable to 1.76551.

THE ROUND-COUNT TRAP. Rounds are fixed at 178 because the early-stopping holdout was
contaminated and removing it was worth -0.00349 (e0017/e0020); `diag_es.json` shows the loss
curve is flat from 80 to 740. But 178 was measured AT lr=0.05. Tuning the learning rate while
holding rounds fixed silently compares an lr=0.01 model at 1/5 of its needed capacity against
an lr=0.15 model at 3x. So rounds are scaled as `178 * 0.05/lr`, clipped to [120, 1200] --
the flat region means the exact value inside it does not matter, but the SCALE does.

Parallel workers share one JournalStorage file (safe on GPFS, unlike SQLite locking).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from data import Panel                      # noqa: E402
from features import build                  # noqa: E402
from metrics import rmsle                   # noqa: E402

BASE_LR, BASE_ROUNDS = 0.05, 178


def log(m):
    print(m, flush=True)


def select(cfg, names, mats):
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


class FoldData:
    """Build every fold ONCE, then reuse across all trials. Feature building is the binding
    cost (the cache makes it ~1 s/anchor, but assembling 25 anchors still costs minutes);
    a trial must be a fit, nothing else."""

    def __init__(self, cfg, spec, fold_ids, subsample, seed=0):
        self.folds = []
        p = Panel()
        rng = np.random.default_rng(seed)
        for k in fold_ids:
            fs = spec["folds"][k]
            va = date.fromisoformat(fs["valid_anchor"])
            tr = [date.fromisoformat(x) for x in fs["train_anchors"]]
            Xs, ys = [], []
            for a in tr:
                ai = p.idx(a); keep = p.active_in(ai - 29, ai)
                Xb, names = build(p, ai, keep, cfg["feature_blocks"])
                Xs.append(Xb); ys.append(np.log1p(p.target(ai)[keep]))
            X = np.concatenate(Xs); y = np.concatenate(ys); del Xs, ys
            vai = p.idx(va); vk = p.active_in(vai - 29, vai)
            Xva, nv = build(p, vai, vk, cfg["feature_blocks"])
            assert nv == names
            names, (X, Xva) = select(cfg, names, (X, Xva))
            if subsample < 1.0:
                # subsample TRAINING rows only -- the validation set stays whole, so the
                # score is on the same rows the confirm run will use
                idx = rng.choice(X.shape[0], int(subsample * X.shape[0]), replace=False)
                X, y = X[idx], y[idx]
            yva = p.target(vai)[vk]
            self.folds.append((X, y, Xva, yva, names))
            log(f"    fold {k}: {X.shape[0]:,} train x {X.shape[1]} feat, "
                f"{Xva.shape[0]:,} valid")
        self.names = names


def scale_rounds(lr: float) -> int:
    return int(np.clip(BASE_ROUNDS * BASE_LR / lr, 120, 1200))


def evaluate(params: dict, fd: FoldData, seed: int = 0, model: str = "lgb",
             return_preds: bool = False):
    rounds = scale_rounds(params["learning_rate"])
    per, preds = [], []
    for (X, y, Xva, yva, names) in fd.folds:
        pr = dict(params)
        if model == "lgb":
            import lightgbm as lgb
            pr.update(objective="regression", metric="rmse", verbosity=-1,
                      num_threads=int(pr.pop("num_threads", 16)), seed=seed)
            m = lgb.train(pr, lgb.Dataset(X, y, feature_name=names),
                          num_boost_round=rounds, callbacks=[lgb.log_evaluation(0)])
            p = m.predict(Xva)
        else:
            from catboost import CatBoostRegressor
            pr.pop("num_threads", None)
            # GPU: border_count caps at 254, and Bayesian bagging is the GPU-supported
            # bootstrap. Depth >10 on GPU with SymmetricTree gets very slow.
            m = CatBoostRegressor(iterations=rounds, loss_function="RMSE",
                                  task_type="GPU", devices="0", random_seed=seed,
                                  verbose=0, allow_writing_files=False, **pr)
            m.fit(X, y)
            p = m.predict(Xva)
        pr_lin = np.maximum(np.expm1(p), 0.0)
        per.append(float(rmsle(yva, pr_lin)))
        preds.append(pr_lin)
    if return_preds:
        return float(np.mean(per)), per, preds
    return float(np.mean(per)), per


# v1 was truncated. Checking the top-8 trials' position within each range afterwards:
#   learning_rate     median  2% of range -> AT THE FLOOR
#   min_data_in_leaf  median 95%          -> AT THE CEILING  (2000 is only 0.03% of 6M rows)
#   feature_fraction  median 14%          -> near the floor
#   bagging_freq      median  0%          -> natural boundary: 1 = bag every iteration, and
#                                            0 DISABLES bagging, so there is nowhere to extend
#   lambda_l2         median 19%          -> soft: 1e-4 is already indistinguishable from 0
# Only the first three are real truncations. v2 widens those and adds max_bin=1023 (the top-8
# chose 127 or 511 but never 255, and 511 was the ceiling).
SPACES = {
    "v1": dict(lr=(0.015, 0.12), leaves=(31, 511), mdl=(20, 2000), ff=(0.3, 1.0),
               bins=[127, 255, 511]),
    "v2": dict(lr=(0.003, 0.06), leaves=(31, 511), mdl=(200, 40000), ff=(0.05, 0.75),
               bins=[127, 255, 511, 1023]),
    # v3: v2's optimum sat at lr~0.010, mdl~1700 (both interior, good) but feature_fraction
    # ~0.20 against a 0.05 floor -- still drifting. Narrow around the optimum and let ff go
    # lower, since with 665 features ff=0.1 is still ~66 features per split.
    "v3": dict(lr=(0.004, 0.03), leaves=(64, 400), mdl=(400, 12000), ff=(0.03, 0.45),
               bins=[255, 511, 1023]),
}


def suggest_cat(t) -> dict:
    """CatBoost on GPU. Deliberately a DIFFERENT inductive bias from LightGBM -- symmetric
    (oblivious) trees, ordered boosting, Bayesian bagging -- which is why it correlated at
    0.974 with the gbdt family rather than the ~0.998 of another LightGBM. §1c showed that
    decorrelation was not enough because the untuned CatBoost was 0.032 WEAKER. This gives it
    the fair shot it never had."""
    gp = t.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"])
    p = {
        "learning_rate": t.suggest_float("learning_rate", 0.006, 0.10, log=True),
        "depth": t.suggest_int("depth", 4, 10),
        "l2_leaf_reg": t.suggest_float("l2_leaf_reg", 0.5, 100.0, log=True),
        "random_strength": t.suggest_float("random_strength", 1e-3, 10.0, log=True),
        "bagging_temperature": t.suggest_float("bagging_temperature", 0.0, 3.0),
        "border_count": t.suggest_categorical("border_count", [64, 128, 254]),
        "min_data_in_leaf": t.suggest_int("min_data_in_leaf", 50, 20000, log=True),
        "grow_policy": gp,
    }
    if gp == "Lossguide":
        p["max_leaves"] = t.suggest_int("max_leaves", 31, 512, log=True)
    return p


def suggest(t, space="v1") -> dict:
    S = SPACES[space]
    return {
        "learning_rate": t.suggest_float("learning_rate", *S["lr"], log=True),
        "num_leaves": t.suggest_int("num_leaves", *S["leaves"], log=True),
        "min_data_in_leaf": t.suggest_int("min_data_in_leaf", *S["mdl"], log=True),
        "feature_fraction": t.suggest_float("feature_fraction", *S["ff"]),
        "bagging_fraction": t.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": t.suggest_int("bagging_freq", 1, 7),
        "lambda_l1": t.suggest_float("lambda_l1", 1e-4, 20.0, log=True),
        "lambda_l2": t.suggest_float("lambda_l2", 1e-4, 50.0, log=True),
        "min_gain_to_split": t.suggest_float("min_gain_to_split", 1e-4, 1.0, log=True),
        "max_bin": t.suggest_categorical("max_bin", S["bins"]),
        "num_threads": 16,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/e0049_nomoment.yaml")
    ap.add_argument("--study", default="lgb_v1")
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--folds", type=int, default=3, help="screen on the LAST n folds")
    ap.add_argument("--subsample", type=float, default=0.5)
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--space", default="v1", choices=["v1", "v2", "v3"])
    ap.add_argument("--model", default="lgb", choices=["lgb", "cat"])
    ap.add_argument("--save-oof", default="",
                    help="exp_id: write oof/<id>.parquet for the BEST confirmed config, so it "
                         "can enter a blend weight fit. Without this a tuned model cannot be "
                         "blended, only scored.")
    ap.add_argument("--confirm", default="", help="JSON params: re-run on all 5 folds, full data")
    ap.add_argument("--confirm-top", type=int, default=0,
                    help="confirm the best N trials of the study (screen noise means the "
                         "top screen trial is not reliably the best true config)")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text())
    spec = json.loads((ROOT / "data" / "fold_spec.json").read_text())
    if args.cache:
        import features as _f
        _f.enable_cache(True)
        log(f"    feature cache {'ON' if _f.CACHE_ENABLED else 'REFUSED'}")

    if args.confirm or args.confirm_top:
        BASE = [1.77312, 1.79400, 1.77715, 1.75088, 1.73239]      # e0049, the incumbent
        cand = []
        if args.confirm:
            cand = [("cli", json.loads(args.confirm))]
        else:
            import optuna
            from optuna.storages import JournalStorage
            from optuna.storages.journal import JournalFileBackend
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            sd = ROOT / "reports" / "optuna"
            st = optuna.load_study(study_name=args.study,
                                   storage=JournalStorage(JournalFileBackend(str(sd / f"{args.study}.log"))))
            done = [t for t in st.trials if t.value is not None]
            done.sort(key=lambda t: t.value)
            log(f"  study has {len(done)} finished trials; confirming the top {args.confirm_top}")
            for t in done[: args.confirm_top]:
                p = dict(t.params); p["num_threads"] = 16
                cand.append((f"trial{t.number}(screen {t.value:.5f})", p))

        fd = FoldData(cfg, spec, list(range(5)), 1.0)
        results, best_preds, best_cv = [], None, float("inf")
        for tag, params in cand:
            log(f"\n=== CONFIRM {tag} on all 5 folds, full data ===\n    {params}")
            t0 = time.time()
            mean, per, preds = evaluate(params, fd, model=args.model, return_preds=True)
            if mean < best_cv:
                best_cv, best_preds, best_tag, best_params = mean, preds, tag, params
            wins = sum(a < b for a, b in zip(per, BASE))
            log(f"  rounds used = {scale_rounds(params['learning_rate'])}")
            log(f"  CV = {mean:.5f}   folds {[round(x,5) for x in per]}")
            log(f"  vs e0049 1.76551  ->  delta {mean-1.76551:+.5f}   wins {wins}/5   "
                f"[{(time.time()-t0)/60:.1f}m]")
            # §3 of EXPERIMENTS.md: project transfer from the LAST fold, not the mean
            log(f"  last fold (most test-like): {per[-1]:.5f} vs {BASE[-1]:.5f} "
                f"-> {per[-1]-BASE[-1]:+.5f}")
            results.append({"tag": tag, "params": params, "cv_mean": mean, "folds": per,
                            "delta": mean - 1.76551, "wins": wins,
                            "last_fold_delta": per[-1] - BASE[-1],
                            "rounds": scale_rounds(params["learning_rate"])})
        results.sort(key=lambda r: r["cv_mean"])
        log("\n=== SUMMARY (confirm tier, all 5 folds, full data) ===")
        log(f"  {'tag':34s} {'CV':>9s} {'delta':>9s} {'wins':>5s} {'lastfold':>9s}")
        log(f"  {'e0049 (incumbent)':34s} {1.76551:>9.5f} {0.0:>+9.5f} {'-':>5s} {0.0:>+9.5f}")
        for r in results:
            log(f"  {r['tag']:34s} {r['cv_mean']:>9.5f} {r['delta']:>+9.5f} "
                f"{r['wins']:>5d} {r['last_fold_delta']:>+9.5f}")
        if args.save_oof and best_preds is not None:
            import pandas as pd
            ref = pd.read_parquet(ROOT / "oof" / "e0049.parquet").sort_values(
                ["fold_id", "user_id"]).reset_index(drop=True)
            rows = []
            for k, pr in enumerate(best_preds):
                m = ref[ref.fold_id == k]
                assert len(m) == len(pr), f"fold {k}: {len(m)} ref rows vs {len(pr)} preds"
                rows.append(pd.DataFrame({"fold_id": m.fold_id.to_numpy(),
                                          "user_id": m.user_id.to_numpy(),
                                          "y_true": m.y_true.to_numpy(),
                                          "y_pred": pr}))
            o = pd.concat(rows, ignore_index=True)
            o.to_parquet(ROOT / "oof" / f"{args.save_oof}.parquet", index=False)
            log(f"\n  wrote oof/{args.save_oof}.parquet from {best_tag} "
                f"({len(o):,} rows) -- keys taken from e0049 so it drops into blend_eval")
            (ROOT / "configs" / f"{args.save_oof}_params.json").write_text(
                json.dumps({"model": args.model, "params": best_params,
                            "rounds": scale_rounds(best_params["learning_rate"]),
                            "cv_mean": best_cv}, indent=2))
        out = ROOT / "reports" / f"tune_confirm_{args.study}.json"
        out.write_text(json.dumps(results, indent=2))
        log(f"\n  wrote {out}")
        return

    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    fold_ids = list(range(5))[-args.folds:]
    log(f"\n=== SCREEN: folds {fold_ids}, subsample {args.subsample}, {args.trials} trials ===")
    fd = FoldData(cfg, spec, fold_ids, args.subsample)

    sd = ROOT / "reports" / "optuna"; sd.mkdir(parents=True, exist_ok=True)
    storage = JournalStorage(JournalFileBackend(str(sd / f"{args.study}.log")))
    study = optuna.create_study(study_name=args.study, storage=storage,
                                direction="minimize", load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=None, n_startup_trials=10))

    # the incumbent, so the study always contains the thing we must beat
    if args.model == 'lgb' and len(study.trials) == 0 and args.space == "v2":
        # start from v1's optimum, which sat against three of its walls
        study.enqueue_trial({"learning_rate": 0.0151, "num_leaves": 133,
                             "min_data_in_leaf": 1614, "feature_fraction": 0.3827,
                             "bagging_fraction": 0.84, "bagging_freq": 1,
                             "lambda_l1": 0.1637, "lambda_l2": 0.000874,
                             "min_gain_to_split": 0.004872, "max_bin": 511})
    if args.model == 'lgb' and len(study.trials) == 0:
        study.enqueue_trial({"learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 200,
                             "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
                             "lambda_l1": 1e-4, "lambda_l2": 1.0, "min_gain_to_split": 1e-4,
                             "max_bin": 255})

    def objective(trial):
        p = suggest_cat(trial) if args.model == 'cat' else suggest(trial, args.space)
        t0 = time.time()
        mean, per = evaluate(p, fd, model=args.model)
        trial.set_user_attr("folds", per)
        trial.set_user_attr("rounds", scale_rounds(p["learning_rate"]))
        extra = (f"leaves={p['num_leaves']} mdl={p['min_data_in_leaf']} "
                 f"ff={p['feature_fraction']:.2f}" if args.model == "lgb" else
                 f"depth={p['depth']} l2={p['l2_leaf_reg']:.2f} mdl={p['min_data_in_leaf']} "
                 f"{p['grow_policy'][:4]}")
        log(f"  trial {trial.number:>3}  {mean:.5f}  lr={p['learning_rate']:.4f} {extra} "
            f"[{(time.time()-t0)/60:.1f}m]")
        return mean

    study.optimize(objective, n_trials=args.trials, gc_after_trial=True)
    log(f"\n  best screen value {study.best_value:.5f}")
    log(f"  best params {json.dumps(study.best_params)}")
    log("\n  SCREEN ONLY -- re-run the winner with --confirm before believing it (§4.2).")


if __name__ == "__main__":
    main()
