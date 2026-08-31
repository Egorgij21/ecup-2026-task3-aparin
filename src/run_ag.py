#!/usr/bin/env python
"""
AutoGluon TabularPredictor on the FROZEN folds (README.md step 5).

The whole point is to get a number comparable to 1.76547. That rules out letting AutoGluon
split the data itself, for a reason specific to this dataset:

    THE SAME USER APPEARS AT ~20 TRAINING ANCHORS.

AutoGluon's default bagging is a random KFold over rows. On this panel that puts user 12345
at anchor 2025-06-03 in the training fold and the same user at 2025-06-10 in the validation
fold -- rows whose 365-day features differ by 7/365 and whose targets overlap 23 of 30 days.
Its internal score would be badly optimistic and, worse, the ensemble WEIGHTS would be tuned
against leaked validation. So we drive the folds ourselves: one predictor per frozen fold,
scored with our own metric on our own validation anchor.

Validation data for AutoGluon itself is the other trap. It structurally needs a `tuning_data`
for early stopping and ensemble weighting, and there are two wrong choices:
  * the fold's OWN validation anchor -> ensemble weights fitted on the rows we then score.
    Straightforwardly optimistic; this is the mistake that makes AutoML look magic.
  * the last training anchor with no gap -> it sits 7 days from its neighbours with 365-day
    lookbacks, i.e. a near-duplicate of training rows. That is exactly e0016/e0017: the
    validation curve falls monotonically, early stopping never fires. Cost us most of a day.
So `tuning_data` is the last training anchor (= A_valid - 30, clean of the scoring fold) and
the fit anchors are embargoed 30 days back from it -- the same `es_gap_days: 30` geometry
that was already validated here.

That embargo costs ~5 of ~25 anchors, and e0020 showed training-set size is worth more than
a holdout (-0.00349). So AutoGluon runs with a real handicap against our LightGBM. `--refit-
full` hands the anchors back by retraining the chosen models on train+tuning; the ensemble
weights still come from the clean holdout.

Target is log1p(gmv) with eval_metric=root_mean_squared_error -- RMSE on log1p IS RMSLE, so
no custom scorer is needed and AutoGluon optimises the competition metric directly.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from data import Panel                      # noqa: E402
from features import build                  # noqa: E402
from metrics import rmsle                   # noqa: E402

GUARD_START = date(2025, 11, 16)            # mirrors predict.py
MIN_HISTORY = 90
TRAIN_STRIDE = 7
EMBARGO_DAYS = 30


def log(m):
    print(m, flush=True)


def select_columns(cfg, names, mats):
    """Same column selection as run.py -- a config's whitelist IS its identity."""
    if cfg.get("feature_exclude_patterns"):
        import re as _re
        pats = [_re.compile(x) for x in cfg["feature_exclude_patterns"]]
        sel = [i for i, n in enumerate(names) if not any(q.search(n) for q in pats)]
        log(f"    exclude patterns: dropping {len(names) - len(sel)} of {len(names)}")
        mats = [m[:, sel] for m in mats]; names = [names[i] for i in sel]
    if cfg.get("feature_whitelist"):
        WL = set(json.loads((ROOT / cfg["feature_whitelist"]).read_text()))
        sel = [i for i, n in enumerate(names) if n in WL]
        log(f"    whitelist: keeping {len(sel)} of {len(names)} features")
        mats = [m[:, sel] for m in mats]; names = [names[i] for i in sel]
    return names, mats


def build_at(p, cfg, anchors):
    """Stack feature matrices at a list of anchors. Returns (X, raw_y, names)."""
    Xs, ys = [], []
    names = None
    for a in anchors:
        ai = p.idx(a)
        keep = p.active_in(ai - 29, ai)
        X, names = build(p, ai, keep, cfg["feature_blocks"])
        Xs.append(X); ys.append(p.target(ai)[keep])
    return np.concatenate(Xs), np.concatenate(ys), names


def to_frame(X, names, y_log=None):
    df = pd.DataFrame(X, columns=names, copy=False)
    if y_log is not None:
        df["__y"] = y_log
    return df


def fit_predict(train_df, tune_df, test_df, args, path, extra_df=None):
    """Fit on the embargoed anchors, then refit the winner on EVERY anchor.

    The two stages exist for different reasons and must not be collapsed:
      * fit()        needs a clean holdout, so it may only see anchors >=30d from the tuning
                     anchor. That is what makes the model selection and ensemble weights
                     honest, and it costs 5 of the fold's anchors.
      * refit_full() no longer needs a holdout -- it reuses the iteration counts already
                     found -- so it can take the lot. `train_data_extra` carries the 4
                     embargoed anchors that are in neither train_data nor tuning_data and
                     would otherwise be silently discarded. Without it the refit still
                     trains on ~20% less data than the LightGBM it is being compared to,
                     and on fold 0, 62% less.
    This is the same trade e0020 made: learn the round count on a holdout, then train on
    everything (-0.00349).
    """
    from autogluon.tabular import TabularPredictor
    pr = TabularPredictor(
        label="__y",
        problem_type="regression",
        eval_metric="root_mean_squared_error",   # on log1p target == RMSLE
        path=str(path),
        verbosity=2,
    )
    fit_kw = dict(
        train_data=train_df,
        tuning_data=tune_df,
        time_limit=args.time_limit,
        presets=args.presets,
        num_bag_folds=0,          # we drive the folds; never let AG resplit these rows
        num_stack_levels=0,       # keeps every model L1, which train_data_extra requires
        excluded_model_types=list(args.exclude),
    )
    if args.hyperparameters:
        # Backward-compatible: a JSON object string (starts with '{') is parsed into the dict
        # AutoGluon expects for an explicit model set, e.g. '{"REALMLP":{},"TABM":{}}' to train
        # ONLY those models. A bare preset-name string is passed through unchanged, so every
        # existing caller (which passes None or a preset name) is unaffected.
        hp = args.hyperparameters
        if isinstance(hp, str) and hp.lstrip().startswith("{"):
            hp = json.loads(hp)
        fit_kw["hyperparameters"] = hp
    pr.fit(**fit_kw)

    try:
        lb_holdout = pr.leaderboard()
        log("\n    leaderboard on the clean holdout:\n" + lb_holdout.to_string(max_rows=25))
    except Exception as e:                       # never lose the run over a display call
        log(f"    (leaderboard unavailable: {type(e).__name__}: {e})")
        lb_holdout = pd.DataFrame()

    if args.refit_full:
        n_extra = 0 if extra_df is None else len(extra_df)
        log(f"    refit_full('best'): retraining on train + tuning + {n_extra:,} embargoed "
            f"rows -> the full anchor set")
        pr.refit_full(model="best", set_best_to_refit_full=True,
                      train_data_extra=extra_df)
        log(f"    model_best is now {pr.model_best}")
    return pr, pr.predict(test_df).to_numpy(), lb_holdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/e0060_filt_top400.yaml")
    ap.add_argument("--exp-id", default="e0064")
    ap.add_argument("--mode", choices=["cv", "full"], default="cv")
    ap.add_argument("--fold", type=int, default=-1, help="cv mode: which frozen fold")
    ap.add_argument("--presets", default="medium_quality")
    ap.add_argument("--time-limit", type=int, default=7200)
    ap.add_argument("--exclude", nargs="*", default=["KNN"],
                    help="KNN is O(n^2)-ish on 4M x 400 and never competitive here")
    ap.add_argument("--hyperparameters", default=None)
    ap.add_argument("--subsample", type=int, default=0,
                    help="cap the TRAINING rows at N (random, fixed seed). Predictions are still "
                         "on the full validation population, so the OOF is comparable. Exists so "
                         "a CPU-only NN (RealMLP/TabM) is feasible; the gate (IDEAS.md §I17) "
                         "measured the row-subsample penalty at ~+0.0007 RMSLE at 1M rows.")
    # ON by default: without it AG trains on 20-62% fewer anchors than the LightGBM it is
    # being compared against, and the comparison measures the embargo, not the model.
    ap.add_argument("--no-refit-full", dest="refit_full", action="store_false")
    ap.set_defaults(refit_full=True)
    ap.add_argument("--keep-models", action="store_true",
                    help="scratch is over its soft quota; models are deleted by default")
    ap.add_argument("--cache", action="store_true",
                    help="feature cache (verify_cache.py: 87x, bit-identical). Six AG jobs "
                         "each rebuilding ~25 anchors at ~100s apiece is ~4h of pure rebuild.")
    args = ap.parse_args()

    if args.cache:
        import features as _feat
        _feat.enable_cache(True)
        log(f"    feature cache {'ON' if _feat.CACHE_ENABLED else 'REFUSED'} "
            f"(gen {_feat._code_hash()})")

    cfg = yaml.safe_load((ROOT / args.config).read_text())
    spec = json.loads((ROOT / "data" / "fold_spec.json").read_text())
    p = Panel()
    t0 = time.time()
    outdir = ROOT / "runs" / "ag"; outdir.mkdir(parents=True, exist_ok=True)

    if args.mode == "cv":
        k = args.fold
        fs = spec["folds"][k]
        va = date.fromisoformat(fs["valid_anchor"])
        tr = [date.fromisoformat(x) for x in fs["train_anchors"]]
        tune_anchor = tr[-1]
        fit_anchors = [a for a in tr[:-1] if (tune_anchor - a).days >= EMBARGO_DAYS]
        embargoed = [a for a in tr[:-1] if (tune_anchor - a).days < EMBARGO_DAYS]
        if not fit_anchors:
            raise ValueError(f"fold {k}: embargo leaves no fit anchors")
        log(f"\n=== {args.exp_id} fold {k}: valid {va} ===")
        log(f"    fit anchors  : {len(fit_anchors)} ({fit_anchors[0]} .. {fit_anchors[-1]})")
        log(f"    tuning anchor: {tune_anchor}  (gap to fits "
            f"{(tune_anchor - fit_anchors[-1]).days}d, gap to valid {(va - tune_anchor).days}d)")
        log(f"    embargoed    : {len(embargoed)} anchors held out of fit(), returned to the "
            f"refit via train_data_extra")
        log(f"    -> refit sees {len(fit_anchors) + 1 + len(embargoed)} of {len(tr)} anchors "
            f"(LightGBM uses all {len(tr)})")

        Xtr, ytr, names = build_at(p, cfg, fit_anchors)
        if args.subsample and Xtr.shape[0] > args.subsample:
            rng = np.random.default_rng(0)
            idx = np.sort(rng.choice(Xtr.shape[0], size=args.subsample, replace=False))
            log(f"    subsample: {Xtr.shape[0]:,} -> {args.subsample:,} training rows "
                f"(random, seed 0; validation population untouched)")
            Xtr, ytr = Xtr[idx], ytr[idx]
        Xtu, ytu, _ = build_at(p, cfg, [tune_anchor])
        Xex, yex = (None, None)
        if embargoed:
            Xex, yex, _ = build_at(p, cfg, embargoed)
        vai = p.idx(va); vkeep = p.active_in(vai - 29, vai)
        Xva, names_v = build(p, vai, vkeep, cfg["feature_blocks"])
        assert names_v == names
        mats = (Xtr, Xtu, Xva) if Xex is None else (Xtr, Xtu, Xva, Xex)
        names, mats = select_columns(cfg, names, mats)
        Xtr, Xtu, Xva = mats[0], mats[1], mats[2]
        Xex = mats[3] if Xex is not None else None
        yva = np.asarray(
            pd.read_parquet(ROOT / "data" / "folds.parquet")
            .query(f"fold_id == {k}").sort_values("user_id")["target"])
        log(f"    train {Xtr.shape[0]:,} x {Xtr.shape[1]}   tune {Xtu.shape[0]:,}   "
            f"valid {Xva.shape[0]:,}")

        path = ROOT / "data" / "ag" / f"{args.exp_id}_f{k}"
        pr, pred_log, lb = fit_predict(
            to_frame(Xtr, names, np.log1p(ytr)),
            to_frame(Xtu, names, np.log1p(ytu)),
            to_frame(Xva, names), args, path,
            extra_df=None if Xex is None else to_frame(Xex, names, np.log1p(yex)))
        pred = np.maximum(np.expm1(pred_log), 0.0)
        score = float(rmsle(yva, pred))
        log(f"\n    fold {k} RMSLE (our metric, our fold) = {score:.5f}")
        log("\n" + lb.to_string(max_rows=25))

        np.save(outdir / f"{args.exp_id}_f{k}_oof.npy", pred)
        (outdir / f"{args.exp_id}_f{k}.json").write_text(json.dumps({
            "exp_id": args.exp_id, "fold": k, "rmsle": score,
            "n_train": int(Xtr.shape[0]), "n_features": int(Xtr.shape[1]),
            "fit_anchors": [str(a) for a in fit_anchors], "tune_anchor": str(tune_anchor),
            "presets": args.presets, "time_limit": args.time_limit,
            "refit_full": bool(args.refit_full),
            "leaderboard": lb.head(25).to_dict("records"),
            "runtime_min": round((time.time() - t0) / 60, 1)}, indent=2, default=str))
        log(f"    wrote runs/ag/{args.exp_id}_f{k}.json")

    else:  # full -- fit at the real test anchor and write a submission
        import polars as pl
        test_anchor = p.dmax; tai = p.idx(test_anchor)
        latest = GUARD_START - timedelta(days=31)
        earliest = p.dmin + timedelta(days=MIN_HISTORY - 1)
        anchors, a = [], latest
        while a >= earliest:
            anchors.append(a); a -= timedelta(days=TRAIN_STRIDE)
        anchors = sorted(anchors)
        tune_anchor = anchors[-1]
        fit_anchors = [x for x in anchors[:-1] if (tune_anchor - x).days >= EMBARGO_DAYS]
        embargoed = [x for x in anchors[:-1] if (tune_anchor - x).days < EMBARGO_DAYS]
        log(f"\n=== {args.exp_id} FULL: test anchor {test_anchor} ===")
        log(f"    fit {len(fit_anchors)} anchors ({fit_anchors[0]} .. {fit_anchors[-1]}), "
            f"tune {tune_anchor}, embargoed {len(embargoed)} -> refit sees all {len(anchors)}")

        Xtr, ytr, names = build_at(p, cfg, fit_anchors)
        Xtu, ytu, _ = build_at(p, cfg, [tune_anchor])
        Xex, yex = (None, None)
        if embargoed:
            Xex, yex, _ = build_at(p, cfg, embargoed)
        keep_test = p.active_in(tai - 29, tai)
        assert keep_test.all(), "some test users fail the population rule"
        Xte, names_t = build(p, tai, keep_test, cfg["feature_blocks"])
        assert names_t == names
        mats = (Xtr, Xtu, Xte) if Xex is None else (Xtr, Xtu, Xte, Xex)
        names, mats = select_columns(cfg, names, mats)
        Xtr, Xtu, Xte = mats[0], mats[1], mats[2]
        Xex = mats[3] if Xex is not None else None
        log(f"    train {Xtr.shape[0]:,} x {Xtr.shape[1]}   test {Xte.shape[0]:,}")

        path = ROOT / "data" / "ag" / f"{args.exp_id}_full"
        pr, pred_log, lb = fit_predict(
            to_frame(Xtr, names, np.log1p(ytr)),
            to_frame(Xtu, names, np.log1p(ytu)),
            to_frame(Xte, names), args, path,
            extra_df=None if Xex is None else to_frame(Xex, names, np.log1p(yex)))
        pred = np.maximum(np.expm1(pred_log), 0.0)
        sub = pl.DataFrame({"user_id": p.users, "predict": pred})
        out = ROOT / "subs" / f"{args.exp_id}.csv"
        sub.write_csv(out)
        log(f"\n    wrote {out}  ({len(sub):,} rows, mean {pred.mean():,.2f}, "
            f"zeros {100 * (pred <= 0).mean():.1f}%)")
        log("\n" + lb.to_string(max_rows=25))
        (outdir / f"{args.exp_id}_full.json").write_text(json.dumps({
            "exp_id": args.exp_id, "mode": "full", "n_train": int(Xtr.shape[0]),
            "n_features": int(Xtr.shape[1]), "presets": args.presets,
            "leaderboard": lb.head(25).to_dict("records"),
            "runtime_min": round((time.time() - t0) / 60, 1)}, indent=2, default=str))

    if not args.keep_models:
        # AutoGluon writes multi-GB model dirs; scratch is shared and over its soft quota.
        shutil.rmtree(path, ignore_errors=True)
        log(f"    removed {path} (pass --keep-models to retain)")
    log(f"\n  total {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
