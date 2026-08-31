#!/usr/bin/env python
"""
The e0141 setup, with a tabular model instead of a GRU: CatBoost / XGBoost / Ridge.

    python src/run_usercv_tab.py --model catboost --variant full

WHY.  Every rho gain of any size in this project came from adding a different model FAMILY:
+0.00143 for the first gbdt+seq blend, +0.00048 for adding the user-split GRU.  Features
(+0.00019 for 480 of them), capacity, seeds, objectives and 42 hand-built behavioural columns
all measured zero.  CatBoost and XGBoost have never been run once here -- 70 of the 88 logged
experiments are LightGBM and 18 are torch -- and CatBoost in particular is structurally distant
from LightGBM: oblivious (symmetric) trees and ordered boosting rather than leaf-wise growth.

Everything except the model is e0141's setup exactly, so the comparison is clean:
  * variant `full` (85 features), target log1p(SUM gmv over [t+1, t+30]);
  * tmask = burn-in 14 from the user's first active day, through 2026-01-14;
  * the same md5("gmv-v1:uid") 5-fold user split -- identical fold membership to e0141;
  * scaling stats fit on train users only (matters for ridge; trees ignore it).

ROW BUDGET.  The full tmask is 80.9M user-days.  Training rows are strided every 7 days, which
e0115 measured as free (+0.00012, 0.6 sigma) -- the targets of adjacent days overlap 29/30, so
the extra rows are nearly duplicates.  EVALUATION uses every scored day of the held-out users,
so the reported number is directly comparable to e0141's 1.74341.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rho_decomp import auc                                                    # noqa: E402
from usercv_features import (Raw, build_features, build_target, build_tmask,  # noqa: E402
                             flag_channels, geo3_log, hash_fold, max_anchor)



# The five frozen validation anchors (data/fold_spec.json).  Saving out-of-sample predictions
# HERE is what makes a user-split model blendable against everything else: each user is held
# out in exactly one user-fold, so stitching the folds together yields a complete out-of-sample
# prediction at each frozen anchor, on the same (anchor, user_id) keys as oof/e0049.parquet.
FROZEN_ANCHORS = {0: "2025-06-18", 1: "2025-07-18", 2: "2025-08-17",
                  3: "2025-09-16", 4: "2025-10-16"}


def log(m: str) -> None:
    print(m, flush=True)


def fit_predict(model: str, Xtr, ytr, Xva_iter, n_va, seed: int, rounds: int,
                device: str = "gpu", tuned: dict | None = None):
    """Returns predictions for the validation rows, streamed in chunks.

    `tuned` overrides the hard-coded defaults. That matters: the defaults here are
    lr=0.05/depth=8, which is the same untuned setting that made CatBoost 0.032 weaker than
    the GRU in the first place -- and §1c concluded decorrelation was worthless BECAUSE of
    that quality gap. On the frozen folds, tuning closed the gap to 0.00017 (1.76473 vs
    1.76456). So a usercv comparison run at lr=0.05 measures the default, not the family."""
    tuned = dict(tuned or {})
    if model == "lightgbm":
        # never available in this harness before: 70 of 88 logged experiments are LightGBM on
        # the FROZEN folds, but none under the user split, which is the protocol that pays
        # (usercv_full beats its frozen sibling e0101 by a flat -0.0028 across all five folds)
        import lightgbm as lgb
        pr = dict(objective="regression", metric="rmse", verbosity=-1, seed=seed,
                  num_threads=int(tuned.pop("num_threads", 16)))
        pr.update(tuned)
        m = lgb.train(pr, lgb.Dataset(Xtr, ytr), num_boost_round=rounds,
                      callbacks=[lgb.log_evaluation(0)])
    elif model == "catboost":
        from catboost import CatBoostRegressor
        kw = dict(learning_rate=0.05, depth=8)
        kw.update({k: v for k, v in tuned.items() if k != "num_threads"})
        m = CatBoostRegressor(iterations=rounds, loss_function="RMSE", random_seed=seed,
                              task_type="GPU" if device == "gpu" else "CPU",
                              devices="0", verbose=max(rounds // 5, 1),
                              allow_writing_files=False, **kw)
        m.fit(Xtr, ytr)
    elif model == "xgboost":
        import xgboost as xgb
        m = xgb.XGBRegressor(n_estimators=rounds, learning_rate=0.05, max_depth=8,
                             subsample=0.8, colsample_bytree=0.8, min_child_weight=50,
                             tree_method="hist",
                             device="cuda" if device == "gpu" else "cpu",
                             random_state=seed,
                             objective="reg:squarederror", verbosity=1)
        m.fit(Xtr, ytr)
    elif model == "ridge":
        # closed form on 85 columns; a deliberately weak, maximally different function class.
        # CLAUDE.md §6 wants a non-GBDT alive for blend diversity at near-zero cost.
        A = np.column_stack([Xtr, np.ones(len(Xtr), np.float32)]).astype(np.float64)
        lam = 1e-3 * len(Xtr)
        G = A.T @ A + lam * np.eye(A.shape[1])
        w = np.linalg.solve(G, A.T @ ytr.astype(np.float64))
        m = w
    else:
        raise ValueError(model)

    out = np.empty(n_va, np.float32)
    i = 0
    for chunk in Xva_iter:
        out[i:i + len(chunk)] = predict_rows(model, m, chunk)
        i += len(chunk)
    assert i == n_va
    return out, m


def predict_rows(kind: str, m, Xr: np.ndarray) -> np.ndarray:
    """Apply a fitted model to a raw feature block (ridge is a plain weight vector)."""
    if kind == "ridge":
        return (np.column_stack([Xr, np.ones(len(Xr), np.float32)]).astype(np.float64) @ m)
    return m.predict(Xr)


def _load_tuned(path: str):
    if not path:
        return None, None
    d = json.loads(Path(path).read_text())
    p = dict(d["params"])
    log(f"  tuned params from {path}: cv_mean {d.get('cv_mean')}, rounds {d.get('rounds')}")
    return p, int(d.get("rounds", 0)) or None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["lightgbm", "catboost", "xgboost", "ridge"])
    ap.add_argument("--predict-test", default="",
                    help="exp_id: ALSO fit on ALL users and write subs/<id>.csv at the test "
                         "anchor. Without this a usercv tabular model has OOF but no test "
                         "prediction, so it can win blend weight and still be unsubmittable "
                         "-- which is the state usercv_catboost/xgboost/ridge are in.")
    ap.add_argument("--tuned-params", default="",
                    help="path to a *_params.json from src/tune.py; without it the model runs "
                         "at the hard-coded lr=0.05 defaults, which is what handicapped "
                         "CatBoost in the first place")
    ap.add_argument("--variant", default="full")
    ap.add_argument("--exp-id", default="")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--stride", type=int, default=7, help="training-day stride (e0115: free)")
    ap.add_argument("--rounds", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--residual", action="store_true",
                    help="predict L - log1p(geo3) and add the baseline back (per-user offset)")
    args = ap.parse_args()
    TUNED, TROUNDS = _load_tuned(args.tuned_params)
    if TROUNDS:
        args.rounds = TROUNDS
    exp = args.exp_id or f"usercv_{args.model}"
    t0 = time.time()
    log(f"\n=== {exp}: e0141 setup, model={args.model}, variant={args.variant} ===")

    raw = Raw()
    last = max_anchor(raw)
    Y = build_target(raw, "sum")
    M = build_tmask(raw, last, burn_in=14, trim_to_first_seen=True)
    X, names = build_features(raw, args.variant)
    # Residual parametrisation: the model learns the DEVIATION from the user's own level.
    # RMSLE is unchanged by construction -- (L - B) - (M - B) = L - M -- so the reported
    # number stays directly comparable; only the learning problem is re-parametrised.
    B = geo3_log(raw) if args.residual else None
    is_flag = flag_channels(names)
    fold_of = hash_fold(raw.users)                      # identical membership to e0141
    log(f"    {len(names)} features | tmask {int(M.sum()):,} user-days | "
        f"folds {np.bincount(fold_of).tolist()}")

    # frozen population rule, for the secondary number run_usercv also reports
    csa = np.concatenate([np.zeros((raw.n, 1), np.int32),
                          np.cumsum(raw.active > 0, 1, dtype=np.int32)], 1)
    lo = np.maximum(np.arange(raw.T) - 29, 0)
    POP = (csa[:, 1:] - csa[:, lo]) > 0
    del csa

    frozen = {k: raw.idx(__import__('datetime').date.fromisoformat(v))
              for k, v in FROZEN_ANCHORS.items()}
    oof_rows = []
    days = np.arange(raw.T)
    tr_days = days[(days % args.stride) == (last % args.stride)]
    per_fold, per_pop, per_auc = [], [], []

    for k in args.folds:
        tr_u = np.flatnonzero(fold_of != k)
        va_u = np.flatnonzero(fold_of == k)
        sub = X[tr_u[::37]].astype(np.float32)
        mu = sub.mean(axis=(0, 1)); sd = np.maximum(sub.std(axis=(0, 1)), 1e-3)
        mu[is_flag] = 0.0; sd[is_flag] = 1.0
        del sub

        xs, ys = [], []
        for t in tr_days:
            m = M[tr_u, t]
            if not m.any():
                continue
            xs.append((X[tr_u[m], t, :].astype(np.float32) - mu) / sd)
            ys.append(Y[tr_u[m], t] - (B[tr_u[m], t] if B is not None else 0.0))
        Xtr = np.concatenate(xs); ytr = np.concatenate(ys)
        del xs, ys

        va_rows = [(t, np.flatnonzero(M[va_u, t])) for t in days]
        va_rows = [(t, r) for t, r in va_rows if r.size]
        n_va = sum(r.size for _, r in va_rows)
        log(f"\n    fold {k}: train {Xtr.shape[0]:,} rows x {Xtr.shape[1]} "
            f"({tr_u.size:,} users, stride {args.stride}) | "
            f"eval {n_va:,} rows ({va_u.size:,} unseen users)")

        def chunks():
            for t, r in va_rows:
                yield (X[va_u[r], t, :].astype(np.float32) - mu) / sd

        pred, fitted = fit_predict(args.model, Xtr, ytr, chunks(), n_va, args.seed,
                                   args.rounds, args.device, tuned=TUNED)
        del Xtr, ytr

        yv = np.concatenate([Y[va_u[r], t] for t, r in va_rows])
        bv = (np.concatenate([B[va_u[r], t] for t, r in va_rows])
              if B is not None else np.zeros_like(yv))
        pred = pred + bv                       # back to the absolute log scale
        pv = np.concatenate([POP[va_u[r], t] for t, r in va_rows])
        rm = float(np.sqrt(np.mean((pred - yv) ** 2)))
        rp = float(np.sqrt(np.mean((pred[pv] - yv[pv]) ** 2)))
        a = auc(pred, (yv > 0).astype(float))
        # out-of-sample predictions at the five frozen anchors, for this fold's held-out
        # users. Stitched across folds these give a complete OOF on the same (anchor, user)
        # keys as oof/e0049.parquet -- which is what makes this family blendable at all.
        for fk, ta in frozen.items():
            r = np.flatnonzero(M[va_u, ta])
            if not r.size:
                continue
            xa = (X[va_u[r], ta, :].astype(np.float32) - mu) / sd
            pa = predict_rows(args.model, fitted, xa)
            if B is not None:
                pa = pa + B[va_u[r], ta]
            oof_rows.append((fk, raw.users[va_u[r]], Y[va_u[r], ta], pa))

        per_fold.append(rm); per_pop.append(rp); per_auc.append(a)
        log(f"    -> fold {k}: UNSEEN {rm:.5f}  (in-population {rp:.5f})  AUC {a:.5f}  "
            f"[{(time.time() - t0) / 60:.1f}m]")

    r, rp_, au = np.array(per_fold), np.array(per_pop), np.array(per_auc)
    log(f"\n  === {exp} ===")
    log(f"  UNSEEN-USER RMSLE = {r.mean():.5f} +/- {r.std():.5f}   folds {np.round(r, 5).tolist()}")
    log(f"  in-population     = {rp_.mean():.5f}")
    log(f"  AUC on y>0        = {au.mean():.5f}")
    log(f"  e0141 (GRU, same setup) was 1.74341 / 1.76808 / 0.84647")
    log(f"  runtime {(time.time() - t0) / 60:.1f} min")

    if oof_rows:
        import pyarrow as pa, pyarrow.parquet as pq
        fk = np.concatenate([np.full(u.size, k, np.int8) for k, u, _, _ in oof_rows])
        uu = np.concatenate([u for _, u, _, _ in oof_rows])
        yy = np.concatenate([y for _, _, y, _ in oof_rows])
        pp = np.concatenate([p for _, _, _, p in oof_rows])
        o = np.lexsort((uu, fk))
        (ROOT / "oof").mkdir(exist_ok=True)
        pq.write_table(pa.table({"fold_id": fk[o], "user_id": uu[o],
                                 "y_true": np.expm1(yy[o]).astype(np.float64),
                                 "y_pred": np.maximum(np.expm1(pp[o]), 0.0).astype(np.float64)}),
                       ROOT / "oof" / f"{exp}.parquet")
        log(f"  wrote oof/{exp}.parquet ({fk.size:,} rows at the 5 frozen anchors)")

    # ---------------------------------------------------------------- test prediction
    if args.predict_test:
        # The CV loop above holds out a fold of USERS; for the submission there is nothing to
        # hold out, so refit on everyone. Two things differ from a fold fit and both matter:
        #   * scaling stats come from ALL users, not the 4/5 training fold;
        #   * features are read at the LAST day of the panel (2026-02-13), which is outside
        #     the tmask -- the mask stops at 2026-01-14 because a TARGET needs [t+1, t+30],
        #     and prediction needs no target. Using `last` here would predict a month stale.
        log(f"\n=== {args.predict_test}: refit on all {raw.n:,} users -> test anchor ===")
        sub = X[::37].astype(np.float32)
        mu = sub.mean(axis=(0, 1)); sd = np.maximum(sub.std(axis=(0, 1)), 1e-3)
        mu[is_flag] = 0.0; sd[is_flag] = 1.0
        del sub
        xs, ys = [], []
        for t in tr_days:
            m = M[:, t]
            if not m.any():
                continue
            xs.append((X[m, t, :].astype(np.float32) - mu) / sd)
            ys.append(Y[m, t] - (B[m, t] if B is not None else 0.0))
        Xall = np.concatenate(xs); yall = np.concatenate(ys); del xs, ys
        t_test = raw.T - 1
        log(f"    train {Xall.shape[0]:,} rows | predicting at day index {t_test} "
            f"({raw.day(t_test) if hasattr(raw, 'day') else 'panel end'}) for {raw.n:,} users")

        def _one():
            yield (X[:, t_test, :].astype(np.float32) - mu) / sd

        pt, _ = fit_predict(args.model, Xall, yall, _one(), raw.n, args.seed,
                            args.rounds, args.device, tuned=TUNED)
        if B is not None:
            pt = pt + B[:, t_test]
        pt = np.maximum(np.expm1(pt), 0.0)
        import polars as _pl
        o = ROOT / "subs" / f"{args.predict_test}.csv"
        _pl.DataFrame({"user_id": raw.users, "predict": pt}).write_csv(o)
        log(f"    wrote {o} ({len(pt):,} rows, mean {pt.mean():,.2f}, "
            f"zeros {100 * (pt <= 0).mean():.2f}%)")
        del Xall, yall

    out = ROOT / "reports" / "eda" / f"{exp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"exp": exp, "model": args.model, "variant": args.variant,
                               "date": datetime.now().isoformat(timespec="seconds"),
                               "unseen_rmsle_mean": float(r.mean()), "per_fold": r.tolist(),
                               "in_population_mean": float(rp_.mean()),
                               "auc_mean": float(au.mean()), "stride": args.stride,
                               "rounds": args.rounds,
                               "runtime_min": round((time.time() - t0) / 60, 1)}, indent=2))
    log(f"  wrote reports/eda/{exp}.json")


if __name__ == "__main__":
    main()
