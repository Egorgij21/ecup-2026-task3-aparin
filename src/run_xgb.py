#!/usr/bin/env python
"""
XGBoost entrypoint on the FROZEN folds -- the sibling of `run.py`, same protocol, one
model family swapped.

    python src/run_xgb.py --config configs/e0210_xgb_base.yaml

Why a separate runner rather than a `model:` switch inside run.py: `run.py` is the workhorse
behind 70+ logged experiments and every published CV number. The repo already keeps one
runner per family (`run_clf.py`, `run_usercv_tab.py`, `run_ag.py`); this follows that.
What is NOT duplicated is the part that must not drift:

  * `assert_no_lookahead` is IMPORTED from run.py, not re-implemented, so the leak guard
    protecting an XGB experiment is byte-identical to the one protecting a LightGBM one;
  * folds come from data/folds.parquet + data/fold_spec.json (rule 3);
  * the metric comes from src/metrics.py (rule 4);
  * OOF lands in oof/<exp_id>.parquet and a row goes to runs/<exp_id>.json -> experiments.csv
    (rule 5), with the identical column set, so XGB rows and LightGBM rows are comparable.

The naive `geo3` reference is re-scored on the identical folds inside every run, so `delta`
is always an exact paired comparison (and, as REVIEW_NOTES.md A1 records, `delta` in the CSV
is vs that naive floor -- NOT vs the parent. Parent deltas are computed in the report below
and belong in EXPERIMENTS.md).

`min_child_weight` is XGB's sum-of-hessians per leaf; for squared error the hessian is 1 per
row, so `min_child_weight: 200` is the exact analogue of LightGBM's `min_data_in_leaf: 200`.
With `grow_policy: lossguide` + `max_leaves: 63` the tree shape mirrors `num_leaves: 63`,
which is what makes e0210 a fair family reference for e0049 rather than a differently-sized
model wearing the same feature set.
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

from data import Panel                          # noqa: E402
from features import build                      # noqa: E402
from metrics import rmsle, score_all            # noqa: E402
from run import assert_no_lookahead             # noqa: E402  -- the SAME guard, not a copy


def log(m: str) -> None:
    print(m, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--screen", action="store_true", help="2 folds only (tier=screen)")
    ap.add_argument("--rounds-curve", default="",
                    help="comma-separated round counts to also score (screen diagnostics "
                         "only -- picking the fixed round count for the family)")
    ap.add_argument("--no-log", action="store_true",
                    help="skip runs/*.json + experiments.csv (smoke tests only)")
    ap.add_argument("--max-train-anchors", type=int, default=0,
                    help="SMOKE TESTS ONLY: cap the fit anchors per fold. This changes the "
                         "training set, so a run using it is not comparable to anything and "
                         "must be paired with --no-log.")
    args = ap.parse_args()

    if args.max_train_anchors and not args.no_log:
        raise SystemExit("--max-train-anchors changes the training set; pass --no-log too, "
                         "or put max_train_anchors in the config as a real experiment.")

    cfg = yaml.safe_load(Path(args.config).read_text())
    exp_id = cfg["exp_id"]
    t0 = time.time()
    log(f"\n=== {exp_id} : {cfg['change']} ===")
    log(f"    parent={cfg['parent_id']}  blocks={cfg['feature_blocks']}  seed={cfg['seed']}")

    spec = json.loads((ROOT / "data" / "fold_spec.json").read_text())
    folds = pl.read_parquet(ROOT / "data" / "folds.parquet")
    fold_ids = sorted(folds["fold_id"].unique().to_list())
    if args.screen:
        fold_ids = fold_ids[-2:]
    tier = "screen" if args.screen else cfg.get("tier", "confirm")

    p = Panel()
    import xgboost as xgb
    log(f"    xgboost {xgb.__version__}")

    FIXED = int(cfg["fixed_rounds"])
    if FIXED <= 0:
        raise ValueError("run_xgb.py requires fixed_rounds > 0: the ES diagnostic "
                         "(reports/eda/diag_es.json) showed a flat validation curve, and a "
                         "variable-length early-stopped fit makes deltas uninterpretable "
                         "(EXPERIMENTS.md §1j).")
    curve = [int(x) for x in args.rounds_curve.split(",") if x.strip()]

    oof, per_fold, per_fold_naive, curves = [], [], [], []
    for k in fold_ids:
        fs = spec["folds"][k]
        va = date.fromisoformat(fs["valid_anchor"])
        vai = p.idx(va)
        tr_anchors = [date.fromisoformat(x) for x in fs["train_anchors"]]
        if cfg.get("max_train_anchors"):
            tr_anchors = tr_anchors[-int(cfg["max_train_anchors"]):]
        if args.max_train_anchors:                    # smoke only; --no-log is enforced above
            tr_anchors = tr_anchors[-args.max_train_anchors:]
        # fixed_rounds > 0 -> no early-stopping holdout, train on every anchor (as e0049 does)
        fit_anchors = tr_anchors

        Xtr, ytr = [], []
        for a in fit_anchors:
            ai = p.idx(a)
            keep = p.active_in(ai - 29, ai)
            p.set_floor(ai, cfg.get("feature_truncate_days"))
            X, names = build(p, ai, keep, cfg["feature_blocks"], cfg.get("feature_max_window"))
            Xtr.append(X); ytr.append(p.target(ai)[keep])
        Xtr = np.concatenate(Xtr); ytr_raw = np.concatenate(ytr)
        ytr = np.log1p(ytr_raw)
        log(f"    fold {k}: {len(fit_anchors)} fit anchors "
            f"({fit_anchors[0]}..{fit_anchors[-1]}), {Xtr.shape[0]:,} train rows x "
            f"{Xtr.shape[1]} features  [t+{(time.time() - t0) / 60:.1f}m]")

        vkeep = p.active_in(vai - 29, vai)
        p.set_floor(vai, cfg.get("feature_truncate_days"))
        Xva, names = build(p, vai, vkeep, cfg["feature_blocks"], cfg.get("feature_max_window"))
        if k == fold_ids[0]:
            assert_no_lookahead(p, vai, Xva, vkeep, cfg["feature_blocks"],
                                cfg.get("feature_max_window"))
            log(f"    look-ahead check passed; {Xva.shape[1]} features")

        if cfg.get("feature_whitelist"):
            WL = set(json.loads(Path(cfg["feature_whitelist"]).read_text()))
            sel = [i for i, n in enumerate(names) if n in WL]
            log(f"    whitelist: keeping {len(sel)} of {len(names)} features")
            Xva = Xva[:, sel]; Xtr = Xtr[:, sel]; names = [names[i] for i in sel]

        fv = folds.filter(pl.col("fold_id") == k).sort("user_id")
        yva = fv["target"].to_numpy()
        assert np.array_equal(fv["user_id"].to_numpy(), p.users[vkeep]), "fold population drift"

        params = dict(cfg["xgb_params"])
        params["seed"] = cfg["seed"]
        dtr = xgb.QuantileDMatrix(Xtr, label=ytr, feature_names=names,
                                  max_bin=int(params.get("max_bin", 256)))
        del Xtr                                   # the quantised copy is what training reads
        dva = xgb.DMatrix(Xva, feature_names=names)
        model = xgb.train(params, dtr, num_boost_round=FIXED, verbose_eval=False)
        pred = np.maximum(np.expm1(model.predict(dva)), 0.0)

        if curve:
            cs = {}
            for r in curve:
                pr = np.maximum(np.expm1(model.predict(dva, iteration_range=(0, r))), 0.0)
                cs[r] = round(rmsle(yva, pr), 5)
            curves.append((k, cs))
            log(f"    rounds curve fold {k}: {cs}")

        # naive geo3 on the identical rows -- never taken from a log (see run.py)
        if "geo3" in names:
            naive = np.maximum(Xva[:, names.index("geo3")].astype(np.float64), 0.0)
        else:
            _fl = p.floor; p.floor = 0
            blks = [p.wsum("gmv", vai - 29 - 30 * j, vai - 30 * j) for j in range(3)]
            p.floor = _fl
            naive = np.maximum(np.expm1(np.mean([np.log1p(b) for b in blks], axis=0))[vkeep], 0.0)

        s, sn = rmsle(yva, pred), rmsle(yva, naive)
        per_fold.append(s); per_fold_naive.append(sn)
        oof.append(pl.DataFrame({"fold_id": np.full(yva.size, k, np.int8),
                                 "anchor_date": pl.Series("anchor_date", [va] * yva.size,
                                                          dtype=pl.Date),
                                 "user_id": fv["user_id"].to_numpy(),
                                 "y_true": yva, "y_pred": pred, "y_naive": naive}))
        log(f"    fold {k} {va}  n={yva.size:>7,} trees={FIXED:>4d}  rmsle={s:.5f}  "
            f"naive={sn:.5f}  delta={s - sn:+.5f}  [t+{(time.time() - t0) / 60:.1f}m]")
        del Xva, dtr, dva

    oof = pl.concat(oof)
    pf = np.array(per_fold); pfn = np.array(per_fold_naive)
    agg = score_all(oof["y_true"].to_numpy(), oof["y_pred"].to_numpy())
    agg_n = score_all(oof["y_true"].to_numpy(), oof["y_naive"].to_numpy())
    wins = int((pf < pfn).sum())
    runtime = (time.time() - t0) / 60

    log(f"\n  cv_mean = {pf.mean():.5f} +/- {pf.std():.5f}   folds {np.round(pf, 5).tolist()}")
    log(f"  naive   = {pfn.mean():.5f} +/- {pfn.std():.5f}   folds {np.round(pfn, 5).tolist()}")
    log(f"  delta vs naive = {pf.mean() - pfn.mean():+.5f}   wins {wins}/{len(pf)} folds")
    log(f"  last fold (most test-like) = {pf[-1]:.5f}  (naive {pfn[-1]:.5f}, "
        f"delta {pf[-1] - pfn[-1]:+.5f})")
    log(f"  tie-breakers: gini_pred={agg['gini_pred']:.4f} (true {agg['gini_true']:.4f})  "
        f"total_rel_err={agg['total_rel_err']:+.4f}  [naive: {agg_n['gini_pred']:.4f}, "
        f"{agg_n['total_rel_err']:+.4f}]")
    log(f"  runtime {runtime:.1f} min")
    imp = sorted(model.get_score(importance_type="gain").items(), key=lambda x: -x[1])[:15]
    log(f"\n  top-15 features by gain:")
    for n, g in imp:
        log(f"    {n:30s} {g:>14,.1f}")

    if args.no_log:
        log("\n  --no-log: nothing written (smoke test)")
        return

    (ROOT / "oof").mkdir(exist_ok=True)
    oof.write_parquet(ROOT / "oof" / f"{exp_id}.parquet")

    row = {
        "exp_id": exp_id, "parent_id": cfg["parent_id"],
        "date": datetime.now().isoformat(timespec="seconds"),
        "approach": cfg["approach"], "change": cfg["change"], "tier": tier,
        "n_features": len(names), "cv_mean": round(float(pf.mean()), 5),
        "cv_std": round(float(pf.std()), 5),
        "folds": json.dumps([round(float(x), 5) for x in pf]),
        "delta": round(float(pf.mean() - pfn.mean()), 5),
        "significant": "yes" if wins >= 4 or abs(pf.mean() - pfn.mean()) > 2 * pf.std() else "no",
        "lb": "", "runtime_min": round(runtime, 1), "seed": cfg["seed"],
        "config": args.config, "verdict": cfg.get("verdict", ""),
        "gini_pred": round(agg["gini_pred"], 4),
        "total_rel_err": round(agg["total_rel_err"], 4),
        "notes": cfg.get("notes", ""),
        "best_iters": json.dumps([FIXED] * len(pf)),
    }
    rd = ROOT / "runs"; rd.mkdir(exist_ok=True)
    (rd / f"{exp_id}.json").write_text(json.dumps(row, indent=2))
    try:
        import subprocess
        subprocess.run([sys.executable, str(ROOT / "src" / "collect.py")], check=False)
    except Exception:
        pass
    log(f"\n  wrote runs/{exp_id}.json -> experiments.csv")


if __name__ == "__main__":
    main()
