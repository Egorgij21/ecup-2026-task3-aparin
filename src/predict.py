#!/usr/bin/env python
"""
Fit the final model at the real test anchor and write subs/<exp_id>.csv.

    python src/predict.py --config configs/e0001_lgbm_base.yaml

The recipe is deliberately the SAME one CV measured, so the submitted number is
comparable to the logged cv_mean (CLAUDE.md rule 2 -- a submission must not quietly
introduce a second change):

  * training anchors  : the frozen 7-day grid, but only anchors whose target window is
                        CLEAN, i.e. A_train + 30 < 2025-11-16 (A_train <= 2025-10-16).
                        Extending into the guaranteed-activity zone would add ~3 months of
                        recent data at the price of guarantee-inflated targets -- that is
                        an untested change and belongs in its own experiment, not here.
  * population        : users active in [A-29, A] -- at the test anchor that is all 250 000.
  * rounds            : the median best_iteration from this exp_id's CV run, so the final
                        model matches what CV converged to instead of an arbitrary cap.
  * seed              : as in the config; no multi-seed averaging (one change at a time).
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

from data import Panel                # noqa: E402
from features import build            # noqa: E402
from metrics import gini              # noqa: E402

GUARD_START = date(2025, 11, 16)
TRAIN_STRIDE = 7
HORIZON = 30
MIN_HISTORY = 90


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--rounds", type=int, default=0,
                    help="0 = median best_iteration from this exp_id's CV run")
    ap.add_argument("--train-through", default=None, metavar="YYYY-MM-DD",
                    help="extend training anchors past the guard-zone boundary "
                         "(default 2025-10-16). Capped at the horizon tail 2026-01-14.")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    exp_id = cfg["exp_id"]

    rounds = args.rounds
    if rounds <= 0:
        import json as _j
        rf = ROOT / "runs" / f"{exp_id}.json"
        bi = _j.loads(_j.loads(rf.read_text())["best_iters"]) if rf.exists() else []
        bi = [b[-1] if isinstance(b, list) else b for b in bi]
        rounds = int(np.median(bi)) if bi else 3000
        print(f"  rounds = {rounds} (median best_iteration across the CV folds)")

    p = Panel()
    import lightgbm as lgb

    test_anchor = p.dmax
    tai = p.idx(test_anchor)
    print(f"\n  test anchor {test_anchor} -> predict "
          f"{test_anchor + timedelta(days=1)} .. {test_anchor + timedelta(days=30)}")

    # clean training anchors on the frozen grid
    latest = GUARD_START - timedelta(days=31)              # 2025-10-16
    if args.train_through:
        # See run_seq.py's docstring: the +0.00189 that justified excluding guard-zone anchors
        # was measured by validating AT 2026-01-14, itself inside the guaranteed-activity zone,
        # and no clean anchor exists to re-test it. e0141 (0.42 of the champion) already trains
        # through 2026-01-14. Default unchanged, so every logged submission is reproducible.
        latest = date.fromisoformat(args.train_through)
        tail = p.dmax - timedelta(days=HORIZON)            # target window must be observed
        assert latest <= tail, (f"train-through {latest} exceeds the horizon tail {tail}: "
                                f"target windows would be truncated")
    earliest = p.dmin + timedelta(days=MIN_HISTORY - 1)
    anchors, a = [], latest
    while a >= earliest:
        anchors.append(a); a -= timedelta(days=TRAIN_STRIDE)
    anchors = sorted(anchors)
    n_dirty = sum(1 for a_ in anchors if a_ + timedelta(days=HORIZON) >= GUARD_START)
    print(f"  training anchors: {len(anchors)}  ({anchors[0]} .. {anchors[-1]}, "
          f"stride {TRAIN_STRIDE}d, "
          + ("all target windows clean)" if n_dirty == 0 else
             f"{n_dirty} of them GUARD-ZONE contaminated -- deliberate, --train-through)"))

    Xtr, ytr = [], []
    for a in anchors:
        ai = p.idx(a)
        keep = p.active_in(ai - 29, ai)
        X, names = build(p, ai, keep, cfg["feature_blocks"])
        Xtr.append(X); ytr.append(np.log1p(p.target(ai)[keep]))
    Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr)

    # The column selectors MUST be applied here as well as in run.py. Without them this script
    # silently fits a different model from the one the config names: e0060 is defined by a
    # 400-feature whitelist over an 809-feature build, so ignoring it ships an 809-feature
    # model under e0060's name and the submission no longer corresponds to any logged CV.
    def _select(names_, mats):
        if cfg.get("feature_exclude_patterns"):
            import re as _re
            pats = [_re.compile(x) for x in cfg["feature_exclude_patterns"]]
            sel = [i for i, n in enumerate(names_) if not any(q.search(n) for q in pats)]
            print(f"  exclude patterns: dropping {len(names_) - len(sel)} of {len(names_)}")
            mats = [m[:, sel] for m in mats]; names_ = [names_[i] for i in sel]
        if cfg.get("feature_whitelist"):
            WL = set(json.loads(Path(cfg["feature_whitelist"]).read_text()))
            sel = [i for i, n in enumerate(names_) if n in WL]
            print(f"  whitelist: keeping {len(sel)} of {len(names_)} features")
            mats = [m[:, sel] for m in mats]; names_ = [names_[i] for i in sel]
        return names_, mats

    names, (Xtr,) = _select(names, (Xtr,))
    print(f"  training matrix: {Xtr.shape[0]:,} rows x {Xtr.shape[1]} features")

    params = dict(cfg["lgb_params"]); params["seed"] = cfg["seed"]
    model = lgb.train(params, lgb.Dataset(Xtr, ytr, feature_name=names),
                      num_boost_round=rounds, callbacks=[lgb.log_evaluation(0)])
    print(f"  trained {rounds} rounds, seed {cfg['seed']}")

    keep_test = p.active_in(tai - 29, tai)
    assert keep_test.all(), "some test users fail the population rule -- unexpected"
    Xte, names2 = build(p, tai, keep_test, cfg["feature_blocks"])
    names2, (Xte,) = _select(names2, (Xte,))
    assert names2 == names, "test columns diverged from training columns"
    pred = np.maximum(np.expm1(model.predict(Xte)), 0.0)

    # geo3 may itself be filtered out of the model's columns, so read the naive reference
    # from the unfiltered build rather than assuming it survived selection.
    Xte_full, names_full = build(p, tai, keep_test, cfg["feature_blocks"])
    naive = np.maximum(Xte_full[:, names_full.index("geo3")].astype(np.float64), 0.0)
    del Xte_full
    last30 = p.wsum("gmv", tai - 29, tai)

    sub = pl.DataFrame({"user_id": p.users, "predict": pred})
    ss = pl.read_csv(ROOT / "data" / "sample_submit.csv")
    assert sub.height == ss.height == 250_000
    assert np.array_equal(sub["user_id"].to_numpy(), ss["user_id"].to_numpy()), "user order"
    assert np.isfinite(pred).all() and (pred >= 0).all()
    out = ROOT / "subs" / f"{exp_id}.csv"
    out.parent.mkdir(exist_ok=True)
    sub.write_csv(out)

    print(f"\n  wrote {out.relative_to(ROOT)}  ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"  {'':22s} {'sum':>16s} {'mean':>10s} {'zero share':>11s} {'gini':>8s} {'p99':>10s}")
    for nm, v in [(f"{exp_id} prediction", pred), ("naive geo3", naive), ("last-30d (=sample)", last30)]:
        print(f"  {nm:22s} {v.sum():>16,.0f} {v.mean():>10.2f} "
              f"{100 * (v == 0).mean():>10.2f}% {gini(v):>8.4f} {np.quantile(v, .99):>10.2f}")
    print(f"\n  reference: last observed 30d GMV = {last30.sum():,.0f}; seasonally calibrated"
          f" estimate for the test window = {last30.sum() * 1.1628:,.0f} (DATA.md 5.4)")
    import json as _json
    rf = ROOT / "runs" / f"{exp_id}.json"
    cvm = float(_json.loads(rf.read_text())["cv_mean"]) if rf.exists() else float("nan")
    print(f"  cv_mean for this recipe = {cvm:.5f}; with the measured +0.102 CV-LB offset the"
          f" expected public LB is ~{cvm - 0.102:.4f}")
    print(f"  NO calibration applied: src/calibrate.py found k* = 1.000 (all forms make CV"
          f" worse); fixing the aggregate would cost +0.163 RMSLE (DATA.md 8.4)")


if __name__ == "__main__":
    main()
