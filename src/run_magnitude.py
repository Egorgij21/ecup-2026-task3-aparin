#!/usr/bin/env python
"""
The MAGNITUDE term: train on buyers, score `corr(L, .|Z=1)` against IDEAS.md §I13's ceiling.

    python src/run_magnitude.py --config configs/e0250_magnitude.yaml

WHY THIS EXISTS. §I13 measured both halves of the task against a test-retest ceiling:

    term                          ceiling   achieved   captured
    buy flag      corr(Z, .)      0.6623    0.5932     89.6 %
    magnitude     corr(L, .|Z=1)  0.6001    0.4814     80.2 %

**The magnitude term is the only quantity in this project with a measured double-digit
relative gap, and it is the one the project never scored directly.** §1b closed the
classification lever on solid evidence (four classifiers inside 0.002 AUC, converting to RMSLE
at ~0) and that closure stands. But it reached "magnitude is the smaller share of covariance"
and stopped, which is not the same as "magnitude is saturated".

The one prior attack is e0010's hurdle (`P(buy) x E[L|buy]`, -0.00012 = no effect). It used the
same features AND the same L2 loss AND was scored end to end, so it showed that *that*
decomposition does not pay -- not that the term is closed. §I13's instruction is explicit:
score `corr(L, .|Z=1)` FIRST, and only then ask whether an improvement survives recombination.
Scoring end-to-end first is what hid this term for the whole project.

WHAT IS DIFFERENT FROM `run.py`, and it is one thing per arm:
  * the TRAINING POPULATION is restricted to rows whose target window is positive (`y > 0`),
    so the loss stops spending capacity on the 44 % of rows that are structurally zero;
  * the SCORING is conditional: `corr(L, pred)` over validation users with `y_true > 0`.

Everything else is held: the frozen folds, the frozen metric, the same anchors, the same
feature blocks, the same seed, `assert_no_lookahead` imported from `run.py`.

MULTI-ARM BY DESIGN. Feature building dominates cost (~25 min/fold at 665 columns) while a fit
is minutes, so an arm-per-job would spend ~90 % of its wall clock rebuilding identical
matrices. This runner builds each fold's matrices ONCE and trains every arm on them. The arms
are still isolated experiments -- identical features, folds, anchors and seed, differing by
exactly one declared thing each (CLAUDE.md 4.1) -- and each writes its own OOF and its own
`runs/<exp_id>.json`. What it removes is the rebuild, not the isolation.

⚠ A magnitude model is NOT a submission. It predicts an amount conditional on buying, and at
test time we do not know who buys. `rho|Z=1` is a DIAGNOSTIC that says whether the term moves
at all. Turning any gain into RMSLE needs the classifier back (`--combine`), and §I13 warns
that the oracle-path conversion rate has historically been ~0. Both numbers are reported so
the difference stays visible.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime
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


# --------------------------------------------------------------------------------- arms
def rank_gauss(v: np.ndarray) -> np.ndarray:
    """Map to normal scores by rank. Kills the heavy right tail of L among buyers without
    changing the ORDER, so `corr` is affected only through the shape of the residual the
    tree fits -- which is the whole point of the arm."""
    from scipy.special import ndtri
    r = np.argsort(np.argsort(v)).astype(np.float64)
    return ndtri((r + 0.5) / len(v))


def fit_arm(arm: dict, Xtr, ytr_L, Xva, seed: int, base_params: dict, rounds: int):
    """Train one arm and return validation predictions on the L scale.

    Every arm returns a prediction of E[L | x, buyer]; the arms differ ONLY in how that
    estimate is learned (loss geometry / target parametrisation / capacity), never in what
    is being estimated. That is the same discipline IDEAS.md §0 applies to the loss axis.
    """
    import lightgbm as lgb
    p = dict(base_params)
    p["seed"] = seed
    kind = arm.get("kind", "l2")
    y = ytr_L
    inv = None

    if kind == "l2":
        pass
    elif kind == "huber":
        # heavy-tailed L among buyers -> a loss that stops chasing the top tail
        p["objective"] = "huber"
        p["alpha"] = float(arm.get("alpha", 1.0))
    elif kind == "l1":
        p["objective"] = "regression_l1"
    elif kind == "rankgauss":
        # monotone reparametrisation of the target; predictions are mapped back by the
        # inverse empirical map so the output is on the L scale and rho is comparable
        srt = np.sort(ytr_L)
        y = rank_gauss(ytr_L)
        ys = np.sort(y)
        inv = lambda q: np.interp(q, ys, srt)          # noqa: E731
    elif kind == "capacity":
        p["num_leaves"] = int(arm.get("num_leaves", 255))
        p["min_data_in_leaf"] = int(arm.get("min_data_in_leaf", 40))
    elif kind == "tuned":
        p.update(arm.get("params", {}))
    else:
        raise ValueError(f"unknown arm kind {kind!r}")

    n = int(arm.get("rounds", rounds))
    m = lgb.train(p, lgb.Dataset(Xtr, y), num_boost_round=n,
                  callbacks=[lgb.log_evaluation(0)])
    pred = m.predict(Xva)
    return (inv(pred) if inv is not None else pred), m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--screen", action="store_true", help="last 2 folds only (tier=screen)")
    ap.add_argument("--max-train-anchors", type=int, default=0,
                    help="SMOKE ONLY: cap fit anchors; requires --no-log")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()
    if args.max_train_anchors and not args.no_log:
        raise SystemExit("--max-train-anchors changes the training set; pass --no-log too.")

    cfg = yaml.safe_load(Path(args.config).read_text())
    arms = cfg["arms"]
    t0 = time.time()
    log(f"\n=== MAGNITUDE SWEEP : {len(arms)} arms ===")
    for a in arms:
        log(f"    {a['exp_id']:8s} {a['change']}")

    spec = json.loads((ROOT / "data" / "fold_spec.json").read_text())
    folds = pl.read_parquet(ROOT / "data" / "folds.parquet")
    fold_ids = sorted(folds["fold_id"].unique().to_list())
    if args.screen:
        fold_ids = fold_ids[-2:]
    tier = "screen" if args.screen else cfg.get("tier", "confirm")

    p = Panel()
    base_params = dict(cfg["lgb_params"])
    ROUNDS = int(cfg["fixed_rounds"])

    # per-arm accumulators
    acc = {a["exp_id"]: {"oof": [], "rho_c": [], "rmsle_c": []} for a in arms}
    base_rho_c = []                                   # e0049-style all-user L2 reference

    for k in fold_ids:
        fs = spec["folds"][k]
        va = date.fromisoformat(fs["valid_anchor"])
        vai = p.idx(va)
        tr_anchors = [date.fromisoformat(x) for x in fs["train_anchors"]]
        if args.max_train_anchors:
            tr_anchors = tr_anchors[-args.max_train_anchors:]

        Xtr, ytr = [], []
        for a in tr_anchors:
            ai = p.idx(a)
            keep = p.active_in(ai - 29, ai)
            X, names = build(p, ai, keep, cfg["feature_blocks"])
            Xtr.append(X); ytr.append(p.target(ai)[keep])
        Xtr = np.concatenate(Xtr); ytr_raw = np.concatenate(ytr)
        log(f"    fold {k}: {len(tr_anchors)} anchors, {Xtr.shape[0]:,} rows x "
            f"{Xtr.shape[1]} feat  [t+{(time.time()-t0)/60:.1f}m]")

        vkeep = p.active_in(vai - 29, vai)
        Xva, names = build(p, vai, vkeep, cfg["feature_blocks"])
        if k == fold_ids[0]:
            assert_no_lookahead(p, vai, Xva, vkeep, cfg["feature_blocks"])
            log(f"    look-ahead check passed; {Xva.shape[1]} features")

        fv = folds.filter(pl.col("fold_id") == k).sort("user_id")
        yva = fv["target"].to_numpy()
        assert np.array_equal(fv["user_id"].to_numpy(), p.users[vkeep]), "fold population drift"
        Lva = np.log1p(yva)
        zva = yva > 0                                  # the scored population for rho|Z=1

        # ---- the one restriction that defines this runner -------------------------------
        buy = ytr_raw > 0
        Xb, Lb = Xtr[buy], np.log1p(ytr_raw[buy])
        log(f"      buyers in train: {buy.sum():,} / {buy.size:,} ({100*buy.mean():.1f}%)  "
            f"| valid buyers {zva.sum():,} / {zva.size:,}")

        # ---- reference: the SAME loss on ALL rows (this is e0049's regime) ---------------
        # Included every fold so the buyers-only comparison is same-session and paired, never
        # against a logged number from another run (BACKLOG: cross-session drift exceeds every
        # effect size being tested here).
        import lightgbm as lgb
        pr = dict(base_params); pr["seed"] = cfg["seed"]
        mref = lgb.train(pr, lgb.Dataset(Xtr, np.log1p(ytr_raw)), num_boost_round=ROUNDS,
                         callbacks=[lgb.log_evaluation(0)])
        pref = mref.predict(Xva)
        rc_ref = float(np.corrcoef(Lva[zva], pref[zva])[0, 1])
        base_rho_c.append(rc_ref)
        log(f"      REFERENCE (all-rows L2)  rho|Z=1 = {rc_ref:.5f}")
        del mref

        for a in arms:
            pv, _ = fit_arm(a, Xb, Lb, Xva, cfg["seed"], base_params, ROUNDS)
            rc = float(np.corrcoef(Lva[zva], pv[zva])[0, 1])
            # RMSLE restricted to buyers -- reported for completeness, NOT the headline:
            # it is not the competition metric and a magnitude model is not a submission.
            rm = rmsle(yva[zva], np.maximum(np.expm1(pv[zva]), 0.0))
            acc[a["exp_id"]]["rho_c"].append(rc)
            acc[a["exp_id"]]["rmsle_c"].append(rm)
            acc[a["exp_id"]]["oof"].append(pl.DataFrame({
                "fold_id": np.full(yva.size, k, np.int8),
                "user_id": fv["user_id"].to_numpy(),
                "y_true": yva, "y_pred": np.maximum(np.expm1(pv), 0.0),
                "is_buyer": zva}))
            log(f"      {a['exp_id']:8s} rho|Z=1 = {rc:.5f}  ({rc - rc_ref:+.5f} vs ref)  "
                f"rmsle_buyers {rm:.5f}")
        del Xtr, Xb, Xva

    # ------------------------------------------------------------------ report + log rows
    ref = np.array(base_rho_c)
    log(f"\n  CEILING (IDEAS.md §I13, test-retest): 0.6001")
    log(f"  REFERENCE all-rows L2: rho|Z=1 = {ref.mean():.5f} +/- {ref.std():.5f}  "
        f"folds {np.round(ref,5).tolist()}")
    log(f"\n  {'arm':10s} {'rho|Z=1':>9s} {'std':>8s} {'d vs ref':>9s} {'wins':>6s} {'captured':>9s}")
    rows = []
    for a in arms:
        e = a["exp_id"]; rc = np.array(acc[e]["rho_c"])
        d = rc.mean() - ref.mean()
        wins = int((rc > ref).sum())
        log(f"  {e:10s} {rc.mean():>9.5f} {rc.std():>8.5f} {d:>+9.5f} {wins:>4d}/{len(rc)} "
            f"{rc.mean()/0.6001:>8.1%}")
        rows.append((a, rc, d, wins))

    if args.no_log:
        log("\n  --no-log: nothing written (smoke)")
        return

    (ROOT / "oof").mkdir(exist_ok=True)
    rd = ROOT / "runs"; rd.mkdir(exist_ok=True)
    runtime = (time.time() - t0) / 60
    for a, rc, d, wins in rows:
        e = a["exp_id"]
        pl.concat(acc[e]["oof"]).write_parquet(ROOT / "oof" / f"{e}.parquet")
        rmc = np.array(acc[e]["rmsle_c"])
        row = {
            "exp_id": e, "parent_id": a.get("parent_id", cfg["parent_id"]),
            "date": datetime.now().isoformat(timespec="seconds"),
            "approach": "gbdt_magnitude", "change": a["change"], "tier": tier,
            "n_features": len(names), "cv_mean": round(float(rmc.mean()), 5),
            "cv_std": round(float(rmc.std()), 5),
            "folds": json.dumps([round(float(x), 5) for x in rmc]),
            "delta": round(float(d), 5),
            # `significant` here is vs the SAME-SESSION all-rows reference on rho|Z=1 --
            # NOT the naive floor run.py uses (REVIEW_NOTES.md A1). Different column meaning
            # in this runner, stated so nobody pools the two.
            "significant": "yes" if wins >= 4 else "no",
            "lb": "", "runtime_min": round(runtime, 1), "seed": cfg["seed"],
            "config": args.config, "verdict": a.get("verdict", ""),
            "gini_pred": 0.0, "total_rel_err": 0.0,
            "notes": (f"rho|Z=1 {rc.mean():.5f} vs same-session all-rows ref "
                      f"{ref.mean():.5f} (d {d:+.5f}, {wins}/{len(rc)} folds); "
                      f"ceiling 0.6001; cv_mean col = RMSLE among BUYERS ONLY, not the "
                      f"competition metric. " + a.get("notes", "")),
            "best_iters": json.dumps([int(a.get("rounds", ROUNDS))] * len(rc)),
        }
        (rd / f"{e}.json").write_text(json.dumps(row, indent=2))
    try:
        import subprocess
        subprocess.run([sys.executable, str(ROOT / "src" / "collect.py")], check=False)
    except Exception:
        pass
    log(f"\n  wrote {len(rows)} run rows -> experiments.csv   runtime {runtime:.1f} min")


if __name__ == "__main__":
    main()
