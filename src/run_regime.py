#!/usr/bin/env python
"""
TRAINING-REGIME sweep on the full population: sample weights, seed bagging, anchor decay.

    python src/run_regime.py --config configs/e0260_regime.yaml

WHY THESE ARMS. `EXPERIMENTS.md` §1q closed the information axis by direct measurement (no
candidate clears incremental R^2 0.0002 against the OOF residual, and 22 have failed), so this
runner changes no features at all -- every arm uses e0049's exact 665 columns. What it varies
is the training regime, where the record still contains items marked untested in the repo's own
words:

  * **Multi-seed averaging on the GBDT side.** §4 note 3: "done for `nn_seq` (-0.00104 for the
    seq family alone) ... **Never run on the GBDT side**; sigma_noise 0.00009 there so the
    ceiling is small, but it is nearly free." That is an explicit open item, and it is the one
    arm here with a *guaranteed* sign -- averaging seeds cannot raise variance.
  * **Recency weighting of the training anchors.** `BACKLOG` Band B, e0012/B3, never run.
    e0070-e0073 tested anchor TRUNCATION (`max_train_anchors` 6/10/14/18) and found the surface
    flat, but truncation is a step function on anchor age; a decay is a different intervention
    and the flatness of one does not imply the flatness of the other.
  * **Buyer up/down-weighting.** §1q measures the task as 81 % classification variance, and
    §I13 measures the magnitude term as the weaker of the two halves. L2 on all rows implicitly
    weights the two terms by their row counts (44 % zeros). Re-weighting is the cheapest way to
    ask whether that implicit split is the right one -- and unlike a hurdle it keeps ONE model,
    so it cannot lose to the composition error that killed e0010.

All arms share one feature build per fold (building dominates cost); each is still an isolated
experiment against the SAME-SESSION reference re-fitted inside every fold, differing by exactly
one declared thing (README.md). The reference is re-fitted rather than read from e0049's
logged 1.76551 because cross-session drift has been measured at +0.00027..+0.00046 for configs
with zero changes -- larger than several of the effects being tested.

Scoring is the real competition metric on the real folds, so unlike `run_magnitude.py` these
numbers ARE comparable to every other row in experiments.csv.
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
from run import assert_no_lookahead             # noqa: E402


def log(m: str) -> None:
    print(m, flush=True)


def arm_weights(arm: dict, ytr_raw: np.ndarray, anchor_age: np.ndarray) -> np.ndarray | None:
    """Per-row sample weights for one arm, or None for the unweighted default."""
    kind = arm.get("kind", "plain")
    if kind == "buyer_weight":
        w = np.ones(ytr_raw.size, np.float64)
        w[ytr_raw > 0] = float(arm["w_buy"])
        return w
    if kind == "anchor_decay":
        # half-life in days on the anchor's age relative to the most recent fit anchor
        hl = float(arm["half_life_days"])
        return 0.5 ** (anchor_age / hl)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--screen", action="store_true")
    ap.add_argument("--max-train-anchors", type=int, default=0)
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()
    if args.max_train_anchors and not args.no_log:
        raise SystemExit("--max-train-anchors changes the training set; pass --no-log too.")

    cfg = yaml.safe_load(Path(args.config).read_text())
    arms = cfg["arms"]
    t0 = time.time()
    log(f"\n=== REGIME SWEEP : {len(arms)} arms ===")
    for a in arms:
        log(f"    {a['exp_id']:8s} {a['change']}")

    spec = json.loads((ROOT / "data" / "fold_spec.json").read_text())
    folds = pl.read_parquet(ROOT / "data" / "folds.parquet")
    fold_ids = sorted(folds["fold_id"].unique().to_list())
    if args.screen:
        fold_ids = fold_ids[-2:]
    tier = "screen" if args.screen else cfg.get("tier", "confirm")

    p = Panel()
    import lightgbm as lgb
    base_params = dict(cfg["lgb_params"])
    ROUNDS = int(cfg["fixed_rounds"])

    acc = {a["exp_id"]: {"oof": [], "pf": []} for a in arms}
    ref_pf, naive_pf = [], []

    for k in fold_ids:
        fs = spec["folds"][k]
        va = date.fromisoformat(fs["valid_anchor"])
        vai = p.idx(va)
        tr_anchors = [date.fromisoformat(x) for x in fs["train_anchors"]]
        if args.max_train_anchors:
            tr_anchors = tr_anchors[-args.max_train_anchors:]

        Xtr, ytr, age = [], [], []
        last = tr_anchors[-1]
        for a in tr_anchors:
            ai = p.idx(a)
            keep = p.active_in(ai - 29, ai)
            X, names = build(p, ai, keep, cfg["feature_blocks"])
            Xtr.append(X); ytr.append(p.target(ai)[keep])
            age.append(np.full(X.shape[0], (last - a).days, np.float64))
        Xtr = np.concatenate(Xtr); ytr_raw = np.concatenate(ytr)
        anchor_age = np.concatenate(age)
        ytr_L = np.log1p(ytr_raw)
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
        naive = np.maximum(Xva[:, names.index("geo3")].astype(np.float64), 0.0)
        naive_pf.append(rmsle(yva, naive))

        # same-session reference: e0049's exact regime, refitted here
        pr = dict(base_params); pr["seed"] = cfg["seed"]
        mref = lgb.train(pr, lgb.Dataset(Xtr, ytr_L), num_boost_round=ROUNDS,
                         callbacks=[lgb.log_evaluation(0)])
        pref = np.maximum(np.expm1(mref.predict(Xva)), 0.0)
        s_ref = rmsle(yva, pref); ref_pf.append(s_ref)
        log(f"      REFERENCE (e0049 regime)  rmsle = {s_ref:.5f}")
        del mref

        for a in arms:
            pr = dict(base_params); pr.update(a.get("params", {}))
            n = int(a.get("rounds", ROUNDS))
            w = arm_weights(a, ytr_raw, anchor_age)
            seeds = a.get("seeds", [cfg["seed"]])
            preds = []
            for sd in seeds:
                pr2 = dict(pr); pr2["seed"] = int(sd)
                # bagging_seed/feature_fraction_seed follow `seed` in LightGBM only if unset;
                # set them explicitly so a seed change actually re-randomises both.
                pr2["bagging_seed"] = int(sd) + 1000
                pr2["feature_fraction_seed"] = int(sd) + 2000
                m = lgb.train(pr2, lgb.Dataset(Xtr, ytr_L, weight=w),
                              num_boost_round=n, callbacks=[lgb.log_evaluation(0)])
                preds.append(m.predict(Xva))
            # average in LOG space: the estimand is E[L|x], so the mean of log-space
            # predictions is the quantity being estimated. Averaging expm1'd values would
            # estimate log1p(E[y]) instead -- the +0.5626 functional error of §1e.
            pv = np.maximum(np.expm1(np.mean(preds, axis=0)), 0.0)
            s = rmsle(yva, pv)
            acc[a["exp_id"]]["pf"].append(s)
            acc[a["exp_id"]]["oof"].append(pl.DataFrame({
                "fold_id": np.full(yva.size, k, np.int8),
                "anchor_date": pl.Series("anchor_date", [va] * yva.size, dtype=pl.Date),
                "user_id": fv["user_id"].to_numpy(),
                "y_true": yva, "y_pred": pv, "y_naive": naive}))
            log(f"      {a['exp_id']:8s} rmsle = {s:.5f}  ({s - s_ref:+.5f} vs ref)"
                + (f"  [{len(seeds)} seeds]" if len(seeds) > 1 else ""))
        del Xtr, Xva

    ref = np.array(ref_pf); nv = np.array(naive_pf)
    log(f"\n  REFERENCE (same-session e0049 regime) = {ref.mean():.5f} +/- {ref.std():.5f}  "
        f"folds {np.round(ref,5).tolist()}")
    log(f"  naive geo3 = {nv.mean():.5f}")
    log(f"\n  {'arm':10s} {'cv_mean':>9s} {'std':>8s} {'d vs ref':>10s} {'wins':>6s} {'last fold':>10s}")
    rows = []
    for a in arms:
        e = a["exp_id"]; pf = np.array(acc[e]["pf"])
        d = pf.mean() - ref.mean(); wins = int((pf < ref).sum())
        sig = "**" if (wins >= 4 and d < 0) or abs(d) > 2 * 0.00009 else ""
        log(f"  {e:10s} {pf.mean():>9.5f} {pf.std():>8.5f} {d:>+10.5f} {wins:>4d}/{len(pf)} "
            f"{pf[-1]:>10.5f} {sig}")
        rows.append((a, pf, d, wins))
    log("\n  ** = >=4/5 folds AND better, or |d| > 2*sigma_noise(0.00009). Per README.md a")
    log("  sub-2sigma delta is `no effect` regardless of sign -- e0060 flipped sign on LB and")
    log("  cost 0.0005 by being promoted on 0.4 sigma.")

    if args.no_log:
        log("\n  --no-log: nothing written (smoke)")
        return

    (ROOT / "oof").mkdir(exist_ok=True)
    rd = ROOT / "runs"; rd.mkdir(exist_ok=True)
    runtime = (time.time() - t0) / 60
    for a, pf, d, wins in rows:
        e = a["exp_id"]
        o = pl.concat(acc[e]["oof"]); o.write_parquet(ROOT / "oof" / f"{e}.parquet")
        agg = score_all(o["y_true"].to_numpy(), o["y_pred"].to_numpy())
        row = {
            "exp_id": e, "parent_id": a.get("parent_id", cfg["parent_id"]),
            "date": datetime.now().isoformat(timespec="seconds"),
            "approach": "gbdt_direct", "change": a["change"], "tier": tier,
            "n_features": len(names), "cv_mean": round(float(pf.mean()), 5),
            "cv_std": round(float(pf.std()), 5),
            "folds": json.dumps([round(float(x), 5) for x in pf]),
            "delta": round(float(pf.mean() - nv.mean()), 5),   # vs naive, as run.py does
            "significant": "yes" if wins >= 4 or abs(d) > 2 * 0.00009 else "no",
            "lb": "", "runtime_min": round(runtime, 1), "seed": cfg["seed"],
            "config": args.config, "verdict": a.get("verdict", ""),
            "gini_pred": round(agg["gini_pred"], 4),
            "total_rel_err": round(agg["total_rel_err"], 4),
            "notes": (f"vs SAME-SESSION refit reference {ref.mean():.5f}: d {d:+.5f}, "
                      f"{wins}/{len(pf)} folds. " + a.get("notes", "")),
            "best_iters": json.dumps([int(a.get("rounds", ROUNDS))] * len(pf)),
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
