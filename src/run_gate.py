#!/usr/bin/env python
"""
SCALE-PENALTY GATE for the tabular-foundation-model direction (IDEAS.md §I7 / §I17), plus a
transductive pseudo-labelling arm (IDEAS.md §I18). No feature changes: every arm runs e0049's
exact 665 columns on the frozen folds against a SAME-SESSION reference refitted in every fold.

    python src/run_gate.py --config configs/e0320_gate.yaml

WHY. TabPFN-3 (arXiv 2605.13986) is validated to 1M rows x 200 features or 100k rows x 2,000;
TabICLv2 (2602.11139) to 1M x 500. Our fold training sets are 1.6M-5.2M rows x 665 features, so
any in-context learner sees a SUBSET of the evidence the LightGBM sees. Before installing
anything, measure what that subset costs the model we already have: a LightGBM trained on the
TFM's input regime (rows subsampled, features cut) scored on the full validation population.
A TFM must first make that penalty back from its inductive bias alone before it can even reach
the "decorrelated but weaker" quadrant EXPERIMENTS.md §1c mapped -- and §1f says a new blend
member needs rho_partial >= 0.04 against the champion.

Arm kinds (each differs from the reference by exactly ONE declared thing, CLAUDE.md §4.1):
  subsample   rows: N        uniform random rows of the fold's training matrix, fixed seed
  topk        feats: K       keep the K highest-gain features of the fold's REFERENCE model
                             (importance from the training set only -- never the fold)
  pseudo      weight: w      self-training: add the validation rows with the reference model's
                             own log-space prediction as label, weight w, refit, re-predict.
                             Transductive and label-free; at submission time the analogue is
                             the 250k test rows at the test anchor.
  params      params: {...}  hyperparameter override (used to give small-N arms a fair config)
Arms may combine `rows` + `feats` (the TabPFN-3 1M x 200 regime is one arm); that is still one
declared regime, not a bundle of hypotheses.

All arms share one feature build per fold (building dominates cost at ~25 min/fold). OOF files
and runs/<exp_id>.json rows are written per arm exactly as run.py writes them, so every number
here is comparable to every other row in experiments.csv.
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--screen", action="store_true", help="last 2 folds only (tier=screen)")
    ap.add_argument("--max-train-anchors", type=int, default=0)
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()
    if args.max_train_anchors and not args.no_log:
        raise SystemExit("--max-train-anchors changes the training set; pass --no-log too.")

    cfg = yaml.safe_load(Path(args.config).read_text())
    arms = cfg["arms"]
    t0 = time.time()
    log(f"\n=== GATE SWEEP : {len(arms)} arms ===")
    for a in arms:
        log(f"    {a['exp_id']:8s} {a['change']}")

    spec = json.loads((ROOT / "data" / "fold_spec.json").read_text())
    folds = pl.read_parquet(ROOT / "data" / "folds.parquet")
    fold_ids = sorted(folds["fold_id"].unique().to_list())
    if args.screen:
        fold_ids = fold_ids[-2:]
    tier = "screen" if args.screen else cfg.get("tier", "confirm")

    p = Panel()
    if cfg.get("feature_cache"):
        import features as _feat
        _feat.enable_cache(True, force=bool(cfg.get("feature_cache_force")))
        log(f"    feature cache {'ON' if _feat.CACHE_ENABLED else 'REFUSED'}  gen={_feat._code_hash()}")
    import lightgbm as lgb
    base_params = dict(cfg["lgb_params"])
    ROUNDS = int(cfg["fixed_rounds"])
    SEED = int(cfg["seed"])

    acc = {a["exp_id"]: {"oof": [], "pf": [], "meta": []} for a in arms}
    ref_pf, naive_pf = [], []

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
        ytr_L = np.log1p(ytr_raw)
        n_tr = Xtr.shape[0]
        log(f"    fold {k}: {len(tr_anchors)} anchors, {n_tr:,} rows x {Xtr.shape[1]} feat  "
            f"[t+{(time.time() - t0) / 60:.1f}m]")

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

        # same-session reference: e0049's exact regime, refitted here. Its gain importance
        # (training rows only) defines the `topk` feature subsets; its log-space prediction on
        # the validation rows is the `pseudo` label. Neither touches y_valid.
        pr = dict(base_params); pr["seed"] = SEED
        mref = lgb.train(pr, lgb.Dataset(Xtr, ytr_L, feature_name=names),
                         num_boost_round=ROUNDS, callbacks=[lgb.log_evaluation(0)])
        ref_log = mref.predict(Xva)
        pref = np.maximum(np.expm1(ref_log), 0.0)
        s_ref = rmsle(yva, pref); ref_pf.append(s_ref)
        gain = mref.feature_importance("gain")
        order = np.argsort(-gain)
        log(f"      REFERENCE (e0049 regime)  rmsle = {s_ref:.5f}   "
            f"[top-5 gain: {', '.join(names[i] for i in order[:5])}]")
        del mref

        rng = np.random.default_rng(SEED + 17 * k)
        for a in arms:
            pr = dict(base_params); pr.update(a.get("params", {})); pr["seed"] = SEED
            n_rounds = int(a.get("rounds", ROUNDS))
            X_fit, y_fit, w_fit, X_pred, nm = Xtr, ytr_L, None, Xva, names
            meta = {}
            # --- rows
            if a.get("rows"):
                N = int(a["rows"])
                if N < n_tr:
                    idx = np.sort(rng.choice(n_tr, size=N, replace=False))
                    X_fit, y_fit = Xtr[idx], ytr_L[idx]
                meta["rows"] = int(X_fit.shape[0])
            # --- features (gain-ranked on the reference model's TRAINING fit)
            if a.get("feats"):
                K = int(a["feats"])
                sel = np.sort(order[:K])
                X_fit, X_pred = X_fit[:, sel], Xva[:, sel]
                nm = [names[i] for i in sel]
                meta["feats"] = int(K)
            # --- transductive self-training on the validation rows
            if a.get("kind") == "pseudo":
                w = float(a["weight"])
                X_fit = np.concatenate([X_fit, X_pred])
                y_fit = np.concatenate([y_fit, ref_log])
                w_fit = np.concatenate([np.ones(n_tr), np.full(X_pred.shape[0], w)])
                meta["pseudo_weight"] = w
            m = lgb.train(pr, lgb.Dataset(X_fit, y_fit, weight=w_fit, feature_name=nm),
                          num_boost_round=n_rounds, callbacks=[lgb.log_evaluation(0)])
            pv_log = m.predict(X_pred)
            pv = np.maximum(np.expm1(pv_log), 0.0)
            s = rmsle(yva, pv)
            # corr with the reference in log space: a subsample arm that is weaker but highly
            # correlated is §1c's worthless quadrant; report it alongside the delta.
            r_ref = float(np.corrcoef(pv_log, ref_log)[0, 1])
            acc[a["exp_id"]]["pf"].append(s)
            acc[a["exp_id"]]["meta"].append({**meta, "r_vs_ref": round(r_ref, 5)})
            acc[a["exp_id"]]["oof"].append(pl.DataFrame({
                "fold_id": np.full(yva.size, k, np.int8),
                "anchor_date": pl.Series("anchor_date", [va] * yva.size, dtype=pl.Date),
                "user_id": fv["user_id"].to_numpy(),
                "y_true": yva, "y_pred": pv, "y_naive": naive}))
            log(f"      {a['exp_id']:8s} rmsle = {s:.5f}  ({s - s_ref:+.5f} vs ref)  "
                f"r_vs_ref={r_ref:.4f}  {meta}")
            del m, X_fit, y_fit
        del Xtr, Xva

    ref = np.array(ref_pf); nv = np.array(naive_pf)
    log(f"\n  REFERENCE (same-session e0049 regime) = {ref.mean():.5f} +/- {ref.std():.5f}  "
        f"folds {np.round(ref, 5).tolist()}")
    log(f"  naive geo3 = {nv.mean():.5f}")
    log(f"\n  {'arm':10s} {'cv_mean':>9s} {'std':>8s} {'d vs ref':>10s} {'wins':>6s} "
        f"{'last fold':>10s} {'r_vs_ref':>9s}")
    rows = []
    for a in arms:
        e = a["exp_id"]; pf = np.array(acc[e]["pf"])
        d = pf.mean() - ref.mean(); wins = int((pf < ref).sum())
        rr = float(np.mean([mm["r_vs_ref"] for mm in acc[e]["meta"]]))
        sig = "**" if (wins >= 4 and d < 0) or abs(d) > 2 * 0.00009 else ""
        log(f"  {e:10s} {pf.mean():>9.5f} {pf.std():>8.5f} {d:>+10.5f} {wins:>4d}/{len(pf)} "
            f"{pf[-1]:>10.5f} {rr:>9.4f} {sig}")
        rows.append((a, pf, d, wins, rr))
    log("\n  ** = >=4/5 folds AND better, or |d| > 2*sigma_noise(0.00009).")

    if args.no_log:
        log("\n  --no-log: nothing written (smoke)")
        return

    (ROOT / "oof").mkdir(exist_ok=True)
    rd = ROOT / "runs"; rd.mkdir(exist_ok=True)
    runtime = (time.time() - t0) / 60
    for a, pf, d, wins, rr in rows:
        e = a["exp_id"]
        o = pl.concat(acc[e]["oof"]); o.write_parquet(ROOT / "oof" / f"{e}.parquet")
        agg = score_all(o["y_true"].to_numpy(), o["y_pred"].to_numpy())
        row = {
            "exp_id": e, "parent_id": a.get("parent_id", cfg["parent_id"]),
            "date": datetime.now().isoformat(timespec="seconds"),
            "approach": "gbdt_direct", "change": a["change"], "tier": tier,
            "n_features": int(a.get("feats", len(names))),
            "cv_mean": round(float(pf.mean()), 5),
            "cv_std": round(float(pf.std()), 5),
            "folds": json.dumps([round(float(x), 5) for x in pf]),
            "delta": round(float(pf.mean() - nv.mean()), 5),   # vs naive, as run.py does
            "significant": "yes" if wins >= 4 or abs(d) > 2 * 0.00009 else "no",
            "lb": "", "runtime_min": round(runtime, 1), "seed": SEED,
            "config": args.config, "verdict": a.get("verdict", ""),
            "gini_pred": round(agg["gini_pred"], 4),
            "total_rel_err": round(agg["total_rel_err"], 4),
            "notes": (f"vs SAME-SESSION refit reference {ref.mean():.5f}: d {d:+.5f}, "
                      f"{wins}/{len(pf)} folds, mean log-corr with reference {rr:.4f}. "
                      + a.get("notes", "")),
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
