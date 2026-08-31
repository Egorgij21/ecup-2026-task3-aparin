#!/usr/bin/env python
"""
The discrimination ceiling: can a DEDICATED classifier beat the regressor's implicit AUC?

    python src/run_clf.py --config configs/e0049_nomoment.yaml --exp-id e0160

WHY THIS IS THE ONLY QUESTION LEFT.  src/rho_decomp.py partitions Cov(L, M) exactly:

    BETWEEN  (separating buyers from non-buyers)  78.6%
    WITHIN   (ranking magnitude among buyers)     21.4%

    oracle split x our magnitudes   rho 0.66247 -> 0.93260   (+0.271)
    our split    x oracle magnitude rho 0.66247 -> 0.71603   (+0.054)

Classification is worth five times magnitude in rho and 7.7x in RMSLE.  Our current AUC on the
`y > 0` event is 0.84322 -- produced by a model that was never asked to classify, only to
minimise L2 on log1p.  If a model trained for the binary task beats that, the largest prize in
the project is open.  If it ties, the ceiling is a property of the data and the 0.271 is
unreachable; that is a result too, and it says stop optimising.

The nearest prior evidence is e0010 (two-part hurdle, "no effect"), but that tested a
DECOMPOSITION -- P(buy) x E[L|buy] against a single regressor -- not whether discrimination
itself can be improved.  A composition can lose while its classifier is better.

DESIGN.  Identical protocol to src/run.py so the AUC is measured on exactly the folds and
populations every other experiment uses: frozen folds, same fit anchors, same feature blocks.
AUC is reported at several round counts from ONE fit, because a fixed 178 rounds was tuned for
the regression objective and a wrong budget is precisely the confound that made the earlier
feature experiments unreadable (EXPERIMENTS.md §3b.1).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data import Panel                     # noqa: E402
from features import build                 # noqa: E402
from rho_decomp import auc                 # noqa: E402

ROUNDS = (100, 178, 300, 500, 800)


def log(m: str) -> None:
    print(m, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/e0049_nomoment.yaml")
    ap.add_argument("--exp-id", default="e0160")
    ap.add_argument("--baseline-oof", nargs="+", default=["e0049", "e0064"],
                    help="regression OOF whose implicit AUC is the number to beat")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    t0 = time.time()
    import lightgbm as lgb

    spec = json.loads((ROOT / "data" / "fold_spec.json").read_text())
    folds = pl.read_parquet(ROOT / "data" / "folds.parquet")
    p = Panel()
    log(f"\n=== {args.exp_id}: dedicated y>0 classifier on {cfg['feature_blocks']} ===")

    oof_rows, per_fold = [], {r: [] for r in ROUNDS}
    for k in sorted(folds["fold_id"].unique().to_list()):
        fs = spec["folds"][k]
        va = date.fromisoformat(fs["valid_anchor"]); vai = p.idx(va)
        tr_anchors = [date.fromisoformat(x) for x in fs["train_anchors"]]

        Xtr, ytr = [], []
        for a in tr_anchors:
            ai = p.idx(a)
            keep = p.active_in(ai - 29, ai)
            X, names = build(p, ai, keep, cfg["feature_blocks"])
            Xtr.append(X); ytr.append(p.target(ai)[keep])
        Xtr = np.concatenate(Xtr); ztr = (np.concatenate(ytr) > 0).astype(np.int8)

        vkeep = p.active_in(vai - 29, vai)
        Xva, names = build(p, vai, vkeep, cfg["feature_blocks"])
        fv = folds.filter(pl.col("fold_id") == k).sort("user_id")
        assert np.array_equal(fv["user_id"].to_numpy(), p.users[vkeep]), "fold population drift"
        zva = (fv["target"].to_numpy() > 0).astype(np.int8)

        params = {**cfg["lgb_params"], "objective": "binary", "metric": "auc",
                  "seed": cfg["seed"]}
        params.pop("objective_alias", None)
        m = lgb.train(params, lgb.Dataset(Xtr, ztr, feature_name=names),
                      num_boost_round=max(ROUNDS), callbacks=[lgb.log_evaluation(0)])
        line = []
        for r in ROUNDS:
            s = m.predict(Xva, num_iteration=r)
            a = auc(s, zva.astype(float))
            per_fold[r].append(a); line.append(f"{r}:{a:.5f}")
        best_r = max(ROUNDS, key=lambda r: per_fold[r][-1])
        oof_rows.append(pl.DataFrame({
            "fold_id": np.full(zva.size, k, np.int8),
            "user_id": fv["user_id"].to_numpy(),
            "z_true": zva.astype(np.int8),
            "p_clf": m.predict(Xva, num_iteration=800).astype(np.float32)}))
        log(f"    fold {k} {va}  n={zva.size:>7,}  P(y>0)={zva.mean():.4f}  AUC  " + "  ".join(line))

    oof = pl.concat(oof_rows)
    (ROOT / "oof").mkdir(exist_ok=True)
    oof.write_parquet(ROOT / "oof" / f"{args.exp_id}_clf.parquet")

    log(f"\n  {'rounds':>7s} {'AUC mean':>10s} {'per fold':>10s}")
    for r in ROUNDS:
        v = np.array(per_fold[r])
        log(f"  {r:>7d} {v.mean():>10.5f}   {[round(x, 5) for x in v]}")
    best_r = max(ROUNDS, key=lambda r: np.mean(per_fold[r]))
    best = float(np.mean(per_fold[best_r]))

    # the number to beat: the implicit AUC of the regression models, same folds, same rows
    import pyarrow.parquet as pq
    Ms = []
    for e in args.baseline_oof:
        t = pq.read_table(ROOT / "oof" / f"{e}.parquet").to_pydict()
        o = np.lexsort((np.array(t["user_id"]), np.array(t["fold_id"])))
        Ms.append(np.log1p(np.array(t["y_pred"])[o]))
        y = np.array(t["y_true"])[o]; fo = np.array(t["fold_id"])[o]
    Mreg = np.mean(Ms, axis=0)
    z = (y > 0).astype(float)
    reg_pf = [auc(Mreg[fo == k], z[fo == k]) for k in np.unique(fo)]
    reg = float(np.mean(reg_pf))

    log(f"\n  ===================== THE ANSWER =====================")
    log(f"  regression models' IMPLICIT AUC  ({'+'.join(args.baseline_oof)})  {reg:.5f}   "
        f"{[round(x, 5) for x in reg_pf]}")
    log(f"  dedicated classifier, best rounds ({best_r})                  {best:.5f}   "
        f"{[round(x, 5) for x in per_fold[best_r]]}")
    log(f"  delta                                                        {best - reg:+.5f}   "
        f"wins {sum(1 for a, b in zip(per_fold[best_r], reg_pf) if a > b)}/5 folds")
    log(f"\n  rho_decomp measured d(rho)/d(AUC) ~ 1.2 along the ORACLE path (an upper bound),")
    log(f"  and d(RMSLE)/d(rho) = -2.30 on the leaderboard. Treat any gain as an upper bound")
    log(f"  until it is realised end-to-end, not as a projected score.")
    log(f"  runtime {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
