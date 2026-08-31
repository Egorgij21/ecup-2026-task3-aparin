#!/usr/bin/env python
"""
TabICLv2 / TabDPT — a tabular FOUNDATION MODEL on the frozen folds, as a blend member.

    python src/run_tabicl.py --config configs/e0310_tabicl.yaml

WHY THIS IS NOT THE §I17 GATE AGAIN. The gate (e0281-e0286) trained *our LightGBM* under the
row/feature budget a TFM would get, and priced the constraint: 50k rows costs +0.01718 RMSLE,
100k costs +0.00887. It concluded "every arm is a weaker near-twin". That conclusion is about
QUALITY and it is correct — but the blend does not buy quality, it buys **disagreement**, and
the gate could not measure disagreement because every arm was still LightGBM.

The requirement, computed against the current champion (rho 0.66335, bar rho_partial 0.04073):

    if r = 0.95  the member needs rho_B 0.63970   <- BELOW the 100k gate arm's 0.65707
    if r = 0.98  the member needs rho_B 0.65615   <- still below it
    if r = 0.99  the member needs rho_B 0.66102
    if r = 0.995 the member needs rho_B 0.66308

**A TFM that is WEAKER than our handicapped LightGBM can still clear the bar if its errors are
genuinely different.** In-context learning is a different function class from gradient
boosting — no split-finding, no greedy loss descent — so it is the one candidate with a
mechanism to land below the r ~ 0.998 floor that every GBDT variant has hit (six families,
three libraries, EXPERIMENTS.md §9d). That is what this measures and the gate did not.

LICENCE. TabPFN's weights are non-commercial and unclear for an Ozon-hosted contest, so this
runner targets **TabICLv2 (BSD-3)** by default and **TabDPT (Apache-2.0)** via `--backend`.
Neither has a licence obstacle to using its predictions in a submission.

PROTOCOL. Frozen folds, frozen metric, the same population rule, OOF written to
oof/<exp_id>.parquet in the identical schema as every other member so it drops straight into
the admissibility and recombination tooling. The context subsample is drawn with a FIXED seed
per fold (rule 9) and is the *only* thing that differs from how a GBDT member sees the data.
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


def fit_predict(backend: str, Xtr, ytr, Xva, device: str, batch: int, seed: int):
    """Return validation predictions on the log1p scale."""
    if backend == "tabicl":
        from tabicl import TabICLRegressor              # BSD-3
        m = TabICLRegressor(device=device, random_state=seed)
        m.fit(Xtr, ytr)
        out = np.empty(len(Xva), np.float64)
        for i in range(0, len(Xva), batch):
            out[i:i + batch] = m.predict(Xva[i:i + batch])
        return out
    if backend == "tabdpt":
        from tabdpt import TabDPTRegressor              # Apache-2.0
        m = TabDPTRegressor(device=device)
        m.fit(Xtr, ytr)
        out = np.empty(len(Xva), np.float64)
        for i in range(0, len(Xva), batch):
            out[i:i + batch] = m.predict(Xva[i:i + batch])
        return out
    raise ValueError(f"unknown backend {backend!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--backend", default="", help="override the config's backend")
    ap.add_argument("--screen", action="store_true", help="last 2 folds only")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    exp_id = cfg["exp_id"]
    backend = args.backend or cfg["backend"]
    n_ctx = int(cfg["context_rows"])
    n_feat = int(cfg.get("top_features", 0))
    device = cfg.get("device", "cuda")
    batch = int(cfg.get("predict_batch", 10000))
    t0 = time.time()
    log(f"\n=== {exp_id} : {cfg.get('change', '(no change string)')} ===")
    log(f"    backend={backend}  context_rows={n_ctx:,}  top_features={n_feat or 'all'}  "
        f"device={device}")

    spec = json.loads((ROOT / "data" / "fold_spec.json").read_text())
    folds = pl.read_parquet(ROOT / "data" / "folds.parquet")
    fold_ids = sorted(folds["fold_id"].unique().to_list())
    if args.screen:
        fold_ids = fold_ids[-2:]

    p = Panel()
    # Feature ranking, if the context is feature-limited. Taken from the null-importance
    # whitelist that already exists, so the choice is not made on this fold's validation data.
    keep_names = None
    if n_feat:
        wl = ROOT / "configs" / "whitelist_top400.json"
        if wl.exists():
            keep_names = set(json.loads(wl.read_text())[:n_feat])
            log(f"    feature subset from {wl.name}: {len(keep_names)} names")

    oof, per_fold, per_fold_naive = [], [], []
    for k in fold_ids:
        fs = spec["folds"][k]
        va = date.fromisoformat(fs["valid_anchor"])
        vai = p.idx(va)
        tr_anchors = [date.fromisoformat(x) for x in fs["train_anchors"]]

        Xtr, ytr = [], []
        for a in tr_anchors:
            ai = p.idx(a)
            keep = p.active_in(ai - 29, ai)
            Xb, names = build(p, ai, keep, cfg["feature_blocks"])
            Xtr.append(Xb); ytr.append(p.target(ai)[keep])
        Xtr = np.concatenate(Xtr); ytr_raw = np.concatenate(ytr)
        ytr_L = np.log1p(ytr_raw)

        vkeep = p.active_in(vai - 29, vai)
        Xva, names = build(p, vai, vkeep, cfg["feature_blocks"])
        if k == fold_ids[0]:
            assert_no_lookahead(p, vai, Xva, vkeep, cfg["feature_blocks"])
            log(f"    look-ahead check passed; {Xva.shape[1]} features")

        if keep_names:
            sel = [i for i, n in enumerate(names) if n in keep_names]
            Xtr = Xtr[:, sel]; Xva = Xva[:, sel]; names = [names[i] for i in sel]

        # context subsample -- the TFM's binding constraint, seeded per fold (rule 9)
        rng = np.random.default_rng(1000 + k)
        idx = rng.choice(len(Xtr), size=min(n_ctx, len(Xtr)), replace=False)
        Xc, yc = Xtr[idx].astype(np.float32), ytr_L[idx].astype(np.float32)
        log(f"    fold {k}: context {Xc.shape[0]:,} x {Xc.shape[1]} "
            f"(from {Xtr.shape[0]:,})   valid {Xva.shape[0]:,}  [t+{(time.time()-t0)/60:.1f}m]")
        del Xtr

        pred_L = fit_predict(backend, Xc, yc, Xva.astype(np.float32), device, batch,
                             int(cfg["seed"]))
        pred = np.maximum(np.expm1(pred_L), 0.0)

        fv = folds.filter(pl.col("fold_id") == k).sort("user_id")
        yva = fv["target"].to_numpy()
        assert np.array_equal(fv["user_id"].to_numpy(), p.users[vkeep]), "fold population drift"
        naive = np.maximum(Xva[:, names.index("geo3")].astype(np.float64), 0.0) \
            if "geo3" in names else np.zeros_like(yva, dtype=np.float64)

        s, sn = rmsle(yva, pred), rmsle(yva, naive)
        per_fold.append(s); per_fold_naive.append(sn)
        rho = float(np.corrcoef(np.log1p(yva), np.log1p(pred))[0, 1])
        oof.append(pl.DataFrame({"fold_id": np.full(yva.size, k, np.int8),
                                 "anchor_date": pl.Series("anchor_date", [va] * yva.size,
                                                          dtype=pl.Date),
                                 "user_id": fv["user_id"].to_numpy(),
                                 "y_true": yva, "y_pred": pred, "y_naive": naive}))
        log(f"    fold {k} {va}  rmsle={s:.5f}  rho={rho:.5f}  "
            f"[t+{(time.time()-t0)/60:.1f}m]")
        del Xva

    oof = pl.concat(oof)
    pf = np.array(per_fold); pfn = np.array(per_fold_naive)
    Ly = np.log1p(oof["y_true"].to_numpy()); Lp = np.log1p(oof["y_pred"].to_numpy())
    rho_all = float(np.corrcoef(Ly, Lp)[0, 1])
    runtime = (time.time() - t0) / 60
    log(f"\n  cv_mean = {pf.mean():.5f} +/- {pf.std():.5f}   folds {np.round(pf,5).tolist()}")
    log(f"  pooled rho = {rho_all:.5f}   runtime {runtime:.1f} min")
    log(f"\n  >>> The number that decides this experiment is NOT cv_mean. It is r against the")
    log(f"      champion and the resulting rho_partial -- run src/admissibility.py on the OOF.")
    log(f"      A member weaker than our GBDTs still clears the bar if r <= 0.98.")

    if args.no_log:
        log("\n  --no-log: nothing written")
        return
    (ROOT / "oof").mkdir(exist_ok=True)
    oof.write_parquet(ROOT / "oof" / f"{exp_id}.parquet")
    agg = score_all(oof["y_true"].to_numpy(), oof["y_pred"].to_numpy())
    row = {"exp_id": exp_id, "parent_id": cfg["parent_id"],
           "date": datetime.now().isoformat(timespec="seconds"),
           "approach": "tfm_icl", "change": cfg.get("change", ""),
           "tier": "screen" if args.screen else "confirm",
           "n_features": len(names), "cv_mean": round(float(pf.mean()), 5),
           "cv_std": round(float(pf.std()), 5),
           "folds": json.dumps([round(float(x), 5) for x in pf]),
           "delta": round(float(pf.mean() - pfn.mean()), 5), "significant": "",
           "lb": "", "runtime_min": round(runtime, 1), "seed": cfg["seed"],
           "config": args.config, "verdict": cfg.get("verdict", ""),
           "gini_pred": round(agg["gini_pred"], 4),
           "total_rel_err": round(agg["total_rel_err"], 4),
           "notes": (f"backend={backend} context={n_ctx} pooled_rho={rho_all:.5f}. "
                     + cfg.get("notes", "")),
           "best_iters": "[]"}
    rd = ROOT / "runs"; rd.mkdir(exist_ok=True)
    (rd / f"{exp_id}.json").write_text(json.dumps(row, indent=2))
    try:
        import subprocess
        subprocess.run([sys.executable, str(ROOT / "src" / "collect.py")], check=False)
    except Exception:
        pass
    log(f"\n  wrote oof/{exp_id}.parquet and runs/{exp_id}.json")


if __name__ == "__main__":
    main()
