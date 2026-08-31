#!/usr/bin/env python
"""
Standalone RealMLP (pytabkit) — bypasses AutoGluon, whose env has a broken torch/torchvision
(e0911: `operator torchvision::nms does not exist`). pytabkit's RealMLP_TD_Regressor is a
sklearn-style estimator that needs no AutoGluon and no torchvision.

IDEAS.md §I22, and it is the ONE genuinely-open modeling lever for top-1. NN_TORCH (e0912) proved
a tabular MLP decorrelates strongly here (r=0.96 vs the blend — the most of any member) but was
too WEAK (rho 0.647) so its excess was negative. RealMLP is documented to MATCH/beat GBDT on
TabArena regression; if it holds rho ≈ 0.67 at NN_TORCH's r ≈ 0.96, the excess turns strongly
positive (e ≈ +0.022, well above the 0.024 rho_partial bar). That is the question this answers.

Fold-4 feasibility (the most test-like anchor), 1.2M-row subsample so CPU is feasible; predict
the full fold-4 population so the OOF is comparable. Writes runs/ag/e0913_f4_oof.npy + rho.
"""
from __future__ import annotations
import json, sys, time
from datetime import date
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from data import Panel                      # noqa: E402
from features import build                  # noqa: E402

BLOCKS = ["base", "counts", "rank", "visit", "channel", "dispersion", "sbcnomoment"]
SUBSAMPLE = 500_000
SEED = 0
FOLD = 4


def log(m): print(m, flush=True)


def main():
    t0 = time.time()
    from pytabkit import RealMLP_TD_Regressor
    import polars as pl
    spec = json.loads((ROOT / "data" / "fold_spec.json").read_text())
    p = Panel()
    fs = spec["folds"][FOLD]
    va = date.fromisoformat(fs["valid_anchor"]); vai = p.idx(va)
    tr = [date.fromisoformat(x) for x in fs["train_anchors"]]
    log(f"=== e0913 RealMLP standalone, fold {FOLD} (valid {va}), {len(tr)} anchors ===")

    Xtr, ytr = [], []
    for a in tr:
        ai = p.idx(a); keep = p.active_in(ai - 29, ai)
        X, names = build(p, ai, keep, BLOCKS)
        Xtr.append(X); ytr.append(p.target(ai)[keep])
    Xtr = np.concatenate(Xtr).astype(np.float32); ytr = np.concatenate(ytr)
    rng = np.random.default_rng(SEED)
    if Xtr.shape[0] > SUBSAMPLE:
        idx = np.sort(rng.choice(Xtr.shape[0], size=SUBSAMPLE, replace=False))
        Xtr, ytr = Xtr[idx], ytr[idx]
    ytr_L = np.log1p(ytr).astype(np.float32)
    log(f"    train {Xtr.shape[0]:,} x {Xtr.shape[1]}  [t+{(time.time()-t0)/60:.1f}m]")

    vkeep = p.active_in(vai - 29, vai)
    Xva, _ = build(p, vai, vkeep, BLOCKS); Xva = Xva.astype(np.float32)
    folds = pl.read_parquet(ROOT / "data" / "folds.parquet")
    fv = folds.filter(pl.col("fold_id") == FOLD).sort("user_id")
    yva = fv["target"].to_numpy()
    assert np.array_equal(fv["user_id"].to_numpy(), p.users[vkeep]), "fold population drift"

    m = RealMLP_TD_Regressor(device="cpu", random_state=SEED)
    log(f"    fitting RealMLP_TD (CPU)...  [t+{(time.time()-t0)/60:.1f}m]")
    m.fit(Xtr, ytr_L)
    pred = np.maximum(np.expm1(m.predict(Xva).ravel()), 0.0)
    log(f"    fit+predict done  [t+{(time.time()-t0)/60:.1f}m]")

    L = np.log1p(yva); M = np.log1p(pred)
    sdL = float(L.std()); rho = float(np.corrcoef(L, M)[0, 1])
    from metrics import rmsle
    log(f"\n  fold {FOLD}: RealMLP rho={rho:.5f}  raw RMSLE={rmsle(yva, pred):.5f}  "
        f"calibrated RMSLE={sdL*np.sqrt(1-rho**2):.5f}  (sd_L={sdL:.4f})")
    out = ROOT / "runs" / "ag"; out.mkdir(parents=True, exist_ok=True)
    np.save(out / "e0913_f4_oof.npy", pred.astype(np.float32))
    (out / "e0913_f4.json").write_text(json.dumps({
        "exp_id": "e0913", "fold": FOLD, "model": "RealMLP_TD (pytabkit, standalone)",
        "rho": rho, "rmsle_cal": sdL*np.sqrt(1-rho**2), "n_train": int(Xtr.shape[0]),
        "runtime_min": round((time.time()-t0)/60, 1)}, indent=2))
    log(f"  wrote runs/ag/e0913_f4.npy + .json   total {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
