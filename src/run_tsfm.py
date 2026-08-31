#!/usr/bin/env python
"""
Frozen TIME-SERIES foundation-model embedding as a blend member (IDEAS.md §I28).

Recipe (Auer et al. 2510.26777): run each user's daily log1p-GMV series through a FROZEN
pretrained TS foundation model (MOMENT-1), take the embedding, feed it to LightGBM, and score
rho + rho_partial vs the champion blend. This injects EXTERNAL temporal priors (MOMENT is
pretrained on a large external TS corpus) -- the one information source the project never tapped.
Leakage-safe: the FM never saw our target, and the series is truncated at each anchor (days <= A).

Fold-4 feasibility (the most test-like anchor). Uses a handful of fit anchors + subsampled train
users to keep the frozen-forward pass tractable on CPU; predicts the FULL fold-4 population so the
OOF aligns with the blend for admissibility.
"""
from __future__ import annotations
import json, sys, time
from datetime import date
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from seqdata import build_seq_panel, CHANNELS         # noqa: E402

FOLD = 4
LAST_CLEAN_ANCHOR = date(2025, 10, 16)
N_FIT_ANCHORS = 6
TRAIN_USERS_PER_ANCHOR = 60_000
SEQ_LEN = 512
MODEL = "AutonLab/MOMENT-1-small"
SEED = 0


def log(m): print(m, flush=True)


def embed_series(model, torch, series_list, device, bs=256):
    """series_list: list of 1-D np arrays (log1p daily gmv, len<=512). Returns [N, d_model]."""
    embs = []
    n = len(series_list)
    for i in range(0, n, bs):
        chunk = series_list[i:i + bs]
        B = len(chunk)
        x = np.zeros((B, 1, SEQ_LEN), np.float32)
        m = np.zeros((B, SEQ_LEN), np.float32)
        for j, s in enumerate(chunk):
            L = min(len(s), SEQ_LEN)
            x[j, 0, SEQ_LEN - L:] = s[-L:]      # right-align: most recent day at position 511
            m[j, SEQ_LEN - L:] = 1.0
        xt = torch.from_numpy(x).to(device); mt = torch.from_numpy(m).to(device)
        with torch.no_grad():
            out = model(x_enc=xt, input_mask=mt)
        embs.append(out.embeddings.float().cpu().numpy())
        if i % (bs * 40) == 0:
            log(f"      embedded {i + B:,}/{n:,}")
    return np.concatenate(embs, axis=0)


def main():
    t0 = time.time()
    import torch, polars as pl, lightgbm as lgb
    from momentfm import MOMENTPipeline
    gmv_ch = [i for i, (nm, _, _) in enumerate(CHANNELS) if nm == "gmv"][0]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"=== e0914 frozen TS-FM (MOMENT-1-small) embedding, fold {FOLD}, device={device} ===")

    model = MOMENTPipeline.from_pretrained(MODEL, model_kwargs={"task_name": "embedding"})
    model.init(); model.to(device); model.eval()

    sp = build_seq_panel()
    Xg = sp.X[:, gmv_ch, :].astype(np.float32)          # (n_users, n_days) log1p daily gmv
    spec = json.loads((ROOT / "data" / "fold_spec.json").read_text())
    fs = spec["folds"][FOLD]
    A = sp.idx(LAST_CLEAN_ANCHOR)
    tr_anchors = [sp.idx(date.fromisoformat(x)) for x in fs["train_anchors"]]
    tr_anchors = tr_anchors[-N_FIT_ANCHORS:]
    rng = np.random.default_rng(SEED)
    log(f"    fit anchors (idx): {tr_anchors}   fold-4 anchor idx {A}")

    # ---- training rows: embeddings at fit anchors, subsampled users
    Etr, ytr = [], []
    for a in tr_anchors:
        keep = np.where(sp.pop[:, a])[0]
        if keep.size > TRAIN_USERS_PER_ANCHOR:
            keep = np.sort(rng.choice(keep, TRAIN_USERS_PER_ANCHOR, replace=False))
        series = [Xg[u, :a + 1] for u in keep]
        log(f"    anchor {a}: embedding {keep.size:,} users  [t+{(time.time()-t0)/60:.1f}m]")
        Etr.append(embed_series(model, torch, series, device))
        ytr.append(sp.Y[keep, a])                        # log1p(sum gmv [a+1, a+30])
    Etr = np.concatenate(Etr); ytr = np.concatenate(ytr)

    # ---- fold-4 test: FULL population, aligned to folds.parquet
    folds = pl.read_parquet(ROOT / "data" / "folds.parquet")
    fv = folds.filter(pl.col("fold_id") == FOLD).sort("user_id")
    keep4 = np.where(sp.pop[:, A])[0]
    assert np.array_equal(sp.users[keep4], fv["user_id"].to_numpy()), "fold-4 population mismatch"
    series4 = [Xg[u, :A + 1] for u in keep4]
    log(f"    fold-4: embedding {keep4.size:,} users  [t+{(time.time()-t0)/60:.1f}m]")
    Eva = embed_series(model, torch, series4, device)
    yva = fv["target"].to_numpy()

    log(f"    training LightGBM on {Etr.shape[0]:,} x {Etr.shape[1]} embeddings  "
        f"[t+{(time.time()-t0)/60:.1f}m]")
    params = dict(objective="regression", metric="rmse", learning_rate=0.05, num_leaves=63,
                  min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=1.0, num_threads=16, verbosity=-1, seed=SEED)
    m = lgb.train(params, lgb.Dataset(Etr, ytr), num_boost_round=300,
                  callbacks=[lgb.log_evaluation(0)])
    pred = np.maximum(np.expm1(m.predict(Eva)), 0.0)

    # ---- score: rho + rho_partial vs the 9-member blend on fold 4
    from metrics import rmsle
    L = np.log1p(yva); M = np.log1p(pred); sdL = float(L.std())
    def rho(a, b): return float(np.corrcoef(a, b)[0, 1])
    rho_fm = rho(L, M)
    E0120 = ["e0049", "e0064", "e0100", "e0101", "e0101s1", "e0101s2", "e0101s3", "e0102", "e0108"]
    def bl(e):
        d = pl.read_parquet(ROOT / "oof" / f"{e}.parquet").filter(pl.col("fold_id") == FOLD).sort("user_id")
        return np.log1p(np.maximum(d["y_pred"].to_numpy(), 0.0))
    B = np.mean([bl(e) for e in E0120], axis=0)
    def resid(x, z): z1 = np.column_stack([z, np.ones_like(z)]); b, *_ = np.linalg.lstsq(z1, x, rcond=None); return x - z1 @ b
    rp = rho(resid(L, B), resid(M, B)); r = rho(M, B); rho_bl = rho(L, B)
    log(f"\n  FOLD {FOLD}: MOMENT-FM member rho={rho_fm:.5f}  raw RMSLE={rmsle(yva,pred):.5f}  "
        f"cal RMSLE={sdL*np.sqrt(1-rho_fm**2):.5f}")
    log(f"  vs 9-member blend (rho={rho_bl:.5f}): r={r:.5f}  excess e={rho_fm-r*rho_bl:+.6f}  "
        f"rho_partial={rp:+.5f}   (bar 0.024; best member ever 0.0127)")
    out = ROOT / "runs" / "ag"; out.mkdir(parents=True, exist_ok=True)
    np.save(out / "e0914_f4_oof.npy", pred.astype(np.float32))
    (out / "e0914_f4.json").write_text(json.dumps({
        "exp_id": "e0914", "fold": FOLD, "model": MODEL, "rho": rho_fm,
        "r_vs_blend": r, "rho_partial": rp, "excess": rho_fm - r * rho_bl,
        "n_train": int(Etr.shape[0]), "emb_dim": int(Etr.shape[1]),
        "runtime_min": round((time.time() - t0) / 60, 1)}, indent=2))
    log(f"  wrote runs/ag/e0914_f4.{{npy,json}}   total {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
