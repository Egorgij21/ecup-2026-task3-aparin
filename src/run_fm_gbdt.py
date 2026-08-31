#!/usr/bin/env python
"""H2a (IDEAS.md §I29): does a frozen TS-FM embedding add signal to the GBDT as EXTRA FEATURES?

The frozen/LoRA FM MEMBER is a weak twin (best +0.0054 pooled) -- but that requires the FM to be
decorrelated-AND-strong on its own. H2a sidesteps that: inject the Chronos mean-pooled embedding
(PCA-reduced) as extra COLUMNS into the 665-feature LightGBM, retrain, and compare CV rho to the
665-only baseline. The FM only needs to carry SOME orthogonal signal the GBDT can pick up. §1v (a
GBDT can't fit its own residual) does NOT cover this -- the embedding is EXTERNAL info, not a
recombination of existing features. Fold-4 feasibility; controlled baseline vs augmented on the SAME
rows/params, so the delta isolates the FM contribution. Also reports the augmented model's
rho_partial vs the family (does it become a better blend member?).
"""
from __future__ import annotations
import json, sys, time
from datetime import date
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from data import Panel                     # noqa: E402
from features import build                 # noqa: E402
from seqdata import build_seq_panel, CHANNELS   # noqa: E402

BLOCKS = ["base", "counts", "rank", "visit", "channel", "dispersion", "sbcnomoment"]
FOLD = 4
PCA_K = 32
TRAIN_SUB = 400_000
CHR_MODEL = "amazon/chronos-bolt-small"
SEED = 0
PARAMS = dict(objective="regression", num_leaves=127, learning_rate=0.05, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, min_data_in_leaf=100, verbose=-1, num_threads=16)
ROUNDS = 300


def log(m): print(m, flush=True)


def main():
    t0 = time.time()
    import torch, polars as pl, lightgbm as lgb
    from chronos import BaseChronosPipeline
    from sklearn.decomposition import PCA
    np.random.seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"=== e0922 FM->GBDT (H2a), fold {FOLD}, dev={dev}, PCA_K={PCA_K} ===")

    p = Panel()
    sp = build_seq_panel()
    assert np.array_equal(p.users, sp.users), "Panel and SeqPanel user order differ"
    gmv_ch = [i for i, (nm, _, _) in enumerate(CHANNELS) if nm == "gmv"][0]
    Xg = sp.X[:, gmv_ch, :].astype(np.float32)
    pipe = BaseChronosPipeline.from_pretrained(CHR_MODEL, device_map=dev, torch_dtype=torch.float32)

    def fm_embed(anchor_idx, keep_bool):
        idx = np.where(keep_bool)[0]
        lo = max(0, anchor_idx + 1 - 512)
        ser = Xg[np.ix_(idx, np.arange(lo, anchor_idx + 1))]
        out = []
        for i in range(0, len(ser), 256):
            ctx = torch.tensor(ser[i:i+256], dtype=torch.float32)
            with torch.no_grad():
                e, _ = pipe.embed(ctx)                    # [b, P, d]
            out.append(e.mean(1).float().cpu().numpy())   # mean-pool tokens -> [b, d]
            if i % 51200 == 0: log(f"      fm {i:,}/{len(ser):,} [t+{(time.time()-t0)/60:.1f}m]")
        return np.concatenate(out).astype(np.float32)

    spec = json.loads((ROOT/"data"/"fold_spec.json").read_text())
    fs = spec["folds"][FOLD]
    tr = [date.fromisoformat(x) for x in fs["train_anchors"]]
    va = date.fromisoformat(fs["valid_anchor"])

    per_cap = max(1, TRAIN_SUB // len(tr))
    rng = np.random.default_rng(SEED)
    Xf, Ef, yt = [], [], []
    for a in tr:
        ai = p.idx(a); keep = p.active_in(ai - 29, ai)
        kidx = np.where(keep)[0]
        if kidx.size > per_cap:                    # subsample users BEFORE extracting (saves GPU)
            kidx = np.sort(rng.choice(kidx, per_cap, replace=False))
            keep = np.zeros(p.n_users, bool); keep[kidx] = True
        X, names = build(p, ai, keep, BLOCKS)
        E = fm_embed(ai, keep)
        log(f"    train anchor {a}: {X.shape[0]:,} rows, {X.shape[1]} feat + {E.shape[1]} fm")
        Xf.append(X.astype(np.float32)); Ef.append(E); yt.append(p.target(ai)[keep])
    Xf = np.concatenate(Xf); Ef = np.concatenate(Ef); yt = np.concatenate(yt)
    ytL = np.log1p(yt)
    log(f"    train {Xf.shape[0]:,} x ({Xf.shape[1]} feat + {Ef.shape[1]} fm)  [t+{(time.time()-t0)/60:.1f}m]")

    vai = p.idx(va); vkeep = p.active_in(vai - 29, vai)
    Xva, _ = build(p, vai, vkeep, BLOCKS); Xva = Xva.astype(np.float32)
    Eva = fm_embed(vai, vkeep)
    folds = pl.read_parquet(ROOT/"data"/"folds.parquet")
    fv = folds.filter(pl.col("fold_id") == FOLD).sort("user_id")
    yva = fv["target"].to_numpy()
    assert np.array_equal(fv["user_id"].to_numpy(), p.users[vkeep]), "val population drift"

    pca = PCA(n_components=PCA_K, random_state=SEED).fit(Ef)
    Efp = pca.transform(Ef).astype(np.float32); Evap = pca.transform(Eva).astype(np.float32)
    log(f"    PCA {Ef.shape[1]}->{PCA_K}, explained var {pca.explained_variance_ratio_.sum():.3f}")

    def fit_score(Xtr, Xval):
        m = lgb.train(PARAMS, lgb.Dataset(Xtr, ytL), num_boost_round=ROUNDS)
        pred = np.maximum(np.expm1(m.predict(Xval)), 0.0)
        L = np.log1p(yva); M = np.log1p(pred)
        return float(np.corrcoef(L, M)[0, 1]), pred

    rho_b, pred_b = fit_score(Xf, Xva)
    rho_f, pred_f = fit_score(np.hstack([Xf, Efp]), np.hstack([Xva, Evap]))
    log(f"\n  BASELINE  [665 feat]         rho = {rho_b:.5f}")
    log(f"  AUGMENTED [665 + FM-PCA{PCA_K}]  rho = {rho_f:.5f}   Δ = {rho_f-rho_b:+.5f}")

    # is the AUGMENTED gbdt a better blend member? rho_partial vs the 9-member family
    E0120 = ["e0049","e0064","e0100","e0101","e0101s1","e0101s2","e0101s3","e0102","e0108"]
    def bl(e):
        d = pl.read_parquet(ROOT/"oof"/f"{e}.parquet").filter(pl.col("fold_id")==FOLD).sort("user_id")
        return np.log1p(np.maximum(d["y_pred"].to_numpy(), 0.0))
    B = np.mean([bl(e) for e in E0120], axis=0); L = np.log1p(yva)
    def resid(x, z): z1 = np.column_stack([z, np.ones_like(z)]); c,*_ = np.linalg.lstsq(z1, x, rcond=None); return x - z1@c
    for nm, pr in [("baseline", pred_b), ("augmented", pred_f)]:
        M = np.log1p(pr); rp = float(np.corrcoef(resid(L, B), resid(M, B))[0, 1]); r = float(np.corrcoef(M, B)[0, 1])
        log(f"  {nm:9s}: r_vs_blend={r:.5f}  rho_partial={rp:+.5f}")
    out = ROOT/"runs"/"ag"; out.mkdir(parents=True, exist_ok=True)
    (out/f"e0922_f{FOLD}.json").write_text(json.dumps({"exp_id": "e0922", "fold": FOLD,
        "rho_baseline": rho_b, "rho_augmented": rho_f, "delta": rho_f-rho_b, "pca_k": PCA_K,
        "runtime_min": round((time.time()-t0)/60, 1)}, indent=2))
    log(f"  wrote runs/ag/e0922_f{FOLD}.json  total {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
