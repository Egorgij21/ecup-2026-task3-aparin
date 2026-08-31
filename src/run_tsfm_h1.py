#!/usr/bin/env python
"""H1 (IDEAS.md §I29): MULTI-CHANNEL frozen-FM member. Feed ALL channels (not just GMV) through the
frozen Chronos-Bolt, mean-pool each channel's embedding, CONCAT across channels, PCA-reduce, +
abs-scale-stats -> MLP head. Fold-4, vs the GMV-only frozen member e0919 (rho 0.664, rho_partial
+0.0124 fold-4, r_vs_blend 0.979).

Prior LOW: the champion's seq-GRU already reads all 13 channels, and §3b found extra channels didn't
help it -- so more channels make the FM member see MORE of what the blend already captures, i.e. MORE
of a twin (higher r), likely a HIGHER solo rho but LOWER blend value. The last cheap check of the FM
direction before declaring it saturated (frozen best +0.0054, LoRA refuted, H2a refuted). Watches the
same decision vars: rho_partial + corr_vs_GRU. --smoke validates the multi-channel path cheaply first.
"""
from __future__ import annotations
import argparse, json, sys, time
from datetime import date
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from seqdata import build_seq_panel, CHANNELS   # noqa: E402

FOLD = 4
N_FIT_ANCHORS = 4
TRAIN_USERS_PER_ANCHOR = 25_000
MAXLEN = 512
MODEL = "amazon/chronos-bolt-small"
SEED = 0


def log(m): print(m, flush=True)


def scale_stats(series, k=8):
    out = np.zeros((len(series), 4 * k), np.float32)
    for j, idx in enumerate(np.array_split(np.arange(series.shape[1]), k)):
        seg = series[:, idx]
        out[:, 4*j] = seg.mean(1); out[:, 4*j+1] = seg.std(1)
        out[:, 4*j+2] = seg.min(1); out[:, 4*j+3] = seg.max(1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="e0923")
    ap.add_argument("--fold", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--pca", type=int, default=256)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    global FOLD; FOLD = args.fold
    n_fit = 2 if args.smoke else N_FIT_ANCHORS
    per_anchor = 3_000 if args.smoke else TRAIN_USERS_PER_ANCHOR
    epochs = 2 if args.smoke else args.epochs
    val_cap = 6_000 if args.smoke else None
    t0 = time.time()
    import torch, torch.nn as nn, polars as pl
    from chronos import BaseChronosPipeline
    from sklearn.decomposition import PCA
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    sp = build_seq_panel()
    X = sp.X.astype(np.float32)               # [users, C, days]  (log1p)
    n_ch = 3 if args.smoke else sp.n_ch
    gmv_ch = [i for i, (nm, _, _) in enumerate(CHANNELS) if nm == "gmv"][0]
    log(f"=== {args.exp_id} multi-channel FM (H1), fold {FOLD}, dev={dev}, channels={n_ch}, PCA={args.pca}, smoke={args.smoke} ===")
    pipe = BaseChronosPipeline.from_pretrained(MODEL, device_map=dev, torch_dtype=torch.float32)

    def embed_channel(ser):                   # [N, T] -> mean-pooled [N, 512]
        out = []
        for i in range(0, len(ser), 256):
            ctx = torch.tensor(ser[i:i+256], dtype=torch.float32)
            with torch.no_grad():
                e, _ = pipe.embed(ctx)
            out.append(e.mean(1).float().cpu().numpy())
        return np.concatenate(out).astype(np.float32)

    def multi_embed(anchor, users):
        lo = max(0, anchor + 1 - MAXLEN)
        embs = []
        for c in range(n_ch):
            embs.append(embed_channel(X[users, c, lo:anchor + 1]))
            log(f"      ch {c+1}/{n_ch} [t+{(time.time()-t0)/60:.1f}m]")
        return np.concatenate(embs, 1), scale_stats(X[users, gmv_ch, lo:anchor + 1])

    spec = json.loads((ROOT/"data"/"fold_spec.json").read_text())
    fs = spec["folds"][FOLD]; A = sp.idx(date.fromisoformat(fs["valid_anchor"]))
    tr_anchors = [sp.idx(date.fromisoformat(x)) for x in fs["train_anchors"]][-n_fit:]
    rng = np.random.default_rng(SEED)

    Etr, Str, ytr = [], [], []
    for a in tr_anchors:
        keep = np.where(sp.pop[:, a])[0]
        if keep.size > per_anchor: keep = np.sort(rng.choice(keep, per_anchor, replace=False))
        log(f"    fit anchor {a}: {keep.size:,} users")
        e, s = multi_embed(a, keep); Etr.append(e); Str.append(s); ytr.append(sp.Y[keep, a])
    Etr = np.concatenate(Etr); Str = np.concatenate(Str); ytr = np.concatenate(ytr).astype(np.float32)

    folds = pl.read_parquet(ROOT/"data"/"folds.parquet")
    fvf = folds.filter(pl.col("fold_id") == FOLD).sort("user_id")
    keep4 = np.where(sp.pop[:, A])[0]
    assert np.array_equal(sp.users[keep4], fvf["user_id"].to_numpy()), "fold population drift"
    yva_full = fvf["target"].to_numpy()
    if val_cap and keep4.size > val_cap:
        sel = np.sort(rng.choice(keep4.size, val_cap, replace=False)); keep4v = keep4[sel]; yva = yva_full[sel]
    else:
        keep4v = keep4; yva = yva_full
    log(f"    fold-{FOLD} val {keep4v.size:,} users")
    Eva, Sva = multi_embed(A, keep4v)

    pca = PCA(n_components=min(args.pca, Etr.shape[1]), random_state=SEED).fit(Etr)
    Etr = pca.transform(Etr).astype(np.float32); Eva = pca.transform(Eva).astype(np.float32)
    log(f"    PCA {pca.n_features_in_}->{Etr.shape[1]}  exp var {pca.explained_variance_ratio_.sum():.3f}")
    ss_mu, ss_sd = Str.mean(0), Str.std(0) + 1e-6
    Etr = np.hstack([Etr, ((Str-ss_mu)/ss_sd).astype(np.float32)])
    Eva = np.hstack([Eva, ((Sva-ss_mu)/ss_sd).astype(np.float32)])

    class Head(nn.Module):
        def __init__(self, d, h=256):
            super().__init__()
            self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, h), nn.GELU(), nn.Dropout(0.1), nn.Linear(h, 1))
        def forward(self, x): return self.net(x).squeeze(1)
    model = Head(Etr.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    def batches(E, y, bs, shuf):
        idx = np.random.permutation(len(E)) if shuf else np.arange(len(E))
        for i in range(0, len(E), bs):
            b = idx[i:i+bs]
            yield torch.from_numpy(E[b]).to(dev), (None if y is None else torch.from_numpy(y[b]).to(dev))

    log(f"    training head d={Etr.shape[1]} on {len(ytr):,} rows")
    model.train()
    for ep in range(epochs):
        tot = 0.0
        for xb, yb in batches(Etr, ytr, 512, True):
            pred = model(xb); loss = ((pred - yb) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(yb)
        log(f"      epoch {ep+1}/{epochs} mse {tot/len(ytr):.4f} [t+{(time.time()-t0)/60:.1f}m]")
    model.eval(); preds = []
    with torch.no_grad():
        for xb, _ in batches(Eva, None, 1024, False): preds.append(model(xb).float().cpu().numpy())
    pred = np.maximum(np.expm1(np.concatenate(preds)), 0.0)

    from metrics import rmsle
    L = np.log1p(yva); M = np.log1p(pred); sdL = float(L.std())
    def rho(a, b): return float(np.corrcoef(a, b)[0, 1])
    rf = rho(L, M)
    log(f"\n  FOLD {FOLD}: rho={rf:.5f}  cal RMSLE={sdL*np.sqrt(1-rf**2):.5f}  raw={rmsle(yva,pred):.5f}  (GMV-only e0919 0.664)")
    if val_cap is None:
        E0120 = ["e0049","e0064","e0100","e0101","e0101s1","e0101s2","e0101s3","e0102","e0108"]
        def bl(e):
            d = pl.read_parquet(ROOT/"oof"/f"{e}.parquet").filter(pl.col("fold_id")==FOLD).sort("user_id")
            return np.log1p(np.maximum(d["y_pred"].to_numpy(), 0.0))
        B = np.mean([bl(e) for e in E0120], axis=0); gru = bl("e0101")
        def resid(x, z): z1 = np.column_stack([z, np.ones_like(z)]); c,*_ = np.linalg.lstsq(z1, x, rcond=None); return x - z1@c
        rp = rho(resid(L, B), resid(M, B)); r = rho(M, B); rhob = rho(L, B); rgru = rho(M, gru)
        log(f"  vs blend (rho={rhob:.5f}): r={r:.5f}  excess={rf-r*rhob:+.6f}  rho_partial={rp:+.5f}  corr_vs_GRU={rgru:.4f}   (GMV-only e0919: r 0.979, rho_partial +0.0124)")
        out = ROOT/"runs"/"ag"; out.mkdir(parents=True, exist_ok=True)
        np.save(out/f"{args.exp_id}_f{FOLD}_oof.npy", pred.astype(np.float32))
        (out/f"{args.exp_id}_f{FOLD}.json").write_text(json.dumps({"exp_id": args.exp_id, "n_channels": int(n_ch),
            "pca": int(Etr.shape[1]), "rho": rf, "r_vs_blend": r, "rho_partial": rp, "corr_vs_gru": rgru,
            "excess": rf-r*rhob, "runtime_min": round((time.time()-t0)/60, 1)}, indent=2))
        log(f"  wrote runs/ag/{args.exp_id}_f{FOLD}.{{npy,json}}")
    log(f"  total {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
