#!/usr/bin/env python
"""H1b / strength-recovery (IDEAS.md §I29): the HYBRID FM member. e0923 (all-13-channel mean-pool +
PCA) was the most DECORRELATED FM member ever (r 0.911) but too WEAK (rho 0.615) -- the mean-pool +
PCA-256 diluted the GMV-target signal (GMV is 1/13 of the concat). This keeps e0919's STRONG GMV
per-patch GRU (rho 0.664) and ADDS the other 12 channels' mean-pooled Chronos embeddings (PCA-reduced)
as decorrelating extras at the head -- strength from GMV, decorrelation from the other channels.

Fold-4, vs e0919 (rho 0.664, r 0.979, rho_partial +0.0124) and e0923 (rho 0.615, r 0.911, +0.0017).
The bet: rho stays ~0.66 (GMV-GRU dominates) while r drops well below 0.979 (multi-channel decorrelates)
-> rho_partial ABOVE +0.0124. Watches rho_partial + corr_vs_GRU. --smoke validates the path.
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
N_FIT_ANCHORS = 6
TRAIN_USERS_PER_ANCHOR = 40_000
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
    ap.add_argument("--exp-id", default="e0924")
    ap.add_argument("--fold", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--mc-pca", type=int, default=128)
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
    X = sp.X.astype(np.float32)
    gmv_ch = [i for i, (nm, _, _) in enumerate(CHANNELS) if nm == "gmv"][0]
    other_ch = [c for c in range(3 if args.smoke else sp.n_ch) if c != gmv_ch]
    log(f"=== {args.exp_id} HYBRID (GMV per-patch GRU + {len(other_ch)} mc mean-pool), fold {FOLD}, dev={dev}, mc_pca={args.mc_pca}, smoke={args.smoke} ===")
    pipe = BaseChronosPipeline.from_pretrained(MODEL, device_map=dev, torch_dtype=torch.float32)

    def embed(series_2d, pool):
        out = []
        for i in range(0, len(series_2d), 256):
            ctx = torch.tensor(series_2d[i:i+256], dtype=torch.float32)
            with torch.no_grad():
                e, _ = pipe.embed(ctx)                       # [b, P, d]
            out.append((e.mean(1) if pool else e).float().cpu().numpy())
        return np.concatenate(out)

    def build(anchor, users):
        lo = max(0, anchor + 1 - MAXLEN)
        gmv = X[users, gmv_ch, lo:anchor + 1]
        toks = embed(gmv, pool=False).astype(np.float16)     # [N, P, d]  per-patch for the GRU
        mc = [embed(X[users, c, lo:anchor + 1], pool=True) for c in other_ch]
        mc = np.concatenate(mc, 1).astype(np.float32)        # [N, |other|*512]
        return toks, scale_stats(gmv), mc

    spec = json.loads((ROOT/"data"/"fold_spec.json").read_text())
    fs = spec["folds"][FOLD]; A = sp.idx(date.fromisoformat(fs["valid_anchor"]))
    tr_anchors = [sp.idx(date.fromisoformat(x)) for x in fs["train_anchors"]][-n_fit:]
    rng = np.random.default_rng(SEED)

    Ptr, Str, Mtr, ytr = [], [], [], []
    for a in tr_anchors:
        keep = np.where(sp.pop[:, a])[0]
        if keep.size > per_anchor: keep = np.sort(rng.choice(keep, per_anchor, replace=False))
        log(f"    fit anchor {a}: {keep.size:,} users")
        tk, ss, mc = build(a, keep); Ptr.append(tk); Str.append(ss); Mtr.append(mc); ytr.append(sp.Y[keep, a])
    Pmax = max(p.shape[1] for p in Ptr)

    folds = pl.read_parquet(ROOT/"data"/"folds.parquet")
    fvf = folds.filter(pl.col("fold_id") == FOLD).sort("user_id")
    keep4 = np.where(sp.pop[:, A])[0]
    assert np.array_equal(sp.users[keep4], fvf["user_id"].to_numpy()), "fold population drift"
    yva_full = fvf["target"].to_numpy()
    if val_cap and keep4.size > val_cap:
        sel = np.sort(rng.choice(keep4.size, val_cap, replace=False)); keep4v = keep4[sel]; yva = yva_full[sel]
    else:
        keep4v = keep4; yva = yva_full
    Pva, Sva, Mva = build(A, keep4v); Pmax = max(Pmax, Pva.shape[1])
    log(f"    fold-{FOLD} val {keep4v.size:,} users")

    def padP(p): return np.pad(p, ((0, 0), (Pmax - p.shape[1], 0), (0, 0))) if p.shape[1] < Pmax else p
    Ptr = np.concatenate([padP(p) for p in Ptr]); Pva = padP(Pva)
    Str = np.concatenate(Str); Mtr = np.concatenate(Mtr); ytr = np.concatenate(ytr).astype(np.float32)

    pca = PCA(n_components=min(args.mc_pca, Mtr.shape[1]), random_state=SEED).fit(Mtr)
    Mtr = pca.transform(Mtr).astype(np.float32); Mva = pca.transform(Mva).astype(np.float32)
    log(f"    mc PCA {pca.n_features_in_}->{Mtr.shape[1]}  exp var {pca.explained_variance_ratio_.sum():.3f}")
    smu, ssd = Str.mean(0), Str.std(0) + 1e-6; mmu, msd = Mtr.mean(0), Mtr.std(0) + 1e-6
    Etr = np.hstack([((Str-smu)/ssd), ((Mtr-mmu)/msd)]).astype(np.float32)
    Eva = np.hstack([((Sva-smu)/ssd), ((Mva-mmu)/msd)]).astype(np.float32)
    extra = Etr.shape[1]

    class GRUHead(nn.Module):
        def __init__(self, c_in, d=128, extra=0):
            super().__init__()
            self.proj = nn.Linear(c_in, d); self.gru = nn.GRU(d, d, 2, batch_first=True, dropout=0.1)
            self.head = nn.Sequential(nn.LayerNorm(d + extra), nn.Linear(d + extra, d), nn.GELU(), nn.Linear(d, 1))
        def forward(self, seq, ex):
            h = self.proj(seq); _, hn = self.gru(h); z = torch.cat([hn[-1], ex], 1)
            return self.head(z).squeeze(1)
    model = GRUHead(Ptr.shape[2], extra=extra).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    def batches(P, E, y, bs, shuf):
        idx = np.random.permutation(len(P)) if shuf else np.arange(len(P))
        for i in range(0, len(P), bs):
            b = idx[i:i+bs]
            yield (torch.from_numpy(P[b].astype(np.float32)).to(dev), torch.from_numpy(E[b]).to(dev),
                   None if y is None else torch.from_numpy(y[b]).to(dev))

    log(f"    training GRU c_in={Ptr.shape[2]} extra={extra} Pmax={Pmax} on {len(ytr):,} rows")
    model.train()
    for ep in range(epochs):
        tot = 0.0
        for seq, ex, yb in batches(Ptr, Etr, ytr, 512, True):
            pred = model(seq, ex); loss = ((pred - yb) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(yb)
        log(f"      epoch {ep+1}/{epochs} mse {tot/len(ytr):.4f} [t+{(time.time()-t0)/60:.1f}m]")
    model.eval(); preds = []
    with torch.no_grad():
        for seq, ex, _ in batches(Pva, Eva, None, 1024, False):
            preds.append(model(seq, ex).float().cpu().numpy())
    pred = np.maximum(np.expm1(np.concatenate(preds)), 0.0)

    from metrics import rmsle
    L = np.log1p(yva); Mp = np.log1p(pred); sdL = float(L.std())
    def rho(a, b): return float(np.corrcoef(a, b)[0, 1])
    rf = rho(L, Mp)
    log(f"\n  FOLD {FOLD}: rho={rf:.5f}  cal RMSLE={sdL*np.sqrt(1-rf**2):.5f}  raw={rmsle(yva,pred):.5f}  (e0919 0.664, e0923 0.615)")
    if val_cap is None:
        E0120 = ["e0049","e0064","e0100","e0101","e0101s1","e0101s2","e0101s3","e0102","e0108"]
        def bl(e):
            d = pl.read_parquet(ROOT/"oof"/f"{e}.parquet").filter(pl.col("fold_id")==FOLD).sort("user_id")
            return np.log1p(np.maximum(d["y_pred"].to_numpy(), 0.0))
        B = np.mean([bl(e) for e in E0120], axis=0); gru = bl("e0101")
        def resid(x, z): z1 = np.column_stack([z, np.ones_like(z)]); c,*_ = np.linalg.lstsq(z1, x, rcond=None); return x - z1@c
        rp = rho(resid(L, B), resid(Mp, B)); r = rho(Mp, B); rhob = rho(L, B); rgru = rho(Mp, gru)
        log(f"  vs blend (rho={rhob:.5f}): r={r:.5f}  excess={rf-r*rhob:+.6f}  rho_partial={rp:+.5f}  corr_vs_GRU={rgru:.4f}   (e0919: r 0.979 +0.0124 | e0923: r 0.911 +0.0017)")
        out = ROOT/"runs"/"ag"; out.mkdir(parents=True, exist_ok=True)
        np.save(out/f"{args.exp_id}_f{FOLD}_oof.npy", pred.astype(np.float32))
        (out/f"{args.exp_id}_f{FOLD}.json").write_text(json.dumps({"exp_id": args.exp_id, "n_other_ch": len(other_ch),
            "mc_pca": int(Mtr.shape[1]), "rho": rf, "r_vs_blend": r, "rho_partial": rp, "corr_vs_gru": rgru,
            "excess": rf-r*rhob, "runtime_min": round((time.time()-t0)/60, 1)}, indent=2))
        log(f"  wrote runs/ag/{args.exp_id}_f{FOLD}.{{npy,json}}")
    log(f"  total {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
