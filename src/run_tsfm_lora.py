#!/usr/bin/env python
"""LoRA-fine-tuned MOMENT as a blend member -- the frozen->LoRA rung (IDEAS.md §I29 fine-tune axis).

Frozen MOMENT (e0915) was a weak twin (pooled rho_partial -0.0004); frozen Chronos+scale-stats (e0919)
reached +0.0054. This tests the survey's core bet: LoRA-adapting the backbone (low r) GAINS rho while
RETAINING decorrelation -- the geometry papers (2405.09673 "LoRA Learns Less and Forgets Less",
2410.21228 "Illusion of Equivalence") say LoRA stays near the pretrained prior, where FULL fine-tune
converges to the from-scratch solution (= a twin of our GRU). So the decision variables are
rho_partial vs the family AND corr(OOF, e0101 GRU) -- NEVER solo rho, NEVER one fold (the e0915
confirm is the cautionary tale).

MOMENT-small (embedding), LoRA(r, q/v) on the T5 encoder, mean-pooled embedding + abs-scale-stats ->
MLP head, trained END-TO-END per fold (grad flows through the LoRA'd encoder), leakage-safe. GPU.
--smoke runs a tiny end-to-end check first (validates the PEFT wiring + GPU path).
"""
from __future__ import annotations
import argparse, json, sys, time
from datetime import date
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from seqdata import build_seq_panel, CHANNELS   # noqa: E402

N_FIT_ANCHORS = 6
TRAIN_USERS_PER_ANCHOR = 40_000
SEQ_LEN = 512
MODEL = "AutonLab/MOMENT-1-small"
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
    ap.add_argument("--exp-id", default="e0920")
    ap.add_argument("--fold", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--no-scale-stats", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    FOLD = args.fold
    use_ss = not args.no_scale_stats
    n_fit = 2 if args.smoke else N_FIT_ANCHORS
    per_anchor = 3_000 if args.smoke else TRAIN_USERS_PER_ANCHOR
    epochs = 2 if args.smoke else args.epochs
    val_cap = 6_000 if args.smoke else None
    t0 = time.time()
    import torch, torch.nn as nn, polars as pl
    from momentfm import MOMENTPipeline
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gmv_ch = [i for i, (nm, _, _) in enumerate(CHANNELS) if nm == "gmv"][0]
    log(f"=== {args.exp_id} LoRA MOMENT r={args.lora_r}, fold {FOLD}, dev={dev}, scale_stats={use_ss}, smoke={args.smoke} ===")

    fm = MOMENTPipeline.from_pretrained(MODEL, model_kwargs={"task_name": "embedding"})
    fm.init()
    lora = LoraConfig(r=args.lora_r, lora_alpha=2*args.lora_r, target_modules=["q", "v"],
                      lora_dropout=0.05, bias="none")
    fm = get_peft_model(fm, lora); fm.to(dev)
    tr_p = sum(p.numel() for p in fm.parameters() if p.requires_grad)
    tot_p = sum(p.numel() for p in fm.parameters())
    log(f"    LoRA trainable {tr_p:,} / {tot_p:,} ({100*tr_p/tot_p:.2f}%)")

    def encode(series_2d):
        B = len(series_2d)
        x = np.zeros((B, 1, SEQ_LEN), np.float32); m = np.zeros((B, SEQ_LEN), np.float32)
        for j, s in enumerate(series_2d):
            L = min(len(s), SEQ_LEN); x[j, 0, SEQ_LEN-L:] = s[-L:]; m[j, SEQ_LEN-L:] = 1.0
        return x, m

    sp = build_seq_panel()
    Xg = sp.X[:, gmv_ch, :].astype(np.float32)
    spec = json.loads((ROOT/"data"/"fold_spec.json").read_text())
    fs = spec["folds"][FOLD]; A = sp.idx(date.fromisoformat(fs["valid_anchor"]))
    tr_anchors = [sp.idx(date.fromisoformat(x)) for x in fs["train_anchors"]][-n_fit:]
    rng = np.random.default_rng(SEED)

    def gather(anchor, users):
        lo = max(0, anchor + 1 - SEQ_LEN)
        ser = Xg[np.ix_(users, np.arange(lo, anchor + 1))]
        x, m = encode(ser)
        return x, m, scale_stats(ser)

    Xtr, Mtr, Str, ytr = [], [], [], []
    for a in tr_anchors:
        keep = np.where(sp.pop[:, a])[0]
        if keep.size > per_anchor: keep = np.sort(rng.choice(keep, per_anchor, replace=False))
        log(f"    fit anchor {a}: {keep.size:,} users")
        x, m, ss = gather(a, keep); Xtr.append(x); Mtr.append(m); Str.append(ss); ytr.append(sp.Y[keep, a])
    Xtr = np.concatenate(Xtr); Mtr = np.concatenate(Mtr); Str = np.concatenate(Str)
    ytr = np.concatenate(ytr).astype(np.float32)

    folds = pl.read_parquet(ROOT/"data"/"folds.parquet")
    fvf = folds.filter(pl.col("fold_id") == FOLD).sort("user_id")
    keep4 = np.where(sp.pop[:, A])[0]
    assert np.array_equal(sp.users[keep4], fvf["user_id"].to_numpy()), "fold population drift"
    yva_full = fvf["target"].to_numpy()
    if val_cap and keep4.size > val_cap:
        sel = np.sort(rng.choice(keep4.size, val_cap, replace=False)); keep4v = keep4[sel]; yva = yva_full[sel]
    else:
        keep4v = keep4; yva = yva_full
    Xva, Mva, Sva = gather(A, keep4v)
    log(f"    fold-{FOLD} eval {keep4v.size:,} users")

    ss_mu, ss_sd = Str.mean(0), Str.std(0) + 1e-6
    Str = ((Str - ss_mu)/ss_sd).astype(np.float32); Sva = ((Sva - ss_mu)/ss_sd).astype(np.float32)
    extra = Str.shape[1] if use_ss else 0

    class Head(nn.Module):
        def __init__(self, d=512, extra=0, h=128):
            super().__init__()
            self.net = nn.Sequential(nn.LayerNorm(d + extra), nn.Linear(d + extra, h), nn.GELU(), nn.Linear(h, 1))
        def forward(self, emb, ex=None):
            z = torch.cat([emb, ex], 1) if ex is not None else emb
            return self.net(z).squeeze(1)
    head = Head(extra=extra).to(dev)

    opt = torch.optim.AdamW([
        {"params": [p for p in fm.parameters() if p.requires_grad], "lr": 2e-4},
        {"params": list(head.parameters()), "lr": 2e-3}], weight_decay=1e-4)

    def run_fm(x, m):
        return fm(x_enc=x, input_mask=m).embeddings          # [B, d]

    def batches(X, M, S, y, bs, shuf):
        idx = np.random.permutation(len(X)) if shuf else np.arange(len(X))
        for i in range(0, len(X), bs):
            b = idx[i:i+bs]
            yield (torch.from_numpy(X[b]).to(dev), torch.from_numpy(M[b]).to(dev),
                   torch.from_numpy(S[b]).to(dev) if extra else None,
                   None if y is None else torch.from_numpy(y[b]).to(dev))

    bs = 64 if args.smoke else 128
    log(f"    training end-to-end (LoRA MOMENT + head) on {len(ytr):,} rows, bs={bs}")
    fm.train(); head.train()
    for ep in range(epochs):
        tot = 0.0
        for x, m, ss, yb in batches(Xtr, Mtr, Str, ytr, bs, True):
            pred = head(run_fm(x, m), ss); loss = ((pred - yb) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(yb)
        log(f"      epoch {ep+1}/{epochs} mse {tot/len(ytr):.4f} [t+{(time.time()-t0)/60:.1f}m]")

    fm.eval(); head.eval(); preds = []
    with torch.no_grad():
        for x, m, ss, _ in batches(Xva, Mva, Sva, None, 256, False):
            preds.append(head(run_fm(x, m), ss).float().cpu().numpy())
    pred = np.maximum(np.expm1(np.concatenate(preds)), 0.0)

    from metrics import rmsle
    L = np.log1p(yva); Mp = np.log1p(pred); sdL = float(L.std())
    def rho(a, b): return float(np.corrcoef(a, b)[0, 1])
    rf = rho(L, Mp)
    log(f"\n  FOLD {FOLD}: rho={rf:.5f}  cal RMSLE={sdL*np.sqrt(1-rf**2):.5f}  raw={rmsle(yva,pred):.5f}  "
        f"(frozen: e0915 0.651, e0919 0.664)")
    if val_cap is None:
        E0120 = ["e0049","e0064","e0100","e0101","e0101s1","e0101s2","e0101s3","e0102","e0108"]
        def bl(e):
            d = pl.read_parquet(ROOT/"oof"/f"{e}.parquet").filter(pl.col("fold_id")==FOLD).sort("user_id")
            return np.log1p(np.maximum(d["y_pred"].to_numpy(), 0.0))
        B = np.mean([bl(e) for e in E0120], axis=0); gru = bl("e0101")
        def resid(x, z): z1 = np.column_stack([z, np.ones_like(z)]); c,*_ = np.linalg.lstsq(z1, x, rcond=None); return x - z1@c
        rp = rho(resid(L, B), resid(Mp, B)); r = rho(Mp, B); rhob = rho(L, B); rgru = rho(Mp, gru)
        log(f"  vs blend (rho={rhob:.5f}): r={r:.5f}  excess={rf-r*rhob:+.6f}  rho_partial={rp:+.5f}  "
            f"corr_vs_GRU(e0101)={rgru:.4f}   (frozen e0919 r 0.979, rho_partial +0.0124 fold4)")
        out = ROOT/"runs"/"ag"; out.mkdir(parents=True, exist_ok=True)
        np.save(out/f"{args.exp_id}_f{FOLD}_oof.npy", pred.astype(np.float32))
        (out/f"{args.exp_id}_f{FOLD}.json").write_text(json.dumps({"exp_id": args.exp_id, "model": MODEL,
            "lora_r": args.lora_r, "scale_stats": use_ss, "rho": rf, "r_vs_blend": r, "rho_partial": rp,
            "corr_vs_gru": rgru, "excess": rf-r*rhob, "runtime_min": round((time.time()-t0)/60, 1)}, indent=2))
        log(f"  wrote runs/ag/{args.exp_id}_f{FOLD}.{{npy,json}}")
    log(f"  total {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
