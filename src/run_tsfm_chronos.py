#!/usr/bin/env python
"""GRU over frozen Chronos-Bolt per-token embeddings + absolute-scale-stats aug (IDEAS.md §I29).

Strengthen of e0915 (frozen MOMENT per-patch GRU, pooled rho_partial ~0 but POSITIVE and rising with
history length -> the test anchor, longest histories, is where it is strongest). Two survey levers:
  (1) BACKBONE: Chronos-Bolt-small = #2 frozen extractor in 2510.26777 (MOMENT was LAST); value-
      tokenised T5 seq2seq -> different objective+corpus -> decorrelation.  embed() -> [B, P, 512].
  (2) ABS-SCALE STATS: every FM instance-normalises, discarding absolute level; our target is a
      30-day SUM (a level). Recover it: k=8 time-patches x {mean,std,min,max}, standardised on train,
      concatenated at the GRU head. e0915 was scale-blind.

Panel is a DENSE time-grid, so at a given anchor every user's series has the SAME length (sparsity is
in values, not length) -> no per-user masking needed. Per-anchor, causal (features <= anchor).
Fold-4 feasibility; scores rho and rho_partial vs the 9-member family -- READ POOLED/5-fold, never one
fold (the e0915 confirm is the cautionary tale). --smoke runs a tiny end-to-end check first.
"""
from __future__ import annotations
import argparse, json, sys, time
from datetime import date
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from seqdata import build_seq_panel, CHANNELS   # noqa: E402

MODEL = "amazon/chronos-bolt-small"
MAXLEN = 512
SEED = 0


def log(m): print(m, flush=True)


def scale_stats(series, k=8):
    """[N,T] raw series -> [N, 4k] per-user abs-scale features (mean/std/min/max over k time patches)."""
    out = np.zeros((len(series), 4 * k), np.float32)
    for j, idx in enumerate(np.array_split(np.arange(series.shape[1]), k)):
        seg = series[:, idx]
        out[:, 4*j+0] = seg.mean(1); out[:, 4*j+1] = seg.std(1)
        out[:, 4*j+2] = seg.min(1);  out[:, 4*j+3] = seg.max(1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="e0919")
    ap.add_argument("--fold", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--no-scale-stats", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--submit", action="store_true",
                    help="predict at the real test anchor (2026-02-13) for the sample_submit "
                         "population and write subs/<exp_id>.csv (train clean-only)")
    args = ap.parse_args()
    FOLD = args.fold
    use_ss = not args.no_scale_stats
    n_fit = 2 if args.smoke else 6
    per_anchor = 4_000 if args.smoke else 40_000
    epochs = 2 if args.smoke else args.epochs
    val_cap = 8_000 if args.smoke else None
    t0 = time.time()
    import torch, torch.nn as nn, polars as pl
    from chronos import BaseChronosPipeline
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gmv_ch = [i for i, (nm, _, _) in enumerate(CHANNELS) if nm == "gmv"][0]
    log(f"=== {args.exp_id} Chronos-Bolt GRU, fold {FOLD}, dev={dev}, scale_stats={use_ss}, smoke={args.smoke} ===")

    pipe = BaseChronosPipeline.from_pretrained(MODEL, device_map=dev, torch_dtype=torch.float32)

    def fm_tokens(series_2d):
        """[N,T] -> per-token embeddings [N, P, d] (float16). Uniform T -> uniform P within a call."""
        toks = []
        for i in range(0, len(series_2d), 256):
            ctx = torch.tensor(series_2d[i:i+256], dtype=torch.float32)   # embed moves to model device
            with torch.no_grad():
                emb, _ = pipe.embed(ctx)
            toks.append(emb.float().cpu().numpy().astype(np.float16))
            if i % 10240 == 0: log(f"      fm {i+min(256,len(series_2d)-i):,}/{len(series_2d):,} [t+{(time.time()-t0)/60:.1f}m]")
        return np.concatenate(toks)

    sp = build_seq_panel()
    Xg = sp.X[:, gmv_ch, :].astype(np.float32)
    spec = json.loads((ROOT / "data" / "fold_spec.json").read_text())
    if args.submit:
        A = sp.idx(date(2026, 2, 13))                          # real test anchor
        lc = sp.idx(date(2025, 10, 16))                        # guard boundary
        tr_anchors = sorted([a for a in range(lc, 120, -30)][:n_fit])
        log(f"    SUBMIT: test anchor {sp.day(A)}, {len(tr_anchors)} clean train anchors")
    else:
        fs = spec["folds"][FOLD]; A = sp.idx(date.fromisoformat(fs["valid_anchor"]))
        tr_anchors = [sp.idx(date.fromisoformat(x)) for x in fs["train_anchors"]][-n_fit:]
    rng = np.random.default_rng(SEED)

    def build(anchor, users):
        lo = max(0, anchor + 1 - MAXLEN)
        ser = Xg[np.ix_(users, np.arange(lo, anchor + 1))]   # [N, T], same T for all users at this anchor
        return fm_tokens(ser), scale_stats(ser)

    def padP(p, P):
        return np.pad(p, ((0, 0), (P - p.shape[1], 0), (0, 0))) if p.shape[1] < P else p

    # ---- train
    Ptr, Str, ytr = [], [], []
    for a in tr_anchors:
        keep = np.where(sp.pop[:, a])[0]
        if keep.size > per_anchor:
            keep = np.sort(rng.choice(keep, per_anchor, replace=False))
        log(f"    fit anchor {a}: {keep.size:,} users")
        p, s = build(a, keep); Ptr.append(p); Str.append(s); ytr.append(sp.Y[keep, a])

    # ---- eval / test population (assert alignment before extraction)
    keep4 = np.where(sp.pop[:, A])[0]
    if args.submit:
        ss = pl.read_csv(ROOT / "data" / "sample_submit.csv")
        assert np.array_equal(sp.users[keep4], ss["user_id"].to_numpy()), "test pop != sample_submit"
        keep4v = keep4; yva = None
        log(f"    SUBMIT eval on {keep4v.size:,} test users at {sp.day(A)}")
    else:
        folds = pl.read_parquet(ROOT / "data" / "folds.parquet")
        fvf = folds.filter(pl.col("fold_id") == FOLD).sort("user_id")
        assert np.array_equal(sp.users[keep4], fvf["user_id"].to_numpy()), "fold population drift"
        yva_full = fvf["target"].to_numpy()
        if val_cap and keep4.size > val_cap:
            sel = np.sort(rng.choice(keep4.size, val_cap, replace=False)); keep4v = keep4[sel]; yva = yva_full[sel]
        else:
            keep4v = keep4; yva = yva_full
        log(f"    fold-{FOLD} eval on {keep4v.size:,} users")
    Pva, Sva = build(A, keep4v)

    Pmax = max([p.shape[1] for p in Ptr] + [Pva.shape[1]])
    Ptr = np.concatenate([padP(p, Pmax) for p in Ptr]); Pva = padP(Pva, Pmax)
    Str = np.concatenate(Str); ytr = np.concatenate(ytr).astype(np.float32)
    ss_mu, ss_sd = Str.mean(0), Str.std(0) + 1e-6
    Str = ((Str - ss_mu) / ss_sd).astype(np.float32); Sva = ((Sva - ss_mu) / ss_sd).astype(np.float32)
    extra = Str.shape[1] if use_ss else 0

    class GRUHead(nn.Module):
        def __init__(self, c_in, d=128, extra=0):
            super().__init__()
            self.proj = nn.Linear(c_in, d); self.gru = nn.GRU(d, d, 2, batch_first=True, dropout=0.1)
            self.head = nn.Sequential(nn.LayerNorm(d + extra), nn.Linear(d + extra, d), nn.GELU(), nn.Linear(d, 1))
        def forward(self, seq, ex=None):
            h = self.proj(seq); _, hn = self.gru(h); z = hn[-1]
            if ex is not None: z = torch.cat([z, ex], 1)
            return self.head(z).squeeze(1)

    c_in = Ptr.shape[2]
    model = GRUHead(c_in, extra=extra).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    def batches(P, S, y, bs, shuf):
        idx = np.random.permutation(len(P)) if shuf else np.arange(len(P))
        for i in range(0, len(P), bs):
            b = idx[i:i+bs]
            seq = torch.from_numpy(P[b].astype(np.float32)).to(dev)
            ex = torch.from_numpy(S[b]).to(dev) if extra else None
            yy = None if y is None else torch.from_numpy(y[b]).to(dev)
            yield seq, ex, yy

    log(f"    training GRU c_in={c_in} extra={extra} Pmax={Pmax} on {len(ytr):,} rows")
    model.train()
    for ep in range(epochs):
        tot = 0.0
        for seq, ex, yb in batches(Ptr, Str, ytr, 512, True):
            pred = model(seq, ex); loss = ((pred - yb) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(yb)
        log(f"      epoch {ep+1}/{epochs} mse {tot/len(ytr):.4f} [t+{(time.time()-t0)/60:.1f}m]")

    model.eval(); preds = []
    with torch.no_grad():
        for seq, ex, _ in batches(Pva, Sva, None, 1024, False):
            preds.append(model(seq, ex).float().cpu().numpy())
    pred = np.maximum(np.expm1(np.concatenate(preds)), 0.0)

    if args.submit:
        (ROOT / "subs").mkdir(exist_ok=True)
        pl.DataFrame({"user_id": sp.users[keep4v], "predict": pred}).write_csv(ROOT / "subs" / f"{args.exp_id}.csv")
        log(f"  wrote subs/{args.exp_id}.csv  n={pred.size:,}  mean={pred.mean():.3f}  "
            f"sum={pred.sum():,.0f}  total {(time.time()-t0)/60:.1f} min")
        return
    from metrics import rmsle
    L = np.log1p(yva); M = np.log1p(pred); sdL = float(L.std())
    def rho(a, b): return float(np.corrcoef(a, b)[0, 1])
    rf = rho(L, M)
    log(f"\n  FOLD {FOLD}: rho={rf:.5f}  cal RMSLE={sdL*np.sqrt(1-rf**2):.5f}  raw={rmsle(yva,pred):.5f}  (e0915 fold4 rho_B 0.651)")
    if val_cap is None:   # full fold -> admissibility vs the 9-member family
        E0120 = ["e0049","e0064","e0100","e0101","e0101s1","e0101s2","e0101s3","e0102","e0108"]
        def bl(e):
            d = pl.read_parquet(ROOT/"oof"/f"{e}.parquet").filter(pl.col("fold_id")==FOLD).sort("user_id")
            return np.log1p(np.maximum(d["y_pred"].to_numpy(), 0.0))
        B = np.mean([bl(e) for e in E0120], axis=0)
        def resid(x, z): z1 = np.column_stack([z, np.ones_like(z)]); b,*_ = np.linalg.lstsq(z1, x, rcond=None); return x - z1@b
        rp = rho(resid(L, B), resid(M, B)); r = rho(M, B); rhob = rho(L, B)
        log(f"  vs blend (rho={rhob:.5f}): r={r:.5f}  excess={rf-r*rhob:+.6f}  rho_partial={rp:+.5f}")
        out = ROOT/"runs"/"ag"; out.mkdir(parents=True, exist_ok=True)
        np.save(out/f"{args.exp_id}_f{FOLD}_oof.npy", pred.astype(np.float32))
        (out/f"{args.exp_id}_f{FOLD}.json").write_text(json.dumps({"exp_id":args.exp_id,"model":MODEL,
            "scale_stats":use_ss,"rho":rf,"r_vs_blend":r,"rho_partial":rp,"excess":rf-r*rhob,
            "runtime_min":round((time.time()-t0)/60,1)}, indent=2))
        log(f"  wrote runs/ag/{args.exp_id}_f{FOLD}.{{npy,json}}")
    log(f"  total {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
