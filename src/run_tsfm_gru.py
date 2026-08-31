#!/usr/bin/env python
"""
GRU over frozen TS-FM (MOMENT-1) representations -- two variants (IDEAS.md §I28):

  --mode patch     : GRU over MOMENT's per-PATCH token sequence ([N, n_patches, d]) -> learned
                     aggregation instead of the mean-pool that e0914 fed to LightGBM.
  --mode channels  : the raw 13-channel daily sequence (<=anchor) fed to a GRU, with MOMENT's
                     mean-pooled embedding broadcast as extra CONSTANT input channels.

Both are PER-ANCHOR models (one prediction per user from history <= A), so causal by construction
(we only read the output at the anchor) and directly comparable/blendable with the frozen-fold OOF.
Fold-4 feasibility; train on subsampled fit anchors, predict the full fold-4 population; score rho
and rho_partial vs the 9-member blend.
"""
from __future__ import annotations
import argparse, json, sys, time
from datetime import date
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from seqdata import build_seq_panel, CHANNELS         # noqa: E402

FOLD = 4
LAST_CLEAN_ANCHOR = date(2025, 10, 16)
N_FIT_ANCHORS = 6
TRAIN_USERS_PER_ANCHOR = 40_000
SEQ_LEN = 512
MODEL = "AutonLab/MOMENT-1-small"
TTM_REPO = "ibm-granite/granite-timeseries-ttm-r2"     # MLP-Mixer, NO attention (IDEAS 129)
SEED = 0
N_SCALE_PATCHES = 8                                    # for --scale-stats


def log(m): print(m, flush=True)


def cat_free(chunks):
    """np.concatenate without its 2x memory peak: allocate once, copy, drop each chunk as it lands.

    e0397 was OOM-killed at exactly this line -- MaxRSS 188.7 GB against a 180 GB request. MOMENT
    emits 64 patches x 512 dims per user, so 1.2M training rows is 78.6 GB of fp16 and plain
    concatenate needs both copies alive at once (157 GB). This keeps the peak at ~1x plus one
    chunk. (TTM is 8 x 192, i.e. 21x smaller, and never had the problem.)
    """
    n = sum(c.shape[0] for c in chunks)
    out = np.empty((n,) + chunks[0].shape[1:], chunks[0].dtype)
    i = 0
    while chunks:
        c = chunks.pop(0)
        out[i:i + c.shape[0]] = c
        i += c.shape[0]
        del c
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["patch", "channels"], required=True)
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--fold", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--submit", action="store_true",
                    help="predict at the real test anchor (2026-02-13) for the sample_submit "
                         "population and write subs/<exp_id>.csv (train clean-only, guard-safe)")
    ap.add_argument("--backbone", choices=["moment", "ttm"], default="moment",
                    help="'moment' reproduces e0914-e0916 byte-for-byte. 'ttm' swaps in "
                         "TinyTimeMixer, an MLP-Mixer with NO ATTENTION -- the most "
                         "architecturally orthogonal option on IDEAS 129's shortlist, against "
                         "MOMENT's T5 encoder (e0915) and Chronos-Bolt's T5 enc-dec (e0919), "
                         "both of which are transformers landing at r 0.96-0.98. e0391's bar "
                         "table says DECORRELATION is the cheap axis (bar 0.6553 -> 0.6441 -> "
                         "0.6324 as r goes 0.98 -> 0.96 -> 0.94) and e0392 showed LoRA moves r "
                         "the WRONG way, so a no-attention backbone is the one lever aimed at "
                         "the axis that is actually cheap. 385k params vs MOMENT-small's 35M.")
    ap.add_argument("--fit-anchors", type=int, default=N_FIT_ANCHORS,
                    help="training anchors for the head (default 6). Every TS-FM arm in this "
                         "project has used 6 x 40,000 = 240,000 training rows, while the causal "
                         "GRU that sits in the champion trains on 80,899,560 user-days. The "
                         "backbone is FROZEN, so only the head is data-limited -- and whether "
                         "these members are short on STRENGTH because the representation is "
                         "weak, or merely because the head saw 240k rows, has never been "
                         "separated. Raising this is the cheapest strength lever available and "
                         "should barely move r (the representation is unchanged).")
    ap.add_argument("--train-users", type=int, default=TRAIN_USERS_PER_ANCHOR,
                    help="users sampled per fit anchor (default 40,000)")
    ap.add_argument("--scale-stats", action="store_true",
                    help="append per-patch ABSOLUTE-SCALE statistics (mean/std/min/max over "
                         f"{N_SCALE_PATCHES} patches = {4 * N_SCALE_PATCHES} features) to the head. "
                         "IDEAS 129 calls this load-bearing and notes e0915 LACKS it: every FM "
                         "instance-normalises internally, which STRIPS absolute scale, while our "
                         "target is a 30-day SUM. Off by default so the backbone swap can be "
                         "tested as ONE change (CLAUDE.md 4.1).")
    ap.add_argument("--deterministic", action="store_true",
                    help="pin cudnn so two runs of one config agree. OFF BY DEFAULT so that "
                         "e0394-e0397 stay reproducible; turn it ON for any NEW single-fold "
                         "comparison. e0396 measured why this matters: re-running e0915's exact "
                         "config reproduced rho to 2e-05 but only corr 0.98997 on the individual "
                         "predictions, giving run-to-run noise on rho_partial of ~0.0014 -- the "
                         "SAME SIZE as the gaps between the arms being compared. Without this, a "
                         "fold-4 arm comparison is read against its own noise.")
    args = ap.parse_args()
    global FOLD
    FOLD = args.fold
    t0 = time.time()
    import torch, torch.nn as nn, polars as pl
    torch.manual_seed(SEED); np.random.seed(SEED)
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gmv_ch = [i for i, (nm, _, _) in enumerate(CHANNELS) if nm == "gmv"][0]
    backbone = args.backbone
    log(f"=== {args.exp_id} TS-FM GRU [{args.mode}] backbone={backbone} "
        f"scale_stats={args.scale_stats}, fold {FOLD}, device={dev} ===")

    if backbone == "moment":
        from momentfm import MOMENTPipeline
        fm = MOMENTPipeline.from_pretrained(MODEL, model_kwargs={"task_name": "embedding"})
        fm.init(); fm.to(dev); fm.eval()
    else:
        # TTM loads through transformers' PatchTSMixerModel -- granite-tsfm is NOT required
        # (verified by src/ttm_gate.py: 384,896 params, context 512, patch 64 -> 8 patches,
        # d_model 192, last_hidden_state [B, C, P, d]).
        from transformers import PatchTSMixerModel
        fm = PatchTSMixerModel.from_pretrained(TTM_REPO)
        fm.to(dev); fm.eval()
        log(f"    TTM: {sum(p.numel() for p in fm.parameters()):,} params  "
            f"ctx {fm.config.context_length}  patch {fm.config.patch_length}  d {fm.config.d_model}")

    # capture per-patch encoder output via a forward hook (for --mode patch)
    _cap = {}
    def _hook(_m, _i, o): _cap["h"] = (o.last_hidden_state if hasattr(o, "last_hidden_state") else o[0]).detach()
    if args.mode == "patch" and backbone == "moment":
        fm.encoder.register_forward_hook(_hook)

    def scale_feats(x, m):
        """Per-patch absolute-scale stats on the RAW (pre-instance-norm) window: the information
        every FM's internal normalisation throws away, and which a 30-day SUM target needs."""
        B = x.shape[0]
        xs = x.reshape(B, N_SCALE_PATCHES, SEQ_LEN // N_SCALE_PATCHES)
        ms = m.reshape(B, N_SCALE_PATCHES, SEQ_LEN // N_SCALE_PATCHES)
        n = np.maximum(ms.sum(2), 1.0)
        mu = (xs * ms).sum(2) / n
        sd = np.sqrt(np.maximum(((xs - mu[:, :, None]) ** 2 * ms).sum(2) / n, 0.0))
        lo = np.where(ms > 0, xs, np.inf).min(2); hi = np.where(ms > 0, xs, -np.inf).max(2)
        lo[~np.isfinite(lo)] = 0.0; hi[~np.isfinite(hi)] = 0.0
        return np.concatenate([mu, sd, lo, hi], 1).astype(np.float32)   # [B, 4*k]

    def fm_forward(series_list, want_patch):
        """Return pooled [N,d], (if want_patch) per-patch [N,P,d], and scale stats [N,4k]."""
        pooled, patches, scales = [], [], []
        for i in range(0, len(series_list), 256):
            chunk = series_list[i:i + 256]; B = len(chunk)
            x = np.zeros((B, 1, SEQ_LEN), np.float32); m = np.zeros((B, SEQ_LEN), np.float32)
            for j, s in enumerate(chunk):
                L = min(len(s), SEQ_LEN); x[j, 0, SEQ_LEN - L:] = s[-L:]; m[j, SEQ_LEN - L:] = 1.0
            if args.scale_stats:
                scales.append(scale_feats(x[:, 0, :], m))
            with torch.no_grad():
                if backbone == "moment":
                    out = fm(x_enc=torch.from_numpy(x).to(dev),
                             input_mask=torch.from_numpy(m).to(dev))
                    emb = out.embeddings.float()
                    h = _cap["h"].float() if want_patch else None      # [B, P, d]
                else:
                    # TTM takes [B, T, C]; last_hidden_state is [B, C, P, d] -> squeeze C=1
                    out = fm(past_values=torch.from_numpy(x).to(dev).transpose(1, 2))
                    h = out.last_hidden_state.float().squeeze(1)       # [B, P, d]
                    emb = h.mean(1)                                    # mean-pool over patches
            pooled.append(emb.cpu().numpy())
            if want_patch:
                patches.append(h.cpu().numpy().astype(np.float16))
            if i % 5120 == 0: log(f"      fm {i+B:,}/{len(series_list):,}  [t+{(time.time()-t0)/60:.1f}m]")
        P = np.concatenate(patches) if want_patch else None
        S = np.concatenate(scales) if args.scale_stats else None
        return np.concatenate(pooled), P, S

    sp = build_seq_panel()
    Xg = sp.X[:, gmv_ch, :].astype(np.float32)
    Xall = sp.X.astype(np.float32) if args.mode == "channels" else None
    spec = json.loads((ROOT / "data" / "fold_spec.json").read_text())
    if args.submit:
        A = sp.idx(date(2026, 2, 13))                      # the real test anchor (= p.dmax)
        lc = sp.idx(LAST_CLEAN_ANCHOR)                      # 2025-10-16, guard-zone boundary
        tr_anchors = sorted([a for a in range(lc, 120, -30)][:args.fit_anchors])
        log(f"    SUBMIT mode: test anchor {sp.day(A)}, {len(tr_anchors)} clean train anchors "
            f"{sp.day(tr_anchors[0])}..{sp.day(tr_anchors[-1])}")
    else:
        fs = spec["folds"][FOLD]; A = sp.idx(date.fromisoformat(fs["valid_anchor"]))
        tr_anchors = [sp.idx(date.fromisoformat(x)) for x in fs["train_anchors"]][-args.fit_anchors:]
    rng = np.random.default_rng(SEED)

    def build_rows(anchor, users):
        series = [Xg[u, :anchor + 1] for u in users]
        pooled, patch, scale = fm_forward(series, args.mode == "patch")
        if args.mode == "patch":
            return patch, scale                              # [N, P, d], [N, 4k] or None
        # channels: raw 13-ch daily seq up to anchor (fixed window = last SEQ_LEN days) + pooled bcast
        T = min(anchor + 1, SEQ_LEN)
        raw = np.zeros((len(users), sp.n_ch, SEQ_LEN), np.float16)
        for j, u in enumerate(users):
            raw[j, :, SEQ_LEN - T:] = Xall[u, :, anchor + 1 - T:anchor + 1]
        return raw, pooled.astype(np.float16), scale

    # ---- assemble train
    Xtr, Ptr, Str, ytr = [], [], [], []
    for a in tr_anchors:
        keep = np.where(sp.pop[:, a])[0]
        if keep.size > args.train_users:
            keep = np.sort(rng.choice(keep, args.train_users, replace=False))
        log(f"    fit anchor {a}: {keep.size:,} users")
        r = build_rows(a, keep)
        if args.mode == "patch": Ptr.append(r[0])
        else: Xtr.append(r[0]); Ptr.append(r[1])
        if args.scale_stats: Str.append(r[-1])
        ytr.append(sp.Y[keep, a])
    ytr = np.concatenate(ytr).astype(np.float32)
    Str = cat_free(Str) if args.scale_stats else None

    # ---- eval / test population  (assert alignment BEFORE the expensive extraction)
    keep4 = np.where(sp.pop[:, A])[0]
    if args.submit:
        ss = pl.read_csv(ROOT / "data" / "sample_submit.csv")
        assert np.array_equal(sp.users[keep4], ss["user_id"].to_numpy()), \
            "test population != sample_submit user order"
        log(f"    SUBMIT: {keep4.size:,} test users at anchor {sp.day(A)}")
        r4 = build_rows(A, keep4); yva = None
    else:
        folds = pl.read_parquet(ROOT / "data" / "folds.parquet")
        fvf = folds.filter(pl.col("fold_id") == FOLD).sort("user_id")
        assert np.array_equal(sp.users[keep4], fvf["user_id"].to_numpy())
        log(f"    fold-{FOLD}: {keep4.size:,} users")
        r4 = build_rows(A, keep4); yva = fvf["target"].to_numpy()

    # ---- model
    class GRUHead(nn.Module):
        def __init__(self, c_in, d=128, extra=0):
            super().__init__()
            self.proj = nn.Linear(c_in, d); self.gru = nn.GRU(d, d, 2, batch_first=True, dropout=0.1)
            self.head = nn.Sequential(nn.LayerNorm(d + extra), nn.Linear(d + extra, d), nn.GELU(), nn.Linear(d, 1))
        def forward(self, seq, extra=None):
            h = self.proj(seq); _, hn = self.gru(h); z = hn[-1]
            if extra is not None: z = torch.cat([z, extra], 1)
            return self.head(z).squeeze(1)

    Sva = r4[-1] if args.scale_stats else None
    if args.mode == "patch":
        Ptr = cat_free(Ptr); Pva = r4[0]
        c_in, extra = Ptr.shape[2], (Str.shape[1] if args.scale_stats else 0)
        def batches(pack, y, bs, shuf):
            P, S = pack
            idx = np.random.permutation(len(P)) if shuf else np.arange(len(P))
            for i in range(0, len(P), bs):
                b = idx[i:i+bs]
                yield (torch.from_numpy(P[b].astype(np.float32)).to(dev),
                       None if S is None else torch.from_numpy(S[b].astype(np.float32)).to(dev),
                       None if y is None else torch.from_numpy(y[b]).to(dev))
    else:
        Xtr = cat_free(Xtr); Ptr = cat_free(Ptr)                  # raw [N,C,T], pooled [N,512]
        Xva, Pva = r4[0], r4[1]
        c_in, extra = sp.n_ch, Ptr.shape[1] + (Str.shape[1] if args.scale_stats else 0)
        def batches(pack, y, bs, shuf):
            X, Pemb, S = pack
            idx = np.random.permutation(len(X)) if shuf else np.arange(len(X))
            for i in range(0, len(X), bs):
                b = idx[i:i+bs]
                seq = torch.from_numpy(X[b].astype(np.float32)).to(dev).transpose(1, 2)  # [B,T,C]
                ex = torch.from_numpy(Pemb[b].astype(np.float32)).to(dev)
                if S is not None:
                    ex = torch.cat([ex, torch.from_numpy(S[b].astype(np.float32)).to(dev)], 1)
                yield (seq, ex, None if y is None else torch.from_numpy(y[b]).to(dev))

    model = GRUHead(c_in, extra=extra).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    tr_pack = (Ptr, Str) if args.mode == "patch" else (Xtr, Ptr, Str)
    va_pack = (Pva, Sva) if args.mode == "patch" else (Xva, Pva, Sva)
    log(f"    training GRU ({args.mode}) c_in={c_in} extra={extra} on {len(ytr):,} rows")
    model.train()
    for ep in range(args.epochs):
        tot = 0.0
        for seq, ex, yb in batches(tr_pack, ytr, 512, True):
            pred = model(seq, ex); loss = ((pred - yb) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(yb)
        log(f"      epoch {ep+1}/{args.epochs} mse {tot/len(ytr):.4f} [t+{(time.time()-t0)/60:.1f}m]")

    model.eval(); preds = []
    with torch.no_grad():
        for seq, ex, _ in batches(va_pack, None, 1024, False):
            preds.append(model(seq, ex).float().cpu().numpy())
    pred = np.maximum(np.expm1(np.concatenate(preds)), 0.0)

    if args.submit:
        (ROOT / "subs").mkdir(exist_ok=True)
        pl.DataFrame({"user_id": sp.users[keep4], "predict": pred}).write_csv(ROOT / "subs" / f"{args.exp_id}.csv")
        log(f"  wrote subs/{args.exp_id}.csv  n={pred.size:,}  mean={pred.mean():.3f}  "
            f"sum={pred.sum():,.0f}  total {(time.time()-t0)/60:.1f} min")
        return
    from metrics import rmsle
    L = np.log1p(yva); M = np.log1p(pred); sdL = float(L.std())
    def rho(a, b): return float(np.corrcoef(a, b)[0, 1])
    rf = rho(L, M)
    E0120 = ["e0049", "e0064", "e0100", "e0101", "e0101s1", "e0101s2", "e0101s3", "e0102", "e0108"]
    def bl(e):
        d = pl.read_parquet(ROOT/"oof"/f"{e}.parquet").filter(pl.col("fold_id")==FOLD).sort("user_id")
        return np.log1p(np.maximum(d["y_pred"].to_numpy(), 0.0))
    B = np.mean([bl(e) for e in E0120], axis=0)
    def resid(x, z): z1=np.column_stack([z,np.ones_like(z)]); b,*_=np.linalg.lstsq(z1,x,rcond=None); return x-z1@b
    rp = rho(resid(L, B), resid(M, B)); r = rho(M, B); rhob = rho(L, B)
    log(f"\n  FOLD {FOLD} [{args.mode}]: rho={rf:.5f}  cal RMSLE={sdL*np.sqrt(1-rf**2):.5f}  raw={rmsle(yva,pred):.5f}")
    log(f"  vs 9-blend (rho={rhob:.5f}): r={r:.5f}  excess={rf-r*rhob:+.6f}  rho_partial={rp:+.5f}")

    # ---- and against the FULL CHAMPION, which is the number that decides (e0391) ----------
    # The 9-member family EXCLUDES the usercv slot that is actually in the champion, so judging a
    # candidate against it flatters the candidate -- exactly the E1 substitution. Judge here.
    rp_c = r_c = rhob_c = float("nan"); margin = float("nan")
    try:
        # ⚠ The usercv OOF covers a DIFFERENT fold-4 population than the seq/gbdt OOF
        # (238,847 vs 225,431 rows), so these must be joined on user_id -- assuming a shared
        # row order raised `operands could not be broadcast` on the first run of this block.
        uid_va = fvf["user_id"].to_numpy()
        def bl_keyed(e):
            d = (pl.read_parquet(ROOT/"oof"/f"{e}.parquet").filter(pl.col("fold_id") == FOLD)
                 .select(["user_id", "y_pred"]))
            return d.rename({"y_pred": e})
        tab = pl.DataFrame({"user_id": uid_va, "_L": L, "_M": M})
        for e in ("e0266", "e0064", "e0100", "e0101", "e0101s1", "e0101s2", "e0101s3",
                  "e0102", "e0108", "usercv_full_h48_e0272"):
            tab = tab.join(bl_keyed(e), on="user_id", how="inner")
        log(f"    champion-aligned population: {tab.height:,} of {len(uid_va):,} fold-{FOLD} users")
        lgc = lambda e: np.log1p(np.maximum(tab[e].to_numpy(), 0.0))
        CH = (0.20 * np.mean([lgc(e) for e in ("e0266", "e0064")], axis=0)
              + 0.38 * np.mean([lgc(e) for e in ("e0100", "e0101", "e0101s1", "e0101s2",
                                                 "e0101s3", "e0102", "e0108")], axis=0)
              + 0.42 * lgc("usercv_full_h48_e0272"))
        L, M = tab["_L"].to_numpy(), tab["_M"].to_numpy()
        rhob_c = rho(L, CH); r_c = rho(M, CH)
        rf = rho(L, M)                       # recomputed on the aligned population
        rp_c = rho(resid(L, CH), resid(M, CH))
        need = rhob_c * r_c + np.sqrt(max((rhob_c + 0.0005) ** 2 - rhob_c ** 2, 0.0) * (1 - r_c ** 2))
        margin = rf - need
        log(f"  vs CHAMPION (rho={rhob_c:.5f}): r={r_c:.5f}  rho_partial={rp_c:+.5f}  "
            f"bar={need:.5f}  MARGIN={margin:+.5f}  {'CLEARS' if margin > 0 else 'short'}")
        log(f"    reference fold-4 margins (e0391/e0392): e0915 -0.0058, e0919 -0.0042 "
            f"| fold-4 -> pooled shift is about -0.008, so a fold-4 rho_partial must exceed "
            f"~+0.008 vs the champion merely to reach ZERO pooled.")
    except Exception as e:
        log(f"  [champion reference unavailable: {type(e).__name__}: {e}]")
    out = ROOT/"runs"/"ag"; out.mkdir(parents=True, exist_ok=True)
    np.save(out/f"{args.exp_id}_f{FOLD}_oof.npy", pred.astype(np.float32))
    (out/f"{args.exp_id}_f{FOLD}.json").write_text(json.dumps({"exp_id":args.exp_id,"mode":args.mode,
        "model":(MODEL if backbone=="moment" else TTM_REPO),"backbone":backbone,
        "scale_stats":bool(args.scale_stats),
        "rho":rf,"r_vs_blend":r,"rho_partial":rp,"excess":rf-r*rhob,
        "rho_vs_champion":rhob_c,"r_vs_champion":r_c,"rho_partial_vs_champion":rp_c,
        "margin_vs_champion_bar":margin,
        "runtime_min":round((time.time()-t0)/60,1)}, indent=2))
    log(f"  wrote runs/ag/{args.exp_id}_f4.{{npy,json}}  total {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
