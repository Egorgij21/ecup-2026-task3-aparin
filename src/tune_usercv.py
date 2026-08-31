#!/usr/bin/env python
"""
Optuna search over the GRU's regularisation and schedule: which regime trains longest?

    python src/tune_usercv.py --trials 6 --study usercv_reg --fold 0

THE QUESTION.  `lr=1e-3, wd=1e-5, dropout=0.1, 60 epochs` has been fixed since the first
CAUSAL_EXP run, and `lr=0.05, num_leaves=63, ... 178 rounds` has been fixed on the GBDT side
since e0001.  Eighty-eight logged experiments, zero hyperparameter tuning -- optuna has been
installed the whole time and never used.  This is the only large lever never pulled.

WHY IT MIGHT PAY, given everything else measured flat.  Mixup moved the best epoch from 13-25
to 17-41 AND improved the score by -0.00047 (5/5 folds).  That is the signature of a model
whose useful training window is bounded by regularisation rather than by information: relax the
bound and it keeps learning.  If that generalises, a stronger regime should push the best epoch
further out and take the score with it.  If instead every configuration tops out at the same
score with the best epoch merely drifting, the bound is informational and this is finished --
which is a result worth having, because it closes the last open lever.

WHAT IS SEARCHED.  Only knobs that trade capacity against overfitting; the features, folds,
target and metric are all frozen.  `max_epochs` is deliberately generous (300) with patience 25,
so a genuinely non-overfitting regime is free to run long -- the search can DISCOVER a
long-training configuration rather than being capped into a short one.

REPORTED PER TRIAL: the best held-out-user RMSLE *and* the epoch it occurred at.  The second
number is the one that answers the question.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rho_decomp import auc                                                    # noqa: E402
from run_usercv import CausalXformer, GRUForecaster                           # noqa: E402
from usercv_features import (Raw, build_features, build_target, build_tmask,  # noqa: E402
                             flag_channels, hash_fold, max_anchor)


def log(m: str) -> None:
    print(m, flush=True)


class Ctx:
    """Data loaded once and shared by every trial."""

    def __init__(self, variant: str, fold: int, device: str):
        raw = Raw()
        self.raw = raw
        Y = build_target(raw, "sum")
        M = build_tmask(raw, max_anchor(raw), burn_in=14, trim_to_first_seen=True)
        X, self.names = build_features(raw, variant)
        is_flag = flag_channels(self.names)
        fo = hash_fold(raw.users)
        tr_u, va_u = np.flatnonzero(fo != fold), np.flatnonzero(fo == fold)
        sub = X[tr_u[::37]].astype(np.float32)
        mu = sub.mean(axis=(0, 1)); sd = np.maximum(sub.std(axis=(0, 1)), 1e-3)
        mu[is_flag] = 0.0; sd[is_flag] = 1.0
        del sub
        self.device = device
        self.Xg = torch.from_numpy(X).to(device)
        self.Yg = torch.from_numpy(Y).to(device)
        self.Mg = torch.from_numpy(M).to(device)
        self.mu = torch.from_numpy(mu).to(device)
        self.sd = torch.from_numpy(sd).to(device)
        self.tr = torch.from_numpy(tr_u).to(device)
        self.va = torch.from_numpy(va_u).to(device)
        log(f"  ctx: {len(self.names)} features | {tr_u.size:,} train / {va_u.size:,} unseen "
            f"users | fold {fold}")


def run_trial(ctx: Ctx, hp: dict, max_epochs: int, patience: int) -> tuple[float, int, float]:
    torch.manual_seed(hp["seed"])
    n_feat = len(ctx.names)
    model = (GRUForecaster(n_feat, hp["hidden"], hp["layers"], hp["dropout"])
             if hp["arch"] == "gru" else
             CausalXformer(n_feat, hp["hidden"], hp["layers"], 4, hp["dropout"])).to(ctx.device)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    if hp["sched"] == "plateau":
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    else:
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    g = torch.Generator().manual_seed(hp["seed"])
    bs = hp["batch"]

    def amp():
        return (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if ctx.device == "cuda" else torch.autocast(device_type="cpu", enabled=False))

    best, best_ep, bad, best_auc = float("inf"), -1, 0, 0.0
    for ep in range(max_epochs):
        model.train()
        perm = ctx.tr[torch.randperm(ctx.tr.numel(), generator=g).to(ctx.device)]
        for i in range(0, perm.numel(), bs):
            b = perm[i:i + bs]
            if hp["mixup_alpha"] > 0:
                b2 = b[torch.randperm(b.numel(), generator=g).to(ctx.device)]
                lam = float(np.random.beta(hp["mixup_alpha"], hp["mixup_alpha"]))
                xb = lam * ctx.Xg[b].float() + (1 - lam) * ctx.Xg[b2].float()
                tb = lam * ctx.Yg[b] + (1 - lam) * ctx.Yg[b2]
                msk = ctx.Mg[b] & ctx.Mg[b2]
            else:
                xb, tb, msk = ctx.Xg[b].float(), ctx.Yg[b], ctx.Mg[b]
            xb = (xb - ctx.mu) / ctx.sd
            if hp["feat_drop"] > 0:                     # drop whole feature CHANNELS
                keep = (torch.rand(1, 1, n_feat, device=ctx.device) > hp["feat_drop"]).float()
                xb = xb * keep / max(1e-6, 1 - hp["feat_drop"])
            m = msk.float()
            if float(m.sum()) == 0:
                continue
            with amp():
                out = model(xb)
            loss = ((out.float() - tb) ** 2 * m).sum() / m.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        num = den = 0.0
        ps, zs = [], []
        with torch.no_grad():
            for i in range(0, ctx.va.numel(), 2048):
                b = ctx.va[i:i + 2048]
                with amp():
                    o = model((ctx.Xg[b].float() - ctx.mu) / ctx.sd).float()
                mm = ctx.Mg[b].float()
                num += float((((o - ctx.Yg[b]) ** 2) * mm).sum()); den += float(mm.sum())
                sel = ctx.Mg[b]
                ps.append(o[sel].cpu().numpy()); zs.append((ctx.Yg[b][sel] > 0).cpu().numpy())
        v = (num / den) ** 0.5
        sch.step(v) if hp["sched"] == "plateau" else sch.step()
        if v < best - 1e-6:
            best, best_ep, bad = v, ep + 1, 0
            best_auc = auc(np.concatenate(ps), np.concatenate(zs).astype(float))
        else:
            bad += 1
            if bad >= patience:
                break
    return best, best_ep, best_auc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--study", default="usercv_reg")
    ap.add_argument("--variant", default="full")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--max-epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=25)
    args = ap.parse_args()
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    log(f"\n=== optuna: {args.study}, fold {args.fold}, up to {args.max_epochs} epochs "
        f"(patience {args.patience}) ===")
    ctx = Ctx(args.variant, args.fold, device)

    # baseline for reference: lr 1e-3, wd 1e-5, dropout 0.1, no mixup -> 1.74033 on fold 0
    def objective(trial):
        hp = dict(
            arch="gru",
            hidden=trial.suggest_categorical("hidden", [64, 128, 256]),
            layers=trial.suggest_int("layers", 1, 3),
            dropout=trial.suggest_float("dropout", 0.0, 0.5),
            wd=trial.suggest_float("wd", 1e-6, 1e-2, log=True),
            lr=trial.suggest_float("lr", 2e-4, 4e-3, log=True),
            mixup_alpha=trial.suggest_float("mixup_alpha", 0.0, 1.0),
            feat_drop=trial.suggest_float("feat_drop", 0.0, 0.3),
            sched=trial.suggest_categorical("sched", ["plateau", "cosine"]),
            batch=trial.suggest_categorical("batch", [128, 256, 512]),
            seed=0)
        v, ep, a = run_trial(ctx, hp, args.max_epochs, args.patience)
        trial.set_user_attr("best_epoch", ep)
        trial.set_user_attr("auc", a)
        log(f"  trial {trial.number:>3d}  RMSLE {v:.5f}  best_epoch {ep:>3d}  AUC {a:.5f}  "
            f"| h{hp['hidden']} L{hp['layers']} do{hp['dropout']:.2f} wd{hp['wd']:.1e} "
            f"lr{hp['lr']:.1e} mix{hp['mixup_alpha']:.2f} fd{hp['feat_drop']:.2f} "
            f"{hp['sched']} b{hp['batch']}  [{(time.time()-t0)/60:.0f}m]")
        return v

    st = optuna.create_study(direction="minimize", study_name=args.study,
                             storage=f"sqlite:///{ROOT}/reports/{args.study}.db",
                             load_if_exists=True)
    st.optimize(objective, n_trials=args.trials)
    df = sorted(st.trials, key=lambda t: t.value if t.value else 9e9)
    log(f"\n  === best 5 of {len(st.trials)} trials ===")
    log(f"  {'RMSLE':>9s} {'epoch':>6s} {'AUC':>9s}  params")
    for t in df[:5]:
        log(f"  {t.value:>9.5f} {t.user_attrs.get('best_epoch', -1):>6d} "
            f"{t.user_attrs.get('auc', 0):>9.5f}  {t.params}")
    log(f"\n  baseline on this fold (fixed hyperparameters): 1.74033 at epoch 13")
    log(f"  correlation between trial score and best_epoch: "
        f"{np.corrcoef([t.value for t in st.trials if t.value], [t.user_attrs.get('best_epoch', 0) for t in st.trials if t.value])[0, 1]:+.3f}")
    log(f"  (negative = better configurations train LONGER, which is the hypothesis)")
    (ROOT / "reports" / "eda" / f"{args.study}.json").write_text(json.dumps(
        [{"value": t.value, **t.params, **t.user_attrs} for t in df], indent=2))


if __name__ == "__main__":
    main()
