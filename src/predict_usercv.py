#!/usr/bin/env python
"""
CAUSAL_EXP.md §7 -- test inference for the user-split variants.

    python src/predict_usercv.py --variant extra --exp-id e0142 --epochs 18

Recipe, deliberately the same one the user-split CV measured, so the submitted number is
comparable to the logged unseen-user score:

  * train on ALL 250,000 users, same anchors and same tmask as the CV (burn-in 14 from first
    seen, last anchor 2026-01-14);
  * epochs = the MEDIAN best epoch over that variant's 15 CV runs.  This mirrors
    src/predict.py's "median best_iteration from CV" convention -- the CV's early stopping
    cannot be reused here because there is no held-out user slice once we train on everyone;
  * scaling stats fit on the training population, which is now all users;
  * one forward pass over the whole 409-day series, prediction read at index 408 = 2026-02-13
    only (§7.3: earlier steps have strictly less information).

ONE CORRECTION TO §7.4.  The doc says "expm1 to get mean daily GMV; multiply by horizon for
the total".  That is right for `target_agg="mean"`.  We train on `sum`, so `expm1(pred)` IS
the 30-day total and multiplying by 30 would inflate every prediction 30x.  Asserted below
against the observed scale rather than left to a comment.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from metrics import gini                                                     # noqa: E402
from run_usercv import GUARD_START, CausalXformer, GRUForecaster            # noqa: E402
from usercv_features import (HORIZON, Raw, build_features, build_target,     # noqa: E402
                             build_tmask, flag_channels, max_anchor)


def log(m: str) -> None:
    print(m, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    choices=["gmv_only", "full", "extra", "extra_nocal", "behav"])
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--epochs", type=int, default=0, help="0 = median best epoch from the CV json")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--model", default="gru", choices=["gru", "lstm", "transformer"],
                    help="must match the CV run whose median best-epoch is passed to "
                         "--epochs; the full-data model has to be the same architecture the "
                         "epoch count was measured on.")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--sched", default="none", choices=["none", "cosine"])
    ap.add_argument("--sched-tmax", type=int, default=0,
                    help="cosine T_max. Set it to the CV run's --epochs, not to the number of "
                         "epochs trained here: the confirmed model stopped PART-WAY down a "
                         "150-epoch cosine, so annealing fully to zero by epoch 71 would be a "
                         "different learning-rate trajectory from the one that was measured.")
    ap.add_argument("--feat-drop", type=float, default=0.0)
    ap.add_argument("--mixup", default="none", choices=["none", "naive"],
                    help="synthetic users by interpolation; measured -0.00047 (5/5 folds) "
                         "and it SUBSUMES seed averaging (1 seed == 3 seeds)")
    ap.add_argument("--mixup-alpha", type=float, default=0.2)
    ap.add_argument("--seeds", type=int, default=3,
                    help="models trained on the FULL data, averaged in log space")
    ap.add_argument("--guard-clean", action="store_true",
                    help="RETRACT the training anchor grid out of the guaranteed-activity zone: "
                         "cap it at the last anchor whose 30-day target window ends before "
                         f"{GUARD_START} (= 2025-10-16), instead of the default 2026-01-14. "
                         "The default INCLUDES guard-zone anchors, whose target windows have "
                         "activity guaranteed by the panel's construction while the real test "
                         "window (2026-02-14+) does not -- the mismatch e0361 measured at "
                         "+0.00035 when the seq half was extended INTO the zone. This is the "
                         "inverse operation on the usercv slot. Costs ~25% of the supervised "
                         "days, and the ones nearest the test anchor, so it is not free.")
    args = ap.parse_args()

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = args.epochs
    if epochs <= 0:
        tag = args.variant + ("_mixnaive" if args.mixup == "naive" else "")
        cv = json.loads((ROOT / "reports" / "eda" / f"usercv_{tag}.json").read_text())
        epochs = int(np.median([r["best_epoch"] for r in cv["runs"]]))
        log(f"  epochs = {epochs} (median best epoch over {len(cv['runs'])} CV runs; "
            f"unseen-user CV {cv['unseen_rmsle_mean']:.5f})")

    raw = Raw()
    last_anchor = max_anchor(raw)
    if args.guard_clean:
        clean_max = raw.idx(GUARD_START) - 1 - HORIZON     # same constant as run_usercv.py:390
        assert clean_max < last_anchor, "guard-clean cap is not a retraction"
        log(f"  --guard-clean: last training anchor {raw.day(last_anchor)} -> "
            f"{raw.day(clean_max)}  (t {last_anchor} -> {clean_max})")
        last_anchor = clean_max
    Y = build_target(raw, "sum")
    M = build_tmask(raw, last_anchor, burn_in=14, trim_to_first_seen=True)
    if args.guard_clean:
        log(f"  supervised user-days: {int(M.sum()):,} "
            f"({100 * M.sum() / build_tmask(raw, max_anchor(raw), 14, True).sum():.1f}% "
            f"of the default grid)")
    Xn, names = build_features(raw, args.variant)
    is_flag = flag_channels(names)
    log(f"\n=== {args.exp_id} [{args.variant}] : {len(names)} features, {epochs} epochs ===")

    sub = Xn[::37].astype(np.float32)
    mu = sub.mean(axis=(0, 1)); sd = np.maximum(sub.std(axis=(0, 1)), 1e-3)
    mu[is_flag] = 0.0; sd[is_flag] = 1.0
    del sub

    Xg = torch.from_numpy(Xn).to(device)
    Yg = torch.from_numpy(Y).to(device)
    Mg = torch.from_numpy(M).to(device)
    mu_g = torch.from_numpy(mu).to(device); sd_g = torch.from_numpy(sd).to(device)
    all_u = torch.arange(raw.n, device=device)
    steps = math.ceil(raw.n / args.batch)

    def amp():
        return (torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device == "cuda"
                else torch.autocast(device_type="cpu", enabled=False))

    tai = raw.T - 1
    assert raw.day(tai) == raw.dmax, "inference index is not the last observed day"
    ckdir = ROOT / "runs" / "usercv"; ckdir.mkdir(parents=True, exist_ok=True)

    # Seeds are averaged in LOG space.  The head already estimates E[log1p(y)|x], which is what
    # RMSLE's optimal point prediction is, so the mean of the outputs is the right average --
    # averaging after expm1 would target E[y|x] instead.  Full data per model, not fold models:
    # the folds existed to make validation honest and each of them sees only 80% of the users.
    logits = []
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        model = (CausalXformer(len(names), args.hidden, args.layers, 4, args.dropout)
                 if args.model == "transformer" else
                 GRUForecaster(len(names), args.hidden, args.layers, args.dropout,
                               cell=args.model)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        sch = (torch.optim.lr_scheduler.CosineAnnealingLR(
                   opt, T_max=(args.sched_tmax or epochs))
               if args.sched == 'cosine' else None)
        g = torch.Generator().manual_seed(seed)
        for ep in range(epochs):
            model.train()
            perm = all_u[torch.randperm(raw.n, generator=g).to(device)]
            num = den = 0.0
            for i in range(steps):
                b = perm[i * args.batch:(i + 1) * args.batch]
                if args.mixup == "none":
                    xb, tb, msk = Xg[b].float(), Yg[b], Mg[b]
                else:
                    b2 = b[torch.randperm(b.numel(), generator=g).to(device)]
                    lam = float(np.random.beta(args.mixup_alpha, args.mixup_alpha))
                    xb = lam * Xg[b].float() + (1 - lam) * Xg[b2].float()
                    tb = lam * Yg[b] + (1 - lam) * Yg[b2]
                    msk = Mg[b] & Mg[b2]
                m = msk.float()
                if float(m.sum()) == 0:
                    continue
                xb = (xb - mu_g) / sd_g
                if args.feat_drop > 0:
                    keep = (torch.rand(1, 1, len(names), device=device)
                            > args.feat_drop).float()
                    xb = xb * keep / max(1e-6, 1 - args.feat_drop)
                with amp():
                    out = model(xb)
                loss = ((out.float() - tb) ** 2 * m).sum() / m.sum()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                num += float(loss.item()) * float(m.sum()); den += float(m.sum())
            if sch is not None:
                sch.step()
            log(f"    seed {seed} epoch {ep + 1:>3d}/{epochs}  train {(num / den) ** 0.5:.5f}  "
                f"[{(time.time() - t0) / 60:.1f}m]")
        model.eval()
        pl_ = np.empty(raw.n, np.float64)
        with torch.no_grad():
            for i in range(0, raw.n, 1024):
                b = all_u[i:i + 1024]
                with amp():
                    out = model((Xg[b].float() - mu_g) / sd_g).float()
                pl_[i:i + b.numel()] = out[:, tai].cpu().numpy()
        logits.append(pl_)
        ck = ckdir / f"{args.exp_id}_full_s{seed}.pt"
        torch.save({"state_dict": model.state_dict(), "variant": args.variant,
                    "exp_id": args.exp_id, "seed": seed, "epochs": epochs,
                    "n_features": len(names), "feature_names": names,
                    "mu": mu, "sd": sd, "trained_on": "all 250k users"}, ck)
        log(f"    saved runs/usercv/{ck.name}")
    L = np.stack(logits)
    if args.seeds > 1:
        c = np.corrcoef(L)
        log(f"    seed log-prediction correlations: "
            f"{[round(float(c[i, j]), 4) for i in range(len(L)) for j in range(i + 1, len(L))]}")
    pred_log = L.mean(0)
    pred = np.maximum(np.expm1(pred_log), 0.0)      # target is the SUM -- do NOT scale by 30

    ss = pl.read_csv(ROOT / "data" / "sample_submit.csv")
    assert np.array_equal(raw.users, ss["user_id"].to_numpy()), "user order differs from sample"
    assert np.isfinite(pred).all() and (pred >= 0).all()
    # scale guard for the §7.4 correction: a x30 slip would put the mean near 1300
    assert 10.0 < pred.mean() < 200.0, f"prediction scale looks wrong: mean {pred.mean():.1f}"

    (ROOT / "subs").mkdir(exist_ok=True)
    out = ROOT / "subs" / f"{args.exp_id}.csv"
    pl.DataFrame({"user_id": raw.users, "predict": pred}).write_csv(out)

    p30 = raw.col["gmv"][:, tai - 29:tai + 1].sum(1)
    log(f"\n  wrote {out.relative_to(ROOT)}")
    log(f"  {'':22s} {'sum':>16s} {'mean':>10s} {'zero share':>11s} {'gini':>8s}")
    for nm, v in [(f"{args.exp_id} ({args.variant})", pred), ("last-30d observed", p30)]:
        log(f"  {nm:22s} {v.sum():>16,.0f} {v.mean():>10.2f} "
            f"{100 * (v == 0).mean():>10.2f}% {gini(v):>8.4f}")
    log(f"  runtime {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
