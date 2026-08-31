#!/usr/bin/env python
"""
Entrypoint for the `seq` approach: causal sequence model on the raw daily panel.

    python src/run_seq.py --config configs/e0100_seq_tcn.yaml            # CV on frozen folds
    python src/run_seq.py --config configs/e0100_seq_tcn.yaml --mode submit

Same protocol contract as src/run.py, and deliberately the same *shape* of output so the two
approaches are comparable through experiments.csv (README.md):

  * folds come from data/folds.parquet and are never recomputed (rule 3)
  * the metric comes from src/metrics.py (rule 4)
  * the naive `geo3` reference is recomputed on the identical population every fold, so
    `delta` is an exact paired comparison rather than a copied number
  * OOF lands in oof/<exp_id>.parquet with run.py's schema, which is what makes a blend
    against e0049/e0064 a five-line script instead of a re-run
  * a row is appended to experiments.csv before anything is reported (rule 5)

CV and submission share ONE training function.  src/predict.py had to grow a duplicated copy
of run.py's column-selection logic and the comment there records what that cost; here `--mode
submit` differs from `--mode cv` only in which day range trains and which day is scored.

WHAT TRAINS ON WHAT
  fold k, valid anchor A:  every day t in [89, A-30] with the user in-population at t.
     - t >= 89 mirrors the frozen min_history_days=90; e0025/e0026 measured shorter histories
       as actively harmful (+0.00026 / +0.00073), so this is not an arbitrary floor.
     - t <= A-30 means every training target window ends on or before A, so no training label
       overlaps the validation label.  This is the frozen fold rule, applied per-day.
  submit mode:            every day t in [89, 288].  288 = 2025-10-16, the last anchor whose
     target window clears the guaranteed-activity zone.  Guard-zone days are excluded from
     TRAINING as well as validation -- measured, not assumed: including them cost +0.00189
     despite adding 35% more rows (reports/eda/guard_test.json).

  --train-through DATE:   overrides the 288 above for SUBMISSION only (default unchanged, so
     every logged submission stays reproducible).  Reopened 2026-08-22 because the +0.00189
     above was measured by validating AT 2026-01-14, an anchor inside the guaranteed-activity
     zone, and NO clean anchor exists to re-test it -- "clean" means the target window ends
     before 2025-11-16, and the last such anchor IS 2025-10-16.  Meanwhile e0141 (weight 0.42
     of the champion) trains through 2026-01-14 and is a top member, so the project already
     contradicts itself across components.  Only the leaderboard can settle it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from metrics import rmsle, score_all                     # noqa: E402
from seqdata import CHANNELS, HORIZON, SeqPanel, build_seq_panel   # noqa: E402
from seqnet import SeqModel, assert_causal               # noqa: E402

MIN_HISTORY_DAYS = 90                      # frozen, mirrors data/fold_spec.json
LAST_CLEAN_ANCHOR = date(2025, 10, 16)     # A + 30 < 2025-11-16 (guard zone)
_T0 = time.time()


def log(m: str) -> None:
    print(m, flush=True)


def within_day_corr(pred: torch.Tensor, targ: torch.Tensor, m: torch.Tensor,
                    eps: float = 1e-6) -> torch.Tensor:
    """Mask-aware Pearson correlation, centred WITHIN each calendar day (column).

    IDEAS.md §I2 / §I19.  After the affine calibration §1b applies to every submission the
    score depends on the prediction only through rho = corr(L, M); MSE additionally pays for a
    level and a spread that calibration then discards.  A correlation term removes exactly that
    waste -- but it must be the correlation the metric is, which §1r proves is the WITHIN-anchor
    one: pooling across days credits a model for knowing December outranks July, and the
    competition scores a single anchor.  So each column (a calendar day) is centred on its own
    masked mean before the covariance is pooled, making the term invariant to a per-day affine
    shift -- the same freedom the calibration has.  Columns with <2 valid rows carry no
    covariance and are dropped.
    """
    cnt = m.sum(0)                                   # (Tt,) users present per day
    denom = cnt.clamp(min=1.0)
    pmean = (pred * m).sum(0) / denom
    ymean = (targ * m).sum(0) / denom
    vcol = (cnt > 1).to(pred.dtype)[None, :]
    pc = (pred - pmean) * m * vcol
    yc = (targ - ymean) * m * vcol
    cov = (pc * yc).sum()
    vp = (pc * pc).sum()
    vy = (yc * yc).sum()
    return cov / torch.sqrt(vp * vy + eps)


def dow_channels(sp: SeqPanel, device: str) -> torch.Tensor:
    """(1, 2, n_days) sin/cos of day-of-week.

    The only calendar signal with no extrapolation risk: it repeats every 7 days, so the test
    window is interpolation.  Day-of-year is deliberately absent -- the model has never seen a
    February target window, and DATA.md §5.4's measured +16% seasonal lift is an aggregate the
    model could only apply by extrapolating a term it has no data for.
    """
    t = np.arange(sp.n_days)
    dow = (np.array([(sp.day(int(i)).weekday()) for i in t]) / 7.0) * 2 * math.pi
    arr = np.stack([np.sin(dow), np.cos(dow)])[None].astype(np.float32)
    return torch.from_numpy(arr).to(device)


@torch.no_grad()
def predict_at(model, Xg: torch.Tensor, dowf, t_score: int, crop: int | None, bs: int,
               amp) -> np.ndarray:
    """Predict at day `t_score`, optionally feeding only the last `crop` days.

    `crop=None` reproduces the original behaviour exactly (the slice starts at day 0), so
    every result logged before this parameter existed is unchanged.  `crop` exists for one
    question: at the test anchor the model consumes a 409-day sequence having trained on at
    most 289, and CV cannot see that mismatch because every fold scores exactly 30 days past
    its own training end.  Cropping the input to the trained length is the direct intervention.
    """
    n_users = Xg.shape[0]
    lo = 0 if crop is None else max(0, t_score - crop + 1)
    Xsc = Xg[:, :, lo: t_score + 1]
    dsc = dowf[:, :, lo: t_score + 1] if dowf is not None else None
    pos = t_score - lo
    out = np.empty(n_users, dtype=np.float64)
    for i in range(0, n_users, bs):
        with amp():
            o = model(Xsc[i: i + bs].float(), dsc)
        out[i: i + bs] = o[:, pos].float().cpu().numpy()
    return np.maximum(np.expm1(out), 0.0)


def train_one(sp: SeqPanel, Xg: torch.Tensor, Yg: torch.Tensor, popg: torch.Tensor,
              cfg: dict, t_lo: int, t_hi: int, score_at, device: str,
              tag: str, user_mask: np.ndarray | None = None) -> tuple[dict, list[float]]:
    """Fit on days [t_lo, t_hi]; return predictions (linear GMV) at each requested day.

    `score_at` is a list of `(day, crop)` pairs; the return value is keyed by that pair.

    Sequences are truncated at the right end to the last day anyone needs.  That is free and
    exact for a causal model -- position t never reads past itself -- and it is what makes
    fold 0 (139 days) three times cheaper than fold 4 (289 days).
    """
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    mu, sd = sp.norm_stats(t_hi)
    model = SeqModel(sp.n_ch, torch.from_numpy(mu), torch.from_numpy(sd),
                     arch=cfg.get("arch", "tcn"), d=int(cfg.get("d_model", 128)),
                     n_blocks=int(cfg.get("n_blocks", 8)), dropout=float(cfg.get("dropout", 0.1)),
                     dow=bool(cfg.get("dow", False)), **cfg.get("arch_kwargs", {})).to(device)

    dowf = dow_channels(sp, device) if cfg.get("dow") else None
    # The guard runs on the REAL model, on this fold, before a single gradient step.  A causal
    # bug found after training is a wasted run; found here it is a five-second failure.
    assert_causal(model, sp.n_ch, min(sp.n_days, 512), device,
                  probes=tuple(p for p in (95, 168, 288, 378) if p < min(sp.n_days, 512) - 1),
                  dow_feat=dowf[:, :, :min(sp.n_days, 512)] if dowf is not None else None)

    n_users = sp.n_users
    # `user_mask` restricts which users contribute GRADIENTS; predictions are always emitted
    # for everyone.  It exists for the user-holdout diagnostic (src/seq_usersplit.py) and is
    # None on every logged experiment, so the training path there is byte-identical.
    train_users = (np.arange(n_users) if user_mask is None else np.flatnonzero(user_mask))
    n_tr_users = int(train_users.size)
    tr_idx = torch.from_numpy(train_users).to(device)
    bs = int(cfg.get("batch_users", 512))
    epochs = int(cfg["epochs"])
    steps_per_epoch = math.ceil(n_tr_users / bs)
    total = epochs * steps_per_epoch
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 3e-3)),
                            weight_decay=float(cfg.get("weight_decay", 1e-4)))
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

    # training positions: [t_lo, t_hi], masked to the in-population user-days
    Xtr = Xg[:, :, : t_hi + 1]
    Ytr = Yg[:, t_lo: t_hi + 1]
    Mtr = popg[:, t_lo: t_hi + 1]
    # `train_stride` supervises only every k-th day.  It exists to PRICE dense supervision:
    # a one-target-per-window design (e.g. a bidirectional encoder over [0, t]) emits one
    # prediction per forward pass instead of 409, so it necessarily trains on an anchor grid.
    # stride=7 reproduces exactly the grid the GBDT uses (data/fold_spec.json), which makes the
    # cost of that trade a measured number rather than an assumption.  Default 1 = unchanged.
    stride = int(cfg.get("train_stride", 1))
    if stride > 1:
        keepcol = torch.zeros(Mtr.shape[1], dtype=torch.bool, device=Mtr.device)
        keepcol[torch.arange(Mtr.shape[1] - 1, -1, -stride, device=Mtr.device)] = True
        Mtr = Mtr & keepcol[None, :]
    n_pos = int(Mtr[tr_idx].sum().item())
    log(f"    {tag}: days [{t_lo}..{t_hi}] = {t_hi - t_lo + 1} positions, "
        f"{n_tr_users:,} training users, {n_pos:,} supervised user-days, "
        f"{epochs} epochs x {steps_per_epoch} steps")

    def amp():
        return (torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device == "cuda"
                else torch.autocast(device_type="cpu", enabled=False))

    # IDEAS.md §I19: blend a within-day correlation term into the MSE loss. corr_lambda=0
    # (the default) is byte-identical to the historical MSE path, so every logged seq run is
    # reproducible. The DRW-Crypto 2025 winner used exactly 0.6*MSE + 0.4*Pearson on its MLP;
    # CISIR (arXiv 2509.16339) uses wMSE + ~0.5*wPCC. Both are NN-only -- a batch-level
    # statistic is natural for SGD and impossible for boosting's per-sample Newton step.
    corr_lambda = float(cfg.get("corr_lambda", 0.0))
    g = torch.Generator(device="cpu").manual_seed(cfg["seed"])
    losses: list[float] = []
    model.train()
    for ep in range(epochs):
        perm = tr_idx[torch.randperm(n_tr_users, generator=g).to(device)]
        tot, cnt = 0.0, 0
        for i in range(steps_per_epoch):
            idx = perm[i * bs: (i + 1) * bs]
            m = Mtr[idx].float()
            n_m = m.sum()
            if float(n_m) == 0.0:
                continue
            with amp():
                out = model(Xtr[idx].float(), dowf[:, :, : t_hi + 1] if dowf is not None else None)
            pred = out[:, t_lo:].float()
            resid = (pred - Ytr[idx]) * m
            mse = (resid ** 2).sum() / n_m
            if corr_lambda > 0.0:
                corr = within_day_corr(pred, Ytr[idx].float(), m)
                loss = (1.0 - corr_lambda) * mse + corr_lambda * (1.0 - corr)
            else:
                loss = mse
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            w = float(n_m)
            tot += float(mse.item()) * w      # track MSE for the progress readout, not the
            cnt += w                          # blended objective, so train_rmsle stays comparable
        losses.append(math.sqrt(tot / max(cnt, 1.0)))
        log(f"      epoch {ep + 1:>3d}/{epochs}  train_rmsle {losses[-1]:.5f}  "
            f"lr {sched.get_last_lr()[0]:.2e}  [{time.time() - _T0:.0f}s]")

    model.eval()
    return {(t, c): predict_at(model, Xg, dowf, t, c, bs, amp) for t, c in score_at}, losses


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["cv", "submit"], default="cv")
    ap.add_argument("--train-through", default=None, metavar="YYYY-MM-DD",
                    help="submit mode only: extend the training window past the guard-zone "
                         "boundary (default 2025-10-16). Capped at the horizon tail "
                         "2026-01-14. Omit for the historical behaviour.")
    ap.add_argument("--screen", action="store_true", help="last 2 folds only (tier=screen)")
    ap.add_argument("--channels", default=None, metavar="a,b,c",
                    help="comma-separated CHANNELS subset (default: all 13). Every seq model in "
                         "this project reads all 13 jointly. A channel-restricted model is a "
                         "DIFFERENT VIEW rather than a degraded copy -- unlike the killed "
                         "lookback-restricted ensemble, whose members had strictly NESTED "
                         "information. ⚠ Several channels are algebraically dependent "
                         "(gmv = gmvs + gmvc, ord = s2o + c2o, buy = 1{gmv>0}), so a per-channel "
                         "split is NOT 13 independent views; the meaningful cut is monetary "
                         "(gmv/ord family) vs behavioural (presence/searches/cat/cart family). "
                         "Names: " + ",".join(c[0] for c in CHANNELS))
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    exp_id = cfg["exp_id"]
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"\n=== {exp_id} [{args.mode}] : {cfg['change']} ===")
    log(f"    parent={cfg['parent_id']}  arch={cfg.get('arch', 'tcn')} d={cfg.get('d_model', 128)} "
        f"blocks={cfg.get('n_blocks', 8)} epochs={cfg['epochs']} seed={cfg['seed']}  device={device}")
    if device == "cuda":
        log(f"    gpu: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    sp = build_seq_panel(derived=bool(cfg.get("derived_channels", False)),
                         ranks=bool(cfg.get("rank_channels", False)),
                         popidx=bool(cfg.get("popidx_channel", False)))
    if args.channels:
        want = [c.strip() for c in args.channels.split(",") if c.strip()]
        names = [c[0] for c in CHANNELS]
        unknown = [w for w in want if w not in names]
        assert not unknown, f"unknown channel(s) {unknown}; available {names}"
        assert not (bool(cfg.get("derived_channels", False))
                    or bool(cfg.get("rank_channels", False))
                    or bool(cfg.get("popidx_channel", False))), \
            "--channels indexes the BASE 13 only; this config appends derived/rank/pop channels"
        keep = [names.index(w) for w in want]
        sp.X = np.ascontiguousarray(sp.X[:, keep, :])
        log(f"    CHANNEL SUBSET: {len(keep)} of {len(names)} -> {want}")
    Xg = torch.from_numpy(sp.X).to(device)
    Yg = torch.from_numpy(sp.Y).to(device)
    popg = torch.from_numpy(sp.pop).to(device)
    if device == "cuda":
        log(f"    gpu resident: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    t_lo = MIN_HISTORY_DAYS - 1

    if args.mode == "submit":
        # `--train-through` extends the SUBMISSION's training window past the guard-zone
        # boundary. Default is unchanged (2025-10-16), so every existing submission is
        # reproducible byte-for-byte.
        #
        # WHY IT IS WORTH A LEADERBOARD SLOT. Two of the champion's three components stop at
        # 2025-10-16; e0141 (weight 0.42) trains through 2026-01-14 and is a top member. The
        # +0.00189 that justified excluding guard-zone anchors was measured by validating AT
        # 2026-01-14 -- an anchor that is itself inside the guaranteed-activity zone -- and no
        # clean anchor exists to re-test it, because "clean" means "target window ends before
        # 2025-11-16" and the last such anchor IS 2025-10-16. So the exclusion rests on a
        # contaminated measurement and CANNOT be checked on any internal fold. Only the
        # leaderboard can settle it, which is exactly what a submission slot is for.
        #
        # The horizon-tail bound still applies: a training day t needs its target window
        # [t+1, t+30] fully observed, so t <= n_days - 31 = 378 = 2026-01-14.
        t_hi = sp.idx(LAST_CLEAN_ANCHOR)
        if args.train_through:
            t_hi = sp.idx(date.fromisoformat(args.train_through))
            tail = sp.n_days - HORIZON - 1
            assert t_hi <= tail, (f"train-through {args.train_through} exceeds the horizon "
                                  f"tail {sp.day(tail)}: targets would be truncated")
            log(f"    TRAIN WINDOW EXTENDED: days [{t_lo}..{t_hi}] "
                f"(through {sp.day(t_hi)}, vs the default {LAST_CLEAN_ANCHOR}) "
                f"-- {t_hi - sp.idx(LAST_CLEAN_ANCHOR)} extra days, guard zone INCLUDED")
        t_score = sp.n_days - 1
        assert t_score == sp.idx(sp.dmax)
        preds, losses = train_one(sp, Xg, Yg, popg, cfg, t_lo, t_hi,
                                  [(t_score, cfg.get("infer_crop"))], device, "submit")
        pred = preds[(t_score, cfg.get("infer_crop"))]
        keep = sp.pop[:, t_score]
        assert keep.all(), "some test users fail the population rule -- unexpected"
        sub = pl.DataFrame({"user_id": sp.users, "predict": pred})
        ss = pl.read_csv(ROOT / "data" / "sample_submit.csv")
        assert sub.height == ss.height == 250_000
        assert np.array_equal(sub["user_id"].to_numpy(), ss["user_id"].to_numpy()), "user order"
        assert np.isfinite(pred).all() and (pred >= 0).all()
        (ROOT / "subs").mkdir(exist_ok=True)
        out = ROOT / "subs" / f"{exp_id}.csv"
        sub.write_csv(out)
        from metrics import gini
        naive = sp.geo3(t_score)
        last30 = sp.wgmv(t_score - 29, t_score)
        log(f"\n  wrote {out.relative_to(ROOT)}")
        log(f"  {'':22s} {'sum':>16s} {'mean':>10s} {'zero share':>11s} {'gini':>8s}")
        for nm, v in [(f"{exp_id} prediction", pred), ("naive geo3", naive),
                      ("last-30d (=sample)", last30)]:
            log(f"  {nm:22s} {v.sum():>16,.0f} {v.mean():>10.2f} "
                f"{100 * (v == 0).mean():>10.2f}% {gini(v):>8.4f}")
        log(f"  runtime {(time.time() - t0) / 60:.1f} min")
        return

    spec = json.loads((ROOT / "data" / "fold_spec.json").read_text())
    folds = pl.read_parquet(ROOT / "data" / "folds.parquet")
    fold_ids = sorted(folds["fold_id"].unique().to_list())
    if args.screen:
        fold_ids = fold_ids[-2:]
    tier = "screen" if args.screen else cfg.get("tier", "confirm")

    oof, per_fold, per_fold_naive, curves = [], [], [], []
    for k in fold_ids:
        fs = spec["folds"][k]
        va = date.fromisoformat(fs["valid_anchor"])
        vai = sp.idx(va)
        t_hi = vai - HORIZON                     # last training day: target ends on or before A
        assert sp.day(t_hi) == va - timedelta(days=HORIZON)
        assert t_hi <= sp.idx(LAST_CLEAN_ANCHOR), "training days reach the guard zone"

        preds, losses = train_one(sp, Xg, Yg, popg, cfg, t_lo, t_hi,
                                  [(vai, cfg.get("infer_crop"))], device, f"fold {k}")
        pred = preds[(vai, cfg.get("infer_crop"))]
        curves.append(losses)

        vkeep = sp.pop[:, vai]
        fv = folds.filter(pl.col("fold_id") == k).sort("user_id")
        assert np.array_equal(fv["user_id"].to_numpy(), sp.users[vkeep]), "fold population drift"
        yva = fv["target"].to_numpy()
        pv = pred[vkeep]
        naive = sp.geo3(vai)[vkeep]

        s, sn = rmsle(yva, pv), rmsle(yva, naive)
        per_fold.append(s); per_fold_naive.append(sn)
        oof.append(pl.DataFrame({
            "fold_id": np.full(yva.size, k, np.int8),
            "anchor_date": pl.Series("anchor_date", [va] * yva.size, dtype=pl.Date),
            "user_id": fv["user_id"].to_numpy(),
            "y_true": yva, "y_pred": pv, "y_naive": naive}))
        log(f"    fold {k} {va}  n={yva.size:>7,}  rmsle={s:.5f}  naive={sn:.5f}  "
            f"delta={s - sn:+.5f}")

    oof = pl.concat(oof)
    (ROOT / "oof").mkdir(exist_ok=True)
    oof.write_parquet(ROOT / "oof" / f"{exp_id}.parquet")

    pf = np.array(per_fold); pfn = np.array(per_fold_naive)
    agg = score_all(oof["y_true"].to_numpy(), oof["y_pred"].to_numpy())
    agg_n = score_all(oof["y_true"].to_numpy(), oof["y_naive"].to_numpy())
    wins = int((pf < pfn).sum())
    runtime = (time.time() - t0) / 60

    log(f"\n  cv_mean = {pf.mean():.5f} +/- {pf.std():.5f}   folds {np.round(pf, 5).tolist()}")
    log(f"  naive   = {pfn.mean():.5f} +/- {pfn.std():.5f}   folds {np.round(pfn, 5).tolist()}")
    log(f"  delta vs naive = {pf.mean() - pfn.mean():+.5f}   wins {wins}/{len(pf)} folds")
    log(f"  last fold (most test-like) = {pf[-1]:.5f}  (naive {pfn[-1]:.5f}, "
        f"delta {pf[-1] - pfn[-1]:+.5f})")
    log(f"  tie-breakers: gini_pred={agg['gini_pred']:.4f} (true {agg['gini_true']:.4f})  "
        f"total_rel_err={agg['total_rel_err']:+.4f}  [naive: {agg_n['gini_pred']:.4f}, "
        f"{agg_n['total_rel_err']:+.4f}]")
    log(f"  runtime {runtime:.1f} min")

    row = {
        "exp_id": exp_id, "parent_id": cfg["parent_id"],
        "date": datetime.now().isoformat(timespec="seconds"),
        "approach": cfg.get("approach", "nn_seq"), "change": cfg["change"], "tier": tier,
        "n_features": sp.n_ch, "cv_mean": round(float(pf.mean()), 5),
        "cv_std": round(float(pf.std()), 5),
        "folds": json.dumps([round(float(x), 5) for x in pf]),
        "delta": round(float(pf.mean() - pfn.mean()), 5),
        "significant": "yes" if wins >= 4 or abs(pf.mean() - pfn.mean()) > 2 * pf.std() else "no",
        "lb": "", "runtime_min": round(runtime, 1), "seed": cfg["seed"],
        "config": args.config, "verdict": cfg.get("verdict", ""),
        "gini_pred": round(agg["gini_pred"], 4),
        "total_rel_err": round(agg["total_rel_err"], 4),
        "notes": cfg.get("notes", ""),
        "best_iters": json.dumps([[round(x, 5) for x in c] for c in curves]),
    }
    rd = ROOT / "runs"; rd.mkdir(exist_ok=True)
    (rd / f"{exp_id}.json").write_text(json.dumps(row, indent=2))
    try:
        import subprocess
        subprocess.run([sys.executable, str(ROOT / "src" / "collect.py")], check=False)
    except Exception:
        pass
    log(f"\n  wrote runs/{exp_id}.json -> experiments.csv")


if __name__ == "__main__":
    main()
