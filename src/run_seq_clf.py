#!/usr/bin/env python
"""
The discrimination ceiling, sequence-model half: a GRU trained with a BCE head on `y > 0`.

    python src/run_seq_clf.py --config configs/e0101_seq_gru.yaml --exp-id e0161

Companion to src/run_clf.py, which asks the same question of LightGBM.  The target is no
longer log1p(GMV) but the binary event, and the loss is BCE rather than L2 -- so this measures
what the seq family can do when it is asked to CLASSIFY instead of to regress.

Standalone on purpose.  run_seq.py's training loop is shared by eighteen logged experiments
and has already had to be re-verified twice after edits; a different objective and a different
metric do not belong inside it.  Everything upstream (the panel, the causality guard, the
frozen folds) is imported, so the two paths cannot silently diverge on the data.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import torch
import yaml
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rho_decomp import auc                                     # noqa: E402
from seqdata import HORIZON, build_seq_panel                   # noqa: E402
from seqnet import SeqModel, assert_causal                     # noqa: E402

MIN_HISTORY_DAYS = 90
LAST_CLEAN_ANCHOR = date(2025, 10, 16)


def log(m: str) -> None:
    print(m, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/e0101_seq_gru.yaml")
    ap.add_argument("--exp-id", default="e0161")
    ap.add_argument("--epochs", type=int, default=0, help="0 = use the config's value")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.epochs:
        cfg["epochs"] = args.epochs
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"\n=== {args.exp_id}: seq GRU with a BCE head on y>0  (arch {cfg.get('arch')}, "
        f"{cfg['epochs']} epochs) ===")

    sp = build_seq_panel(derived=bool(cfg.get("derived_channels", False)),
                         ranks=bool(cfg.get("rank_channels", False)))
    Xg = torch.from_numpy(sp.X).to(device)
    Zg = torch.from_numpy((sp.Y > 0).astype(np.float32)).to(device)   # the binary event
    popg = torch.from_numpy(sp.pop).to(device)

    spec = json.loads((ROOT / "data" / "fold_spec.json").read_text())
    folds = pl.read_parquet(ROOT / "data" / "folds.parquet")
    t_lo = MIN_HISTORY_DAYS - 1
    bs = int(cfg.get("batch_users", 512))
    per_fold, oof = [], []

    for k in sorted(folds["fold_id"].unique().to_list()):
        fs = spec["folds"][k]
        va = date.fromisoformat(fs["valid_anchor"]); vai = sp.idx(va)
        t_hi = vai - HORIZON
        assert sp.day(t_hi) == va - timedelta(days=HORIZON)
        assert t_hi <= sp.idx(LAST_CLEAN_ANCHOR), "training days reach the guard zone"

        torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
        mu, sd = sp.norm_stats(t_hi)
        model = SeqModel(sp.n_ch, torch.from_numpy(mu), torch.from_numpy(sd),
                         arch=cfg.get("arch", "gru"), d=int(cfg.get("d_model", 128)),
                         n_blocks=int(cfg.get("n_blocks", 2)),
                         dropout=float(cfg.get("dropout", 0.1)),
                         **cfg.get("arch_kwargs", {})).to(device)
        assert_causal(model, sp.n_ch, min(sp.n_days, 512), device,
                      probes=tuple(q for q in (95, 168, 288, 378) if q < min(sp.n_days, 512) - 1))

        n_users, epochs = sp.n_users, int(cfg["epochs"])
        steps = math.ceil(n_users / bs)
        opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 3e-3)),
                                weight_decay=float(cfg.get("weight_decay", 1e-4)))
        warm = max(1, int(0.05 * epochs * steps))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / warm if s < warm
            else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, epochs * steps - warm))))
        Xtr, Ztr, Mtr = Xg[:, :, : t_hi + 1], Zg[:, t_lo: t_hi + 1], popg[:, t_lo: t_hi + 1]
        log(f"\n    fold {k} {va}: days [{t_lo}..{t_hi}], "
            f"{int(Mtr.sum().item()):,} supervised user-days, base rate "
            f"{float((Ztr * Mtr).sum() / Mtr.sum()):.4f}")

        def amp():
            return (torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device == "cuda"
                    else torch.autocast(device_type="cpu", enabled=False))

        g = torch.Generator(device="cpu").manual_seed(cfg["seed"])
        bce = nn.BCEWithLogitsLoss(reduction="none")
        model.train()
        for ep in range(epochs):
            perm = torch.randperm(n_users, generator=g).to(device)
            tot = cnt = 0.0
            for i in range(steps):
                idx = perm[i * bs:(i + 1) * bs]
                m = Mtr[idx].float(); nm = m.sum()
                if float(nm) == 0:
                    continue
                with amp():
                    out = model(Xtr[idx].float())
                loss = (bce(out[:, t_lo:].float(), Ztr[idx]) * m).sum() / nm
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step()
                tot += float(loss.item()) * float(nm); cnt += float(nm)
            log(f"      epoch {ep + 1:>3d}/{epochs}  bce {tot / cnt:.5f}  "
                f"[{(time.time() - t0) / 60:.1f}m]")

        model.eval()
        logits = np.empty(n_users, np.float64)
        Xsc = Xg[:, :, : vai + 1]
        with torch.no_grad():
            for i in range(0, n_users, bs):
                with amp():
                    o = model(Xsc[i:i + bs].float())
                logits[i:i + bs] = o[:, vai].float().cpu().numpy()

        vkeep = sp.pop[:, vai]
        fv = folds.filter(pl.col("fold_id") == k).sort("user_id")
        assert np.array_equal(fv["user_id"].to_numpy(), sp.users[vkeep]), "fold population drift"
        z = (fv["target"].to_numpy() > 0).astype(float)
        a = auc(logits[vkeep], z)
        per_fold.append(a)
        oof.append(pl.DataFrame({"fold_id": np.full(z.size, k, np.int8),
                                 "user_id": fv["user_id"].to_numpy(),
                                 "z_true": z.astype(np.int8),
                                 "p_clf": logits[vkeep].astype(np.float32)}))
        log(f"    fold {k} AUC = {a:.5f}")

    pl.concat(oof).write_parquet(ROOT / "oof" / f"{args.exp_id}_clf.parquet")
    v = np.array(per_fold)
    log(f"\n  {args.exp_id} seq-BCE AUC = {v.mean():.5f}   folds {[round(x, 5) for x in v]}")
    log(f"  the number to beat is the regressors' implicit AUC = 0.84322 "
        f"(src/rho_decomp.py, same folds)")
    log(f"  runtime {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
