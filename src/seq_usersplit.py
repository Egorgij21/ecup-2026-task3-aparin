#!/usr/bin/env python
"""
Is the seq model's overfitting USER-wise or TIME-wise?  And would a user-holdout have caught it?

    python src/seq_usersplit.py --config configs/e0101_seq_gru.yaml --epochs 6 12 20 30

WHY THIS EXISTS.  A user-split CV cannot be this competition's reported metric: the
public/private split is by customer, but both leaderboards cover the SAME future window and
every one of the 250k test users is a user we already hold history for.  The train->test
relationship is therefore *same users, later date* -- pure temporal extrapolation, with no
user-generalisation gap to measure.  CLAUDE.md §3.1 requires CV to reproduce that relationship,
and rule 3 freezes the folds that do.

But a user holdout is still worth exactly one thing, and it is a thing we are missing.  The seq
models train for a fixed 12 epochs chosen by fiat, with no honest signal behind it, and e0106
measured 30 epochs at +0.0204 -- the model sits on an overfitting cliff and we cannot see the
edge.  If the overfitting is user-wise, a held-out slice of users at the SAME anchor detects it
without touching the frozen folds or spending a training anchor, which is what an ordinary
temporal early-stopping split would cost (and what e0017/e0020 showed is expensive here).

WHAT IT MEASURES.  Train on 80% of users over the fold's normal day range, then score at the
fold's own anchor twice: on users the model trained on, and on users it never saw.  The gap is
the user-wise memorisation.  Sweeping the epoch budget answers the question that matters:

  * gap ~ 0 and flat in epochs  -> the model does not memorise users, a user holdout is a
    useless early-stopping signal, and a user-split CV would measure nothing the frozen folds
    do not already measure.  The idea is dead for both purposes.
  * gap grows with epochs       -> user memorisation IS the overfitting channel, and the
    held-out-user curve is a free, honest epoch selector.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from metrics import rmsle                                   # noqa: E402
from run_seq import HORIZON, MIN_HISTORY_DAYS, train_one    # noqa: E402
from seqdata import build_seq_panel                         # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/e0101_seq_gru.yaml")
    ap.add_argument("--fold", type=int, default=4, help="frozen fold to run on")
    ap.add_argument("--epochs", type=int, nargs="+", default=[6, 12, 20, 30])
    ap.add_argument("--holdout", type=float, default=0.2)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=== user-holdout diagnostic : {cfg['exp_id']} ({cfg.get('arch')}) "
          f"fold {args.fold} ===", flush=True)

    sp = build_seq_panel(derived=bool(cfg.get("derived_channels", False)),
                         ranks=bool(cfg.get("rank_channels", False)))
    Xg = torch.from_numpy(sp.X).to(device)
    Yg = torch.from_numpy(sp.Y).to(device)
    popg = torch.from_numpy(sp.pop).to(device)

    spec = json.loads((ROOT / "data" / "fold_spec.json").read_text())
    folds = pl.read_parquet(ROOT / "data" / "folds.parquet")
    fs = spec["folds"][args.fold]
    vai = sp.idx(date.fromisoformat(fs["valid_anchor"]))
    t_lo, t_hi = MIN_HISTORY_DAYS - 1, vai - HORIZON

    fv = folds.filter(pl.col("fold_id") == args.fold).sort("user_id")
    yva = fv["target"].to_numpy()
    vkeep = sp.pop[:, vai]
    assert np.array_equal(fv["user_id"].to_numpy(), sp.users[vkeep]), "fold population drift"

    # The split is over ALL panel users, then intersected with the fold population, so the two
    # scored groups are drawn from one distribution and differ only in whether the model was
    # fit on them.  Seeded off the config seed, fixed for every epoch budget.
    rng = np.random.default_rng(1234 + int(cfg["seed"]))
    is_train_user = rng.random(sp.n_users) >= args.holdout
    seen = is_train_user[vkeep]
    print(f"    train users {is_train_user.sum():,} / {sp.n_users:,}   "
          f"fold population {vkeep.sum():,}  ->  seen {seen.sum():,}  held-out {(~seen).sum():,}")

    rows = []
    for ep in args.epochs:
        c = dict(cfg, epochs=ep)
        preds, losses = train_one(sp, Xg, Yg, popg, c, t_lo, t_hi, [(vai, None)], device,
                                  f"epochs={ep}", user_mask=is_train_user)
        pv = preds[(vai, None)][vkeep]
        s_seen, s_held = rmsle(yva[seen], pv[seen]), rmsle(yva[~seen], pv[~seen])
        rows.append({"epochs": ep, "seen": s_seen, "held_out": s_held,
                     "gap": s_held - s_seen, "train_rmsle": losses[-1]})
        print(f"    epochs {ep:>3d}  train-loss {losses[-1]:.5f}  seen-users {s_seen:.5f}  "
              f"HELD-OUT {s_held:.5f}  gap {s_held - s_seen:+.5f}", flush=True)

    print(f"\n  {'epochs':>7s} {'seen':>9s} {'held-out':>10s} {'gap':>9s}   "
          f"{'d(held-out) vs best':>20s}")
    best = min(r["held_out"] for r in rows)
    for r in rows:
        print(f"  {r['epochs']:>7d} {r['seen']:>9.5f} {r['held_out']:>10.5f} {r['gap']:>+9.5f}   "
              f"{r['held_out'] - best:>+20.5f}")
    gaps = np.array([r["gap"] for r in rows])
    print(f"\n  gap range over the sweep: {gaps.min():+.5f} .. {gaps.max():+.5f}  "
          f"(sigma_noise for nn_seq = 0.00020)")
    argbest = min(rows, key=lambda r: r["held_out"])["epochs"]
    print(f"  held-out-user optimum at {argbest} epochs; the frozen-fold optimum is 12 "
          f"(e0101 1.76458 vs e0106 1.78500 at 30)")
    if gaps.max() - gaps.min() < 4e-4:
        print("  VERDICT: the gap is flat inside the noise floor -- the model does NOT memorise\n"
              "           users, so a user holdout is useless as an early-stopping signal and a\n"
              "           user-split CV would measure nothing the frozen folds do not.")
    else:
        print("  VERDICT: the gap moves with the epoch budget -- user memorisation is a real\n"
              "           overfitting channel and the held-out-user curve is a usable, free\n"
              "           epoch selector that costs no training anchor.")

    out = ROOT / "reports" / "eda" / "seq_usersplit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": args.config, "fold": args.fold,
                               "holdout": args.holdout, "rows": rows}, indent=2))
    print(f"\n  wrote reports/eda/seq_usersplit.json   runtime {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
