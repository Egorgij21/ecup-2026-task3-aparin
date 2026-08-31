#!/usr/bin/env python
"""
Architecture + hyperparameter search for the seq family, on the FROZEN date folds.

    python src/tune_seq.py --trials 8 --study seqarch --fold 4

TWO DEFECTS OF THE PREVIOUS SEARCH, BOTH FIXED HERE.

1. **Protocol.** `tune_usercv.py` searched on the user-split CV, which measures unseen users at
   calendar times the model trained on -- not the same users at a future date, which is the
   task.  Its winner improved that CV by -0.00099 (5/5 folds) and made rho on the leaderboard
   WORSE by -0.00101.  This searches the frozen date folds, the protocol every logged
   experiment uses and the one the LightGBM tuning transferred on (+0.00038 rho).

2. **A biased objective.**  That search scored each trial by the MINIMUM of an early-stopped
   validation curve.  The curve is noise (per-evaluation sd 0.0014-0.0027), so taking the min
   of N draws is optimistically biased by about `sigma*sqrt(2*ln N)` -- and configurations that
   train longer get more draws.  The winner had 100 evaluations against the baseline's 21, and
   most of its "gain" was that asymmetry.  `src/run_seq.py` trains a FIXED number of epochs and
   scores once at the end, so no such selection exists; `epochs` is searched as an ordinary
   hyperparameter and every trial is judged on the same single measurement.

SEARCHED: architecture (8 of them), embedding size, depth, dropout, learning rate, weight
decay, batch size, and the epoch budget.  Everything else -- features, folds, target, metric --
is frozen (CLAUDE.md rules 3 and 4).

Search runs on ONE fold (default 4, the most test-like per EXPERIMENTS.md §3.2); the winner is
confirmed on all five before it means anything (§4.2).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from metrics import rmsle                                            # noqa: E402
from run_seq import HORIZON, LAST_CLEAN_ANCHOR, MIN_HISTORY_DAYS, train_one   # noqa: E402
from seqdata import build_seq_panel                                  # noqa: E402

ARCHS = ["gru", "lstm", "rnn", "tcn", "cnngru",
         "xformer_alibi", "xformer_learned", "xformer_rope"]


def log(m: str) -> None:
    print(m, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--study", default="seqarch")
    ap.add_argument("--fold", type=int, default=4)
    ap.add_argument("--derived", action="store_true")
    args = ap.parse_args()
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    log(f"\n=== seq architecture search : frozen fold {args.fold}, fixed-epoch objective ===")

    sp = build_seq_panel(derived=args.derived)
    Xg = torch.from_numpy(sp.X).to(device)
    Yg = torch.from_numpy(sp.Y).to(device)
    popg = torch.from_numpy(sp.pop).to(device)

    spec = json.loads((ROOT / "data" / "fold_spec.json").read_text())
    folds = pl.read_parquet(ROOT / "data" / "folds.parquet")
    fs = spec["folds"][args.fold]
    va = date.fromisoformat(fs["valid_anchor"]); vai = sp.idx(va)
    t_lo, t_hi = MIN_HISTORY_DAYS - 1, vai - HORIZON
    assert sp.day(t_hi) == va - timedelta(days=HORIZON)
    assert t_hi <= sp.idx(LAST_CLEAN_ANCHOR)
    fv = folds.filter(pl.col("fold_id") == args.fold).sort("user_id")
    yva = fv["target"].to_numpy()
    vkeep = sp.pop[:, vai]
    assert np.array_equal(fv["user_id"].to_numpy(), sp.users[vkeep]), "fold population drift"
    naive = rmsle(yva, sp.geo3(vai)[vkeep])
    log(f"    anchor {va} | {vkeep.sum():,} users | geo3 {naive:.5f} | "
        f"e0101 (gru d128 L2, 12ep) scored 1.73240 on this fold")

    def objective(trial):
        arch = trial.suggest_categorical("arch", ARCHS)
        cfg = {
            "arch": arch,
            "d_model": trial.suggest_categorical("d_model", [32, 64, 128, 256]),
            "n_blocks": trial.suggest_int("n_blocks", 1, 4),
            "dropout": trial.suggest_float("dropout", 0.0, 0.4),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "batch_users": trial.suggest_categorical("batch_users", [256, 512, 1024]),
            "epochs": trial.suggest_int("epochs", 8, 200, log=True),
            "seed": 0, "dow": False,
            "arch_kwargs": {"n_heads": 4} if arch.startswith("xformer") else {},
        }
        if arch == "tcn":                      # dilations 2^i: depth sets the receptive field
            cfg["n_blocks"] = trial.suggest_int("tcn_blocks", 4, 9)
        try:
            preds, _ = train_one(sp, Xg, Yg, popg, cfg, t_lo, t_hi, [(vai, None)], device,
                                 f"trial{trial.number}")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            log(f"  trial {trial.number:>3d}  OOM ({arch} d{cfg['d_model']} "
                f"L{cfg['n_blocks']} b{cfg['batch_users']}) -> pruned")
            raise optuna.TrialPruned()
        v = rmsle(yva, preds[(vai, None)][vkeep])
        trial.set_user_attr("arch", arch)
        log(f"  trial {trial.number:>3d}  {v:.5f}  | {arch:15s} d{cfg['d_model']:<4d} "
            f"L{cfg['n_blocks']} do{cfg['dropout']:.2f} lr{cfg['lr']:.1e} "
            f"wd{cfg['weight_decay']:.1e} b{cfg['batch_users']:<5d} ep{cfg['epochs']:<4d} "
            f"[{(time.time() - t0) / 60:.0f}m]")
        return v

    st = optuna.create_study(direction="minimize", study_name=args.study,
                             storage=f"sqlite:///{ROOT}/reports/{args.study}.db",
                             load_if_exists=True)
    st.optimize(objective, n_trials=args.trials, catch=(RuntimeError,))
    done = [t for t in st.trials if t.value is not None]
    done.sort(key=lambda t: t.value)
    log(f"\n  === best 8 of {len(done)} completed trials (fold {args.fold}) ===")
    for t in done[:8]:
        log(f"  {t.value:.5f}  {t.params}")
    log(f"\n  per-architecture best:")
    for a in ARCHS:
        w = [t.value for t in done if t.user_attrs.get("arch") == a]
        log(f"    {a:16s} n={len(w):>2d}  best {min(w):.5f}" if w else f"    {a:16s} n= 0")
    log(f"\n  reference on this fold: e0101 1.73240 | e0100 1.73525 | e0102 1.73743 | "
        f"geo3 {naive:.5f}")
    (ROOT / "reports" / "eda" / f"{args.study}_fold{args.fold}.json").write_text(
        json.dumps([{"value": t.value, **t.params} for t in done], indent=2))


if __name__ == "__main__":
    main()
