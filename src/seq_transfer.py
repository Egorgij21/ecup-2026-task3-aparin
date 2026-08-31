#!/usr/bin/env python
"""
Why did e0120's CV gain not transfer?  Cut-off-distance stress test for the `seq` family.

    python src/seq_transfer.py --config configs/e0101_seq_gru.yaml

THE BLIND SPOT THIS EXISTS TO MEASURE.  Every frozen fold scores exactly 30 days past its own
last training day (t_hi = A - 30, scored at A).  The submission scores **120 days** past its
last training day (train to 2025-10-16, predict at 2026-02-13) and, for a recurrent model,
runs its recurrence for 409 steps having never trained past 289.  CV is structurally incapable
of seeing either mismatch: there is no fold where the gap is anything but 30.

Measured on the LB, e0120 gained -0.00060 against e0064 where the CV delta was -0.00239 -- a
0.25x transfer where every previous significant delta ran 1.5-1.8x.  And -0.00060 is exactly
what the GBDT half of the blend would have produced alone (its own CV delta is -0.00041, which
at 1.5-1.8x projects -0.00062/-0.00074).  So the working hypothesis is that the seq family's
contribution at the test cut-off is ~zero, and this script asks whether cut-off distance is
the reason.

WHAT IT PRODUCES.  The same triangular transfer matrix `src/eda/transfer_test.py` already
built for the GBDT (reports/eda/transfer_test.json), so the two degradation curves are
directly comparable: model trained with fold i's day budget, evaluated at fold j's anchor for
every j >= i.  Fold 0's model at fold 4's anchor is a 150-day gap -- the closest analogue the
data admits to the submission's 120-day gap.

It also scores every off-diagonal cell a second time with the input left-cropped to the
model's own trained sequence length.  If cropping recovers the loss, the failure is sequence
length (fixable at inference, for free); if it does not, the failure is genuine cut-off drift
(fixable only by retraining closer to the test date, or by making the model level-invariant).
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
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=== seq cut-off transfer test : {cfg['exp_id']} ({cfg.get('arch')}) ===", flush=True)

    sp = build_seq_panel()
    Xg = torch.from_numpy(sp.X).to(device)
    Yg = torch.from_numpy(sp.Y).to(device)
    popg = torch.from_numpy(sp.pop).to(device)

    spec = json.loads((ROOT / "data" / "fold_spec.json").read_text())
    folds = pl.read_parquet(ROOT / "data" / "folds.parquet")
    anchors = [sp.idx(date.fromisoformat(f["valid_anchor"])) for f in spec["folds"]]
    truth = {}
    for k in range(len(anchors)):
        fv = folds.filter(pl.col("fold_id") == k).sort("user_id")
        truth[k] = (fv["target"].to_numpy(), sp.pop[:, anchors[k]])

    n = len(anchors)
    M = np.full((n, n), np.nan)          # [train fold][eval fold], full sequence
    Mc = np.full((n, n), np.nan)         # same, input cropped to the trained length
    t_lo = MIN_HISTORY_DAYS - 1

    for i in range(n):
        t_hi = anchors[i] - HORIZON
        trained_len = t_hi + 1
        score_at = [(anchors[j], None) for j in range(i, n)]
        score_at += [(anchors[j], trained_len) for j in range(i + 1, n)]
        preds, _ = train_one(sp, Xg, Yg, popg, cfg, t_lo, t_hi, score_at, device,
                             f"train-budget of fold {i}")
        for j in range(i, n):
            y, keep = truth[j]
            M[i, j] = rmsle(y, preds[(anchors[j], None)][keep])
            if j > i:
                Mc[i, j] = rmsle(y, preds[(anchors[j], trained_len)][keep])
        print(f"    fold-{i} budget (days<= {t_hi}, len {trained_len}): "
              + "  ".join(f"@f{j}={M[i, j]:.5f}" for j in range(i, n)), flush=True)

    print("\n  TRANSFER MATRIX  rows = training day budget, cols = evaluation anchor")
    print("  " + " " * 10 + "".join(f"{'@f' + str(j):>12s}" for j in range(n)))
    for i in range(n):
        print(f"  budget f{i} " + "".join(
            f"{M[i, j]:>12.5f}" if not np.isnan(M[i, j]) else f"{'':>12s}" for j in range(n)))

    print("\n  DEGRADATION vs the fold's own model, by cut-off gap (days)")
    print(f"  {'eval':>6s} {'gap':>5s} {'own':>10s} {'transferred':>12s} {'loss':>10s} "
          f"{'cropped':>10s} {'crop helps':>11s}")
    rows = []
    for j in range(n):
        own = M[j, j]
        for i in range(j):
            gap = (anchors[j] - anchors[i]) + HORIZON     # train ends at A_i-30, eval at A_j
            d_full = M[i, j] - own
            d_crop = Mc[i, j] - own
            rows.append({"train_fold": i, "eval_fold": j, "gap_days": int(gap),
                         "own": float(own), "transferred": float(M[i, j]),
                         "cropped": float(Mc[i, j])})
            print(f"  f{j:<5d} {gap:>5d} {own:>10.5f} {M[i, j]:>12.5f} {d_full:>+10.5f} "
                  f"{Mc[i, j]:>10.5f} {d_crop - d_full:>+11.5f}")

    # headline: the widest gap available, which is the closest analogue to the submission
    wide = max(rows, key=lambda r: r["gap_days"])
    print(f"\n  WIDEST GAP  {wide['gap_days']}d (train fold {wide['train_fold']} -> eval fold "
          f"{wide['eval_fold']}):")
    print(f"    own model            {wide['own']:.5f}")
    print(f"    transferred          {wide['transferred']:.5f}  "
          f"({wide['transferred'] - wide['own']:+.5f})")
    print(f"    transferred, cropped {wide['cropped']:.5f}  "
          f"({wide['cropped'] - wide['own']:+.5f})")

    gaps = np.array([r["gap_days"] for r in rows], float)
    degr = np.array([r["transferred"] - r["own"] for r in rows], float)
    print(f"\n  corr(gap, degradation) = {np.corrcoef(gaps, degr)[0, 1]:+.4f}   "
          f"slope = {np.polyfit(gaps, degr, 1)[0] * 100:+.5f} RMSLE per 100 days")

    ref = ROOT / "reports" / "eda" / "transfer_test.json"
    if ref.exists():
        G = np.array(json.loads(ref.read_text())["transfer"], float)
        gg, gd = [], []
        for j in range(n):
            for i in range(j):
                gg.append((anchors[j] - anchors[i]) + HORIZON)
                gd.append(G[i, j] - G[j, j])
        print(f"  GBDT (e0020) same statistic:      "
              f"corr {np.corrcoef(gg, gd)[0, 1]:+.4f}   "
              f"slope {np.polyfit(gg, gd, 1)[0] * 100:+.5f} RMSLE per 100 days")
        print("  -> if the seq slope is materially steeper, cut-off distance is the reason "
              "the CV gain did not transfer.")

    out = ROOT / "reports" / "eda" / "seq_transfer.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"config": args.config, "arch": cfg.get("arch"), "anchors": anchors,
         "transfer": M.tolist(), "transfer_cropped": Mc.tolist(), "rows": rows}, indent=2))
    print(f"\n  wrote reports/eda/seq_transfer.json   runtime {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
