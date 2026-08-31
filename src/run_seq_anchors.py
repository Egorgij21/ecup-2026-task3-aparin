#!/usr/bin/env python
"""
Multi-anchor test-time augmentation: predict at A, A-1, ... A-(K-1) and recombine.

    python src/run_seq_anchors.py --config configs/e0101_seq_gru.yaml --anchors 30

THE IDEA.  We currently emit ONE prediction, made at the final anchor A, for the window
[A+1, A+30].  But the same trained model can also be run at anchor A-k for any k, and at the
real prediction time we know everything up to A -- so the realised GMV over [A-k+1, A] is
observed, not forecast.  That gives, for each k, an independent-ish view of the same user:

    pred_k                  ~= GMV over [A-k+1, A-k+30]          (model output at anchor A-k)
    known_k                  = GMV over [A-k+1, A]               (OBSERVED, k days)
    pred_k - known_k        ~= GMV over [A+1, A-k+30]            (the first 30-k target days)

Rescaling by 30/(30-k) puts every k on the target's scale.  k=0 is exactly today's prediction,
so the baseline is nested inside the family and the comparison is paired.

WHY IT MIGHT PAY.  Thirty overlapping views of one user, averaged, have less variance than one.
WHY IT MIGHT NOT.  Same weights, near-identical inputs -> highly correlated errors (seed
averaging is worth ~0.00003), and the k-th view covers only 30-k of the 30 target days, so its
information decays to nothing at k=30.  The subtraction also injects realised-spend variance.

TRAINING IS UNTOUCHED.  `t_hi = vai - HORIZON` exactly as in `run_seq.py`, same seed, same
config -- this is a pure TEST-TIME change, so it is one change per CLAUDE.md rule 2 and the
result is directly comparable to the parent's logged row.

NO LEAKAGE.  Every scoring anchor A-k (k <= 29) is strictly LATER than the last training day
A-30, and `known_k` uses only days <= A, which are observed at prediction time.  Asserted below.

THIS SCRIPT ONLY DUMPS THE RAW ARRAYS.  Weighting and smoothing are searched offline by
`src/anchor_blend.py` on the saved parquet -- one GPU run, then unlimited free exploration.
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
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from metrics import rmsle                                                      # noqa: E402
from run_seq import (HORIZON, LAST_CLEAN_ANCHOR, MIN_HISTORY_DAYS, train_one)  # noqa: E402
from seqdata import build_seq_panel                                            # noqa: E402


def log(m: str) -> None:
    print(m, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--anchors", type=int, default=30, help="K: score at A-0 .. A-(K-1)")
    ap.add_argument("--tag", default=None, help="output name (default: <exp_id>_anchors)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    exp_id = cfg["exp_id"]
    tag = args.tag or f"{exp_id}_anchors"
    K = int(args.anchors)
    assert 1 <= K <= HORIZON, f"k must stay inside the horizon; got {K}"

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"\n=== {tag}: multi-anchor TTA over K={K} anchors, base config {exp_id} ===")
    log(f"    arch={cfg.get('arch')} d={cfg.get('d_model')} blocks={cfg.get('n_blocks')} "
        f"epochs={cfg['epochs']} seed={cfg['seed']}  device={device}")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    sp = build_seq_panel(derived=bool(cfg.get("derived_channels", False)),
                         ranks=bool(cfg.get("rank_channels", False)))
    Xg = torch.from_numpy(sp.X).to(device)
    Yg = torch.from_numpy(sp.Y).to(device)
    popg = torch.from_numpy(sp.pop).to(device)

    spec = json.loads((ROOT / "data" / "fold_spec.json").read_text())
    folds = pl.read_parquet(ROOT / "data" / "folds.parquet")
    fold_ids = sorted(folds["fold_id"].unique().to_list())
    t_lo = MIN_HISTORY_DAYS - 1

    out_frames, base_per_fold = [], []
    for f in fold_ids:
        fs = spec["folds"][f]
        va = date.fromisoformat(fs["valid_anchor"])
        vai = sp.idx(va)
        t_hi = vai - HORIZON
        assert sp.day(t_hi) == va - timedelta(days=HORIZON)
        assert t_hi <= sp.idx(LAST_CLEAN_ANCHOR), "training days reach the guard zone"
        # Every scoring anchor must lie strictly after the last TRAINING day, or the model
        # would be scored where it was fitted.  K <= HORIZON guarantees it; assert anyway.
        assert vai - (K - 1) > t_hi, f"scoring anchor {vai - (K - 1)} <= last train day {t_hi}"

        score_at = [(vai - k, None) for k in range(K)]
        preds, _ = train_one(sp, Xg, Yg, popg, cfg, t_lo, t_hi, score_at, device, f"fold {f}")

        vkeep = sp.pop[:, vai]
        fv = folds.filter(pl.col("fold_id") == f).sort("user_id")
        assert np.array_equal(fv["user_id"].to_numpy(), sp.users[vkeep]), "fold population drift"
        yva = fv["target"].to_numpy()

        d = {"fold_id": np.full(yva.size, f, np.int8),
             "user_id": fv["user_id"].to_numpy(), "y_true": yva}
        for k in range(K):
            pk = preds[(vai - k, None)][vkeep]
            # realised GMV over [A-k+1, A]: k observed days.  k=0 -> empty window -> exactly 0.
            gk = sp.wgmv(vai - k + 1, vai)[vkeep]
            if k == 0:
                assert np.all(gk == 0.0), "k=0 correction window must be empty"
            d[f"p{k:02d}"] = pk.astype(np.float32)
            d[f"g{k:02d}"] = gk.astype(np.float32)
        out_frames.append(pl.DataFrame(d))

        s0 = rmsle(yva, preds[(vai, None)][vkeep])
        base_per_fold.append(s0)
        log(f"    fold {f} {va}  n={yva.size:>7,}  k=0 rmsle={s0:.5f}  "
            f"[{(time.time() - t0) / 60:.1f}m]")

    out = pl.concat(out_frames)
    (ROOT / "oof").mkdir(exist_ok=True)
    dest = ROOT / "oof" / f"{tag}.parquet"
    out.write_parquet(dest)
    pf = np.array(base_per_fold)
    log(f"\n  wrote {dest.relative_to(ROOT)}  ({out.height:,} rows x {out.width} cols)")
    log(f"  baseline (k=0, = {exp_id}): cv_mean {pf.mean():.5f}  folds {np.round(pf, 5).tolist()}")
    log(f"  runtime {(time.time() - t0) / 60:.1f} min")
    log(f"  next: python src/anchor_blend.py --tag {tag}")


if __name__ == "__main__":
    main()
