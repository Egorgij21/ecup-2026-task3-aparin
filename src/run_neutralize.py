#!/usr/bin/env python
"""
FEATURE NEUTRALISATION test (IDEAS.md §I21). No training: pure linear algebra on the frozen-fold
OOF predictions of an existing model plus the per-fold feature matrix.

    python src/run_neutralize.py --oof oof/e0049.parquet

WHY THIS IS NOT KILLED BY THE CALIBRATION ALGEBRA. §1b/§1t/§1q closed affine, monotone and
per-segment calibration: any function of the prediction M alone leaves rho unchanged (isotonic
-0.00006 vs a +0.000000 no-op control), because rho is invariant to monotone maps of M. Feature
neutralisation is NOT a function of M alone -- it subtracts M's linear projection onto a set of
FEATURE columns:

    M' = M - p * N @ lstsq(N, M) ,   N = [z-scored selected features | 1]

so it re-ranks users using information outside M, which is exactly the freedom a monotone map
lacks. It uses no labels (the projection is of the prediction onto features, computable at test
time byte-identically), so it is leakage-safe and applies unchanged to the 2026-02-13 submission.

WHY IT MIGHT HELP HERE SPECIFICALLY. The mechanism (Numerai; arXiv 2303.16117) trades mean
correlation for correlation STABILITY under distribution shift -- full projection cost ~22% of
mean corr but cut corr volatility ~35% and raised Sharpe ~20% out-of-sample. Our test anchor is
a measured 3.9x feature-space outlier driven by the lifetime/365-day features (anchor_drift.py,
e0056/e0057), and the GBDT is the cut-off-sensitive family (+0.00428 RMSLE per 100 days of gap,
§3b). Neutralising M against ONLY those drifting features should, if the mechanism is real,
RAISE rho on the most-shifted fold (4, 2025-10-16) while LOWERING it on the early folds -- a
signature no in-sample metric can fake. That per-fold divergence, not the mean, is the test.

ARMS. feature set in {drift = (_total$|^tenure|_365$), all 665, RANDOM matched-count control},
proportion p in {0, 0.2, 0.35, 0.5, 0.7, 1.0}. The random control (e0214 discipline) shows
whether any change is drift-specific or just the projection removing variance.

Each fold is a single anchor, so per-fold rho is already the WITHIN-anchor rho the metric is
(§1r) -- no pooling confound.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data import Panel                     # noqa: E402
from features import build                 # noqa: E402

BLOCKS = ["base", "counts", "rank", "visit", "channel", "dispersion", "sbcnomoment"]
DRIFT_PATTERNS = [r"_total$", r"^tenure", r"_365$"]
PROPORTIONS = [0.0, 0.2, 0.35, 0.5, 0.7, 1.0]


def log(m: str) -> None:
    print(m, flush=True)


def neutralise(M: np.ndarray, F: np.ndarray, p: float) -> np.ndarray:
    """M' = M - p * N (N^+ M), N = [z(F) | 1]. Labels never enter."""
    if p == 0.0 or F.shape[1] == 0:
        return M.copy()
    Fz = (F - F.mean(0)) / (F.std(0) + 1e-9)
    N = np.column_stack([Fz, np.ones(Fz.shape[0])])
    beta, *_ = np.linalg.lstsq(N, M, rcond=1e-6)
    return M - p * (N @ beta)


def rho(L: np.ndarray, M: np.ndarray) -> float:
    return float(np.corrcoef(L, M)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default="oof/e0049.parquet")
    ap.add_argument("--exp-id", default="e0294")
    args = ap.parse_args()
    t0 = time.time()

    spec = json.loads((ROOT / "data" / "fold_spec.json").read_text())
    oof = pl.read_parquet(ROOT / args.oof)
    fold_ids = sorted(oof["fold_id"].unique().to_list())
    p = Panel()
    rng = np.random.default_rng(0)

    # per (feature_set, p) -> list of per-fold rho
    sets = ["drift", "all", "random"]
    res = {s: {q: [] for q in PROPORTIONS} for s in sets}
    n_drift_cols = None

    for k in fold_ids:
        fs = spec["folds"][k]
        va = date.fromisoformat(fs["valid_anchor"])
        vai = p.idx(va)
        vkeep = p.active_in(vai - 29, vai)
        Xva, names = build(p, vai, vkeep, BLOCKS)
        users = p.users[vkeep]

        o = oof.filter(pl.col("fold_id") == k).sort("user_id")
        assert np.array_equal(o["user_id"].to_numpy(), users), f"fold {k} user mismatch"
        L = np.log1p(o["y_true"].to_numpy())
        M = np.log1p(np.maximum(o["y_pred"].to_numpy(), 0.0))

        drift_idx = [i for i, n in enumerate(names) if any(re.search(q, n) for q in DRIFT_PATTERNS)]
        n_drift_cols = len(drift_idx)
        Fdrift = Xva[:, drift_idx].astype(np.float64)
        Fall = Xva.astype(np.float64)
        Frand = rng.standard_normal((Xva.shape[0], len(drift_idx)))   # matched count

        base = rho(L, M)
        log(f"  fold {k} {va}  n={M.size:,}  drift_cols={len(drift_idx)}/{len(names)}  base rho={base:.5f}")
        for s, F in (("drift", Fdrift), ("all", Fall), ("random", Frand)):
            for q in PROPORTIONS:
                res[s][q].append(rho(L, neutralise(M, F, q)))

    log(f"\n  drift columns matched: {n_drift_cols}  (patterns {DRIFT_PATTERNS})")
    log(f"\n  per-fold rho by proportion p  (fold 4 = most test-like)\n")
    for s in sets:
        log(f"  [{s}]")
        log(f"    {'p':>5s} " + " ".join(f"f{k}".rjust(8) for k in fold_ids)
            + f" {'mean':>8s} {'f4-f0':>8s}")
        for q in PROPORTIONS:
            r = np.array(res[s][q])
            log(f"    {q:>5.2f} " + " ".join(f"{x:8.5f}" for x in r)
                + f" {r.mean():8.5f} {r[-1] - r[0]:+8.5f}")
        log("")

    # LOFO-honest p selection per feature set: pick p max mean rho on the other folds, apply held-out
    log("  LOFO-honest: choose p on the other 4 folds, score the held-out fold (vs p=0)\n")
    for s in ["drift", "all"]:
        held, base_held = [], []
        R = {q: np.array(res[s][q]) for q in PROPORTIONS}
        for k in fold_ids:
            others = [j for j in range(len(fold_ids)) if j != k]
            best_q = max(PROPORTIONS, key=lambda q: R[q][others].mean())
            held.append(R[best_q][k]); base_held.append(R[0.0][k])
        held = np.array(held); base_held = np.array(base_held)
        d = held.mean() - base_held.mean()
        log(f"  [{s}]  LOFO rho {held.mean():.5f} vs p=0 {base_held.mean():.5f}  "
            f"Δ {d:+.5f}  per-fold Δ {np.round(held - base_held, 5).tolist()}")

    log(f"\n  runtime {(time.time() - t0) / 60:.1f} min  (no training, no OOF written -- diagnostic only)")


if __name__ == "__main__":
    main()
