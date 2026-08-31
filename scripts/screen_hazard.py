#!/usr/bin/env python
"""
E-IDEA-03 -- discrete-time HAZARD supervision for the buy/no-buy term.   IDEAS.md §I9.

`scripts/noise_ceiling.py` measures the ceiling on corr(1(y>0), predictor) at 0.662 against a
best achieved 0.593 (e0160).  That is the largest measured gap in the project, on the term
EXPERIMENTS.md §1b puts at 78.6% of Cov(L, M).  Every classifier built here -- e0160 LightGBM
binary, e0161/e0162 GRU-BCE -- was trained on ONE label per user: did they buy in the 30-day
window.  A user who bought on day 2 and a user who bought on day 29 are the same training
example.

Discrete-time survival uses the timing.  Split the horizon into J intervals, expand each user
into one row per interval they survive into, and fit a binary model to the per-interval
hazard.  Then

    P(buy within 30d | x) = 1 - prod_j (1 - h_j(x))

Same estimand as the binary classifier, strictly more supervision from the same data, and no
extra information.  This is "survival stacking" (2107.13480): it recasts survival as
classification, so it runs on the existing feature matrix with the existing LightGBM.

Refs: Gensheimer & Narasimhan, Nnet-survival (1805.00917); Kvamme & Borgan (1910.06724);
Craig, Zhong & Tibshirani, survival stacking (2107.13480); "Buy when?" (2308.14343).
PAPERS.md 6.2 rated this `P1` and it was never built.

TIER = SCREEN (CLAUDE.md §4.2).  Three-way user split, no arm ever sees the scored users.
The reported statistic is corr(Z, p) on the score split, which is directly comparable to the
0.662 ceiling and to the 0.593 e0160 achieves.

Run:  python3.11 scripts/screen_hazard.py --anchor 2025-10-16
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from screen_features import cap_memory, log, make_subset, rss_gb        # noqa: E402
from screen_loss import EXISTING, PARAMS                                # noqa: E402


def first_buy_day(p, ai: int, horizon: int = 30) -> np.ndarray:
    """Day offset (1..horizon) of the first buy-day after the anchor; horizon+1 if none."""
    buys = np.stack([(p.cs_buy[:, ai + d + 1] - p.cs_buy[:, ai + d]) > 0
                     for d in range(1, horizon + 1)], axis=1)          # (users, horizon)
    any_buy = buys.any(axis=1)
    return np.where(any_buy, buys.argmax(axis=1) + 1, horizon + 1)


def stack_survival(fb: np.ndarray, rows: np.ndarray, n_int: int, width: int):
    """Survival stacking: user u contributes one row per interval it enters.

    Interval j covers days [j*width+1, (j+1)*width].  Label 1 on the interval containing the
    first buy, 0 for every earlier interval; a user censored at the horizon contributes all
    J intervals with label 0.  Returns (row_index_into_users, interval_index, label).
    """
    ui, ji, yi = [], [], []
    ev = np.minimum((fb - 1) // width, n_int)          # interval of the event, n_int = censored
    for j in range(n_int):
        alive = rows[ev[rows] >= j]                    # still at risk at the start of interval j
        ui.append(alive)
        ji.append(np.full(alive.size, j, np.int32))
        yi.append((ev[alive] == j).astype(np.int8))
    return np.concatenate(ui), np.concatenate(ji), np.concatenate(yi)


def main(anchor: str, n_users: int, n_int: int, mtrees: int, max_gb: float):
    import lightgbm as lgb
    from data import Panel
    from features import build

    cap_memory(max_gb)
    t0 = time.time()
    p = Panel(path=make_subset(n_users))
    ai = p.idx(date.fromisoformat(anchor))
    keep = p.active_in(ai - 29, ai)
    X, names = build(p, ai, keep, EXISTING)
    X = X.astype(np.float32)
    y = p.target(ai, 30)[keep]
    Z = (y > 0).astype(np.float64)
    L = np.log1p(y)
    fb = first_buy_day(p, ai, 30)[keep]
    width = 30 // n_int
    log(f"  anchor {anchor}: {X.shape[0]:,} users x {X.shape[1]} features   "
        f"buy-share {Z.mean():.4f}   RSS {rss_gb():.2f} GB")
    log(f"  first-buy-day distribution: median {np.median(fb[fb <= 30]):.0f} among buyers, "
        f"{n_int} intervals of {width}d")

    rng = np.random.default_rng(1)
    u = rng.random(X.shape[0])
    tr, es, sc = np.flatnonzero(u < 0.50), np.flatnonzero((u >= 0.50) & (u < 0.75)), \
        np.flatnonzero(u >= 0.75)
    log(f"  split train {tr.size:,} / earlystop {es.size:,} / score {sc.size:,}")

    def rho_Z(pred):
        return float(np.corrcoef(pred, Z[sc])[0, 1])

    def rho_L(pred):
        return float(np.corrcoef(pred, L[sc])[0, 1])

    results = {}

    # ---------------- arm 1: the installed design -- one binary label per user -------------
    pr = dict(PARAMS, objective="binary", metric="auc")
    d = lgb.Dataset(X[tr], Z[tr])
    v = lgb.Dataset(X[es], Z[es], reference=d)
    m = lgb.train(pr, d, num_boost_round=mtrees, valid_sets=[v],
                  callbacks=[lgb.early_stopping(80, verbose=False)])
    results["binary_30d"] = m.predict(X[sc], num_iteration=m.best_iteration)
    log(f"    binary_30d      iters {m.best_iteration}")

    # ---------------- arm 2: discrete-time hazard, survival-stacked -----------------------
    su, sj, sy = stack_survival(fb, tr, n_int, width)
    eu, ej, ey = stack_survival(fb, es, n_int, width)
    Xs = np.column_stack([X[su], sj.astype(np.float32)])
    Xe = np.column_stack([X[eu], ej.astype(np.float32)])
    log(f"    hazard stack: {Xs.shape[0]:,} train rows from {tr.size:,} users "
        f"({Xs.shape[0]/tr.size:.1f} per user), event rate {sy.mean():.4f}")
    hp = dict(PARAMS, objective="binary", metric="binary_logloss")
    # no feature_name: the screen's block list emits a few duplicate names across blocks
    # (e.g. ewm_gmv_hl7 from both `trend` and `ewm`), which LightGBM rejects.  Harmless here
    # -- nothing in this script reads importances by name.
    dh = lgb.Dataset(Xs, sy)
    vh = lgb.Dataset(Xe, ey, reference=dh)
    mh = lgb.train(hp, dh, num_boost_round=mtrees, valid_sets=[vh],
                   callbacks=[lgb.early_stopping(80, verbose=False)])
    del Xs, Xe
    surv = np.ones(sc.size)
    for j in range(n_int):
        Xj = np.column_stack([X[sc], np.full(sc.size, j, np.float32)])
        surv *= 1.0 - mh.predict(Xj, num_iteration=mh.best_iteration)
    results["hazard"] = 1.0 - surv
    log(f"    hazard          iters {mh.best_iteration}")

    log(f"\n  === corr with the buy flag Z on the held-out score split ===")
    base = None
    for k, v_ in results.items():
        r = rho_Z(v_)
        extra = ""
        if base is None:
            base = r
        else:
            rr = float(np.corrcoef(v_, results["binary_30d"])[0, 1])
            extra = f"   d {r - base:+.5f}   corr_vs_binary {rr:.5f}"
        log(f"    {k:14s} corr(Z, p) {r:.5f}   corr(L, p) {rho_L(v_):.5f}{extra}")
    log(f"\n    ceiling on corr(Z, .) from scripts/noise_ceiling.py: "
        f"0.6623 conservative / 0.6755 drift-corrected")
    log(f"    e0160 (full-data LightGBM binary, frozen folds) achieves 0.59317")
    log(f"\n  total {time.time() - t0:.0f}s   peak RSS {rss_gb():.2f} GB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", default="2025-10-16")
    ap.add_argument("--n", type=int, default=15000)
    ap.add_argument("--intervals", type=int, default=6)
    ap.add_argument("--mtrees", type=int, default=1500)
    ap.add_argument("--max-gb", type=float, default=5.5)
    a = ap.parse_args()
    main(a.anchor, a.n, a.intervals, a.mtrees, a.max_gb)
