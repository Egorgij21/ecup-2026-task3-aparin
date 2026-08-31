#!/usr/bin/env python
"""
E-IDEA-07 -- is the MAGNITUDE term reachable?   IDEAS.md §I13.

§I13 measured, for the first time, a ceiling on the magnitude term:

    corr(L, predictor | y > 0)   <=  0.6001      achieved 0.4814 (e0049 OOF)  =  80.2% captured
    corr(Z, predictor)           <=  0.6623      achieved 0.5932              =  89.6% captured

So the model is proportionally weakest on magnitude -- the term EXPERIMENTS.md §1b never
measured, because it is only 21.4% of Cov(L, M) and attention went to the 78.6% share.

The first question is not "build a better magnitude model".  It is whether the joint model is
already extracting everything a DEDICATED one could: every model in this project is fitted on
all users at once, so its capacity is spent mostly on separating buyers from non-buyers, and
its ranking among buyers is a by-product.

    joint            train on all users, target L                  -- the installed design
    buyers_only      train on buyers only, target L                -- the dedicated model
    joint_subsample  train on a random subset of the SAME SIZE     -- SIZE CONTROL: buyers_only
                     as the buyers set                                sees ~57% of the rows, so
                                                                      any deficit could be data
                                                                      volume rather than design

All three are scored the same way: corr(L, pred) among held-out users with y > 0.  Three-way
user split; nothing is fitted or early-stopped on the scored users.

Run:  python3.11 scripts/screen_magnitude.py --anchor 2025-10-16
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


def main(anchor: str, n_users: int, mtrees: int, max_gb: float):
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
    L = np.log1p(p.target(ai, 30))[keep]
    Z = L > 0

    rng = np.random.default_rng(1)
    u = rng.random(X.shape[0])
    tr, es, sc = np.flatnonzero(u < 0.50), np.flatnonzero((u >= 0.50) & (u < 0.75)), \
        np.flatnonzero(u >= 0.75)
    sc_pos = sc[Z[sc]]
    tr_pos, es_pos = tr[Z[tr]], es[Z[es]]
    log(f"  anchor {anchor}: {X.shape[0]:,} users, buy-share {Z.mean():.4f}   RSS {rss_gb():.2f} GB")
    log(f"  train {tr.size:,} ({tr_pos.size:,} buyers) / es {es.size:,} / "
        f"score {sc.size:,} ({sc_pos.size:,} buyers)")

    def fit(rows, es_rows):
        d = lgb.Dataset(X[rows], L[rows])
        v = lgb.Dataset(X[es_rows], L[es_rows], reference=d)
        m = lgb.train(dict(PARAMS, objective="regression", metric="rmse"), d,
                      num_boost_round=mtrees, valid_sets=[v],
                      callbacks=[lgb.early_stopping(80, verbose=False)])
        return m.predict(X[sc_pos], num_iteration=m.best_iteration), m.best_iteration

    sub = rng.choice(tr, size=tr_pos.size, replace=False)          # matched-size control
    arms = {
        "joint": (tr, es),
        "buyers_only": (tr_pos, es_pos),
        "joint_subsample": (sub, es),
    }
    log(f"\n  === corr(L, pred) among held-out BUYERS ===")
    base = None
    for k, (r_, e_) in arms.items():
        pred, it = fit(r_, e_)
        r = float(np.corrcoef(pred, L[sc_pos])[0, 1])
        if base is None:
            base = r
        log(f"    {k:16s} corr {r:.5f}   d {r - base:+.5f}   n_train {r_.size:,}  iters {it}")
    log(f"\n    measured ceiling on this term (250k, conservative): 0.6001"
        f"   |  e0049 frozen-fold achieved: 0.4814")
    log(f"    NOTE the screen's own level is not comparable to 0.4814 (different population and"
        f"\n    protocol) -- the arm-to-arm delta is what this run is for.")
    log(f"\n  total {time.time() - t0:.0f}s   peak RSS {rss_gb():.2f} GB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", default="2025-10-16")
    ap.add_argument("--n", type=int, default=15000)
    ap.add_argument("--mtrees", type=int, default=1500)
    ap.add_argument("--max-gb", type=float, default=5.5)
    a = ap.parse_args()
    main(a.anchor, a.n, a.mtrees, a.max_gb)
