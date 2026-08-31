#!/usr/bin/env python
"""
E-IDEA-04 -- inverse-variance sample weighting from MEASURED per-user label noise.  IDEAS.md §I10.

`scripts/noise_ceiling.py` measured that ~45% of Var(L) is single-window realisation noise.
That noise is wildly heteroscedastic across users: someone who buys 20x a month has a nearly
deterministic log1p(30-day sum), while someone at p(buy) ~ 0.5 has a target that flips between
0 and ~4.2 -- almost pure noise.  L2 weights both equally, so most of the gradient signal from
the second group is noise.

Batch Inverse-Variance Weighting (2107.04497) is the standard response: weight each sample by
1 / (sigma^2_noise + c).  Its stated precondition is "the labelling process can estimate the
variance of the noise distribution for each label" -- normally the blocker.  Here it is free:
the panel contains many past 30-day windows per user, so the within-user variance of L over
windows strictly BEFORE the anchor estimates sigma^2(u) causally.

The weights depend on x only (past windows), never on y, so the estimand stays E[L|x] -- the
same discipline as IDEAS.md §I1.  What changes is estimation efficiency, not what is estimated.

FOUR ARMS, and the last two exist because this screen has burned me twice:
    unweighted   the control
    inv_var      w = 1/(sigma^2 + c)      the BIV hypothesis: trust the quiet users
    prop_var     w =  sigma^2 + c         the OPPOSITE direction.  EXPERIMENTS.md §1b puts
                                          78.6% of Cov(L,M) in the buy/no-buy term, and the
                                          high-noise users ARE the undecided ones -- so the
                                          sign is genuinely not obvious and must be measured
                                          rather than assumed.
    shuffled     inv_var's weights, permuted across users -- same weight MARGINAL, no
                 relationship to the user.  If this moves rho, the screen cannot resolve
                 weighting at all and neither real arm means anything.

TIER = SCREEN.  Three-way user split; nothing is early-stopped or fitted on the scored users.

Run:  python3.11 scripts/screen_weights.py --anchor 2025-10-16
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


def past_window_noise(p, ai: int, n_win: int = 6, width: int = 30):
    """Within-user sd of log1p(30-day GMV) over `n_win` disjoint windows ending at the anchor.

    Strictly causal: window j covers [ai - (j+1)*width + 1, ai - j*width], all <= ai.  This is
    Var(theta drift) + Var(eps) rather than Var(eps) alone, but the lag curve in
    noise_ceiling.py shows theta moves ~19% over five months, so the drift share is small.
    """
    Ls = np.stack([np.log1p(p.wsum("gmv", ai - (j + 1) * width + 1, ai - j * width))
                   for j in range(n_win)], axis=1)
    return Ls.std(axis=1), Ls


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
    sd_hist, _ = past_window_noise(p, ai)
    sd_hist = sd_hist[keep]
    var_hist = sd_hist ** 2
    c = float(np.median(var_hist))                      # BIV's regulariser, set from the data
    log(f"  anchor {anchor}: {X.shape[0]:,} users x {X.shape[1]} features   RSS {rss_gb():.2f} GB")
    log(f"  per-user historical sd(L) over 6 past 30d windows: "
        f"median {np.median(sd_hist):.3f}  q10 {np.quantile(sd_hist,0.1):.3f}  "
        f"q90 {np.quantile(sd_hist,0.9):.3f}   regulariser c = {c:.4f}")

    # does the estimate actually predict this window's error?  If not, the premise is false and
    # every arm below is noise-fitting -- so check before spending the fits.
    rng = np.random.default_rng(1)
    u = rng.random(X.shape[0])
    tr, es, sc = u < 0.50, (u >= 0.50) & (u < 0.75), u >= 0.75

    W = {
        "unweighted": np.ones(X.shape[0]),
        "inv_var": 1.0 / (var_hist + c),
        "prop_var": var_hist + c,
    }
    W["shuffled"] = W["inv_var"][rng.permutation(X.shape[0])]
    for k in W:                                          # mean-1 so the effective lr matches
        W[k] = W[k] / W[k].mean()

    def rho(pred):
        return float(np.corrcoef(pred, L[sc])[0, 1])

    log(f"  split train {tr.sum():,} / earlystop {es.sum():,} / score {sc.sum():,}")
    log(f"\n  === rho on the held-out score split ===")
    base, preds = None, {}
    for k, w in W.items():
        pr = dict(PARAMS, objective="regression", metric="rmse")
        d = lgb.Dataset(X[tr], L[tr], weight=w[tr])
        v = lgb.Dataset(X[es], L[es], weight=w[es], reference=d)
        m = lgb.train(pr, d, num_boost_round=mtrees, valid_sets=[v],
                      callbacks=[lgb.early_stopping(80, verbose=False)])
        preds[k] = m.predict(X[sc], num_iteration=m.best_iteration)
        r = rho(preds[k])
        if base is None:
            base = r
        log(f"    {k:12s} rho {r:.5f}   d {r - base:+.5f}   iters {m.best_iteration:4d}"
            f"   w range [{w.min():.3f}, {w.max():.3f}]")

    # the premise, checked directly: does measured past noise predict this window's |residual|?
    res = np.abs(L[sc] - preds["unweighted"])
    log(f"\n  premise check: corr(historical sd(L), |residual| of the unweighted model) = "
        f"{np.corrcoef(sd_hist[sc], res)[0, 1]:+.4f}")
    log(f"                 corr(historical sd(L), L) = {np.corrcoef(sd_hist[sc], L[sc])[0,1]:+.4f}"
        f"   (if this is large the weight is just a level proxy)")
    log(f"\n  total {time.time() - t0:.0f}s   peak RSS {rss_gb():.2f} GB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", default="2025-10-16")
    ap.add_argument("--n", type=int, default=15000)
    ap.add_argument("--mtrees", type=int, default=1500)
    ap.add_argument("--max-gb", type=float, default=5.5)
    a = ap.parse_args()
    main(a.anchor, a.n, a.mtrees, a.max_gb)
