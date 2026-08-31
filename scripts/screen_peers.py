#!/usr/bin/env python
"""
E-IDEA-06 -- does CROSS-USER information exist?   IDEAS.md §I12.  Closes BACKLOG Band D e0030.

Every feature in the 1021-column set is a function of ONE user's own history.  §I11's headroom
map says the model captures 63% of the reliable variance for 8+-buy-day users but only 22% for
0-buy-day users -- the shape you get when what binds is *how much evidence exists per user*.
Borrowing strength from similar users is the standard response and the one information axis
this project has never opened (`BACKLOG` Band D e0030 k-NN / e0031 cluster-id, costed at 5c and
2c, neither ever run).

CONSTRAINT that shapes the design: at test time NO user's target is known -- all 250k are
predicted at once.  So a peer feature may only use neighbours' *observed past*, never their
outcome in the scored window.  Every candidate below obeys that.

    peer_gmv30      mean of neighbours' log1p(GMV in the last 30d)      -- the peer level
    peer_rel        the user's own log1p(GMV last 30d) MINUS peer_gmv30 -- position within the
                    local peer group.  This is the one that is not reconstructible from
                    per-user features: the `rank` block (e0004) gives POPULATION-wide rank,
                    never rank inside a behavioural neighbourhood.
    peer_buyrate    fraction of neighbours with any buy-day in the last 30d
    peer_dist       mean distance to the k neighbours -- how typical this user is

Screened as single columns against the installed set, which is the regime `FEATURES.md`
calibrated at ~+-0.001, plus an i.i.d. noise column measured in the SAME run as the reference.
(BACKLOG.md: the model-level band on this harness is ~+-0.004, four times wider -- adding a
column is the one thing this screen can still resolve.)

Run:  python3.11 scripts/screen_peers.py --anchor 2025-10-16
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


def peer_features(p, ai: int, keep: np.ndarray, k: int):
    """k-NN in a small standardised behaviour space built ONLY from data <= ai."""
    from sklearn.neighbors import NearestNeighbors

    def w(col, a, b):
        return np.log1p(p.wsum(col, ai - a + 1, ai - b))

    cols = [w("gmv", 30, 0), w("gmv", 90, 0), w("gmv", 365, 0),
            w("ord", 90, 0), w("cart", 90, 0), w("srch", 90, 0),   # Panel.cs uses the SHORT names
            (p.cs_buy[:, ai + 1] - p.cs_buy[:, ai - 89]).astype(np.float64),
            (p.cs_days[:, ai + 1] - p.cs_days[:, ai - 89]).astype(np.float64),
            (ai - p.last_act[:, ai]).astype(np.float64)]
    E = np.column_stack([c[keep] for c in cols])
    E = (E - E.mean(0)) / np.maximum(E.std(0), 1e-9)

    nn = NearestNeighbors(n_neighbors=k + 1).fit(E)
    dist, idx = nn.kneighbors(E)
    idx, dist = idx[:, 1:], dist[:, 1:]                  # drop self

    gmv30 = np.log1p(p.wsum("gmv", ai - 29, ai))[keep]
    buy30 = ((p.cs_buy[:, ai + 1] - p.cs_buy[:, ai - 29]) > 0).astype(np.float64)[keep]
    peer_gmv30 = gmv30[idx].mean(1)
    return [("peer_gmv30", peer_gmv30),
            ("peer_rel", gmv30 - peer_gmv30),
            ("peer_buyrate", buy30[idx].mean(1)),
            ("peer_dist", dist.mean(1))]


def main(anchor: str, n_users: int, k: int, mtrees: int, max_gb: float):
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
    cands = peer_features(p, ai, keep, k)
    log(f"  anchor {anchor}: {X.shape[0]:,} users x {X.shape[1]} features, k={k} neighbours"
        f"   RSS {rss_gb():.2f} GB")

    rng = np.random.default_rng(1)
    u = rng.random(X.shape[0])
    tr, es, sc = u < 0.50, (u >= 0.50) & (u < 0.75), u >= 0.75

    def fit_rho(Xa):
        d = lgb.Dataset(Xa[tr], L[tr])
        v = lgb.Dataset(Xa[es], L[es], reference=d)
        m = lgb.train(dict(PARAMS, objective="regression", metric="rmse"), d,
                      num_boost_round=mtrees, valid_sets=[v],
                      callbacks=[lgb.early_stopping(80, verbose=False)])
        return float(np.corrcoef(m.predict(Xa[sc], num_iteration=m.best_iteration), L[sc])[0, 1])

    base = fit_rho(X)
    log(f"\n  baseline ({X.shape[1]} features)   rho {base:.5f}")
    noiz = fit_rho(np.column_stack([X, rng.standard_normal(X.shape[0]).astype(np.float32)]))
    log(f"  noise control (1 i.i.d. col)     rho {noiz:.5f}   d {noiz - base:+.5f}"
        f"   <-- the reference every candidate must beat")
    for nm, c in cands:
        r = fit_rho(np.column_stack([X, c.astype(np.float32)]))
        flag = "  <<" if (r - base) > max(2e-4, abs(noiz - base)) else ""
        log(f"  + {nm:16s}               rho {r:.5f}   d {r - base:+.5f}"
            f"   corr_with_L {np.corrcoef(c, L)[0,1]:+.4f}{flag}")
    allc = np.column_stack([X] + [c.astype(np.float32) for _, c in cands])
    r = fit_rho(allc)
    log(f"  + all 4 peer columns             rho {r:.5f}   d {r - base:+.5f}")
    log(f"\n  total {time.time() - t0:.0f}s   peak RSS {rss_gb():.2f} GB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", default="2025-10-16")
    ap.add_argument("--n", type=int, default=15000)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--mtrees", type=int, default=1500)
    ap.add_argument("--max-gb", type=float, default=5.5)
    a = ap.parse_args()
    main(a.anchor, a.n, a.k, a.mtrees, a.max_gb)
