#!/usr/bin/env python
"""
Search weighting and smoothing schemes over the multi-anchor predictions.

    python src/anchor_blend.py --tag e0101_anchors

Consumes `oof/<tag>.parquet` from `src/run_seq_anchors.py`: for each user, K predictions made
at anchors A-0 .. A-(K-1) plus the realised GMV over each [A-k+1, A] correction window.
Nothing here touches a GPU, so the scheme space is free to explore exhaustively.

THE ESTIMATOR FAMILY.  For anchor k,

    raw_k  = p_k                                  no correction: pure test-time augmentation
    sub_k  = max(p_k - g_k, 0)                    subtract the observed [A-k+1, A] spend
    resc_k = sub_k * H / (H - k)                  ... and rescale to the full-horizon scale

`raw` is the control that isolates whether the SUBTRACTION helps or whether any gain is just
averaging.  `resc` is the proposal as stated.  Note rho is invariant to a global affine map, so
the rescale can only matter through its effect on the RELATIVE weighting of different k -- which
is exactly why it has to be tested rather than assumed.

HONESTY.  Every fixed scheme is parameter-free, but CHOOSING among ~40 of them on the same five
folds is selection, and §1k is this project's lesson about exactly that.  So the headline number
is the leave-one-fold-out one: pick the best scheme on four folds, score it on the fifth.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from metrics import rmsle          # noqa: E402

H = 30                              # competition horizon


def opt_affine(Ly, Lp, fold):
    """RMSLE after the per-fold optimal affine map in log space -- the §1b ceiling.

    Closed form rather than polyfit: with the map applied in log space the metric is exactly
    the residual RMS, since log1p(expm1(z)) = z.  ~50x faster, and this runs ~450 times.
    """
    out = []
    for k in np.unique(fold):
        m = fold == k
        x, t = Lp[m], Ly[m]
        xm, tm = x.mean(), t.mean()
        v = ((x - xm) ** 2).mean()
        a = (((x - xm) * (t - tm)).mean() / v) if v > 0 else 0.0
        r = t - (a * x + (tm - a * xm))
        out.append(float(np.sqrt((r ** 2).mean())))
    return float(np.mean(out)), out


def build_estimators(P, G, kind):
    """(n_users, K) matrix of per-anchor estimates of the 30-day target, in RAW gmv units."""
    K = P.shape[1]
    if kind == "raw":
        return P
    S = np.maximum(P - G, 0.0)
    if kind == "sub":
        return S
    if kind == "resc":
        sc = np.array([H / (H - k) for k in range(K)], dtype=np.float32)
        return S * sc
    raise ValueError(kind)


def weights(name, K):
    k = np.arange(K, dtype=np.float64)
    if name == "k0":
        w = np.zeros(K); w[0] = 1.0
    elif name.startswith("linear"):                       # the proposal: w_k = (H-k)/H
        cut = K if name == "linear" else int(name.split("_")[1])
        w = np.maximum(H - k, 0.0) / H
        w[cut:] = 0.0
    elif name.startswith("uniform_"):
        cut = int(name.split("_")[1]); w = np.zeros(K); w[:cut] = 1.0
    elif name.startswith("exp_"):
        tau = float(name.split("_")[1]); w = np.exp(-k / tau)
    elif name.startswith("quad"):                          # w_k = ((H-k)/H)^2
        w = (np.maximum(H - k, 0.0) / H) ** 2
    elif name.startswith("sqrt"):
        w = np.sqrt(np.maximum(H - k, 0.0) / H)
    else:
        raise ValueError(name)
    s = w.sum()
    assert s > 0, name
    return w / s


def smooth(E, kind):
    """Smoothing ACROSS the anchor axis, before weighting.

    Cumsum form, not np.convolve: convolve's 'same' ZERO-PADS the edges, and k=0 IS an edge --
    it would drag the single most informative anchor toward zero.  This shrinks the window at
    the boundary instead, so every output is a mean of real values only.  Also vectorised;
    apply_along_axis over 1.07M rows is minutes.
    """
    if kind == "none":
        return E
    assert kind.startswith("ma"), kind
    n = int(kind[2:]); half = n // 2
    K = E.shape[1]
    C = np.concatenate([np.zeros((E.shape[0], 1)), np.cumsum(E, axis=1)], axis=1)
    lo = np.clip(np.arange(K) - half, 0, K)
    hi = np.clip(np.arange(K) + half + 1, 0, K)
    return (C[:, hi] - C[:, lo]) / (hi - lo)[None, :]


def combine(E, w, space, LE=None):
    """`LE` = precomputed log1p(E).  Passing it turns each call into one matvec; recomputing
    log1p per (weight, space) is 675 passes over a 32M-element array and dominates everything."""
    if LE is None:
        LE = np.log1p(E)
    if space == "log":
        return LE @ w
    if space == "raw":
        return np.log1p(E @ w)
    if space == "extrap":
        # A weighted mean shrinks variance but keeps whatever bias grows with k.  Instead fit
        # log1p(est_k) = a + b*k per user (weighted) and take the INTERCEPT: all K anchors
        # contribute to the fit, but the reported value is the unbiased k=0 point.
        k = np.arange(E.shape[1], dtype=np.float64)
        mk = float((w * k).sum())
        vk = float((w * k * k).sum()) - mk ** 2
        if vk <= 1e-12:                       # degenerate (e.g. w = k0): no slope to fit
            return None
        mL = LE @ w
        ckL = (LE @ (w * k)) - mk * mL
        return mL + (ckL / vk) * (0.0 - mk)   # intercept = value extrapolated to k = 0
    raise ValueError(space)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="e0101_anchors")
    ap.add_argument("--sigma", type=float, default=0.00023,
                    help="seq-family sigma_noise on the 5-fold mean (EXPERIMENTS.md §1l)")
    args = ap.parse_args()

    d = pd.read_parquet(ROOT / "oof" / f"{args.tag}.parquet").sort_values(["fold_id", "user_id"])
    K = sum(c.startswith("p") and c[1:].isdigit() for c in d.columns)
    P = d[[f"p{k:02d}" for k in range(K)]].values.astype(np.float64)
    G = d[[f"g{k:02d}" for k in range(K)]].values.astype(np.float64)
    y, fold = d.y_true.values, d.fold_id.values
    Ly = np.log1p(y)
    print(f"\n  {args.tag}: {len(d):,} rows, K={K} anchors, {len(np.unique(fold))} folds")

    base_L = np.log1p(P[:, 0])
    b_mean, b_per = opt_affine(Ly, base_L, fold)
    b_rho = float(np.corrcoef(Ly, base_L)[0, 1])
    b_raw = float(np.mean([rmsle(y[fold == k], P[fold == k, 0]) for k in np.unique(fold)]))
    print(f"  BASELINE k=0 : raw {b_raw:.5f}   calibrated {b_mean:.5f}   rho {b_rho:.5f}")
    print(f"               folds {[round(v, 5) for v in b_per]}")

    # how much does the correction term actually move things?
    print(f"\n  correction magnitude: mean g_k / mean p_k")
    for k in (0, 1, 5, 10, 20, K - 1):
        print(f"    k={k:>2d}   mean p {P[:, k].mean():>9.2f}   mean g {G[:, k].mean():>9.2f}"
              f"   share of p_k clipped to 0: {100 * (P[:, k] <= G[:, k]).mean():>5.2f}%")

    WNAMES = (["k0", "linear", "quad", "sqrt"]
              + [f"uniform_{c}" for c in (2, 3, 5, 7, 10, 15, 20, 30) if c <= K]
              + [f"linear_{c}" for c in (3, 5, 7, 10, 15, 20) if c <= K]
              + [f"exp_{t}" for t in (1, 2, 3, 5, 7, 10, 15)])
    rows = []
    import time
    t_start = time.time()
    for kind in ("raw", "sub", "resc"):
        E0 = build_estimators(P, G, kind)
        for sm in ("none", "ma3", "ma5"):
            print(f"    [{time.time()-t_start:6.1f}s] {kind}/{sm} ...", flush=True)
            E = smooth(E0, sm)
            LE = np.log1p(E)              # hoisted: 9 log1p passes total, not 675
            for wn in WNAMES:
                w = weights(wn, K)
                for space in ("log", "raw", "extrap"):
                    L = combine(E, w, space, LE)
                    if L is None or not np.isfinite(L).all():
                        continue
                    m, per = opt_affine(Ly, L, fold)
                    rows.append(dict(kind=kind, smooth=sm, w=wn, space=space, cal=m,
                                     rho=float(np.corrcoef(Ly, L)[0, 1]),
                                     per=per, delta=m - b_mean,
                                     wins=sum(a < b for a, b in zip(per, b_per))))
    R = pd.DataFrame(rows).sort_values("cal")
    print(f"\n  === {len(R)} schemes, best 20 by calibrated 5-fold mean ===")
    print(f"  {'kind':5s} {'smooth':7s} {'weights':11s} {'space':5s} {'cal':>9s} {'delta':>9s} "
          f"{'rho':>9s} {'wins':>5s}")
    for _, r in R.head(20).iterrows():
        print(f"  {r['kind']:5s} {r['smooth']:7s} {r['w']:11s} {r['space']:5s} {r.cal:>9.5f} "
              f"{r.delta:>+9.5f} {r.rho:>9.5f} {r.wins}/5")
    print(f"\n  the proposal as stated (resc / linear / log):")
    q = R[(R.kind == 'resc') & (R.w == 'linear') & (R.smooth == 'none') & (R.space == 'log')]
    for _, r in q.iterrows():
        print(f"    cal {r.cal:.5f}  delta {r.delta:+.5f}  rho {r.rho:.5f}  wins {r.wins}/5  "
              f"folds {[round(v, 5) for v in r['per']]}")
    print(f"\n  best per estimator kind (isolates the subtraction):")
    for kind in ("raw", "sub", "resc"):
        r = R[R.kind == kind].iloc[0]
        print(f"    {kind:5s}  {r['smooth']:5s} {r['w']:11s} {r['space']:4s}  cal {r.cal:.5f}  "
              f"delta {r.delta:+.5f}  wins {r.wins}/5")

    # ------------------------------------------------------------------ honest selection
    print(f"\n  === LOFO: choose the scheme on 4 folds, score it on the 5th ===")
    fids = list(np.unique(fold))
    tot = []
    for hold in fids:
        tr = [i for i, k in enumerate(fids) if k != hold]
        hi = fids.index(hold)
        best = min(rows, key=lambda r: np.mean([r["per"][i] for i in tr]))
        gain = best["per"][hi] - b_per[hi]
        tot.append(gain)
        print(f"    hold f{hold}: picks {best['kind']}/{best['smooth']}/{best['w']}/{best['space']}"
              f"   held-out {best['per'][hi]:.5f} vs baseline {b_per[hi]:.5f}   {gain:+.5f}")
    print(f"\n  HONEST mean gain vs k=0 baseline: {np.mean(tot):+.5f}"
          f"   (sigma_noise on the mean = {args.sigma:.5f}, so 2 sigma = {2*args.sigma:.5f})")
    verdict = ("KEEP" if abs(np.mean(tot)) > 2 * args.sigma and np.mean(tot) < 0
               else "no effect")
    print(f"  verdict per CLAUDE.md §3.4: {verdict}")
    R.drop(columns=["per"]).to_csv(ROOT / "reports" / "eda" / f"{args.tag}_schemes.csv",
                                   index=False)
    print(f"\n  full table -> reports/eda/{args.tag}_schemes.csv")


if __name__ == "__main__":
    main()
