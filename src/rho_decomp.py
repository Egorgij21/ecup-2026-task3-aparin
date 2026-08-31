#!/usr/bin/env python
"""
Where does the missing rho live -- the buy/no-buy decision, or the magnitude?

    python src/rho_decomp.py

THE QUESTION.  The score is exactly `RMSLE = sd_L * sqrt(1 - rho^2)` (EXPERIMENTS.md §2), and
rho has moved 0.698 -> 0.704 across ninety experiments spanning LightGBM, AutoGluon, TCN,
transformer and GRU, two CV protocols and 800+ features.  A four-feature GRU reaches 0.7007.
Meanwhile DATA.md §8.2's oracle -- know only WHO buys, predict one constant for them -- implies
rho ~ 0.81.  So the headroom is real and it is not in the places we have been looking.  Before
spending another hour, partition it.

THE PARTITION, exact by the law of total covariance.  With `Z = 1{y > 0}` and `L = log1p(y)`
(so L = 0 whenever Z = 0):

    Cov(L, M) = P(Z=1) * Cov(L, M | Z=1)   +   Cov( E[L|Z], E[M|Z] )
                \_______ WITHIN ________/       \______ BETWEEN ______/
                 ranking magnitudes              separating buyers
                 among actual buyers             from non-buyers

Every point of rho comes from one of those two terms, and they are measured here rather than
argued about.  Then three oracle substitutions bound what fixing each half is worth:

    perfect split + our magnitudes   M = Z * ghat(M)      ghat = E[L | M, Z=1], isotonic
    perfect split + one constant     M = Z * c            reproduces DATA.md §8.2's bound
    our split     + perfect magnitude M = phat(M) * L     phat = P(Z=1 | M), isotonic

Scored on the frozen-fold OOF, so the absolute level is the CV scale, not the leaderboard's.
The RATIO between the two terms is what transfers, not the numbers themselves.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from metrics import rmsle                      # noqa: E402

GBDT = ["e0049", "e0064"]
SEQ = ["e0100", "e0101", "e0101s1", "e0101s2", "e0101s3", "e0102", "e0108"]


def load(exp: str):
    t = pq.read_table(ROOT / "oof" / f"{exp}.parquet").to_pydict()
    o = np.lexsort((np.array(t["user_id"]), np.array(t["fold_id"])))
    return (np.array(t["fold_id"])[o], np.array(t["y_true"])[o],
            np.log1p(np.array(t["y_pred"])[o]))


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Rank-based AUC; O(n log n), exact with tie handling."""
    r = np.empty(score.size, float)
    order = np.argsort(score, kind="stable")
    s = score[order]
    i = 0
    while i < s.size:                                   # average ranks within ties
        j = i
        while j + 1 < s.size and s[j + 1] == s[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    n1 = float(label.sum()); n0 = float(label.size - n1)
    return float((r[label == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def binned_mean(x: np.ndarray, y: np.ndarray, xq: np.ndarray, bins: int = 200) -> np.ndarray:
    """E[y | x] by equal-count bins, evaluated at xq.

    Deliberately NOT the PAVA isotonic fit that was here first: that version repeated pooled
    blocks by observation WEIGHT rather than by bin count, which collapsed the lookup table to
    a near-constant and made two different oracles return an identical rho.  A binned
    conditional mean needs no monotonicity assumption and is checkable in one line, which the
    assert below does.
    """
    q = np.quantile(x, np.linspace(0, 1, bins + 1)[1:-1])
    b = np.searchsorted(q, x)
    cnt = np.bincount(b, minlength=bins).astype(float)
    tot = np.bincount(b, weights=y, minlength=bins)
    m = np.divide(tot, cnt, out=np.full(bins, y.mean()), where=cnt > 0)
    out = m[np.clip(np.searchsorted(q, xq), 0, bins - 1)]
    assert out.std() > 1e-6, "conditional mean is constant -- binning is broken"
    return out


def main() -> None:
    fold, y, G = load(GBDT[0])
    G = np.mean([load(e)[2] for e in GBDT], axis=0)
    S = np.mean([load(e)[2] for e in SEQ], axis=0)
    M = 0.5 * G + 0.5 * S                       # the e0120 recipe, on OOF
    L = np.log1p(y)
    Z = (y > 0).astype(float)
    n = L.size
    print(f"\n  OOF: {n:,} user-anchors over {len(np.unique(fold))} frozen folds   "
          f"P(y>0) = {Z.mean():.4f}")

    rho = np.corrcoef(L, M)[0, 1]
    sd_L, sd_M = L.std(), M.std()
    print(f"  current model (e0120 recipe): rho = {rho:.5f}   RMSLE = {rmsle(y, np.expm1(M)):.5f}")

    # ---- exact partition of Cov(L, M) ---------------------------------------------------
    p = Z.mean()
    b = Z == 1
    cov_within = p * np.cov(L[b], M[b])[0, 1]
    eL = np.array([L[~b].mean(), L[b].mean()])
    eM = np.array([M[~b].mean(), M[b].mean()])
    cov_between = (eL[1] - eL[0]) * (eM[1] - eM[0]) * p * (1 - p)
    tot = np.cov(L, M)[0, 1]
    print(f"\n  COV(L, M) = {tot:.5f}")
    print(f"    BETWEEN (separating buyers from non-buyers) {cov_between:>9.5f}   "
          f"{100 * cov_between / tot:>5.1f}%")
    print(f"    WITHIN  (ranking magnitude among buyers)    {cov_within:>9.5f}   "
          f"{100 * cov_within / tot:>5.1f}%")
    print(f"    check: sum = {cov_between + cov_within:.5f}")

    print(f"\n  our implicit classifier: AUC(M -> y>0) = {auc(M, Z):.5f}")
    print(f"  magnitude ranking among true buyers: corr(L, M | y>0) = "
          f"{np.corrcoef(L[b], M[b])[0, 1]:.5f}")

    # ---- oracle substitutions -----------------------------------------------------------
    print(f"\n  {'predictor':44s} {'rho':>8s} {'RMSLE':>9s}")
    def row(nm, pred_log):
        r = np.corrcoef(L, pred_log)[0, 1]
        print(f"  {nm:44s} {r:>8.5f} {rmsle(y, np.expm1(pred_log)):>9.5f}")

    row("our model (baseline)", M)
    ghat = binned_mean(M[b], L[b], M)          # E[L | M, buyer] -- our magnitude model
    phat = binned_mean(M, Z, M)                # P(buy | M)      -- our implicit classifier
    row("ORACLE split  x  our magnitudes", Z * ghat)
    row("ORACLE split  x  one constant", Z * L[b].mean())
    # our classification, perfect magnitude: we do not know Z, so non-buyers get the average
    # buyer magnitude -- the honest "we can size a buyer perfectly but still cannot spot one"
    row("our split     x  ORACLE magnitudes",
        phat * (Z * L + (1 - Z) * L[b].mean()))
    row("ORACLE split  x  ORACLE magnitudes (= truth)", L)

    # ---- payoff curve: rho as a function of realised classifier AUC ----------------------
    print(f"\n  PAYOFF CURVE -- interpolate our model toward the perfect-split oracle")
    print(f"  {'lambda':>7s} {'AUC':>8s} {'rho':>8s} {'RMSLE(oof)':>11s} {'d(rho)/d(AUC)':>14s}")
    prev = None
    for lam in (0.0, 0.02, 0.05, 0.10, 0.20, 0.40, 1.0):
        Mx = (1 - lam) * M + lam * (Z * ghat)
        a, r = auc(Mx, Z), np.corrcoef(L, Mx)[0, 1]
        slope = "" if prev is None else f"{(r - prev[1]) / max(a - prev[0], 1e-9):>14.2f}"
        print(f"  {lam:>7.2f} {a:>8.5f} {r:>8.5f} {rmsle(y, np.expm1(Mx)):>11.5f} {slope}")
        prev = (a, r)
    print(f"\n  On the leaderboard d(RMSLE)/d(rho) = -2.30, so multiply the last column by 2.30")
    print(f"  to read '-RMSLE per +1.00 AUC'. Compare against: everything since e0020 = 0.006 rho.")


if __name__ == "__main__":
    main()
