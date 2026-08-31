"""Histogram-loss (HL-Gauss) regression head for LightGBM.  IDEAS.md §I1.

Discretise the target L = log1p(y) into bins, train softmax cross-entropy against a
Gaussian-smoothed target distribution over those bins, and read the prediction out as

    M(x) = sum_k p_k(x) * c_k          c_k = bin centre

so the estimand stays E[L|x] -- the functional RMSLE elicits.  That is what separates this
from ZILN / OptDist / the hurdle model, all of which estimate E[y|x] and were killed here for
exactly that reason (PAPERS_FEATURES_AND_IDEAS.md §0.1).  Only the loss geometry changes.

References: Imani & White, ICML 2018 (1806.04613); Farebrother et al., ICML 2024 (2403.03950);
Wang et al., JMLR 2026 (2402.13425) -- the last of which finds the benefit is optimisation
rather than extra information, which is the honest prior on this whole direction.

Used by `scripts/screen_loss.py` (screen) and `src/run.py` (`model: hlgauss`, confirm).
"""
from __future__ import annotations

import numpy as np


def make_bins(L_tr: np.ndarray, k: int, hi_q: float = 0.999, min_frac: float = 0.002):
    """Uniform grid on [0, vmax], then merge away under-occupied bins.

    Uniform (not quantile) edges because HL-Gauss's Gaussian smoothing is defined in target
    units: with quantile edges the same sigma would mean a different amount of smoothing in
    every bin.  vmax comes off a quantile, not the max, so the ~0.1% extreme tail does not eat
    the resolution the 44% zero atom and the L ~ 4.2 bulk actually need.

    THE MERGE IS NOT COSMETIC.  `DATA.md` §6.1 measures a near-empty region between the zero
    atom and the bulk (0.127% of users), so a uniform grid reliably produces a bin with no
    training mass.  An empty softmax class has gradient p and hessian ~p with no counterweight,
    its raw score runs away (measured: -34 after 3 rounds), and because softmax shares a
    normaliser that corrupts EVERY other class -- which is what made the port check fail at
    K=16 on real data while passing on synthetic data with no empty bin.  A degenerate class
    is a modelling defect, not something to reproduce faithfully.

    Returns (edges, centres, vmax); len(centres) may be < k after merging.
    """
    pos = L_tr[L_tr > 0]
    vmax = float(np.quantile(pos, hi_q)) if pos.size else 1.0
    edges = list(np.linspace(0.0, vmax, k + 1))
    floor = max(20, int(min_frac * L_tr.size))
    while True:
        e = np.asarray(edges)
        cnt = np.histogram(np.clip(L_tr, 0, vmax), bins=e)[0]
        if len(cnt) <= 2 or cnt.min() >= floor:
            break
        j = int(np.argmin(cnt))                      # drop the interior edge that isolates it
        edges.pop(j + 1 if j == 0 else j)
    edges = np.asarray(edges)
    return edges, 0.5 * (edges[:-1] + edges[1:]), vmax


def soft_targets(L: np.ndarray, edges: np.ndarray, sigma: float) -> np.ndarray:
    """HL-Gauss target distribution: N(L, sigma^2) integrated over each bin, truncated to
    [edges[0], edges[-1]] and renormalised.  sigma -> 0 degenerates to a one-hot vector.

    Unequal bin widths (which the merge above can produce) are handled exactly, because this
    integrates the CDF over each bin's own edges rather than assuming a common width.
    """
    from scipy.stats import norm
    Lc = np.clip(L, edges[0], edges[-1])
    nb = len(edges) - 1
    if sigma <= 1e-9:
        idx = np.clip(np.searchsorted(edges, Lc, side="right") - 1, 0, nb - 1)
        q = np.zeros((L.size, nb))
        q[np.arange(L.size), idx] = 1.0
        return q
    cdf = norm.cdf((edges[None, :] - Lc[:, None]) / sigma)
    q = np.diff(cdf, axis=1)
    s = q.sum(axis=1, keepdims=True)
    return np.where(s > 1e-12, q / np.maximum(s, 1e-12), 1.0 / nb)


def hl_objective(q: np.ndarray, k: int):
    """LightGBM custom multiclass objective: softmax cross-entropy against soft targets q.

    grad = p - q,  hess = (K/(K-1)) * p * (1 - p).

    That hessian coefficient is NOT the textbook Newton step and NOT the factor 2 that
    XGBoost's softmax uses -- it is what LightGBM's own MulticlassSoftmax works out to, and
    getting it wrong is a silent step-size error that still trains and still scores.  It was
    established by bisection against the built-in objective, not read off the docs:
    `port_exact_check` reproduces the built-in arm to max|d raw| = 0.0 for K in {3, 5, 8, 16}
    at matched (zero) init.  BACKLOG.md, 2026-08-20: "a wrong hand-port trains, scores, and
    returns a plausible null" -- so it is re-verified on every run, not once.
    """
    fac = k / (k - 1.0)

    def _obj(preds, _dataset):
        z = preds.reshape(-1, k) if preds.ndim == 1 else preds
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        g = p - q
        h = np.maximum(fac * p * (1.0 - p), 1e-6)
        return (g.reshape(preds.shape), h.reshape(preds.shape)) if preds.ndim == 1 else (g, h)
    return _obj


def softmax_rows(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def prior_init(q: np.ndarray) -> np.ndarray:
    """Per-class init score = log of the marginal target distribution.

    LightGBM's built-in multiclass boosts from the class prior; a custom objective starts at 0
    instead, and that difference alone is worth several rounds of boosting.  `predict` does NOT
    include init_score, so callers must add this back before the softmax at readout.
    """
    init = np.log(np.maximum(q.mean(axis=0), 1e-12))
    return init - init.max()


def port_exact_check(X, y_int, k, params, rounds: int = 3) -> float:
    """Gate: the hand-written objective must reproduce LightGBM's built-in `multiclass`
    EXACTLY on this data, at matched (zero) init.  Returns max|difference| in raw score;
    anything above ~1e-6 means the objective is mis-specified and every number is void.
    """
    import lightgbm as lgb
    n = X.shape[0]
    q = np.zeros((n, k))
    q[np.arange(n), y_int] = 1.0
    pr = dict(params, num_class=k, metric="None", deterministic=True, force_row_wise=True,
              feature_fraction=1.0, bagging_fraction=1.0, verbosity=-1)
    pr.pop("early_stopping_round", None)
    m1 = lgb.train(dict(pr, objective="multiclass", boost_from_average=False),
                   lgb.Dataset(X, y_int), num_boost_round=rounds)
    m2 = lgb.train(dict(pr, objective=hl_objective(q, k)),
                   lgb.Dataset(X, np.zeros(n)), num_boost_round=rounds)
    r1 = m1.predict(X, raw_score=True).reshape(-1, k)
    r2 = m2.predict(X, raw_score=True).reshape(-1, k)
    return float(np.abs(r1 - r2).max())
