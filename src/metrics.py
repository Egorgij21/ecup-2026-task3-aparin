#!/usr/bin/env python
"""
FROZEN competition metric (README.md). Do not reimplement inline anywhere else.

Official definition (E-CUP 2026, task 3):

    RMSLE = sqrt( (1/n) * sum_i ( log(1 + y_i) - log(1 + yhat_bar_i) )^2 )
    yhat_bar_i = max(0, yhat_i)
    n = number of customers

Notes on exactness:
  * `log` is the natural logarithm. Confirmed empirically: a base-10 reading would put
    every observed score at 1/ln(10) = 0.434 of its actual value (the LB reads 2.12 for a
    predictor we measure at 2.2468, not 0.976).
  * Only the PREDICTION is clipped at 0, per the formula. The truth is not clipped --
    `gmv` has zero negative values in the data, so it makes no difference here, but the
    code stays literal.

Also provides the two jury tie-breaker metrics (TASK.md "prize determination"):
Gini over customer predictions, and the relative error of total predicted GMV.
"""

from __future__ import annotations

import numpy as np

__all__ = ["rmsle", "gini", "total_gmv_rel_err", "rmspe", "score_all"]


def rmsle(y_true, y_pred) -> float:
    """The competition metric. Lower is better."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.maximum(np.asarray(y_pred, dtype=np.float64), 0.0)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)))


def gini(x) -> float:
    """Gini concentration coefficient over per-customer values (tie-breaker)."""
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = x.size
    s = x.sum()
    if n == 0 or s <= 0:
        return 0.0
    return float(2.0 * np.sum(np.arange(1, n + 1) * x) / (n * s) - (n + 1) / n)


def total_gmv_rel_err(y_true, y_pred) -> float:
    """(sum predicted - sum true) / sum true. Tie-breaker: aggregate calibration."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.maximum(np.asarray(y_pred, dtype=np.float64), 0.0)
    s = y_true.sum()
    return float(np.nan) if s == 0 else float(y_pred.sum() / s - 1.0)


def rmspe(y_true, y_pred, eps: float = 1.0) -> float:
    """Root mean squared percentage error, guarded for zero truths."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.maximum(np.asarray(y_pred, dtype=np.float64), 0.0)
    return float(np.sqrt(np.mean(((y_true - y_pred) / np.maximum(y_true, eps)) ** 2)))


def score_all(y_true, y_pred) -> dict:
    """Everything we track per experiment, in one call."""
    return {
        "rmsle": rmsle(y_true, y_pred),
        "gini_pred": gini(y_pred),
        "gini_true": gini(y_true),
        "total_rel_err": total_gmv_rel_err(y_true, y_pred),
        "rmspe": rmspe(y_true, y_pred),
    }


def _self_test() -> None:
    # 1. hand-computed case
    y = np.array([0.0, 1.0, 3.0])
    p = np.array([0.0, 1.0, 3.0])
    assert abs(rmsle(y, p)) < 1e-15, rmsle(y, p)

    y = np.array([0.0, 0.0])
    p = np.array([np.e - 1.0, np.e - 1.0])          # log1p -> 1 exactly
    assert abs(rmsle(y, p) - 1.0) < 1e-12, rmsle(y, p)

    # 2. negative predictions are clipped, not squared as-is
    assert abs(rmsle(np.array([0.0]), np.array([-5.0]))) < 1e-15

    # 3. natural log, not log10
    y = np.array([0.0]); p = np.array([9.0])        # log1p(9) = ln(10) = 2.302585
    assert abs(rmsle(y, p) - np.log(10.0)) < 1e-12

    # 4. matches the explicit sum form on random data
    rng = np.random.default_rng(0)
    yt = rng.gamma(0.5, 200.0, 10_000) * (rng.random(10_000) > 0.45)
    yp = rng.gamma(0.5, 200.0, 10_000) - 20.0
    ref = np.sqrt(np.sum((np.log(1 + yt) - np.log(1 + np.maximum(yp, 0))) ** 2) / yt.size)
    assert abs(rmsle(yt, ref * 0 + yp) - ref) < 1e-12

    # 5. gini bounds
    assert abs(gini(np.ones(1000))) < 1e-9
    assert gini(np.concatenate([np.zeros(999), [1.0]])) > 0.99

    # 6. known anchor from the EDA: predicting 0 everywhere gives sqrt(E[log1p(y)^2])
    yt = rng.gamma(0.7, 100.0, 50_000) * (rng.random(50_000) > 0.46)
    assert abs(rmsle(yt, np.zeros_like(yt)) - np.sqrt(np.mean(np.log1p(yt) ** 2))) < 1e-12

    print("src/metrics.py: all self-tests passed")


if __name__ == "__main__":
    _self_test()
