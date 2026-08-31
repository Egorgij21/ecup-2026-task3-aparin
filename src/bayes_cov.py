#!/usr/bin/env python
"""
BAYES_EXP.md B1 (`hier_cov`): the §3.1 generative process with covariates on every latent.

Spec: BAYES_EXP.md.  Status header on that file records what was already killed; this builds
the one variant the review left open.

--------------------------------------------------------------------------------------------
DELIBERATE DEPARTURE FROM §4.2, AND IT IS AN UPGRADE
--------------------------------------------------------------------------------------------
§4.2 prescribes NumPyro + `AutoNormal` mean-field SVI over per-user random effects
`z_lambda,u`, `z_theta,u`, `z_m,u`, and then §4.2 has to defend against that choice: floor the
scales at 0.15, warn when the floor binds, check for posterior correlation between `mu_lam`
and `mu_theta`, sanity-fit NUTS on a subsample.  Every one of those is a patch for the same
thing -- a mean-field guide approximating latents that this model can integrate EXACTLY.

BG/NBD's Gamma prior on the rate and Beta prior on the dropout ARE the per-user random
effects, and the marginal likelihood is closed form.  So instead of putting covariates on a
random effect and approximating the integral, put them on the PRIOR and keep the integral:

    lambda_u ~ Gamma(r, alpha_u),     log alpha_u = alpha_0 + X_u . beta_alpha
    p_u      ~ Beta(a_u, b_u),        log a_u     = a_0     + X_u . beta_a
                                      log b_u     = b_0     + X_u . beta_b
    value    ~ Gamma-Gamma,           log nu_u    = nu_0    + X_u . beta_nu

The BG/NBD likelihood formula holds verbatim for per-user (r, alpha, a, b), so this is a
hierarchical covariate model whose random effects are marginalised in closed form.  It is
strictly tighter than mean-field SVI, has no ELBO, no guide, no seed-to-seed plateau, and no
variance floor to bind -- there is no free `sigma` to collapse, because the prior spread is
`1/r` and `a+b`, which are fitted.

§3.3's diagnostic still applies in translated form.  §3.3 asks that `sigma_u` SHRINK when
covariates enter.  Here the equivalent is that `r` and `a+b` should GROW: a tighter Gamma /
Beta prior is the same statement that the covariates have absorbed per-user dispersion.
Reported every run.

Also skipped, deliberately: §3.2's `s_next` seasonal extrapolation.  It is in the graveyard
twice (e0142 scored 1.6785, our worst since e0001, doing exactly this; and BACKLOG Band A+
retracted the year-lag mechanism with best multiplier k = 1.00).
"""

from __future__ import annotations

import numpy as np
import polars as pl

__all__ = ["COVARIATE_COLS", "build_covariates"]

EPS = 1e-6

COVARIATE_COLS = [
    "log_recency_act", "log_recency_ord", "log_recency_cart",
    "log_tenure", "log_active_days", "active_rate",
    "log_gmv_30", "log_gmv_90", "log_gmv_365",
    "log_ord_30", "log_ord_90", "log_ord_365",
    "log_days_30", "log_days_90",
    "log_buy_days", "log_srch_30",
    "ord_per_cart", "cart_per_srch", "ord_per_buyday", "gmv_per_buyday",
    "search_share_gmv", "search_share_ord",
    "gap_cv", "gmv_cv_90",
    "trend_gmv", "trend_days",
    "log_geo3", "is_browser",
]


def _safe_log1p(v):
    return np.log1p(np.maximum(np.asarray(v, np.float64), 0.0))


def _ratio(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return a / (b + 1.0)


def build_covariates(events: pl.DataFrame, users: np.ndarray, anchor: int
                     ) -> tuple[np.ndarray, list[str]]:
    """§3.3's covariate set at `anchor`, from days <= anchor only.

    `events` carries (user_id, di, gmv, to_ord, to_cart, searches, gmv_search, search_to_ord).
    Built entirely with sparse groupbys -- BAYES_EXP §1.1: never materialise the dense panel.

    Standardisation uses this anchor's own population.  That is NOT leakage: no covariate and
    no moment touches a day after `anchor`, and the target is never referenced.  The same
    property is why §6's train/val user split is unnecessary for the FIT -- nothing in the
    likelihood or the covariates is supervised.
    """
    past = events.filter(pl.col("di") <= anchor)
    base = pl.DataFrame({"user_id": users})

    def win(lo: int, tag: str) -> pl.DataFrame:
        w = past.filter(pl.col("di") >= anchor - lo + 1)
        return w.group_by("user_id").agg(
            pl.col("gmv").sum().alias(f"gmv_{tag}"),
            pl.col("to_ord").sum().alias(f"ord_{tag}"),
            pl.col("to_cart").sum().alias(f"cart_{tag}"),
            pl.col("searches").sum().alias(f"srch_{tag}"),
            pl.len().alias(f"days_{tag}"),
            (pl.col("gmv") ** 2).sum().alias(f"gmv2_{tag}"),
        )

    agg = past.group_by("user_id").agg(
        pl.col("di").max().alias("last_act"),
        pl.col("di").min().alias("first_act"),
        pl.len().alias("n_act"),
        pl.col("gmv").sum().alias("gmv_all"),
        pl.col("to_ord").sum().alias("ord_all"),
        pl.col("to_cart").sum().alias("cart_all"),
        pl.col("searches").sum().alias("srch_all"),
        pl.col("gmv_search").sum().alias("gmvs_all"),
        pl.col("search_to_ord").sum().alias("s2o_all"),
        (pl.col("di") * (pl.col("gmv") > 0)).max().alias("last_ord_raw"),
        (pl.col("gmv") > 0).sum().alias("n_buy"),
        (pl.col("di") * (pl.col("to_cart") > 0)).max().alias("last_cart_raw"),
        (pl.col("to_cart") > 0).sum().alias("n_cart_days"),
        # inter-active gap dispersion (§3.3 "CV of inter-active gaps")
        pl.col("di").sort().diff().mean().alias("gap_mean"),
        pl.col("di").sort().diff().std().alias("gap_std"),
    )
    df = base.join(agg, on="user_id", how="left")
    for lo, tag in ((30, "30"), (90, "90"), (365, "365")):
        df = df.join(win(lo, tag), on="user_id", how="left")
    # the three 30-day blocks behind geo3, the naive baseline every model is handed
    for k in range(3):
        b = past.filter((pl.col("di") >= anchor - 29 - 30 * k) & (pl.col("di") <= anchor - 30 * k))
        df = df.join(b.group_by("user_id").agg(pl.col("gmv").sum().alias(f"blk{k}")),
                     on="user_id", how="left")
    df = df.fill_null(0.0)
    assert df.height == users.size

    g = {c: df[c].to_numpy().astype(np.float64) for c in df.columns if c != "user_id"}
    A = float(anchor)
    n_buy = g["n_buy"]
    has_buy = n_buy > 0
    has_cart = g["n_cart_days"] > 0

    rec_act = A - g["last_act"]
    rec_ord = np.where(has_buy, A - g["last_ord_raw"], A + 1.0)
    rec_cart = np.where(has_cart, A - g["last_cart_raw"], A + 1.0)
    tenure = np.maximum(A - g["first_act"], 1.0)

    mean90 = g["gmv_90"] / 90.0
    var90 = np.maximum(g["gmv2_90"] / 90.0 - mean90 ** 2, 0.0)
    geo3 = np.expm1(np.mean([np.log1p(g[f"blk{k}"]) for k in range(3)], axis=0))

    cols = {
        "log_recency_act": np.log1p(rec_act),
        "log_recency_ord": np.log1p(rec_ord),
        "log_recency_cart": np.log1p(rec_cart),
        "log_tenure": np.log1p(tenure),
        "log_active_days": np.log1p(g["n_act"]),
        "active_rate": g["n_act"] / tenure,
        "log_gmv_30": _safe_log1p(g["gmv_30"]),
        "log_gmv_90": _safe_log1p(g["gmv_90"]),
        "log_gmv_365": _safe_log1p(g["gmv_365"]),
        "log_ord_30": _safe_log1p(g["ord_30"]),
        "log_ord_90": _safe_log1p(g["ord_90"]),
        "log_ord_365": _safe_log1p(g["ord_365"]),
        "log_days_30": _safe_log1p(g["days_30"]),
        "log_days_90": _safe_log1p(g["days_90"]),
        "log_buy_days": _safe_log1p(n_buy),
        "log_srch_30": _safe_log1p(g["srch_30"]),
        "ord_per_cart": _ratio(g["ord_all"], g["cart_all"]),
        "cart_per_srch": _ratio(g["cart_all"], g["srch_all"]),
        "ord_per_buyday": _ratio(g["ord_all"], n_buy),
        "gmv_per_buyday": _safe_log1p(_ratio(g["gmv_all"], n_buy)),
        "search_share_gmv": _ratio(g["gmvs_all"], g["gmv_all"]),
        "search_share_ord": _ratio(g["s2o_all"], g["ord_all"]),
        "gap_cv": np.where(g["gap_mean"] > 0, g["gap_std"] / (g["gap_mean"] + EPS), 0.0),
        "gmv_cv_90": np.sqrt(var90) / (mean90 + 1.0),
        "trend_gmv": _ratio(g["gmv_30"], g["gmv_90"] / 3.0),
        "trend_days": _ratio(g["days_30"], g["days_90"] / 3.0),
        "log_geo3": _safe_log1p(geo3),
        "is_browser": (~has_buy).astype(np.float64),
    }
    assert list(cols) == COVARIATE_COLS, "COVARIATE_COLS out of sync with the builder"
    X = np.column_stack([cols[c] for c in COVARIATE_COLS])
    assert np.all(np.isfinite(X)), "non-finite covariate"

    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    X = (X - mu) / sd
    # Winsorise at +/-8 sd.  Measured before adding this: `ord_per_buyday` reaches |z| = 249
    # and `gmv_per_buyday` 31.5 -- ratio features on this panel are violently heavy-tailed.
    # An unbounded z there makes `X . beta` explode for a handful of users and lets them
    # dominate the likelihood, which is also why BAYES_EXP §3.1's Normal(0, 0.5) prior on
    # beta would not have saved it: the problem is the design matrix, not the prior.
    X = np.clip(X, -8.0, 8.0)
    # `is_browser` is a cohort flag; keep it as a clean 0/1 contrast, not a standardised float
    X[:, COVARIATE_COLS.index("is_browser")] = cols["is_browser"]
    return X, list(COVARIATE_COLS)
