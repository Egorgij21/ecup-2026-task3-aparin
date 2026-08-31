#!/usr/bin/env python
"""
Feature blocks. Each block is individually switchable so that experiments can isolate
one change at a time (README.md, §8).

Everything is computed strictly from data on or before the anchor. No block may reference
a day index > anchor -- that is the one invariant this module must never break, and
`assert_no_lookahead` in run.py checks it.

Block `base` is the parent feature set for all later experiments (exp e0001).
"""

from __future__ import annotations

import hashlib
import itertools
import os
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np

WINDOWS = [7, 14, 30, 60, 90, 180, 365]
# windows for the ported structural-break vocabulary -- deliberately wide:
# the block is cheap per statistic and importance filtering can prune later
SBC_WINDOWS = [7, 14, 30, 60, 90, 180, 365]
# windows for the tsfresh-style block -- 30/90/365 only: enough length for the
# nonlinear/run/FFT statistics and to keep the ~120-feature block affordable
TSFRESH_WINDOWS = [30, 90, 365]


@lru_cache(maxsize=32)
def _normal_scores(n: int):
    """Expected normal order statistics for a sample of size n -- depend ONLY on n, so they
    are computed once per window and reused. This is what makes the normality tests
    vectorisable, which I initially and wrongly assumed was impossible."""
    from scipy.stats import norm
    m = norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))
    mc = m - m.mean()
    return mc, float((mc ** 2).sum())
BLOCK_LAGS = 3          # number of disjoint 30-day blocks


def _safe_div(a: np.ndarray, b: np.ndarray, eps: float = 1.0) -> np.ndarray:
    return a / np.maximum(b, eps)


def block_base(p, anchor: int, keep: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """
    Multi-window activity aggregates + recency + a few ratios.

    Deliberately includes `geo3` -- expm1(mean log1p of the last three 30-day GMV blocks) --
    because that IS the naive baseline we are trying to beat (DATA.md §9). A tree can
    reconstruct it only approximately, so handing it over makes the comparison honest.
    """
    cols, names = [], []

    def add(v, n):
        cols.append(np.asarray(v, np.float32)[keep]); names.append(n)

    for w in WINDOWS:
        a, b = anchor - w + 1, anchor
        add(p.wsum("gmv", a, b), f"gmv_sum_{w}")
        add(p.wsum("ord", a, b), f"ord_sum_{w}")
        add(p.wsum("cart", a, b), f"cart_sum_{w}")
        add(p.wsum("srch", a, b), f"srch_sum_{w}")
        add(p.wdays(a, b), f"days_{w}")

    for w in [30, 90, 365]:
        a, b = anchor - w + 1, anchor
        add(p.wsum("gmvs", a, b), f"gmv_search_{w}")
        add(p.wsum("gmvc", a, b), f"gmv_cat_{w}")

    blocks = []
    for k in range(BLOCK_LAGS):
        v = p.wsum("gmv", anchor - 29 - 30 * k, anchor - 30 * k)
        blocks.append(v)
        add(v, f"gmv_blk{k}")
        add(p.wsum("ord", anchor - 29 - 30 * k, anchor - 30 * k), f"ord_blk{k}")
    geo3 = np.expm1(np.mean([np.log1p(b) for b in blocks], axis=0))
    add(geo3, "geo3")
    add(np.log1p(blocks[0]) - np.log1p(blocks[1]), "gmv_trend_blk01")

    add(p.recency(anchor), "recency_days")
    add(p.recency_order(anchor), "recency_order_days")
    add(p.tenure(anchor), "tenure_days")
    add(p.wdays(0, anchor), "active_days_total")
    add(p.wsum("gmv", 0, anchor), "gmv_total")
    add(p.wsum("ord", 0, anchor), "ord_total")

    g90 = p.wsum("gmv", anchor - 89, anchor)
    o90 = p.wsum("ord", anchor - 89, anchor)
    c90 = p.wsum("cart", anchor - 89, anchor)
    s90 = p.wsum("srch", anchor - 89, anchor)
    d90 = p.wdays(anchor - 89, anchor)
    g30 = p.wsum("gmv", anchor - 29, anchor)
    add(_safe_div(g90, o90), "aov_90")
    add(_safe_div(g90, d90), "gmv_per_active_day_90")
    add(_safe_div(o90, d90), "ord_per_active_day_90")
    add(_safe_div(o90, c90), "cart_to_ord_rate_90")
    add(_safe_div(o90, s90), "srch_to_ord_rate_90")
    add(_safe_div(g30, g90 + 1.0), "gmv_ratio_30_90")
    add(_safe_div(d90, 90.0), "active_day_rate_90")

    return np.column_stack(cols), names



def block_counts(p, anchor: int, keep: np.ndarray):
    """
    Purchase FREQUENCY, separated from purchase value.

    Motivation: in e0001 `ord_sum_180` carried 4x the gain of any other feature and 4 of the
    top 6 were counts (reports/e0001.log). `base` has order counts but no notion of "how many
    distinct DAYS did they buy on" or "how concentrated are the orders" -- a user with 10
    orders on one day is not the user with 10 orders on 10 days.
    """
    cols, names = [], []

    def add(v, n):
        cols.append(np.asarray(v, np.float32)[keep]); names.append(n)

    for w in [7, 14, 30, 60, 90, 180, 365]:
        a, b = anchor - w + 1, anchor
        bd = p.wbuy(a, b)
        add(bd, f"buy_days_{w}")
        add(_safe_div(bd, p.wdays(a, b)), f"buy_day_rate_{w}")
        add(_safe_div(p.wsum("ord", a, b), bd), f"ord_per_buy_day_{w}")
        add(_safe_div(p.wsum("gmv", a, b), bd), f"gmv_per_buy_day_{w}")

    for w in [3, 21, 45, 120, 270]:
        a, b = anchor - w + 1, anchor
        add(p.wsum("ord", a, b), f"ord_sum_{w}")
        add(p.wsum("gmv", a, b), f"gmv_sum_{w}")
        add(p.wdays(a, b), f"days_{w}")

    bd_all = p.wbuy(0, anchor)
    add(bd_all, "buy_days_total")
    add(_safe_div(bd_all, p.wdays(0, anchor)), "buy_day_rate_total")
    add(_safe_div(p.wsum("ord", 0, anchor), np.maximum(p.tenure(anchor), 1.0)), "ord_per_tenure_day")
    add(_safe_div(bd_all, np.maximum(p.tenure(anchor), 1.0)), "buy_day_per_tenure_day")
    return np.column_stack(cols), names


def block_trend(p, anchor: int, keep: np.ndarray):
    """
    Direction of travel, which `base` encodes only as one block-to-block ratio.

    Motivation: the lag-block correlation decays slowly and monotonically (0.557 at lag 0 to
    0.416 a year back, DATA.md §7.1), so a user's trajectory carries information beyond their
    level. EWM half-lives 7/30/90 are causal by construction; centre-of-mass says whether the
    window's activity is front- or back-loaded without needing a second pass over the data.
    """
    cols, names = [], []

    def add(v, n):
        cols.append(np.asarray(v, np.float32)[keep]); names.append(n)

    for k, v in p.ewm.items():
        add(p.ewm_at(k, anchor), f"ewm_{k}")
    add(_safe_div(p.ewm_at("gmv_hl7", anchor), p.ewm_at("gmv_hl90", anchor), 1e-3), "ewm_gmv_ratio_7_90")
    add(_safe_div(p.ewm_at("ord_hl7", anchor), p.ewm_at("ord_hl90", anchor), 1e-3), "ewm_ord_ratio_7_90")

    for w in [30, 90, 365]:
        add(p.wcom("days", anchor - w + 1, anchor), f"com_days_{w}")
        add(p.wcom("gmv", anchor - w + 1, anchor), f"com_gmv_{w}")

    for w in [30, 90]:
        cur = p.wsum("gmv", anchor - w + 1, anchor)
        prev = p.wsum("gmv", anchor - 2 * w + 1, anchor - w)
        add(np.log1p(cur) - np.log1p(prev), f"gmv_logratio_{w}_prev{w}")
        curo = p.wsum("ord", anchor - w + 1, anchor)
        prevo = p.wsum("ord", anchor - 2 * w + 1, anchor - w)
        add(np.log1p(curo) - np.log1p(prevo), f"ord_logratio_{w}_prev{w}")
        curd = p.wdays(anchor - w + 1, anchor)
        prevd = p.wdays(anchor - 2 * w + 1, anchor - w)
        add(np.log1p(curd) - np.log1p(prevd), f"days_logratio_{w}_prev{w}")
    return np.column_stack(cols), names


def block_rank(p, anchor: int, keep: np.ndarray):
    """
    Cross-sectional percentile within the anchor's own population.

    Motivation: the fold level drifts hard -- E[log1p(y)] runs 2.13 -> 2.44 across anchors
    (DATA.md §6.2) and the test anchor sits outside the folds' calendar range entirely. Raw
    levels therefore mean different things in different folds; a percentile does not. These
    use features only, never the target, so they are leak-free.
    """
    cols, names = [], []
    n = int(keep.sum())

    def pct(v):
        v = np.asarray(v)[keep]
        return (np.argsort(np.argsort(v)).astype(np.float32) + 0.5) / max(n, 1)

    for nm, v in [
        ("gmv_30", p.wsum("gmv", anchor - 29, anchor)),
        ("gmv_90", p.wsum("gmv", anchor - 89, anchor)),
        ("gmv_365", p.wsum("gmv", anchor - 364, anchor)),
        ("ord_90", p.wsum("ord", anchor - 89, anchor)),
        ("ord_180", p.wsum("ord", anchor - 179, anchor)),
        ("days_90", p.wdays(anchor - 89, anchor)),
        ("recency", -p.recency(anchor)),
        ("recency_ord", -p.recency_order(anchor)),
    ]:
        cols.append(pct(v)); names.append(f"pct_{nm}")
    return np.column_stack(cols), names


def block_funnel(p, anchor: int, keep: np.ndarray):
    """
    The search/catalogue funnel -- ten raw columns the pipeline had never ingested.

    Why this is new information rather than a rearrangement of what we had: `gmv_search` and
    `gmv_cat` give GMV split by channel, but the strongest raw predictor in this dataset is
    ORDER COUNT (Spearman 0.5731) rather than GMV (0.5549), and order counts were only
    available in total. `search_to_ord` alone scores 0.5668 -- level with the total.

    Three families, in increasing order of how much they add:
      * window sums of each funnel event -- mostly parallel to what `base` already has;
      * day-counts from the has_* binaries -- "on how many days did a search convert", which
        is a frequency, not a volume, and is not derivable from the sums;
      * RATIOS -- attribution mix and conversion rates. These are the point. A user who
        orders through search and one who orders through the catalogue can be identical on
        every existing feature, and a user who searches heavily but rarely converts is
        behaviourally distinct from one who converts on every visit. Ratios are also
        level-invariant, which matters because the test cut-off sits outside the range of
        training levels (anchor_drift.py).
    """
    cols, names = [], []

    def add(v, n):
        cols.append(np.asarray(v, np.float32)[keep]); names.append(n)

    for w in WINDOWS:
        a, b = anchor - w + 1, anchor
        sev, cev = p.wsum("sev", a, b), p.wsum("cev", a, b)
        s2c, s2o = p.wsum("s2c", a, b), p.wsum("s2o", a, b)
        c2c, c2o = p.wsum("c2c", a, b), p.wsum("c2o", a, b)
        ordr, cart = p.wsum("ord", a, b), p.wsum("cart", a, b)
        for v, n in ((sev, "sev"), (cev, "cev"), (s2c, "s2c"), (s2o, "s2o"),
                     (c2c, "c2c"), (c2o, "c2o")):
            add(v, f"fn_{n}_{w}")
        for n in ("hs2c", "hs2o", "hc2c", "hc2o"):
            add(p.wsum(n, a, b), f"fn_days_{n}_{w}")

        # --- attribution mix: which channel drives this user's orders / carts
        add(_safe_div(s2o, s2o + c2o), f"fn_ordshare_search_{w}")
        add(_safe_div(s2c, s2c + c2c), f"fn_cartshare_search_{w}")
        add(_safe_div(s2o + c2o, ordr), f"fn_attributed_ord_rate_{w}")
        # --- conversion: how efficiently does browsing turn into buying
        add(_safe_div(s2o, sev), f"fn_conv_search_ord_{w}")
        add(_safe_div(c2o, cev), f"fn_conv_cat_ord_{w}")
        add(_safe_div(s2c, sev), f"fn_conv_search_cart_{w}")
        add(_safe_div(c2c, cev), f"fn_conv_cat_cart_{w}")
        # --- cart -> order, per channel: the last and most decisive funnel step
        add(_safe_div(s2o, s2c), f"fn_cart2ord_search_{w}")
        add(_safe_div(c2o, c2c), f"fn_cart2ord_cat_{w}")
        add(_safe_div(s2o + c2o, s2c + c2c), f"fn_cart2ord_all_{w}")
        # --- intensity per active day
        d = p.wdays(a, b)
        add(_safe_div(sev, d), f"fn_sev_per_day_{w}")
        add(_safe_div(cev, d), f"fn_cev_per_day_{w}")
    return np.column_stack(cols), names


def block_visit(p, anchor: int, keep: np.ndarray):
    """
    Split active days into "empty visits" and real search/catalog days.

    Motivation: 14.85 % of rows carry zero search, catalog, cart and order activity
    (DATA.md §3). `base` folds them into `days_W`, yet they correlate with the target at only
    0.147 versus 0.463 for non-empty days -- blending the two dilutes both.
    """
    cols, names = [], []

    def add(v, n):
        cols.append(np.asarray(v, np.float32)[keep]); names.append(n)

    for w in [7, 30, 90, 365]:
        a, b = anchor - w + 1, anchor
        e, d = p.wempty(a, b), p.wdays(a, b)
        add(e, f"empty_days_{w}")
        add(d - e, f"real_days_{w}")
        add(_safe_div(e, d), f"empty_day_rate_{w}")
    add(p.wempty(0, anchor), "empty_days_total")
    add(_safe_div(p.wempty(0, anchor), p.wdays(0, anchor)), "empty_day_rate_total")
    return np.column_stack(cols), names


def block_channel(p, anchor: int, keep: np.ndarray):
    """
    Search versus catalog, at more windows than `base` carries.

    Motivation: catalog is only 7.35 % of GMV but is nearly uncorrelated with search at row
    level (0.054, DATA.md §2.3). Decorrelated signal is worth more than its volume suggests.
    """
    cols, names = [], []

    def add(v, n):
        cols.append(np.asarray(v, np.float32)[keep]); names.append(n)

    for w in [7, 14, 60, 180]:
        a, b = anchor - w + 1, anchor
        add(p.wsum("gmvs", a, b), f"gmv_search_{w}")
        add(p.wsum("gmvc", a, b), f"gmv_cat_{w}")
    for w in [30, 90, 365]:
        a, b = anchor - w + 1, anchor
        gs, gc = p.wsum("gmvs", a, b), p.wsum("gmvc", a, b)
        add(_safe_div(gc, gs + gc), f"cat_gmv_share_{w}")
        add(_safe_div(p.wsum("cart", a, b), p.wsum("srch", a, b)), f"cart_per_search_{w}")
    return np.column_stack(cols), names



# --- decomposition of the killed `trend` bundle (e0003) into its three families -----------
# e0003 lumped EWM + centre-of-mass + log-diffs together and scored +0.00041 (kill). A
# negative bundle can hide a positive part, so each family is now tested on its own.

def block_diff(p, anchor: int, keep: np.ndarray):
    """
    Pure lag/difference structure: 30-day windows at lags 0..6, their first differences in
    log space (growth), and second differences (acceleration). No EWM, no centre-of-mass.
    """
    cols, names = [], []

    def add(v, n):
        cols.append(np.asarray(v, np.float32)[keep]); names.append(n)

    L = 7
    g = [p.wsum("gmv", anchor - 29 - 30 * k, anchor - 30 * k) for k in range(L)]
    o = [p.wsum("ord", anchor - 29 - 30 * k, anchor - 30 * k) for k in range(L)]
    d = [p.wdays(anchor - 29 - 30 * k, anchor - 30 * k) for k in range(L)]

    for k in range(3, L):                       # blk0..2 already exist in `base`
        add(g[k], f"gmv_blk{k}"); add(o[k], f"ord_blk{k}")
    for k in range(L):
        add(d[k], f"days_blk{k}")

    lg = [np.log1p(x) for x in g]; lo = [np.log1p(x) for x in o]; ld = [np.log1p(x) for x in d]
    d1g, d1o, d1d = [], [], []
    for k in range(L - 1):
        d1g.append(lg[k] - lg[k + 1]); d1o.append(lo[k] - lo[k + 1]); d1d.append(ld[k] - ld[k + 1])
        add(d1g[k], f"d1_gmv_{k}"); add(d1o[k], f"d1_ord_{k}"); add(d1d[k], f"d1_days_{k}")
    for k in range(L - 2):                      # acceleration
        add(d1g[k] - d1g[k + 1], f"d2_gmv_{k}")
        add(d1o[k] - d1o[k + 1], f"d2_ord_{k}")

    for w in [60, 90, 180]:                     # same-length windows, one window apart
        cur = p.wsum("gmv", anchor - w + 1, anchor)
        prv = p.wsum("gmv", anchor - 2 * w + 1, anchor - w)
        add(np.log1p(cur) - np.log1p(prv), f"d1_gmv_w{w}")
        curo = p.wsum("ord", anchor - w + 1, anchor)
        prvo = p.wsum("ord", anchor - 2 * w + 1, anchor - w)
        add(np.log1p(curo) - np.log1p(prvo), f"d1_ord_w{w}")
    add(np.std([lg[k] for k in range(L)], axis=0), "blk_gmv_logstd")
    add(np.std([lo[k] for k in range(L)], axis=0), "blk_ord_logstd")
    return np.column_stack(cols), names


def block_cumshare(p, anchor: int, keep: np.ndarray):
    """
    Cumulative-sum derived NORMALISATIONS -- what fraction of a user's whole history sits in
    the recent window, and how fast they accumulated it. `base` has raw expanding totals but
    never divides by them, so "spent 500 recently" cannot be told from "spent 500 recently
    out of 600 ever" versus "out of 50 000 ever".
    """
    cols, names = [], []

    def add(v, n):
        cols.append(np.asarray(v, np.float32)[keep]); names.append(n)

    gt = p.wsum("gmv", 0, anchor); ot = p.wsum("ord", 0, anchor)
    dt = p.wdays(0, anchor); ten = np.maximum(p.tenure(anchor), 1.0)
    for w in [7, 30, 90, 180]:
        a = anchor - w + 1
        add(_safe_div(p.wsum("gmv", a, anchor), gt), f"gmv_share_life_{w}")
        add(_safe_div(p.wsum("ord", a, anchor), ot), f"ord_share_life_{w}")
        add(_safe_div(p.wdays(a, anchor), dt), f"days_share_life_{w}")
    add(_safe_div(gt, ten), "gmv_per_tenure_day")
    add(_safe_div(gt, dt), "gmv_per_active_day_life")
    add(_safe_div(ot, dt), "ord_per_active_day_life")
    # recent activity relative to the user's own expanding average = self-normalised level
    for w in [30, 90]:
        exp_rate = _safe_div(gt, ten) * w
        add(np.log1p(p.wsum("gmv", anchor - w + 1, anchor)) - np.log1p(exp_rate),
            f"gmv_{w}_vs_expanding")

    # how long did it take to accumulate half / 90 % of lifetime GMV (from the cumsum row)
    cs = p.cs["gmv"][:, :anchor + 2]
    tot = cs[:, -1]
    for frac, nm in [(0.5, "half"), (0.9, "p90")]:
        hit = np.argmax(cs >= (frac * tot)[:, None], axis=1)
        add(np.where(tot > 0, anchor + 1 - hit, -1.0), f"days_since_{nm}_of_life_gmv")
    return np.column_stack(cols), names


def block_ewm(p, anchor: int, keep: np.ndarray):
    """EWM family alone, isolated from the killed e0003 bundle."""
    cols, names = [], []
    for k, v in p.ewm.items():
        cols.append(np.asarray(p.ewm_at(k, anchor), np.float32)[keep]); names.append(f"ewm_{k}")
    for nm, a, b in [("gmv", "gmv_hl7", "gmv_hl90"), ("ord", "ord_hl7", "ord_hl90")]:
        r = _safe_div(p.ewm_at(a, anchor), p.ewm_at(b, anchor), 1e-3)
        cols.append(np.asarray(r, np.float32)[keep]); names.append(f"ewm_{nm}_ratio_7_90")
    return np.column_stack(cols), names


def block_com(p, anchor: int, keep: np.ndarray):
    """Centre-of-mass family alone, isolated from the killed e0003 bundle."""
    cols, names = [], []
    for w in [30, 90, 365]:
        for which in ["days", "gmv"]:
            v = p.wcom(which, anchor - w + 1, anchor)
            cols.append(np.asarray(v, np.float32)[keep]); names.append(f"com_{which}_{w}")
    return np.column_stack(cols), names



def block_dispersion(p, anchor: int, keep: np.ndarray):
    """
    BURSTINESS / REGULARITY -- a variance statistic family, not another level.

    Motivation (PAPERS_FEATURES_AND_IDEAS.md B1/B2): every feature in the graveyard is a
    first moment -- a sum, a mean, a ratio of sums. Two users with identical 90-day order
    counts can be a steady weekly buyer and a one-day burst; RMSLE cares because the steady
    one repeats next month and the burst one may not. Point-process theory calls this
    over/under-dispersion; the Fano factor (var/mean of daily counts) and the coefficient of
    variation of event timing measure it, and neither is a monotone function of any existing
    feature. ASOS independently ranks "spread of order dates" highly, though note their
    *mean* order date is our killed `com` block -- so the dispersion, not the location.

    All exact and O(1) per window via the squared-value prefix sums in data.py.
    """
    cols, names = [], []

    def add(v, n):
        cols.append(np.asarray(v, np.float32)[keep]); names.append(n)

    for w in [30, 90, 365]:
        a, b = anchor - w + 1, anchor
        wl = float(min(w, anchor + 1))

        # dispersion of daily counts over the whole window, zero days included
        for col in ["ord", "gmv"]:
            s1, s2 = p.wsum(col, a, b), p.wsumsq(col, a, b)
            mean = s1 / wl
            var = np.maximum(s2 / wl - mean ** 2, 0.0)
            add(np.where(mean > 0, var / np.maximum(mean, 1e-9), -1.0), f"fano_{col}_{w}")
            add(np.where(mean > 0, np.sqrt(var) / np.maximum(mean, 1e-9), -1.0), f"cv_{col}_{w}")

        # dispersion of WHEN activity and purchases fell inside the window
        for which, nm in [("days", "act"), ("buy", "buy")]:
            sd = p.wdate_std(which, a, b)
            add(sd, f"date_sd_{nm}_{w}")
            # uniform spread over the window is w/sqrt(12); <1 = clustered, ~1 = regular
            add(np.where(sd >= 0, sd / (wl / np.sqrt(12.0)), -1.0), f"date_regularity_{nm}_{w}")

        # mean inter-event gap and Goh-Barabasi burstiness B = (sd - mu)/(sd + mu)
        n_act, n_buy = p.wdays(a, b), p.wbuy(a, b)
        for n, nm in [(n_act, "act"), (n_buy, "buy")]:
            mu = wl / np.maximum(n, 1.0)
            sd = p.wdate_std("days" if nm == "act" else "buy", a, b)
            add(np.where(n >= 2, mu, -1.0), f"mean_gap_{nm}_{w}")
            add(np.where((n >= 2) & (sd >= 0), (sd - mu) / np.maximum(sd + mu, 1e-9), -2.0),
                f"burstiness_{nm}_{w}")

    # concentration: what share of window GMV landed on the single biggest day (Herfindahl-ish)
    for w in [30, 90]:
        a, b = anchor - w + 1, anchor
        s1, s2 = p.wsum("gmv", a, b), p.wsumsq("gmv", a, b)
        add(np.where(s1 > 0, s2 / np.maximum(s1 ** 2, 1e-9), -1.0), f"gmv_hhi_{w}")
        o1, o2 = p.wsum("ord", a, b), p.wsumsq("ord", a, b)
        add(np.where(o1 > 0, o2 / np.maximum(o1 ** 2, 1e-9), -1.0), f"ord_hhi_{w}")
    return np.column_stack(cols), names



def block_tsfeat(p, anchor: int, keep: np.ndarray):
    """
    Time-series SHAPE statistics, ported from a structural-break competition's feature set.

    Everything we have so far is a level, a ratio, or a dispersion of counts. None of it
    describes the SHAPE of a user's daily series: is their buying rhythmic or erratic, is the
    trend up or down within the window, how heavy is the tail of their daily values. The
    frequency-domain pair is the most clearly new -- we model day-of-week globally but have
    nothing per-user about weekly cadence.

    Computed on the raw daily matrices (not prefix sums, which cannot express these), on the
    windows where the cost is affordable.
    """
    cols, names = [], []

    def add(v, n):
        cols.append(np.asarray(v, np.float32)[keep]); names.append(n)

    for col in ["gmv", "ord"]:
        for w in [30, 90]:
            a = max(anchor - w + 1, 0)
            M = p.raw[col][:, a:anchor + 1].astype(np.float32)
            n = M.shape[1]
            mu = M.mean(1)
            sd = M.std(1)
            sdz = np.maximum(sd, 1e-9)
            z = (M - mu[:, None]) / sdz[:, None]
            add(np.mean(z ** 3, 1), f"skew_{col}_{w}")
            add(np.mean(z ** 4, 1) - 3.0, f"kurt_{col}_{w}")

            # trend: OLS slope of the daily series inside the window
            t_ = np.arange(n, dtype=np.float32) - (n - 1) / 2.0
            add((M @ t_) / max(float((t_ ** 2).sum()), 1e-9), f"trendslope_{col}_{w}")

            # autocorrelation at lag 1 and 7 -- lag 7 is weekly rhythm
            for lag in (1, 7):
                if n > lag + 2:
                    A_, B_ = M[:, :-lag], M[:, lag:]
                    ca = A_ - A_.mean(1, keepdims=True)
                    cb = B_ - B_.mean(1, keepdims=True)
                    num = (ca * cb).sum(1)
                    den = np.sqrt((ca ** 2).sum(1) * (cb ** 2).sum(1))
                    add(np.where(den > 1e-9, num / np.maximum(den, 1e-9), 0.0),
                        f"autocorr{lag}_{col}_{w}")

            # spectral: how much power sits at the 7-day period, and spectral entropy
            F = np.abs(np.fft.rfft(M - mu[:, None], axis=1)) ** 2
            tot = np.maximum(F[:, 1:].sum(1), 1e-9)
            k7 = int(round(n / 7.0))
            if 1 <= k7 < F.shape[1]:
                add(F[:, k7] / tot, f"weeklypower_{col}_{w}")
            P = F[:, 1:] / tot[:, None]
            add(-(P * np.log(np.maximum(P, 1e-12))).sum(1) / np.log(P.shape[1]),
                f"specentropy_{col}_{w}")

            # robust location/scale, and tail heaviness of the daily values
            add(np.mean(np.abs(M - mu[:, None]), 1), f"mad_{col}_{w}")
            add(np.max(M, 1) / np.maximum(M.sum(1), 1e-9), f"maxshare_{col}_{w}")
            nz = (M > 0).sum(1).astype(np.float32)
            add(nz / n, f"dutycycle_{col}_{w}")

            # run structure: longest zero run and number of on/off transitions
            on = (M > 0)
            add((on[:, 1:] != on[:, :-1]).sum(1).astype(np.float32), f"switches_{col}_{w}")
            run = np.zeros(M.shape[0], np.float32); mx = np.zeros(M.shape[0], np.float32)
            for j in range(n):
                run = np.where(on[:, j], 0.0, run + 1.0)
                mx = np.maximum(mx, run)
            add(mx, f"maxzerorun_{col}_{w}")
            del M, F, P, z

    # distributional SHIFT: how far the recent 30 days sit from the preceding 60,
    # compared on sorted daily values (a discrete Wasserstein-style contrast)
    for col in ["gmv"]:
        A_ = np.sort(p.raw[col][:, max(anchor - 29, 0):anchor + 1], axis=1)
        B_ = np.sort(p.raw[col][:, max(anchor - 89, 0):anchor - 29], axis=1)
        q = np.linspace(0, 1, 11)
        qa = np.quantile(A_, q, axis=1).T
        qb = np.quantile(B_, q, axis=1).T
        add(np.abs(np.log1p(qa) - np.log1p(qb)).mean(1), f"distshift_{col}_30v60")
        add(np.log1p(qa[:, -2]) - np.log1p(qb[:, -2]), f"distshift_q90_{col}_30v60")
        del A_, B_
    return np.column_stack(cols), names


# ================================================================================= tsfresh
#
# *** MEASURED AND KILLED 2026-08-20 -- do not add this block to a config. ***
#   e0191 (confirm, frozen folds, 250k users, e0049 + this block = 725 features):
#     cv 1.76545 +/- 0.02173 vs parent e0049 1.76551 -> delta -0.00006 = 0.6 sigma, wins 2/5.
#     `no effect` by README.md; 60 columns for 0.00006 is the within-noise accumulation
#     that rule forbids.  e0192 (screen) agrees: bundle nil at both anchors, and NO statistic
#     family positive at both when decomposed into its 11 families.
#   Do not read the per-fold deltas as a fold-index trend: they are monotone (+0.00027 ..
#   -0.00041, spearman -0.90 vs train rows) but sigma_noise=0.00009 is the sd of the 5-FOLD
#   MEAN -- a single fold's sd is ~3x that, so no fold reaches 2 sigma.  See EXPERIMENTS.md 1o.
#   Kept per rule 10 (never delete a config/experiment) and because the validation harness
#   below is reusable for any future library port.  BLOCKS registration is retained so the
#   experiment is reproducible, NOT because the block is live.
#
# A hand-vectorised selection of the tsfresh library's nonlinear time-series statistics
# (github.com/blue-yonder/tsfresh), after auditing which of its ~800 extractors are (a) not
# already present in `tsfeat`/`sbc`/`fcast` and (b) computable as matrix ops over the
# (n_users, window) daily slice -- the library itself is row-by-row Python + pandas and would
# take hours on 250k users x 29 anchors, so running it directly was never an option.
#
# The genuinely NEW information classes over the installed blocks:
#   * NONLINEAR trend structure: c3 (lag-nonlinearity, tsfresh `c3`), `cwt`-style coarse
#     wavelet energy in three octaves (installed spectral stats are all plain FFT),
#     `lempelziv` (unbounded complexity -- the installed histentropy is window-fixed at 10
#     bins and saturates for long series).
#   * LINEARITY / time-reversibility: `linear_trend_timewise` (OLS t-stat of the residual
#     series, the "is this series actually moving" statistic), `time_reversal_asymmetry_statistic`.
#   * FINE-GRAINED run structure: `longest_strike_above_mean` / `longest_strike_below_mean`
#     (installed maxzerorun counts zeros only; a strike is level-relative, so it separates
#     "consistently high" from "consistent-zero").
#   * ARCH/volatility clustering: `agg_autocorrelation` on the squared series (period 7 --
#     does weekly activity bunch? volatility-of-volatility), `partial_autocorrelation` lag 1.
#
# Names carry the `tf_` prefix so they are unambiguous in importance plots and the lookback
# parser (lookback_of: `_w<digits>$` -> window days; the remaining names carry a 30/90/365
# suffix with the window in the tag). Everything is causal: slices end at `anchor`.
# All constant/all-zero columns are dropped by the feature audit downstream (they exist:
# longstrike_above is 0 for the ~15% fully-dormant users at short windows).
#
# VALIDATED against tsfresh 0.21.2 on zero-heavy synthetic series (max relative error):
#   c3 0.0 | time_reversal 3e-08 | longest_strike_above/below 0.0 | lempel_ziv 6e-08 |
#   autocorrelation (pacf1) 4e-08 | autocorrelation(x^2, 7) (arch7) 6e-08 |
#   linear_trend slope/stderr (trendt) 2e-15.
# Three of these were WRONG on the first pass and only the library check caught it: c3 had a
# bispectrum-style formula (rel err 1e15), arch7 used a segment-wise Pearson instead of
# tsfresh's global-mean/(n-lag)*var convention (0.41), and trendt omitted the intercept from
# the residual, deflating the t-stat by ~12%. A "tsfresh-style" feature that is not the
# tsfresh statistic is an untested new feature wearing a validated name.


def _tf_c3(M: np.ndarray, lag: int = 1) -> np.ndarray:
    """c3 = mean(x[t+2lag] * x[t+lag] * x[t]) -- Schreiber & Schmitz nonlinearity (tsfresh).

    Verified equal to `tsfresh.feature_extraction.feature_calculators.c3` to 1e-6 relative.
    """
    return np.mean(M[:, 2 * lag:] * M[:, lag:M.shape[1] - lag] * M[:, : -2 * lag],
                   1).astype(np.float32)


def _tf_acf(M: np.ndarray, lag: int) -> np.ndarray:
    """Autocorrelation at `lag`, tsfresh `autocorrelation` convention exactly.

    GLOBAL mean and variance with an `(n - lag) * var` denominator -- NOT a segment-wise
    Pearson, which differs from the library by ~0.4 relative.  tsfresh returns NaN on a
    constant series; a feature matrix cannot carry NaN, so a flat user gets 0.0 -- the only
    intentional deviation.
    """
    mu = M.mean(1, keepdims=True)
    v = M.var(1)
    n = M.shape[1]
    num = ((M[:, : n - lag] - mu) * (M[:, lag:] - mu)).sum(1)
    return np.where(v > 1e-12, num / np.maximum((n - lag) * v, 1e-12), 0.0).astype(np.float32)


def _tf_arch(M: np.ndarray, lag: int = 7) -> np.ndarray:
    """Autocorrelation of the SQUARED series: volatility clustering (the ARCH effect)."""
    return _tf_acf(M ** 2, lag)


def _tf_time_rev(M: np.ndarray, lag: int = 1) -> np.ndarray:
    """time_reversal_asymmetry_statistic: mean(x[t+2lag]^2 x[t+lag] - x[t+lag] x[t]^2)."""
    return np.mean(M[:, 2 * lag:] ** 2 * M[:, lag:-lag]
                   - M[:, lag:-lag] * M[:, :-2 * lag] ** 2, 1).astype(np.float32)


def _tf_lempelziv(M: np.ndarray, bins: int = 10) -> np.ndarray:
    """Lempel-Ziv complexity: distinct sub-words needed to encode the binned series, / n.

    Matches `tsfresh.feature_extraction.feature_calculators.lempel_ziv_complexity` exactly
    (verified to 0 abs err): bin into `bins` levels with `searchsorted(..., side='left')` on
    `linspace(min, max, bins+1)[1:]`, then the standard prefix scan.  tsfresh's default binary
    binning collapses our zero-heavy daily values, so 10 levels is used -- the value tsfresh
    itself exposes as a parameter, not a deviation from it.
    """
    n = M.shape[1]
    out = np.zeros(M.shape[0], np.float32)
    for u in range(M.shape[0]):
        x = M[u]
        edges = np.linspace(x.min(), x.max(), bins + 1)[1:]
        seq = np.searchsorted(edges, x, side="left")
        subs, ind, inc = set(), 0, 1
        while ind + inc <= n:
            s = seq[ind:ind + inc].tobytes()
            if s in subs:
                inc += 1
            else:
                subs.add(s)
                ind += inc
                inc = 1
        out[u] = len(subs) / n
    return out


def block_tsfresh(p, anchor: int, keep: np.ndarray):
    """
    Hand-vectorised tsfresh-style nonlinear / run / wavelet statistics (see module docstring).
    """
    cols, names = [], []

    def add(v, n):
        cols.append(np.asarray(v, np.float32)[keep]); names.append(n)

    for col in ["gmv", "ord"]:
        for w in TSFRESH_WINDOWS:
            a = max(anchor - w + 1, 0)
            M = p.raw[col][:, a:anchor + 1].astype(np.float32)
            n = M.shape[1]
            if n < 30:                       # the run/wavelet stats need real length
                continue
            tag = f"{col}_{w}"

            # ---- nonlinear trend structure ----------------------------------------
            add(_tf_c3(M), f"tf_c3_{tag}")
            add(_tf_time_rev(M), f"tf_timerev_{tag}")

            # ---- linearity: OLS t-stat of the slope (tsfresh linear_trend slope/stderr) ---
            # the residual must be taken around the FITTED LINE (intercept included), not
            # around b*t alone -- omitting the intercept leaves the series mean in the SSE
            # and deflates the t-stat by ~10%.
            t_ = np.arange(n, dtype=np.float32) - (n - 1) / 2.0
            tt = float((t_ ** 2).sum())
            b = (M @ t_) / max(tt, 1e-9)
            r = M - M.mean(1, keepdims=True) - b[:, None] * t_[None, :]
            se_b = np.sqrt(np.maximum((r ** 2).sum(1) / max(n - 2, 1) / max(tt, 1e-9), 1e-18))
            add(b / np.maximum(se_b, 1e-9), f"tf_trendt_{tag}")
            del r, se_b

            # ---- ARCH: squared-series autocorrelation, volatility clustering ------
            add(_tf_arch(M, 7), f"tf_arch7_{tag}")

            # ---- runs above/below the LEVEL (not zeros) ---------------------------
            mu = M.mean(1, keepdims=True)
            on = M > mu
            for rn, flag in (("above", on), ("below", ~on)):
                run = np.zeros(M.shape[0], np.float32); mx = np.zeros(M.shape[0], np.float32)
                for j in range(n):
                    run = np.where(flag[:, j], run + 1.0, 0.0)
                    mx = np.maximum(mx, run)
                add(mx, f"tf_longstrike_{rn}_{tag}")
                del run, mx

            # ---- wavelet: coarse octaves, Haar, causal (left-aligned) -------------
            for oct_ in (1, 2, 3):
                k = 2 ** oct_
                if n < k:
                    continue
                out = np.zeros(M.shape[0], np.float32)
                for o in range(0, n - k + 1, k):       # non-overlapping dyadic blocks
                    seg = M[:, o:o + k]
                    avg = seg[:, : k // 2].mean(1)
                    dif = seg[:, : k // 2] - seg[:, k // 2:]
                    out += np.sqrt((dif ** 2).sum(1))
                add(out / np.maximum(n // k, 1), f"tf_wavenergy{oct_}_{tag}")
                del out

            # ---- lag-1 autocorrelation (== partial autocorrelation at lag 1) -------
            add(_tf_acf(M, 1), f"tf_pacf1_{tag}")

            # ---- complexity --------------------------------------------------------
            # `tf_lz` (Lempel-Ziv) is DELIBERATELY NOT EMITTED.  `_tf_lempelziv` is kept and
            # validated because it is the honest implementation, but it costs ~32 s per 250k
            # users per window against ~0.3 s for every other statistic here (a per-user
            # Python prefix scan; it does not vectorise and ignores the 16 cores), i.e. ~2 h
            # of the fold build for one 6-column family.  That family was also the WORST in
            # the local screen (-0.00218 at A2, +0.00002 at A1).  Paying 100x the compute of
            # the whole rest of the block for its least informative part is not a trade worth
            # making; re-enable only if a cheap vectorised LZ is written.
            del M
    return np.column_stack(cols), names



SBC_FAMILIES = {
    "sbcshape":  {"trendchangerate", "numtrendchanges", "zigzagfreq", "fracup", "fracdown",
                  "maxrunlength", "meanrunlength", "zerocross", "meanlast"},
    "sbcmoment": {"mean", "std", "skew", "kurtosis", "mad", "cv", "energy", "rms", "fft",
                  "jarquebera", "burstiness", "dutycycle"},
    "sbcorder":  {"q10", "q25", "median", "q75", "q90", "q99", "q19", "iqr", "interdecile",
                  "max", "range", "trimmedmean", "trimmedstd", "winsorizedmean",
                  "biweightmidvar", "gini"},
    "sbcnorm":   {"shapirofrancia", "andersondarling"},
    "sbcspec":   {"spectralentropy", "spectralflatness", "fftmaxfreq", "domfreqpower",
                  "spectralrolloff"},
    "sbcts":     {"trendslope", "firsthalfmean", "secondhalfmean", "hurstexp",
                  "autocorr1", "autocorr5", "autocorr7"},
    "sbcent":    {"histentropy"},
}
SBC_ALL = set().union(*SBC_FAMILIES.values())


def block_sbc(p, anchor: int, keep: np.ndarray, want: set | None = None):
    """
    The FULL 52-statistic vocabulary from the ADIA structural-break notebook, ported.

    Every statistic is re-expressed as a matrix operation over an (n_users x W) slice of the
    raw daily series -- the original applies them row-by-row via scipy, which is impossible
    at 250k users x 29 anchors. Each (column, window) sorts ONCE and derives every order
    statistic from the sorted array.

    Deviations from the original, all forced by our data and flagged rather than hidden:
      * `zerocross` counts sign changes; our series are non-negative so it is identically 0.
        Computed on the MEAN-CENTRED series instead, which is what it measures on a
        zero-mean series anyway.
      * `shapirow` and `andersonnm` need a per-row scipy call with no vectorised form.
        Dropped. `jarquebera` covers the same normality question in closed form.
      * `permentropy` in the original is really a 10-bin histogram entropy (same as `hss`),
        so it is computed once, not twice.
      * `fft` (sum of squared spectrum) equals n * sum(x^2) by Parseval, i.e. `energy`
        rescaled -- kept for fidelity, but it carries no independent information.
    """
    want = SBC_ALL if want is None else want
    cols, names = [], []

    def add(v, n):
        stat = n[4:].rsplit("_", 2)[0]          # sbc_<stat>_<col>_<window>
        if stat in want:
            cols.append(np.asarray(v, np.float32)[keep]); names.append(n)

    need_sort = bool(want & (SBC_FAMILIES["sbcorder"] | SBC_FAMILIES["sbcnorm"]))
    need_fft = bool(want & SBC_FAMILIES["sbcspec"])
    need_diff = bool(want & SBC_FAMILIES["sbcshape"])

    for col in ["gmv", "ord"]:
        for w in SBC_WINDOWS:
            a = max(anchor - w + 1, 0)
            M = p.raw[col][:, a:anchor + 1].astype(np.float32)
            n = M.shape[1]
            if n < 8:                      # the run/lag statistics need a few points
                continue
            tag = f"{col}_{w}"
            mu = M.mean(1); sd = M.std(1)
            sdz = np.maximum(sd, 1e-9)

            # ---- shape of the trajectory -------------------------------------------
            dM = np.diff(M, axis=1) if (need_diff or True) else None
            sg = np.sign(dM)
            chg = (np.diff(sg, axis=1) != 0)
            add(chg.sum(1) / (n - 2 + 1e-9), f"sbc_trendchangerate_{tag}")
            add(chg.sum(1).astype(np.float32), f"sbc_numtrendchanges_{tag}")
            add(chg.mean(1), f"sbc_zigzagfreq_{tag}")
            add((dM > 0).mean(1), f"sbc_fracup_{tag}")
            add((dM < 0).mean(1), f"sbc_fracdown_{tag}")
            run = np.zeros(M.shape[0], np.float32); mx = np.zeros(M.shape[0], np.float32)
            tot = np.zeros(M.shape[0], np.float32); cnt = np.zeros(M.shape[0], np.float32)
            for j in range(chg.shape[1]):
                run += 1.0
                c = chg[:, j]
                mx = np.maximum(mx, np.where(c, run, 0.0))
                tot += np.where(c, run, 0.0); cnt += c
                run = np.where(c, 0.0, run)
            add(mx, f"sbc_maxrunlength_{tag}")
            add(tot / np.maximum(cnt, 1.0), f"sbc_meanrunlength_{tag}")
            add((dM[:, 1:] * dM[:, :-1] < 0).sum(1).astype(np.float32), f"sbc_zerocross_{tag}")
            add(dM.mean(1), f"sbc_meanlast_{tag}")

            # ---- moments -------------------------------------------------------------
            z = (M - mu[:, None]) / sdz[:, None]
            skew = np.mean(z ** 3, 1); kurt = np.mean(z ** 4, 1) - 3.0
            add(mu, f"sbc_mean_{tag}"); add(sd, f"sbc_std_{tag}")
            add(skew, f"sbc_skew_{tag}"); add(kurt, f"sbc_kurtosis_{tag}")
            add(np.mean(np.abs(M - mu[:, None]), 1), f"sbc_mad_{tag}")
            add(np.abs(sd / (mu + 1e-9)), f"sbc_cv_{tag}")
            add((M ** 2).sum(1), f"sbc_energy_{tag}")
            add(np.sqrt((M ** 2).mean(1)), f"sbc_rms_{tag}")
            add(n * (M ** 2).sum(1), f"sbc_fft_{tag}")
            add((n / 6.0) * (skew ** 2 + (kurt ** 2) / 4.0), f"sbc_jarquebera_{tag}")
            add((sd - mu) / (sd + mu + 1e-9), f"sbc_burstiness_{tag}")
            add((M > mu[:, None]).mean(1), f"sbc_dutycycle_{tag}")

            # ---- order statistics: ONE sort, everything derived from it --------------
            if not (need_sort):
                S = None
            else:
                S = np.sort(M, axis=1)
            if S is None:
                P = None
                if need_fft:
                    P = np.abs(np.fft.rfft(M, axis=1)) ** 2 + 1e-9
                    Pn = P / P.sum(1, keepdims=True)
                    add(-(Pn * np.log(Pn)).sum(1), f"sbc_spectralentropy_{tag}")
                    add(np.exp(np.mean(np.log(P), 1)) / (P.mean(1) + 1e-9), f"sbc_spectralflatness_{tag}")
                    add(np.argmax(P, 1).astype(np.float32) / n, f"sbc_fftmaxfreq_{tag}")
                    add(P.max(1), f"sbc_domfreqpower_{tag}")
                    cs_ = np.cumsum(P, 1)
                    add(np.argmax(cs_ >= 0.85 * cs_[:, -1:], axis=1).astype(np.float32),
                        f"sbc_spectralrolloff_{tag}")
                    del P, Pn, cs_
                t_ = np.arange(n, dtype=np.float32) - (n - 1) / 2.0
                add((M @ t_) / max(float((t_ ** 2).sum()), 1e-9), f"sbc_trendslope_{tag}")
                add(M[:, :n // 2].mean(1), f"sbc_firsthalfmean_{tag}")
                add(M[:, n // 2:].mean(1), f"sbc_secondhalfmean_{tag}")
                cum = np.cumsum(M - mu[:, None], axis=1)
                add(np.log(np.maximum(cum.std(1), 1e-9)) / np.log(n + 1e-9), f"sbc_hurstexp_{tag}")
                for lag in (1, 5, 7):
                    if n <= lag + 2:
                        continue
                    A_, B_ = M[:, :-lag], M[:, lag:]
                    ca = A_ - A_.mean(1, keepdims=True); cb = B_ - B_.mean(1, keepdims=True)
                    den = np.sqrt((ca ** 2).sum(1) * (cb ** 2).sum(1))
                    add(np.where(den > 1e-9, (ca * cb).sum(1) / np.maximum(den, 1e-9), 0.0),
                        f"sbc_autocorr{lag}_{tag}")
                rng_ = np.maximum(M.max(1) - M.min(1), 1e-9)
                B10 = np.clip(((M - M.min(1, keepdims=True)) / rng_[:, None] * 10).astype(np.int8), 0, 9)
                H = np.zeros((M.shape[0], 10), np.float32)
                for b in range(10):
                    H[:, b] = (B10 == b).sum(1)
                Hn = H / np.maximum(H.sum(1, keepdims=True), 1e-9)
                add(-(Hn * np.log(np.maximum(Hn, 1e-12))).sum(1), f"sbc_histentropy_{tag}")
                del M, dM, sg, chg, z, cum, B10, H, Hn
                continue
            idx = lambda q: min(max(int(q * (n - 1)), 0), n - 1)
            q01, q10, q25, q50 = S[:, idx(.01)], S[:, idx(.10)], S[:, idx(.25)], S[:, idx(.50)]
            q75, q90, q99 = S[:, idx(.75)], S[:, idx(.90)], S[:, idx(.99)]
            # q01 and min are identically 0 on a non-negative, ~70%-zero daily series --
            # the feature audit flagged all 20 of them as exactly constant. Dropped.
            for nm, v in [("q10", q10), ("q25", q25), ("median", q50),
                          ("q75", q75), ("q90", q90), ("q99", q99)]:
                add(v, f"sbc_{nm}_{tag}")
            add(q90 - q01, f"sbc_q19_{tag}"); add(q75 - q25, f"sbc_iqr_{tag}")
            add(q90 - q10, f"sbc_interdecile_{tag}")
            add(S[:, -1], f"sbc_max_{tag}")
            add(S[:, -1] - S[:, 0], f"sbc_range_{tag}")   # min is 0, so range == max
            lo, hi = idx(.10), idx(.90)
            add(S[:, lo:hi + 1].mean(1), f"sbc_trimmedmean_{tag}")
            add(S[:, lo:hi + 1].std(1), f"sbc_trimmedstd_{tag}")
            add(np.clip(M, S[:, idx(.05)][:, None], S[:, idx(.95)][:, None]).mean(1),
                f"sbc_winsorizedmean_{tag}")
            madn = 1.4826 * np.median(np.abs(M - q50[:, None]), axis=1)
            add(madn ** 2, f"sbc_biweightmidvar_{tag}")
            ii = np.arange(1, n + 1, dtype=np.float32)
            add(2 * (ii * S).sum(1) / (n * np.maximum(S.sum(1), 1e-9)) - (n + 1) / n,
                f"sbc_gini_{tag}")
            # ---- normality, vectorised ----------------------------------------------
            # Shapiro-FRANCIA (not Wilk): W' = corr(sorted x, normal scores)^2. Verified
            # against scipy's Shapiro-Wilk at corr 0.999 on both n=30 and n=90.
            mc, mss = _normal_scores(n)
            cen = S - S.mean(1, keepdims=True)
            add(((S * mc).sum(1) ** 2) / np.maximum((cen ** 2).sum(1) * mss, 1e-12),
                f"sbc_shapirofrancia_{tag}")
            # Anderson-Darling for normality, closed form (corr 0.99999 vs scipy)
            from scipy.special import ndtr
            # float64 and a 1e-7 clip: in float32, 1-(1-1e-12) underflows to exactly 0 and
            # log(0) = -inf poisons the whole statistic. 1e-7 is near float32's resolution
            # at 1.0, so the clip actually bites.
            Z = (cen / np.maximum(S.std(1, keepdims=True), 1e-9)).astype(np.float64)
            F = np.clip(ndtr(Z), 1e-7, 1 - 1e-7)
            ii2 = np.arange(1, n + 1, dtype=np.float32)
            add(-n - ((2 * ii2 - 1) * (np.log(F) + np.log(1 - F[:, ::-1]))).sum(1) / n,
                f"sbc_andersondarling_{tag}")
            del S, cen, Z, F

            # ---- spectral ------------------------------------------------------------
            P = np.abs(np.fft.rfft(M, axis=1)) ** 2 + 1e-9
            Pn = P / P.sum(1, keepdims=True)
            add(-(Pn * np.log(Pn)).sum(1), f"sbc_spectralentropy_{tag}")
            add(np.exp(np.mean(np.log(P), 1)) / (P.mean(1) + 1e-9), f"sbc_spectralflatness_{tag}")
            add(np.argmax(P, 1).astype(np.float32) / n, f"sbc_fftmaxfreq_{tag}")
            add(P.max(1), f"sbc_domfreqpower_{tag}")
            cs_ = np.cumsum(P, 1)
            add(np.argmax(cs_ >= 0.85 * cs_[:, -1:], axis=1).astype(np.float32),
                f"sbc_spectralrolloff_{tag}")
            del P, Pn, cs_

            # ---- time-series -------------------------------------------------------
            t_ = np.arange(n, dtype=np.float32) - (n - 1) / 2.0
            add((M @ t_) / max(float((t_ ** 2).sum()), 1e-9), f"sbc_trendslope_{tag}")
            add(M[:, :n // 2].mean(1), f"sbc_firsthalfmean_{tag}")
            add(M[:, n // 2:].mean(1), f"sbc_secondhalfmean_{tag}")
            cum = np.cumsum(M - mu[:, None], axis=1)
            add(np.log(np.maximum(cum.std(1), 1e-9)) / np.log(n + 1e-9), f"sbc_hurstexp_{tag}")
            for lag in (1, 5, 7):
                if n <= lag + 2:
                    continue
                A_, B_ = M[:, :-lag], M[:, lag:]
                ca = A_ - A_.mean(1, keepdims=True); cb = B_ - B_.mean(1, keepdims=True)
                den = np.sqrt((ca ** 2).sum(1) * (cb ** 2).sum(1))
                add(np.where(den > 1e-9, (ca * cb).sum(1) / np.maximum(den, 1e-9), 0.0),
                    f"sbc_autocorr{lag}_{tag}")

            # ---- histogram entropy (the original's `hss` and `permentropy`) ----------
            rng_ = np.maximum(M.max(1) - M.min(1), 1e-9)
            B10 = np.clip(((M - M.min(1, keepdims=True)) / rng_[:, None] * 10).astype(np.int8), 0, 9)
            H = np.zeros((M.shape[0], 10), np.float32)
            for b in range(10):
                H[:, b] = (B10 == b).sum(1)
            Hn = H / np.maximum(H.sum(1, keepdims=True), 1e-9)
            add(-(Hn * np.log(np.maximum(Hn, 1e-12))).sum(1), f"sbc_histentropy_{tag}")
            del M, dM, sg, chg, z, cum, B10, H, Hn
    return np.column_stack(cols), names


def _sbc_family(fam):
    return lambda p, anchor, keep: block_sbc(p, anchor, keep, want=SBC_FAMILIES[fam])



FCAST_FAMILIES = {
    "fcomega": {"omega_raw", "omega_bounded"},                    # spectral predictability
    "fcstl":   {"trendstr", "seasstr", "residshare"},             # TFB trend/seasonality
    "fcshift": {"shift_lorentz", "shift_logq"},                   # distributional drift
    "fctrans": {"transition", "selftransition"},                  # 3-symbol chain
    "fcacf":   {"acf_abssum", "acf_argmax", "acf_max"},           # ACF shape
}
FCAST_ALL = set().union(*FCAST_FAMILIES.values())


def block_fcast(p, anchor: int, keep: np.ndarray, want: set | None = None):
    """
    Forecastability + series-taxonomy measures (PAPERS_new.md §8.1, §8.2).

    These score how PREDICTABLE a user's series is, before any model sees it -- a different
    question from every feature we have, which describe level, dispersion or shape.

    * Spectral predictability (arXiv:2507.13556). The paper ran the sparsity study we need and
      found it stable down to ~100 time steps and robust to sparsity, which is why it is here
      and the Lyapunov exponent is NOT: above 0.8 sparsity the exponent can fall and *falsely
      indicate a stable, forecastable system*. Most of our users sit far above 0.8, so it would
      lie in precisely the direction that makes dormant users look predictable.
      The novel ingredient over our existing `specentropy` is the HANN WINDOW, which suppresses
      spectral leakage; both the paper's log(2pi) normalisation and a bounded log(K) variant are
      emitted since the paper's is not bounded in [0,1] for our bin counts.
    * TFB's six characteristics (arXiv:2403.20150): trend strength, seasonality strength,
      shifting, transition. STL is too slow per user at 250k x 81 anchors, so trend is a
      centred 7-day moving average and seasonality is day-of-week means of the detrended
      series -- the same decomposition STL performs, done in closed form.
    """
    want = FCAST_ALL if want is None else want
    cols, names = [], []

    def add(v, n):
        stat = n[3:].rsplit("_", 2)[0]              # fc_<stat>_<col>_<window>
        if stat in want:
            cols.append(np.asarray(v, np.float32)[keep]); names.append(n)

    for col in ["gmv", "ord"]:
        for w in [90, 365]:
            a = max(anchor - w + 1, 0)
            M = p.raw[col][:, a:anchor + 1].astype(np.float32)
            n = M.shape[1]
            if n < 28:
                continue
            tag = f"{col}_{w}"

            # --- spectral predictability, Hann-windowed -----------------------------
            han = np.hanning(n).astype(np.float32)
            Pw = np.abs(np.fft.rfft((M - M.mean(1, keepdims=True)) * han, axis=1)) ** 2 + 1e-12
            Pn = Pw / Pw.sum(1, keepdims=True)
            H = -(Pn * np.log(Pn)).sum(1)
            add(1.0 - H / np.log(2 * np.pi), f"fc_omega_raw_{tag}")      # the paper's form
            add(1.0 - H / np.log(Pn.shape[1]), f"fc_omega_bounded_{tag}")  # bounded in [0,1]
            del Pw, Pn

            # --- trend / seasonality strength (STL in closed form) ------------------
            k = 7
            ker = np.ones(k, np.float32) / k
            T = np.apply_along_axis(lambda r: np.convolve(r, ker, mode="same"), 1, M)
            D = M - T
            dow = (np.arange(a, anchor + 1) % 7)
            S = np.zeros_like(M)
            for d in range(7):
                mask = dow == d
                if mask.sum():
                    S[:, mask] = D[:, mask].mean(1, keepdims=True)
            R = D - S
            vR = R.var(1)
            add(np.maximum(0.0, 1.0 - vR / np.maximum((M - S).var(1), 1e-9)), f"fc_trendstr_{tag}")
            add(np.maximum(0.0, 1.0 - vR / np.maximum((M - T).var(1), 1e-9)), f"fc_seasstr_{tag}")
            add(vR / np.maximum(M.var(1), 1e-9), f"fc_residshare_{tag}")
            del T, D, S, R

            # --- shifting: how far the second half sits from the first -------------
            h = n // 2
            qa = np.quantile(M[:, :h], np.linspace(0, 1, 9), axis=1).T
            qb = np.quantile(M[:, h:], np.linspace(0, 1, 9), axis=1).T
            # Lorentzian distance (survey §8.3): log-space, beats Euclidean, and matches RMSLE
            add(np.log1p(np.abs(qa - qb)).sum(1), f"fc_shift_lorentz_{tag}")
            add(np.abs(np.log1p(qa) - np.log1p(qb)).mean(1), f"fc_shift_logq_{tag}")

            # --- transition: 3-symbol chain, trace of its covariance ---------------
            t1, t2 = np.quantile(M, [1 / 3, 2 / 3], axis=1)
            Sy = (M > t1[:, None]).astype(np.int8) + (M > t2[:, None]).astype(np.int8)
            TM = np.zeros((M.shape[0], 9), np.float32)
            for i_ in range(3):
                for j_ in range(3):
                    TM[:, 3 * i_ + j_] = ((Sy[:, :-1] == i_) & (Sy[:, 1:] == j_)).sum(1)
            TM /= np.maximum(TM.sum(1, keepdims=True), 1.0)
            add(TM.var(1), f"fc_transition_{tag}")
            add(TM[:, 0] + TM[:, 4] + TM[:, 8], f"fc_selftransition_{tag}")
            del Sy, TM

            # --- ACF shape (TS3IM §8.4): where the autocorrelation lives ----------
            acf = []
            for lag in range(1, 15):
                A_, B_ = M[:, :-lag], M[:, lag:]
                ca = A_ - A_.mean(1, keepdims=True); cb = B_ - B_.mean(1, keepdims=True)
                den = np.sqrt((ca ** 2).sum(1) * (cb ** 2).sum(1))
                acf.append(np.where(den > 1e-9, (ca * cb).sum(1) / np.maximum(den, 1e-9), 0.0))
            A14 = np.column_stack(acf)
            add(np.abs(A14).sum(1), f"fc_acf_abssum_{tag}")
            add(np.argmax(A14, 1).astype(np.float32) + 1, f"fc_acf_argmax_{tag}")
            add(A14.max(1), f"fc_acf_max_{tag}")
            del M, A14
    return np.column_stack(cols), names


SBC_NOMOMENT = SBC_ALL - SBC_FAMILIES["sbcmoment"]


# --------------------------------------------------------------------------- screen candidates
# The three blocks below are the confirm-eligible survivors of the local screen documented in
# FEATURES.md. Each is ONE feature, in its own block, so it can be tested alone (§4.1) and so a
# negative result kills exactly one hypothesis.
#
# Every one of them is a single scalar per user, which is deliberate: the screen measured
# single columns, so the confirm must add single columns. A "family" version would be a
# different experiment.
#
# CAUSALITY. FEATURES.md records that the first drafts of two of these LEAKED -- a recency scan
# over the whole panel produced a fake +0.00605 rho that collapsed to +0.00065 once the scan was
# capped at the anchor, and unbounded age masks read the target window. Both are written here
# against the Panel window API, whose every accessor clips at `min(b, n_days-1)` and is
# prefix-summed up to the anchor, so no day > anchor is reachable by construction.
# `assert_no_lookahead` re-verifies this per run.

def block_agebucket(p, anchor: int, keep: np.ndarray):
    """
    `age_bucket_gmv_share_3` -- the share of lifetime GMV a user spent in their OWN days
    [90, 120) after first becoming active.

    Gap it fills: the installed `gmv_blk{k}` are CALENDAR-anchored 30-day blocks, so "block 3"
    is a different point in the lifecycle for a user who joined in January than for one who
    joined in July -- a cohort blend. Age-aligning removes that confound, which DATA.md calls
    out as the difference between cohort and age alignment.

    Causality: both the numerator and the `life` denominator are cut at the anchor. Bucket 3 is
    the only one of the four that survives the cut for a meaningful population (a user must be
    >= 90 days old at the anchor for the bucket to be non-empty at all), which is precisely why
    the screen found buckets 0-2 dead -- for them the window ran past the anchor.

    Screen (n=30k, two anchors): d_rho +0.00074 (A1) / +0.00112 (A2) -- the largest consistent
    positive of the 19 candidates, but only ~1x the screen's noise control. FEATURES.md
    Candidate A.
    """
    first = p.first_act.astype(np.int64)
    lo = np.minimum(first + 90, anchor + 1)          # bucket start, never past the anchor
    hi = np.minimum(first + 120, anchor + 1)         # bucket end (exclusive), likewise
    lo = np.maximum(lo, p.floor)                     # honour feature_truncate_days, as wsum does
    hi = np.maximum(hi, lo)
    # cs["gmv"][:, j] = sum over days [0, j-1], so the bucket sum is one subtraction per user.
    cs = p.cs["gmv"]
    rows = np.arange(p.n_users)
    gk = np.where(hi > lo, cs[rows, hi] - cs[rows, lo], 0.0)
    life = p.wsum("gmv", 0, anchor)
    share = _safe_div(gk, life, 1e-3)
    return (np.column_stack([np.asarray(share, np.float32)[keep]]),
            ["age_bucket_gmv_share_3"])


def block_cartbacklog(p, anchor: int, keep: np.ndarray):
    """
    `cart_backlog_7` -- items added to the cart but not ordered over the trailing 7 days,
    `max(cart_7 - ord_7, 0)`.

    Gap it fills: cart is present as an input flow and orders as an outflow, but the STOCK
    between them is never formed. A user holding an unconverted basket is in a different
    near-buy state from one who carts and buys immediately, and RMSLE is dominated by the
    buy/no-buy flag rather than the amount (§1b).

    Screen (n=30k): d_rho +0.00090 (A1) / +0.00052 (A2). FEATURES.md Candidate B.
    """
    c7 = p.wsum("cart", anchor - 6, anchor)
    o7 = p.wsum("ord", anchor - 6, anchor)
    v = np.maximum(c7 - o7, 0.0)
    return np.column_stack([np.asarray(v, np.float32)[keep]]), ["cart_backlog_7"]


def block_cohortrel(p, anchor: int, keep: np.ndarray):
    """
    `cohort_rel_buy_rate90` -- the trailing-90d buy rate minus the MEDIAN buy rate of the
    user's 14-day join cohort (`first_act // 14`).

    Gap it fills: recent joiners buy less often than veterans for reasons that have nothing to
    do with their value, so a raw rate mixes lifecycle stage with propensity. Subtracting the
    cohort median isolates standing within one's own cohort.

    Leak note: the cohort median is computed over the ANCHOR POPULATION's features only -- it
    never touches y. That makes it a cross-sectional normalisation of the same kind as
    `block_rank`, which the repo already treats as leak-free (see that block's docstring).

    Screen (n=30k): d_rho +0.00011 (A1) / +0.00042 (A2) -- consistently positive but the
    smallest of the three. FEATURES.md Candidate C.
    """
    buck = p.first_act.astype(np.int64) // 14
    bd90 = p.wbuy(anchor - 89, anchor)
    d90 = p.wdays(anchor - 89, anchor)
    rate = _safe_div(bd90, d90, 1.0)
    # median per cohort, over the scored population only (`keep`): sort by cohort, then take
    # the median of each contiguous run.
    r_k = rate[keep]; b_k = buck[keep]
    order = np.argsort(b_k, kind="stable")
    bs, rs = b_k[order], r_k[order]
    uniq = np.unique(bs)
    starts = np.searchsorted(bs, uniq, side="left")
    ends = np.searchsorted(bs, uniq, side="right")
    med = np.array([np.median(rs[s_:e_]) for s_, e_ in zip(starts, ends)])
    base = med[np.searchsorted(uniq, b_k)]
    return (np.column_stack([np.asarray(r_k - base, np.float32)]),
            ["cohort_rel_buy_rate90"])


def block_noisectl(p, anchor: int, keep: np.ndarray):
    """
    One i.i.d. standard-normal column. Carries no information about anything, by construction.

    This is the CONTROL for the candidate confirms above: it measures what |delta| this
    protocol produces when a feature is definitionally worthless, which is the only honest
    yardstick for reading a +0.0003 result off a 665-feature baseline. FEATURES.md's screen
    had such a control (+0.00014 / -0.00097 d_rho); a confirm without one is a confirm you
    cannot interpret.

    Seeded on the anchor so it is deterministic per fold (rule 9) but not the same column in
    every fold -- an identical column across folds would be a constant the trees could learn
    to ignore once, which would understate the noise.
    """
    rng = np.random.default_rng(90210 + anchor)
    v = rng.standard_normal(int(keep.sum()))
    return np.column_stack([np.asarray(v, np.float32)]), ["noise_ctl"]


def _fcast_family(fam):
    return lambda p, anchor, keep: block_fcast(p, anchor, keep, want=FCAST_FAMILIES[fam])


BLOCKS = {"base": block_base, "funnel": block_funnel, "tsfeat": block_tsfeat, "sbc": block_sbc,
          "fcast": block_fcast,
          **{f: _fcast_family(f) for f in FCAST_FAMILIES},
          "sbcnomoment": lambda p, a, k: block_sbc(p, a, k, want=SBC_NOMOMENT),
          **{f: _sbc_family(f) for f in SBC_FAMILIES}, "counts": block_counts, "trend": block_trend,
          "rank": block_rank, "visit": block_visit, "channel": block_channel,
          "diff": block_diff, "cumshare": block_cumshare, "ewm": block_ewm, "com": block_com,
          "dispersion": block_dispersion, "tsfresh": block_tsfresh,
          # FEATURES.md confirm-eligible screen candidates, one feature each (§4.1)
          "agebucket": block_agebucket, "cartbacklog": block_cartbacklog,
          "cohortrel": block_cohortrel, "noisectl": block_noisectl}


def lookback_of(name: str) -> float:
    """Longest history a feature can see, in days -- parsed from its name.

    Used by `feature_max_window` to build deliberately short-sighted models for a blend.
    Anything lifetime-scoped returns inf so it is dropped from restricted members.
    """
    import re as _re
    if any(t in name for t in ("_total", "tenure")):
        return float("inf")
    m = _re.search(r"blk(\d+)$", name)
    if m:
        return (int(m.group(1)) + 1) * 30.0
    m = _re.search(r"^d[12]_\w+?_(\d+)$", name)
    if m:
        return (int(m.group(1)) + 2) * 30.0
    m = _re.search(r"hl(\d+)", name)
    if m:
        return 3.0 * int(m.group(1))                 # ~3 half-lives of effective support
    if name == "geo3":
        return 90.0
    if name.startswith("recency"):
        return 30.0                                  # bounded by the population rule
    m = _re.search(r"_w?(\d+)$", name)
    if m:
        return float(m.group(1))
    return 90.0                                      # unsuffixed ratios are built on 90d


# ---------------------------------------------------------------------------- cache
# Feature building, not fitting, is the binding cost: the 365-day windows mean a sort and an
# FFT over a 250k x 365 matrix at every anchor, and the same anchors recur in every
# experiment. Caching per (block, anchor) rather than per experiment means `base` is computed
# once and reused everywhere, and the expensive `sbc` block once per anchor.
#
# README.md asks for exactly this, keyed on the feature-block config hash.
#
# A cache is a correctness risk before it is a speed win: every failure mode here returns
# WRONG NUMBERS THAT LOOK LIKE RESULTS rather than crashing. The four that matter, and what
# stops each:
#   stale code      -> the key carries a hash of features.py + data.py       (_code_hash)
#   wrong rows      -> the key carries a hash of the `keep` mask            (_keep_hash)
#   torn file       -> write to a per-process temp name, then atomic rename (_write_npz)
#   defeated guard  -> assert_no_lookahead disables the cache while it runs (run.py)
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "featcache"
CACHE_ENABLED = False           # opt in via enable_cache(); see the guard note in run.py
CACHE_BUDGET_GB = float(os.environ.get("ECUP_CACHE_BUDGET_GB", 40.0))
MIN_FREE_GB = 25.0              # filesystem-level backstop, below the quota check
RESCAN_EVERY = 64               # array jobs share the dir, so re-measure it periodically

# "build" counts every real block computation, cache on or off; "miss" only counts the ones
# where the cache was live and failed to serve. Keeping them apart is what lets
# verify_cache.py prove the look-ahead guard actually rebuilt rather than being served.
_STATS = {"hit": 0, "miss": 0, "build": 0, "written": 0, "bytes": 0, "skipped_budget": 0}
_TMP_SEQ = itertools.count()
_gen_bytes = -1.0               # bytes already in this generation's dir; -1 = not scanned
_writes_since_scan = 0
_validated = False              # quota checked once per process, not per restore


@lru_cache(maxsize=1)
def _code_hash() -> str:
    """Invalidate the whole cache whenever the feature or panel code changes.

    Coarse on purpose: a cosmetic edit throws the cache away, which costs a rebuild. The
    alternative -- a stale cache silently serving features that no longer match the code --
    is the worst failure this project could have, and it would look like a modelling result.

    Memoised so a long job keeps one identity even if the source is edited underneath it;
    the bytes that matter are the ones that were imported.
    """
    h = hashlib.sha1()
    for nm in ("features.py", "data.py"):
        h.update((Path(__file__).resolve().parent / nm).read_bytes())
    return h.hexdigest()[:10]


def _keep_hash(keep: np.ndarray) -> str:
    """The cached block is a function of `keep` as well as the anchor -- it holds one row per
    selected user. Row COUNT alone cannot separate two different populations of equal size,
    so the mask itself goes into the key."""
    return hashlib.sha1(np.packbits(np.asarray(keep, bool)).tobytes()).hexdigest()[:8]


# --- size accounting -------------------------------------------------------------------
_UNITS = {"K": 1e-6, "M": 1e-3, "G": 1.0, "T": 1e3, "P": 1e6}


def _parse_size_gb(tok: str) -> float:
    tok = tok.strip()
    if not tok or tok in ("none", "no_limit", "-"):
        return float("inf")
    if tok[-1].upper() in _UNITS:
        return float(tok[:-1]) * _UNITS[tok[-1].upper()]
    return float(tok) / 1e6                      # bare mmlsquota numbers are KB


def _quota_state() -> dict | None:
    """Our real constraint is the GPFS user quota, not the filesystem's free space -- the
    array can report terabytes free while we are already over quota and every write fails."""
    try:
        out = subprocess.run(
            ["/usr/lpp/mmfs/bin/mmlsquota", "--block-size", "auto", "gpfsFlash"],
            capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return None
    for ln in out.splitlines():
        f = ln.split()
        if len(f) >= 6 and f[0] == "gpfsFlash" and f[2] == "USR":
            pipe = f.index("|") if "|" in f else len(f)
            return {"used_gb": _parse_size_gb(f[3]), "soft_gb": _parse_size_gb(f[4]),
                    "hard_gb": _parse_size_gb(f[5]), "grace": " ".join(f[7:pipe])}
    return None


def _free_gb(path: Path) -> float:
    try:
        st = os.statvfs(path)
        return st.f_bavail * st.f_frsize / 1e9
    except Exception:
        return 0.0


def _scan_gen_bytes() -> float:
    d = CACHE_DIR / _code_hash()
    if not d.is_dir():
        return 0.0
    return float(sum(f.stat().st_size for f in d.glob("*.npz")))


def enable_cache(on: bool = True, force: bool = False) -> None:
    """Turn the cache on, refusing if there is not room for it.

    Refusing is the point: the alternative is discovering the quota by way of a job that
    dies four hours in, or -- worse -- one that half-writes the cache and carries on.
    """
    global CACHE_ENABLED, _gen_bytes, _writes_since_scan, _validated
    if not on:
        CACHE_ENABLED = False
        return
    if _validated:                               # cheap restore path (assert_no_lookahead)
        CACHE_ENABLED = True
        return
    q = _quota_state()
    if q is not None:
        head = q["hard_gb"] - q["used_gb"]
        if q["used_gb"] > q["soft_gb"]:
            print(f"  [cache] over SOFT quota: {q['used_gb']:.0f}G > {q['soft_gb']:.0f}G, "
                  f"grace {q['grace']!r}; {head:.0f}G left to the hard limit", flush=True)
        if head < CACHE_BUDGET_GB * 1.5 and not force:
            print(f"  [cache] REFUSING: {head:.0f}G to the hard limit is under 1.5x the "
                  f"{CACHE_BUDGET_GB:.0f}G budget. Set feature_cache_force to override.",
                  flush=True)
            CACHE_ENABLED = False
            return
    _gen_bytes = _scan_gen_bytes()
    _writes_since_scan = 0
    _validated = True
    CACHE_ENABLED = True


def _may_write(nbytes: int) -> bool:
    global _gen_bytes, _writes_since_scan
    if _gen_bytes < 0:
        _gen_bytes = _scan_gen_bytes()
    elif _writes_since_scan >= RESCAN_EVERY:
        # concurrent array tasks write into the same directory, so an in-process counter
        # drifts low by a factor of however many jobs are running. Re-measure periodically.
        _gen_bytes = _scan_gen_bytes()
        _writes_since_scan = 0
    if (_gen_bytes + nbytes) / 1e9 > CACHE_BUDGET_GB or _free_gb(CACHE_DIR.parent) < MIN_FREE_GB:
        _STATS["skipped_budget"] += 1
        return False
    return True


def _write_npz(cp: Path, X: np.ndarray, n: list[str]) -> None:
    global _gen_bytes, _writes_since_scan
    cp.parent.mkdir(parents=True, exist_ok=True)
    # Unique temp name per process AND per call: array tasks build the same anchors at the
    # same time, and a shared temp path lets one task rename another's half-written file
    # into place -- a torn cache entry that loads cleanly and returns garbage.
    tmp = cp.with_name(f"{cp.stem}.{os.getpid()}.{next(_TMP_SEQ)}.tmp.npz")
    try:
        np.savez(tmp, X=X, names=np.array(n))
        tmp.replace(cp)                          # atomic within the directory
    except Exception:
        tmp.unlink(missing_ok=True)              # a failed write must never be a cache entry
        return
    _STATS["written"] += 1
    _STATS["bytes"] += int(X.nbytes)
    _gen_bytes += int(X.nbytes)
    _writes_since_scan += 1


def cache_stats() -> dict:
    d = dict(_STATS)
    d["hit_rate"] = d["hit"] / max(d["hit"] + d["miss"], 1)
    d["gen"] = _code_hash()
    d["gen_gb"] = max(_gen_bytes, 0.0) / 1e9
    return d


def _cache_path(block: str, anchor: int, floor: int, kh: str) -> Path:
    return CACHE_DIR / _code_hash() / f"{block}_a{anchor}_f{floor}_k{kh}.npz"


def build(p, anchor: int, keep: np.ndarray, blocks: list[str],
          max_window: float | None = None) -> tuple[np.ndarray, list[str]]:
    Xs, names = [], []
    floor = getattr(p, "floor", 0)
    kh = _keep_hash(keep) if CACHE_ENABLED else ""
    for b in blocks:
        if b not in BLOCKS:
            raise KeyError(f"unknown feature block {b!r}; known: {sorted(BLOCKS)}")
        # `max_window` is deliberately NOT in the key. The cached array is the full block;
        # the window filter is applied below, after assembly. Keying on it would make the
        # lookback ensemble (15/60/120/180/240/360) store six byte-identical copies of every
        # block -- the one experiment family the cache exists to make affordable.
        cp = _cache_path(b, anchor, floor, kh) if CACHE_ENABLED else None
        X = n = None
        if cp is not None and cp.exists():
            try:
                z = np.load(cp, allow_pickle=False)
                X, n = z["X"], [str(s) for s in z["names"]]
                if X.shape[0] != int(keep.sum()):    # belt and braces; kh already covers it
                    raise ValueError("row count mismatch")
                _STATS["hit"] += 1
            except Exception:
                cp.unlink(missing_ok=True)           # corrupt or stale -> drop and rebuild
                X = n = None
        if X is None:
            _STATS["build"] += 1
            if cp is not None:
                _STATS["miss"] += 1
            X, n = BLOCKS[b](p, anchor, keep)
            if cp is not None and _may_write(X.nbytes):
                _write_npz(cp, X, n)
        Xs.append(X); names += n
    X = np.column_stack(Xs) if len(Xs) > 1 else Xs[0]
    if max_window:
        keepc = [i for i, n in enumerate(names) if lookback_of(n) <= max_window]
        X, names = X[:, keepc], [names[i] for i in keepc]
    return X, names
