#!/usr/bin/env python
"""
Panel, target, tmask and the A/B/C feature sets of CAUSAL_EXP.md.

Implements that document literally, with the three decisions taken on 2026-08-13 folded in:

  * `target_agg = "sum"` (not the doc's "mean" default).  The competition metric is RMSLE on
    the 30-day SUM, so with `sum` the training loss IS the metric.  With `mean` it is not:
    log1p(30m) != log1p(m) + log(30), so `expm1(pred)*30` would not preserve the log-space
    error the model was fitted to.
  * last train/val anchor = 2026-01-14 (index 378) -- the doc's own horizon-tail bound,
    T - horizon - 1.  Its target window is exactly 2026-01-15 .. 2026-02-13, the last observed
    day.  One day later would give a 29-day window that reads as a low-spending user rather
    than a truncated one, which is the failure §2.1 exists to prevent.  Inference still rolls
    the anchor forward to the last available day (2026-02-13), per §7.
    NOTE the consequence, flagged and not fixed: every anchor from 2025-11-16 on has a target
    window inside the guaranteed-activity zone, where 100% of users are active by construction
    (DATA.md §4).  On the date-split protocol that bias measured +0.041.
  * `FLAG_COLS` is 6, not the doc's "six has_* plus search, cat" -- the source table has FOUR
    `has_*` columns (DATA.md §2.1).  6 flags + 10 counts = 16 value columns, not 18.

AMBIGUITY RESOLVED EXPLICITLY: §1 says counts are log1p-ed, §3 builds rolling means over
COUNT_COLS, and the two orders are not the same function.  Window aggregates here are computed
on RAW values and log1p-ed afterwards -- log1p(mean of x), not mean of log1p(x) -- which is
what the tabular pipeline does for its window features and keeps a window's level on the same
scale as the day-t value.

Every feature is causal: the value at t reads only days <= t.  §9's look-ahead assert is
implemented in `assert_causal_features` and run by the self-test, not left as hope.
"""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "data" / "train.parquet"

HORIZON = 30
WINDOWS = (7, 30, 60, 90)
FLAG_COLS = ("search", "cat", "has_search_to_cart", "has_search_to_ord",
             "has_cat_to_cart", "has_cat_to_ord")
COUNT_COLS = ("searches", "search_to_cart", "search_to_ord", "cat_to_cart", "cat_to_ord",
              "to_cart", "to_ord", "gmv_search", "gmv_cat", "gmv")
RAW_COLS = FLAG_COLS + COUNT_COLS                      # 6 + 10 = 16
EPS = 1e-6


def hash_fold(user_ids: np.ndarray, salt: str = "gmv-v1", n_folds: int = 5) -> np.ndarray:
    """CAUSAL_EXP.md §4's deterministic md5 user split, generalised to n_folds.

    Lives here rather than in run_usercv.py so the tabular runner can import it without
    pulling in torch -- and, more importantly, so both runners get byte-identical fold
    membership. If they diverged, no comparison against e0141 would mean anything.
    """
    out = np.empty(user_ids.size, np.int8)
    for i, u in enumerate(user_ids):
        h = int(hashlib.md5(f"{salt}:{u}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        out[i] = min(int(h * n_folds), n_folds - 1)
    return out


# ------------------------------------------------------------------ causal window helpers
def rolling_mean(x: np.ndarray, W: int) -> np.ndarray:
    """CAUSAL_EXP.md §3, verbatim: denominator clipped at the start of the series."""
    cs = np.concatenate([np.zeros((x.shape[0], 1)), np.cumsum(x, 1, dtype=np.float64)], 1)
    idx = np.arange(x.shape[1])
    lo = np.maximum(0, idx + 1 - W)
    return ((cs[:, idx + 1] - cs[:, lo]) / (idx + 1 - lo)).astype(np.float32)


def rolling_std(x: np.ndarray, W: int) -> np.ndarray:
    m = rolling_mean(x, W)
    m2 = rolling_mean(x.astype(np.float64) ** 2, W)
    return np.sqrt(np.maximum(m2 - m.astype(np.float64) ** 2, 0.0)).astype(np.float32)


def days_since(flag: np.ndarray, T: int) -> np.ndarray:
    """Days since the last True, evaluated at every t; T is the 'never' sentinel."""
    t = np.arange(T, dtype=np.int32)[None, :]
    last = np.maximum.accumulate(np.where(flag, t, -1), axis=1)
    return np.where(last < 0, T, t - last).astype(np.float32)


def fifo_queue_age(to_cart: np.ndarray, to_ord: np.ndarray) -> np.ndarray:
    """Days the item converting at t waited in the cart, by FIFO matching.

    `to_ord <= to_cart` holds exactly on this data (DATA.md 2.1, 0 violations on 30.6M rows),
    so the cumulative curves nest and the match is well defined without any item ids: the
    n-th order is matched to the n-th cart event, and

        age[t] = t - min{ j : cumcart[j] >= cumord[t] }

    is the wait experienced by the item converting at t -- equivalently, the age of the oldest
    still-unconverted selection.  This is the only construction in the project that measures
    the DELAY between selecting and buying rather than the rate of it.

    Causal: cumord[t] <= cumcart[t], so the answer is always <= t and never reads a future day.
    (The look-ahead assert checks this rather than trusting the argument.)
    """
    n, T = to_cart.shape
    cc = np.cumsum(to_cart, 1, dtype=np.float64)
    co = np.cumsum(to_ord, 1, dtype=np.float64)
    age = np.empty((n, T), np.float32)
    t = np.arange(T, dtype=np.float32)
    for u in range(n):                      # ~5 us per user; a vectorised two-pointer is
        age[u] = t - np.searchsorted(cc[u], co[u], side="left")   # slower here in practice
    return np.maximum(age, 0.0)


class Raw:
    """Dense (n_users, T) matrix per source column, plus the `active` mask."""

    def __init__(self, path: Path = TRAIN, verbose: bool = True):
        df = pl.read_parquet(path)
        self.dmin: date = df["event_date"].min()
        self.dmax: date = df["event_date"].max()
        self.T = (self.dmax - self.dmin).days + 1
        self.users = np.sort(df["user_id"].unique().to_numpy())
        self.n = self.users.size
        ui = np.searchsorted(self.users, df["user_id"].to_numpy())
        di = (df["event_date"].to_numpy().astype("datetime64[D]")
              - np.datetime64(self.dmin)).astype(np.int32)
        # DATA.md §3: zero duplicate (user_id, date) pairs, so the doc's groupby-sum is a
        # no-op here and scatter-assign is exact.  Asserted rather than assumed.
        flat = ui.astype(np.int64) * self.T + di
        assert np.unique(flat).size == flat.size, "duplicate (user_id, event_date) rows"
        self.col: dict[str, np.ndarray] = {}
        for c in RAW_COLS:
            A = np.zeros((self.n, self.T), np.float32)
            A[ui, di] = df[c].to_numpy().astype(np.float32)
            if c in FLAG_COLS:
                A = (A > 0).astype(np.float32)
            self.col[c] = A
        self.active = np.zeros((self.n, self.T), np.float32)
        self.active[ui, di] = 1.0
        if verbose:
            print(f"  Raw panel: {self.n:,} users x {self.T} days ({self.dmin} .. {self.dmax}), "
                  f"{len(RAW_COLS)} value columns, active {self.active.mean():.3f} dense",
                  flush=True)

    def idx(self, d: date) -> int:
        return (d - self.dmin).days

    def day(self, i: int):
        from datetime import timedelta
        return self.dmin + timedelta(days=int(i))


# ---------------------------------------------------------------------------- target/mask
def build_target(raw: Raw, agg: str = "sum") -> np.ndarray:
    """log1p of gmv aggregated over [t+1, t+horizon].  Day t excluded (§2)."""
    cs = np.concatenate([np.zeros((raw.n, 1)), np.cumsum(raw.col["gmv"], 1, dtype=np.float64)], 1)
    Y = np.zeros((raw.n, raw.T), np.float32)
    hi = raw.T - HORIZON
    w = cs[:, 1 + HORIZON: 1 + HORIZON + hi] - cs[:, 1:1 + hi]
    if agg == "mean":
        w = w / HORIZON
    elif agg != "sum":
        raise ValueError(agg)
    Y[:, :hi] = np.log1p(w).astype(np.float32)
    return Y


def geo3_log(raw: Raw) -> np.ndarray:
    """log1p of the geo3 naive baseline at every (user, day).

    geo3 = expm1(mean of log1p over the three trailing 30-day blocks) -- the strongest single
    scalar this project has (DATA.md §9, RMSLE 1.919 on its own) and the reference every
    experiment is scored against.  Used as a per-user OFFSET so a model can learn the deviation
    from a user's own level instead of the level itself.

    Causal: block b covers [t-29-30b, t-30b], all <= t.  Blocks falling entirely before day 0
    contribute 0, which is log1p of an empty window, not a clamped garbage value.
    """
    cs = np.concatenate([np.zeros((raw.n, 1)),
                         np.cumsum(raw.col["gmv"], 1, dtype=np.float64)], 1)
    t = np.arange(raw.T)
    g = np.zeros((raw.n, raw.T), np.float64)
    for b in range(3):
        hi, lo = t - 30 * b, t - 29 - 30 * b
        g += np.log1p((cs[:, np.maximum(hi, 0) + 1] - cs[:, np.maximum(lo, 0)]) * (hi >= 0))
    return (g / 3.0).astype(np.float32)


def max_anchor(raw: Raw) -> int:
    """Last t whose target window [t+1, t+HORIZON] is fully observed: T - HORIZON - 1."""
    return raw.T - HORIZON - 1


def build_tmask(raw: Raw, last_anchor: int, burn_in: int = 14,
                trim_to_first_seen: bool = True) -> np.ndarray:
    """§2.1: burn-in at the head, anchor cap at the tail."""
    t = np.arange(raw.T)[None, :]
    m = (t <= last_anchor)
    if trim_to_first_seen:
        first = np.argmax(raw.active > 0, axis=1)[:, None]
        m = m & (t >= first + burn_in)
    else:
        m = m & (t >= burn_in)
    return np.broadcast_to(m, (raw.n, raw.T)).copy()


# ------------------------------------------------------------------------- feature sets
def _log(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(x, 0.0)).astype(np.float32)


BEHAV_LOCAL = ["cart_minus_ord_d", "queue_age", "cart_backlog"]
BEHAV_WIN = ["cart_minus_ord", "conv", "aov", "basket", "gmv_per_buyday",
             "cart_no_ord_days", "queue_age_mean", "cart_per_srch", "srch_per_cart"]


def feature_names(variant: str) -> list[str]:
    n: list[str] = []
    if variant == "gmv_only":
        return ["gmv", "gmv_rm7", "gmv_rm30", "active"]
    n += [f"raw_{c}" for c in RAW_COLS]
    n += [f"{c}_rm{w}" for c in COUNT_COLS for w in WINDOWS]
    n += [f"{c}_rate{w}" for c in FLAG_COLS for w in WINDOWS]
    n += ["active"] + [f"active_rm{w}" for w in WINDOWS]
    if variant == "full":
        return n
    if variant == "full_dso":
        # FEATURES_CAUSAL.md: EXACTLY ONE channel on top of `full` (README.md). The proxy
        # screen put ds_order at -0.00360 while ds_cart/ds_active measured nil -- the claim is
        # that order-conversion recency is the one long-memory scalar the recurrence cannot
        # cheaply carry. It has only ever been tested BUNDLED (e0142, 143 feat); this isolates it.
        return n + ["dso"]
    if variant == "full_backlog":
        # secondary candidate, -0.00033 on the proxy (~6x its noise). Separate run, never
        # combined with dso: e0110 died from exactly that kind of accumulation.
        return n + ["cart_backlog"]
    if variant == "full_popidx":
        # ONE broadcast channel: the population's trailing-30d mean log1p(gmv) at each day.
        #
        # WHY. e0215/e0216 measured the 5-channel calendar block at +0.00049 rho on a
        # forward-in-calendar anchor -- the largest feature effect on this path -- but e0142
        # proves moy_sin/moy_cos cost -0.00262 rho at the YEAR boundary, because day-of-year
        # maps 2026-02 onto 2025-02's LEVEL and the platform grew ~15% in between. This
        # channel carries the same "what regime are we in" information with the year bug
        # removed: the level is MEASURED, not mapped, so at the test anchor it reports
        # Jan-Feb 2026's actual level instead of Feb 2025's.
        #
        # It is also strictly outside a per-user GRU's hypothesis space -- the recurrence sees
        # one user and can never compute a cross-sectional quantity (that is why e0114's rank
        # channels were tried at all). Causal: day t uses days [t-29, t] only.
        return n + ["pop_gmv_30"]
    if variant == "behav":
        # B + interpretable behavioural blocks. Calendar is deliberately absent: e0142 showed
        # day-of-year maps the test anchor onto Feb-Mar 2025 and applies that level to 2026.
        n += ["dsa", "dso", "dsc"] + BEHAV_LOCAL
        n += [f"{b}_{w}" for b in BEHAV_WIN for w in WINDOWS]
        return n
    n += ["dsa", "dso", "dsc"]
    n += [f"gmv_rs{w}" for w in WINDOWS] + [f"active_rs{w}" for w in WINDOWS] \
        + [f"dsa_rs{w}" for w in WINDOWS]
    n += [f"gmv_cv{w}" for w in WINDOWS] + [f"ord_cv{w}" for w in WINDOWS]
    n += [f"{r}_{w}" for r in ("ord_per_cart", "aov", "cart_per_srch", "s_ord_per_cart",
                               "c_ord_per_cart", "search_share") for w in WINDOWS]
    n += ["trend_7_30", "trend_30_90"]
    n += ["tenure", "cum_active", "exp_mean_gmv", "exp_max_gmv"]
    if variant == "extra_nodoy":
        # `extra` minus ONLY the two day-of-year channels. Rationale, measured (e0215/e0216):
        # the 5-channel calendar block is worth +0.00049 rho on a forward-in-calendar anchor,
        # but e0142 shows it costs -0.00262 rho at the YEAR boundary. dow (period 7) and dom
        # (period ~30) repeat dozens of times inside the training range, so they carry no
        # year-scale extrapolation risk; moy_sin/moy_cos (period 365.25) appear ONCE and are
        # the only channels that can map 2026-02 onto 2025-02's level. This variant asks
        # whether the +0.00049 lives in the safe half.
        return n + ["dow_sin", "dow_cos", "dom"]
    if variant == "extra_nocal":
        # `extra` minus its calendar block, and NOTHING else -- the one change (README.md).
        # Why this variant exists: `extra` beats `full` by -0.00080 on 5/5 folds of the
        # user-split CV and is -0.00262 rho WORSE on the leaderboard (e0142 vs e0141). The
        # whole CV win was attributed to the calendar block's interpolation artefact and the
        # variant was abandoned -- but that attribution was never MEASURED, and the other 53
        # channels went into the graveyard with it. This is the e0003 lesson exactly
        # ("my error -- a negative bundle hid a positive part": trend = ewm + com + diff).
        return n
    n += ["dow_sin", "dow_cos", "dom", "moy_sin", "moy_cos"]
    return n


def build_features(raw: Raw, variant: str, verbose: bool = True) -> tuple[np.ndarray, list[str]]:
    """(n_users, T, F) float16.  Channels written one at a time to keep peak RSS down."""
    names = feature_names(variant)
    X = np.zeros((raw.n, raw.T, len(names)), np.float16)
    k = 0

    def put(mat: np.ndarray) -> None:
        nonlocal k
        X[:, :, k] = mat.astype(np.float16)
        k += 1

    if variant == "gmv_only":
        put(_log(raw.col["gmv"]))
        put(_log(rolling_mean(raw.col["gmv"], 7)))
        put(_log(rolling_mean(raw.col["gmv"], 30)))
        put(raw.active)
        assert k == len(names)
        return X, names

    for c in RAW_COLS:
        put(raw.col[c] if c in FLAG_COLS else _log(raw.col[c]))
    for c in COUNT_COLS:
        for w in WINDOWS:
            put(_log(rolling_mean(raw.col[c], w)))
    for c in FLAG_COLS:                       # usage RATE over the window, not the flag
        for w in WINDOWS:
            put(rolling_mean(raw.col[c], w))
    put(raw.active)
    for w in WINDOWS:
        put(rolling_mean(raw.active, w))

    if variant == "full":
        assert k == len(names), f"{k} != {len(names)}"
        return X, names

    if variant == "full_dso":
        # days since the last day with to_ord > 0; T = "never ordered" sentinel, log1p-scaled.
        # Same construction as the `dso` channel of behav/extra, so a win here is directly
        # attributable to the channel and not to a different definition of it.
        put(_log(days_since(raw.col["to_ord"] > 0, raw.T)))
        assert k == len(names), f"{k} != {len(names)}"
        if verbose:
            print(f"  features[{variant}]: {len(names)} channels, "
                  f"{X.nbytes / 1e9:.2f} GB fp16", flush=True)
        return X, names

    if variant == "full_backlog":
        # running stock of selected-but-not-yet-converted items. DATA.md 2.1 verifies
        # to_ord <= to_cart per day (0 violations), so the per-day gap is already >= 0 and the
        # cumulative sum is non-decreasing; the max() is belt-and-braces, matching `behav`.
        gap = np.maximum(raw.col["to_cart"] - raw.col["to_ord"], 0.0)
        put(_log(np.cumsum(gap, 1, dtype=np.float64)))
        assert k == len(names), f"{k} != {len(names)}"
        if verbose:
            print(f"  features[{variant}]: {len(names)} channels, "
                  f"{X.nbytes / 1e9:.2f} GB fp16", flush=True)
        return X, names

    if variant == "full_popidx":
        # population trailing-30d mean of log1p(gmv), broadcast to every user.
        # Order matters and is deliberate: log1p FIRST, then average across users, so a few
        # whales cannot dominate the index (raw-space pooling would make this a whale tracker
        # rather than a regime index -- the same heavy-tail trap that put |z|=249 covariates
        # into e0180).  Then a trailing 30-day mean over days <= t, using the same
        # clipped-denominator helper as every other window feature.
        pop_day = _log(raw.col["gmv"]).mean(axis=0, keepdims=True)        # (1, T)
        pop = rolling_mean(pop_day, 30)                                   # (1, T), causal
        put(np.broadcast_to(pop, (raw.n, raw.T)))
        assert k == len(names), f"{k} != {len(names)}"
        if verbose:
            print(f"  features[{variant}]: {len(names)} channels "
                  f"(= full + population trailing-30d level), "
                  f"{X.nbytes / 1e9:.2f} GB fp16", flush=True)
        return X, names

    dsa = days_since(raw.active > 0, raw.T)
    dso = days_since(raw.col["to_ord"] > 0, raw.T)
    dsc = days_since(raw.col["to_cart"] > 0, raw.T)
    for m in (dsa, dso, dsc):
        put(_log(m))

    if variant == "behav":
        cart, ordr, gmv = raw.col["to_cart"], raw.col["to_ord"], raw.col["gmv"]
        srch = raw.col["searches"]
        gap = np.maximum(cart - ordr, 0.0)            # selected but not (yet) bought
        qage = fifo_queue_age(cart, ordr)
        buyday = (gmv > 0).astype(np.float32)
        cart_no_ord = ((cart > 0) & (ordr == 0)).astype(np.float32)
        put(_log(gap))                                                    # local
        put(_log(qage))
        put(_log(np.cumsum(gap, 1, dtype=np.float64)))                    # the cart "stock"
        for b in BEHAV_WIN:                                               # windowed
            for w in WINDOWS:
                c_, o_, g_, s_ = (rolling_mean(cart, w), rolling_mean(ordr, w),
                                  rolling_mean(gmv, w), rolling_mean(srch, w))
                bd = rolling_mean(buyday, w)
                if b == "cart_minus_ord":
                    put(_log(rolling_mean(gap, w)))
                elif b == "conv":            # P(buy | selected)
                    put(np.clip(o_ / (c_ + EPS), 0.0, 1.5))
                elif b == "aov":             # GMV per item purchased
                    put(_log(np.clip(g_ / (o_ + EPS), 0.0, 5e4)))
                elif b == "basket":          # items per purchase occasion
                    put(np.clip(o_ / (bd + EPS), 0.0, 100.0))
                elif b == "gmv_per_buyday":
                    put(_log(np.clip(g_ / (bd + EPS), 0.0, 5e4)))
                elif b == "cart_no_ord_days":
                    put(rolling_mean(cart_no_ord, w))
                elif b == "queue_age_mean":
                    put(_log(rolling_mean(qage, w)))
                elif b == "cart_per_srch":
                    put(np.clip(c_ / (s_ + EPS), 0.0, 20.0))
                elif b == "srch_per_cart":
                    put(np.clip(s_ / (c_ + EPS), 0.0, 200.0))
        assert k == len(names), f"{k} != {len(names)}"
        return X, names
    for w in WINDOWS:
        put(_log(rolling_std(raw.col["gmv"], w)))
    for w in WINDOWS:
        put(rolling_std(raw.active, w))
    for w in WINDOWS:
        put(_log(rolling_std(dsa, w)))
    for c in ("gmv", "to_ord"):
        for w in WINDOWS:
            put(np.clip(rolling_std(raw.col[c], w) / (rolling_mean(raw.col[c], w) + EPS),
                        0.0, 20.0))
    # ratios on the ROLLING sums, never per-day (§3.1C: per-day ratios are 0/0 most days)
    for num, den in (("to_ord", "to_cart"), ("gmv", "to_ord"), ("to_cart", "searches"),
                     ("search_to_ord", "search_to_cart"), ("cat_to_ord", "cat_to_cart"),
                     ("gmv_search", "gmv")):
        for w in WINDOWS:
            put(np.clip(rolling_mean(raw.col[num], w) / (rolling_mean(raw.col[den], w) + EPS),
                        0.0, 20.0))
    put(np.clip(rolling_mean(raw.col["gmv"], 7) / (rolling_mean(raw.col["gmv"], 30) + EPS),
                0.0, 20.0))
    put(np.clip(rolling_mean(raw.col["gmv"], 30) / (rolling_mean(raw.col["gmv"], 90) + EPS),
                0.0, 20.0))
    t = np.arange(raw.T, dtype=np.float32)[None, :]
    # Tenure must use a RUNNING first-active day, not `argmax` over the whole row.  The value
    # of `max(t - first_global, 0)` happens to be right, but the quantity is computed from days
    # > t, and §9's assert catches exactly that -- it was the only feature in any variant that
    # failed.  Running minimum of the active-day index, with T as the "not seen yet" sentinel.
    act_idx = np.where(raw.active > 0, np.arange(raw.T)[None, :], raw.T).astype(np.int32)
    first_causal = np.minimum.accumulate(act_idx, axis=1).astype(np.float32)
    put(_log(np.where(first_causal <= t, t - first_causal, 0.0)))
    cum = np.cumsum(raw.active, 1, dtype=np.float64)
    put(_log(cum))
    put(_log(np.cumsum(raw.col["gmv"], 1, dtype=np.float64) / (t + 1.0)))
    put(_log(np.maximum.accumulate(raw.col["gmv"], axis=1)))
    if variant == "extra_nodoy":
        dow_ = np.array([raw.day(i).weekday() for i in range(raw.T)], np.float32)[None, :]
        dom_ = np.array([raw.day(i).day for i in range(raw.T)], np.float32)[None, :]
        one_ = np.ones((raw.n, 1), np.float32)
        put(one_ * np.sin(2 * np.pi * dow_ / 7))
        put(one_ * np.cos(2 * np.pi * dow_ / 7))
        put(one_ * (dom_ / 31.0))
        assert k == len(names), f"{k} != {len(names)}"
        if verbose:
            print(f"  features[{variant}]: {len(names)} channels "
                  f"(= extra minus moy_sin/moy_cos), {X.nbytes / 1e9:.2f} GB fp16", flush=True)
        return X, names
    if variant == "extra_nocal":
        assert k == len(names), f"{k} != {len(names)}"
        if verbose:
            print(f"  features[{variant}]: {len(names)} channels "
                  f"(= extra minus dow/dom/day-of-year), {X.nbytes / 1e9:.2f} GB fp16",
                  flush=True)
        return X, names
    dow = np.array([raw.day(i).weekday() for i in range(raw.T)], np.float32)[None, :]
    doy = np.array([raw.day(i).timetuple().tm_yday for i in range(raw.T)], np.float32)[None, :]
    dom = np.array([raw.day(i).day for i in range(raw.T)], np.float32)[None, :]
    ones = np.ones((raw.n, 1), np.float32)
    put(ones * np.sin(2 * np.pi * dow / 7))
    put(ones * np.cos(2 * np.pi * dow / 7))
    put(ones * (dom / 31.0))
    put(ones * np.sin(2 * np.pi * doy / 365.25))
    put(ones * np.cos(2 * np.pi * doy / 365.25))
    assert k == len(names), f"{k} != {len(names)}"
    if verbose:
        print(f"  features[{variant}]: {len(names)} channels, "
              f"{X.nbytes / 1e9:.2f} GB fp16", flush=True)
    return X, names


def flag_channels(names: list[str]) -> np.ndarray:
    """Binary channels stay unscaled (§3.2): mu=0, sigma=1."""
    return np.array([n.startswith("raw_") and n[4:] in FLAG_COLS
                     or n == "active" or n.endswith(("_sin", "_cos")) for n in names])


# --------------------------------------------------------------------------- §9 asserts
def assert_causal_features(variant: str, n_users: int = 300, probes=(120, 250, 348)) -> None:
    """§9: zero the panel from t+1 on, rebuild, require columns at t to be identical."""
    raw = Raw(verbose=False)
    keep = np.arange(0, raw.n, max(1, raw.n // n_users))[:n_users]
    raw.users = raw.users[keep]; raw.n = keep.size
    raw.active = raw.active[keep]
    raw.col = {c: v[keep] for c, v in raw.col.items()}
    full, names = build_features(raw, variant, verbose=False)
    for t in probes:
        cut = Raw.__new__(Raw)
        cut.__dict__.update(raw.__dict__)
        cut.active = raw.active.copy(); cut.active[:, t + 1:] = 0.0
        cut.col = {c: v.copy() for c, v in raw.col.items()}
        for v in cut.col.values():
            v[:, t + 1:] = 0.0
        part, _ = build_features(cut, variant, verbose=False)
        d = np.abs(full[:, :t + 1, :].astype(np.float32) - part[:, :t + 1, :].astype(np.float32))
        bad = np.where(d.max(axis=(0, 1)) > 1e-2)[0]
        assert bad.size == 0, (f"LOOK-AHEAD in {variant} at t={t}: "
                               f"{[names[i] for i in bad][:8]}")
    print(f"  [{variant}] causality assert passed at t={list(probes)} ({len(names)} features)")


def _self_test() -> None:
    raw = Raw()
    T = raw.T
    last_anchor = max_anchor(raw)
    assert raw.day(last_anchor) == date(2026, 1, 14), raw.day(last_anchor)
    assert raw.day(last_anchor + HORIZON) == raw.dmax, "target window overruns"
    print(f"  T={T}  last train/val anchor = {raw.day(last_anchor)} (index {last_anchor})")

    Y = build_target(raw, "sum")
    rng = np.random.default_rng(0)
    for _ in range(200):                      # §9: hand-check the target
        u, t = int(rng.integers(0, raw.n)), int(rng.integers(0, T - HORIZON))
        assert t <= last_anchor
        want = np.log1p(raw.col["gmv"][u, t + 1:t + 1 + HORIZON].sum())
        assert abs(float(Y[u, t]) - float(want)) < 1e-3, f"target mismatch at ({u},{t})"
    assert (Y[:, T - HORIZON:] == 0).all(), "target not zeroed past the horizon tail"

    m = build_tmask(raw, last_anchor)
    assert not m[:, last_anchor + 1:].any(), "tmask leaks past the anchor cap"
    first = np.argmax(raw.active > 0, axis=1)
    assert not any(m[u, : first[u] + 14].any() for u in range(0, raw.n, 5000)), "burn-in leak"
    print(f"  tmask: {m.sum():,} scored user-days "
          f"({m.sum() / (raw.n * T):.1%} of the panel)")

    # cross-check the target against the frozen folds -- same quantity, independent code path
    import polars as _pl
    folds = _pl.read_parquet(ROOT / "data" / "folds.parquet")
    fv = folds.filter(_pl.col("fold_id") == 4).sort("user_id")
    a = raw.idx(date(2025, 10, 16))
    sel = np.searchsorted(raw.users, fv["user_id"].to_numpy())
    assert np.allclose(np.expm1(Y[sel, a]), fv["target"].to_numpy(), rtol=1e-4, atol=1e-3), \
        "target disagrees with data/folds.parquet"
    print("  target agrees with data/folds.parquet at anchor 2025-10-16")

    VARIANTS = ("gmv_only", "full", "full_dso", "full_backlog", "full_popidx",
                "extra", "extra_nocal", "extra_nodoy", "behav")
    for v in VARIANTS:
        print(f"  {v:12s} n_features = {len(feature_names(v))}")
    for v in VARIANTS:
        assert_causal_features(v)
    print("src/usercv_features.py: all self-tests passed")


if __name__ == "__main__":
    _self_test()
