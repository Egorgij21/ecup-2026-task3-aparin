#!/usr/bin/env python
"""
Dense (user x channel x day) tensor for the `seq` approach.

This is the input side of a genuinely different model class (EXPERIMENTS.md §4.2): instead of
collapsing a user's history into ~800 hand-built scalars at one anchor, we hand the model the
raw daily matrix and let a causal network build its own summary.

Two things this buys that the tabular pipeline structurally cannot have:

  1. DENSE SUPERVISION.  A causal architecture emits a prediction at EVERY day in one forward
     pass, so every day t with >=90d of history is a training example, not just the 7-day
     anchor grid.  Fold 4 goes from 25 anchors to 170 days -- ~6.8x more supervised positions
     for the same compute.  (The targets overlap heavily, so this is variance reduction rather
     than 6.8x more independent information; that is still the cheapest kind of gain there is.)

  2. CALENDAR TRANSLATION INVARIANCE.  No feature here references absolute time.  The tabular
     model's `gmv_365` / lifetime columns grow mechanically with the calendar, which is why
     anchor_drift.py measured the test cut-off as a 3.92x feature-space outlier.  A relative-
     only sequence model has no such drift by construction: at the test anchor it applies the
     identical function to a later window.  Dropping the drifting features from the GBDT cost
     CV (e0056/e0057) because they carry signal; here we get the invariance for free instead
     of paying for it.

Layout: X is (n_users, C, n_days) float16 -- channels-first, which is what Conv1d wants and
what makes a batch a contiguous slice.  250k x 13 x 409 x 2B = 2.66 GB, so the whole panel
lives on one H200 and there is no dataloader at all: a "batch" is a row slice of a GPU tensor.

Channel semantics follow DATA.md §2.1.  Six raw columns are exact functions of others
(`to_cart`, `to_ord` are channel sums; the four `has_*` are `1{count>0}`); the sums are kept
because they are what the target is built from and a network should not have to rediscover
them, the `has_*` flags are dropped as strictly redundant.  `presence` and `cat` are NOT
derivable from any count column and matter most: 14.85% of rows are visits with no search and
no catalogue interaction at all, and `cat` is the only catalogue-browsing signal that never
reached a cart.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "data" / "train.parquet"

HORIZON = 30

# (channel name, source column, transform).  "log" = log1p, "raw" = as-is (already 0/1),
# "flag" = 1{col > 0}.  Order is fixed: it defines the channel axis and the normalisation file.
CHANNELS: list[tuple[str, str, str]] = [
    ("presence", "__row__", "raw"),    # a row exists for this user-day (sparse panel!)
    ("buy",      "gmv",     "flag"),   # gmv > 0  <=>  to_ord > 0 (DATA.md §2.1, exact)
    ("searches", "searches", "log"),
    ("cat",      "cat",      "raw"),   # catalogue visit flag; no count column behind it
    ("s2c",      "search_to_cart", "log"),
    ("c2c",      "cat_to_cart",    "log"),
    ("s2o",      "search_to_ord",  "log"),
    ("c2o",      "cat_to_ord",     "log"),
    ("gmvs",     "gmv_search",     "log"),
    ("gmvc",     "gmv_cat",        "log"),
    ("cart",     "to_cart",        "log"),
    ("ord",      "to_ord",         "log"),
    ("gmv",      "gmv",            "log"),
]
C_IN = len(CHANNELS)

# --------------------------------------------------------------------------------------------
# Derived channels -- the tabular feature set's top-ranked family, evaluated at EVERY day.
#
# DATA.md §9.2 ranks "multi-window GMV/order/cart/search aggregates in log1p space" first by
# measured evidence, and §7.1 shows why: the 90-day sum (corr 0.589) beats the 30-day sum
# (0.557), and the lag-block correlation is still 0.416 a year back.  A GRU has to learn that
# integration through its recurrence; a prefix-sum difference just *is* it.
#
# Every one of these is `cs[:, t+1] - cs[:, t-W+1]`, so computing them for all 409 days costs
# one subtraction per channel and they are causal by construction -- window [t-W+1, t] cannot
# see past t.  That is the whole point: the anchor-grid version of this idea (feed e0049's 665
# columns at ~29 anchor dates) would throw away the dense per-day supervision that is the seq
# approach's structural advantage, and it would cost 133 GB to materialise besides.
#
# DELIBERATELY ABSENT: tenure, lifetime totals, and anything else indexed on absolute time.
# `anchor_drift.py` blamed exactly those for the test cut-off being a 3.92x feature-space
# outlier, and `src/seq_transfer.py` measured the seq family as 6.6x more cut-off-robust than
# the GBDT precisely because it has none of them.  Importing them here would trade away the one
# property the sequence approach has that the tabular one does not.
DERIVED_WINDOWS = (7, 30, 90, 365)
DERIVED_COLS = ("gmv", "ord", "cart", "srch")


def derived_channel_names() -> list[str]:
    n = [f"{c}_{w}" for c in DERIVED_COLS for w in DERIVED_WINDOWS]
    n += [f"days_{w}" for w in DERIVED_WINDOWS]
    n += [f"buydays_{w}" for w in DERIVED_WINDOWS]
    n += ["recency", "recency_ord", "geo3"]
    return n


N_DERIVED = len(derived_channel_names())

# --------------------------------------------------------------------------------------------
# Cross-sectional rank channels -- the one family a sequence model CANNOT build for itself.
#
# e0110/e0113 settled what the derived channels above are: not new information, but a
# substitute for long-range integration.  The TCN gained -0.00165 from them (5/5 folds); the
# GRU, which already integrates through its recurrence, LOST +0.00137 (0/5).  Redundant inputs
# dilute; that is the same lesson `sbcmoment` and `funnel` taught on the GBDT side.
#
# So the filter for anything added from here on is: can the network compute it itself?  For
# window aggregates the answer was yes.  For a user's rank AMONG OTHER USERS on the same day it
# is emphatically no -- every architecture here processes one user's sequence in isolation, so
# cross-sectional position is strictly outside the hypothesis space, at any depth or width.
#
# README.md lists "cross-sectional rank within timestamp" as a first-class feature family,
# and the GBDT's own `rank` block (e0004) was a keep.  BACKLOG's rationale applies here too and
# with more force: fold level drifts hard (E[L] 2.13 -> 2.44 across anchors) and the test sits
# outside that range, so ranks are level-invariant where raw levels are not.
#
# LEAKAGE NOTE: the rank at day t is computed across users using ONLY day <= t data, so it is
# cross-sectional, never temporal.  This is not the "global rank" README.md forbids -- that
# prohibition is on statistics computed over the full time range.
RANK_CHANNELS = ("rk_gmv30", "rk_ord30", "rk_days30", "rk_recency", "rk_geo3")
N_RANK = len(RANK_CHANNELS)


def _xs_rank(mat: np.ndarray) -> np.ndarray:
    """Per-day cross-sectional rank in [0, 1]: the fraction of users scoring strictly lower.

    Tie-aware by construction -- `searchsorted(..., 'left')` gives every tied user the same
    value.  That matters more than it sounds: ~44% of users have zero 30-day GMV on any given
    day, and an argsort-based rank would scatter that block arbitrarily across [0, 0.44] and
    feed the network pure noise.
    """
    n, T = mat.shape
    out = np.empty((n, T), np.float32)
    for t in range(T):
        col = mat[:, t]
        out[:, t] = np.searchsorted(np.sort(col), col, side="left") / n
    return out


@dataclass
class SeqPanel:
    """Everything the training loop needs, all as plain numpy."""

    X: np.ndarray          # (n_users, C, n_days)   float16, log1p-transformed
    Y: np.ndarray          # (n_users, n_days)      float32, log1p(sum gmv [t+1, t+30])
    pop: np.ndarray        # (n_users, n_days)      bool, >=1 active day in [t-29, t]
    cs_gmv: np.ndarray     # (n_users, n_days+1)    float64 prefix sum of raw gmv
    users: np.ndarray      # (n_users,)             int64, sorted
    dmin: date
    dmax: date

    @property
    def n_users(self) -> int:
        return self.X.shape[0]

    @property
    def n_days(self) -> int:
        return self.X.shape[2]

    @property
    def n_ch(self) -> int:
        return self.X.shape[1]

    def idx(self, d: date) -> int:
        return (d - self.dmin).days

    def day(self, i: int) -> date:
        from datetime import timedelta
        return self.dmin + timedelta(days=int(i))

    @property
    def last_target_day(self) -> int:
        """Largest t whose full 30-day target window fits inside the history."""
        return self.n_days - 1 - HORIZON

    def wgmv(self, a: int, b: int) -> np.ndarray:
        """Exact sum of raw gmv over day-index window [a, b] inclusive."""
        a = max(a, 0)
        b = min(b, self.n_days - 1)
        if b < a:
            return np.zeros(self.n_users)
        return self.cs_gmv[:, b + 1] - self.cs_gmv[:, a]

    def geo3(self, anchor: int) -> np.ndarray:
        """The naive reference every experiment is scored against (DATA.md §9.1):
        expm1 of the mean of log1p over the last three 30-day blocks."""
        blocks = [self.wgmv(anchor - 29 - 30 * j, anchor - 30 * j) for j in range(3)]
        return np.maximum(np.expm1(np.mean([np.log1p(b) for b in blocks], axis=0)), 0.0)

    def norm_stats(self, t_hi: int) -> tuple[np.ndarray, np.ndarray]:
        """Per-channel mean/std over days [0, t_hi] only.

        README.md: scalers are fit on the train part only.  The split here is temporal,
        so "train part" means days at or before the fold's last training anchor -- never the
        validation window.  Computed on a user subsample; these are normalisation constants,
        not estimates anyone acts on, and the subsample keeps the pass cheap.
        """
        sub = self.X[::7, :, : t_hi + 1].astype(np.float32)      # ~36k users
        mu = sub.mean(axis=(0, 2))
        sd = sub.std(axis=(0, 2))
        return mu, np.maximum(sd, 1e-3)


def _windowed(cs: np.ndarray, n_days: int, W: int) -> np.ndarray:
    """Trailing-window sum over [t-W+1, t] for every t, from a (n_users, n_days+1) prefix sum."""
    lo = np.maximum(np.arange(n_days) - W + 1, 0)
    return cs[:, 1:] - cs[:, lo]


def _since(flag: np.ndarray, n_days: int) -> np.ndarray:
    """Days since the last True in `flag`, evaluated at every t (n_days if never)."""
    out = np.empty(flag.shape, np.float32)
    cur = np.full(flag.shape[0], -(10 ** 6), np.int32)
    idx = np.arange(flag.shape[0])
    for t in range(n_days):
        cur = np.where(flag[:, t], t, cur)
        out[:, t] = np.minimum(t - cur, n_days)
    del idx
    return out


POPIDX_CHANNELS = ("pop_gmv_30",)
N_POPIDX = len(POPIDX_CHANNELS)


def _pop_index(x_gmv_log: np.ndarray, n_days: int, win: int = 30) -> np.ndarray:
    """Population trailing-`win`-day mean of log1p(gmv), one scalar per day.

    WHY THIS CHANNEL. A per-user recurrence structurally cannot compute a cross-sectional
    quantity -- it sees one user. The calendar block is the only user-invariant broadcast this
    project ever gave a sequence model, and on the forward-in-calendar protocol it is worth
    +0.00049 rho (e0215 vs e0216, the largest feature effect measured on that path). But
    e0142 proves day-of-year costs -0.00262 rho at the YEAR boundary, because doy maps
    2026-02 onto 2025-02's LEVEL and the platform level rose from 0.1093 to 0.1773 (+62%)
    between them. This channel carries the same "what regime is this" information with the
    year bug removed: the level is MEASURED at every day, so at the test anchor it reads
    2026's value instead of 2025's.

    log1p FIRST, then average over users, so whales cannot turn a regime index into a whale
    tracker. Causal: day t uses days [t-win+1, t] only, with the clipped denominator the rest
    of the pipeline uses.
    """
    day = x_gmv_log.mean(axis=0, dtype=np.float32).astype(np.float64)      # (n_days,)
    cs = np.concatenate([[0.0], np.cumsum(day)])
    t = np.arange(n_days)
    lo = np.maximum(0, t + 1 - win)
    return ((cs[t + 1] - cs[lo]) / (t + 1 - lo)).astype(np.float32)


def build_seq_panel(path: Path = TRAIN, verbose: bool = True, derived: bool = False,
                    ranks: bool = False, popidx: bool = False) -> SeqPanel:
    df = pl.read_parquet(path)
    dmin: date = df["event_date"].min()
    dmax: date = df["event_date"].max()
    n_days = (dmax - dmin).days + 1
    users = np.sort(df["user_id"].unique().to_numpy())
    n_users = users.size

    ui = np.searchsorted(users, df["user_id"].to_numpy())
    di = (df["event_date"].to_numpy().astype("datetime64[D]")
          - np.datetime64(dmin)).astype(np.int32)

    # DATA.md §3 measured 0 duplicate (user_id, date) pairs, so scatter-assign is exact and
    # ~20x faster than np.add.at.  Assert it rather than trust it: a silent duplicate would
    # drop activity instead of summing it.
    flat = ui.astype(np.int64) * n_days + di
    assert np.unique(flat).size == flat.size, "duplicate (user_id, event_date) rows"
    del flat

    RAW_SRC = {"gmv": "gmv", "ord": "to_ord", "cart": "to_cart", "srch": "searches"}
    want_raw = set(RAW_SRC.values()) if (derived or ranks) else {"gmv", "to_ord"}
    raw: dict[str, np.ndarray] = {}

    X = np.zeros((n_users, C_IN, n_days), dtype=np.float16)
    gmv_raw = None
    presence = None
    for c, (name, src, how) in enumerate(CHANNELS):
        A = np.zeros((n_users, n_days), dtype=np.float32)
        if src == "__row__":
            A[ui, di] = 1.0
            presence = A > 0
        else:
            v = df[src].to_numpy().astype(np.float32)
            A[ui, di] = v
            if src in want_raw and src not in raw:
                raw[src] = A.copy()
            if src == "gmv" and gmv_raw is None:
                gmv_raw = A.copy()
            if how == "log":
                np.log1p(A, out=A)
            elif how == "flag":
                A = (A > 0).astype(np.float32)
        X[:, c, :] = A.astype(np.float16)
        del A

    # target: log1p(sum gmv over [t+1, t+30]).  float64 cumsum -- a user's lifetime GMV
    # reaches ~1e5 and float32 would round the differences that make up small windows.
    cs = np.concatenate([np.zeros((n_users, 1)), np.cumsum(gmv_raw, axis=1, dtype=np.float64)],
                        axis=1)
    Y = np.zeros((n_users, n_days), dtype=np.float32)
    hi = n_days - HORIZON                       # t may run to n_days-1-HORIZON inclusive
    Y[:, :hi] = np.log1p(cs[:, 1 + HORIZON: 1 + HORIZON + hi] - cs[:, 1:1 + hi]).astype(np.float32)
    del gmv_raw

    # population rule, evaluated at every day: >=1 active day in [t-29, t].  This is the
    # frozen fold rule (data/fold_spec.json) applied to all t, so training positions are drawn
    # from exactly the distribution the folds score on.
    csp = np.concatenate([np.zeros((n_users, 1), np.int32),
                          np.cumsum(presence, axis=1, dtype=np.int32)], axis=1)
    lo = np.maximum(np.arange(n_days) - 29, 0)
    pop = (csp[:, 1:] - csp[:, lo]) > 0
    del csp

    if derived:
        D = np.zeros((n_users, N_DERIVED, n_days), dtype=np.float16)
        t = np.arange(n_days)
        j = 0
        for cname in DERIVED_COLS:
            csc = np.concatenate([np.zeros((n_users, 1)),
                                  np.cumsum(raw[RAW_SRC[cname]], axis=1, dtype=np.float64)], axis=1)
            for W in DERIVED_WINDOWS:
                D[:, j, :] = np.log1p(_windowed(csc, n_days, W)).astype(np.float16); j += 1
            del csc
        buy = raw["gmv"] > 0
        for flag in (presence, buy):
            csf = np.concatenate([np.zeros((n_users, 1)),
                                  np.cumsum(flag, axis=1, dtype=np.float64)], axis=1)
            for W in DERIVED_WINDOWS:
                D[:, j, :] = np.log1p(_windowed(csf, n_days, W)).astype(np.float16); j += 1
            del csf
        D[:, j, :] = np.log1p(_since(presence, n_days)).astype(np.float16); j += 1
        D[:, j, :] = np.log1p(_since(buy, n_days)).astype(np.float16); j += 1
        # geo3, the naive baseline itself as a channel: mean of log1p over three 30-day blocks.
        # Blocks that fall entirely before day 0 contribute 0 (= log1p of an empty window),
        # rather than the clamped garbage a bare max(.,0) would give at t < 60.
        g = np.zeros((n_users, n_days), np.float64)
        for b in range(3):
            hi_b, lo_b = t - 30 * b, t - 29 - 30 * b
            ok = hi_b >= 0
            blk = (cs[:, np.maximum(hi_b, 0) + 1] - cs[:, np.maximum(lo_b, 0)]) * ok
            g += np.log1p(blk)
        D[:, j, :] = (g / 3.0).astype(np.float16); j += 1
        assert j == N_DERIVED, f"built {j} derived channels, expected {N_DERIVED}"
        X = np.concatenate([X, D], axis=1)
        del D, g

    if ranks:
        R = np.zeros((n_users, N_RANK, n_days), dtype=np.float16)
        t = np.arange(n_days)
        w30 = lambda m: _windowed(                                              # noqa: E731
            np.concatenate([np.zeros((n_users, 1)), np.cumsum(m, axis=1, dtype=np.float64)],
                           axis=1), n_days, 30)
        g3 = np.zeros((n_users, n_days), np.float64)
        for b in range(3):
            hi_b, lo_b = t - 30 * b, t - 29 - 30 * b
            g3 += np.log1p((cs[:, np.maximum(hi_b, 0) + 1] - cs[:, np.maximum(lo_b, 0)])
                           * (hi_b >= 0))
        mats = [w30(raw["gmv"]), w30(raw["to_ord"]), w30(presence.astype(np.float32)),
                -_since(presence, n_days), g3 / 3.0]
        for c, m in enumerate(mats):
            R[:, c, :] = _xs_rank(np.ascontiguousarray(m, dtype=np.float32)).astype(np.float16)
        X = np.concatenate([X, R], axis=1)
        del R, mats, g3
    if popidx:
        gi_ = [n for n, _, _ in CHANNELS].index("gmv")
        # NB: NOT named `pop` -- that is the population MASK this function returns
        # (line 316, SeqPanel.pop). Shadowing it silently replaced a (n_users, n_days) bool
        # mask with this 1-D vector and killed e0242 at the first indexing of popg.
        pop_idx = _pop_index(X[:, gi_, :], n_days)                 # (n_days,) float32
        P = np.broadcast_to(pop_idx, (n_users, N_POPIDX, n_days)).astype(np.float16)
        X = np.concatenate([X, P], axis=1)
        del P
    if derived:
        del buy
    del presence, raw

    if verbose:
        print(f"  SeqPanel: {n_users:,} users x {X.shape[1]} channels x {n_days} days "
              f"({dmin} .. {dmax})"
              + f"  [{C_IN} base"
              + (f" + {N_DERIVED} derived" if derived else "")
              + (f" + {N_RANK} rank" if ranks else "")
              + (f" + {N_POPIDX} popidx" if popidx else "") + "]",
              flush=True)
        print(f"            X {X.nbytes / 1e9:.2f} GB fp16 | Y {Y.nbytes / 1e6:.0f} MB | "
              f"pop {pop.mean():.3f} dense", flush=True)
    return SeqPanel(X=X, Y=Y, pop=pop, cs_gmv=cs, users=users, dmin=dmin, dmax=dmax)


def _self_test() -> None:
    """Cross-check the tensor against src/data.py's independently-built prefix panel."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from data import Panel

    sp = build_seq_panel(derived=True, ranks=True)
    p = Panel()
    assert sp.n_ch == C_IN + N_DERIVED + N_RANK

    assert np.array_equal(sp.users, p.users), "user axis differs from data.Panel"
    assert sp.n_days == p.n_days

    rng = np.random.default_rng(0)
    for a in (168, 288, 378):                       # 2025-06-18, 2025-10-16, last valid t
        ref = np.log1p(p.target(a, HORIZON))
        assert np.allclose(sp.Y[:, a], ref, atol=1e-4), f"target mismatch at t={a}"
        ref_pop = p.active_in(a - 29, a)
        assert np.array_equal(sp.pop[:, a], ref_pop), f"population mismatch at t={a}"

    # channel spot-checks against window sums
    gi = [n for n, _, _ in CHANNELS].index("gmv")
    oi = [n for n, _, _ in CHANNELS].index("ord")
    for _ in range(5):
        u = int(rng.integers(0, sp.n_users))
        a, b = 100, 200
        got = float(np.expm1(sp.X[u, gi, a:b + 1].astype(np.float64)).sum())
        want = float(p.wsum("gmv", a, b)[u])
        assert abs(got - want) <= 1e-2 * max(1.0, want), f"gmv channel {got} vs {want}"
        got = float(np.expm1(sp.X[u, oi, a:b + 1].astype(np.float64)).sum())
        want = float(p.wsum("ord", a, b)[u])
        assert abs(got - want) <= 1e-2 * max(1.0, want), f"ord channel {got} vs {want}"

    # geo3 must reproduce run.py's naive reference bit-for-bit, or the `delta` column of the
    # seq experiments would not be comparable to the gbdt ones
    for a in (168, 288):
        blks = [p.wsum("gmv", a - 29 - 30 * j, a - 30 * j) for j in range(3)]
        ref = np.maximum(np.expm1(np.mean([np.log1p(b) for b in blks], axis=0)), 0.0)
        assert np.allclose(sp.geo3(a), ref, rtol=1e-9, atol=1e-9), f"geo3 mismatch at t={a}"

    # Y must be exactly zero where the window runs off the end of the history
    assert (sp.Y[:, sp.last_target_day + 1:] == 0).all()

    # --- derived channels: every one must reproduce the tabular pipeline's own definition ---
    # Everything is compared IN LOG SPACE, which is the quantity the model consumes.  fp16
    # carries ~11 bits of mantissa, so a stored log1p value is good to ~5e-4 relative -- but
    # expm1 amplifies that by (1+x), which for a 365-day window is an absolute error of order
    # one day.  Asserting on the expm1'd value would therefore be testing float16's dynamic
    # range, not our arithmetic.  ATOL 0.01 in log space is ~2x the worst fp16 step at the
    # largest value any of these channels reaches (log1p of lifetime gmv, ~11.5).
    ATOL = 0.01
    dn = derived_channel_names()
    ix = {n: C_IN + i for i, n in enumerate(dn)}
    for a in (95, 168, 288, 378):
        for c in DERIVED_COLS:
            for W in DERIVED_WINDOWS:
                got = sp.X[:, ix[f"{c}_{W}"], a].astype(np.float64)
                want = np.log1p(p.wsum(c, a - W + 1, a))
                assert np.abs(got - want).max() < ATOL, \
                    f"{c}_{W} @t={a}: max |d| = {np.abs(got - want).max():.4f}"
        for W in DERIVED_WINDOWS:
            got = sp.X[:, ix[f"days_{W}"], a].astype(np.float64)
            assert np.abs(got - np.log1p(p.wdays(a - W + 1, a))).max() < ATOL, f"days_{W} @t={a}"
            got = sp.X[:, ix[f"buydays_{W}"], a].astype(np.float64)
            assert np.abs(got - np.log1p(p.wbuy(a - W + 1, a))).max() < ATOL, f"buydays_{W} @t={a}"
        got = sp.X[:, ix["recency"], a].astype(np.float64)
        assert np.abs(got - np.log1p(np.minimum(p.recency(a), sp.n_days))).max() < ATOL, \
            f"recency @t={a}"
        got = sp.X[:, ix["recency_ord"], a].astype(np.float64)
        assert np.abs(got - np.log1p(np.minimum(p.recency_order(a), sp.n_days))).max() < ATOL, \
            f"recency_ord @t={a}"
        # the geo3 channel is the naive baseline in log space: mean log1p of the three blocks
        got = sp.X[:, ix["geo3"], a].astype(np.float64)
        assert np.abs(got - np.log1p(sp.geo3(a))).max() < ATOL, f"geo3 channel @t={a}"
    print(f"  derived channels verified against data.Panel at 4 anchors "
          f"({N_DERIVED} channels x {len(DERIVED_WINDOWS)} windows)")

    # --- rank channels: bounded, tie-consistent, and ordered like the quantity they rank ---
    r0 = C_IN + N_DERIVED
    for a in (95, 288, 378):
        for c, nm in enumerate(RANK_CHANNELS):
            v = sp.X[:, r0 + c, a].astype(np.float64)
            # <= 1.0, not < 1.0: the top rank is (n-1)/n = 0.999996 and fp16's step near 1 is
            # 2^-10, so it stores as exactly 1.0.  That rounding is also why a rank channel
            # carries ~1024 distinct levels rather than 250,000 -- ample for a model that
            # normalises the channel anyway, but it is a real property of the encoding.
            assert v.min() >= 0.0 and v.max() <= 1.0, f"{nm} @t={a}: out of [0,1]"
        # rk_gmv30 must be a monotone function of the thing it ranks, ties included
        g30 = p.wsum("gmv", a - 29, a)
        rk = sp.X[:, r0, a].astype(np.float64)
        o = np.argsort(g30, kind="stable")
        assert np.all(np.diff(rk[o]) >= -2e-3), f"rk_gmv30 @t={a} not monotone in gmv_30"
        zero = g30 <= 0
        assert rk[zero].max() - rk[zero].min() < 2e-3, \
            f"rk_gmv30 @t={a}: {zero.sum():,} tied zeros were not given one rank"
        assert abs(rk[zero].max() - 0.0) < 1e-6, f"rk_gmv30 @t={a}: zeros must rank at 0"
    print(f"  rank channels verified at 3 anchors ({N_RANK} channels, tie-consistent)")
    print("src/seqdata.py: all self-tests passed")


if __name__ == "__main__":
    _self_test()
