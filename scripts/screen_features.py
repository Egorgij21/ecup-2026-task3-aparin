#!/usr/bin/env python
"""
LOCAL feature-informativeness screen (laptop, 8 GB, no GPU).

What this is and is not:
  This is a TIER=SCREEN instrument (CLAUDE.md §4.2). It measures whether a candidate feature
  carries information for the target *beyond what the installed feature set already carries*,
  using incremental log-space correlation (rho) -- the quantity the metric actually is
  (EXPERIMENTS.md §1b). It deliberately does NOT use the frozen folds: a screen may subsample
  and re-split (§4.2). Any candidate that clears the bar here is written to FEATURES.md /
  FEATURES_CAUSAL.md as "requires validation" -- the confirm is the real frozen-fold CV run.

  The anchor (tabular) screen:  users subsampled, Panel built on the subset, the FULL e0049
  feature set built at a strict clean anchor (A <= 2025-10-16), LightGBM fit on a random
  user-split train half, rho vs the held-out half measured; then candidate added alone and the
  SAME split re-measured.  delta_rho is the informative statistic.

  The causal (user-split) screen:  per-user-day feature matrix (e0141's `full` variant) at a
  grid of training days + the candidate as extra channels, LightGBM under a user split, target
  log1p(sum gmv over [t+1, t+30]).  delta on a held-out user fold.  This is the tabular proxy
  for the GRU's information (label the difference honestly; a GRU confirm still required).

Run:
  python3.11 scripts/screen_features.py --screen tabular --n 30000 [--anchor 2025-10-16]
  python3.11 scripts/screen_features.py --screen causal  --n 20000
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

SUBSET = ROOT / "data" / "_screen_subset.parquet"


def log(m: str) -> None:
    print(m, flush=True)


def cap_memory(gb: float) -> None:
    """Abort this process if its RSS exceeds `gb`, via a background watchdog.

    This screen killed an 8 GB laptop once (a full-tensor float32 copy in run_causal_lgb),
    so the ceiling is enforced rather than assumed.  RLIMIT_AS/RLIMIT_DATA are NOT usable
    here: macOS starts them at RLIM_INFINITY and refuses to lower them ("current limit
    exceeds maximum limit"), so the rlimit route silently no-ops -- tested, it does not work.
    A 0.5 s RSS poll that calls os._exit is crude but it actually fires, and it fires while
    the machine is still responsive instead of after it has started swapping.

    os._exit is deliberate: a MemoryError inside numpy can leave the interpreter unable to
    unwind cleanly, and the point is to stop consuming RAM immediately.
    """
    import os
    import threading
    import psutil

    proc = psutil.Process(os.getpid())
    avail = psutil.virtual_memory().available / 1e9
    log(f"  memory watchdog: abort above {gb:.1f} GB RSS ({avail:.1f} GB free now)")
    if avail < gb:
        log(f"  WARNING: only {avail:.1f} GB is free -- the cap is above what the machine has")

    def _watch():
        while True:
            try:
                r = proc.memory_info().rss / 1e9
            except psutil.Error:
                return
            if r > gb:
                print(f"\n!! ABORT: RSS {r:.2f} GB exceeded the {gb:.1f} GB cap "
                      f"-- killed to protect the machine", flush=True)
                os._exit(137)
            time.sleep(0.5)

    threading.Thread(target=_watch, daemon=True).start()


def rss_gb() -> float:
    """Peak resident set size in GB (macOS reports ru_maxrss in bytes, Linux in KiB)."""
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if sys.platform == "darwin" else r * 1024 / 1e9


def make_subset(n_users: int, seed: int = 0) -> Path:
    """Deterministic subsample of users -> a small parquet both entrypoints can read.

    MEMORY: building it reads all 30.6M rows of train.parquet (~2 GB in RAM), so a cached
    subset with a DIFFERENT user count is reused with a warning rather than silently rebuilt
    -- an accidental `--n` mismatch used to trigger the full load on top of a live screen.
    Delete data/_screen_subset.parquet to force a rebuild at a new size.
    """
    import polars as pl
    if SUBSET.exists():
        u = pl.read_parquet(SUBSET, columns=["user_id"])["user_id"].n_unique()
        if u == n_users:
            log(f"  subset {SUBSET.name} already has {u:,} users -- reusing")
        else:
            log(f"  subset {SUBSET.name} has {u:,} users, not {n_users:,} -- REUSING ANYWAY "
                f"(rebuilding loads all 30.6M rows; rm the file to force it)")
        return SUBSET
    log(f"  building subset from train.parquet (loads ~30.6M rows, needs ~2 GB free)")
    df = pl.read_parquet(ROOT / "train.parquet")
    rng = np.random.default_rng(seed)
    users = np.sort(rng.choice(df["user_id"].unique().to_numpy(), size=n_users, replace=False))
    sub = df.filter(pl.col("user_id").is_in(users.tolist()))
    SUBSET.parent.mkdir(exist_ok=True)
    sub.write_parquet(SUBSET)
    log(f"  wrote {SUBSET.name}: {sub.height:,} rows, {n_users:,} users")
    return SUBSET


# ================================================================================== tabular
def screen_tabular(n_users: int, anchor: str, mtrees: int, cands_on: bool = True,
                   decompose: bool = False):
    import lightgbm as lgb
    from data import Panel
    from features import build

    ai_anchor = date.fromisoformat(anchor)
    sub = make_subset(n_users)

    p = Panel(path=sub)
    ai = p.idx(ai_anchor)
    keep = p.active_in(ai - 29, ai)
    log(f"  anchor {anchor} idx {ai}  users {p.n_users}  keep(active) {int(keep.sum()):,}")

    EXISTING = ["base", "counts", "trend", "rank", "visit", "channel", "diff",
                "cumshare", "ewm", "com", "dispersion", "sbcnomoment", "tsfeat",
                "fcast", "funnel"]
    X, names = build(p, ai, keep, EXISTING)
    X = X.astype(np.float64)
    L = np.log1p(p.target(ai, 30))[keep]
    log(f"  built {X.shape[1]} existing features x {X.shape[0]} users")

    # tsfresh-style block: a hand-vectorised port of the library's nonlinear/run/wavelet
    # statistics (see src/features.py block_tsfresh).  Screened as ONE bundle first -- if
    # the bundle wins, decompose into its families (§4.1).
    Xt, namest = build(p, ai, keep, ["tsfresh"])
    Xt = Xt.astype(np.float64)
    log(f"  tsfresh block: {Xt.shape[1]} features x {Xt.shape[0]} users "
        f"({namest[0]} .. {namest[-1]})")

    y_bool = L > 0
    log(f"  zero-share of held-out-style target: {(1 - y_bool.mean()):.3%}  sd_L={L.std():.4f}")

    rng = np.random.default_rng(1)
    tr = rng.random(X.shape[0]) < 0.5
    va = ~tr
    log(f"  split train {tr.sum():,} / val {va.sum():,}")

    params = dict(objective="regression", metric="rmse", learning_rate=0.05,
                  num_leaves=63, min_data_in_leaf=40, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, num_threads=8,
                  verbosity=-1, seed=0)

    def rho(pred):
        return float(np.corrcoef(pred, L[va])[0, 1])

    def fit(dfeats):
        d = lgb.Dataset(dfeats[tr], L[tr])
        v = lgb.Dataset(dfeats[va], L[va], reference=d)
        m = lgb.train(params, d, num_boost_round=mtrees,
                      valid_sets=[v], callbacks=[lgb.early_stopping(80, verbose=False)])
        return m.predict(dfeats[va])

    log(f"\n  baseline: {X.shape[1]} features")
    p_base = rho(fit(X))
    log(f"    baseline rho = {p_base:.5f}")

    # ---- noise control: an i.i.d. column, same row structure as a candidate ----
    # A screen whose informative statistic penalises even a pure-noise column is measuring
    # over-fit capacity of the baseline, not candidate information.  If noise shows <= -1e-3,
    # negative candidate deltas are artifacts and only large POSITIVE deltas mean anything.
    noiz = rng.standard_normal(keep.sum())
    pn = rho(fit(np.column_stack([X, noiz])))
    log(f"\n  noise control:          rho {pn:.5f}  d_rho {pn - p_base:+.5f}  <-- overfit sensitivity")

    # ---- tsfresh bundle: add the whole block to the base set (one bundle screen) ----
    pt = rho(fit(np.column_stack([X, Xt])))
    d_bundle = pt - p_base
    log(f"\n  tsfresh bundle:         rho {pt:.5f}  d_rho {d_bundle:+.5f}"
        f"{'  <<' if d_bundle > 2e-4 else ''}  (screen of the {Xt.shape[1]}-feature block)")

    # Decompose the block into its statistic families and screen each alone (§4.1: a bundle
    # is never evidence as a unit).  Forced when --decompose: a 66-column bundle carries 66x
    # the capacity cost of the 1-column noise control on this overfit-starved screen, so a
    # null BUNDLE delta does not clear the individual 6-column families.
    if d_bundle > abs(pn - p_base) or decompose:
        fams = {}
        for j, nm_ in enumerate(namest):
            fams.setdefault(nm_.split("_gmv_")[0].split("_ord_")[0], []).append(j)
        log(f"  decomposing into {len(fams)} statistic families (6 cols each):")
        for fname, idx in sorted(fams.items()):
            pf = rho(fit(np.column_stack([X, Xt[:, idx]])))
            df = pf - p_base
            log(f"    {fname:44s} rho {pf:.5f}  d_rho {df:+.5f}"
                f"{'  <<' if df > 2e-4 else ''}  ({len(idx)} cols)")
    else:
        log("  bundle <= noise control -> no decomposition (nothing to attribute)")
    del Xt

    if not cands_on:
        return 0

    # ---- candidates + base feature set ----
    cands = build_candidates_tabular(p, ai, keep)
    log(f"\n  {len(cands)} candidate families x base set (each added alone):")
    for cname, C in cands:
        Xc = np.column_stack([X, C.astype(np.float64)])
        pc = rho(fit(Xc))
        d = pc - p_base
        flag = "  <<" if d > 2e-4 else ""
        log(f"    {cname:44s} rho {pc:.5f}  d_rho {d:+.5f}{flag}")

    return 0


# ================================================================================== causal
def screen_causal(n_users: int, mtrees: int, which: str = 'orig'):
    import lightgbm as lgb
    from usercv_features import (Raw, build_features, build_target, build_tmask,
                                 hash_fold, max_anchor)

    sub = make_subset(n_users)
    raw = Raw(path=sub)
    X, names = build_features(raw, "full")
    Y = build_target(raw, "sum")
    last = max_anchor(raw)                                    # T-31: last fully-observed target day
    M = build_tmask(raw, last, burn_in=14, trim_to_first_seen=True)
    fold_of = hash_fold(raw.users)
    log(f"  Raw: {raw.n} users x {raw.T} days, {X.shape[-1]} full-variant features: {names}")

    # train days: stride 7 grid up to the last day with an observed target; eval all scored days
    tr_days = np.arange(89, last + 1, 7)
    days = np.arange(89, last + 1)

    # Build the candidate channels NOW and drop `raw`: the raw panel is ~0.9 GB at n=15k and
    # is only needed to derive them, while X alone is ~1.0 GB.  Holding both plus LightGBM's
    # copy is what took the machine down.
    cands = (build_candidates_tsfresh_causal(raw) if which == 'tsfresh'
             else build_candidates_causal(raw))
    n_u, n_t = raw.n, raw.T                                   # keep the shape, drop the panel
    del raw
    log(f"  candidates built ({len(cands)}), raw panel released -- RSS {rss_gb():.2f} GB peak")
    log(f"\n  baseline: {X.shape[-1]} full-variant features")
    r_base = run_causal_lgb(X, Y, M, fold_of, tr_days, days, mtrees)
    log(f"    baseline unseen-user RMSLE = {r_base:.5f}   peak RSS {rss_gb():.2f} GB")

    # noise control: an i.i.d. per-user-day channel.  If it moves RMSLE materially, the
    # proxy is over-parameterised and even large candidate deltas are capacity artefacts.
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((n_u, n_t)).astype(np.float16)
    r_noise = run_causal_lgb(X, Y, M, fold_of, tr_days, days, mtrees, extra=noise)
    log(f"  noise control:                rmsle {r_noise:.5f}  d_rmsle {r_noise - r_base:+.5f}")
    del noise

    for cname, Ch in cands:
        rc = run_causal_lgb(X, Y, M, fold_of, tr_days, days, mtrees, extra=Ch)
        d = rc - r_base
        flag = "  <<" if d < -5e-4 else ""
        log(f"    {cname:44s} rmsle {rc:.5f}  d_rmsle {d:+.5f}{flag}"
            f"   peak {rss_gb():.2f} GB")
        del Ch
    return 0


def _gather(X, extra, Y, M, users, days):
    """Rows (user, day) selected by M -> (rows, C) float32 + target, gathering BEFORE promotion.

    MEMORY (this function exists only because of an OOM): X is (n_users, T, C) float16, ~1 GB
    at n=15k.  The previous version standardised the WHOLE tensor -- `(X.astype(np.float32)
    - mu) / sd` builds a 2 GB float32 copy plus a 2 GB transient for the astype, while X and
    the concatenated candidate tensor are both still alive.  ~6 GB of peaks on an 8 GB laptop,
    and it killed it.  Only ~10% of user-days are ever scored, so gather the rows first and
    promote the small result: peak is now ~0.2 GB.  `extra` is the candidate channel, gathered
    the same way instead of being concatenated onto the full tensor (which cost another 1 GB).
    """
    xs, ys = [], []
    for t in days:
        r = np.flatnonzero(M[users, t])
        if not r.size:
            continue
        u = users[r]
        blk = X[u, t, :].astype(np.float32)
        if extra is not None:
            blk = np.column_stack([blk, extra[u, t].astype(np.float32)])
        xs.append(blk)
        ys.append(Y[u, t])
    return np.concatenate(xs), np.concatenate(ys)


def run_causal_lgb(X, Y, M, fold_of, tr_days, days, mtrees, extra=None):
    """One user-split LightGBM: holds out fold 0, trains on the rest, scores unseen users."""
    import lightgbm as lgb
    k = 0
    tr_u = np.flatnonzero(fold_of != k)
    va_u = np.flatnonzero(fold_of == k)
    Xtr, ytr = _gather(X, extra, Y, M, tr_u, tr_days)
    Xva, yva = _gather(X, extra, Y, M, va_u, days)
    # standardise from the TRAIN rows only.  (For LightGBM this is a no-op up to float
    # rounding -- bins are quantile-based, so a per-column affine map leaves splits identical
    # -- but it is kept so the numbers stay comparable with the recorded 1.75027 baseline.)
    mu = Xtr.mean(0)
    sd = np.maximum(Xtr.std(0), 1e-3)
    Xtr = (Xtr - mu) / sd
    Xva = (Xva - mu) / sd
    d = lgb.Dataset(Xtr, ytr)
    v = lgb.Dataset(Xva, yva, reference=d)
    params = dict(objective="regression", metric="rmse", learning_rate=0.05,
                  num_leaves=63, min_data_in_leaf=40, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, num_threads=8,
                  verbosity=-1, seed=0)
    m = lgb.train(params, d, num_boost_round=mtrees,
                  valid_sets=[v], callbacks=[lgb.early_stopping(60, verbose=False)])
    pred = m.predict(Xva)
    return float(np.sqrt(np.mean((pred - yva) ** 2)))


# ============================================================================= candidates
def build_candidates_tabular(p, ai, keep) -> list[tuple[str, np.ndarray]]:
    """Candidate features for the anchor (tabular) screen, computed via the Panel API only."""
    out = []
    EPS = 1e-3

    def add(nm, v):
        out.append((nm, np.asarray(v, np.float64)[keep]))

    # ---- 1. cohort-normalised levels: relative to users whose FIRST ACTIVE day is in the
    # same 14-day bucket (tenure-matched).  Anchor levels drift with tenure (a user who joined
    # Oct 2025 and one who joined Jan 2025 have very different lifetime accumulation), and the
    # raw levels mean different things by cohort.  Removing the cohort mean isolates standing
    # within one's own lifecycle stage.
    first = p.first_act.astype(np.int64)                 # (n_users,) first active day
    buck = first // 14                                   # 14-day tenure bucket id
    g30 = p.wsum("gmv", ai - 29, ai)
    g90 = p.wsum("gmv", ai - 89, ai)
    o90 = p.wsum("ord", ai - 89, ai)
    bd90 = p.wbuy(ai - 89, ai)
    d90 = p.wdays(ai - 89, ai)
    for v, nm in [(g30, "gmv30"), (g90, "gmv90"), (o90, "ord90"), (bd90, "buy_days90")]:
        mu = np.full(p.n_users, np.nan)
        for b in np.unique(buck):
            m = buck == b
            mu[m] = v[m].mean()
        add(f"cohort_rel_{nm}", np.log1p(v) - np.log1p(np.where(np.isnan(mu), 0.0, mu)))

    # cohort-relative activity rate (buying frequency vs people who joined at the same time)
    rate = np.divide(bd90, np.maximum(d90, 1))
    mu_rate = np.full(p.n_users, np.nan)
    for b in np.unique(buck):
        m = buck == b
        mu_rate[m] = np.nanmedian(rate[m])
    add("cohort_rel_buy_rate90", rate - np.where(np.isnan(mu_rate), 0.0, mu_rate))

    # ---- 2. weekday / weekend spend shape.  The 665-feature set has NO day-of-week anything
    # (no dow, no weekend) -- only global weekly-power in tsfeat (a user-agnostic autocorr).
    # Whether a user shops on weekends vs workdays is a purchasing-pattern trait orthogonal to
    # level.  Computed from the raw daily GMV matrix (requires the truth day-of-week).
    t = np.arange(p.n_days)
    dow = np.array([p.day(i).weekday() for i in t])       # 0=Mon..6=Sun
    week = (dow >= 5).astype(np.float64)                  # Sat/Sun
    raw = p.raw["gmv"]                                     # (n_users, n_days_cut?) raw daily
    # raw is (n_users, n_days) -- but built on the SUBSAMPLE panel so rows == subsample
    W = week[ai - 89: ai + 1]
    Rw = raw[:, ai - 89: ai + 1]
    g = Rw.sum(1)
    gw = (Rw * W[None]).sum(1)
    add("weekend_gmv_share_90", np.divide(gw, np.maximum(g, EPS)))

    # day-of-week variety: how many distinct weekdays GMV falls on (entropy-ish).  1 if all
    # spend on one weekday, 7 if perfectly spread.
    V = np.zeros(p.n_users)
    submat = raw[:, ai - 89: ai + 1]
    for w_ in range(7):
        sel = dow[ai - 89: ai + 1] == w_
        V += (submat[:, sel].sum(1) > 0)
    add("distinct_dow_with_buy_90", V)

    # ---- 3. recency of cart / search (a second decay channel).  The installed set has only
    # activity and order recency.  Someone who added to cart yesterday but never ordered is a
    # distinct pre-purchase-intent state, and the cs prefix sums span the whole subsample db,
    # so a running-last-day index is rebuilt from the dense daily matrix below.
    P_raw = p.raw  # {gmv, ord} only -- reconstruct cart/srch recency from the source
    import polars as _pl
    _df = _pl.read_parquet(SUBSET).sort("user_id", "event_date")  # subsample rows
    _ui = np.searchsorted(p.users, _df["user_id"].to_numpy())
    _di = (_df["event_date"].to_numpy().astype("datetime64[D]")
           - np.datetime64(p.dmin)).astype(np.int32)
    for col, rc in (("to_cart", "cart"), ("searches", "srch")):
        _A = np.zeros((p.n_users, p.n_days), np.float32)
        _A[_ui, _di] = _df[col].to_numpy().astype(np.float32)
        _last = np.full(p.n_users, -1, np.int32)
        for t in range(ai + 1):                            # scan only days <= ai: causal
            _last = np.where(_A[:, t] > 0, t, _last)
        _r = np.where(_last < 0, p.n_days, ai - _last)   # capped at n_days == tenure sentinel
        add(f"recency_{rc}", _r)
    del _A, _last

    # ---- 4. undelivered cart backlog: how much is sitting in the basket unconverted.
    cart_w7 = p.wsum("cart", ai - 6, ai)
    ord_w7 = p.wsum("ord", ai - 6, ai)
    add("cart_backlog_7", np.maximum(cart_w7 - ord_w7, 0.0))
    add("cart_to_ord_ratio_l7", np.divide(cart_w7, np.maximum(ord_w7, EPS)))

    # ---- 5. age-aligned lifecycle buckets: the user's OWN 30-day age buckets since first
    # active (calendar-free).  The existing gmv_blk are calendar-anchored, which for a recent
    # joiner is "their first month" and for an old user "their 10th+".  Age-aligning removes
    # that confound.  gmv + buy-day share per age bucket 0..3 (first 120 days after joining).
    # CAUSALITY: only days <= ai may enter the numerator AND the denominator.  A user whose
    # first_act sits < 30 days before the anchor has bucket-0 ages that extend past ai into
    # the target window -- summing the unbounded age mask would read future GMV into the
    # feature (the same look-ahead class that cost the repo a day, DATA.md §10).  The mask
    # caps the age span at days <= ai; the "lifetime" denominator is likewise cut at ai.
    first = p.first_act.astype(np.int64)                 # (n,) first active day
    t = np.arange(p.n_days)[:, None]
    age = t - first[None]                                # (n_days, n) days since first active
    past = (t <= ai)                                     # (n_days, n) row mask: no future days
    rawA = p.raw["gmv"]                                  # (n, n_days) daily GMV
    life = rawA[:, : ai + 1].sum(1)                      # lifetime GMV seen at the anchor
    for k, (lo_, hi_) in enumerate([(0, 30), (30, 60), (60, 90), (90, 120)]):
        sel = (age >= lo_) & (age < hi_) & past           # (n_days, n), future days zeroed
        gk = (rawA.T * sel).sum(0)                        # (n,) causal GMV in this age bucket
        add(f"age_bucket_gmv_{k}", gk)
        add(f"age_bucket_gmv_share_{k}", np.divide(gk, np.maximum(life, EPS)))
    return out


def build_candidates_causal(raw) -> list[tuple[str, np.ndarray]]:
    """Candidate daily channels for the user-split (causal GRU) screen.

    The e0141 GRU runs the `full` variant: 16 raw daily columns, window means, flag-rate
    windows, active + active-window means -- NO recency, NO composition.  Every prior
    engineered ADDITION hurt (e0110 derived +0.00137, e0114 rank channels lost, e0142's 143
    features were the worst model since e0001).  The channel classes the installed `full` set
    lacks and that a recurrent model might genuinely lack are (a) RECENCY channels -- the GRU
    must otherwise churn a long history to discover "days since X"; (b) COMPOSITIONS the raw
    fields only expose arithmetically (AOV, conversion, cart backlog stock).

    Measurement is the tabular proxy for the GRU's information: same per-user-day features,
    LightGBM under a user split.  Only channels with measured incremental signal belong in
    FEATURES_CAUSAL.md, and even then the GRU itself must confirm -- the GRU is not a stack of
    fixed windows and e0110/e0142 prove that surplus channels are *worse* than nothing.
    """
    out = []
    EPS = 1e-6
    cart, ordr, gmv = raw.col["to_cart"], raw.col["to_ord"], raw.col["gmv"]
    srch = raw.col["searches"]
    active = raw.active

    def _ds(flag: np.ndarray) -> np.ndarray:
        """days since last True, per (user, day); raw.T = 'never' sentinel."""
        T = raw.T
        t = np.arange(T, dtype=np.int32)[None, :]
        last = np.maximum.accumulate(np.where(flag > 0, t, -1), axis=1)
        return np.where(last < 0, T, t - last).astype(np.float32)

    # (a) recency channels
    out.append(("ds_cart", _ds(cart)))
    out.append(("ds_order", _ds(ordr)))
    out.append(("ds_active", _ds(active)))
    # (b) compositions -- per-day, future-blind arithmetic the recurrence must discover
    out.append(("aov_daily", np.clip(np.divide(gmv, ordr + EPS), 0.0, 5e4).astype(np.float32)))
    out.append(("conv_daily", np.clip(np.divide(ordr, cart + EPS), 0.0, 1.5).astype(np.float32)))
    out.append(("cart_per_srch_d", np.clip(np.divide(cart, srch + EPS), 0.0, 20.0).astype(np.float32)))
    # (c) undelivered-selection stock (cumulative cart - cumulative orders -> running backlog)
    cscr = np.concatenate([np.zeros((raw.n, 1)),
                           np.cumsum(cart, 1, dtype=np.float64)], 1)
    csor = np.concatenate([np.zeros((raw.n, 1)),
                           np.cumsum(ordr, 1, dtype=np.float64)], 1)
    back = (cscr[:, 1:] - csor[:, 1:]).astype(np.float32)
    out.append(("cart_backlog_running", back))
    return out


def build_candidates_tsfresh_causal(raw) -> list[tuple[str, np.ndarray]]:
    """tsfresh nonlinear statistics as ROLLING per-user-day channels (30-day trailing window).

    Only the statistics that survive as cheap matrix ops at (n_users x T) resolution are
    here: c3 and time_reversal_asymmetry are rolling MEANS of a product series, and ARCH is a
    rolling covariance of the squared series -- all three are cumsum-able.  (lempel_ziv and
    longest_strike are per-(user,day) loops = 6M Python iterations at screen size, and their
    tabular screen was nil, so they are not worth the compute.)

    CAUSALITY: the product series P[s] = x[s]x[s+1]x[s+2] is only KNOWN at day s+2, so the
    trailing window at day t runs over s in [t-w+1, t-2], never s > t-2.  Getting this wrong
    is the same look-ahead class that faked +0.006 rho in the tabular screen.

    MEMORY: each channel is float32 (n, T) ~25 MB at n=15k, and the float64 temporaries are
    freed as they are consumed -- the ARCH branch in particular used to hold gmv, S, S**2, mu,
    m2, PX and cr live simultaneously (~7 x 50 MB float64), which is fine alone but not on top
    of a 1 GB feature tensor.
    """
    out = []
    W = 30
    gmv = raw.col["gmv"].astype(np.float64)
    n, T = gmv.shape

    def _roll_mean_delayed(P: np.ndarray, delay: int, w: int) -> np.ndarray:
        """Trailing-w mean at each day t of a series whose element s is known only at s+delay."""
        cs = np.concatenate([np.zeros((n, 1)), np.cumsum(P, 1)], 1)      # cs[:, i] = sum P[0..i-1]
        t = np.arange(T)
        a = np.maximum(t - w + 1, 0)                                     # window start
        b = t - delay                                                    # last KNOWN index
        s = cs[:, np.maximum(b + 1, 0)] - cs[:, a]
        cnt = np.maximum(b - a + 1, 0)
        return (s / np.maximum(cnt, 1)).astype(np.float32)

    # c3 (lag 1): mean(x[s] x[s+1] x[s+2]) -- nonlinearity, known at s+2
    P3 = np.zeros_like(gmv)
    P3[:, : T - 2] = gmv[:, : T - 2] * gmv[:, 1:T - 1] * gmv[:, 2:]
    out.append(("tf_c3_roll30", np.log1p(_roll_mean_delayed(P3, 2, W))))
    del P3

    # time reversal asymmetry: mean(x[s+2]^2 x[s+1] - x[s+1] x[s]^2), known at s+2
    PT = np.zeros_like(gmv)
    PT[:, : T - 2] = (gmv[:, 2:] ** 2 * gmv[:, 1:T - 1]
                      - gmv[:, 1:T - 1] * gmv[:, : T - 2] ** 2)
    tr = _roll_mean_delayed(PT, 2, W)
    del PT
    out.append(("tf_timerev_roll30", np.sign(tr) * np.log1p(np.abs(tr))))
    del tr

    # ARCH lag 7: rolling autocorrelation of the SQUARED series (volatility clustering).
    # cross term x[s]^2 x[s+7]^2 is known at s+7.
    S = gmv ** 2
    del gmv
    mu = _roll_mean_delayed(S, 0, W).astype(np.float64)
    m2 = _roll_mean_delayed(S ** 2, 0, W).astype(np.float64)
    PX = np.zeros_like(S)
    PX[:, : T - 7] = S[:, : T - 7] * S[:, 7:]
    del S
    cr = _roll_mean_delayed(PX, 7, W).astype(np.float64)
    del PX
    var = np.maximum(m2 - mu ** 2, 0.0)
    del m2
    out.append(("tf_arch7_roll30",
                np.where(var > 1e-9, (cr - mu ** 2) / np.maximum(var, 1e-9), 0.0
                         ).astype(np.float32)))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", required=True, choices=["tabular", "causal"])
    ap.add_argument("--n", type=int, default=25000)
    ap.add_argument("--anchor", default="2025-10-16")
    ap.add_argument("--mtrees", type=int, default=800, help="max boosting rounds to try")
    ap.add_argument("--max-gb", type=float, default=5.0,
                    help="hard address-space cap; exceeding it raises MemoryError")
    ap.add_argument("--causal-cands", default="orig", choices=["orig", "tsfresh"],
                    help="causal: which candidate channel set to screen")
    ap.add_argument("--decompose", action="store_true",
                    help="tabular: always screen the tsfresh families individually")
    ap.add_argument("--no-cands", action="store_true",
                    help="tabular: baseline + noise + tsfresh only (skip the e0189 candidates)")
    args = ap.parse_args()
    cap_memory(args.max_gb)
    if args.screen == "tabular":
        raise SystemExit(screen_tabular(args.n, args.anchor, args.mtrees,
                                        not args.no_cands, args.decompose))
    else:
        raise SystemExit(screen_causal(args.n, args.mtrees, args.causal_cands))
