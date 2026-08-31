#!/usr/bin/env python
"""
E-CUP 2026 / Task 3 (Search LTV) -- full exploratory data analysis.

Runs on the QMUL `compute` partition (needs ~40 GB RAM, no GPU).
Everything is printed to stdout (-> the slurm .out log) and mirrored into
reports/eda/*.json so that DATA.md can be written from hard numbers only.

Stages
  1  schema / integrity / column semantics / invariants
  2  temporal structure (daily panel, seasonality, calendar events)
  3  user panel (sparsity, tenure, recency, submission-set match)
  4  target construction at rolling anchors + naive-baseline scoreboard
  5  predictability: autocorrelation, zero-inflation conditioning,
     RMSLE decomposition, tie-breaker metrics (Gini / total-GMV RMSPE)
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path("/path/to/ecup")
DATA = ROOT / "data" / "train.parquet"
SAMPLE_SUB = ROOT / "data" / "sample_submit.csv"
OUT = ROOT / "reports" / "eda"
FIGS = ROOT / "reports" / "figs"
OUT.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

REPORT: dict = {}
T0 = time.time()


def hdr(txt: str) -> None:
    print(f"\n{'=' * 78}\n{txt}   [t+{time.time() - T0:.0f}s]\n{'=' * 78}", flush=True)


def sub(txt: str) -> None:
    print(f"\n--- {txt} " + "-" * max(0, 70 - len(txt)), flush=True)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (date,)):
        return o.isoformat()
    if isinstance(o, (np.ndarray,)):
        return jsonable(o.tolist())
    return o


def dump(name: str, obj) -> None:
    with open(OUT / f"{name}.json", "w") as f:
        json.dump(jsonable(obj), f, indent=2, default=str)


def q(a: np.ndarray, qs=(0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 0.9999, 1)) -> dict:
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return {}
    return {f"p{x * 100:g}": float(np.quantile(a, x)) for x in qs}


def gini(y: np.ndarray) -> float:
    """Gini over predictions/values (concentration of GMV across users)."""
    y = np.sort(np.asarray(y, dtype=np.float64))
    n = y.size
    s = y.sum()
    if s <= 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * y).sum()) / (n * s) - (n + 1) / n)


def rmsle(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.log1p(np.clip(y, 0, None)) - np.log1p(np.clip(p, 0, None))) ** 2)))


# ============================================================================
# STAGE 1 -- schema, integrity, column semantics
# ============================================================================
hdr("STAGE 1 -- SCHEMA / INTEGRITY / COLUMN SEMANTICS")

df = pl.read_parquet(DATA)
n_rows = df.height
print(f"shape            : {df.shape}")
print(f"est. memory      : {df.estimated_size('gb'):.2f} GB (int64/float64 as stored)")

sub("dtypes")
for c, t in zip(df.columns, df.dtypes):
    print(f"  {c:22s} {t}")

sub("null counts")
nulls = df.null_count().row(0)
print({c: n for c, n in zip(df.columns, nulls)})

sub("per-column summary (min / max / mean / #distinct / #zeros / #negatives)")
col_stats = {}
num_cols = [c for c in df.columns if c != "event_date"]
for c in num_cols:
    s = df[c]
    st = {
        "dtype": str(s.dtype),
        "min": s.min(),
        "max": s.max(),
        "mean": float(s.mean()),
        "n_distinct": int(s.n_unique()),
        "n_zero": int((s == 0).sum()),
        "n_neg": int((s < 0).sum()),
        "n_null": int(s.null_count()),
    }
    col_stats[c] = st
    print(
        f"  {c:22s} min={st['min']!s:>12} max={st['max']!s:>18} mean={st['mean']:>12.4f} "
        f"distinct={st['n_distinct']:>9} zeros={st['n_zero']:>10} neg={st['n_neg']:>8}"
    )
REPORT["col_stats"] = col_stats

sub("value counts of low-cardinality columns")
low_card = {}
for c in num_cols:
    if col_stats[c]["n_distinct"] <= 30:
        vc = df[c].value_counts().sort(c)
        d = {str(r[0]): int(r[1]) for r in vc.iter_rows()}
        low_card[c] = d
        print(f"  {c:22s} {d}")
REPORT["low_cardinality"] = low_card

sub("key uniqueness")
n_users = df["user_id"].n_unique()
n_dates = df["event_date"].n_unique()
n_pairs = df.select(["user_id", "event_date"]).n_unique()
print(f"  n_unique user_id            : {n_users:,}")
print(f"  n_unique event_date         : {n_dates:,}")
print(f"  n_unique (user_id, date)    : {n_pairs:,}")
print(f"  rows                        : {n_rows:,}")
print(f"  duplicated (user,date) rows : {n_rows - n_pairs:,}")
print(f"  full dense panel would be   : {n_users * n_dates:,} rows "
      f"(density = {100 * n_rows / (n_users * n_dates):.2f}%)")
REPORT["keys"] = {
    "n_rows": n_rows, "n_users": n_users, "n_dates": n_dates,
    "n_user_date_pairs": n_pairs, "n_dup_rows": n_rows - n_pairs,
    "dense_panel_rows": n_users * n_dates,
    "density_pct": 100 * n_rows / (n_users * n_dates),
}

sub("column-identity checks (are the aggregates internally consistent?)")
ident = {}
checks = [
    ("gmv == gmv_search + gmv_cat", (pl.col("gmv") - pl.col("gmv_search") - pl.col("gmv_cat")).abs() > 1e-6),
    ("to_cart == search_to_cart + cat_to_cart", pl.col("to_cart") != pl.col("search_to_cart") + pl.col("cat_to_cart")),
    ("to_ord  == search_to_ord + cat_to_ord", pl.col("to_ord") != pl.col("search_to_ord") + pl.col("cat_to_ord")),
    ("has_search_to_cart == (search_to_cart>0)", pl.col("has_search_to_cart") != (pl.col("search_to_cart") > 0).cast(pl.Int64)),
    ("has_search_to_ord  == (search_to_ord>0)", pl.col("has_search_to_ord") != (pl.col("search_to_ord") > 0).cast(pl.Int64)),
    ("has_cat_to_cart    == (cat_to_cart>0)", pl.col("has_cat_to_cart") != (pl.col("cat_to_cart") > 0).cast(pl.Int64)),
    ("has_cat_to_ord     == (cat_to_ord>0)", pl.col("has_cat_to_ord") != (pl.col("cat_to_ord") > 0).cast(pl.Int64)),
    ("search >= has_search_to_cart", pl.col("search") < pl.col("has_search_to_cart")),
    ("searches >= search", pl.col("searches") < pl.col("search")),
    ("gmv_search>0 => search_to_ord>0", (pl.col("gmv_search") > 0) & (pl.col("search_to_ord") <= 0)),
    ("gmv_cat>0    => cat_to_ord>0", (pl.col("gmv_cat") > 0) & (pl.col("cat_to_ord") <= 0)),
    ("search_to_ord>0 => gmv_search>0", (pl.col("search_to_ord") > 0) & (pl.col("gmv_search") <= 0)),
    ("cat_to_ord>0    => gmv_cat>0", (pl.col("cat_to_ord") > 0) & (pl.col("gmv_cat") <= 0)),
    ("to_ord>0 => gmv>0", (pl.col("to_ord") > 0) & (pl.col("gmv") <= 0)),
    ("gmv>0 => to_ord>0", (pl.col("gmv") > 0) & (pl.col("to_ord") <= 0)),
    ("to_ord <= to_cart", pl.col("to_ord") > pl.col("to_cart")),
]
for name, viol_expr in checks:
    v = int(df.select(viol_expr.sum()).item())
    ident[name] = v
    flag = "OK " if v == 0 else "VIOL"
    print(f"  [{flag}] {name:46s} violations = {v:,} ({100 * v / n_rows:.4f}%)")
REPORT["identities"] = ident

sub("all-zero-activity rows (is the series really thinned?)")
act_cols = ["search", "cat", "to_cart", "to_ord", "searches", "gmv"]
allzero = int(df.select((sum((pl.col(c) == 0).cast(pl.Int8) for c in act_cols) == len(act_cols)).sum()).item())
print(f"  rows with search=cat=to_cart=to_ord=searches=gmv=0 : {allzero:,} ({100 * allzero / n_rows:.4f}%)")
REPORT["all_zero_rows"] = allzero

sub("row 'type' composition")
comp = (
    df.select(
        (pl.col("gmv") > 0).alias("has_gmv"),
        (pl.col("to_cart") > 0).alias("has_cart"),
        (pl.col("searches") > 0).alias("has_searches"),
        (pl.col("search") > 0).alias("has_search_sess"),
        (pl.col("cat") > 0).alias("has_cat_sess"),
    )
    .group_by(["has_gmv", "has_cart", "has_searches", "has_search_sess", "has_cat_sess"])
    .len()
    .sort("len", descending=True)
)
print(comp.head(20))
REPORT["row_composition"] = comp.head(25).to_dicts()

sub("gmv value granularity")
gmv_pos = df.filter(pl.col("gmv") > 0)["gmv"].to_numpy()
print(f"  positive-gmv rows : {gmv_pos.size:,}")
print(f"  quantiles         : {json.dumps({k: round(v, 3) for k, v in q(gmv_pos).items()})}")
frac = gmv_pos - np.floor(gmv_pos)
print(f"  share integer-valued gmv       : {100 * np.mean(frac < 1e-9):.3f}%")
print(f"  share 2-dp-rounded gmv         : {100 * np.mean(np.abs(gmv_pos * 100 - np.round(gmv_pos * 100)) < 1e-6):.3f}%")
print(f"  n distinct positive gmv values : {np.unique(gmv_pos).size:,}")
print(f"  smallest 10 positive values    : {np.unique(gmv_pos)[:10]}")
REPORT["gmv_row_level"] = {
    "n_pos_rows": int(gmv_pos.size), "quantiles": q(gmv_pos),
    "share_integer": float(np.mean(frac < 1e-9)),
    "share_2dp": float(np.mean(np.abs(gmv_pos * 100 - np.round(gmv_pos * 100)) < 1e-6)),
    "n_distinct_pos": int(np.unique(gmv_pos).size),
}

sub("average order value (gmv / to_ord on rows with orders)")
aov = df.filter(pl.col("to_ord") > 0).select((pl.col("gmv") / pl.col("to_ord")).alias("aov"))["aov"].to_numpy()
print(f"  {json.dumps({k: round(v, 2) for k, v in q(aov).items()})}")
REPORT["aov"] = q(aov)

sub("channel split of GMV (search vs catalog)")
tot = df.select(pl.col("gmv").sum(), pl.col("gmv_search").sum(), pl.col("gmv_cat").sum()).row(0)
print(f"  total gmv={tot[0]:,.0f}  search={tot[1]:,.0f} ({100 * tot[1] / tot[0]:.2f}%)  "
      f"cat={tot[2]:,.0f} ({100 * tot[2] / tot[0]:.2f}%)")
REPORT["channel_split"] = {"gmv": tot[0], "gmv_search": tot[1], "gmv_cat": tot[2]}

sub("pairwise correlation of row-level columns (Pearson, on log1p for heavy tails)")
corr_cols = ["search", "cat", "searches", "to_cart", "to_ord", "gmv", "gmv_search", "gmv_cat",
             "search_to_cart", "search_to_ord", "cat_to_cart", "cat_to_ord"]
M = np.log1p(np.clip(df.select(corr_cols).to_numpy().astype(np.float64), 0, None))
C = np.corrcoef(M, rowvar=False)
print("      " + " ".join(f"{c[:7]:>8s}" for c in corr_cols))
for i, c in enumerate(corr_cols):
    print(f"  {c[:5]:5s} " + " ".join(f"{C[i, j]:8.3f}" for j in range(len(corr_cols))))
REPORT["row_corr_log1p"] = {"cols": corr_cols, "matrix": C}
del M, C

dump("stage1_schema", REPORT)


# ============================================================================
# STAGE 2 -- temporal structure
# ============================================================================
hdr("STAGE 2 -- TEMPORAL STRUCTURE")

dmin, dmax = df["event_date"].min(), df["event_date"].max()
span = (dmax - dmin).days + 1
print(f"  date range : {dmin} .. {dmax}   ({span} calendar days, {n_dates} distinct present)")
print(f"  missing calendar days : {span - n_dates}")
REPORT["dates"] = {"min": dmin, "max": dmax, "span_days": span, "n_present": n_dates}

daily = (
    df.group_by("event_date")
    .agg(
        pl.len().alias("n_rows"),
        pl.col("user_id").n_unique().alias("n_active_users"),
        (pl.col("gmv") > 0).sum().alias("n_buyers"),
        pl.col("gmv").sum().alias("gmv"),
        pl.col("gmv_search").sum().alias("gmv_search"),
        pl.col("gmv_cat").sum().alias("gmv_cat"),
        pl.col("to_ord").sum().alias("to_ord"),
        pl.col("to_cart").sum().alias("to_cart"),
        pl.col("searches").sum().alias("searches"),
        pl.col("search").sum().alias("search_sess"),
        pl.col("cat").sum().alias("cat_sess"),
    )
    .sort("event_date")
)
daily = daily.with_columns(
    pl.col("event_date").dt.weekday().alias("dow"),
    pl.col("event_date").dt.month().alias("month"),
    (pl.col("gmv") / pl.col("n_active_users")).alias("gmv_per_active"),
    (pl.col("gmv") / pl.col("n_buyers")).alias("gmv_per_buyer"),
    (pl.col("n_buyers") / pl.col("n_active_users")).alias("buyer_rate"),
)
daily.write_parquet(OUT / "daily_panel.parquet")

sub("first / last 10 days")
print(daily.head(10))
print(daily.tail(10))

sub("monthly aggregates")
monthly = (
    daily.group_by(pl.col("event_date").dt.strftime("%Y-%m").alias("ym"))
    .agg(pl.len().alias("days"), pl.col("gmv").sum(), pl.col("to_ord").sum(),
         pl.col("n_active_users").mean().alias("mean_dau"), pl.col("searches").sum())
    .sort("ym")
)
print(monthly)
REPORT["monthly"] = monthly.to_dicts()

sub("day-of-week effect (1=Mon .. 7=Sun), medians over all days")
dow = daily.group_by("dow").agg(
    pl.col("gmv").median().alias("gmv_med"),
    pl.col("n_active_users").median().alias("dau_med"),
    pl.col("to_ord").median().alias("ord_med"),
).sort("dow")
gmed = daily["gmv"].median()
for r in dow.iter_rows(named=True):
    print(f"  dow={r['dow']}  gmv_med={r['gmv_med']:>14,.0f} ({100 * r['gmv_med'] / gmed:6.1f}% of overall median)  "
          f"dau_med={r['dau_med']:>9,.0f}  ord_med={r['ord_med']:>9,.0f}")
REPORT["dow"] = dow.to_dicts()

sub("top-20 GMV days (promo / holiday detection)")
top = daily.sort("gmv", descending=True).head(20).select(
    ["event_date", "dow", "gmv", "n_active_users", "n_buyers", "to_ord", "gmv_per_buyer"])
print(top)
REPORT["top_gmv_days"] = top.to_dicts()

sub("bottom-10 GMV days")
print(daily.sort("gmv").head(10).select(["event_date", "dow", "gmv", "n_active_users", "n_buyers"]))

sub("THE TEST WINDOW vs THE SAME WINDOW ONE YEAR EARLIER")
# test target window = 2026-02-14 .. 2026-03-15 (unobserved).
# observed analogue    = 2025-02-14 .. 2025-03-15.
win_pairs = [
    ("test-window-1y (2025-02-14..2025-03-15)", date(2025, 2, 14), date(2025, 3, 15)),
    ("last-observed-30d (2026-01-15..2026-02-13)", date(2026, 1, 15), date(2026, 2, 13)),
    ("prev-30d (2025-12-16..2026-01-14)", date(2025, 12, 16), date(2026, 1, 14)),
    ("1y-before-last-obs (2025-01-15..2025-02-13)", date(2025, 1, 15), date(2025, 2, 13)),
    ("dec-2025 (2025-12-01..2025-12-30)", date(2025, 12, 1), date(2025, 12, 30)),
]
win_stats = {}
for name, a, b in win_pairs:
    w = daily.filter(pl.col("event_date").is_between(a, b))
    st = {"days": w.height, "gmv": float(w["gmv"].sum()), "orders": int(w["to_ord"].sum()),
          "mean_dau": float(w["n_active_users"].mean())}
    win_stats[name] = st
    print(f"  {name:46s} days={st['days']:3d}  gmv={st['gmv']:>16,.0f}  orders={st['orders']:>10,}  meanDAU={st['mean_dau']:>9,.0f}")
yoy = win_stats["test-window-1y (2025-02-14..2025-03-15)"]["gmv"] / win_stats["1y-before-last-obs (2025-01-15..2025-02-13)"]["gmv"]
print(f"\n  seasonal ratio (Feb14-Mar15) / (Jan15-Feb13) measured on 2025 : {yoy:.4f}")
print("  -> this is the calendar-shift multiplier the test window carries vs the last observed 30d block.")
REPORT["window_stats"] = win_stats
REPORT["seasonal_ratio_2025"] = yoy

sub("30-day rolling GMV totals sampled every 30 days back from the last observed day")
rolls = []
end = dmax
while (end - timedelta(days=29)) >= dmin:
    st = end - timedelta(days=29)
    w = daily.filter(pl.col("event_date").is_between(st, end))
    rolls.append({"start": st, "end": end, "gmv": float(w["gmv"].sum()),
                  "orders": int(w["to_ord"].sum()), "act_users": None})
    end = end - timedelta(days=30)
for r in reversed(rolls):
    print(f"  {r['start']} .. {r['end']}   gmv={r['gmv']:>16,.0f}  orders={r['orders']:>10,}")
REPORT["rolling30"] = rolls

# figures
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = daily.to_pandas()
    fig, ax = plt.subplots(4, 1, figsize=(15, 14), sharex=True)
    ax[0].plot(d.event_date, d.n_active_users, lw=.9); ax[0].set_ylabel("active users"); ax[0].grid(alpha=.3)
    ax[1].plot(d.event_date, d.gmv, lw=.9, color="darkorange"); ax[1].set_ylabel("GMV"); ax[1].grid(alpha=.3)
    ax[2].plot(d.event_date, d.to_ord, lw=.9, color="firebrick"); ax[2].set_ylabel("orders"); ax[2].grid(alpha=.3)
    ax[3].plot(d.event_date, d.searches, lw=.9, color="purple"); ax[3].set_ylabel("searches"); ax[3].grid(alpha=.3)
    for a in ax:
        a.axvspan(date(2025, 2, 14), date(2025, 3, 15), color="green", alpha=.12)
    ax[0].set_title("Daily panel (green band = calendar analogue of the test window, one year earlier)")
    fig.tight_layout(); fig.savefig(FIGS / "daily_panel.png", dpi=110); plt.close(fig)
    print("  saved reports/figs/daily_panel.png")
except Exception as e:  # pragma: no cover
    print(f"  [plot skipped] {e}")

dump("stage2_time", {k: REPORT[k] for k in ["dates", "monthly", "dow", "top_gmv_days",
                                            "window_stats", "seasonal_ratio_2025", "rolling30"]})


# ============================================================================
# STAGE 3 -- user panel
# ============================================================================
hdr("STAGE 3 -- USER PANEL / SPARSITY")

upanel = (
    df.group_by("user_id")
    .agg(
        pl.len().alias("n_rows"),
        pl.col("event_date").min().alias("first_date"),
        pl.col("event_date").max().alias("last_date"),
        pl.col("gmv").sum().alias("gmv_total"),
        (pl.col("gmv") > 0).sum().alias("n_buy_days"),
        pl.col("to_ord").sum().alias("n_orders"),
        pl.col("searches").sum().alias("n_searches"),
        pl.col("to_cart").sum().alias("n_cart"),
    )
    .sort("user_id")
)
upanel = upanel.with_columns(
    ((pl.lit(dmax) - pl.col("last_date")).dt.total_days()).alias("recency_days"),
    ((pl.col("last_date") - pl.col("first_date")).dt.total_days() + 1).alias("tenure_days"),
)
upanel.write_parquet(OUT / "user_panel.parquet")

sub("active days per user (rows per user)")
nr = upanel["n_rows"].to_numpy()
print(f"  {json.dumps({k: round(v, 1) for k, v in q(nr).items()})}")
print(f"  mean={nr.mean():.1f}  users with 1 row={int((nr == 1).sum()):,}  "
      f"users with >=300 rows={int((nr >= 300).sum()):,}")
REPORT["rows_per_user"] = {"quantiles": q(nr), "mean": float(nr.mean()),
                           "n_users_1row": int((nr == 1).sum()),
                           "n_users_ge300": int((nr >= 300).sum())}

sub("lifetime GMV per user")
gt = upanel["gmv_total"].to_numpy()
print(f"  {json.dumps({k: round(v, 2) for k, v in q(gt).items()})}")
print(f"  users with zero lifetime GMV : {int((gt <= 0).sum()):,} ({100 * (gt <= 0).mean():.2f}%)")
print(f"  Gini of lifetime GMV         : {gini(gt):.4f}")
top1 = np.sort(gt)[::-1][: int(0.01 * gt.size)].sum() / gt.sum()
top10 = np.sort(gt)[::-1][: int(0.10 * gt.size)].sum() / gt.sum()
print(f"  top-1% users hold  {100 * top1:.2f}% of GMV;  top-10% hold {100 * top10:.2f}%")
REPORT["lifetime_gmv"] = {"quantiles": q(gt), "n_zero_users": int((gt <= 0).sum()),
                          "gini": gini(gt), "top1pct_share": float(top1), "top10pct_share": float(top10)}

sub("orders / searches per user over the whole history")
for c in ["n_orders", "n_searches", "n_buy_days", "n_cart"]:
    a = upanel[c].to_numpy()
    print(f"  {c:12s} {json.dumps({k: round(v, 1) for k, v in q(a).items()})}")
    REPORT.setdefault("user_totals", {})[c] = q(a)

sub("recency (days since last activity, relative to 2026-02-13) and tenure")
rec = upanel["recency_days"].to_numpy()
ten = upanel["tenure_days"].to_numpy()
print(f"  recency  {json.dumps({k: round(v, 1) for k, v in q(rec).items()})}")
print(f"  tenure   {json.dumps({k: round(v, 1) for k, v in q(ten).items()})}")
for thr in [0, 1, 3, 7, 14, 30, 60, 90, 180, 365]:
    print(f"    active within last {thr:>3d} days : {int((rec <= thr).sum()):>8,} ({100 * (rec <= thr).mean():5.2f}%)")
REPORT["recency"] = {"quantiles": q(rec),
                     "active_within": {int(t): int((rec <= t).sum()) for t in [0, 1, 3, 7, 14, 30, 60, 90, 180, 365]}}
REPORT["tenure"] = q(ten)

sub("first-activity cohort (does the user base grow over time = new signups?)")
coh = (
    upanel.group_by(pl.col("first_date").dt.strftime("%Y-%m").alias("cohort")).len().sort("cohort")
)
print(coh)
REPORT["first_date_cohorts"] = coh.to_dicts()
lastcoh = (
    upanel.group_by(pl.col("last_date").dt.strftime("%Y-%m").alias("last_month")).len().sort("last_month")
)
print(lastcoh)
REPORT["last_date_cohorts"] = lastcoh.to_dicts()

sub("user_id: is the identifier informative (ordering / range / correlation with target)?")
uid = upanel["user_id"].to_numpy()
print(f"  min={uid.min():,}  max={uid.max():,}  n={uid.size:,}  "
      f"contiguous={'yes' if uid.max() - uid.min() + 1 == uid.size else 'no'}")
print(f"  corr(user_id, log1p(lifetime_gmv))  = {np.corrcoef(uid, np.log1p(gt))[0, 1]:.4f}")
print(f"  corr(user_id, first_date_ordinal)   = "
      f"{np.corrcoef(uid, upanel['first_date'].to_numpy().astype('datetime64[D]').astype(np.int64))[0, 1]:.4f}")
print(f"  corr(user_id, n_rows)               = {np.corrcoef(uid, nr)[0, 1]:.4f}")
REPORT["user_id_info"] = {
    "min": int(uid.min()), "max": int(uid.max()),
    "contiguous": bool(uid.max() - uid.min() + 1 == uid.size),
    "corr_with_log_lifetime_gmv": float(np.corrcoef(uid, np.log1p(gt))[0, 1]),
    "corr_with_n_rows": float(np.corrcoef(uid, nr)[0, 1]),
}

sub("does the submission set match the training users exactly?")
ss = pl.read_csv(SAMPLE_SUB)
ss_ids = set(ss["user_id"].to_list())
tr_ids = set(uid.tolist())
print(f"  sample_submit rows={ss.height:,}  unique={len(ss_ids):,}")
print(f"  train users={len(tr_ids):,}")
print(f"  in submit but not in train : {len(ss_ids - tr_ids):,}")
print(f"  in train but not in submit : {len(tr_ids - ss_ids):,}")
print(f"  identical sets             : {ss_ids == tr_ids}")
print(f"  sample_submit 'predict' stats: min={ss['predict'].min():.4f} max={ss['predict'].max():.4f} "
      f"mean={ss['predict'].mean():.4f} zeros={int((ss['predict'] == 0).sum()):,}")
REPORT["submission_match"] = {
    "n_submit": ss.height, "n_train_users": len(tr_ids),
    "submit_minus_train": len(ss_ids - tr_ids), "train_minus_submit": len(tr_ids - ss_ids),
    "identical": bool(ss_ids == tr_ids),
    "sample_predict": {"min": float(ss["predict"].min()), "max": float(ss["predict"].max()),
                       "mean": float(ss["predict"].mean()),
                       "n_zero": int((ss["predict"] == 0).sum())},
}

sub("gap structure between consecutive active days (per user) -- sample of 20k users")
rng = np.random.default_rng(0)
samp = rng.choice(uid, size=min(20000, uid.size), replace=False)
gsub = (
    df.filter(pl.col("user_id").is_in(samp.tolist()))
    .select(["user_id", "event_date"])
    .sort(["user_id", "event_date"])
    .with_columns((pl.col("event_date").diff().over("user_id").dt.total_days()).alias("gap"))
    .drop_nulls("gap")
)
gaps = gsub["gap"].to_numpy()
print(f"  n gaps={gaps.size:,}  {json.dumps({k: round(v, 1) for k, v in q(gaps).items()})}")
print(f"  share gap==1 (consecutive days) : {100 * (gaps == 1).mean():.2f}%")
print(f"  share gap<=7                    : {100 * (gaps <= 7).mean():.2f}%")
REPORT["activity_gaps"] = {"quantiles": q(gaps), "share_gap1": float((gaps == 1).mean()),
                           "share_le7": float((gaps <= 7).mean())}
del gsub, gaps

dump("stage3_users", {k: REPORT[k] for k in
                      ["rows_per_user", "lifetime_gmv", "user_totals", "recency", "tenure",
                       "first_date_cohorts", "last_date_cohorts", "user_id_info",
                       "submission_match", "activity_gaps"]})


# ============================================================================
# STAGE 4 -- target construction at rolling anchors + naive baselines
# ============================================================================
hdr("STAGE 4 -- TARGET AT ROLLING ANCHORS + NAIVE BASELINE SCOREBOARD")

HORIZON = 30
ALL_USERS = np.sort(uid)
uidx = {u: i for i, u in enumerate(ALL_USERS.tolist())}
N = ALL_USERS.size

slim = df.select(["user_id", "event_date", "gmv", "to_ord", "searches", "to_cart"])
del df


def window_sum(col: str, a: date, b: date) -> np.ndarray:
    """Dense per-user sum of `col` over [a, b] inclusive, aligned to ALL_USERS."""
    g = (
        slim.filter(pl.col("event_date").is_between(a, b))
        .group_by("user_id")
        .agg(pl.col(col).sum().alias("v"))
    )
    out = np.zeros(N, dtype=np.float64)
    ids = g["user_id"].to_numpy()
    vals = g["v"].to_numpy().astype(np.float64)
    pos = np.searchsorted(ALL_USERS, ids)
    out[pos] = vals
    return out


def last_active(anchor: date) -> np.ndarray:
    """Days since last activity as of `anchor` (999 if never active)."""
    g = (
        slim.filter(pl.col("event_date") <= anchor)
        .group_by("user_id")
        .agg(pl.col("event_date").max().alias("d"))
    )
    out = np.full(N, 999.0)
    pos = np.searchsorted(ALL_USERS, g["user_id"].to_numpy())
    out[pos] = np.array([(anchor - x).days for x in g["d"].to_list()], dtype=np.float64)
    return out


# anchors: last one with a complete 30-day target, then every 30 days back
LAST_ANCHOR = dmax - timedelta(days=HORIZON)      # 2026-01-14
anchors = [LAST_ANCHOR - timedelta(days=30 * k) for k in range(0, 11)]
anchors = [a for a in anchors if a - timedelta(days=89) >= dmin]  # need >= 90d history
anchors = sorted(anchors)
# plus the exact calendar analogue of the test cut-off
cal_anchor = date(2025, 2, 13)
print(f"  test cut-off (real)      : {dmax}  -> predict {dmax + timedelta(days=1)} .. {dmax + timedelta(days=HORIZON)}")
print(f"  usable anchors (>=90d hist, full target): {len(anchors)}")
for a in anchors:
    print(f"    anchor {a}  target {a + timedelta(days=1)} .. {a + timedelta(days=HORIZON)}")
print(f"  calendar-analogue anchor (short history): {cal_anchor}")
REPORT["anchors"] = {"last": LAST_ANCHOR, "list": anchors, "calendar_analogue": cal_anchor}

targ_stats, board = {}, []
cache = {}
for a in anchors + [cal_anchor]:
    y = window_sum("gmv", a + timedelta(days=1), a + timedelta(days=HORIZON))
    p30 = window_sum("gmv", a - timedelta(days=29), a)
    p60 = window_sum("gmv", a - timedelta(days=59), a - timedelta(days=30))
    p90 = window_sum("gmv", a - timedelta(days=89), a - timedelta(days=60))
    o30 = window_sum("to_ord", a - timedelta(days=29), a)
    s30 = window_sum("searches", a - timedelta(days=29), a)
    rec = last_active(a)
    cache[a] = dict(y=y, p30=p30, p60=p60, p90=p90, o30=o30, s30=s30, rec=rec)

    ly = np.log1p(y)
    st = {
        "anchor": a,
        "n_users": int(N),
        "share_zero_target": float((y <= 0).mean()),
        "mean_target": float(y.mean()),
        "sum_target": float(y.sum()),
        "mean_log1p": float(ly.mean()),
        "std_log1p": float(ly.std()),
        "gini": gini(y),
        "quantiles": q(y),
        "quantiles_pos": q(y[y > 0]),
    }
    targ_stats[str(a)] = st
    print(f"\n  ANCHOR {a}: zero-share={st['share_zero_target']:.4f}  mean={st['mean_target']:.2f}  "
          f"sum={st['sum_target']:,.0f}  mean_log1p={st['mean_log1p']:.4f}  gini={st['gini']:.4f}")
    print(f"     target quantiles (all)  {json.dumps({k: round(v, 2) for k, v in st['quantiles'].items()})}")
    print(f"     target quantiles (y>0)  {json.dumps({k: round(v, 2) for k, v in st['quantiles_pos'].items()})}")

REPORT["target_stats"] = targ_stats
dump("stage4_targets", targ_stats)

sub("NAIVE BASELINE SCOREBOARD (RMSLE, lower is better)")
print("  Baselines are evaluated on every anchor. `k*` = the multiplicative shrink applied")
print("  in *linear* space to the last-30-day GMV.\n")

for a in anchors + [cal_anchor]:
    c = cache[a]
    y, p30, p60, p90 = c["y"], c["p30"], c["p60"], c["p90"]
    ly = np.log1p(y)

    # optimal constant in log space
    c_star = float(np.expm1(ly.mean()))
    # optimal global multiplier on p30 (grid, in log space)
    ks = np.linspace(0.0, 2.0, 81)
    kbest, sbest = 0.0, 1e9
    for k in ks:
        s = rmsle(y, k * p30)
        if s < sbest:
            kbest, sbest = float(k), s
    # optimal shrink applied to log1p(p30):  pred = expm1(alpha*log1p(p30))
    abest, sabest = 0.0, 1e9
    for al in np.linspace(0, 1.2, 61):
        s = rmsle(y, np.expm1(al * np.log1p(p30)))
        if s < sabest:
            abest, sabest = float(al), s

    rows = [
        ("zero (predict 0)", np.zeros(N)),
        (f"constant c*={c_star:.3f}", np.full(N, c_star)),
        ("last-30d gmv (k=1)", p30),
        (f"last-30d gmv * k*={kbest:.3f}", kbest * p30),
        (f"expm1({abest:.2f}*log1p(p30))", np.expm1(abest * np.log1p(p30))),
        ("mean(p30,p60,p90) linear", (p30 + p60 + p90) / 3),
        ("expm1(mean log1p(p30,p60,p90))", np.expm1((np.log1p(p30) + np.log1p(p60) + np.log1p(p90)) / 3)),
        ("last-90d gmv / 3", (p30 + p60 + p90) / 3),
    ]
    print(f"  --- anchor {a} ---")
    for name, pred in rows:
        sc = rmsle(y, pred)
        tot_err = (pred.sum() - y.sum()) / y.sum() if y.sum() > 0 else np.nan
        board.append({"anchor": a, "baseline": name, "rmsle": sc, "gini_pred": gini(pred),
                      "total_gmv_rel_err": float(tot_err)})
        print(f"      {name:36s} RMSLE={sc:.5f}   gini={gini(pred):.4f}   sum(pred)/sum(y)-1={tot_err:+.3f}")
    print(f"      [best k on p30 = {kbest:.3f}]  [best alpha on log1p(p30) = {abest:.3f}]")

REPORT["baselines"] = board
dump("stage4_baselines", board)


# ============================================================================
# STAGE 5 -- predictability structure
# ============================================================================
hdr("STAGE 5 -- PREDICTABILITY / ZERO-INFLATION / METRIC DECOMPOSITION")

A = LAST_ANCHOR
c = cache[A]
y, p30, p60, p90, o30, s30, rec = c["y"], c["p30"], c["p60"], c["p90"], c["o30"], c["s30"], c["rec"]
ly, lp30 = np.log1p(y), np.log1p(p30)

sub(f"autocorrelation of the 30-day GMV block (anchor {A})")
print(f"  corr(log1p(next30), log1p(prev30))  = {np.corrcoef(ly, lp30)[0, 1]:.4f}")
print(f"  corr(log1p(next30), log1p(prev60))  = {np.corrcoef(ly, np.log1p(p60))[0, 1]:.4f}")
print(f"  corr(log1p(next30), log1p(prev90))  = {np.corrcoef(ly, np.log1p(p90))[0, 1]:.4f}")
print(f"  corr(log1p(next30), log1p(sum90))   = {np.corrcoef(ly, np.log1p(p30 + p60 + p90))[0, 1]:.4f}")
print(f"  corr(log1p(next30), log1p(orders30))= {np.corrcoef(ly, np.log1p(o30))[0, 1]:.4f}")
print(f"  corr(log1p(next30), log1p(search30))= {np.corrcoef(ly, np.log1p(s30))[0, 1]:.4f}")
print(f"  corr(log1p(next30), -recency)       = {np.corrcoef(ly, -rec)[0, 1]:.4f}")
print(f"  corr(linear next30, linear prev30)  = {np.corrcoef(y, p30)[0, 1]:.4f}")
REPORT["autocorr"] = {
    "log_prev30": float(np.corrcoef(ly, lp30)[0, 1]),
    "log_prev60": float(np.corrcoef(ly, np.log1p(p60))[0, 1]),
    "log_prev90": float(np.corrcoef(ly, np.log1p(p90))[0, 1]),
    "log_sum90": float(np.corrcoef(ly, np.log1p(p30 + p60 + p90))[0, 1]),
    "log_orders30": float(np.corrcoef(ly, np.log1p(o30))[0, 1]),
    "log_searches30": float(np.corrcoef(ly, np.log1p(s30))[0, 1]),
    "recency": float(np.corrcoef(ly, -rec)[0, 1]),
    "linear_prev30": float(np.corrcoef(y, p30)[0, 1]),
}

sub("zero-inflation conditioned on past behaviour")
buckets = [
    ("no activity in 90d", rec > 90),
    ("active 31-90d ago", (rec > 30) & (rec <= 90)),
    ("active 8-30d ago", (rec > 7) & (rec <= 30)),
    ("active 1-7d ago", rec <= 7),
    ("no order in prev30", o30 == 0),
    ("1 order in prev30", o30 == 1),
    ("2-4 orders in prev30", (o30 >= 2) & (o30 <= 4)),
    (">=5 orders in prev30", o30 >= 5),
    ("prev30 gmv == 0", p30 == 0),
    ("prev30 gmv in (0,1k]", (p30 > 0) & (p30 <= 1000)),
    ("prev30 gmv in (1k,10k]", (p30 > 1000) & (p30 <= 10000)),
    ("prev30 gmv > 10k", p30 > 10000),
    ("ALL USERS", np.ones(N, dtype=bool)),
]
zi = []
for name, m in buckets:
    if m.sum() == 0:
        continue
    r = {"bucket": name, "n": int(m.sum()), "share_users": float(m.mean()),
         "p_target_pos": float((y[m] > 0).mean()),
         "mean_target": float(y[m].mean()),
         "mean_target_given_pos": float(y[m][y[m] > 0].mean()) if (y[m] > 0).any() else 0.0,
         "share_of_total_gmv": float(y[m].sum() / y.sum())}
    zi.append(r)
    print(f"  {name:26s} n={r['n']:>8,} ({100 * r['share_users']:5.2f}%)  "
          f"P(y>0)={r['p_target_pos']:.4f}  E[y]={r['mean_target']:>10.2f}  "
          f"E[y|y>0]={r['mean_target_given_pos']:>10.2f}  share_of_GMV={100 * r['share_of_total_gmv']:5.2f}%")
REPORT["zero_inflation"] = zi

sub("RMSLE decomposition: where does the error live?")
for name, pred in [("predict 0", np.zeros(N)),
                   ("last-30d gmv", p30),
                   ("constant c*", np.full(N, float(np.expm1(ly.mean()))))]:
    e2 = (np.log1p(y) - np.log1p(np.clip(pred, 0, None))) ** 2
    mz, mp = (y <= 0), (y > 0)
    print(f"  [{name}] total RMSLE={np.sqrt(e2.mean()):.5f}")
    print(f"      zero-target users  ({100 * mz.mean():5.2f}% of users) contribute {100 * e2[mz].sum() / e2.sum():5.2f}% of SSE")
    print(f"      pos-target  users  ({100 * mp.mean():5.2f}% of users) contribute {100 * e2[mp].sum() / e2.sum():5.2f}% of SSE")
REPORT["rmsle_decomposition_note"] = "see log"

sub("target distribution in log1p space (histogram, anchor %s)" % A)
hist, edges = np.histogram(ly, bins=40)
for h, lo, hi in zip(hist, edges[:-1], edges[1:]):
    print(f"  log1p in [{lo:6.2f},{hi:6.2f})  n={h:>8,}  {'#' * int(60 * h / hist.max())}")
REPORT["log1p_hist"] = {"counts": hist, "edges": edges}

sub("stability of the target across anchors (is the zero-share / level drifting?)")
print(f"  {'anchor':12s} {'zero-share':>11s} {'mean':>12s} {'sum':>16s} {'mean_log1p':>11s} {'gini':>8s}")
for a in anchors:
    st = targ_stats[str(a)]
    print(f"  {str(a):12s} {st['share_zero_target']:11.4f} {st['mean_target']:12.2f} "
          f"{st['sum_target']:16,.0f} {st['mean_log1p']:11.4f} {st['gini']:8.4f}")
st = targ_stats[str(cal_anchor)]
print(f"  {str(cal_anchor):12s} {st['share_zero_target']:11.4f} {st['mean_target']:12.2f} "
      f"{st['sum_target']:16,.0f} {st['mean_log1p']:11.4f} {st['gini']:8.4f}   <- calendar analogue of test")

sub("user-level persistence: transition matrix of 30d-GMV deciles (prev30 -> next30)")
nz = p30 > 0
dec = np.zeros(N, dtype=int)
if nz.sum() > 0:
    dec[nz] = 1 + np.digitize(p30[nz], np.quantile(p30[nz], np.linspace(0, 1, 10)[1:]), right=True)
ydec = np.zeros(N, dtype=int)
nzy = y > 0
if nzy.sum() > 0:
    ydec[nzy] = 1 + np.digitize(y[nzy], np.quantile(y[nzy], np.linspace(0, 1, 10)[1:]), right=True)
print("  rows = prev30 decile (0 = zero), cols = next30 decile (0 = zero); values = row %")
print("       " + "".join(f"{j:>6d}" for j in range(11)))
tm = np.zeros((11, 11))
for i in range(11):
    m = dec == i
    if m.sum() == 0:
        continue
    for j in range(11):
        tm[i, j] = (ydec[m] == j).mean()
    print(f"  {i:>3d}  " + "".join(f"{100 * tm[i, j]:6.1f}" for j in range(11)) + f"   n={int(m.sum()):,}")
REPORT["transition_matrix"] = tm

sub("tie-breaker metrics on the naive baselines (anchor %s)" % A)
print(f"  TRUE: sum(y)={y.sum():,.0f}   gini(y)={gini(y):.4f}   n_pos={int((y > 0).sum()):,}")
for name, pred in [("last-30d gmv", p30),
                   ("last-30d * k*", p30 * 1.0),
                   ("constant c*", np.full(N, float(np.expm1(ly.mean()))))]:
    print(f"  {name:20s} sum={pred.sum():>16,.0f}  rel.err={100 * (pred.sum() / y.sum() - 1):+7.2f}%  "
          f"gini={gini(pred):.4f}")

sub("how much of next-30d GMV comes from users who were INACTIVE in the prev 30 days?")
inact = p30 == 0
print(f"  users with prev30 gmv == 0 : {int(inact.sum()):,} ({100 * inact.mean():.2f}%)")
print(f"  their share of next-30d GMV: {100 * y[inact].sum() / y.sum():.2f}%")
print(f"  their P(y>0)               : {(y[inact] > 0).mean():.4f}")
noact = rec > 90
print(f"  users with NO activity at all in prev 90d : {int(noact.sum()):,} ({100 * noact.mean():.2f}%)")
print(f"  their share of next-30d GMV               : {100 * y[noact].sum() / y.sum():.2f}%")
print(f"  their P(y>0)                              : {(y[noact] > 0).mean():.4f}")

hdr("DONE")
dump("stage5_predictability", {k: REPORT[k] for k in
                               ["autocorr", "zero_inflation", "log1p_hist", "transition_matrix"]})
dump("eda_full", REPORT)
print(f"total runtime {time.time() - T0:.0f}s")
