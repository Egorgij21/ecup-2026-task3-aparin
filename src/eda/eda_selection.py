#!/usr/bin/env python
"""
E-CUP 2026 / Task 3 -- EDA pass 2: the *sampling rule* and what it does to CV.

Pass 1 turned up a structural fact with big consequences: every one of the 250 000
users has at least one active day in the final 30 days of the history
(max recency = 29 days). The panel is therefore conditioned on end-of-window
activity, which (a) makes the apparent "growth" trend partly an artefact and
(b) contaminates the most test-like CV anchor, because that anchor's *target*
window IS the selection window.

This pass measures all of that, plus the metric's sampling noise floor.
"""

from __future__ import annotations

import json
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path("/path/to/ecup")
OUT = ROOT / "reports" / "eda"
FIGS = ROOT / "reports" / "figs"
OUT.mkdir(parents=True, exist_ok=True)
R: dict = {}
T0 = time.time()


def hdr(t):
    print(f"\n{'=' * 78}\n{t}   [t+{time.time() - T0:.0f}s]\n{'=' * 78}", flush=True)


def sub(t):
    print(f"\n--- {t} " + "-" * max(0, 70 - len(t)), flush=True)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, date):
        return o.isoformat()
    return o


def rmsle(y, p):
    return float(np.sqrt(np.mean((np.log1p(np.clip(y, 0, None)) - np.log1p(np.clip(p, 0, None))) ** 2)))


def gini(y):
    y = np.sort(np.asarray(y, np.float64)); n = y.size; s = y.sum()
    if s <= 0:
        return 0.0
    return float(2 * ((np.arange(1, n + 1) * y).sum()) / (n * s) - (n + 1) / n)


df = pl.read_parquet(ROOT / "data" / "train.parquet")
DMIN, DMAX = df["event_date"].min(), df["event_date"].max()
users = np.sort(df["user_id"].unique().to_numpy())
N = users.size
slim = df.select(["user_id", "event_date", "gmv", "to_ord", "searches", "to_cart", "search", "cat"])


def wsum(col, a, b, src=None):
    src = slim if src is None else src
    g = src.filter(pl.col("event_date").is_between(a, b)).group_by("user_id").agg(pl.col(col).sum().alias("v"))
    out = np.zeros(N)
    out[np.searchsorted(users, g["user_id"].to_numpy())] = g["v"].to_numpy().astype(np.float64)
    return out


def wdays(a, b):
    """number of rows (= present days) per user in [a,b]"""
    g = slim.filter(pl.col("event_date").is_between(a, b)).group_by("user_id").len()
    out = np.zeros(N)
    out[np.searchsorted(users, g["user_id"].to_numpy())] = g["len"].to_numpy().astype(np.float64)
    return out


# ============================================================================
hdr("A -- IS sample_submit.csv THE NAIVE 'REPEAT LAST 30 DAYS' PREDICTION?")
ss = pl.read_csv(ROOT / "data" / "sample_submit.csv").sort("user_id")
assert np.array_equal(ss["user_id"].to_numpy(), users), "user ordering mismatch"
p_ss = ss["predict"].to_numpy()
last30 = wsum("gmv", DMAX - timedelta(days=29), DMAX)
d = np.abs(p_ss - last30)
print(f"  max |sample_submit - sum(gmv over 2026-01-15..2026-02-13)| = {d.max():.3e}")
print(f"  exact matches (atol=1e-9) : {int((d < 1e-9).sum()):,} / {N:,}")
print(f"  -> sample_submit IS the last-30-day GMV: {bool((d < 1e-6).all())}")
R["sample_submit_is_last30d"] = bool((d < 1e-6).all())

# ============================================================================
hdr("B -- THE SAMPLING RULE: WHO IS IN THE 250k PANEL?")
last_act = (
    slim.group_by("user_id").agg(pl.col("event_date").max().alias("d")).sort("user_id")["d"].to_numpy()
)
rec = np.array([(DMAX - x).days for x in last_act.astype("datetime64[D]").astype(object)], float)
first_act = (
    slim.group_by("user_id").agg(pl.col("event_date").min().alias("d")).sort("user_id")["d"].to_numpy()
)
firstoff = np.array([(x - DMIN).days for x in first_act.astype("datetime64[D]").astype(object)], float)
nrows = slim.group_by("user_id").len().sort("user_id")["len"].to_numpy().astype(float)

print(f"  recency (days since last activity @ {DMAX}) : max={rec.max():.0f}  "
      f"share <=29d = {100 * (rec <= 29).mean():.2f}%   share <=30d = {100 * (rec <= 30).mean():.2f}%")
print(f"  rows per user      : min={nrows.min():.0f}  p1={np.quantile(nrows, .01):.0f}")
print(f"  first-activity offset from 2025-01-01 : max={firstoff.max():.0f} days "
      f"(i.e. latest first-seen date = {DMIN + timedelta(days=int(firstoff.max()))})")
print("  => inclusion rule looks like: >=1 active day in the LAST 30 DAYS of history.")
R["selection"] = {"max_recency": float(rec.max()), "min_rows": float(nrows.min()),
                  "latest_first_seen": str(DMIN + timedelta(days=int(firstoff.max())))}

sub("how binding is that rule at earlier dates? (share of the panel active in [A-29, A])")
LAST_ANCHOR = DMAX - timedelta(days=30)
anchors = sorted([LAST_ANCHOR - timedelta(days=30 * k) for k in range(0, 10)])
act_share = {}
for a in anchors + [DMAX]:
    m = wdays(a - timedelta(days=29), a) > 0
    act_share[str(a)] = float(m.mean())
    print(f"  A={a}   active in [A-29,A]: {int(m.sum()):>8,} / {N:,}  ({100 * m.mean():5.2f}%)")
R["active_share_by_anchor"] = act_share

sub("is the 'growth' real, or an artefact of selecting on end-of-window activity?")
print("  Test: take the sub-cohort selected by the SAME rule but at an earlier date")
print("  (>=1 active day in [S-29, S]) and look at its daily active-user count before and after S.")
for S in [date(2025, 6, 30), date(2025, 9, 30)]:
    m = wdays(S - timedelta(days=29), S) > 0
    sel = users[m]
    d2 = slim.filter(pl.col("user_id").is_in(sel.tolist())).group_by("event_date").len().sort("event_date")
    dd = d2.to_pandas().set_index("event_date")["len"]
    pts = [S - timedelta(days=x) for x in (120, 90, 60, 30, 0)] + [S + timedelta(days=x) for x in (30, 60, 90, 120)]
    pts = [p for p in pts if DMIN <= p <= DMAX]
    line = "  ".join(f"{p}:{dd.get(p, 0) / len(sel) * 100:5.1f}%" for p in pts)
    print(f"  S={S}  cohort n={len(sel):,}\n     DAU%/day around S: {line}")
print("\n  (If the % peaks at S and decays both ways, the 'growth' in the full panel is")
print("   selection, not business growth. If it keeps rising after S, growth is real.)")

# ============================================================================
hdr("C -- THE CONTAMINATED ANCHOR: target window overlapping the selection window")
SEL_A, SEL_B = DMAX - timedelta(days=29), DMAX   # 2026-01-15 .. 2026-02-13
print(f"  global selection window = [{SEL_A}, {SEL_B}]")
print(f"  {'anchor':12s} {'target window':26s} {'overlap days':>12s} {'P(active in target)':>20s} {'zero-share':>11s}")
overlap_tbl = []
for a in anchors:
    t0, t1 = a + timedelta(days=1), a + timedelta(days=30)
    ov = max(0, (min(t1, SEL_B) - max(t0, SEL_A)).days + 1)
    y = wsum("gmv", t0, t1)
    act = wdays(t0, t1) > 0
    overlap_tbl.append({"anchor": a, "overlap_days": ov, "p_active_in_target": float(act.mean()),
                        "zero_share": float((y <= 0).mean())})
    print(f"  {str(a):12s} {str(t0) + '..' + str(t1):26s} {ov:>12d} {100 * act.mean():>19.2f}% {100 * (y <= 0).mean():>10.2f}%")
R["overlap_table"] = overlap_tbl
print("\n  -> any anchor with overlap>0 is optimistically biased: its users are GUARANTEED")
print("     to be active inside the target window. Safe anchors have overlap = 0.")

# ============================================================================
hdr("D -- CV-COMPARABLE ANCHORS: re-select the population at each anchor")
print("  For each anchor we mirror the test's inclusion rule: keep only users with")
print("  >=1 active day in [A-29, A]. Baselines are then re-scored on that subset.\n")
rows = []
for a in anchors:
    t0, t1 = a + timedelta(days=1), a + timedelta(days=30)
    ov = max(0, (min(t1, SEL_B) - max(t0, SEL_A)).days + 1)
    keep = wdays(a - timedelta(days=29), a) > 0
    y = wsum("gmv", t0, t1)[keep]
    p30 = wsum("gmv", a - timedelta(days=29), a)[keep]
    p60 = wsum("gmv", a - timedelta(days=59), a - timedelta(days=30))[keep]
    p90 = wsum("gmv", a - timedelta(days=89), a - timedelta(days=60))[keep]
    geo = np.expm1((np.log1p(p30) + np.log1p(p60) + np.log1p(p90)) / 3)
    ks = np.linspace(0.05, 1.5, 30)
    kbest = float(ks[np.argmin([rmsle(y, k * geo) for k in ks])])
    r = {"anchor": a, "overlap_days": ov, "n_users": int(keep.sum()),
         "zero_share": float((y <= 0).mean()), "mean_log1p": float(np.log1p(y).mean()),
         "sum_y": float(y.sum()), "gini_y": gini(y),
         "rmsle_zero": rmsle(y, np.zeros_like(y)),
         "rmsle_p30": rmsle(y, p30),
         "rmsle_geo3": rmsle(y, geo),
         "rmsle_geo3_k": rmsle(y, kbest * geo), "k_best": kbest}
    rows.append(r)
    print(f"  A={a} ov={ov:2d} n={r['n_users']:>7,} zero={100 * r['zero_share']:5.2f}%  "
          f"RMSLE: zero={r['rmsle_zero']:.4f} p30={r['rmsle_p30']:.4f} geo3={r['rmsle_geo3']:.4f} "
          f"geo3*{kbest:.2f}={r['rmsle_geo3_k']:.4f}")
R["cv_comparable"] = rows
safe = [r for r in rows if r["overlap_days"] == 0]
print(f"\n  SAFE anchors (overlap=0): {[str(r['anchor']) for r in safe]}")
if safe:
    v = np.array([r["rmsle_geo3"] for r in safe])
    print(f"  geo3 baseline over safe anchors: mean={v.mean():.4f}  std={v.std():.4f}  "
          f"range=[{v.min():.4f}, {v.max():.4f}]")

# ============================================================================
hdr("E -- METRIC NOISE FLOOR (how big must a CV delta be to be real?)")
A = anchors[-1]
keep = wdays(A - timedelta(days=29), A) > 0
y = wsum("gmv", A + timedelta(days=1), A + timedelta(days=30))[keep]
p30 = wsum("gmv", A - timedelta(days=29), A)[keep]
p60 = wsum("gmv", A - timedelta(days=59), A - timedelta(days=30))[keep]
p90 = wsum("gmv", A - timedelta(days=89), A - timedelta(days=60))[keep]
pred = np.expm1((np.log1p(p30) + np.log1p(p60) + np.log1p(p90)) / 3)
rng = np.random.default_rng(0)
n = y.size
for m in [50_000, 200_000, n]:
    if m > n:
        continue
    vals = [rmsle(y[i], pred[i]) for i in (rng.choice(n, m, replace=False) for _ in range(60))]
    v = np.array(vals)
    print(f"  subsample n={m:>7,}: RMSLE mean={v.mean():.5f}  std={v.std():.5f}  "
          f"(2 sigma = {2 * v.std():.5f})")
    R.setdefault("noise", {})[str(m)] = {"mean": float(v.mean()), "std": float(v.std())}
print("\n  -> public LB (50k users) sampling noise is the `std` at n=50,000.")
print("     A CV improvement smaller than ~2x the n=250k std is not real.")

# ============================================================================
hdr("F -- HEADROOM: what do oracles score?")
ly = np.log1p(y)
print(f"  naive geo3                                  RMSLE = {rmsle(y, pred):.4f}")
o = pred.copy(); o[y <= 0] = 0.0
print(f"  oracle zero/positive split + geo3 magnitude RMSLE = {rmsle(y, o):.4f}")
o2 = np.where(y > 0, np.expm1(ly[y > 0].mean()), 0.0)
print(f"  oracle split + constant magnitude           RMSLE = {rmsle(y, o2):.4f}")
q10 = np.quantile(p30, np.linspace(0, 1, 11))
bins = np.clip(np.digitize(p30, q10[1:-1]), 0, 9)
grp = np.array([np.expm1(ly[bins == b].mean()) if (bins == b).any() else 0 for b in range(10)])
print(f"  per-decile-of-p30 optimal constant          RMSLE = {rmsle(y, grp[bins]):.4f}")
print(f"  perfect prediction                          RMSLE = 0")
R["headroom"] = {"geo3": rmsle(y, pred), "oracle_split_geo3": rmsle(y, o),
                 "oracle_split_const": rmsle(y, o2), "decile_const": rmsle(y, grp[bins])}

# ============================================================================
hdr("G -- WHAT ARE THE 15% ALL-ZERO ROWS?")
z = df.with_columns(
    ((pl.col("searches") == 0) & (pl.col("to_cart") == 0) & (pl.col("to_ord") == 0) &
     (pl.col("search") == 0) & (pl.col("cat") == 0)).alias("empty")
)
print(f"  empty rows: {int(z['empty'].sum()):,} ({100 * z['empty'].mean():.2f}%)")
byday = z.group_by("event_date").agg(pl.col("empty").mean().alias("share")).sort("event_date")
print(f"  share of empty rows per day: min={byday['share'].min():.4f} max={byday['share'].max():.4f} "
      f"first={byday['share'][0]:.4f} last={byday['share'][-1]:.4f}")
byuser = z.group_by("user_id").agg(pl.col("empty").mean().alias("share"))["share"].to_numpy()
print(f"  per-user share of empty rows: {json.dumps({f'p{p}': round(float(np.quantile(byuser, p / 100)), 3) for p in [1, 25, 50, 75, 99]})}")
print("  -> a row with zero search/catalog activity still marks a VISIT: row presence is a")
print("     feature in its own right, distinct from 'searched' or 'bought'.")
R["empty_rows"] = {"n": int(z["empty"].sum()), "share": float(z["empty"].mean())}

sub("does an empty-row day predict the target? (anchor %s)" % A)
emp30 = np.zeros(N)
g = (z.filter(pl.col("event_date").is_between(A - timedelta(days=29), A))
       .group_by("user_id").agg(pl.col("empty").sum().alias("v")))
emp30[np.searchsorted(users, g["user_id"].to_numpy())] = g["v"].to_numpy().astype(float)
emp30 = emp30[keep]
act30 = wdays(A - timedelta(days=29), A)[keep]
print(f"  corr(log1p(next30 gmv), n_empty_days_30)  = {np.corrcoef(ly, np.log1p(emp30))[0, 1]:.4f}")
print(f"  corr(log1p(next30 gmv), n_active_days_30) = {np.corrcoef(ly, np.log1p(act30))[0, 1]:.4f}")
print(f"  corr(log1p(next30 gmv), n_nonempty_30)    = {np.corrcoef(ly, np.log1p(act30 - emp30))[0, 1]:.4f}")

# ============================================================================
hdr("H -- WITHIN-HORIZON STRUCTURE + ANCHOR-TO-ANCHOR PERSISTENCE")
sub("share of the 30-day target GMV landing in each week after the anchor")
tot = 0.0; parts = []
for w in range(4):
    a0 = A + timedelta(days=1 + 7 * w)
    a1 = min(A + timedelta(days=min(7 + 7 * w, 30)), A + timedelta(days=30))
    s = wsum("gmv", a0, a1)[keep].sum()
    parts.append((a0, a1, s)); tot += s
for a0, a1, s in parts:
    print(f"  {a0}..{a1}  {100 * s / tot:5.2f}% of horizon GMV")

sub("correlation between targets at consecutive anchors (same users)")
ys = {}
for a in anchors:
    ys[a] = np.log1p(wsum("gmv", a + timedelta(days=1), a + timedelta(days=30)))
ks = list(ys)
print("        " + " ".join(f"{str(k)[5:]:>7s}" for k in ks))
for i, k1 in enumerate(ks):
    print(f"  {str(k1)[5:]:6s} " + " ".join(f"{np.corrcoef(ys[k1], ys[k2])[0, 1]:7.3f}" for k2 in ks))

sub("lag decay: corr(log1p(target@A), log1p(gmv in [A-30k-29, A-30k])) at anchor %s" % A)
for k in range(0, 12):
    b = A - timedelta(days=30 * k)
    a0 = b - timedelta(days=29)
    if a0 < DMIN:
        break
    v = wsum("gmv", a0, b)[keep]
    print(f"  lag {k:2d} block [{a0}..{b}]  corr={np.corrcoef(ly, np.log1p(v))[0, 1]:.4f}")

# ============================================================================
hdr("I -- LEAKAGE CHECKS")
sub("duplicate users (identical full activity vector)?")
sig = (
    df.group_by("user_id")
    .agg(pl.col("gmv").sum().alias("g"), pl.len().alias("n"), pl.col("searches").sum().alias("s"),
         pl.col("to_ord").sum().alias("o"), pl.col("event_date").min().alias("f"),
         pl.col("event_date").max().alias("l"))
)
ndup = sig.height - sig.select(["g", "n", "s", "o", "f", "l"]).n_unique()
print(f"  users sharing an identical (gmv,n_rows,searches,orders,first,last) signature: {ndup:,}")
print(f"  (users with zero lifetime activity collide trivially; {int((sig['g'] == 0).sum()):,} have gmv=0)")

sub("row order / user_id vs target")
yfull = wsum("gmv", A + timedelta(days=1), A + timedelta(days=30))
print(f"  corr(user_id, log1p(target@{A})) = {np.corrcoef(users.astype(float), np.log1p(yfull))[0, 1]:.5f}")
print(f"  corr(rank(user_id), rank(target)) = "
      f"{np.corrcoef(np.arange(N), np.argsort(np.argsort(yfull)))[0, 1]:.5f}")

sub("are there rows dated after the cut-off, or any future-dated leak?")
print(f"  max event_date = {DMAX} (cut-off). rows after cut-off: 0 by construction.")

hdr("DONE")
with open(OUT / "eda_selection.json", "w") as f:
    json.dump(jsonable(R), f, indent=2, default=str)
print(f"total runtime {time.time() - T0:.0f}s")
