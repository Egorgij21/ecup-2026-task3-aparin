#!/usr/bin/env python
"""
E-CUP 2026 / Task 3 -- EDA pass 3: pin down the exact panel-inclusion rule.

Pass 2 showed the share of users active in a trailing 30-day window is 100 % for the
last THREE disjoint 30-day blocks and only 93 % / 90 % / 87 % before that. That is not
a single "active in the last 30 days" filter -- it is a stronger guarantee that
contaminates more CV anchors than expected. This pass finds the exact boundary.
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
R: dict = {}
T0 = time.time()


def hdr(t):
    print(f"\n{'=' * 78}\n{t}   [t+{time.time() - T0:.0f}s]\n{'=' * 78}", flush=True)


def sub(t):
    print(f"\n--- {t} " + "-" * max(0, 70 - len(t)), flush=True)


def rmsle(y, p):
    return float(np.sqrt(np.mean((np.log1p(np.clip(y, 0, None)) - np.log1p(np.clip(p, 0, None))) ** 2)))


df = pl.read_parquet(ROOT / "data" / "train.parquet").select(
    ["user_id", "event_date", "gmv", "to_ord"])
DMIN, DMAX = df["event_date"].min(), df["event_date"].max()
users = np.sort(df["user_id"].unique().to_numpy())
N = users.size

# dense user x day presence matrix as a bitset: 250k x 409 bools = 102M bytes -> fine
hdr("BUILD DENSE PRESENCE MATRIX (250k x 409)")
uidx = np.searchsorted(users, df["user_id"].to_numpy())
didx = (df["event_date"].to_numpy().astype("datetime64[D]") - np.datetime64(DMIN)).astype(int)
P = np.zeros((N, 409), dtype=bool)
P[uidx, didx] = True
print(f"  presence matrix {P.shape}, density={100 * P.mean():.2f}%")
gmv = np.zeros((N, 409), dtype=np.float32)
np.add.at(gmv, (uidx, didx), df["gmv"].to_numpy().astype(np.float32))
print(f"  gmv matrix built, total={gmv.sum():,.0f}")
del uidx, didx

D = [DMIN + timedelta(days=i) for i in range(409)]

# ============================================================================
hdr("A -- TRAILING-30d ACTIVE SHARE BY END DATE  (where does the 100% zone start?)")
csum = np.concatenate([np.zeros((N, 1), np.int32), np.cumsum(P, axis=1, dtype=np.int32)], axis=1)


def active_in(a_idx, b_idx):
    """boolean: user active in day-index window [a_idx, b_idx] inclusive"""
    a_idx = max(a_idx, 0)
    return (csum[:, b_idx + 1] - csum[:, a_idx]) > 0


shares = []
for e in range(29, 409):
    shares.append((D[e], float(active_in(e - 29, e).mean())))
first100 = next((d for d, s in shares if s >= 1.0), None)
print(f"  first end-date with 100% trailing-30d activity : {first100}")
print(f"  last  end-date with <100%                      : "
      f"{[d for d, s in shares if s < 1.0][-1] if any(s < 1 for _, s in shares) else None}")
print("\n  trailing-30d active share, sampled every 15 days:")
for d, s in shares[::15]:
    bar = "#" * int(60 * s)
    print(f"    {d}  {100 * s:6.2f}%  {bar}")
print("\n  ...and daily around the boundary:")
for d, s in shares:
    if first100 and abs((d - first100).days) <= 8:
        print(f"    {d}  {100 * s:7.3f}%")
R["first_100pct_trailing30"] = str(first100)

# ============================================================================
hdr("B -- MAXIMUM ACTIVITY GAP INSIDE THE LAST K DAYS")
for K in [30, 60, 90, 120, 150, 180, 409]:
    lo = 409 - K
    blk = P[:, lo:]
    # longest run of zeros per user within the block, plus leading/trailing distance
    maxgap = np.zeros(N, np.int32)
    run = np.zeros(N, np.int32)
    for j in range(blk.shape[1]):
        run = np.where(blk[:, j], 0, run + 1)
        maxgap = np.maximum(maxgap, run)
    nact = blk.sum(1)
    print(f"  last {K:3d} days: users with 0 activity = {int((nact == 0).sum()):>7,} | "
          f"max gap  p50={np.median(maxgap):5.0f}  p90={np.quantile(maxgap, .9):5.0f}  "
          f"p99={np.quantile(maxgap, .99):5.0f}  MAX={maxgap.max():5.0f}")
print("\n  -> if MAX gap over the last 90 days is < 30 for every user, the inclusion rule")
print("     is a no-30-day-gap-in-the-last-90-days condition, not a single 30-day filter.")

# ============================================================================
hdr("C -- ACTIVITY IN EACH DISJOINT 30-DAY BLOCK BACK FROM THE CUT-OFF")
print(f"  {'block':28s} {'share active':>13s} {'share with gmv>0':>18s}")
blocks = []
for k in range(0, 13):
    b = 408 - 30 * k
    a = b - 29
    if a < 0:
        break
    act = active_in(a, b)
    g = gmv[:, max(a, 0):b + 1].sum(1)
    blocks.append({"k": k, "start": str(D[max(a, 0)]), "end": str(D[b]),
                   "share_active": float(act.mean()), "share_buy": float((g > 0).mean())})
    print(f"  k={k:2d} [{D[max(a, 0)]}..{D[b]}] {100 * act.mean():12.2f}% {100 * (g > 0).mean():17.2f}%")
R["blocks"] = blocks

# ============================================================================
hdr("D -- IS THE 'GROWTH' REAL OR A SELECTION ARTEFACT?")
print("  Re-select a cohort by the SAME rule at an earlier date S (active in [S-29,S])")
print("  and track its daily active share before and after S. A peak AT S means selection.\n")
for S_i in [180, 270, 330]:
    S = D[S_i]
    coh = active_in(S_i - 29, S_i)
    sub_daily = P[coh].mean(0)
    pts = [S_i - x for x in (150, 120, 90, 60, 30, 0)] + [S_i + x for x in (30, 60, 90, 120)]
    pts = [p for p in pts if 0 <= p < 409]
    print(f"  S={S} cohort n={int(coh.sum()):,}")
    print("     " + "  ".join(f"{D[p]}:{100 * sub_daily[p]:5.2f}%" for p in pts))
print("\n  full panel daily active share for reference:")
allp = P.mean(0)
print("     " + "  ".join(f"{D[p]}:{100 * allp[p]:5.2f}%" for p in range(0, 409, 45)))

# ============================================================================
hdr("E -- FINAL SAFE-ANCHOR LIST AND THE BASELINE IT IMPLIES")
GUARD = first100  # target windows must end strictly before this date
print(f"  guaranteed-activity zone starts at {GUARD}")
print(f"  => a CV target window [A+1, A+30] is clean only if A+30 < {GUARD}, i.e. A <= "
      f"{GUARD - timedelta(days=31)}\n")
rows = []
for k in range(0, 12):
    A_i = 378 - 30 * k          # 378 = index of 2026-01-14
    if A_i - 89 < 0:
        break
    A = D[A_i]
    keep = active_in(A_i - 29, A_i)
    y = gmv[:, A_i + 1:A_i + 31].sum(1)[keep].astype(np.float64)
    p30 = gmv[:, A_i - 29:A_i + 1].sum(1)[keep].astype(np.float64)
    p60 = gmv[:, A_i - 59:A_i - 29].sum(1)[keep].astype(np.float64)
    p90 = gmv[:, A_i - 89:A_i - 59].sum(1)[keep].astype(np.float64)
    geo = np.expm1((np.log1p(p30) + np.log1p(p60) + np.log1p(p90)) / 3)
    clean = D[min(A_i + 30, 408)] < GUARD
    tgt_act = float((P[:, A_i + 1:A_i + 31].sum(1) > 0)[keep].mean())
    r = {"anchor": str(A), "clean": bool(clean), "n": int(keep.sum()),
         "p_active_in_target": tgt_act, "zero_share": float((y <= 0).mean()),
         "rmsle_p30": rmsle(y, p30), "rmsle_geo3": rmsle(y, geo)}
    rows.append(r)
    print(f"  A={A} {'CLEAN ' if clean else 'DIRTY '} n={r['n']:>7,} "
          f"P(act in tgt)={100 * tgt_act:6.2f}%  zero={100 * r['zero_share']:5.2f}%  "
          f"RMSLE p30={r['rmsle_p30']:.4f} geo3={r['rmsle_geo3']:.4f}")
R["anchor_table"] = rows
cl = [r for r in rows if r["clean"]]
if cl:
    v = np.array([r["rmsle_geo3"] for r in cl])
    z = np.array([r["zero_share"] for r in cl])
    print(f"\n  CLEAN anchors: {len(cl)}  ({cl[-1]['anchor']} .. {cl[0]['anchor']})")
    print(f"  geo3 RMSLE over clean anchors : mean={v.mean():.4f} std={v.std():.4f}")
    print(f"  zero-share over clean anchors : mean={z.mean():.4f} std={z.std():.4f}")
dirty = [r for r in rows if not r["clean"]]
if dirty and cl:
    vd = np.array([r["rmsle_geo3"] for r in dirty])
    print(f"  geo3 RMSLE over DIRTY anchors : mean={vd.mean():.4f}  "
          f"=> optimism bias {v.mean() - vd.mean():+.4f} RMSLE")

# ============================================================================
hdr("F -- TEST-TIME POPULATION SANITY")
A_i = 408
keep = active_in(A_i - 29, A_i)
print(f"  test anchor {D[A_i]}: users satisfying 'active in [A-29,A]' = {int(keep.sum()):,} (must be 250,000)")
print(f"  the model must score ALL 250,000 users regardless -- the rule is already satisfied by all.")
p30 = gmv[:, A_i - 29:A_i + 1].sum(1).astype(np.float64)
p60 = gmv[:, A_i - 59:A_i - 29].sum(1).astype(np.float64)
p90 = gmv[:, A_i - 89:A_i - 59].sum(1).astype(np.float64)
geo = np.expm1((np.log1p(p30) + np.log1p(p60) + np.log1p(p90)) / 3)
print(f"  test-time feature stats: p30 zero-share={100 * (p30 == 0).mean():.2f}%  "
      f"sum(p30)={p30.sum():,.0f}")
print(f"  geo3 prediction for the real test: sum={geo.sum():,.0f}  "
      f"mean={geo.mean():.2f}  zero-share={100 * (geo == 0).mean():.2f}%")
print(f"  last observed 30d GMV total = {p30.sum():,.0f}")
print(f"  2025 analogue seasonal ratio (Feb14-Mar15)/(Jan15-Feb13) = 1.1628")
print(f"  => a *level-calibrated* point estimate of test-window total GMV is "
      f"{p30.sum() * 1.1628:,.0f}")

with open(OUT / "eda_rule.json", "w") as f:
    json.dump(R, f, indent=2, default=str)
hdr("DONE")
print(f"total runtime {time.time() - T0:.0f}s")
