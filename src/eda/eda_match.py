#!/usr/bin/env python
"""
E-CUP 2026 / Task 3 -- why does CV sit above the LB, and what population is right?

The re-selection rule of DATA.md §4.4 ("keep users active in [A-29, A]") matched CV folds
to the test on ACTIVITY. But RMSLE is driven by the joint distribution of
(feature, target), and the dominant feature is p30 = GMV in the last 30 days.
Being active is not the same as having bought: 45.93 % of the test users are active yet
have p30 = 0, and a user with p30 = 0 who also has y = 0 contributes exactly zero error.
How many such "free" users a fold contains sets its RMSLE level.

Crucially the TEST FEATURES ARE FULLY OBSERVABLE -- only the target is hidden. So the
fold population can be matched to the test's feature distribution offline, with no
submissions spent.

Outputs
  1  feature distribution of the real test anchor vs each CV anchor x population
  2  RMSLE re-weighted to the test's feature-cell frequencies (covariate-shift correction)
  3  which population choice reproduces the observed LB of 2.12
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
LB = 2.12
R: dict = {}
T0 = time.time()


def hdr(t):
    print(f"\n{'=' * 78}\n{t}   [t+{time.time() - T0:.0f}s]\n{'=' * 78}", flush=True)


def sub(t):
    print(f"\n--- {t} " + "-" * max(0, 70 - len(t)), flush=True)


df = pl.read_parquet(ROOT / "data" / "train.parquet").select(
    ["user_id", "event_date", "gmv", "to_ord"])
DMIN, DMAX = df["event_date"].min(), df["event_date"].max()
users = np.sort(df["user_id"].unique().to_numpy())
N = users.size
ui = np.searchsorted(users, df["user_id"].to_numpy())
di = (df["event_date"].to_numpy().astype("datetime64[D]") - np.datetime64(DMIN)).astype(int)
P = np.zeros((N, 409), bool); P[ui, di] = True
G = np.zeros((N, 409), np.float32); np.add.at(G, (ui, di), df["gmv"].to_numpy().astype(np.float32))
O = np.zeros((N, 409), np.float32); np.add.at(O, (ui, di), df["to_ord"].to_numpy().astype(np.float32))
D = [DMIN + timedelta(days=i) for i in range(409)]
cs = np.concatenate([np.zeros((N, 1), np.int32), np.cumsum(P, 1, dtype=np.int32)], 1)
print(f"  matrices built ({N} x 409)")

TEST_A = 408                                    # 2026-02-13
ANCH = [378 - 30 * k for k in range(0, 10)]     # 2026-01-14 .. 2025-04-19
CLEAN = D.index(date(2025, 10, 16))


def act(a, b):
    return (cs[:, b + 1] - cs[:, max(a, 0)]) > 0


def feats(A):
    """the features that matter, as of anchor index A"""
    p30 = G[:, max(A - 29, 0):A + 1].sum(1).astype(np.float64)
    o30 = O[:, max(A - 29, 0):A + 1].sum(1).astype(np.float64)
    d30 = P[:, max(A - 29, 0):A + 1].sum(1).astype(np.float64)
    return p30, o30, d30


def cells(p30, o30):
    """20 feature cells: p30 bucket (0 + quartiles of positive) x orders bucket"""
    pb = np.zeros(p30.shape, int)
    pos = p30 > 0
    if pos.any():
        qs = np.quantile(P30_REF[P30_REF > 0], [.25, .5, .75])
        pb[pos] = 1 + np.digitize(p30[pos], qs)
    ob = np.digitize(o30, [1, 2, 5])            # 0 | 1 | 2-4 | 5+
    return pb * 4 + ob


# reference quartiles come from the TEST anchor so cells mean the same thing everywhere
P30_REF, O30_REF, D30_REF = feats(TEST_A)

# ============================================================================
hdr("1 -- FEATURE DISTRIBUTION: the real test anchor vs each CV fold")
print("  (features are observable at test time; only the target is hidden)\n")
print(f"  {'anchor':12s} {'pop':>9s} {'n':>8s} {'p30=0':>7s} {'ord30=0':>8s} "
      f"{'E log1p(p30)':>13s} {'E days30':>9s} {'cell L1 dist':>13s}")
ref_cell = np.bincount(cells(P30_REF, O30_REF), minlength=20) / N
print(f"  {'2026-02-13':12s} {'TEST':>9s} {N:>8,} {100 * (P30_REF == 0).mean():>6.2f}% "
      f"{100 * (O30_REF == 0).mean():>7.2f}% {np.log1p(P30_REF).mean():>13.4f} "
      f"{D30_REF.mean():>9.2f} {0.0:>13.4f}")
tbl = []
for A in ANCH:
    p30, o30, d30 = feats(A)
    for pop in ["all250k", "reselect"]:
        k = np.ones(N, bool) if pop == "all250k" else act(A - 29, A)
        c = np.bincount(cells(p30[k], o30[k]), minlength=20) / k.sum()
        l1 = float(np.abs(c - ref_cell).sum())
        row = {"anchor": str(D[A]), "pop": pop, "n": int(k.sum()),
               "p30_zero": float((p30[k] == 0).mean()), "ord_zero": float((o30[k] == 0).mean()),
               "E_logp30": float(np.log1p(p30[k]).mean()), "E_days": float(d30[k].mean()),
               "cell_l1": l1}
        tbl.append(row)
        print(f"  {row['anchor']:12s} {pop:>9s} {row['n']:>8,} {100 * row['p30_zero']:>6.2f}% "
              f"{100 * row['ord_zero']:>7.2f}% {row['E_logp30']:>13.4f} {row['E_days']:>9.2f} "
              f"{l1:>13.4f}")
R["feature_match"] = tbl
best = min(tbl, key=lambda r: r["cell_l1"])
print(f"\n  closest feature match to the test anchor: {best['anchor']} / {best['pop']} "
      f"(L1 = {best['cell_l1']:.4f})")

# ============================================================================
hdr("2 -- RMSLE RE-WEIGHTED TO THE TEST'S FEATURE DISTRIBUTION")
print("  Weighted RMSLE = sqrt( sum_cell  w_test(cell) * mean squared log-error in cell ).")
print("  This is what the p30 baseline would score on a fold whose feature mix equals")
print("  the test's -- directly comparable to the LB.\n")
print(f"  {'anchor':12s} {'clean':6s} {'pop':>9s} {'raw':>8s} {'reweighted':>11s} "
      f"{'vs LB':>8s} {'cells used':>11s}")
res = []
for A in ANCH:
    p30, o30, d30 = feats(A)
    y = G[:, A + 1:A + 31].sum(1).astype(np.float64)
    for pop in ["all250k", "reselect"]:
        k = np.ones(N, bool) if pop == "all250k" else act(A - 29, A)
        cc = cells(p30[k], o30[k])
        se = (np.log1p(y[k]) - np.log1p(p30[k])) ** 2
        raw = float(np.sqrt(se.mean()))
        num, wsum_, used = 0.0, 0.0, 0
        for c in range(20):
            m = cc == c
            if m.sum() >= 30 and ref_cell[c] > 0:
                num += ref_cell[c] * se[m].mean(); wsum_ += ref_cell[c]; used += 1
        rw = float(np.sqrt(num / wsum_)) if wsum_ > 0 else np.nan
        res.append({"anchor": str(D[A]), "clean": A <= CLEAN, "pop": pop,
                    "raw": raw, "reweighted": rw})
        print(f"  {str(D[A]):12s} {str(A <= CLEAN):6s} {pop:>9s} {raw:>8.4f} {rw:>11.4f} "
              f"{rw - LB:>+8.4f} {used:>11d}")
R["reweighted"] = res
for pop in ["all250k", "reselect"]:
    v = np.array([r["reweighted"] for r in res if r["clean"] and r["pop"] == pop])
    w = np.array([r["raw"] for r in res if r["clean"] and r["pop"] == pop])
    print(f"\n  CLEAN anchors, {pop:>9s}: raw mean={w.mean():.4f}  reweighted mean={v.mean():.4f} "
          f"(sd {v.std():.4f})  LB={LB}  diff={v.mean() - LB:+.4f}")

# ============================================================================
hdr("3 -- WHAT DOES THE REWEIGHTING ACTUALLY CHANGE?")
A = CLEAN
p30, o30, d30 = feats(A)
y = G[:, A + 1:A + 31].sum(1).astype(np.float64)
k = act(A - 29, A)
print(f"  anchor {D[A]}, re-selected population (n={int(k.sum()):,})\n")
print(f"  {'cell (p30 bucket, ord bucket)':34s} {'w_fold':>8s} {'w_test':>8s} {'mean SE':>9s} {'contrib':>9s}")
cc = cells(p30[k], o30[k])
se = (np.log1p(y[k]) - np.log1p(p30[k])) ** 2
names_p = ["p30=0", "p30 Q1", "p30 Q2", "p30 Q3", "p30 Q4"]
names_o = ["ord=0", "ord=1", "ord2-4", "ord5+"]
for c in range(20):
    m = cc == c
    if m.sum() < 30:
        continue
    wf = m.mean()
    print(f"  {names_p[c // 4] + ', ' + names_o[c % 4]:34s} {wf:>8.4f} {ref_cell[c]:>8.4f} "
          f"{se[m].mean():>9.4f} {ref_cell[c] * se[m].mean():>9.4f}")
print("\n  -> the fold and the test differ most in the zero-feature cells, and those cells")
print("     carry the lowest squared error. Getting their weight wrong moves the LEVEL of")
print("     the metric without saying anything about model quality.")

# ============================================================================
hdr("4 -- CONCLUSION FOR THE PROTOCOL")
cl_rs = np.array([r["reweighted"] for r in res if r["clean"] and r["pop"] == "reselect"])
cl_al = np.array([r["reweighted"] for r in res if r["clean"] and r["pop"] == "all250k"])
raw_rs = np.array([r["raw"] for r in res if r["clean"] and r["pop"] == "reselect"])
raw_al = np.array([r["raw"] for r in res if r["clean"] and r["pop"] == "all250k"])
print(f"  observed public LB for p30            : {LB}")
print(f"  clean/reselect  raw                   : {raw_rs.mean():.4f}   ({raw_rs.mean() - LB:+.4f})")
print(f"  clean/all250k   raw                   : {raw_al.mean():.4f}   ({raw_al.mean() - LB:+.4f})")
print(f"  clean/reselect  reweighted to test    : {cl_rs.mean():.4f}   ({cl_rs.mean() - LB:+.4f})")
print(f"  clean/all250k   reweighted to test    : {cl_al.mean():.4f}   ({cl_al.mean() - LB:+.4f})")
print("\n  If the two reweighted columns agree with each other and with the LB, the")
print("  population choice was a LEVEL artefact and feature-reweighting removes it.")
print("  If a gap survives, the test period itself differs and only a submission can say how.")

with open(OUT / "eda_match.json", "w") as f:
    json.dump(R, f, indent=2, default=str)
hdr("DONE")
print(f"total runtime {time.time() - T0:.0f}s")
