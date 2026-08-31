"""What is the BEST blend obtainable from OOF files we already have? No new training."""
import sys, itertools
import numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "src"))
from metrics import rmsle

E0120 = ["e0049", "e0064", "e0100", "e0101", "e0101s1", "e0101s2", "e0101s3", "e0102", "e0108"]
ALL = ["e0020", "e0049", "e0064", "e0100", "e0101", "e0101s1", "e0101s2", "e0101s3", "e0102",
       "e0108", "e0110", "e0170", "e0170ey"] + [f"e018{i}" for i in range(8)]
# e0111 (cnngru) EXCLUDED: oof/e0111.parquet is truncated (2.8MB vs ~13.9MB) and unreadable.

base = pd.read_parquet(ROOT / "oof" / "e0049.parquet").sort_values(["fold_id", "user_id"])
y, fold = base.y_true.values, base.fold_id.values
uid = base.user_id.values
Ly = np.log1p(y)
FOLDS = np.unique(fold)

cache, skipped = {}, []
for e in ALL:
    d = pd.read_parquet(ROOT / "oof" / f"{e}.parquet").sort_values(["fold_id", "user_id"])
    if len(d) != len(base) or not np.array_equal(d.user_id.values, uid):
        skipped.append((e, len(d))); continue
    cache[e] = np.log1p(d.y_pred.values)
print(f"  usable on the frozen folds: {len(cache)}   skipped: {skipped}\n")


def cal(Lp, fit=None):
    """per-fold optimal affine calibration; `fit` = folds to fit a,b on (None = in-fold)"""
    sc = []
    for k in FOLDS:
        m = fold == k
        if fit is None:
            a, b = np.polyfit(Lp[m], Ly[m], 1)
        else:
            f = np.isin(fold, fit)
            a, b = np.polyfit(Lp[f], Ly[f], 1)
        sc.append(rmsle(np.expm1(Ly[m]), np.expm1(a * Lp[m] + b)))
    return float(np.mean(sc)), sc


def rho(Lp):
    return float(np.corrcoef(Ly, Lp)[0, 1])


print(f"  {'member':12s} {'rho':>9s}  {'calibrated':>10s}")
for e in sorted(cache, key=lambda e: -rho(cache[e])):
    print(f"  {e:12s} {rho(cache[e]):>9.5f}  {cal(cache[e])[0]:>10.5f}"
          + ("   <- in e0120" if e in E0120 else ""))

M0 = np.mean([cache[e] for e in E0120], axis=0)
r0, c0 = rho(M0), cal(M0)[0]
print(f"\n  e0120 as built (9 equal):  rho {r0:.5f}   cal {c0:.5f}")

# ---------------------------------------------------------------- greedy equal-weight
print(f"\n  === greedy forward selection, EQUAL weights (no fitted parameters) ===")
sel, best = [], None
pool = list(cache)
for step in range(12):
    cands = []
    for e in pool:
        if e in sel:
            continue
        M = np.mean([cache[x] for x in sel + [e]], axis=0)
        cands.append((cal(M)[0], rho(M), e))
    cands.sort()
    s, r, e = cands[0]
    if best is not None and s >= best - 1e-7:
        print(f"  stop: no member improves ({cands[0][2]} would give {s:.5f})")
        break
    sel.append(e); best = s
    print(f"  {step+1:>2d}. +{e:12s} rho {r:.5f}  cal {s:.5f}   (vs e0120 {s - c0:+.5f})")
Mg = np.mean([cache[e] for e in sel], axis=0)
print(f"\n  greedy subset ({len(sel)}): {sel}")
print(f"  rho {rho(Mg):.5f} ({rho(Mg) - r0:+.5f})   cal {cal(Mg)[0]:.5f} ({cal(Mg)[0] - c0:+.5f})")

# ---------------------------------------------------------------- honest: LOFO selection
print(f"\n  === same greedy, but the SUBSET is chosen on 4 folds and scored on the 5th ===")
tot = []
for hold in FOLDS:
    tr = [k for k in FOLDS if k != hold]
    s_, b_ = [], None
    for _ in range(12):
        cs = []
        for e in cache:
            if e in s_:
                continue
            M = np.mean([cache[x] for x in s_ + [e]], axis=0)
            sc = np.mean([cal(M)[1][k] for k in tr])
            cs.append((sc, e))
        cs.sort()
        if b_ is not None and cs[0][0] >= b_ - 1e-7:
            break
        b_ = cs[0][0]; s_.append(cs[0][1])
    M = np.mean([cache[e] for e in s_], axis=0)
    held = cal(M)[1][hold]
    base_held = cal(M0)[1][hold]
    tot.append(held - base_held)
    print(f"  hold f{hold}: picked {len(s_):>2d} -> {s_[:6]}{'...' if len(s_) > 6 else ''}")
    print(f"            held-out {held:.5f}  vs e0120 {base_held:.5f}   {held - base_held:+.5f}")
print(f"\n  mean honest gain over e0120: {np.mean(tot):+.5f}")

# ---------------------------------------------------------------- the unconstrained trap
print(f"\n  === unconstrained OLS over all {len(cache)} members ===")
X = np.column_stack([cache[e] for e in cache])
w = np.linalg.lstsq(np.column_stack([X, np.ones(len(X))]), Ly, rcond=None)[0]
ins = cal(X @ w[:-1] + w[-1])[0]
oof = []
for hold in FOLDS:
    f = fold != hold
    ww = np.linalg.lstsq(np.column_stack([X[f], np.ones(f.sum())]), Ly[f], rcond=None)[0]
    P = X @ ww[:-1] + ww[-1]
    oof.append(cal(P)[1][hold])
print(f"  in-sample {ins:.5f}   leave-one-fold-out {np.mean(oof):.5f}   "
      f"vs e0120 {np.mean(oof) - c0:+.5f}")
