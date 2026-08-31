#!/usr/bin/env python
"""Verdict for the TabICL member: r, rho_partial, and the blend test. Pre-registered bars."""
import numpy as np, pandas as pd, sys
from pathlib import Path
EXP = sys.argv[1] if len(sys.argv) > 1 else "e0340"
SD = 2.3178
d = pd.read_parquet('oof/e0049.parquet').sort_values(["fold_id","user_id"])
x = pd.read_parquet('oof/usercv_full.parquet')
j = d[['fold_id','user_id','y_true']].merge(x, on=['fold_id','user_id'], how='inner', suffixes=('','_u'))
key = j[['fold_id','user_id']]; Ly = np.log1p(j.y_true.values); fold = j.fold_id.values
def gk(e):
    a = pd.read_parquet(f'oof/{e}.parquet')[['fold_id','user_id','y_pred']]
    v = key.merge(a, on=['fold_id','user_id'], how='left').y_pred.values
    return np.log1p(v) if not np.isnan(v).any() else None
seq = np.mean([gk(e) for e in ['e0100','e0101','e0101s1','e0101s2','e0101s3','e0102','e0108']], axis=0)
CH = 0.20*np.mean([gk('e0266'), gk('e0064')], axis=0) + 0.38*seq + 0.42*gk('usercv_full')
rc = float(np.corrcoef(Ly, CH)[0,1])
need = float(np.sqrt(((rc+0.00070)**2 - rc**2)/(1-rc**2)))
v = gk(EXP)
if v is None:
    sys.exit(f"{EXP}: OOF has NaN on the intersection")
rb = float(np.corrcoef(Ly, v)[0,1]); r = float(np.corrcoef(CH, v)[0,1])
rp = (rb - r*rc)/(np.sqrt(1-rc**2)*np.sqrt(1-r**2))
print(f"\n=== {EXP} VERDICT ===")
print(f"  champion rho   {rc:.5f}      bar rho_partial {need:.5f}")
print(f"  member rho_B   {rb:.5f}")
print(f"  r vs champion  {r:.5f}   <<< THE decision number (GBDTs all sit at 0.998)")
print(f"  rho_partial    {rp:+.5f}   = {100*rp/need:.1f}% of the bar")
print(f"  VERDICT: {'CLEARS THE BAR' if rp >= need else 'below the bar'}")
# blend test, weights fixed then re-optimised out of fold
def sc(z): return np.array([np.corrcoef(Ly[fold==k], z[fold==k])[0,1] for k in sorted(set(fold))])
b = sc(CH)
per, ws = [], []
for k in sorted(set(fold)):
    tr = fold != k; te = fold == k
    grid = np.linspace(0, 0.5, 51)
    s = [np.corrcoef(Ly[tr], ((1-w)*CH[tr] + w*v[tr]))[0,1] for w in grid]
    w = float(grid[int(np.argmax(s))]); ws.append(w)
    per.append(np.corrcoef(Ly[te], (1-w)*CH[te] + w*v[te])[0,1])
per = np.array(per); dd = per.mean() - b.mean()
print(f"\n  LOFO blend test: d {dd:+.6f}  wins {int((per>b).sum())}/5  weights {np.round(ws,2).tolist()}")
print(f"  projected LB {SD*np.sqrt(1-(0.70379+dd)**2):.6f}   (champion 1.646589, target 1.6450)")
