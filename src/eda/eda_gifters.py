#!/usr/bin/env python
"""
Is "gift buyer" a real user trait? A properly-powered retest.

The first pass (reports/holiday.log) used ONE pair of windows -- Feb-March gifting vs New
Year -- and found corr = +0.0016. That is a single noisy measurement: a per-user lift built
from two sparse windows is a terrible estimator, so a weak-but-real trait could hide under it.

This retest uses SEVEN celebration windows across the history and asks the question the right
way round, with classical test theory:

  * split-half reliability -- does a gifter score built from half the holidays predict the
    score built from the other half? This is the only question that matters. If the trait is
    real but each window is noisy, the SPLIT-HALF correlation will be clearly positive even
    when any single pair is not.
  * Spearman-Brown -- extrapolate the reliability of the full composite from the split-half.
  * disattenuation -- given the measured reliability, what is the largest TRUE correlation
    consistent with the observed single-pair value?
  * restriction to measurable users -- a user with no orders in either window contributes
    pure noise; the trait can only exist among people who actually buy.

If reliability is near zero, no amount of feature engineering can extract a gifter score,
because there is nothing stable to extract. If it is clearly positive, a per-user holiday
feature is justified for the test window even though our Jun-Oct CV folds cannot validate it.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path("/path/to/ecup")
sys.path.insert(0, str(ROOT / "src"))
from data import Panel                      # noqa: E402

OUT = ROOT / "reports" / "eda"
R: dict = {}

# celebration run-ups (spending happens BEFORE the date -- reports/holiday.log §1)
# each: name, holiday window, matched baseline window of the same length
HOL = [
    ("feb23_2025",  (date(2025, 2, 13), date(2025, 2, 22)), (date(2025, 1, 23), date(2025, 2, 1))),
    ("mar8_2025",   (date(2025, 2, 26), date(2025, 3, 7)),  (date(2025, 4, 2),  date(2025, 4, 11))),
    ("may9_2025",   (date(2025, 4, 30), date(2025, 5, 8)),  (date(2025, 6, 4),  date(2025, 6, 12))),
    ("sep1_2025",   (date(2025, 8, 22), date(2025, 8, 31)), (date(2025, 7, 23), date(2025, 8, 1))),
    ("nov11_2025",  (date(2025, 11, 4), date(2025, 11, 11)), (date(2025, 10, 7), date(2025, 10, 14))),
    ("blackfri_25", (date(2025, 11, 24), date(2025, 11, 30)), (date(2025, 10, 20), date(2025, 10, 26))),
    ("newyear_25",  (date(2025, 12, 15), date(2025, 12, 31)), (date(2025, 10, 25), date(2025, 11, 10))),
]


def hdr(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}", flush=True)


p = Panel()
lift, active, gmv_h = {}, {}, {}
for nm, (ha, hb), (ba, bb) in HOL:
    gh = p.wsum("gmv", p.idx(ha), p.idx(hb))
    gb = p.wsum("gmv", p.idx(ba), p.idx(bb))
    nh = (hb - ha).days + 1
    nb = (bb - ba).days + 1
    lift[nm] = np.log1p(gh / nh * 10) - np.log1p(gb / nb * 10)      # per-10-day rate
    active[nm] = (p.wbuy(p.idx(ha), p.idx(hb)) + p.wbuy(p.idx(ba), p.idx(bb))) > 0
    gmv_h[nm] = gh

hdr("1 -- THE WINDOWS")
print(f"  {'holiday':14s} {'window':26s} {'baseline':26s} {'lift(agg)':>10s} {'buyers':>9s}")
for nm, (ha, hb), (ba, bb) in HOL:
    gh, gb = gmv_h[nm].sum(), p.wsum("gmv", p.idx(ba), p.idx(bb)).sum()
    nh, nb = (hb - ha).days + 1, (bb - ba).days + 1
    print(f"  {nm:14s} {f'{ha}..{hb}':26s} {f'{ba}..{bb}':26s} "
          f"{(gh / nh) / (gb / nb):>10.4f} {int(active[nm].sum()):>9,}")

hdr("2 -- PAIRWISE CORRELATION OF PER-USER LIFTS")
names = [h[0] for h in HOL]
# restrict to users who bought in at least one window of BOTH holidays being compared
print("      " + "".join(f"{n[:9]:>11s}" for n in names))
C = np.zeros((len(names), len(names)))
for i, a in enumerate(names):
    row = []
    for j, b in enumerate(names):
        m = active[a] & active[b]
        C[i, j] = np.corrcoef(lift[a][m], lift[b][m])[0, 1] if m.sum() > 1000 else np.nan
        row.append(f"{C[i, j]:>11.4f}")
    print(f"  {a[:5]:5s} " + "".join(row))
off = C[~np.eye(len(names), dtype=bool)]
print(f"\n  mean off-diagonal correlation = {np.nanmean(off):+.4f}  "
      f"(max {np.nanmax(off):+.4f})")
R["pairwise_mean"] = float(np.nanmean(off))

hdr("3 -- SPLIT-HALF RELIABILITY OF A MULTI-HOLIDAY GIFTER SCORE  (the real test)")
buyers = np.ones(p.n_users, bool)
for nm in names:
    buyers &= active[nm]
print(f"  users who bought in at least one window of EVERY holiday pair: {int(buyers.sum()):,}")
for label, sel in [("all users", np.ones(p.n_users, bool)),
                   ("users measurable in every holiday", buyers)]:
    A = np.mean([lift[n][sel] for n in names[0::2]], axis=0)     # holidays 1,3,5,7
    B = np.mean([lift[n][sel] for n in names[1::2]], axis=0)     # holidays 2,4,6
    r = float(np.corrcoef(A, B)[0, 1])
    sb = 2 * r / (1 + r) if r > -1 else np.nan                  # Spearman-Brown
    print(f"\n  [{label}]  n={int(sel.sum()):,}")
    print(f"    split-half corr (odd vs even holidays) = {r:+.4f}")
    print(f"    Spearman-Brown reliability of the full 7-holiday score = {sb:+.4f}")
    if sb > 0.01:
        print(f"    max TRUE corr consistent with the observed single-pair 0.0016 = "
              f"{0.0016 / max(sb, 1e-6):+.4f}")
    R[f"splithalf_{label.replace(' ', '_')}"] = r

hdr("4 -- CONTROL: THE SAME TEST ON PLAIN SPENDING LEVEL")
print("  If the machinery works at all, plain spending level must show high split-half")
print("  reliability -- it is the most persistent user trait there is.\n")
lev = {nm: np.log1p(gmv_h[nm]) for nm in names}
for label, sel in [("all users", np.ones(p.n_users, bool)),
                   ("users measurable in every holiday", buyers)]:
    A = np.mean([lev[n][sel] for n in names[0::2]], axis=0)
    B = np.mean([lev[n][sel] for n in names[1::2]], axis=0)
    print(f"  [{label}] split-half corr of LEVEL = {np.corrcoef(A, B)[0, 1]:+.4f}")

hdr("5 -- VERDICT")
A = np.mean([lift[n][buyers] for n in names[0::2]], axis=0)
B = np.mean([lift[n][buyers] for n in names[1::2]], axis=0)
r = float(np.corrcoef(A, B)[0, 1])
print(f"  gifter-score split-half reliability (measurable users) = {r:+.4f}")
if r < 0.05:
    print("  -> NO stable gift-responsiveness trait exists at the resolution of this data.")
    print("     Seven celebrations, restricted to users who actually buy, still cannot")
    print("     reproduce a user's own holiday lift. A per-user gifter feature has nothing")
    print("     to learn; the holiday effect is aggregate-only.")
else:
    print("  -> A trait IS detectable. Build the gifter score from all pre-anchor holidays")
    print("     and add it as a feature; note it cannot be validated on Jun-Oct folds.")
(OUT).mkdir(parents=True, exist_ok=True)
(OUT / "gifters.json").write_text(json.dumps(R, indent=2, default=str))
print("\n  wrote reports/eda/gifters.json")
