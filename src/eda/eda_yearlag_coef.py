#!/usr/bin/env python
"""
Measure the year-lag adjustment coefficient at anchor 2026-01-14.

Anchor 2026-01-14 is the ONLY anchor with both
  * a complete target window   : 2026-01-15 .. 2026-02-13
  * an available year-lag window: 2025-01-15 .. 2025-02-13  (same calendar position)

so it is the only place the effect can be measured against a real model residual rather
than in the abstract. Caveat: that target window sits inside the guaranteed-activity zone
(DATA.md §4), so its LEVEL is optimistically biased. A *relative* coefficient on one extra
feature should be far less affected than the level is, but it is a real caveat and the
number should be read as indicative, not as a validated CV gain.

Procedure
  1. train the e0020 recipe on clean anchors <= 2025-10-16 (targets end <= 2025-11-15,
     so the model never sees the 2026-01-15..02-13 target)
  2. predict at anchor 2026-01-14
  3. residual = log1p(y) - log1p(pred)
  4. regress the residual on each candidate lag feature and report the coefficient, the
     t-stat and the RMSLE the adjustment would deliver

Candidate lag definitions include the narrow pre-holiday run-ups, because that is the
mechanism proposed: gift buying happens in the 1-2 weeks BEFORE 23 Feb and 8 March
(reports/holiday.log measured the peak at d-7 and d-5 respectively).
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import yaml

ROOT = Path("/path/to/ecup")
sys.path.insert(0, str(ROOT / "src"))
from data import Panel                      # noqa: E402
from features import build                  # noqa: E402
from metrics import rmsle                   # noqa: E402

OUT = ROOT / "reports" / "eda"
R: dict = {}
GUARD_START, TRAIN_STRIDE, MIN_HISTORY = date(2025, 11, 16), 7, 90


def hdr(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}", flush=True)


cfg = yaml.safe_load((ROOT / "configs" / "e0020_fixedrounds.yaml").read_text())
p = Panel()
import lightgbm as lgb

A = date(2026, 1, 14)
ai = p.idx(A)
hdr("1 -- TRAIN THE e0020 RECIPE, PREDICT AT 2026-01-14")
latest, earliest = GUARD_START - timedelta(days=31), p.dmin + timedelta(days=MIN_HISTORY - 1)
anchors, a = [], latest
while a >= earliest:
    anchors.append(a); a -= timedelta(days=TRAIN_STRIDE)
anchors = sorted(anchors)
print(f"  training anchors: {len(anchors)} ({anchors[0]}..{anchors[-1]}); "
      f"their targets end <= {anchors[-1] + timedelta(days=30)}, before the eval window")

Xtr, ytr = [], []
for x in anchors:
    xi = p.idx(x); keep = p.active_in(xi - 29, xi)
    Xb, names = build(p, xi, keep, cfg["feature_blocks"])
    Xtr.append(Xb); ytr.append(np.log1p(p.target(xi)[keep]))
Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr)
params = dict(cfg["lgb_params"]); params["seed"] = cfg["seed"]
model = lgb.train(params, lgb.Dataset(Xtr, ytr, feature_name=names),
                  num_boost_round=int(cfg["fixed_rounds"]), callbacks=[lgb.log_evaluation(0)])

keep = p.active_in(ai - 29, ai)
Xev, _ = build(p, ai, keep, cfg["feature_blocks"])
y = p.target(ai)[keep]
pred = np.maximum(np.expm1(model.predict(Xev)), 0.0)
L, P = np.log1p(y), np.log1p(pred)
resid = L - P
print(f"  eval users {int(keep.sum()):,}   RMSLE at this anchor = {rmsle(y, pred):.5f}")
print(f"  residual mean {resid.mean():+.4f}  sd {resid.std():.4f}")

hdr("2 -- CANDIDATE YEAR-LAG FEATURES (all from the SAME calendar window one year back)")
W = lambda a1, b1: p.wsum("gmv", p.idx(a1), p.idx(b1))
REF = np.log1p(p.wsum("gmv", p.idx(date(2025, 6, 1)), p.idx(date(2025, 12, 15))) / 197)

cands = {
    "lag_full_month  (2025-01-15..02-13)": np.log1p(W(date(2025, 1, 15), date(2025, 2, 13))),
    "lag_last2w      (2025-01-31..02-13)": np.log1p(W(date(2025, 1, 31), date(2025, 2, 13))),
    "lag_first2w     (2025-01-15..01-28)": np.log1p(W(date(2025, 1, 15), date(2025, 1, 28))),
    "lag_spike_month (vs own reference)":  np.log1p(W(date(2025, 1, 15), date(2025, 2, 13)) / 30) - REF,
    "lag_spike_last2w(vs own reference)":  np.log1p(W(date(2025, 1, 31), date(2025, 2, 13)) / 14) - REF,
}
print(f"  {'candidate':38s} {'coef':>9s} {'t-stat':>9s} {'corr w/ resid':>14s} {'RMSLE after':>12s}")
best = None
for nm, v in cands.items():
    v = v[keep]
    Xd = np.column_stack([np.ones(v.size), v])
    coef, *_ = np.linalg.lstsq(Xd, resid, rcond=None)
    fit = Xd @ coef
    se = np.sqrt(((resid - fit) ** 2).sum() / (v.size - 2) / max(((v - v.mean()) ** 2).sum(), 1e-9))
    adj = np.maximum(np.expm1(P + fit), 0.0)
    s = rmsle(y, adj)
    print(f"  {nm:38s} {coef[1]:>9.4f} {coef[1] / se:>9.1f} "
          f"{np.corrcoef(v, resid)[0, 1]:>14.4f} {s:>12.5f}")
    R[nm] = {"coef": float(coef[1]), "t": float(coef[1] / se), "rmsle_after": float(s)}
    if best is None or s < best[1]:
        best = (nm, s)
print(f"\n  baseline RMSLE at this anchor       = {rmsle(y, pred):.5f}")
print(f"  best adjusted                        = {best[1]:.5f}  ({best[0]})")
print(f"  gain                                 = {best[1] - rmsle(y, pred):+.5f}")
R["baseline_rmsle"] = float(rmsle(y, pred))
R["best"] = {"name": best[0], "rmsle": float(best[1])}

hdr("3 -- IS THIS JUST THE 365-DAY WINDOW IN DISGUISE?")
p365 = np.log1p(p.wsum("gmv", ai - 364, ai))[keep]
v = cands["lag_full_month  (2025-01-15..02-13)"][keep]
Xd = np.column_stack([np.ones(v.size), p365])
c1, *_ = np.linalg.lstsq(Xd, resid, rcond=None)
r1 = resid - Xd @ c1
Xd2 = np.column_stack([np.ones(v.size), p365, v])
c2, *_ = np.linalg.lstsq(Xd2, resid, rcond=None)
r2 = resid - Xd2 @ c2
print(f"  residual SS after p365 only         = {(r1 ** 2).sum():,.1f}")
print(f"  residual SS after p365 + year-lag   = {(r2 ** 2).sum():,.1f}")
print(f"  incremental R^2 of the year-lag     = {1 - (r2 ** 2).sum() / (r1 ** 2).sum():.6f}")
print(f"  year-lag coefficient given p365     = {c2[2]:+.4f}")

hdr("4 -- CAVEAT")
print("  This anchor's target window is inside the guaranteed-activity zone, so its LEVEL is")
print("  optimistic by ~0.04 RMSLE (DATA.md §4.3). The coefficient is a relative quantity and")
print("  should be much less affected, but this is ONE anchor and cannot be cross-validated.")
print("  Treat the number as a magnitude estimate to size an LB test, not as a CV result.")

(OUT).mkdir(parents=True, exist_ok=True)
(OUT / "yearlag_coef.json").write_text(json.dumps(R, indent=2, default=str))
print("\n  wrote reports/eda/yearlag_coef.json")
