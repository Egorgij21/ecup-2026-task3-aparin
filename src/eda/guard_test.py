#!/usr/bin/env python
"""
Should guard-zone anchors be used as TRAINING data?

We exclude every anchor whose 30-day target window touches the guaranteed-activity zone
(from 2025-11-16). That is unarguably right for VALIDATION -- DATA.md §4 measures +0.041 of
optimism. But it was then applied to training too, and those are different questions. The
evidence that they are different:

    anchor       zone    P(target=0)   mean log1p
    2025-10-16   CLEAN      42.8%         2.4417
    2025-12-15   GUARD      43.7%         2.4170
    2026-01-14   GUARD      45.9%         2.2421

The target distribution barely moves, because the guarantee is on ACTIVITY (any event), not
on PURCHASE -- and ~44% of active users buy nothing regardless. So the objection "the model
would learn that everyone stays active" is not supported by the data.

Rolling-origin CV cannot settle this: guard anchors sit after every validation anchor, so
using them to predict 2025-10-16 would be training on the future. Instead, validate at the
LATEST fully-observed anchor, which is also the most test-like one available:

    validation anchor 2026-01-14, target 2026-01-15 .. 2026-02-13
    A: train anchors <= 2025-10-16   (29 anchors -- the current recipe)
    B: train anchors <= 2025-12-15   (37 anchors -- adds 8 recent ones)

B's last target window ends exactly on 2026-01-14, so the 30-day embargo holds and no
training row's target overlaps the validation features.

Both scores are optimistic in absolute terms (the validation population is guard-selected).
That does not matter: the two models are scored on IDENTICAL rows, so the delta is clean.
Read the delta, never the level.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import yaml

ROOT = Path("/path/to/ecup")
sys.path.insert(0, str(ROOT / "src"))
from data import Panel                      # noqa: E402
from features import build                  # noqa: E402
from metrics import rmsle                   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/e0049_nomoment.yaml")
ap.add_argument("--valid-anchor", default="2026-01-14")
ap.add_argument("--stride", type=int, default=7)
ap.add_argument("--min-history", type=int, default=90)
ap.add_argument("--cache", action="store_true")
args = ap.parse_args()

cfg = yaml.safe_load((ROOT / args.config).read_text())
if args.cache:
    import features as _f
    _f.enable_cache(True)

p = Panel()
import lightgbm as lgb                       # noqa: E402

VA = date.fromisoformat(args.valid_anchor)
vai = p.idx(VA)
ROUNDS = int(cfg["fixed_rounds"])
EMBARGO = 30


def hdr(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}", flush=True)


def anchors_upto(last: date):
    out, a = [], last
    earliest = p.dmin + timedelta(days=args.min_history - 1)
    while a >= earliest:
        out.append(a)
        a -= timedelta(days=args.stride)
    return sorted(out)


def select(names, mats):
    if cfg.get("feature_exclude_patterns"):
        import re as _re
        pats = [_re.compile(x) for x in cfg["feature_exclude_patterns"]]
        s = [i for i, n in enumerate(names) if not any(q.search(n) for q in pats)]
        mats = [m[:, s] for m in mats]; names = [names[i] for i in s]
    if cfg.get("feature_whitelist"):
        WL = set(json.loads((ROOT / cfg["feature_whitelist"]).read_text()))
        s = [i for i, n in enumerate(names) if n in WL]
        mats = [m[:, s] for m in mats]; names = [names[i] for i in s]
    return names, mats


hdr("0 -- VALIDATION SET (the most test-like fully-observed anchor)")
vkeep = p.active_in(vai - 29, vai)
Xva, names = build(p, vai, vkeep, cfg["feature_blocks"])
yva = p.target(vai)[vkeep]
print(f"  anchor {VA}  target {VA + timedelta(days=1)} .. {VA + timedelta(days=30)}")
print(f"  {int(vkeep.sum()):,} users, P(y=0) = {(yva <= 0).mean():.1%}, "
      f"mean log1p = {np.log1p(np.maximum(yva, 0)).mean():.4f}")

# the naive floor on THIS anchor, so the two models have a common reference
naive = np.maximum(Xva[:, names.index("geo3")].astype(np.float64), 0.0)
print(f"  naive geo3 on this anchor: RMSLE = {rmsle(yva, naive):.5f}")

results = {}
LAST = {"A_clean_only": date(2025, 10, 16), "B_plus_guard": VA - timedelta(days=EMBARGO)}
for tag, last in LAST.items():
    hdr(f"{tag}: train anchors <= {last}")
    anc = anchors_upto(last)
    # hard guarantee: no training target window may reach the validation anchor
    bad = [a for a in anc if (VA - a).days < EMBARGO]
    assert not bad, f"embargo violated by {bad}"
    print(f"  {len(anc)} anchors, {anc[0]} .. {anc[-1]}")
    t0 = time.time()
    Xs, ys = [], []
    for a in anc:
        ai = p.idx(a)
        k = p.active_in(ai - 29, ai)
        Xb, nb = build(p, ai, k, cfg["feature_blocks"])
        Xs.append(Xb); ys.append(np.log1p(p.target(ai)[k]))
    X = np.concatenate(Xs); y = np.concatenate(ys)
    del Xs, ys
    n2, (X2, Xv2) = select(nb, (X, Xva))
    print(f"  {X2.shape[0]:,} rows x {X2.shape[1]} features   "
          f"[build {(time.time() - t0) / 60:.1f}m]")
    params = dict(cfg["lgb_params"]); params["seed"] = cfg["seed"]
    m = lgb.train(params, lgb.Dataset(X2, y, feature_name=n2),
                  num_boost_round=ROUNDS, callbacks=[lgb.log_evaluation(0)])
    pred = np.maximum(np.expm1(m.predict(Xv2)), 0.0)
    r = float(rmsle(yva, pred))
    results[tag] = {"rmsle": r, "n_anchors": len(anc), "n_rows": int(X2.shape[0]),
                    "first": str(anc[0]), "last": str(anc[-1]),
                    "runtime_min": round((time.time() - t0) / 60, 1)}
    print(f"  RMSLE on {VA} = {r:.5f}")
    del X, X2, y

hdr("VERDICT")
a, b = results["A_clean_only"], results["B_plus_guard"]
d = b["rmsle"] - a["rmsle"]
print(f"  A (clean only, {a['n_anchors']} anchors, {a['n_rows']:,} rows) = {a['rmsle']:.5f}")
print(f"  B (+guard,     {b['n_anchors']} anchors, {b['n_rows']:,} rows) = {b['rmsle']:.5f}")
print(f"\n  delta (B - A) = {d:+.5f}   {'B WINS' if d < 0 else 'A wins'}")
print(f"  extra anchors bought by using the guard zone: "
      f"{b['n_anchors'] - a['n_anchors']} ({b['n_rows'] - a['n_rows']:,} rows)")
print("\n  Read the DELTA only. Both levels are optimistic -- the validation population is")
print("  guard-selected -- but the two models are scored on identical rows.")
if d < -0.0002:
    print("\n  -> recent targets DO help. The exclusion is costing us, and the final model")
    print("     should train on anchors up to 2026-01-14, not stop at 2025-10-16.")
elif d > 0.0002:
    print("\n  -> the contamination outweighs the recency. Keep the current cut-off.")
else:
    print("\n  -> no measurable difference; the 4 months of extra data are not worth much.")

out = ROOT / "reports" / "eda" / "guard_test.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"valid_anchor": str(VA), "results": results, "delta": d,
                           "config": args.config}, indent=2))
print(f"\n  wrote {out}")
