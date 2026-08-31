#!/usr/bin/env python
"""
Export a compact, self-contained extract for the LOCAL exploration notebook.

The laptop has train.parquet (the raw events) but not the engineered features -- rebuilding
them locally means the Panel, polars, and a 250k x 409 prefix-sum build, which is not what a
notebook is for. So the cluster builds them once and ships a single small parquet.

Deliberately NOT the full 809 columns: that is 728 MB at float32 and the laptop has ~21 GB
free. The top --n-features by gain (among those that beat their null) covers everything the
pair grid and the feature-vs-target views need.

Notebooks are for LOOKING at data, never for producing a result that enters the log
(CLAUDE.md §8) -- so this ships facts, not fitted anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path("/path/to/ecup")
sys.path.insert(0, str(ROOT / "src"))
from data import Panel                      # noqa: E402
from features import build                  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/e0039_sbc.yaml")
ap.add_argument("--n-features", type=int, default=120)
ap.add_argument("--fold", type=int, default=-1)
ap.add_argument("--out", default="reports/local_extract.parquet")
args = ap.parse_args()

import yaml                                 # noqa: E402
cfg = yaml.safe_load((ROOT / args.config).read_text())
spec = json.loads((ROOT / "data" / "fold_spec.json").read_text())

imp = json.loads((ROOT / "reports" / "importance.json").read_text())
names_all, gain, score = imp["names"], np.asarray(imp["actual"], float), np.asarray(imp["score"], float)
eligible = np.flatnonzero(score > 0)
ranked = [int(i) for i in eligible[np.argsort(-gain[eligible])]]

p = Panel()
fs = spec["folds"][args.fold]
va = date.fromisoformat(fs["valid_anchor"])
ai = p.idx(va)
keep = p.active_in(ai - 29, ai)
X, names = build(p, ai, keep, cfg["feature_blocks"])
y = p.target(ai)[keep]
print(f"  anchor {va}: {int(keep.sum()):,} users x {X.shape[1]} features", flush=True)

have = set(names)
sel = [names_all[i] for i in ranked if names_all[i] in have][: args.n_features]
idx = [names.index(n) for n in sel]
print(f"  exporting {len(sel)} features", flush=True)

d = {"user_id": p.users[keep].astype(np.int64), "target": y.astype(np.float32)}
# a few raw reference columns so the notebook can sanity-check the target end to end
d["gmv_last30"] = p.wsum("gmv", ai - 29, ai)[keep].astype(np.float32)
d["gmv_last90"] = p.wsum("gmv", ai - 89, ai)[keep].astype(np.float32)
d["ord_last30"] = p.wsum("ord", ai - 29, ai)[keep].astype(np.float32)
for j, n in zip(idx, sel):
    d[n] = X[:, j].astype(np.float32)
out = ROOT / args.out
pl.DataFrame(d).write_parquet(out, compression="zstd")
print(f"  wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)", flush=True)

# the importance table, so the notebook can rank/filter without the cluster
rows = [{"feature": names_all[i], "gain": float(gain[i]), "null_score": float(score[i]),
         "beats_null": bool(score[i] > 0), "in_extract": names_all[i] in set(sel)}
        for i in np.argsort(-gain)]
pl.DataFrame(rows).write_csv(ROOT / "reports" / "local_importance.csv")
print(f"  wrote reports/local_importance.csv ({len(rows)} features)")

# fold geometry, so the CV-scheme plot needs no cluster access
meta = {"valid_anchor": str(va), "n_users": int(keep.sum()),
        "panel_start": str(p.dmin), "panel_end": str(p.dmax),
        "guard_zone_start": spec.get("guard_zone_start"),
        "horizon_days": spec.get("horizon_days"),
        "folds": [{"valid_anchor": f["valid_anchor"],
                   "train_anchors": f["train_anchors"]} for f in spec["folds"]]}
(ROOT / "reports" / "local_meta.json").write_text(json.dumps(meta, indent=2))
print("  wrote reports/local_meta.json")
