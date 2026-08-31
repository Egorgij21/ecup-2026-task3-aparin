#!/usr/bin/env python
"""Merge runs/*.json into experiments.csv. Parallel-safe: each run owns one file."""
from __future__ import annotations
import csv, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COLS = ["exp_id", "parent_id", "date", "approach", "change", "tier", "n_features",
        "cv_mean", "cv_std", "folds", "delta", "significant", "lb", "runtime_min",
        "seed", "config", "verdict", "gini_pred", "total_rel_err", "best_iters", "notes"]

rows = {}
exp = ROOT / "experiments.csv"
if exp.exists():
    for r in csv.DictReader(exp.open()):
        rows[r["exp_id"]] = r
for p in sorted((ROOT / "runs").glob("*.json")):
    r = json.loads(p.read_text())
    prev = rows.get(r["exp_id"], {})
    for k in ("lb", "verdict"):                       # never clobber hand-entered fields
        if prev.get(k) and not r.get(k):
            r[k] = prev[k]
    rows[r["exp_id"]] = r
with exp.open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
    w.writeheader()
    for k in sorted(rows):
        w.writerow({c: rows[k].get(c, "") for c in COLS})
print(f"  experiments.csv: {len(rows)} rows")
