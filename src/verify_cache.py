#!/usr/bin/env python
"""
Prove the feature cache returns EXACTLY what building from scratch returns.

Every way this cache can fail returns wrong numbers that look like results -- there is no
crash to notice. So it does not get switched on until a job has shown, on the real panel and
the real feature blocks, that cold and warm builds are bit-identical.

Six checks:
  1. cold (cache off)  vs  cold (cache on, miss)   -- writing must not perturb the result
  2. cold (miss)       vs  warm (hit)              -- the round-trip through .npz is exact
  3. names identical and in the same ORDER         -- a silent permutation would remap every
                                                      feature importance we have logged
  4. a different `keep` mask must MISS, not serve the previous mask's rows
  5. `max_window` filtering agrees cold vs warm    -- it is applied after assembly and is
                                                      deliberately absent from the cache key
  6. the guard still guards: assert_no_lookahead must report a cache-disabled rebuild

Exact equality is the bar, not np.isclose. float32 through npz is lossless; anything less
than identity means something is wrong and "close enough" would hide it.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import features as F                          # noqa: E402
from data import Panel                        # noqa: E402


def hdr(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}", flush=True)


def same(a: np.ndarray, b: np.ndarray) -> bool:
    """Bit-identical, treating NaN as equal to NaN (features legitimately carry NaN)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return bool(np.array_equal(a, b, equal_nan=True))


import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/e0049_nomoment.yaml")
ap.add_argument("--force", action="store_true", help="enable the cache despite the quota")
args = ap.parse_args()

cfg = yaml.safe_load((ROOT / args.config).read_text())
blocks = cfg["feature_blocks"]
spec = json.loads((ROOT / "data" / "fold_spec.json").read_text())
p = Panel()

anchor = p.idx(date.fromisoformat(spec["folds"][0]["valid_anchor"]))
keep = p.active_in(anchor - 29, anchor)
print(f"  config {args.config}")
print(f"  blocks {blocks}")
print(f"  anchor index {anchor}, {int(keep.sum()):,} rows")

fails = []

hdr("0 -- COLD BUILD WITH THE CACHE OFF (the reference)")
F.enable_cache(False)
t = time.time()
X0, n0 = F.build(p, anchor, keep, blocks)
t_off = time.time() - t
print(f"  {X0.shape[0]:,} x {X0.shape[1]}  dtype={X0.dtype}  {t_off:.1f}s")
print(f"  in-memory size {X0.nbytes / 1e9:.2f} G per anchor")

hdr("1 -- ENABLE THE CACHE")
F.enable_cache(True, force=args.force)
if not F.CACHE_ENABLED:
    print("\n  cache REFUSED to enable (quota). Re-run with --force once there is room.")
    sys.exit(1)
gen = F.CACHE_DIR / F._code_hash()
print(f"  generation {F._code_hash()}  budget {F.CACHE_BUDGET_GB:.0f}G  dir {gen}")
# start from a clean slate for THIS anchor so check 2 is a genuine cold->warm transition
kh = F._keep_hash(keep)
for b in blocks:
    F._cache_path(b, anchor, getattr(p, "floor", 0), kh).unlink(missing_ok=True)

hdr("2 -- COLD BUILD WITH THE CACHE ON (miss + write)")
t = time.time()
X1, n1 = F.build(p, anchor, keep, blocks)
t_cold = time.time() - t
s = F.cache_stats()
print(f"  {t_cold:.1f}s   hits={s['hit']} misses={s['miss']} written={s['written']} "
      f"({s['bytes'] / 1e9:.2f}G)")
if s["written"] != len(blocks):
    fails.append(f"expected {len(blocks)} cache files written, got {s['written']}")
if not same(X0, X1):
    fails.append("cold-with-cache differs from cold-without-cache")
if n0 != n1:
    fails.append("names differ between cold builds")

hdr("3 -- WARM BUILD (must be all hits, and identical)")
t = time.time()
X2, n2 = F.build(p, anchor, keep, blocks)
t_warm = time.time() - t
s2 = F.cache_stats()
print(f"  {t_warm:.1f}s   hits={s2['hit'] - s['hit']} misses={s2['miss'] - s['miss']}")
if s2["miss"] != s["miss"]:
    fails.append("warm build missed the cache")
if not same(X0, X2):
    d = np.argwhere(~np.isclose(np.nan_to_num(X0), np.nan_to_num(X2)))
    fails.append(f"WARM BUILD DIFFERS from the reference at {len(d)} cells")
if n0 != n2:
    fails.append("names differ or are reordered between cold and warm")
else:
    print(f"  names identical and in order ({len(n2)} features)")
print(f"  speed-up {t_off / max(t_warm, 1e-9):.1f}x  ({t_off:.1f}s -> {t_warm:.1f}s)")

hdr("4 -- A DIFFERENT `keep` MASK MUST NOT HIT")
keep_b = keep.copy()
on = np.flatnonzero(keep_b)
keep_b[on[: max(len(on) // 2, 1)]] = False        # same anchor, different population
before = F.cache_stats()["miss"]
Xb, nb = F.build(p, anchor, keep_b, blocks)
after = F.cache_stats()["miss"]
print(f"  rows {int(keep.sum()):,} -> {int(keep_b.sum()):,}   misses {after - before} "
      f"(want {len(blocks)})")
if after - before != len(blocks):
    fails.append("a changed keep mask served cached rows instead of rebuilding")
if Xb.shape[0] != int(keep_b.sum()):
    fails.append("row count wrong for the second mask")
# and the original must still be intact afterwards
X3, _ = F.build(p, anchor, keep, blocks)
if not same(X0, X3):
    fails.append("the original entry was corrupted by the second mask")
else:
    print("  original entry still intact")

hdr("5 -- max_window AGREES COLD vs WARM (it is not part of the key, by design)")
MW = 90.0
F.enable_cache(False)
Xm0, nm0 = F.build(p, anchor, keep, blocks, MW)
F.enable_cache(True, force=args.force)
Xm1, nm1 = F.build(p, anchor, keep, blocks, MW)
print(f"  max_window={MW:.0f} -> {Xm1.shape[1]} of {X0.shape[1]} features")
if not same(Xm0, Xm1) or nm0 != nm1:
    fails.append("max_window filtering differs between cold and warm builds")
else:
    print("  identical -- one cached copy serves every lookback")

hdr("6 -- THE LOOK-AHEAD GUARD STILL DISABLES THE CACHE")
sys.path.insert(0, str(ROOT / "src"))
from run import assert_no_lookahead              # noqa: E402
F.enable_cache(True, force=args.force)
before = F.cache_stats()["build"]                # "build", not "miss": the guard turns the
assert_no_lookahead(p, anchor, X0, keep, blocks)  # cache OFF, so a served entry would show
after = F.cache_stats()["build"]                  # as neither a hit nor a miss
print(f"  guard forced {after - before} rebuilds (want {len(blocks)}); "
      f"cache re-enabled afterwards: {F.CACHE_ENABLED}")
if after - before != len(blocks):
    fails.append("assert_no_lookahead was served from the cache -- the guard is not guarding")
if not F.CACHE_ENABLED:
    fails.append("guard left the cache disabled")

hdr("VERDICT")
if fails:
    for f in fails:
        print(f"  FAIL: {f}")
    sys.exit(1)
print("  all six checks passed -- the cache is safe to enable")
print(f"  projected: {t_off:.0f}s -> {t_warm:.0f}s per anchor, "
      f"{X0.nbytes / 1e9 * 81:.0f}G for all 81 anchors at this feature set")
