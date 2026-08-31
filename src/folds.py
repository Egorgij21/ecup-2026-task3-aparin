#!/usr/bin/env python
"""
FROZEN fold definition (README.md). Writes data/folds.parquet + data/fold_spec.json.

Derived from DATA.md §4 and §9.4:

  * Population at any anchor A = users with >= 1 active day in [A-29, A]. This mirrors the
    test's inclusion rule and reproduces the test target's marginal (E[log1p y] 2.330 vs
    the LB-measured 2.320). Scoring all 250k at a past anchor is optimistic by ~0.10 and
    the stricter three-block rule is worse still (2.279) -- both tested and rejected.
  * A target window must not touch the guaranteed-activity zone [2025-11-16, 2026-02-13],
    so validation anchors satisfy A + 30 < 2025-11-16, i.e. A <= 2025-10-16.
  * Expanding-origin: fold k trains only on anchors A_train <= A_valid - 30, which makes
    the training targets end on or before the validation anchor -- no target overlap.
  * The two earliest clean anchors are training-only so that every validation fold has at
    least ~10 training anchors behind it. That leaves 5 validation folds.

Changing anything here invalidates every logged experiment (README.md).
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from data import Panel

ROOT = Path(__file__).resolve().parent.parent

HORIZON = 30
MIN_HISTORY_DAYS = 90                      # anchors need >=90d of lookback
GUARD_START = date(2025, 11, 16)           # start of the guaranteed-activity zone
VALID_ANCHORS = [date(2025, 6, 18), date(2025, 7, 18), date(2025, 8, 17),
                 date(2025, 9, 16), date(2025, 10, 16)]
TRAIN_STRIDE_DAYS = 7                      # grid for training anchors
TRAIN_GAP_DAYS = 30                        # A_train <= A_valid - 30


def training_anchors(valid_anchor: date, dmin: date) -> list[date]:
    """Every TRAIN_STRIDE_DAYS-th day in [dmin + 89, valid_anchor - TRAIN_GAP_DAYS]."""
    earliest = dmin + timedelta(days=MIN_HISTORY_DAYS - 1)
    latest = valid_anchor - timedelta(days=TRAIN_GAP_DAYS)
    out, a = [], latest
    while a >= earliest:
        out.append(a)
        a -= timedelta(days=TRAIN_STRIDE_DAYS)
    return sorted(out)


def main() -> None:
    p = Panel()
    for a in VALID_ANCHORS:
        assert a + timedelta(days=HORIZON) < GUARD_START, f"anchor {a} touches the guard zone"
        assert p.idx(a) - (MIN_HISTORY_DAYS - 1) >= 0, f"anchor {a} lacks history"

    rows, spec_folds = [], []
    print(f"\n  {'fold':5s} {'valid anchor':13s} {'target window':26s} {'n users':>9s} "
          f"{'zero%':>7s} {'n train anchors':>16s}")
    for k, a in enumerate(VALID_ANCHORS):
        ai = p.idx(a)
        keep = p.active_in(ai - 29, ai)
        y = p.target(ai, HORIZON)[keep]
        uid = p.users[keep]
        rows.append(pl.DataFrame({
            "fold_id": np.full(uid.size, k, np.int8),
            "anchor_date": pl.Series("anchor_date", [a] * uid.size, dtype=pl.Date),
            "user_id": uid,
            "target": y,
        }))
        ta = training_anchors(a, p.dmin)
        spec_folds.append({
            "fold_id": k, "valid_anchor": a.isoformat(),
            "target_window": [(a + timedelta(days=1)).isoformat(),
                              (a + timedelta(days=HORIZON)).isoformat()],
            "n_users": int(keep.sum()), "zero_share": float((y <= 0).mean()),
            "train_anchors": [x.isoformat() for x in ta],
        })
        print(f"  {k:<5d} {a.isoformat():13s} "
              f"{(a + timedelta(days=1)).isoformat() + '..' + (a + timedelta(days=HORIZON)).isoformat():26s} "
              f"{int(keep.sum()):>9,} {100 * (y <= 0).mean():>6.2f}% {len(ta):>16d}")

    folds = pl.concat(rows)
    out = ROOT / "data" / "folds.parquet"
    folds.write_parquet(out)

    spec = {
        "frozen_on": "2026-08-11",
        "horizon_days": HORIZON,
        "population_rule": "users with >=1 active day in [A-29, A]",
        "target": "sum(gmv) over [A+1, A+30]",
        "guard_zone_start": GUARD_START.isoformat(),
        "guard_zone_reason": "all 250k users are active in each 30d block of "
                             "[2025-11-16, 2026-02-13]; a target window overlapping it is "
                             "optimistically biased by ~0.041 RMSLE (DATA.md 4.3)",
        "valid_anchors": [a.isoformat() for a in VALID_ANCHORS],
        "train_stride_days": TRAIN_STRIDE_DAYS,
        "train_gap_days": TRAIN_GAP_DAYS,
        "min_history_days": MIN_HISTORY_DAYS,
        "expected_cv_lb_offset": "+0.11 (DATA.md 9.4): fold-period predictability is lower "
                                 "than the test's; compare deltas, never levels",
        "folds": spec_folds,
    }
    (ROOT / "data" / "fold_spec.json").write_text(json.dumps(spec, indent=2))
    print(f"\n  wrote {out.relative_to(ROOT)}  ({folds.height:,} rows)")
    print(f"  wrote data/fold_spec.json")

    # frozen-ness check: a re-run must reproduce byte-identical content
    import hashlib
    h = hashlib.sha256(out.read_bytes()).hexdigest()[:16]
    print(f"  folds.parquet sha256[:16] = {h}")


if __name__ == "__main__":
    main()
