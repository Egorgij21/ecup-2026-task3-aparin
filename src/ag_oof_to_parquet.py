#!/usr/bin/env python
"""
Assemble AutoGluon's per-fold OOF dumps into run.py's OOF schema so they can be blended.

    python src/ag_oof_to_parquet.py e0064

src/run_ag.py writes one `runs/ag/<exp>_f<k>_oof.npy` per fold -- linear-space predictions,
ordered by sorted user_id within the fold, i.e. exactly folds.parquet's per-fold order.
src/blend.py wants the columns run.py emits.  This bridges the two rather than teaching either
side about the other's format.

Nothing is recomputed here: the assembled file must reproduce the fold RMSLEs already logged
in runs/ag/<exp>_f<k>.json, and the script asserts that it does.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from metrics import rmsle       # noqa: E402


def main() -> None:
    exp = sys.argv[1] if len(sys.argv) > 1 else "e0064"
    folds = pl.read_parquet(ROOT / "data" / "folds.parquet")
    spec = json.loads((ROOT / "data" / "fold_spec.json").read_text())

    # geo3 comes from the seq panel's exact prefix sums -- the same reference run.py and
    # run_seq.py score against, so `delta` stays comparable across all three approaches.
    from seqdata import build_seq_panel
    sp = build_seq_panel(verbose=False)

    out, scores = [], []
    for k in sorted(folds["fold_id"].unique().to_list()):
        f = ROOT / "runs" / "ag" / f"{exp}_f{k}_oof.npy"
        if not f.exists():
            print(f"  fold {k}: MISSING {f.relative_to(ROOT)} -- skipped")
            continue
        pred = np.load(f).astype(np.float64)
        fv = folds.filter(pl.col("fold_id") == k).sort("user_id")
        yva = fv["target"].to_numpy()
        assert pred.size == yva.size, f"fold {k}: {pred.size} preds vs {yva.size} users"
        va = date.fromisoformat(spec["folds"][k]["valid_anchor"])
        naive = sp.geo3(sp.idx(va))[sp.pop[:, sp.idx(va)]]

        s = rmsle(yva, pred)
        logged = json.loads((ROOT / "runs" / "ag" / f"{exp}_f{k}.json").read_text())["rmsle"]
        assert abs(s - logged) < 1e-5, f"fold {k}: recomputed {s:.5f} vs logged {logged:.5f}"
        scores.append(s)
        out.append(pl.DataFrame({
            "fold_id": np.full(yva.size, k, np.int8),
            "anchor_date": pl.Series("anchor_date", [va] * yva.size, dtype=pl.Date),
            "user_id": fv["user_id"].to_numpy(),
            "y_true": yva, "y_pred": pred, "y_naive": naive}))
        print(f"  fold {k} {va}  n={yva.size:>7,}  rmsle={s:.5f}  (matches logged)")

    df = pl.concat(out)
    dst = ROOT / "oof" / f"{exp}.parquet"
    dst.parent.mkdir(exist_ok=True)
    df.write_parquet(dst)
    print(f"\n  wrote oof/{exp}.parquet  ({df.height:,} rows, {len(scores)} folds)")
    print(f"  mean over the folds present = {np.mean(scores):.5f}")


if __name__ == "__main__":
    main()
