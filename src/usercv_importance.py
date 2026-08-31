#!/usr/bin/env python
"""
Which of the behavioural features actually carry signal, and where do they rank?

    python src/usercv_importance.py --variant behav

The `behav` block adds 42 interpretable features over `full`: cart-minus-order as a level,
the cart backlog stock, the FIFO cart-to-order delay, and the per-window conversion rate,
AOV, basket size, GMV per buying day, cart-without-order days and the two search/cart ratios.

A GRU gives no feature attribution, so importance is read off a LightGBM fitted on the SAME
matrix, sliced at the frozen anchors.  Three views, because each is wrong on its own:

  * GAIN            -- biased toward high-cardinality continuous columns (EXPERIMENTS.md §5.6:
                       it cannot see redundancy, and `sbc_dutycycle_ord_180` topped it at 2.4x
                       the next feature while its whole family was worth -0.00008).
  * PERMUTATION     -- measures what the model would LOSE without the column, so twins that
                       carry the same information both score ~0.  That is the point.
  * UNIVARIATE      -- Spearman against the target, ignoring the model entirely, so a feature
                       with real signal that the model already gets elsewhere still shows up.

Read them together: high univariate + low permutation = real signal, already covered.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from usercv_features import (HORIZON, Raw, build_features, build_target,   # noqa: E402
                             build_tmask, feature_names, max_anchor)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    def rank(v):
        o = np.argsort(v, kind="stable"); r = np.empty(v.size)
        r[o] = np.arange(v.size)
        return r
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="behav")
    ap.add_argument("--anchors", type=int, default=12, help="anchors sampled for the matrix")
    ap.add_argument("--rounds", type=int, default=400)
    args = ap.parse_args()
    t0 = time.time()
    import lightgbm as lgb

    raw = Raw()
    last = max_anchor(raw)
    Y = build_target(raw, "sum")
    M = build_tmask(raw, last, burn_in=14, trim_to_first_seen=True)
    X, names = build_features(raw, args.variant)
    base = set(feature_names("full"))
    new = [n for n in names if n not in base]
    print(f"  {len(names)} features, {len(new)} new in `{args.variant}` vs `full`")

    # sample anchors on the frozen 7-day grid so the matrix matches the CV protocol's shape
    grid = list(range(89, last + 1, 7))
    anchors = grid[-args.anchors:]
    rows_x, rows_y = [], []
    for t in anchors:
        m = M[:, t]
        rows_x.append(X[m, t, :].astype(np.float32)); rows_y.append(Y[m, t])
    Xt = np.concatenate(rows_x); yt = np.concatenate(rows_y)
    print(f"  matrix {Xt.shape[0]:,} x {Xt.shape[1]}  from anchors "
          f"{raw.day(anchors[0])} .. {raw.day(anchors[-1])}")

    # hold out the last anchor so permutation importance is measured out of sample
    ntr = sum(r.shape[0] for r in rows_x[:-1])
    m = lgb.train({"objective": "regression", "metric": "rmse", "learning_rate": 0.05,
                   "num_leaves": 63, "min_data_in_leaf": 200, "feature_fraction": 0.8,
                   "verbose": -1, "seed": 0},
                  lgb.Dataset(Xt[:ntr], yt[:ntr], feature_name=names),
                  num_boost_round=args.rounds)
    Xv, yv = Xt[ntr:], yt[ntr:]
    base_rmse = float(np.sqrt(np.mean((m.predict(Xv) - yv) ** 2)))
    print(f"  holdout RMSE {base_rmse:.5f}  (anchor {raw.day(anchors[-1])})")

    gain = dict(zip(names, m.feature_importance("gain")))
    gsum = sum(gain.values())
    rng = np.random.default_rng(0)
    sub = rng.choice(Xv.shape[0], min(60_000, Xv.shape[0]), replace=False)
    Xs, ys = Xv[sub], yv[sub]
    b = float(np.sqrt(np.mean((m.predict(Xs) - ys) ** 2)))
    perm, uni = {}, {}
    order = sorted(names, key=lambda n: -gain[n])
    probe = set(new) | set(order[:25])
    for n in probe:
        j = names.index(n)
        keep = Xs[:, j].copy()
        Xs[:, j] = rng.permutation(keep)
        perm[n] = float(np.sqrt(np.mean((m.predict(Xs) - ys) ** 2))) - b
        Xs[:, j] = keep
        uni[n] = spearman(Xt[ntr:][sub][:, j], ys)

    print(f"\n  === TOP 20 BY GAIN (of {len(names)}) ===")
    print(f"  {'#':>3s} {'feature':26s} {'gain %':>8s} {'perm dRMSE':>11s} {'spearman':>9s}  new?")
    for i, n in enumerate(order[:20], 1):
        print(f"  {i:>3d} {n:26s} {100 * gain[n] / gsum:>8.2f} "
              f"{perm.get(n, float('nan')):>11.5f} {uni.get(n, float('nan')):>9.4f}"
              f"  {'NEW' if n not in base else ''}")

    print(f"\n  === THE {len(new)} NEW BEHAVIOURAL FEATURES, by permutation importance ===")
    print(f"  {'feature':26s} {'gain %':>8s} {'gain rank':>10s} {'perm dRMSE':>11s} {'spearman':>9s}")
    for n in sorted(new, key=lambda z: -perm[z]):
        print(f"  {n:26s} {100 * gain[n] / gsum:>8.2f} {order.index(n) + 1:>10d} "
              f"{perm[n]:>11.5f} {uni[n]:>9.4f}")

    tot_new = 100 * sum(gain[n] for n in new) / gsum
    print(f"\n  the {len(new)} new features hold {tot_new:.1f}% of total gain "
          f"({len(new) / len(names):.0%} of the columns)")
    print(f"  best new feature by permutation: "
          f"{max(new, key=lambda z: perm[z])} ({max(perm[z] for z in new):+.5f} RMSE)")
    print(f"  for scale: the whole `sbc` family (144 features) was worth -0.00008 CV (e0049)")

    out = ROOT / "reports" / "eda" / f"usercv_importance_{args.variant}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"variant": args.variant, "n_features": len(names),
                               "new": new, "gain": {k: float(v) for k, v in gain.items()},
                               "perm": perm, "spearman": uni,
                               "holdout_rmse": base_rmse}, indent=2))
    print(f"\n  wrote {out.relative_to(ROOT)}   runtime {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
