#!/usr/bin/env python
"""
BTYD on the frozen folds -> CV row + oof/e0170.parquet.   Spec: BTYD.md §4, §6, §7.

    python src/run_btyd.py [--draws 200] [--seed 0] [--gg-count buy|repeat]

Mirrors src/run.py's contract: same 5 anchors, same populations, same metric, same OOF
schema, so the output is directly blendable with oof/e0049.parquet and oof/e0101.parquet.

The fold populations and targets are taken from oof/e0049.parquet rather than rebuilt.
That file was written by run.py from data/folds.parquet and reproduces both frozen
reference numbers exactly (geo3 1.92862, e0049 1.76551), so it IS the frozen fold
definition -- and reading it makes population drift impossible rather than merely checked.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from btyd import (assert_no_lookahead, build_rfm, fit_bgnbd, fit_gg,  # noqa: E402
                  simulate_elog1p)
from metrics import rmsle, score_all                                  # noqa: E402

DMIN = date(2025, 1, 1)
HORIZON = 30.0
EXP_ID = "e0170"


def load_events() -> pl.DataFrame:
    for p in (ROOT / "data" / "train.parquet", ROOT / "train.parquet"):
        if p.exists():
            df = pl.read_parquet(p, columns=["user_id", "event_date", "gmv"])
            break
    else:
        raise FileNotFoundError("train.parquet not found")
    di = ((df["event_date"].to_numpy().astype("datetime64[D]") - np.datetime64(DMIN))
          .astype(np.int32))
    return df.select(["user_id", "gmv"]).with_columns(pl.Series("di", di))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gg-count", choices=["buy", "repeat"], default="buy",
                    help="observations behind m_x: all buy-days (x+1, consistent) or the "
                         "literal BTYD.md §3.2 repeat count x")
    ap.add_argument("--skip-guard", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    events = load_events()
    folds = pd.read_parquet(ROOT / "oof" / "e0049.parquet").sort_values(["fold_id", "user_id"])
    print(f"  events {events.height:,} rows   folds {len(folds):,} rows   "
          f"{time.time() - t0:.1f}s", flush=True)

    rows, diag = [], []
    for k, g in folds.groupby("fold_id"):
        anchor_date = g["anchor_date"].iloc[0]
        anchor = (anchor_date - DMIN).days
        users = g["user_id"].to_numpy()
        y_true = g["y_true"].to_numpy()
        y_naive = g["y_naive"].to_numpy()
        tf = time.time()

        if k == 0 and not args.skip_guard:
            assert_no_lookahead(events, users, anchor)
            print("  look-ahead guard: PASSED (fold 0)", flush=True)

        rfm = build_rfm(events, users, anchor)
        assert np.array_equal(rfm.user_id, users), "fold population mismatch"

        buyer = rfm.buyer
        bg = fit_bgnbd(rfm.x[buyer], rfm.t_x[buyer], rfm.T[buyer])
        n_gg = (rfm.n_buy.astype(np.float64) if args.gg_count == "buy" else rfm.x)
        gg = fit_gg(n_gg, rfm.m_x)

        assert min(bg.r, bg.alpha, bg.a, bg.b) > 0 and min(gg.p, gg.q, gg.nu) > 0
        pa = bg.p_alive(rfm.x, rfm.t_x, rfm.T)

        # Gamma-Gamma's own assumption: frequency independent of monetary value
        m = buyer & (rfm.m_x > 0)
        rho_fm = float(np.corrcoef(rfm.x[m], rfm.m_x[m])[0, 1])
        rho_fm_log = float(np.corrcoef(np.log1p(rfm.x[m]), np.log(rfm.m_x[m]))[0, 1])

        sim = simulate_elog1p(bg, gg, rfm, n_gg, HORIZON, args.draws, args.seed + k)
        pred_elog = np.expm1(sim["e_log1p"])                       # E[log1p y], the RMSLE optimum
        e_m = gg.expected_m(n_gg, rfm.m_x)
        if bg.a > 1.0:
            e_x = bg.expected_x(rfm.x, rfm.t_x, rfm.T, HORIZON)
        else:
            e_x = sim["e_n"]                                       # closed form diverges; MC is exact
        pred_ey = e_x * e_m                                        # E[y], the naive BTYD output
        assert np.all(np.isfinite(pred_elog)) and np.all(pred_elog >= 0)
        assert np.all(np.isfinite(pred_ey)) and np.all(pred_ey >= 0)

        s_elog, s_ey = rmsle(y_true, pred_elog), rmsle(y_true, pred_ey)
        s_naive = rmsle(y_true, y_naive)
        rho = float(np.corrcoef(np.log1p(y_true), np.log1p(pred_elog))[0, 1])
        print(f"  fold {k} {anchor_date}  n {users.size:,}  "
              f"never-buyers {100 * (~buyer).mean():5.2f}%  "
              f"E[log1p] {s_elog:.5f}  log1p(E) {s_ey:.5f}  geo3 {s_naive:.5f}  "
              f"rho {rho:.4f}  [{time.time() - tf:.0f}s]", flush=True)
        print(f"          BG/NBD r={bg.r:.4f} alpha={bg.alpha:.4f} a={bg.a:.4f} b={bg.b:.4f}"
              f"  mean P(alive) {pa.mean():.4f}  P(alive)>0.99 {100 * (pa > 0.99).mean():.1f}%"
              f"  | GG p={gg.p:.4f} q={gg.q:.4f} nu={gg.nu:.3f}"
              f"  corr(x,m_x) {rho_fm:+.4f} (log {rho_fm_log:+.4f})", flush=True)

        rows.append(pl.DataFrame({
            "fold_id": np.full(users.size, k, np.int8),
            "anchor_date": pl.Series("anchor_date", [anchor_date] * users.size, dtype=pl.Date),
            "user_id": users, "y_true": y_true,
            "y_pred": pred_elog, "y_naive": y_naive,
            "y_pred_ey": pred_ey, "p_alive": pa, "e_x30": e_x, "e_m": e_m,
            "sd_log1p": sim["sd_log1p"], "p_zero": sim["p_zero"],
        }))
        diag.append({"fold": int(k), "anchor": str(anchor_date), "n": int(users.size),
                     "never_buyer_share": float((~buyer).mean()),
                     "rmsle_elog1p": s_elog, "rmsle_log1pey": s_ey, "rmsle_geo3": s_naive,
                     "rho": rho, "mean_p_alive": float(pa.mean()),
                     "share_p_alive_gt_099": float((pa > 0.99).mean()),
                     "corr_x_mx": rho_fm, "corr_x_mx_log": rho_fm_log,
                     "bgnbd": bg.as_dict(), "gg": gg.as_dict()})

    oof = pl.concat(rows)
    (ROOT / "oof").mkdir(exist_ok=True)
    oof.write_parquet(ROOT / "oof" / f"{EXP_ID}.parquet")
    # a second file so src/blend.py can use the E[y] variant unchanged
    oof.drop("y_pred").rename({"y_pred_ey": "y_pred"}).write_parquet(
        ROOT / "oof" / f"{EXP_ID}ey.parquet")

    e_log = np.array([d["rmsle_elog1p"] for d in diag])
    e_ey = np.array([d["rmsle_log1pey"] for d in diag])
    e_n = np.array([d["rmsle_geo3"] for d in diag])
    print(f"\n  {'variant':22s} {'cv_mean':>9s} {'cv_std':>8s}   folds")
    for nm, v in (("E[log1p y]  (primary)", e_log), ("log1p(E[y])", e_ey), ("geo3 reference", e_n)):
        print(f"  {nm:22s} {v.mean():>9.5f} {v.std(ddof=1):>8.5f}   {np.round(v, 5).tolist()}")
    print(f"\n  delta vs geo3: E[log1p y] {e_log.mean() - e_n.mean():+.5f}   "
          f"log1p(E[y]) {e_ey.mean() - e_n.mean():+.5f}")
    print(f"  functional gap (log1p(E[y]) - E[log1p y]) = {e_ey.mean() - e_log.mean():+.5f}")

    agg = score_all(oof["y_true"].to_numpy(), oof["y_pred"].to_numpy())
    print(f"  aggregate: " + "  ".join(f"{k}={v:.4f}" for k, v in agg.items()))

    summary = {"exp_id": EXP_ID, "draws": args.draws, "seed": args.seed,
               "gg_count": args.gg_count,
               "cv_mean_elog1p": float(e_log.mean()), "cv_std_elog1p": float(e_log.std(ddof=1)),
               "cv_mean_log1pey": float(e_ey.mean()), "cv_std_log1pey": float(e_ey.std(ddof=1)),
               "cv_mean_geo3": float(e_n.mean()),
               "runtime_min": (time.time() - t0) / 60.0, "folds": diag, "aggregate": agg}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / f"{EXP_ID}_btyd.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  wrote oof/{EXP_ID}.parquet, oof/{EXP_ID}ey.parquet, "
          f"reports/{EXP_ID}_btyd.json   [{(time.time() - t0) / 60:.1f} min]")


if __name__ == "__main__":
    main()
