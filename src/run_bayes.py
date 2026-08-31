#!/usr/bin/env python
"""
BAYES_EXP.md B1 on the frozen folds -> CV row + oof/<exp>.parquet.

    python src/run_bayes.py --which full --draws 1000        # B1  -> e0180
    python src/run_bayes.py --which lam  --draws 1000        # ablation, rate latent only
    python src/run_bayes.py --which none --draws 1000        # CONTROL: must reproduce e0170

Same contract as src/run_btyd.py: fold populations and targets read from oof/e0049.parquet,
metric from src/metrics.py, OOF in run.py's schema so it is directly blendable.

Why there is no train/val user split here (BAYES_EXP §6 asks for one): nothing in this fit is
supervised.  The likelihood sees only the feature window, the covariates see only the feature
window, and the standardisation moments are computed on feature-window quantities.  The
target is referenced exactly once, at scoring.  A user split would protect against nothing
and cost population.  §6's split exists for the calibration layer, which the review measured
at -0.00001 and which this run does not use.
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

from bayes_cov import COVARIATE_COLS, build_covariates            # noqa: E402
from bayes_model import fit_bayes, fit_bayes_loc                  # noqa: E402
from btyd import BGNBD, GammaGamma, build_rfm, simulate_elog1p    # noqa: E402
from metrics import rmsle, score_all                              # noqa: E402

DMIN = date(2025, 1, 1)
HORIZON = 30.0
EXP = {"full": "e0180", "lam": "e0181", "none": "e0182", "loc": "e0183"}
COLS = ["user_id", "event_date", "gmv", "to_ord", "to_cart", "searches",
        "gmv_search", "search_to_ord"]


def load_events() -> pl.DataFrame:
    for p in (ROOT / "data" / "train.parquet", ROOT / "train.parquet"):
        if p.exists():
            df = pl.read_parquet(p, columns=COLS)
            break
    else:
        raise FileNotFoundError("train.parquet not found")
    di = ((df["event_date"].to_numpy().astype("datetime64[D]") - np.datetime64(DMIN))
          .astype(np.int32))
    return df.drop("event_date").with_columns(pl.Series("di", di))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["full", "lam", "none", "loc"], default="full")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--maxiter", type=int, default=2000)
    ap.add_argument("--cold", action="store_true", help="skip the B0 warm start")
    args = ap.parse_args()
    exp_id = EXP[args.which]

    t0 = time.time()
    events = load_events()
    folds = pd.read_parquet(ROOT / "oof" / "e0049.parquet").sort_values(["fold_id", "user_id"])
    print(f"  {exp_id} ({args.which})   events {events.height:,}   folds {len(folds):,}   "
          f"{time.time() - t0:.1f}s", flush=True)

    rows, diag = [], []
    for k, g in folds.groupby("fold_id"):
        anchor_date = g["anchor_date"].iloc[0]
        anchor = (anchor_date - DMIN).days
        users = g["user_id"].to_numpy()
        y_true, y_naive = g["y_true"].to_numpy(), g["y_naive"].to_numpy()
        tf = time.time()

        rfm = build_rfm(events.select(["user_id", "di", "gmv"]), users, anchor)
        assert np.array_equal(rfm.user_id, users), "fold population mismatch"
        X, names = build_covariates(events, users, anchor)
        assert X.shape == (users.size, len(COVARIATE_COLS))
        t_cov = time.time() - tf

        n_gg = rfm.n_buy.astype(np.float64)
        # BAYES_EXP §4.2 is right about this: initialise from B0.  Cold-started, the 119-
        # parameter fit wanders (r -> 144, p_gg -> 161) and is still moving at 150 iterations.
        init, b0 = None, None
        if args.which != "none" and not args.cold:
            b0 = fit_bayes(X, rfm.x, rfm.t_x, rfm.T, n_gg, rfm.m_x, which="none",
                           maxiter=args.maxiter)
            init = b0.params
        if args.which == "loc":
            assert b0 is not None, "--which loc needs the B0 warm start (do not pass --cold)"
            fit = fit_bayes_loc(X, rfm.x, rfm.t_x, rfm.T, n_gg, rfm.m_x, b0,
                                maxiter=args.maxiter)
        else:
            fit = fit_bayes(X, rfm.x, rfm.t_x, rfm.T, n_gg, rfm.m_x,
                            which=args.which, init=init, maxiter=args.maxiter)
        if init is not None:
            assert fit.nll <= b0.nll + 1e-6, (
                f"warm-started {args.which} fit ({fit.nll:,.1f}) is worse than the B0 it "
                f"started from ({b0.nll:,.1f}) -- the optimiser moved uphill")
        A = fit.arrays(X)
        if args.which == "loc":
            A["a"], A["b"] = fit.extra["a_u"], fit.extra["b_u"]
        t_fit = time.time() - tf - t_cov

        bg = BGNBD(A["r"], A["alpha"], A["a"], A["b"], fit.nll, users.size, fit.converged)
        gg = GammaGamma(A["p_gg"], A["q_gg"], A["nu"], 0.0, users.size, fit.converged)
        pa = bg.p_alive(rfm.x, rfm.t_x, rfm.T)
        sim = simulate_elog1p(bg, gg, rfm, n_gg, HORIZON, args.draws, args.seed + k)
        pred = np.expm1(sim["e_log1p"])
        assert np.all(np.isfinite(pred)) and np.all(pred >= 0)

        s = rmsle(y_true, pred)
        s_naive = rmsle(y_true, y_naive)
        rho = float(np.corrcoef(np.log1p(y_true), np.log1p(pred))[0, 1])
        sm = fit.summary(X)
        print(f"  fold {k} {anchor_date}  n {users.size:,}  RMSLE {s:.5f}  geo3 {s_naive:.5f}  "
              f"rho {rho:.4f}  [cov {t_cov:.0f}s fit {t_fit:.0f}s tot {time.time() - tf:.0f}s]",
              flush=True)
        print(f"          r={sm['r']:.4f}  mean alpha={sm['mean_alpha']:.3f}  "
              f"mean a+b={sm['mean_a_plus_b']:.3f}  p={sm['p_gg']:.4f} q={sm['q_gg']:.4f}  "
              f"mean nu={sm['mean_nu']:.2f}  nll={fit.nll:,.0f}  iters={fit.n_iter}"
              f"{'' if fit.converged else '  <- NOT CONVERGED'}", flush=True)

        rows.append(pl.DataFrame({
            "fold_id": np.full(users.size, k, np.int8),
            "anchor_date": pl.Series("anchor_date", [anchor_date] * users.size, dtype=pl.Date),
            "user_id": users, "y_true": y_true, "y_pred": pred, "y_naive": y_naive,
            "p_alive": pa, "sd_log1p": sim["sd_log1p"], "p_zero": sim["p_zero"],
            "e_n30": sim["e_n"],
        }))
        diag.append({"fold": int(k), "anchor": str(anchor_date), "n": int(users.size),
                     "rmsle": s, "rmsle_geo3": s_naive, "rho": rho,
                     "mean_p_alive": float(pa.mean()), "runtime_s": time.time() - tf,
                     **sm})

    oof = pl.concat(rows)
    oof.write_parquet(ROOT / "oof" / f"{exp_id}.parquet")
    v = np.array([d["rmsle"] for d in diag])
    n = np.array([d["rmsle_geo3"] for d in diag])
    print(f"\n  {exp_id} ({args.which}, {len(COVARIATE_COLS)} covariates)")
    print(f"  cv {v.mean():.5f} +/- {v.std(ddof=1):.5f}   folds {np.round(v, 5).tolist()}")
    print(f"  delta vs geo3 {v.mean() - n.mean():+.5f}   vs e0170 (1.83569) "
          f"{v.mean() - 1.83569:+.5f}")
    agg = score_all(oof["y_true"].to_numpy(), oof["y_pred"].to_numpy())
    print(f"  aggregate: " + "  ".join(f"{a}={b:.4f}" for a, b in agg.items()))

    summary = {"exp_id": exp_id, "which": args.which, "draws": args.draws,
               "n_covariates": len(COVARIATE_COLS), "covariates": names,
               "cv_mean": float(v.mean()), "cv_std": float(v.std(ddof=1)),
               "folds": [float(z) for z in v], "runtime_min": (time.time() - t0) / 60,
               "per_fold": diag, "aggregate": agg}
    (ROOT / "reports" / f"{exp_id}_bayes.json").write_text(json.dumps(summary, indent=2))
    print(f"  wrote oof/{exp_id}.parquet, reports/{exp_id}_bayes.json   "
          f"[{(time.time() - t0) / 60:.1f} min]")


if __name__ == "__main__":
    main()
