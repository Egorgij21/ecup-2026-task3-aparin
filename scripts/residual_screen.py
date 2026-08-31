"""Screen a candidate feature by its correlation with the MODEL RESIDUAL.

EXPERIMENTS.md 1q: the model is within 0.0010 RMSLE of perfectly predicting the
conditional mean of its own 200 prediction bins, so a new feature can only help by
separating users INSIDE a bin.  A feature the model already knows correlates with its
PREDICTION, not its ERROR -- e.g. every BTYD gap feature scores |corr(x,pred)| 0.35-0.90
and |corr(x,residual)| < 0.007.

Kill bar: incremental R^2 over [1, model_pred] must clear ~0.0002.  The year-lag was
killed at 0.000135; all 22 hand-designed candidates and 60 tsfresh statistics are below it.

Runs on the laptop in seconds against an OOF parquet -- use it BEFORE spending a CV run.

    python3 scripts/residual_screen.py --oof oof/e0049.parquet --fold 4
"""
import argparse
import numpy as np
import polars as pl
from numpy.linalg import lstsq


def r2(X, y):
    b, *_ = lstsq(X, y, rcond=None)
    return 1.0 - (y - X @ b).var() / y.var()


def screen(cands: dict, oof_path: str, fold: int = 4, bar: float = 2e-4):
    """cands: {name: (user_id array, value array)}.  Returns rows sorted worst-first."""
    o = pl.read_parquet(oof_path).filter(pl.col("fold_id") == fold)
    base_df = o.select(["user_id", "y_true", "y_pred"])
    out = []
    for name, (uid, val) in cands.items():
        j = base_df.join(pl.DataFrame({"user_id": uid, "x": val.astype(np.float64)}), on="user_id")
        M = np.log1p(np.maximum(j["y_pred"].to_numpy(), 0.0))
        L = np.log1p(j["y_true"].to_numpy())
        x = np.nan_to_num(j["x"].to_numpy())
        X0 = np.column_stack([np.ones(len(M)), M])
        inc = r2(np.column_stack([X0, x]), L) - r2(X0, L)
        out.append(dict(name=name, n=len(M),
                        corr_resid=float(np.corrcoef(x, L - M)[0, 1]),
                        corr_pred=float(np.corrcoef(x, M)[0, 1]),
                        incr_r2=float(inc), verdict="LOOK" if inc > bar else "kill"))
    return sorted(out, key=lambda r: -r["incr_r2"])


def report(rows):
    print(f"{'candidate':28}{'corr(resid)':>12}{'corr(pred)':>12}{'incr_R2':>10}  verdict")
    for r in rows:
        print(f"{r['name']:28}{r['corr_resid']:+12.4f}{r['corr_pred']:+12.4f}"
              f"{r['incr_r2']:10.5f}  {r['verdict']}")
    print("\nkill bar incr_R2 = 0.0002 (year-lag died at 0.000135; see EXPERIMENTS.md 1q)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default="oof/e0049.parquet")
    ap.add_argument("--fold", type=int, default=4)
    a = ap.parse_args()
    o = pl.read_parquet(a.oof).filter(pl.col("fold_id") == a.fold)
    uid = o["user_id"].to_numpy()
    rng = np.random.default_rng(0)
    report(screen({"__noise_control__": (uid, rng.standard_normal(len(uid)))}, a.oof, a.fold))
