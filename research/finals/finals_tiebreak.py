#!/usr/bin/env python
"""
FINALS — the jury's LTV tie-breakers, applied to the candidate submissions.

TASK.md: among the top-15 private teams the jury weighs, alongside private rank,
"LTV-specific tie-breaker metrics: the Gini coefficient over customer predictions and a
comparison of total predicted GMV across all customers (by RMSPE)".

Neither has ever been used to choose between finals. Both are computable on the submission
files themselves; the only unknown is the true test total GMV, which is anchored here two
independent ways:
  (a) from the frozen-fold OOF truth (fold 4 = the most test-like anchor), per-user mean
      scaled to 250k users;
  (b) from the probe-solved test log-moments, via the OOF's own log->level relationship
      (the empirical E[y] / exp(E[log1p y]) ratio, which is distribution-shape driven and
      far more stable than a lognormal assumption).
"""
import numpy as np, pandas as pd, polars as pl

CANDS = ["e0301_usercv48_cal", "e0303_arch4_cal", "e0300_cal", "e0162", "e0161",
         "e0150", "e0152", "e0201_cal", "e0200_cal", "e0270_cal",
         "e0266_cal", "e0141", "e0120", "e0049"]
MU_L, SD_L = 2.3303, 2.3178

def gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x)
    if x.sum() <= 0: return 0.0
    idx = np.arange(1, n + 1)
    return float((2.0 * (idx * x).sum()) / (n * x.sum()) - (n + 1.0) / n)

def main():
    # --- anchor the true total from OOF fold 4
    d = pl.read_parquet("oof/e0049.parquet").filter(pl.col("fold_id") == 4)
    y = d["y_true"].to_numpy().astype(np.float64)
    ly = np.log1p(y)
    print("== truth anchor, frozen-fold OOF fold 4 (anchor 2025-10-16, the most test-like) ==")
    print(f"   n={len(y):,}  mean(y)={y.mean():.3f}  E[log1p y]={ly.mean():.4f}  sd={ly.std():.4f}")
    print(f"   Gini(truth)={gini(y):.5f}")
    ratio = y.mean() / np.expm1(ly.mean())          # level-vs-log shape factor
    print(f"   shape factor  E[y] / expm1(E[log1p y]) = {ratio:.4f}")
    # (b) transport that shape factor to the test moments
    est_mean_test = np.expm1(MU_L) * ratio
    tot_b = est_mean_test * 250_000
    tot_a = y.mean() * 250_000
    print(f"\n== estimated TRUE total test GMV over 250k users ==")
    print(f"   (a) OOF fold-4 level, scaled:            {tot_a:,.0f}")
    print(f"   (b) probe-solved test mu + shape factor: {tot_b:,.0f}   "
          f"(test mu {MU_L:.4f} vs fold4 {ly.mean():.4f} -> level ratio {np.expm1(MU_L)/np.expm1(ly.mean()):.3f})")
    print("   -> the test anchor sits ABOVE fold 4 in level, so (b) is the better anchor;")
    print("      (a) is kept as a floor.\n")

    rows = []
    for c in CANDS:
        p = pd.read_csv(f"subs/{c}.csv")
        v = np.maximum(p.iloc[:, 1].astype(np.float64).to_numpy(), 0.0)
        tot = v.sum()
        rows.append(dict(sub=c, total=tot, mean=v.mean(), gini=gini(v),
                         rel_a=tot / tot_a - 1.0, rel_b=tot / tot_b - 1.0,
                         zeros=float((v <= 1e-9).mean()), p99=np.quantile(v, 0.99),
                         mx=v.max()))
    R = pd.DataFrame(rows)
    print("== tie-breaker table (Gini over predictions; total predicted GMV vs truth anchors) ==")
    print(R.to_string(index=False, formatters={
        "total": "{:,.0f}".format, "mean": "{:.2f}".format, "gini": "{:.5f}".format,
        "rel_a": "{:+.1%}".format, "rel_b": "{:+.1%}".format, "zeros": "{:.3%}".format,
        "p99": "{:,.0f}".format, "mx": "{:,.0f}".format}))
    print(f"\n   Gini(truth, fold 4) = {gini(y):.5f}  <- the number predictions are judged against")

if __name__ == "__main__":
    main()
