#!/usr/bin/env python
"""
Validate the analytic paired-delta estimator  sd(dRMSLE) = rms(d)/sqrt(n)  against a direct
bootstrap, on the frozen-fold OOF where the truth L IS known.

Why this is needed: my first cross-check used the e0120-vs-e0162 submission pair and looked
3x off. That pair carries a 0.148 LEVEL offset, and scripts/robustness.py bootstraps rho,
which is affine-invariant and therefore blind to level. This script compares like with like:
same vectors, same quantity (RMSLE), bootstrap vs formula, with and without the level term.
"""
import polars as pl, numpy as np

GBDT = ["e0049", "e0064"]
SEQ = ["e0100", "e0101", "e0101s1", "e0101s2", "e0101s3", "e0102", "e0108"]

def main():
    rng = np.random.default_rng(0)
    base = pl.read_parquet("oof/e0049.parquet").sort(["fold_id", "user_id"])
    uf = pl.read_parquet("oof/usercv_full.parquet").sort(["fold_id", "user_id"])
    keys = (base.select(["fold_id", "user_id"])
            .join(uf.select(["fold_id", "user_id"]), on=["fold_id", "user_id"], how="inner")
            .sort(["fold_id", "user_id"]))

    def aligned(e, src=None):
        d = src if src is not None else pl.read_parquet(f"oof/{e}.parquet").sort(["fold_id", "user_id"])
        j = keys.join(d, on=["fold_id", "user_id"], how="left").sort(["fold_id", "user_id"])
        return np.log1p(np.maximum(j["y_pred"].to_numpy(), 0.0))

    L = np.log1p(keys.join(base, on=["fold_id", "user_id"], how="left")
                 .sort(["fold_id", "user_id"])["y_true"].to_numpy())
    fold = keys["fold_id"].to_numpy()
    f4 = np.where(fold == 4)[0]

    gbdt = np.mean([aligned(e) for e in GBDT], axis=0)
    seq = np.mean([aligned(e) for e in SEQ], axis=0)
    ufl = aligned(None, src=uf)
    A = np.mean([aligned(e) for e in GBDT + SEQ], axis=0)          # e0120 structure
    B = 0.20 * gbdt + 0.38 * seq + 0.42 * ufl                       # e0162 structure

    L4, A4, B4 = L[f4], A[f4], B[f4]
    print(f"n(fold4) = {len(f4):,}\n")

    def rmsle(p, y): return float(np.sqrt(np.mean((y - p) ** 2)))

    for tag, (X, Y) in [("RAW (level offset present)", (A4, B4)),
                        ("LEVEL-MATCHED (both re-centred, as calibrated subs are)",
                         (A4 - A4.mean() + L4.mean(), B4 - B4.mean() + L4.mean()))]:
        d = Y - X
        rms = float(np.sqrt((d ** 2).mean()))
        analytic = rms / np.sqrt(50_000)
        boots = []
        for _ in range(2000):
            idx = rng.integers(0, len(f4), size=50_000)
            boots.append(rmsle(Y[idx], L4[idx]) - rmsle(X[idx], L4[idx]))
        boots = np.array(boots)
        print(f"-- {tag}")
        print(f"   mean|d| {np.abs(d).mean():.5f}   rms(d) {rms:.5f}   level offset {Y.mean()-X.mean():+.5f}")
        print(f"   bootstrap 50k paired sd = {boots.std():.6f}")
        print(f"   analytic  rms(d)/sqrt(n) = {analytic:.6f}   ratio {analytic/boots.std():.3f}\n")

if __name__ == "__main__":
    main()
