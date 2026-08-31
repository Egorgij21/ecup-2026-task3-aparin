#!/usr/bin/env python
"""
FINAL-SUBMISSION ROBUSTNESS: which candidates are distinguishable on the private 200k?

The public LB is 50k users, private is 200k DISJOINT users, both from the test anchor. Our top
candidates' public-LB scores sit within 0.0002 of each other while the 50k sampling noise is
~0.012 (DATA.md 8.3). This bootstraps the frozen-fold OOF to measure the PAIRED-delta noise band
at 50k and 200k, so we can say which LB gaps (if any) are real and pick the finals on robustness
rather than public-LB luck.

Method: the score depends only on rho after calibration (RMSLE = sd_L*sqrt(1-rho^2)), and rho is
affine-invariant, so we bootstrap rho directly (calibration-free). We compare two champion-CLASS
blends built from members we hold; since all real candidates correlate >0.99, the paired-delta sd
between two champion-class predictors IS the noise band for comparing any two of them.
"""
import polars as pl, numpy as np

SEED = 0
SD_L = 2.3178                  # probe-solved test moment (EXPERIMENTS.md 1i)
GBDT = ["e0049", "e0064"]
SEQ  = ["e0100", "e0101", "e0101s1", "e0101s2", "e0101s3", "e0102", "e0108"]

def logpred(e):
    d = pl.read_parquet(f"oof/{e}.parquet").sort(["fold_id", "user_id"])
    return d

def main():
    rng = np.random.default_rng(SEED)
    # align everything on the (fold_id,user_id) intersection with usercv_full (the e0141 slot)
    base = logpred("e0049")
    uf = pl.read_parquet("oof/usercv_full.parquet").sort(["fold_id", "user_id"])
    keys = base.select(["fold_id", "user_id"]).join(
        uf.select(["fold_id", "user_id"]), on=["fold_id", "user_id"], how="inner")
    keys = keys.sort(["fold_id", "user_id"])
    def aligned(e, src=None):
        d = src if src is not None else logpred(e)
        j = keys.join(d, on=["fold_id", "user_id"], how="left").sort(["fold_id", "user_id"])
        return np.log1p(np.maximum(j["y_pred"].to_numpy(), 0.0))
    L = np.log1p(keys.join(base, on=["fold_id", "user_id"], how="left")
                 .sort(["fold_id", "user_id"])["y_true"].to_numpy())
    fold = keys["fold_id"].to_numpy()

    gbdt = np.mean([aligned(e) for e in GBDT], axis=0)
    seq  = np.mean([aligned(e) for e in SEQ], axis=0)
    ufl  = aligned(None, src=uf)
    # Champion-A: the 9-member equal log blend (e0120 structure, gbdt+seq only)
    A = np.mean([aligned(e) for e in GBDT + SEQ], axis=0)
    # Champion-B: e0162 structure (0.20 gbdt / 0.38 seq / 0.42 usercv_full)
    B = 0.20 * gbdt + 0.38 * seq + 0.42 * ufl

    def rho(y, m, idx): return float(np.corrcoef(y[idx], m[idx])[0, 1])
    def drmsle_drho(r): return SD_L * (-r / np.sqrt(1 - r * r))

    print("== full-population rho (fold 4, the most test-like anchor) ==")
    f4 = np.where(fold == 4)[0]
    rA, rB = rho(L, A, f4), rho(L, B, f4)
    print(f"  n(fold4)={len(f4):,}  Champion-A(e0120) rho={rA:.5f}  Champion-B(e0162) rho={rB:.5f}"
          f"  corr(A,B)={np.corrcoef(A[f4],B[f4])[0,1]:.5f}")
    print(f"  RMSLE(A)={SD_L*np.sqrt(1-rA**2):.5f}  RMSLE(B)={SD_L*np.sqrt(1-rB**2):.5f}")

    for n in (50_000, 200_000):
        dr = []
        pool = f4
        for _ in range(2000):
            idx = pool[rng.integers(0, len(pool), size=n)]   # bootstrap with replacement
            dr.append(rho(L, A, idx) - rho(L, B, idx))
        dr = np.array(dr)
        drmsle = dr * drmsle_drho((rA + rB) / 2)
        print(f"\n== bootstrap {n:,}-user paired delta (A - B), 2000 draws ==")
        print(f"  rho delta:   mean {dr.mean():+.6f}   sd {dr.std():.6f}")
        print(f"  RMSLE delta: mean {drmsle.mean():+.6f}   sd {drmsle.std():.6f}   "
              f"(2sigma band = {2*drmsle.std():.5f})")
        print(f"  P(A better than B) = {(dr>0).mean():.3f}")

    print("\n== the top candidates' public-LB gaps vs the 200k noise band ==")
    lb = {"e0301_usercv48_cal":1.646456,"e0300_cal":1.646589,"e0162":1.646602,
          "e0161":1.646670,"e0150":1.646700,"e0201_cal":1.646831}
    best = min(lb.values())
    for k,v in sorted(lb.items(), key=lambda x:x[1]):
        print(f"  {k:20s} LB={v:.6f}  gap vs best {v-best:+.6f}")

if __name__ == "__main__":
    main()
