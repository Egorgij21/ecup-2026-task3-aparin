#!/usr/bin/env python
"""
FINALS — TRUTH-ANCHORED PARTITION SIMULATION.

The public/private split here is a RANDOM SPLIT BY USER over the same 250k users and the same
30-day window: there is no distribution shift, no time gap, no regime change. So the only two
mechanisms that can reorder the finals on private are

  (i)  sampling noise: public measured 50k, private measures the complementary 200k;
  (ii) anything we fitted ON the public 50k (calibration constants, LB-fitted weights).

`finals_private_risk.py` handles (i) analytically. This script measures it DIRECTLY, on the
frozen-fold OOF where the truth is known, using the actual structural difference between the two
finals candidates: the usercv slot at GRU d128 (e0141 -> e0300_cal) vs GRU d48 (e0295 -> e0301).

Procedure, repeated N times on fold 4 (anchor 2025-10-16, the most test-like):
   * draw a 20/80 partition of the users;
   * affine-calibrate BOTH blends on the 20% part only (that is what we actually did: mu/sd came
     from public probes), then score both parts;
   * record the public delta and the private delta.
That gives P(private reverses | public sign) and sd(d_priv | d_pub) with no normal assumption
and no unknown eps -- the truth is in hand.
"""
import numpy as np, polars as pl

GBDT = ["e0266", "e0064"]                                  # e0300/e0301 gbdt slot
SEQ = ["e0100", "e0101", "e0101s1", "e0101s2", "e0101s3", "e0102", "e0108"]
W = (0.20, 0.38, 0.42)
NDRAW = 400
RNG = np.random.default_rng(7)


def main():
    base = pl.read_parquet("oof/e0049.parquet").sort(["fold_id", "user_id"])
    uf = pl.read_parquet("oof/usercv_full.parquet").sort(["fold_id", "user_id"])
    u48 = pl.read_parquet("oof/usercv_full_h48_e0272.parquet").sort(["fold_id", "user_id"])
    keys = (base.select(["fold_id", "user_id"])
            .join(uf.select(["fold_id", "user_id"]), on=["fold_id", "user_id"], how="inner")
            .sort(["fold_id", "user_id"]))

    def aligned(name=None, src=None):
        d = src if src is not None else pl.read_parquet(f"oof/{name}.parquet").sort(["fold_id", "user_id"])
        j = keys.join(d, on=["fold_id", "user_id"], how="left").sort(["fold_id", "user_id"])
        return np.log1p(np.maximum(j["y_pred"].to_numpy(), 0.0))

    L = np.log1p(keys.join(base, on=["fold_id", "user_id"], how="left")
                 .sort(["fold_id", "user_id"])["y_true"].to_numpy())
    fold = keys["fold_id"].to_numpy()

    gbdt = np.mean([aligned(e) for e in GBDT], axis=0)
    seq = np.mean([aligned(e) for e in SEQ], axis=0)
    d128 = aligned(src=uf)
    d48 = aligned(src=u48)

    A = W[0] * gbdt + W[1] * seq + W[2] * d128        # e0300_cal structure
    B = W[0] * gbdt + W[1] * seq + W[2] * d48         # e0301         structure

    for f in [4, None]:
        idx = np.where(fold == f)[0] if f is not None else np.arange(len(L))
        tag = f"fold {f}" if f is not None else "pooled 5 folds"
        La, Aa, Ba = L[idx], A[idx], B[idx]
        n = len(idx)

        def cal(x, m, sd):                                  # affine calibration in log space
            return (x - x.mean()) / x.std() * sd + m

        # full-sample calibrated scores, for reference
        Ac = cal(Aa, La.mean(), La.std() * np.corrcoef(Aa, La)[0, 1])
        Bc = cal(Ba, La.mean(), La.std() * np.corrcoef(Ba, La)[0, 1])
        rA = float(np.sqrt(np.mean((La - Ac) ** 2)))
        rB = float(np.sqrt(np.mean((La - Bc) ** 2)))
        d = Bc - Ac
        print(f"== {tag}: n={n:,} ==")
        print(f"   RMSLE  d128 slot {rA:.6f}   d48 slot {rB:.6f}   delta {rB-rA:+.6f}"
              f"   (negative = d48 better, matches the LB's -0.000133)")
        print(f"   corr(A,B) = {np.corrcoef(Ac,Bc)[0,1]:.6f}   rms(d) = {np.sqrt((d**2).mean()):.5f}")

        if f is None:
            continue

        # --- 20/80 partition simulation
        pub, priv = [], []
        for _ in range(NDRAW):
            perm = RNG.permutation(n)
            k = int(round(0.2 * n))
            ip, iq = perm[:k], perm[k:]
            # calibrate on the PUBLIC part only (mu, sd solved from public probes), score both
            mp, sp = La[ip].mean(), La[ip].std()
            res = {}
            for name, X in (("A", Aa), ("B", Ba)):
                rho = np.corrcoef(X[ip], La[ip])[0, 1]
                xc = (X - X[ip].mean()) / X[ip].std() * (sp * rho) + mp
                res[name] = (float(np.sqrt(np.mean((La[ip] - xc[ip]) ** 2))),
                             float(np.sqrt(np.mean((La[iq] - xc[iq]) ** 2))))
            pub.append(res["B"][0] - res["A"][0])
            priv.append(res["B"][1] - res["A"][1])
        pub, priv = np.array(pub), np.array(priv)
        print(f"\n   -- {NDRAW} random 20/80 partitions, calibration fitted on the 20% part --")
        print(f"      public  delta: mean {pub.mean():+.6f}  sd {pub.std():.6f}")
        print(f"      private delta: mean {priv.mean():+.6f}  sd {priv.std():.6f}")
        print(f"      corr(public, private delta) = {np.corrcoef(pub, priv)[0,1]:+.3f}"
              f"   (negative: they PARTITION the same users)")
        sign_agree = float(np.mean(np.sign(pub) == np.sign(priv)))
        print(f"      P(private sign == public sign) = {sign_agree:.3f}")
        print(f"      P(private favours d48)         = {float(np.mean(priv < 0)):.3f}")
        # conditional: given the public delta observed on the real LB (-0.000133)
        obs = -0.000133
        sel = np.argsort(np.abs(pub - obs))[:max(20, NDRAW // 10)]
        print(f"      given a public delta near {obs:+.6f}: private mean {priv[sel].mean():+.6f}"
              f"  sd {priv[sel].std():.6f}  P(reversal) {float(np.mean(priv[sel] > 0)):.3f}")
        print()


if __name__ == "__main__":
    main()
