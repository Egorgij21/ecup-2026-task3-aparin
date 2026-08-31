#!/usr/bin/env python
"""
Is the public -> private jump (1.646483 -> 1.6616148454, i.e. +0.01513) a normal amount of
sampling shift, or a sign that something in the pipeline did not transfer?

The public 50k and the private 200k PARTITION the same 250k users, so the absolute score is
allowed to move purely because the two user subsets differ in how hard they are. That shift is
COMMON to every team (same users, and all top predictions correlate >0.999), so it cannot move
the ranking -- but its size is checkable: replay 20/80 partitions on the frozen-fold OOF, where
the truth is known, with a champion-structure blend calibrated exactly the way the submissions
were (moments taken from the 20% "public" part only).

Reports the distribution of RMSLE(80%) - RMSLE(20%) and where +0.01513 falls in it.
"""
import numpy as np, polars as pl

GBDT = ["e0266", "e0064"]
SEQ = ["e0100", "e0101", "e0101s1", "e0101s2", "e0101s3", "e0102", "e0108"]
W = (0.20, 0.38, 0.42)
OBSERVED = 1.6616148454 - 1.646483        # private(best of the two finals) - public(e0303)
NDRAW = 2000
RNG = np.random.default_rng(11)


def main():
    base = pl.read_parquet("oof/e0049.parquet").sort(["fold_id", "user_id"])
    uf = pl.read_parquet("oof/usercv_full.parquet").sort(["fold_id", "user_id"])
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
    P = (W[0] * np.mean([aligned(e) for e in GBDT], axis=0)
         + W[1] * np.mean([aligned(e) for e in SEQ], axis=0)
         + W[2] * aligned(src=uf))

    idx = np.where(fold == 4)[0]
    L4, P4 = L[idx], P[idx]
    n = len(idx)
    print(f"OOF fold 4 (anchor 2025-10-16): n = {n:,}   sd(log1p y) = {L4.std():.4f}\n")

    dif, r20, r80, sdrat = [], [], [], []
    for _ in range(NDRAW):
        perm = RNG.permutation(n)
        k = int(round(0.2 * n))
        ip, iq = perm[:k], perm[k:]
        rho = np.corrcoef(P4[ip], L4[ip])[0, 1]                 # calibrate on the 20% part only
        pc = (P4 - P4[ip].mean()) / P4[ip].std() * (L4[ip].std() * rho) + L4[ip].mean()
        a = float(np.sqrt(np.mean((L4[ip] - pc[ip]) ** 2)))
        b = float(np.sqrt(np.mean((L4[iq] - pc[iq]) ** 2)))
        dif.append(b - a); r20.append(a); r80.append(b)
        sdrat.append(L4[iq].std() / L4[ip].std())
    dif, r20, r80, sdrat = map(np.array, (dif, r20, r80, sdrat))

    print("== 2000 random 20/80 partitions, calibration fitted on the 20% part ==")
    print(f"   RMSLE on the 20% part : {r20.mean():.4f} +- {r20.std():.4f}")
    print(f"   RMSLE on the 80% part : {r80.mean():.4f} +- {r80.std():.4f}")
    print(f"   shift  (80%) - (20%)  : {dif.mean():+.5f} +- {dif.std():.5f}")
    print(f"   |shift| >= {abs(OBSERVED):.5f} in {float(np.mean(np.abs(dif) >= abs(OBSERVED))):.1%} of draws"
          f"   -> observed sits at {OBSERVED/dif.std():+.2f} sd")
    for q in (1, 5, 25, 50, 75, 95, 99):
        print(f"      p{q:<3d} {np.percentile(dif, q):+.5f}")
    print(f"\n   sd(log1p y) ratio 80%/20% : {sdrat.mean():.5f} +- {sdrat.std():.5f}")
    need = 1.6616148454 / 1.646483
    print(f"   the observed jump implies a target-sd ratio of {need:.5f} at constant rho"
          f"  -> {(need-1)/sdrat.std():+.2f} sd of that ratio")

    print("\n== how much of the jump could be OUR doing? ==")
    SD_L = 2.3178
    v = SD_L ** 2 * (1 / 50_000 + 1 / 200_000)
    print(f"   calibrating on public-solved moments costs {v/(2*1.6465):+.6f} on private"
          f"  = {100*v/(2*1.6465)/OBSERVED:.2f}% of the observed jump")
    print("   everything else is a property of WHICH USERS landed in each half, identical for")
    print("   every team, and therefore rank-neutral.")


if __name__ == "__main__":
    main()
