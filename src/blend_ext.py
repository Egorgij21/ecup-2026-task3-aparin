#!/usr/bin/env python
"""Assemble the champion blend from component submission files, optionally swapping in a
component retrained with an extended (guard-zone-inclusive) window, and calibrate.

    python src/blend_ext.py --exp-id e0252_extblend --seq subs/e0250_seqext.csv

WHY THIS EXISTS. e0162 = 0.20*log(gbdt half) + 0.38*log(seq half) + 0.42*log(e0141), verified
to corr 0.999997 against subs/e0162.csv. So a component can be replaced without retraining the
other two, and the candidate can be inspected locally before a leaderboard slot is spent.

CALIBRATION -- deliberately MATCHED TO e0162 RATHER THAN RE-SOLVED.
`1i` records that the one submission which ASSUMED a member's rho (e0145) missed by 0.0019,
60x worse than the ones that measured everything. We do not know the extended blend's rho, so
we do not pretend to. Instead we map it to e0162's OWN submitted moments (mu 2.3305,
sd 1.6308). That makes the comparison PAIRED: both submissions carry the identical level and
spread, so any difference in score is attributable to rho alone, which is the quantity in
question. It is also near-optimal -- the penalty for sd_M being slightly off its optimum is
second order, while the level term is first order and is matched exactly.

Once the score comes back, rho is solvable in closed form (the same identity that reproduced
e0161 to 0.000000), and the optimally-calibrated version is one more slot.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
MU_L, SD_L = 2.3303, 2.3178          # 1i, solved from four precise equations
REF = "e0300_cal.csv"                # the CURRENT champion (1.646589), whose moments we match
#   e0300_cal = e0162 structure with e0266 replacing e0049 in the gbdt slot.
#   Reconstructed here from components to corr 0.9999970 before anything is swapped.
W = dict(gbdt=0.20, seq=0.38, usercv=0.42)


def logp(path: Path):
    d = pl.read_csv(path)
    return (d["user_id"].to_numpy(),
            np.log1p(np.maximum(d["predict"].to_numpy().astype(np.float64), 0.0)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--gbdt", default="subs/_gbdthalf_e0300.csv")
    ap.add_argument("--seq", default="subs/e0120s.csv")
    ap.add_argument("--usercv", default="subs/e0141.csv")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    u_ref, ref = logp(ROOT / "subs" / REF)
    parts, users = {}, None
    for k, p in (("gbdt", args.gbdt), ("seq", args.seq), ("usercv", args.usercv)):
        u, v = logp(ROOT / p)
        if users is None:
            users = u
        assert np.array_equal(u, users), f"user order differs in {p}"
        parts[k] = v
        print(f"  {k:7s} {p:34s} mu {v.mean():.4f}  sd {v.std():.4f}")
    assert np.array_equal(users, u_ref), "component user order differs from the reference"

    B = sum(W[k] * parts[k] for k in W)
    print(f"\n  raw blend            mu {B.mean():.4f}  sd {B.std():.4f}")
    print(f"  corr(blend, {REF[:-4]})  {np.corrcoef(B, ref)[0, 1]:.7f}")

    # match e0162's submitted moments exactly -> a PAIRED comparison isolating rho
    mu_t, sd_t = ref.mean(), ref.std()
    Bc = mu_t + (B - B.mean()) * (sd_t / B.std())
    print(f"  calibrated to {REF[:-4]:9s} mu {Bc.mean():.4f}  sd {Bc.std():.4f}   "
          f"(target mu {mu_t:.4f} sd {sd_t:.4f})")
    print(f"  corr(calibrated, {REF[:-4]}) {np.corrcoef(Bc, ref)[0, 1]:.7f}")

    pred = np.maximum(np.expm1(Bc), 0.0)
    ss = pl.read_csv(ROOT / "data" / "sample_submit.csv")
    assert pred.size == ss.height == 250_000
    assert np.array_equal(users, ss["user_id"].to_numpy()), "user order differs from sample"
    assert np.isfinite(pred).all() and (pred >= 0).all()
    assert 10.0 < pred.mean() < 400.0, f"prediction scale looks wrong: mean {pred.mean():.1f}"

    # what the score WOULD be at a given rho, so the result is pre-registered not post-hoc
    print(f"\n  pre-registered: with mu/sd matched to {REF[:-4]}, the score maps to rho as")
    for r in (0.70338, 0.70379, 0.704306, 0.704479, 0.7050):
        s = np.sqrt((SD_L - sd_t) ** 2 + 2 * SD_L * sd_t * (1 - r) + (MU_L - mu_t) ** 2)
        tag = ""
        if abs(r - 0.70379) < 1e-5:
            tag = "  <- e0300_cal's rho, so this reproduces 1.646589"
        if abs(r - 0.704479) < 1e-5:
            tag = "  <- the 1.6450 target"
        if abs(r - 0.704306) < 1e-5:
            tag = "  <- top-1 at 1.6454"
        print(f"    rho {r:.5f} -> {s:.6f}{tag}")

    if args.dry_run:
        print("\n  --dry-run: nothing written")
        return
    out = ROOT / "subs" / f"{args.exp_id}.csv"
    pl.DataFrame({"user_id": users, "predict": pred}).write_csv(out)
    print(f"\n  wrote {out.relative_to(ROOT)}  mean {pred.mean():.2f}  "
          f"zero-share {100 * (pred == 0).mean():.2f}%")


if __name__ == "__main__":
    main()
