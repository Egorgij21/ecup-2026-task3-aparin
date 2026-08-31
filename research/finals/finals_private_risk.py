#!/usr/bin/env python
"""
FINALS — PRIVATE-LB RISK, PAIR BY PAIR, AND THE VALUE OF THE SECOND PICK.

The generic "200k noise band" in scripts/robustness.py is one number for all pairs. That is too
blunt here: the candidates differ by wildly different amounts (r spans 0.99945 to 0.99987), so
each pair has its OWN band, and it is computable EXACTLY from the submitted files without the
test truth.

ESTIMATOR.  For two log-space predictions X, Y with residual eps = X - L, d = Y - X:
    RMSLE_Y^2 - RMSLE_X^2 = mean_i[ d_i (X_i + Y_i - 2 L_i) ] ~ mean_i[ 2 d_i eps_i ]
so over a random n-user sample sd(dRMSLE) = rms(d)/sqrt(n)  -- a paired t-test, and rms(d) is
known exactly from the two csv files.
VALIDATED in scripts/finals_estimator_check.py against a direct 2000-draw bootstrap on the
frozen-fold OOF (where L is known): analytic/bootstrap = 0.90 (raw) and 0.90 (level-matched),
i.e. the formula runs ~10% LOW. CAL = 1.10 corrects it, conservatively.
(My first cross-check used the e0120-vs-e0162 submission pair and looked 3x off; that pair
carries a 0.148 level offset and robustness.py bootstraps rho, which is affine-invariant and
blind to level. Not the same quantity -- the OOF check above is the apples-to-apples one.)

PUBLIC/PRIVATE COMPLEMENTARITY.  Public 50k and private 200k PARTITION the same 250k, so
    d_priv = 1.25 * D_250 - 0.25 * d_pub
with D_250 the unknown population delta. With a diffuse prior on D_250,
    d_priv | d_pub ~ N( d_pub, (rms(d)/200)^2 )
The uncertainty on the PRIVATE delta is 1.118x the naive public sd: a public lead is worth less
than it looks, because part of it is public-sample luck that must REVERSE on the complement.

DECISION QUANTITY.  Assuming the better of the two finals is what counts, holding a second file
Y alongside the champion X is worth  E[min(0, d_priv)] = m*Phi(-m/s) - s*phi(m/s).
That -- not P(win) -- is what should choose the second pick.
"""
import numpy as np, pandas as pd
from math import erf, sqrt, pi, exp

CAL = 1.10          # estimator correction, from finals_estimator_check.py
TAU = 0.00015       # prior sd on a real champion-class effect (the scale of every one measured)

SUBS = ["e0301_usercv48_cal", "e0303_arch4_cal", "e0300_cal", "e0162", "e0161",
        "e0152", "e0150", "e0361_seqext_cal", "e0201_cal", "e0270_cal",
        "e0200_cal", "e0302_add_d48_cal"]
LB = {"e0301_usercv48_cal": 1.646456, "e0303_arch4_cal": 1.646483, "e0300_cal": 1.646589,
      "e0162": 1.646602, "e0161": 1.646670, "e0152": 1.646697, "e0150": 1.646700,
      "e0361_seqext_cal": 1.646806, "e0201_cal": 1.646831, "e0270_cal": 1.646836,
      "e0200_cal": 1.646868, "e0302_add_d48_cal": 1.647049}
REF = "e0301_usercv48_cal"

def load(name):
    df = pd.read_csv(f"subs/{name}.csv").sort_values("user_id" if "user_id" in
        pd.read_csv(f"subs/{name}.csv", nrows=1).columns else 0)
    return np.log1p(np.maximum(df.iloc[:, 1].astype(np.float64).to_numpy(), 0.0))

def Phi(z): return 0.5 * (1.0 + erf(z / sqrt(2.0)))
def phi(z): return exp(-0.5 * z * z) / sqrt(2 * pi)
def emin(m, s):                       # E[min(0, d)] for d ~ N(m, s^2)
    return m * Phi(-m / s) - s * phi(m / s)

def main():
    P = {n: load(n) for n in SUBS}
    ref = P[REF]
    print(f"reference (public champion): {REF}  LB {LB[REF]:.6f}\n")
    hdr = (f"{'candidate':22s} {'LB':>9s} {'gap':>10s} {'r':>9s} {'rms(d)':>8s} {'sd_priv':>8s} "
           f"{'P(win)':>7s} {'P shr':>6s} {'E[gain]':>9s} {'E[gain]shr':>11s}")
    print("== per-pair private risk and the value of holding it as the 2nd final ==")
    print("   gap>0 = worse on public.  E[gain] = expected RMSLE improvement from holding it too.\n")
    print(hdr); print("-" * len(hdr))
    out = []
    for n in SUBS:
        if n == REF: continue
        d = P[n] - ref
        rms = float(np.sqrt((d ** 2).mean()))
        r = float(np.corrcoef(P[n], ref)[0, 1])
        s = CAL * rms / 200.0                                   # sd(d_priv | d_pub), diffuse
        m = LB[n] - LB[REF]
        se = CAL * rms / np.sqrt(50_000) * np.sqrt(0.8)          # sd(d_pub - D_250)
        m_shr = m * TAU ** 2 / (TAU ** 2 + se ** 2)              # shrunk posterior mean
        row = dict(n=n, lb=LB[n], m=m, r=r, rms=rms, s=s,
                   p=Phi(-m / s), p_shr=Phi(-m_shr / s),
                   g=emin(m, s), g_shr=emin(m_shr, s))
        out.append(row)
        print(f"{n:22s} {LB[n]:9.6f} {m:+10.6f} {r:9.5f} {rms:8.4f} {s:8.6f} "
              f"{row['p']:7.3f} {row['p_shr']:6.3f} {row['g']:+9.6f} {row['g_shr']:+11.6f}")

    best = max(out, key=lambda x: -x["g"])
    best_shr = max(out, key=lambda x: -x["g_shr"])
    print(f"\n   best 2nd pick by E[gain] (diffuse prior): {best['n']}  ({best['g']:+.6f})")
    print(f"   best 2nd pick by E[gain] (shrunk  prior): {best_shr['n']}  ({best_shr['g_shr']:+.6f})")

    print("\n== how identical are the two files the last session recommended? ==")
    a, b = P["e0300_cal"], P["e0162"]
    print(f"   corr(e0300_cal, e0162) = {np.corrcoef(a, b)[0,1]:.8f}   rms(d) = "
          f"{float(np.sqrt(((a-b)**2).mean())):.5f}   max|d| = {np.abs(a-b).max():.4f}")
    print(f"   for scale, corr(e0301, e0303) = {np.corrcoef(ref, P['e0303_arch4_cal'])[0,1]:.8f}"
          f"   corr(e0301, e0300) = {np.corrcoef(ref, a)[0,1]:.8f}")

    print("\n== uniform private penalty from calibrating on public-solved moments ==")
    SD_L = 2.3178
    v = SD_L ** 2 * (1 / 50_000 + 1 / 200_000)
    print(f"   E[(mu_pub - mu_priv)^2] = {v:.3e}  ->  expected RMSLE penalty "
          f"{v / (2 * 1.6465):+.6f} for EVERY calibrated file (does not change the ranking)")

if __name__ == "__main__":
    main()
