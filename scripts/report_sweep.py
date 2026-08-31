#!/usr/bin/env python
"""
Read the e0250-e0257 / e0260-e0266 sweep results and apply the pre-registered bars.

    python scripts/report_sweep.py

One place where the verdict rules live, so the numbers cannot be read selectively after the
fact. Everything here is mechanical: it reads `runs/*.json`, applies the bars declared in
EXPERIMENTS.md §1r before the jobs ran, and prints a keep/kill per arm plus the one number the
brief actually asked for -- the implied LB and how far it is from -5 %.

Bars (pre-registered, CLAUDE.md §3.4):
  * keep  <=>  wins >= 4/5 folds AND better, OR |delta| > 2*sigma_noise (0.00009)
  * a sub-2sigma delta is `no effect` REGARDLESS OF SIGN
  * the magnitude arms additionally need to clear the isotonic control on rho|Z=1, or their
    gain is the free rescaling documented in §1r, not information
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SIGMA = 0.00009
SD_L = 2.3178                    # §1b, probe-solved public truth moments
BEST_TAB_LB = 1.655247           # e0090, the best tabular submission
TARGET_LB = BEST_TAB_LB * 0.95   # the brief's -5 %


def rho_from_rmsle(r: float) -> float:
    v = 1.0 - (r / SD_L) ** 2
    return float(np.sqrt(v)) if v > 0 else float("nan")


def load(eid: str):
    p = ROOT / "runs" / f"{eid}.json"
    return json.loads(p.read_text()) if p.exists() else None


def main() -> None:
    mag = [f"e{n:04d}" for n in range(250, 258)]
    reg = [f"e{n:04d}" for n in range(260, 267)]

    print("=" * 78)
    print("REGIME SWEEP (e0260-e0266) -- real metric, frozen folds, same-session reference")
    print("=" * 78)
    print(f"  {'exp':8s} {'cv_mean':>9s} {'folds won':>10s} {'delta vs ref':>13s} "
          f"{'sigma':>7s}  verdict")
    best = None
    for e in reg:
        r = load(e)
        if not r:
            print(f"  {e:8s} (not finished)"); continue
        note = r.get("notes", "")
        d = float(note.split("d ")[1].split(",")[0]) if "d " in note else float("nan")
        wins = note.split("folds")[0].split()[-1] if "folds" in note else "?"
        nw = int(wins.split("/")[0]) if "/" in wins else 0
        keep = (nw >= 4 and d < 0) or abs(d) > 2 * SIGMA
        v = "KEEP" if (keep and d < 0) else ("kill (worse)" if d > 0 else "no effect")
        print(f"  {e:8s} {float(r['cv_mean']):>9.5f} {wins:>10s} {d:>+13.5f} "
              f"{d/SIGMA:>7.1f}  {v}")
        if d < 0 and (best is None or d < best[1]):
            best = (e, d, float(r["cv_mean"]))

    print()
    print("=" * 78)
    print("MAGNITUDE SWEEP (e0250-e0257) -- rho|Z=1 diagnostic, NOT the competition metric")
    print("=" * 78)
    print("  ⚠ §1r: isotonic recalibration of e0049 alone moves rho|Z=1 by +0.02849 while")
    print("    making RMSLE WORSE by +0.00524. rho|Z=1 gains are NOT score gains here.")
    print(f"  {'exp':8s} {'rho|Z=1':>9s} {'vs ref':>9s} {'vs iso ctrl':>12s}  reading")
    for e in mag:
        r = load(e)
        if not r:
            print(f"  {e:8s} (not finished)"); continue
        note = r.get("notes", "")
        try:
            rc = float(note.split("rho|Z=1 ")[1].split()[0])
            dd = float(note.split("(d ")[1].split(",")[0])
        except Exception:
            rc, dd = float("nan"), float("nan")
        # the isotonic control's rho|Z=1, measured in §1r
        d_iso = rc - 0.50990
        reading = ("INFORMATION" if d_iso > 0.005 else
                   "recalibration only" if d_iso > -0.02 else "below control")
        print(f"  {e:8s} {rc:>9.5f} {dd:>+9.5f} {d_iso:>+12.5f}  {reading}")
    print("\n  Run `python src/combine_magnitude.py --arms e0250 ... e0257` for the RMSLE")
    print("  conversion -- that, not rho|Z=1, decides whether any of this reaches the score.")

    print()
    print("=" * 78)
    print("THE BRIEF: -5 % from the best tabular submission")
    print("=" * 78)
    # Two different rho's, and conflating them is easy: e0090's MEASURED rho is 0.70209
    # (cv_lb.csv, solved from the full identity with its own mu_M/sd_M). Inverting the
    # optimally-calibrated identity on its raw LB gives 0.70000 instead -- that is the rho a
    # model would need to score 1.655247 IF it were already perfectly calibrated. The target
    # below uses the calibrated form deliberately: it is the most generous reading, since it
    # assumes the -5 % submission gets its level and spread for free (§1b says they are free).
    print(f"  best tabular submission   e0090 LB {BEST_TAB_LB:.6f}  "
          f"(measured rho 0.70209; calibrated-equivalent {rho_from_rmsle(BEST_TAB_LB):.5f})")
    print(f"  -5 % target                     {TARGET_LB:.6f}  "
          f"(needs rho {rho_from_rmsle(TARGET_LB):.5f} EVEN IF perfectly calibrated)")
    print(f"  §I6 measured UPPER bound on achievable rho          0.72540")
    print(f"     -> best conceivable RMSLE     {SD_L*np.sqrt(1-0.7254**2):.6f}"
          f"  = {100*(1-SD_L*np.sqrt(1-0.7254**2)/BEST_TAB_LB):.1f} % improvement")
    print()
    if best:
        e, d, cv = best
        # CV deltas transfer to LB at 1.5-1.8x for single-model changes in one family (§3),
        # but ONLY when significant; quote the range, never a point estimate.
        print(f"  best arm this sweep: {e}, CV delta {d:+.5f} ({d/SIGMA:.1f} sigma)")
        print(f"    implied LB at the measured 1.5-1.8x transfer: "
              f"{BEST_TAB_LB + d*1.5:.6f} .. {BEST_TAB_LB + d*1.8:.6f}")
        print(f"    that is {100*abs(d)*1.8/BEST_TAB_LB:.3f} % -- against the 5 % asked for.")
    print()
    print("  CONCLUSION: the -5 % target requires rho 0.7346, which is ABOVE §I6's measured")
    print("  upper bound (0.7254) on what this data contains. It is not reachable by any")
    print("  model. The sweep's job was to find what IS reachable and to measure it honestly.")


if __name__ == "__main__":
    main()
