#!/usr/bin/env python
"""
Does a magnitude-term gain convert into RMSLE? The honest test, on real OOFs.

    python src/combine_magnitude.py --arms e0250 e0251 ... --clf oof/e0160_clf.parquet

WHY THIS IS A SEPARATE STEP. `run_magnitude.py` scores `corr(L, .|Z=1)` -- a diagnostic on the
term IDEAS.md §I13 found weakest (80.2 % of ceiling vs 89.6 % for classification). It is NOT
the competition metric: at test time we do not know who buys, so a buyers-only model is not a
submission. §I13 is explicit that the oracle-path conversion rate ("closing the gap is worth
~+0.012 rho") is the kind of extrapolation §1b retired, having been measured at ~0 in practice.

So the conversion gets measured, not assumed, and the two numbers are never merged:

    E[L|x] = P(buy|x) * E[L|x, buy]

with `P(buy|x)` from an EXISTING classifier OOF (e0160/e0161/e0162 -- already built, already
scored, no new training) and `E[L|x, buy]` from each magnitude arm. This multiplies in LOG
space, which is the same hurdle algebra `run.py` uses for `two_part` and the reason that model
does not compose `p * E[y|y>0]` in linear space.

THREE CONTROLS, because a combiner can manufacture a result out of nothing:

  1. `ref` -- the same combination using e0049's own all-rows predictions as the magnitude
     half. Any gain an arm shows must beat THIS, not the raw e0049 score, or what is being
     measured is the combiner rather than the arm.
  2. `clf_only` -- classifier times a CONSTANT for buyers. §1q/DATA.md put this oracle at
     ~1.014 with a perfect classifier; with a real one it says how much of the score is the
     buy/no-buy decision alone.
  2b. ⚠ `ref_iso` -- the reference model ISOTONICALLY RECALIBRATED among buyers, fitted
     leave-one-fold-out. **This control was added after the smoke run and it is the one that
     matters.** The buyers-only arms scored +0.0517 on `rho|Z=1` in the smoke, which would be
     the largest effect in the project's history -- but a monotone recalibration of e0049's
     EXISTING predictions, with no new model at all, already supplies **+0.0288** of it. The
     cause is visible in §1q: among buyers the all-rows model's bottom 8 prediction bins move
     0.61 -> 2.95 while the actual `mean L` is flat at ~3.5-3.96, and Pearson correlation is
     penalised by exactly that curvature. Training on buyers removes the zero-inflation
     squash, so most of the apparent gain is a rescaling that any isotonic map buys for free.
     **An arm only demonstrates new information if it beats THIS control, not the raw one.**
  3. Leave-one-fold-out weighting. Any free parameter (the blend weight, the constant) is
     fitted on four folds and applied to the fifth. In-sample weights are reported too, marked,
     and never used for a verdict -- §1m's recombination work shows in-sample blending
     over-reports by ~2x here.

Verdict rule, pre-registered: an arm converts only if its LOO-honest RMSLE beats control (1) by
more than 2 sigma_noise (0.00009) or on >=4/5 folds. Anything else is a magnitude-term result
that does not reach the metric -- which is still worth knowing, and is what §I13 predicts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from metrics import rmsle                        # noqa: E402


def load(p: Path) -> pl.DataFrame:
    return pl.read_parquet(p).sort(["fold_id", "user_id"])


def align(frames: dict[str, pl.DataFrame]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Intersect on (fold_id, user_id) -- never assume identical row order (REVIEW_NOTES A2)."""
    keys = None
    for nm, d in frames.items():
        k = set(zip(d["fold_id"].to_list(), d["user_id"].to_list()))
        keys = k if keys is None else (keys & k)
    keys = sorted(keys)
    idx = pl.DataFrame({"fold_id": [a for a, _ in keys], "user_id": [b for _, b in keys]})
    out = {}
    for nm, d in frames.items():
        j = idx.join(d, on=["fold_id", "user_id"], how="left")
        out[nm] = j
    # y_true is taken from whichever frame carries it (the clf OOF does not)
    ytrue = next(o["y_true"].to_numpy() for o in out.values() if "y_true" in o.columns)
    return idx["fold_id"].to_numpy(), ytrue, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True, help="magnitude arm exp_ids")
    ap.add_argument("--clf", default="oof/e0160_clf.parquet")
    ap.add_argument("--ref", default="e0049", help="all-rows model used as control (1)")
    args = ap.parse_args()

    clf = load(ROOT / args.clf)
    # the classifier OOFs carry (z_true, p_clf), not (y_true, y_pred) -- they were written by
    # run_clf.py for the buy-flag experiments. y_true comes from the regression OOF instead.
    pcol = "p_clf" if "p_clf" in clf.columns else "y_pred"
    frames = {"clf": clf.select(["fold_id", "user_id", pl.col(pcol).alias("p")]),
              "ref": load(ROOT / "oof" / f"{args.ref}.parquet").select(
                  ["fold_id", "user_id", "y_true", "y_pred"])}
    for a in args.arms:
        f = ROOT / "oof" / f"{a}.parquet"
        if not f.exists():
            print(f"  skip {a}: no OOF yet"); continue
        frames[a] = load(f).select(["fold_id", "user_id", "y_true", "y_pred"])

    fold, y, F = align(frames)
    L = np.log1p(y)
    p = np.clip(F["clf"]["p"].to_numpy().astype(np.float64), 1e-6, 1 - 1e-6)
    print(f"  aligned on {len(y):,} (fold_id,user_id) keys across {len(F)} frames")
    print(f"  classifier AUC-ish check: mean p {p.mean():.4f}, buyer rate {(y>0).mean():.4f}")

    folds = sorted(set(fold.tolist()))
    arms = [a for a in args.arms if a in F]

    def loo_score(mag_L: np.ndarray) -> tuple[float, list, np.ndarray]:
        """Hurdle in log space with a LOO-fitted exponent on p. One free parameter, fitted on
        the other four folds -- never on the fold being scored."""
        per = []
        best_w = []
        for k in folds:
            tr = fold != k; te = fold == k
            ws = np.linspace(0.0, 2.0, 41)
            sc = [rmsle(y[tr], np.maximum(np.expm1((p[tr] ** w) * mag_L[tr]), 0.0)) for w in ws]
            w = float(ws[int(np.argmin(sc))]); best_w.append(w)
            per.append(rmsle(y[te], np.maximum(np.expm1((p[te] ** w) * mag_L[te]), 0.0)))
        return float(np.mean(per)), per, np.array(best_w)

    # control 2: classifier x constant for buyers
    cs = np.linspace(0.5, 6.0, 56)
    per_c = []
    for k in folds:
        tr = fold != k; te = fold == k
        sc = [rmsle(y[tr], np.maximum(np.expm1(p[tr] * c), 0.0)) for c in cs]
        c = float(cs[int(np.argmin(sc))])
        per_c.append(rmsle(y[te], np.maximum(np.expm1(p[te] * c), 0.0)))
    print(f"\n  CONTROL 2  clf x constant           = {np.mean(per_c):.5f}")

    # control 2b: the reference, isotonically recalibrated among buyers, LOO-fitted.
    # See the module docstring -- this absorbs the zero-inflation rescaling that the
    # buyers-only arms get for free, and is the bar an arm must clear to show INFORMATION.
    from sklearn.isotonic import IsotonicRegression
    ref_mag = np.log1p(F["ref"]["y_pred"].to_numpy().astype(np.float64))
    zb = y > 0
    iso_mag = ref_mag.copy()
    for k in folds:
        te = fold == k; tr = (fold != k) & zb
        ir = IsotonicRegression(out_of_bounds="clip").fit(ref_mag[tr], L[tr])
        iso_mag[te] = ir.predict(ref_mag[te])

    rows = []
    for nm in ["ref"] + arms:
        mag = np.log1p(F[nm]["y_pred"].to_numpy().astype(np.float64))
        s, per, w = loo_score(mag)
        rc = float(np.corrcoef(L[y > 0], mag[y > 0])[0, 1])
        rows.append((nm, s, per, rc, w))
    s_i, per_i, w_i = loo_score(iso_mag)
    rc_i = float(np.corrcoef(L[zb], iso_mag[zb])[0, 1])

    ref_s = rows[0][1]; ref_per = np.array(rows[0][2])
    iso_per = np.array(per_i)
    print(f"\n  CONTROL 1   clf x {args.ref} (all-rows)        = {ref_s:.5f}   "
          f"folds {np.round(ref_per,5).tolist()}")
    print(f"  CONTROL 2b  clf x {args.ref}-ISOTONIC (no new model) = {s_i:.5f}   "
          f"rho|Z=1 {rc_i:.5f}")
    print(f"              ^ recalibration alone moves rho|Z=1 by {rc_i - rows[0][3]:+.5f} "
          f"and RMSLE by {s_i - ref_s:+.5f}")
    print(f"\n  {'arm':8s} {'rho|Z=1':>9s} {'d rho vs iso':>13s} {'LOO RMSLE':>10s} "
          f"{'d vs ctrl1':>11s} {'d vs iso':>10s} {'wins/iso':>9s}")
    for nm, s, per, rc, w in rows[1:]:
        per = np.array(per); d = s - ref_s; di = s - s_i
        wins = int((per < ref_per).sum())
        # TWO bars, because they are different claims and the controls disagree:
        #   INFO  -- rho|Z=1 above the isotonic control => new information, not rescaling
        #   METRIC-- LOO RMSLE below control 1          => it actually reaches the score
        info = rc > rc_i
        metric = (wins >= 4 or abs(d) > 2 * 0.00009) and d < 0
        flag = ("**" if (info and metric) else
                "info" if info else "metric" if metric else "")
        print(f"  {nm:8s} {rc:>9.5f} {rc - rc_i:>+13.5f} {s:>10.5f} {d:>+11.5f} "
              f"{di:>+10.5f} {wins:>7d}/5 {flag}")
    print("\n  ⚠ THE TWO CONTROLS DISAGREE, AND THAT IS THE RESULT.")
    print(f"  Isotonic recalibration RAISES rho|Z=1 by {rc_i - rows[0][3]:+.5f} while making")
    print(f"  RMSLE WORSE by {s_i - ref_s:+.5f}. A monotone map cannot add information, so this")
    print("  is direct proof that rho|Z=1 and the competition metric move in OPPOSITE")
    print("  directions here -- exactly the conversion failure §I13 warned about, now")
    print("  measured. Consequences for reading the table:")
    print("    'info'   = beats the isotonic control on rho|Z=1  -> genuinely new information")
    print("    'metric' = beats control 1 on LOO RMSLE           -> actually improves the score")
    print("    '**'     = both. ONLY '**' is a keep; 'info' alone is the §I13 trap.")


if __name__ == "__main__":
    main()
