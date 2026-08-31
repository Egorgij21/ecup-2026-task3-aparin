#!/usr/bin/env python
"""
E-IDEA-02 -- how much of the target is predictable AT ALL?   See IDEAS.md §I6.

Ten model families, four feature regimes, two CV protocols and 200 experiments all land at
rho ~ 0.704 (EXPERIMENTS.md §1b).  The project reads that as saturation.  It has never been
MEASURED: the classification term was bounded with an oracle (§1b) but rho itself never was.

This bounds it from the data alone, with no model.  Write

    L_t = theta_t + eps_t          theta = the part any predictor could know at the cut-off
                                   eps   = the realisation noise of one 30-day window

For two disjoint windows of equal length at the same instant, eps is independent and theta is
shared, so their correlation IS the reliability of a single window, and classical attenuation
gives the ceiling on any predictor's correlation with L:

    rho_max = sqrt(reliability)

"At the same instant" is not directly observable -- two disjoint windows are necessarily
separated in time, and theta drifts.  So measure corr at lags 1, 2, 3 ... windows and
extrapolate the decay to lag 0.  Doing it at several window LENGTHS is the consistency check:
reliability must rise with window length (more days, less sampling noise), and if it does not,
the estimator is broken.

FALSIFIER, and it is a real one: the estimated ceiling must exceed the rho our models already
achieve on the same period (e0049 OOF: 0.653-0.673 per fold).  A ceiling below the achieved
value means the design is wrong, not that the models are impossible.

All windows are kept inside the CLEAN region (target end <= 2025-11-15).  Everything after
that is the guaranteed-activity zone (DATA.md §4), where every user is active by construction
and every correlation is inflated.

Run:
  python3.11 scripts/noise_ceiling.py                       # 15k local subset
  python3.11 scripts/noise_ceiling.py --full                # all 250k (cluster / big RAM)
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# `screen_features` is a laptop-only helper (psutil memory watchdog, cached subset builder)
# and is not deployed to the cluster.  Import it when present, fall back to no-ops otherwise,
# so this script runs unchanged in both places -- the --full path needs none of it.
try:
    from screen_features import cap_memory, log, make_subset, rss_gb   # noqa: E402
except ModuleNotFoundError:                                            # cluster
    def log(m: str) -> None:
        print(m, flush=True)

    def cap_memory(gb: float) -> None:
        log(f"  (no memory watchdog on this host; requested cap {gb:.0f} GB ignored)")

    def make_subset(n: int):
        raise SystemExit("subset mode needs scripts/screen_features.py; use --full here")

    def rss_gb() -> float:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 1e9

CLEAN_END = date(2025, 11, 15)      # last day outside the guaranteed-activity zone
POP_ANCHOR = date(2025, 4, 19)      # population = active in [A-29, A]; windows start A+1


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def window_sums(p, start_idx: int, length: int, n_win: int):
    """L = log1p(sum gmv) for `n_win` consecutive disjoint windows of `length` days, all users.

    Also returns, per window, the frozen-fold population mask -- users active in the 30 days
    immediately BEFORE that window. Selecting one fixed cohort at the start instead would
    measure reliability on a population the folds never score: the fold rule re-selects on
    recent activity at every anchor (DATA.md §4.4), and that selection is not neutral.
    """
    Ls, pops = [], []
    for j in range(n_win):
        a = start_idx + j * length
        Ls.append(np.log1p(p.wsum("gmv", a, a + length - 1)))
        pops.append(p.active_in(a - 30, a - 1))
    return np.array(Ls), np.array(pops)


def lag_curve(Ls, pops, length, n_win, cond_pos=False):
    """corr(L_j, L_{j+m}) on the population of the EARLIER window -- i.e. exactly the users a
    model at that cut-off would have been asked to predict."""
    lags, vals = [], []
    for m in range(1, n_win):
        cs = []
        for j in range(n_win - m):
            # condition on the LATER window being positive, because that is the window the
            # achieved number conditions on (corr(L, M) among fold users with y > 0).
            # Conditioning on the earlier one instead leaves zeros in the target and mixes the
            # buy/no-buy decision back into what is meant to be a magnitude-only ceiling.
            sel = pops[j] & (Ls[j + m] > 0) if cond_pos else pops[j]
            if sel.sum() > 500:
                cs.append(corr(Ls[j][sel], Ls[j + m][sel]))
        lags.append(m * length)
        vals.append(float(np.mean(cs)))
    return np.asarray(lags, float), np.asarray(vals, float)


def signal_power_ceiling(Ls, pop, n_win):
    """Sahani-Linden / Schoppe signal-power ceiling, using all N windows jointly.

    The noise-ceiling problem is solved in sensory neuroscience, where N repeats of the same
    stimulus give an unbiased split of total power into signal and noise:

        SP = (N * Var(mean over repeats) - mean(Var of each repeat)) / (N - 1)
        ceiling = sqrt(SP / TP),   TP = mean over repeats of Var(L_j)

    Refs: Sahani & Linden (NIPS 2003) signal power; Schoppe et al. (2016) CC_max / CC_norm;
    Pospisil & Bair (PLOS Comp Biol 2021) on the bias of the naive version.

    THE DIRECTION OF ITS BIAS HERE IS THE POINT, and it is opposite to the lag-0 extrapolation.
    The neuroscience setting assumes the signal is IDENTICAL across repeats.  Our windows span
    210 days and theta drifts, so what survives as "signal" is only the component common to ALL
    seven windows -- the persistent trait.  A predictor at the cut-off also knows recent state,
    which this discards.  So this is a LOWER end and the lag-0 extrapolation is an UPPER end;
    together they bracket the ceiling instead of either one claiming it.
    """
    A = Ls[:, pop]                                    # (n_win, users) -- same users throughout
    tp = float(np.mean(A.var(axis=1)))
    sp = (n_win * float(A.mean(axis=0).var()) - tp) / (n_win - 1)
    return sp, tp, float(np.sqrt(max(sp, 0.0) / tp))


def analyse(p, start_idx: int, length: int, n_win: int, label: str, binary: bool = False,
            cond_pos: bool = False):
    """Reliability vs lag for one window length, extrapolated to lag 0, three ways."""
    Ls, pops = window_sums(p, start_idx, length, n_win)
    if binary:
        Ls = (Ls > 0).astype(np.float64)
    lags_a, vals_a = lag_curve(Ls, pops, length, n_win, cond_pos)

    lin = float(np.polyval(np.polyfit(lags_a, vals_a, 1), 0.0))
    exp_ = float(np.exp(np.polyval(np.polyfit(lags_a, np.log(np.maximum(vals_a, 1e-6)), 1), 0.0)))
    # CONSERVATIVE variant.  The lag-1 point is the one that a short-lived shared component
    # -- a purchasing burst straddling a window boundary -- would inflate, and such a
    # component is NOT knowable at the cut-off.  Refitting on lags >= 2 and extrapolating
    # through the lag-1 point removes it.  If lin and cons agree the curve has no such kink.
    cons = (float(np.polyval(np.polyfit(lags_a[1:], vals_a[1:], 1), 0.0))
            if len(lags_a) > 3 else float("nan"))

    # DRIFT correction.  c(k) = r * rho_theta(k*length): the reliability r times the
    # autocorrelation of the knowable state.  A predictor at the cut-off knows theta then,
    # while the target window is centred half a window later, so it loses rho_theta(length/2):
    #     rho_max = sqrt(r) * rho_theta(length/2),  rho_theta(length/2) ~ sqrt(c(1)/r)
    r = max(lin, 1e-9)
    rho_theta_half = float(np.sqrt(max(min(vals_a[0] / r, 1.0), 0.0)))
    drift = float(np.sqrt(r) * rho_theta_half)

    log(f"\n  --- {label}: {length}-day windows, {n_win} of them"
        f"{'  [BINARY 1(y>0)]' if binary else ''} ---")
    log("      lag(d)  " + "  ".join(f"{x:5.0f}" for x in lags_a))
    log("      corr    " + "  ".join(f"{v:.3f}" for v in vals_a))
    log(f"      reliability at lag 0:  linear {lin:.4f}   exponential {exp_:.4f}"
        f"   conservative(lags>=2) {cons:.4f}")
    log(f"      => ceiling on corr(target, any predictor):"
        f"  {np.sqrt(max(cons, 0)):.4f} (conservative)"
        f"  {np.sqrt(max(lin, 0)):.4f} (linear)"
        f"  {drift:.4f} (linear, drift-corrected)")
    return dict(lin=lin, exp=exp_, cons=cons, drift=drift, c1=float(vals_a[0]))


def main(full: bool, max_gb: float):
    from data import Panel
    cap_memory(max_gb)
    # --full uses Panel's own default (src/data.py TRAIN = data/train.parquet), which is where
    # the cluster keeps it; the laptop copy at the repo root is only for the subset builder.
    p = Panel() if full else Panel(path=make_subset(15000))

    ai = p.idx(POP_ANCHOR)
    end_i = p.idx(CLEAN_END)
    span = end_i - ai                       # usable clean days after the population anchor
    log(f"  {p.n_users:,} users; population re-selected per window (active in the prior 30d)")
    log(f"  clean span after {POP_ANCHOR}: {span} days (to {CLEAN_END})")

    res = {}
    for length in (7, 15, 30):
        res[length] = analyse(p, ai + 1, length, span // length, f"W={length}")

    log("\n  --- consistency check: reliability must RISE with window length ---")
    ok = all(res[a]["lin"] <= res[b]["lin"] for a, b in ((7, 15), (15, 30)))
    for W in (7, 15, 30):
        log(f"      W={W:2d}d  r0 {res[W]['lin']:.4f}   ceiling {np.sqrt(max(res[W]['lin'],0)):.4f}")
    log(f"      monotone in window length: {'YES' if ok else 'NO -- estimator suspect'}")

    # The same measurement for the buy/no-buy event.  EXPERIMENTS.md §1b puts 78.6% of
    # Cov(L, M) in this term and declares AUC 0.845 a wall on the evidence that four model
    # classes landed within 0.002 of each other.  That is agreement between estimators, not a
    # bound; this is the bound.
    b30 = analyse(p, ai + 1, 30, span // 30, "W=30 buy/no-buy", binary=True)

    # robustness: the same 30-day measurement on a disjoint, later calendar span.
    # A ceiling that moves with the period is a period artefact, not a bound.
    ai2 = p.idx(date(2025, 6, 18))
    span2 = p.idx(CLEAN_END) - ai2
    res2 = analyse(p, ai2 + 1, 30, span2 // 30, "W=30 @ 2025-06-18 (robustness)")

    # bracket the ceiling from the other side, with all 7 windows at once
    Lw, Pw = window_sums(p, ai + 1, 30, span // 30)
    common = np.all(Pw, axis=0)                       # users present before every window
    sp, tp, sl = signal_power_ceiling(Lw, common, span // 30)
    log(f"\n  --- Sahani-Linden / Schoppe signal-power ceiling (all {span//30} windows, "
        f"{int(common.sum()):,} users present throughout) ---")
    log(f"      signal power {sp:.4f} / total power {tp:.4f} = {sp/tp:.4f}"
        f"   -> ceiling {sl:.4f}")
    log(f"      this DISCARDS recent state (only the component common to all 7 windows "
        f"survives), so it is the LOWER end of the bracket")

    # ---- split the gap into its two terms.  EXPERIMENTS.md 1b puts 78.6% of Cov(L,M) in
    # the buy/no-buy decision and 21.4% in ranking magnitude among buyers, and closed the first
    # on the grounds that four classifiers agreed within 0.002 AUC.  Now both terms have a
    # MEASURED ceiling instead, so the question "which term should a new idea target" has an
    # answer rather than an argument.
    mag = analyse(p, ai + 1, 30, span // 30, "W=30 magnitude | bought in the SCORED window",
                  cond_pos=True)

    d = res[30]
    log(f"\n  === CEILINGS (30-day window, clean region) ===")
    log(f"      GMV      corr(L, predictor)  <=  {np.sqrt(max(d['cons'],0)):.4f} conservative"
        f" | {np.sqrt(max(d['lin'],0)):.4f} linear | {d['drift']:.4f} drift-corrected")
    log(f"      buy flag corr(Z, predictor)  <=  {np.sqrt(max(b30['cons'],0)):.4f} conservative"
        f" | {np.sqrt(max(b30['lin'],0)):.4f} linear | {b30['drift']:.4f} drift-corrected")

    # achieved, on the same quantity, from the OOF this repo already owns
    try:
        import polars as pl
        o = pl.read_parquet(ROOT / "oof" / "e0049.parquet")
        L = np.log1p(o["y_true"].to_numpy()); M = np.log1p(o["y_pred"].to_numpy())
        log(f"\n      ACHIEVED  e0049 OOF: corr(L, M) = {corr(L, M):.4f}"
            f"   corr(1(y>0), M) = {corr((L > 0).astype(float), M):.4f}")
    except Exception as e:                                   # OOF is optional, not required
        log(f"      (achieved comparison skipped: {e})")
    log(f"      ACHIEVED  best submission e0162 on the TEST anchor: rho = 0.70378"
        f"  (EXPERIMENTS.md §1b)")
    sd = 2.3178                                              # solved test sd_L, §1b
    for nm, rr in (("conservative", np.sqrt(max(d['cons'], 0))),
                   ("drift-corrected", d['drift'])):
        log(f"      implied RMSLE floor at the {nm} ceiling: "
            f"{sd * np.sqrt(max(0.0, 1 - rr ** 2)):.4f}   (best submission 1.6466)")

    log(f"      magnitude|buyer corr(L, predictor) <= {np.sqrt(max(mag['cons'],0)):.4f} "
        f"conservative | {np.sqrt(max(mag['lin'],0)):.4f} linear")
    log(f"\n      BRACKET  {sl:.4f} (signal-power, persistent trait only)  ..  "
        f"{np.sqrt(max(d['cons'],0)):.4f} (lag-0 extrapolation, conservative)")
    log(f"\n      ROBUSTNESS  same measurement starting 2025-06-18: "
        f"r0 {res2['lin']:.4f} vs {d['lin']:.4f}  -> ceiling {np.sqrt(max(res2['lin'],0)):.4f}")
    log(f"\n  FALSIFIER: every ceiling above must exceed the achieved value. "
        f"If it does not, the design is wrong -- not the models.")
    log(f"  peak RSS {rss_gb():.2f} GB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="use all 250k users, not the 15k subset")
    ap.add_argument("--max-gb", type=float, default=5.5)
    a = ap.parse_args()
    main(a.full, a.max_gb)
