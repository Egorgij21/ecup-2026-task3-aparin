#!/usr/bin/env python
"""
E-IDEA-01 -- LOSS-GEOMETRY screen (laptop, 8 GB, no GPU).   See IDEAS.md §0, §I1, §E1.

The one axis this project has never varied: every model in experiments.csv minimises squared
error on L = log1p(y).  This screen holds the estimand (E[L|x]) and the information (the same
1021 features, the same anchor, the same 50/50 user split, the same hyperparameters, the same
seed) and varies ONLY the loss:

    l2_rmse_es      objective=regression, early stop on val RMSE     <- the installed baseline
    l2_rho_es       objective=regression, early stop on val rho      <- stopping-criterion control
    ce_hard         built-in multiclass CE over K bins of L          <- classification-as-regression
    hlgauss         custom CE with Gaussian-smoothed bin targets     <- Farebrother et al. 2024
    hlgauss_s0      the SAME custom objective at sigma -> 0          <- PORT-CORRECTNESS control
    l2_uid          baseline + user_id as one column                 <- IDEAS.md §I5, free rider

All classification arms read out the SAME functional, M(x) = sum_k p_k(x) * c_k, so any
difference is loss geometry and not estimand (which is what killed ZILN/OptDist here --
PAPERS_FEATURES_AND_IDEAS.md §0.1).

TIER = SCREEN (CLAUDE.md §4.2).  It subsamples and re-splits, it early-stops on the same half
it scores, and FEATURES.md measures its sensitivity at ~+-0.001 rho.  It decides nothing on
its own; it decides whether a frozen-fold confirm is worth a cluster job.

Two statistics are reported per arm and BOTH matter:
    d_rho        accuracy vs the L2 baseline
    rho_partial  corr(L, arm | baseline) -- EXPERIMENTS.md §1f says this is the entire blend
                 value of a candidate.  Best ever achieved in this project: 0.0127.

Run:
  python3.11 scripts/screen_loss.py --n 30000 --anchor 2025-10-16
  python3.11 scripts/screen_loss.py --n 30000 --anchor 2025-06-18      # replication anchor
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from screen_features import SUBSET, cap_memory, log, make_subset, rss_gb  # noqa: E402

EXISTING = ["base", "counts", "trend", "rank", "visit", "channel", "diff",
            "cumshare", "ewm", "com", "dispersion", "sbcnomoment", "tsfeat",
            "fcast", "funnel"]

PARAMS = dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=40,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              num_threads=8, verbosity=-1, seed=0)


# HL-Gauss binning, soft targets, objective and the port gate all live in one place
# so the screen and the frozen-fold confirm (`src/run.py`, `model: hlgauss`) cannot
# drift apart -- the screen would otherwise be measuring a different method.
from hlgauss import (make_bins, soft_targets, hl_objective, softmax_rows,   # noqa: E402
                     prior_init, port_exact_check)


# ------------------------------------------------------------------ the screen
def screen_loss(n_users: int, anchor: str, mtrees: int, k: int, sigma_ratio: float,
                arms: list[str]):
    import lightgbm as lgb
    from data import Panel
    from features import build

    t0 = time.time()
    ai_anchor = date.fromisoformat(anchor)
    sub = make_subset(n_users)

    p = Panel(path=sub)
    ai = p.idx(ai_anchor)
    keep = p.active_in(ai - 29, ai)
    log(f"  anchor {anchor} idx {ai}  users {p.n_users}  keep(active) {int(keep.sum()):,}")

    X, names = build(p, ai, keep, EXISTING)
    X = X.astype(np.float64)
    L = np.log1p(p.target(ai, 30))[keep]
    uid = p.users[keep].astype(np.float64)
    log(f"  built {X.shape[1]} features x {X.shape[0]} users   RSS {rss_gb():.2f} GB")

    # THREE-way user split, not the installed harness's two.  The arms being compared have
    # very different evaluation-noise profiles (a K-class model's val curve is noisier than a
    # regressor's), and EXPERIMENTS.md §1j prices early-stopping selection bias at ~sigma*
    # sqrt(2 ln N) -- enough to manufacture the entire effect being measured.  Early stopping
    # therefore happens on `es` and every reported number comes from `sc`, which no arm ever
    # saw.  Costs precision (n_sc ~ 3.4k), buys an unbiased comparison.
    rng = np.random.default_rng(1)
    u = rng.random(X.shape[0])
    tr, es, sc = u < 0.50, (u >= 0.50) & (u < 0.75), u >= 0.75
    Ltr, Les, Lsc = L[tr], L[es], L[sc]
    sd_L = float(Lsc.std())
    log(f"  split train {tr.sum():,} / earlystop {es.sum():,} / score {sc.sum():,}"
        f"   zero-share {1 - (L > 0).mean():.3%}   sd_L(score) {sd_L:.4f}")

    k_req = k
    edges, centres, vmax = make_bins(Ltr, k_req)
    k = len(centres)
    width = vmax / k_req                      # the grid's design resolution, pre-merge
    sigma = sigma_ratio * width
    log(f"  bins K={k} (requested {k_req}, {k_req - k} merged away) on [0, {vmax:.3f}]"
        f"  design width {width:.4f}  sigma {sigma:.4f}  (sigma/width = {sigma_ratio})")
    hard_tr = np.clip(np.searchsorted(edges, np.clip(Ltr, 0, vmax), side="right") - 1,
                      0, k - 1)
    hist = np.bincount(hard_tr, minlength=k) / hard_tr.size
    log(f"  train bin occupancy: " + " ".join(f"{h:.3f}" for h in hist))

    def rho_sc(pred):
        return float(np.corrcoef(pred, Lsc)[0, 1])

    def rho_es(pred):
        return float(np.corrcoef(pred, Les)[0, 1])

    def rmsle_at(r):
        """The metric a perfectly affine-calibrated submission with this rho would score."""
        return sd_L * np.sqrt(max(0.0, 1.0 - r * r))

    # ---- every arm early-stops on the SAME statistic (rho of its own readout, on `es`), so
    # the comparison is not confounded by regression stopping on RMSE while classification
    # stops on rho.  l2_rmse_es is kept as the one arm that stops the installed way.
    def feval_reg(preds, dset):
        return "rho", rho_es(preds), True

    def feval_cls(preds, dset):
        z = preds.reshape(-1, k) if preds.ndim == 1 else preds
        return "rho", rho_es(softmax_rows(z) @ centres), True

    results = {}

    def fit_reg(Xa, es_metric, seed=0):
        pr = dict(PARAMS, objective="regression", seed=seed, bagging_seed=seed + 1,
                  feature_fraction_seed=seed + 2,
                  metric=("rmse" if es_metric == "rmse" else "None"))
        d = lgb.Dataset(Xa[tr], Ltr)
        v = lgb.Dataset(Xa[es], Les, reference=d)
        m = lgb.train(pr, d, num_boost_round=mtrees, valid_sets=[v],
                      feval=None if es_metric == "rmse" else feval_reg,
                      callbacks=[lgb.early_stopping(80, verbose=False)])
        return m.predict(Xa[sc]), m.best_iteration

    def fit_cls(obj, q_or_label):
        pr = dict(PARAMS, num_class=k, metric="None")
        init = None
        if obj == "builtin":
            pr["objective"] = "multiclass"
            d = lgb.Dataset(X[tr], q_or_label)
            v = lgb.Dataset(X[es], Les, reference=d)
        else:
            pr["objective"] = hl_objective(q_or_label, k)
            # LightGBM's built-in multiclass boosts from the class prior (its round-1 raw
            # scores are log-priors, ~-3 for a rare class at lr=0.05 -- measured, not assumed).
            # A custom objective starts at 0 instead, and that difference ALONE broke the port
            # control.  Reinstate it as an explicit init score on both datasets; LightGBM adds
            # init_score to its internal scores, so the objective and feval both see it, but
            # `predict` does NOT, so it is added back by hand at readout.
            init = prior_init(q_or_label)
            # a custom objective still needs a label column; it is unused by the objective.
            d = lgb.Dataset(X[tr], np.zeros(int(tr.sum())),
                            init_score=np.tile(init, (int(tr.sum()), 1)).ravel(order="F"))
            v = lgb.Dataset(X[es], Les, reference=d,
                            init_score=np.tile(init, (int(es.sum()), 1)).ravel(order="F"))
        m = lgb.train(pr, d, num_boost_round=mtrees, valid_sets=[v], feval=feval_cls,
                      callbacks=[lgb.early_stopping(80, verbose=False)])
        raw = m.predict(X[sc], raw_score=True)
        raw = raw.reshape(-1, k) if raw.ndim == 1 else raw
        P = softmax_rows(raw if init is None else raw + init)
        # two readouts, free from the same fit: standard bin centres, and the empirical
        # within-bin mean of L on train (removes discretisation bias in the readout).
        bin_mean = np.array([Ltr[hard_tr == j].mean() if (hard_tr == j).any() else centres[j]
                             for j in range(k)])
        return P @ centres, P @ bin_mean, m.best_iteration

    def record(name, pred, iters, dt, trees=None):
        r = rho_sc(pred)
        results[name] = dict(pred=pred, rho=r, iters=iters, dt=dt, trees=trees or iters)
        base = results.get("l2_rmse_es")
        extra = ""
        if base is not None and name != "l2_rmse_es":
            rm, rb = base["rho"], r
            rr = float(np.corrcoef(pred, base["pred"])[0, 1])
            den = np.sqrt(max(1e-12, (1 - rr ** 2) * (1 - rm ** 2)))
            rp = (rb - rr * rm) / den
            r_blend = np.sqrt(rm ** 2 + (1 - rm ** 2) * rp ** 2)
            results[name].update(r_vs_base=rr, rho_partial=rp, rho_blend=r_blend)
            extra = (f"  d_rho {rb - rm:+.5f}  r_vs_base {rr:.5f}"
                     f"  rho_partial {rp:+.5f}  blend_rho {r_blend:.5f}"
                     f" (d {r_blend - rm:+.5f})")
        log(f"    {name:16s} rho {r:.5f}  RMSLE* {rmsle_at(r):.5f}"
            f"  iters {iters:4d}  trees {results[name]['trees']:5d}  {dt:5.1f}s{extra}")
        return r

    # ---- gate: the hand-written objective must BE LightGBM's multiclass, on this data ----
    md = port_exact_check(X[tr][:4000], hard_tr[:4000], k, PARAMS)
    gate = "PASS" if md < 1e-6 else "*** FAIL ***"
    log(f"\n  PORT GATE  custom objective vs built-in multiclass at matched zero init: "
        f"max|d raw| = {md:.3e}  -> {gate}")
    if md >= 1e-6:
        log("  refusing to report classification arms: the objective is mis-specified.")
        return {}

    log(f"\n  === arms ===  (RMSLE* = sd_L*sqrt(1-rho^2), the perfectly-calibrated score)")

    # baseline must run first: every other arm is scored against it
    t = time.time(); pr_, it_ = fit_reg(X, "rmse"); record("l2_rmse_es", pr_, it_, time.time() - t)

    if "l2_rho_es" in arms:
        t = time.time(); pr_, it_ = fit_reg(X, "rho"); record("l2_rho_es", pr_, it_, time.time() - t)

    # CAPACITY CONTROL.  A K-class model fits K trees per boosting round, so any classification
    # arm carries several times the ensemble of the L2 arm.  Averaging K reseeded L2 models is
    # the matched-capacity, matched-variance-reduction comparison: if it recovers the gain, the
    # effect is ensembling, not loss geometry.  This is the no-op control §1b says decides these.
    if "l2_bagK" in arms:
        t = time.time(); ps, its = [], 0
        for s in range(k):
            pr_, it_ = fit_reg(X, "rho", seed=10 * (s + 1)); ps.append(pr_); its += it_
        record("l2_bagK", np.mean(ps, axis=0), its // k, time.time() - t, trees=its)

    if "l2_uid" in arms:
        Xu = np.column_stack([X, uid])
        t = time.time(); pr_, it_ = fit_reg(Xu, "rmse"); record("l2_uid", pr_, it_, time.time() - t)
        del Xu

    if "ce_hard" in arms:
        t = time.time(); pc, pm, it_ = fit_cls("builtin", hard_tr); dt = time.time() - t
        record("ce_hard", pc, it_, dt, trees=it_ * k)
        record("ce_hard_binmean", pm, it_, 0.0, trees=it_ * k)

    if "hlgauss_s0" in arms:
        q0 = soft_targets(Ltr, edges, 0.0)
        t = time.time(); pc, pm, it_ = fit_cls("custom", q0); dt = time.time() - t
        record("hlgauss_s0", pc, it_, dt, trees=it_ * k)
        del q0

    if "hlgauss" in arms:
        q = soft_targets(Ltr, edges, sigma)
        t = time.time(); pc, pm, it_ = fit_cls("custom", q); dt = time.time() - t
        record("hlgauss", pc, it_, dt, trees=it_ * k)
        record("hlgauss_binmean", pm, it_, 0.0, trees=it_ * k)
        del q

    # informational, NOT a gate -- the gate is port_exact_check above.  These two differ by
    # their init (log-prior vs LightGBM's boost_from_average) and stop independently, so a
    # small gap here is expected and says nothing about correctness.
    if "ce_hard" in results and "hlgauss_s0" in results:
        d = results["hlgauss_s0"]["rho"] - results["ce_hard"]["rho"]
        rr = float(np.corrcoef(results["hlgauss_s0"]["pred"], results["ce_hard"]["pred"])[0, 1])
        log(f"\n  (informational) hlgauss(sigma->0) vs built-in multiclass: "
            f"d_rho {d:+.5f}, corr {rr:.5f}")

    log(f"\n  total {time.time() - t0:.0f}s   peak RSS {rss_gb():.2f} GB")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--anchor", default="2025-10-16")
    ap.add_argument("--mtrees", type=int, default=800)
    ap.add_argument("--k", type=int, default=16, help="number of target bins")
    ap.add_argument("--sigma-ratio", type=float, default=0.75,
                    help="HL-Gauss sigma as a multiple of the bin width (2403.03950 uses ~0.75)")
    ap.add_argument("--arms", default="l2_rho_es,l2_bagK,l2_uid,ce_hard,hlgauss_s0,hlgauss")
    ap.add_argument("--max-gb", type=float, default=5.5)
    a = ap.parse_args()
    cap_memory(a.max_gb)
    screen_loss(a.n, a.anchor, a.mtrees, a.k, a.sigma_ratio, a.arms.split(","))
