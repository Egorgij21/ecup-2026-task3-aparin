#!/usr/bin/env python
"""
Two ways to identify "gifters", one shared multiplier, two submissions.

APPROACH 1 -- statistical shift.
    A gifter is someone who visibly spiked in the 2025 gift run-ups (9-22 Feb and
    23 Feb-7 Mar) relative to their OWN later-year baseline. Assumption: the behaviour
    repeats in 2026.

APPROACH 2 -- learned classifier.
    Train on 2025: label = "bought more in 14 Feb-15 Mar 2025 than their pre-window level
    predicts" (top decile of the residual, so the label is level-free, not "rich").
    Features = 1 Jan - 13 Feb 2025 ONLY. Then apply the same feature recipe to
    1 Jan - 13 Feb 2026 and take the top decile by predicted probability.
    The 44-day feature window is identical in both years by construction, so the model is
    not extrapolating onto a different feature distribution.

Both then multiply the e0020 prediction by k for their gifter set and leave every other
user untouched.

Honest context, so the results are read correctly: on the frozen folds the model's residual
is uncorrelated with spiking (corr +0.0001) and every k > 1 made CV worse. These submissions
test the holiday-specific version of the idea, which CV cannot reach, and the upper bound on
the prize is about -0.0008 RMSLE -- below LB noise. Treat the two scores as a directional
read, not a significance test.
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from data import Panel                      # noqa: E402

OUT = ROOT / "reports" / "eda"
R: dict = {}


def hdr(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}", flush=True)


p = Panel()
import lightgbm as lgb

# ---------------------------------------------------------------- feature recipe
def feats(cut: date):
    """Features from 1 January of `cut`'s year up to `cut` -- identical shape in both years."""
    ci = p.idx(cut)
    y0 = p.idx(date(cut.year, 1, 1))
    span = ci - y0 + 1
    cols, names = [], []

    def add(v, n):
        cols.append(np.asarray(v, np.float32)); names.append(n)

    for w in [7, 14, 30, span]:
        a = ci - w + 1
        nm = "all" if w == span else str(w)
        add(p.wsum("gmv", a, ci), f"gmv_{nm}")
        add(p.wsum("ord", a, ci), f"ord_{nm}")
        add(p.wsum("cart", a, ci), f"cart_{nm}")
        add(p.wsum("srch", a, ci), f"srch_{nm}")
        add(p.wdays(a, ci), f"days_{nm}")
        add(p.wbuy(a, ci), f"buydays_{nm}")
    g, o, d = p.wsum("gmv", y0, ci), p.wsum("ord", y0, ci), p.wdays(y0, ci)
    add(g / np.maximum(o, 1), "aov")
    add(g / np.maximum(d, 1), "gmv_per_day")
    add(o / np.maximum(d, 1), "ord_per_day")
    add(d / span, "active_rate")
    add(p.wbuy(y0, ci) / np.maximum(d, 1), "buy_rate")
    add(np.minimum(p.recency(ci), span), "recency")
    add(np.minimum(p.recency_order(ci), span), "recency_ord")
    add(p.wsum("gmvc", y0, ci) / np.maximum(g, 1e-9), "cat_share")
    add(p.wdate_std("days", y0, ci), "date_sd_act")
    add(p.wdate_std("buy", y0, ci), "date_sd_buy")
    s1, s2 = p.wsum("ord", y0, ci), p.wsumsq("ord", y0, ci)
    mu = s1 / span
    add(np.where(mu > 0, (s2 / span - mu ** 2) / np.maximum(mu, 1e-9), -1), "fano_ord")
    add(p.wsum("gmv", ci - 6, ci) / np.maximum(p.wsum("gmv", y0, ci), 1e-9), "gmv_share_last7")
    return np.column_stack(cols), names


CUT25, CUT26 = date(2025, 2, 13), date(2026, 2, 13)
TGT25 = (date(2025, 2, 14), date(2025, 3, 15))

hdr("1 -- BUILD THE 2025 TRAINING SET")
X25, names = feats(CUT25)
X26, n2 = feats(CUT26)
assert names == n2
pre25 = p.wsum("gmv", p.idx(date(2025, 1, 1)), p.idx(CUT25))
tgt25 = p.wsum("gmv", p.idx(TGT25[0]), p.idx(TGT25[1]))
Lt, Lp = np.log1p(tgt25), np.log1p(pre25)
A = np.column_stack([np.ones(Lp.size), Lp])
coef, *_ = np.linalg.lstsq(A, Lt, rcond=None)
resid25 = Lt - A @ coef                       # "bought more than their level predicts"
thr = np.quantile(resid25, 0.90)
lab = (resid25 >= thr).astype(np.int8)
print(f"  features: {X25.shape[1]} columns, window 1 Jan - 13 Feb of each year")
print(f"  label = top decile of residual(log1p target ~ log1p pre-window level)")
print(f"  positives: {int(lab.sum()):,} / {lab.size:,}")
print(f"  mean residual  positives {resid25[lab == 1].mean():+.4f}   "
      f"others {resid25[lab == 0].mean():+.4f}")

hdr("2 -- TRAIN AND HONESTLY EVALUATE THE CLASSIFIER (user-split, 2025 only)")
rng = np.random.default_rng(0)
fold = rng.integers(0, 5, size=lab.size)
params = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, num_threads=16, verbosity=-1, seed=0)
aucs, oof_pred = [], np.zeros(lab.size)
for k in range(5):
    tr, va = fold != k, fold == k
    m = lgb.train(params, lgb.Dataset(X25[tr], lab[tr], feature_name=names),
                  num_boost_round=400,
                  valid_sets=[lgb.Dataset(X25[va], lab[va], feature_name=names)],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    oof_pred[va] = m.predict(X25[va], num_iteration=m.best_iteration)
    aucs.append(m.best_score["valid_0"]["auc"])
print(f"  5-fold AUC on 2025: {np.mean(aucs):.4f}  (folds {np.round(aucs, 4).tolist()})")
print(f"  -> AUC 0.5 would mean gifting is unpredictable from pre-window behaviour")
R["auc"] = float(np.mean(aucs))

hdr("3 -- HOW MUCH DO PREDICTED GIFTERS ACTUALLY OVER-SPEND? (the multiplier evidence)")
q = np.quantile(oof_pred, 0.90)
sel = oof_pred >= q
print(f"  top-decile predicted gifters (2025, out-of-fold): {int(sel.sum()):,}")
print(f"  their mean residual = {resid25[sel].mean():+.4f}   others {resid25[~sel].mean():+.4f}")
excess = resid25[sel].mean() - resid25[~sel].mean()
print(f"  excess = {excess:+.4f} log-points  ->  implied multiplier exp({excess:.4f}) = "
      f"{np.exp(excess):.3f}")
R["excess_log"] = float(excess); R["implied_k"] = float(np.exp(excess))

hdr("4 -- BUILD THE TWO GIFTER SETS FOR 2026")
full = lgb.train(params, lgb.Dataset(X25, lab, feature_name=names), num_boost_round=400,
                 callbacks=[lgb.log_evaluation(0)])
prob26 = full.predict(X26)
S_clf = prob26 >= np.quantile(prob26, 0.90)

PRE23, PRE8M = (date(2025, 2, 9), date(2025, 2, 22)), (date(2025, 2, 23), date(2025, 3, 7))
REF = (date(2025, 6, 1), date(2025, 12, 15))
runup = (p.wsum("gmv", p.idx(PRE23[0]), p.idx(PRE23[1]))
         + p.wsum("gmv", p.idx(PRE8M[0]), p.idx(PRE8M[1]))) / 27.0
ref = p.wsum("gmv", p.idx(REF[0]), p.idx(REF[1])) / 197.0
spike = np.log1p(runup) - np.log1p(ref)
elig = ref > 0
S_stat = elig & (spike >= np.quantile(spike[elig], 0.90))

print(f"  approach 1 (statistical shift) : {int(S_stat.sum()):,} users")
print(f"  approach 2 (classifier)        : {int(S_clf.sum()):,} users")
inter = int((S_stat & S_clf).sum())
print(f"  overlap                        : {inter:,} "
      f"({100 * inter / max(int(S_stat.sum()), 1):.1f}% of set 1)")
print(f"  -> low overlap means the two submissions really are different hypotheses")
R["n_stat"], R["n_clf"], R["overlap"] = int(S_stat.sum()), int(S_clf.sum()), inter

hdr("5 -- WRITE THE SUBMISSIONS")
K = 1.10
print(f"  multiplier k = {K} for BOTH, so the two runs differ only in WHO is a gifter.")
print(f"  Why 1.10 and not the implied {np.exp(excess):.2f}: the payoff is asymmetric.")
print(f"    if the effect is real  -> about -0.0006 RMSLE")
print(f"    if it is not           -> about +0.0002 RMSLE")
print(f"  A larger k raises both sides; 1.10 keeps the downside small while capturing most")
print(f"  of the upside, and CV said the model's spiker residual is only +0.012 (k~1.01).")
base = pl.read_csv(ROOT / "subs" / "e0020.csv")
assert np.array_equal(base["user_id"].to_numpy(), p.users)
b = base["predict"].to_numpy()
for nm, S in [("e0021_gifters_stat", S_stat), ("e0022_gifters_clf", S_clf)]:
    q_ = b.copy(); q_[S] *= K
    out = ROOT / "subs" / f"{nm}.csv"
    pl.DataFrame({"user_id": p.users, "predict": q_}).write_csv(out)
    print(f"  wrote subs/{nm}.csv   changed {int(S.sum()):,} users, "
          f"sum {q_.sum():,.0f} (was {b.sum():,.0f})")

(OUT).mkdir(parents=True, exist_ok=True)
(OUT / "gifters_build.json").write_text(json.dumps(R, indent=2, default=str))
print(f"\n  every other user is byte-identical to e0020 (LB 1.6578).")
print(f"  expected move if the effect is real: ~-0.0006; LB 2-sigma noise is 0.0118, so a")
print(f"  single score cannot confirm it -- read the two together as a direction, not proof.")
