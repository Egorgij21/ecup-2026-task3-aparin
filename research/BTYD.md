# BTYD.md — Buy-Till-You-Die as a blend member

Spec for `BACKLOG` Band D item **e0033**, the last untried entry. Written to be built from,
in the style of `CAUSAL_EXP.md`. Strategy lives in `CLAUDE.md`; measured facts in `DATA.md`.

---

## RESULT (2026-08-16) — built, measured, killed. Full write-up: `EXPERIMENTS.md` §1e.

```
e0170 / geo3 / BG-NBD + Gamma-Gamma MLE per anchor, prediction = E[log1p y] by simulation
cv: 1.83569 ± 0.02371 | folds [1.84580, 1.86232, 1.84792, 1.81782, 1.80458]
    Δ vs geo3 = -0.09293 | Δ vs e0049 = +0.07018 | significant: yes
    fitted LOFO blend gain = -0.00006 vs a pre-registered threshold of -0.00050
verdict: kill
```

**Scorecard against this document's own pre-registrations** — five calls, four correct:

| §  | called in advance | outcome |
|---|---|---|
| §0 | ~15% that BTYD clears +0.0005 in a fitted blend | **resolved NO.** −0.00006, off by 8× |
| §0 | it would land in Ridge's quadrant if it failed | **exactly.** +0.0702 / 0.9423 / −0.00006 against Ridge's +0.0726 / 0.9433 / 0.00000 |
| §5.1 | dropout biased toward immortality; report `a, b`, mean `P(alive)` | **confirmed, and worse.** `a` = 0.012–0.020, mean `P(alive)` 0.978–0.983, `P(alive) > 0.99` for 74–85%. `a < 1` ⇒ §3.1's closed form **diverges and is unusable**; only the MC route works |
| §3.2 | check `corr(x, m_x)`; say so if non-zero | **misspecified.** `corr(log x, log m_x)` = +0.23 at every anchor. Note the raw-space check §3.2 literally asks for gives +0.01…+0.03 and **would have passed it** |
| §0/§3.3 | the simulated `E[log1p y]` is "a genuinely new capability" | **half wrong — and this is the finding.** Not new *relative to the family*: L2-on-`log1p` already targets `E[log1p y]`, so every model here had it for free. But as a correction to **BTYD's own** output it is worth **+0.5626** — `log1p(E[y])` scores 2.39829, worse than the optimal constant and worse than `sample_submit`. Largest single effect in the project |

**Two deliberate deviations from the spec below, both needed to make it correct:**

1. **§3.2's Gamma-Gamma weight is internally inconsistent with §2.** §2 defines `m_x` over all
   buy-days (`x+1` of them) while §3.2 writes the credibility weight with the repeat count `x`.
   The weight must use the count `m_x` actually averages. Implemented with `n = x + 1`; the
   literal variant is available as `--gg-count repeat`.
2. **§3.3's simulation sketch is not the BG/NBD process.** `n ~ Poisson(λ·30) · alive` lets a
   live customer buy for the whole horizon, ignoring that they can die *during* it — so it
   overstates `E[X(30)]` and would fail §8's own "MC must reproduce the closed form" check by
   construction. Implemented as the true forward process (Exp(λ) interarrivals, dropout w.p.
   `p ~ Beta(a, b+x)` after each transaction); MC then matches the closed form to 0.01–0.04%.

Deliverables landed as `src/btyd.py`, `src/run_btyd.py`, `src/btyd_blend.py`,
`oof/e0170.parquet`, `reports/e0170_btyd.json`. **No `scripts/btyd.slurm`** — §7 assumed a
cluster job, but all five folds fit in **8.6 minutes on the laptop**, which is the more useful
fact for the resource-efficiency criterion.

---

## 0. Read this before writing code

**The expected value is low and the reason is measured, not guessed.** Ridge already ran the
experiment this is a variant of: it reached log-prediction correlation **0.943** against the
existing family — by far the most decorrelated model this project has produced — and
contributed **exactly nothing** to a leave-one-fold-out blend, because at +0.073 worse it was
too weak. Ten model families now span ρ ∈ [0.7007, 0.7040] and the fitted blend does not move.

> **The requirement is not decorrelation. It is decorrelation AT COMPARABLE QUALITY.**
> `rho_blend = rho * sqrt(2/(1+r))` assumes members of equal quality. A weak member's
> disagreement with the family is mostly its own error, not a different view of the truth.

Pre-registered prior: **~15%** that BTYD clears **+0.0005** in a fitted blend. Log that number
before running, so the result means something either way.

**Why build it anyway.** Two reasons, both specific:

1. `CLAUDE.md` §1 — for cheap ideas, cost of testing is the *only* filter. This is cheap.
2. **§3.3 below is a genuinely new capability**, not just another function class: a generative
   model gives the whole predictive distribution, so `E[log1p(y)]` — the exact Bayes-optimal
   prediction under RMSLE — can be obtained by simulation. Every discriminative model here
   estimates it by fitting it directly. This is the one thing in the backlog that attacks the
   metric from a different direction rather than the features from a different direction.

---

## 1. The idea

BTYD models the *process* rather than the conditional mean. Each user has a latent purchase
rate and a latent dropout propensity; both are fitted by maximum likelihood from four numbers
per user. `DATA.md` describes this task as literally a buy-till-you-die problem.

Two components, fitted independently:

* **BG/NBD** — transaction *timing*: how many purchases in the next 30 days, and `P(alive)`.
* **Gamma-Gamma** — transaction *value*: expected spend per purchase occasion.

---

## 2. Data → RFM

At an anchor day `A`, for each user in the fold population (`>= 1 active day in [A-29, A]`),
using **only** days `<= A`:

| symbol | definition | source |
|---|---|---|
| `x` | number of **repeat** purchase occasions = (buy-days) − 1, floored at 0 | `gmv > 0` per day |
| `t_x` | days from first buy-day to **last** buy-day | |
| `T` | days from first buy-day to `A` | |
| `m_x` | mean GMV per buy-day over the observed buy-days (0 if `x = 0`) | `gmv` |

**A purchase occasion is a buy-day, not an order.** `to_ord` counts items; BG/NBD's Poisson
assumption is about shopping *events*. `DATA.md` §2.1 verifies `gmv > 0 <=> to_ord > 0`
exactly (0 violations in 30.6M rows), so a buy-day is unambiguous.

**`T` is measured from the user's first buy-day, not from 2025-01-01.** Standard BTYD
convention, and it matters here: first activity varies across the panel.

Users with **zero** buy-days have no `t_x` and no `m_x`. They are 12.33% of the panel
(`DATA.md` §3) and BG/NBD assigns them all one identical prediction. Handle them explicitly —
do not let them silently become NaN.

---

## 3. The models

### 3.1 BG/NBD — likelihood and prediction

Parameters `r, alpha` (Gamma prior on purchase rate) and `a, b` (Beta prior on dropout).
Per-user likelihood, with `d = 1{x > 0}`:

```
L = A1 * A2 * (A3 + d * A4)
A1 = gamma(r+x)/gamma(r) * alpha**r
A2 = gamma(a+b)*gamma(b+x) / (gamma(b)*gamma(a+b+x))
A3 = (1/(alpha+T))**(r+x)
A4 = (a/(b+x-1)) * (1/(alpha+t_x))**(r+x)
```

Fit by minimising `-sum(log L)` over the population with `scipy.optimize.minimize`
(L-BFGS-B, all four parameters `> 0`, log-parametrised). Work in log-gamma
(`scipy.special.gammaln`) throughout — the raw products underflow at this sample size.

Expected repeat transactions in the next `t = 30` days, and the survival probability:

```
P(alive | x,t_x,T) = 1 / (1 + d * (a/(b+x-1)) * ((alpha+T)/(alpha+t_x))**(r+x))

E[X(t) | x,t_x,T] =
    ( (a+b+x-1)/(a-1)
      * (1 - ((alpha+T)/(alpha+T+t))**(r+x)
             * hyp2f1(r+x, b+x, a+b+x-1, t/(alpha+T+t))) )
    / (1 + d * (a/(b+x-1)) * ((alpha+T)/(alpha+t_x))**(r+x))
```

`scipy.special.hyp2f1`, and guard `a > 1` (the expression diverges otherwise).

### 3.2 Gamma-Gamma — expected spend per occasion

Parameters `p, q, nu`. Fitted on users with `x >= 1` only. The conditional expectation is a
credibility-weighted average of the population mean and the user's own observed mean:

```
w = (q - 1) / (p*x + q - 1)
E[M | x, m_x] = w * (p*nu/(q - 1))  +  (1 - w) * m_x
```

**Check the model's own assumption before trusting it:** Gamma-Gamma requires frequency and
monetary value to be independent. Compute `corr(x, m_x)` on the fitted population and report
it. If it is materially non-zero the model is misspecified — say so in the results rather than
shipping it silently.

### 3.3 Turning it into an RMSLE prediction — the part that matters

The naive output is `E[y] = E[X(30)] * E[M]`. **That is the wrong functional.** RMSLE is
minimised by `E[log1p(y)]`, and `log1p(E[y]) != E[log1p(y)]` — the same error
`PAPERS_FEATURES_AND_IDEAS.md` §6 flags for simulation averaging and `src/blend.py` avoids by
averaging in log space.

Because the model is generative, the correct quantity is directly available:

```
for s in 1..S:                      # S = 200 is plenty
    lambda_i ~ Gamma(r + x, alpha + T)          # posterior purchase rate
    alive_i  ~ Bernoulli(P(alive | x,t_x,T))
    n_i      ~ Poisson(lambda_i * 30) * alive_i
    m_i      ~ posterior spend per occasion (Gamma-Gamma posterior, or fix at E[M|x,m_x])
    y_s      = n_i * m_i
prediction = mean over s of log1p(y_s)          # then expm1 for the submission scale
```

**Produce both** — `log1p(E[y])` and `E[log1p(y)]` — and report both. If they differ
materially, that difference is the one advantage a generative model has here, and it should be
measured rather than assumed.

---

## 4. Protocol

Frozen and non-negotiable (`CLAUDE.md` rules 3 and 4):

* folds from `data/folds.parquet`; 5 anchors 2025-06-18 … 2025-10-16
* population at anchor `A`: `>= 1 active day in [A-29, A]`
* target: `sum(gmv)` over `[A+1, A+30]`
* metric from `src/metrics.py`; the `geo3` naive reference recomputed on the same population
* OOF written to `oof/e0170.parquet` in `run.py`'s schema — **this is what makes it blendable**,
  and it is the omission that cost two re-runs this week

Fit the parameters **per anchor**, on that anchor's population, from days `<= A` only. Four
parameters over ~200k users is seconds; there is no reason to share a fit across anchors.

---

## 5. Hazards specific to this dataset

1. **The panel is conditioned on end-of-window activity.** All 250k users are active in each
   30-day block ending 2026-02-13 (`DATA.md` §4). **BG/NBD's dropout parameter will be biased
   toward immortality**, because churned users were excluded by construction. This is the
   single biggest threat to the model being meaningful — report the fitted `a, b` and
   `mean P(alive)`; if `P(alive)` is ~1 for nearly everyone, the dropout half is dead and the
   model has collapsed to an NBD.
2. **Selection also truncates `T`.** No user is absent for the final 30 days, so recency is
   capped at 29 days at the test anchor.
3. **12.33% of users never purchased** — one identical prediction for all of them.
4. **GMV is rescaled/anonymised** (`DATA.md` §2.2, all 4.7M positive values distinct). Fine for
   Gamma-Gamma, but do not round or clip to currency conventions.
5. **Do not build a two-part gate.** `BACKLOG` e0033 is explicit and e0010 measured it: the
   hurdle is dead (−0.00012, 1.3σ). `P(alive)` may enter as a blend member or a column, never
   as a multiplicative gate on another model.

---

## 6. Evaluation and the pre-registered decision rule

Report on the frozen folds: `cv_mean ± std`, per-fold, Δ vs `geo3`, and

* **ρ against truth** in log space, and
* **`corr(log BTYD, log e0049)` and `corr(log BTYD, log e0101)`** — the decision numbers.

Thresholds, from this project's own history:

```
e0049 <-> e0064   0.9983  -> blend gain ~0        (twins)
gbdt  <-> e0101   0.9951  -> +0.00048 rho         (paid)
usercv_ridge      0.9433  -> 0.00000              (decorrelated but too weak)
```

> **Decision rule, fixed in advance.** BTYD is kept only if a leave-one-fold-out fitted blend
> including it beats the same blend without it by **> 0.0005** on the frozen folds. Correlation
> alone is not sufficient evidence — Ridge already proved that. If it lands in Ridge's quadrant
> (decorrelated, much weaker, zero blend gain), record it in the graveyard and stop.

---

## 7. Deliverables

```
src/btyd.py        RFM extraction, BG/NBD + Gamma-Gamma fit, both prediction functionals
src/run_btyd.py    frozen-fold CV -> cv row + oof/e0170.parquet   (mirrors src/run.py)
scripts/btyd.slurm compute partition, CPU only -- this is scipy, not a GPU job (CLUSTER.md 5b)
```

---

## 8. Asserts, not hope

* **Look-ahead:** zero the panel after day `A`, recompute the RFM summary, require it unchanged.
  This is the same guard `assert_no_lookahead` / `assert_causal_features` apply elsewhere, and
  it caught a real bug in `tenure` (a global `argmax` reading the whole row).
* `0 <= t_x <= T` for every user; `x >= 0`; `m_x > 0` wherever `x >= 1`.
* `E[X(t)]` matches a Monte-Carlo simulation of the fitted process for several parameter sets
  (this is the only check that the `hyp2f1` expression is transcribed correctly).
* Fitted `r, alpha, a, b, p, q, nu` all `> 0`, and `a > 1`.
* Predictions finite and `>= 0`; `E[log1p(y)]` and `log1p(E[y])` both produced.
* Fold populations match `data/folds.parquet` exactly (`array_equal` on `user_id`).
