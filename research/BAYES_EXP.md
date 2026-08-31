# Experiment: Hierarchical Bayesian BTYD for 30-day forward GMV

> ## STATUS 2026-08-16 — reviewed against the record. Read this before building anything.
>
> Full write-up: `EXPERIMENTS.md` §1f (the frontier) and §1g (this review).
>
> **B0 is already built and killed — it is e0170.** Same model, same order-day counting, same
> cohort split. CV **1.83569**, +0.0702 behind e0049, fitted blend gain **−0.00006**.
> See `BTYD.md` and `EXPERIMENTS.md` §1e. §5.2's Gauss–Laguerre readout would replace our
> 1000-draw MC, which is measured at 0.0009 above the noise-free limit: worth having, far too
> small to change a verdict.
>
> **§10's uncertainty columns — the one genuinely new claim — measure +0.00001 (e0172).**
> `sd_log1p` is a real signal (`corr` +0.3187 with the blend's absolute residual) and §10 is
> right that a GBDT cannot build it. The *reason* given is wrong: **under squared error the
> Bayes action is the posterior mean, so the posterior sd cannot move the optimal point
> prediction** — the shrinkage §10 invokes is already inside `E[log1p Y]`. These columns pay
> for pinball / CRPS / expectiles, never for RMSLE.
>
> **§5.3's calibration wrapper: −0.00001 (e0173), and its own blocking rule never fires**
> (`a_c` = 0.9932…1.0005 on all five folds). That is `EXPERIMENTS.md` §2's `k*=1.000`; §1b
> already made the test-anchor shift standing protocol.
>
> ### ⚠ Two sections must not be built as written
>
> * **§6 (folds).** F2/F3/F4 target windows lie **inside** the guaranteed-activity zone
>   `[2025-11-16, 2026-02-13]`. `DATA.md` §4.3 prices that bias at **+0.041** — 80× the effect
>   being chased — and §6 nominates **F4 as the primary validation fold**. Only F1 is clean,
>   and it is already frozen fold 4. Rule 3: use `data/folds.parquet` unchanged.
> * **§3.2 (`s_next` from the year-earlier block).** Already falsified twice: **e0142** did
>   exactly this via day-of-year features and scored **1.6785, our worst since e0001**
>   (−0.410 log-mean shift); and `BACKLOG` Band A+ retracted the year-lag mechanism on CV
>   (best multiplier k = **1.00**, every k > 1 worse). §3.2 calls it "the single most
>   falsifiable claim in the plan" — correct instinct, already falsified.
>
> ### Right, and worth keeping
>
> §1.3 order-days not orders (matches e0170) · §1.4 explicit cohorts (15–21% never-buyers at
> the frozen anchors) · §8.1 the transductive refit — features, never targets, so legitimate
> and worth saying out loud · §7's `P(alive)` reliability curve as jury material · §11–§13's
> asserts, sanity checks and regression tests.
>
> ### B1 IS NOW BUILT AND MEASURED (2026-08-17) — `e0180`, full write-up in EXPERIMENTS.md §1h
>
> ```
> e0180 / e0170 / B1 hier_cov, 28 covariates on the rate/dropout/spend priors
> raw cv 1.87825   CV* (optimally calibrated) 1.81822   rho 0.63452
>     vs B0:  CV* -0.01420   rho +0.00743 (5/5 folds)   <- the covariates WORK
>     r vs family 0.94266 -> 0.95638, so rho_partial 0.01017 -> 0.00400
>     blend gain +0.00000
> verdict: kill.  Pre-registered prior ~10%; resolved NO, for exactly the stated reason.
> ```
>
> **A better model and a worse blend member.** Judge it on rho / CV\*, never raw RMSLE: raw got
> 0.043 *worse* purely through level and spread, which §1b establishes is free to fix.
>
> **The mechanism, in one regression:**
> `log B1 = -0.715 + 0.560*log B0 + 0.642*log family`, **R² = 0.938**. B1 is a 53/47 mixture of
> B0 and the models we already have — the covariates did not add a direction, they rotated B0
> toward the family.
>
> **§4.2's variance floor is mandatory, and I was wrong to drop it.** Unconstrained, `r` climbs
> 114 → 144 and is still rising at 2000 iterations on every fold: the rate prior collapses to a
> point mass and the per-user random effect is optimised out of existence. I argued the floor
> was unnecessary because marginalising the latents leaves "no free sigma to collapse" — wrong,
> `1/r` and `1/(a+b)` play sigma's role.
>
> **`e0183` — the floored variant: fixes the fit, changes nothing.** `r` 114 → 0.53, `a+b`
> 389 → 2.0, `p_gg` 162 → 1.11 (a 200× difference), and the predictions correlate at
> **0.98497** — rho 0.63435 vs 0.63452, `r` vs family 0.95641 vs 0.95638, blend +0.00000 both.
> So the collapse was a real pathology and **not** the cause of the blend failure.
> **Once 28 covariates sit on the locations, the generative process is only a link function**;
> the covariate regression decides the prediction. 19.4 min for five folds — use this variant
> if B2 is ever attempted.
>
> **Two bugs worth carrying into any future NumPyro work:** a JAX where-trick NaN gradient that
> left the objective finite so L-BFGS-B stopped at iteration 1 reporting `success=True` (check
> gradients, not just objectives); and covariates reaching |z| = 249, which §3.1's
> `Normal(0, 0.5)` prior would not have saved — the problem is the design matrix.
>
> ### B1/B2: the bar, pre-registered
>
> `EXPERIMENTS.md` §1f derives the exact admissibility condition — a candidate's entire blend
> value is `rho_partial = corr(L, B | M)`, since `R² = rho_M² + (1−rho_M²)·rho_partial²`.
>
> ```
> B1 must reach   rho_partial >= 0.02383     (= -0.0005)
> best ever here            e0064  0.01269   (53% of the bar)
> B0 as built               e0170  0.01017   (43%)
> ```
>
> **The one live argument for building B1**: B0 already carries `e = +0.00254`, the **largest
> excess correlation on record here, 3.4× e0049's**. Its entire problem is the `1/(1−r²)`
> divisor at r = 0.943. If B1 preserves that excess while raising quality to r ≈ 0.99, it
> clears the bar. Excess usually collapses as models converge, so: **pre-registered prior
> ~10%**, below `BTYD.md`'s 15% for B0. Logged before any decision to build.
>
> And note what §1f found about the bar itself: **no single member has ever cleared 0.0005,
> ours included** (e0049 is worth −0.00007). Judge B1 against `rho_partial ≈ 0.013`, the best
> ever achieved, not against a threshold nothing has met.


Generative latent-variable model of the purchase process. One sample = one user's
*sufficient statistics* at a cutoff date. The model does not emit a number; it emits a
**posterior predictive distribution** over GMV in the next `horizon` days, and the number
submitted is a functional of that distribution chosen to match the metric.

Implement as a runnable pipeline. Three model variants (B0/B1/B2) share everything else —
cutoff construction, sufficient statistics, covariates, splits, readout, evaluation,
inference — so build those once and switch models by config.

The point of contrast with `CAUSAL_EXP.md`: the GRU learns the map history → number. This
learns history → *parameters of a process*, and the number falls out. That buys three
things the GRU cannot give you: the RMSLE-optimal point estimate computed rather than
approximated, a calibrated `E[Y]` for the aggregate tie-breaker from the same fit, and
predictive intervals. It costs you flexibility — the process is assumed, not learned.

---

## 1. Data

Same raw table as `CAUSAL_EXP.md` §1: one row per `(user_id, event_date)` where the user
was active, days with no activity absent.

### 1.1 No dense panel

**Never materialise `(n_users, T, n_raw)`.** The whole reason this model family is cheap is
that it compresses each user's history to a handful of numbers. The dense panel is 250k ×
409 × 18 float32 ≈ 7.4 GB and buys nothing here. Work on the sparse table with groupbys.

If a helper needs a per-day series (block counts, gap distributions), build it as a sparse
`(n_users, K)` block matrix with `K ≈ 14`, not a `(n_users, 409)` daily one.

### 1.2 Cutoff and blocks

A **cutoff** `C` splits every user's history into a feature window `[start_date, C]` and a
target window `[C+1, C+horizon]`. `horizon = 30`.

Blocks are laid out **backwards from the cutoff** so the last block is exactly the 30 days
immediately before `C` and the first block absorbs the remainder:

```
block_index(d) = K - 1 - floor((C - d).days / 30)      # d <= C, clipped at 0
K = ceil((C - start_date + 1).days / 30)
L_k = number of calendar days in block k               # L_0 may be < 30
```

Backward alignment matters: forward alignment from 2025-01-01 puts an arbitrary partial
block at the recency end, which is precisely where the model is most sensitive.

### 1.3 Sufficient statistics

Per user, at cutoff `C`:

| symbol | column | meaning |
|---|---|---|
| `n_k` | `orders_block_k` | order-days in block *k* (`to_ord > 0`), `(n_users, K)` |
| `g_k` | `gmv_block_k` | total GMV in block *k*, `(n_users, K)` |
| `x` | `n_order_days` | total order-days in the feature window |
| `t_x` | `recency_days` | days from first order to last order |
| `T_u` | `tenure_days` | days from first *active* day to `C` |
| `m̄` | `mean_order_value` | `total_gmv / x`, undefined when `x = 0` |
| `k_last` | `last_active_block` | highest *k* with `n_k > 0`, `-1` if never |

Collapse to **order-days, not orders**: two orders on the same day are one transaction
event. The classic BTYD counting process is continuous-time with rate λ; daily
aggregation already discretised it, and treating a 3-item day as 3 events inflates λ for
bulk buyers without any corresponding change in their inter-purchase timing. Keep the item
count as a covariate instead.

`m̄` is the average value *per order-day*, so `E[GMV] = E[N] · E[m̄]` stays consistent.

### 1.4 The zero-purchase cohort

A large fraction of 250k Search/Catalog users will have **never purchased** in 409 days.
Classic BTYD is undefined for them: `t_x`, `m̄` and the recency-based `P(alive)` all
require a first transaction.

Do not drop them and do not predict 0 for all of them. Split the population explicitly:

- **`cohort = "buyer"`** — `x ≥ 1`. Full model.
- **`cohort = "browser"`** — `x = 0`, but `active_days ≥ 1`. No per-user frequency latent
  is identifiable; the covariate regression and the population prior do all the work. In
  the hierarchical variants this happens automatically — the per-user random effect
  collapses to its prior — which is exactly the behaviour you want and is a good reason to
  prefer B1/B2 over B0.
- **`cohort = "dormant"`** — no activity at all in the feature window (possible at earlier
  cutoffs). Predict from the population prior conditioned on covariates only.

Log the three cohort sizes at every cutoff. Report metrics sliced by cohort — a variant
that only helps buyers is a different result from one that helps everywhere.

---

## 2. Target

```
y[u] = sum(gmv[u, C+1 : C+1+horizon])          # total, not mean
```

**Total, not mean daily.** `CAUSAL_EXP.md` predicts mean daily GMV because a sequence model
emits one value per timestep and the horizon has to be normalised out. Here the target is
the competition target directly, and the generative model produces a total by construction
(a sum over a random number of orders). Do not divide by 30 and multiply back — the `log1p`
makes those two scales non-equivalent in loss space, and the competition scores the total.

Do **not** apply `log1p` to the target. The model has a likelihood; the target enters it on
its natural scale. `log1p` appears in exactly one place, the readout (§5), and in the
metric (§7). This is the structural difference from a regression pipeline and the reason
retransformation bias is a computed quantity here rather than a tuned one.

### 2.1 Which users are scored

Every user with at least one row anywhere in `[start_date, C]`. No burn-in, no horizon-tail
mask — those exist in the GRU spec because it scores every timestep. Here there is exactly
one prediction per user per cutoff.

Users who first appear *after* `C` are excluded from that cutoff's fold entirely (they have
no feature window). At the final cutoff this cannot happen — all 250k are present by
2026-02-13 by construction — but at earlier cutoffs it does, and silently including them as
zero-history users biases the population prior toward inactivity. Assert on it.

---

## 3. Model

### 3.1 Generative process

For user *u*, with per-user latents λ_u (order-day rate, per day), θ_u (per-block churn
hazard), μ_u (mean value per order-day), and a shared spend dispersion φ:

```
alive in block k        d_u ~ Geometric(θ_u)      # dies at the start of block d_u
n_k | alive             ~ Poisson(λ_u · L_k · s_k)
n_k | dead              = 0
g_k | n_k               ~ Gamma(n_k · φ, φ / μ_u)      # 0 when n_k = 0
```

`s_k` is a shared calendar multiplier (B2 only; `s_k ≡ 1` in B0/B1).

The death time is an absorbing state, so marginalise it exactly — no latent discrete
variable in the guide, no enumeration overhead beyond a `logsumexp` over `K - k_last`
terms:

```python
# log P(data_u | lam, theta, mu) — vectorised over users, K+1 death hypotheses
ll_pois   = poisson_logpmf(n, lam[:, None] * L * s)          # (U, K)
cum_ll    = cumsum(ll_pois, axis=1)                          # alive through block k
log_die_d = log(theta) + (arange(K) * log1p(-theta))         # die at start of block d
log_alive = K * log1p(-theta)                                # never died
# a death hypothesis d is only admissible if d > k_last
branches  = where(arange(K) > k_last, cum_ll_shifted + log_die_d, -inf)
loglik    = logsumexp(concat([branches, cum_ll[:, -1:] + log_alive]), axis=1)
```

The posterior weight on the final branch is `P(alive at C | data)` — read it off directly,
do not estimate it by sampling.

Forecast window = one block of length `horizon`:

```
P(active next block) = P(alive at C) · (1 - θ_u)
N ~ Poisson(λ_u · horizon · s_next)        conditional on active, else N = 0
Y = Σ_{i=1..N} M_i,     M_i ~ Gamma(φ, φ/μ_u)
```

`Y | N=n` is `Gamma(nφ, φ/μ_u)`, which is what makes §5.2 cheap.

### 3.2 Three variants

**Experiment B0 — `btyd_classic`.** Closed-form empirical Bayes, no covariates. The
floor, and the resource-efficiency exhibit.
- BG/NBD on `(x, t_x, T_u)`: λ ~ Gamma(r, α), per-transaction death prob p ~ Beta(a, b).
- Gamma-Gamma on `(x, m̄)`: value per order-day ~ Gamma(φ, φ/μ), μ ~ Gamma(q, γ).
- 4 + 3 population hyperparameters fitted by MLE over all users (`scipy.optimize`,
  L-BFGS-B on the log-likelihood, or `pymc-marketing`'s CLV module).
- Per-user posteriors then closed-form. `E[M | x, m̄]` is the shrinkage estimator
  `w · m̄ + (1 - w) · γφ/(q-1)` with `w = φx / (φx + q - 1)` — a user with one order is
  pulled hard to the population mean, a user with fifty is not.
- Buyers only. Browsers and dormants get the population prior.
- Fit time target: **under 2 minutes, CPU, 250k users.**

**Experiment B1 — `hier_cov`.** The §3.1 model, fitted by SVI, with covariates on all
three latents. This is the variant that should actually be competitive.
- Non-centred parameterisation, per-user random effects:
  ```
  log λ_u    = μ_λ + Xᵤ·β_λ + σ_λ·z_λ,u
  logit θ_u  = μ_θ + Xᵤ·β_θ + σ_θ·z_θ,u
  log μ_u    = μ_m + Xᵤ·β_m + σ_m·z_m,u
  ```
- Priors: `μ_• ~ Normal(0, 2)`, `β_• ~ Normal(0, 0.5)` on standardised covariates,
  `σ_• ~ HalfNormal(1)`, `φ ~ Gamma(2, 0.1)`. Validate with a prior predictive check (§4.4)
  before fitting anything.
- All three cohorts in one fit. For a browser, `n_k ≡ 0`, the likelihood carries no
  information about `z_λ,u`, the guide returns the prior, and the prediction is driven by
  `Xᵤ·β_λ`. That is the correct answer and it is free.
- NumPyro + `AutoNormal` (or `AutoLowRankNormal`, rank 10) + minibatch over the user plate.

**Experiment B2 — `hier_seasonal`.** B1 plus the blocks that should matter for *this*
competition specifically. Add as named blocks so each can be ablated.
- **Calendar multipliers `s_k`.** Shared across users, `log s_k ~ Normal(0, 0.3)`, one per
  block, soft-constrained to mean zero in log space for identifiability against `μ_λ`.
  **Do not build a GP for this.** The Ariel solution replaced its 2024 GP drift model with
  plain cubic polynomials in 2025 and moved from 2nd to 1st — the flexible nuisance model was
  over-engineering, and the win came from getting the *structural* part (real transit physics)
  right instead. Block multipliers are our cubic polynomial. Spend the complexity budget on
  the churn and spend process.
- **Seasonal extrapolation for the forecast block.** The forecast window is
  2026-02-14 → 2026-03-15. **The same calendar window one year earlier is inside the
  training data** (2025-02-14 → 2025-03-15). It contains 23 February and 8 March, both
  major gifting peaks in the Russian market — the forecast window is emphatically not a
  neutral month, and a model with `s_next = 1` will under-predict it. Estimate `s_next`
  from the year-earlier block, with a `Normal(log s_2025, 0.2)` prior rather than pinning it
  hard. Ablate this: it is the single most falsifiable claim in the plan.
- **Negative-binomial counts.** Replace `Poisson(λ L s)` with `NegBinomial(λ L s, κ)` to
  absorb within-block burstiness. Poisson will be over-confident about `N`, which matters
  because `Var[N]` feeds straight into the left tail of `Y` and therefore into
  `E[log1p Y]`.
- **Tenure-varying hazard.** Constant `θ` is a strong claim. Allow
  `logit θ_u,k = logit θ_u + δ · log1p(k)`, `δ ~ Normal(0, 0.3)`. Costs one parameter,
  tests whether churn risk decays with tenure.

### 3.3 Covariates `X`

Standardised at the cutoff, train-users-only statistics (§6). Keep `d ≈ 20–40`. Piling in
300 engineered features turns this into a badly-regularised GLM and throws away the reason
you built a generative model. If you want 300 features, use the GBDT — and see §10.

Reuse the definitions from `CAUSAL_EXP.md` §3.1 variant C, evaluated at `t = C`:

- recency: `log1p(days_since_active)`, `log1p(days_since_order)`, `log1p(days_since_cart)`
- tenure and intensity: `log1p(T_u)`, `log1p(active_days)`, `active_days / T_u`
- funnel: `to_ord / (to_cart + eps)`, `to_cart / (searches + eps)`, `log1p(searches_30d)`
- channel mix: `gmv_search / (gmv + eps)`, `search_to_ord / (to_ord + eps)`
- dispersion: CV of inter-active gaps, `rolling_std(gmv, 90) / (rolling_mean(gmv, 90) + eps)`
- trend: `gmv_30d / (gmv_90d/3 + eps)`, `active_days_30d / (active_days_90d/3 + eps)`
- one-hot cohort (buyer / browser / dormant)

**Causality rule is identical to the GRU spec**: no covariate may read a day `> C`. Same
assert (§11).

**Report `σ_u` before and after adding covariates.** Adding structure must remove variance
from whatever that structure replaces — the Ariel PCA block explicitly rescales the residual
GP prior by `sqrt(1 - ratio)` after the components absorb part of the signal, precisely to
stop the two components double-counting. Here the equivalent is: `σ_λ`, `σ_θ`, `σ_m` must be
re-estimated when covariates enter, never carried over from B0, and they should visibly
shrink. If `σ_u` is unchanged after adding 30 covariates, the covariates explain nothing and
B1 is B0 with extra steps — report that plainly rather than shipping it.

---

## 4. Inference

### 4.1 B0 — empirical Bayes

MLE on the population hyperparameters, closed-form per-user posteriors. Parameterise in log
space (`log r, log α, log a, log b`) and bound the optimiser. The BG/NBD likelihood needs
the Gaussian hypergeometric `₂F₁`; use the standard series with a convergence guard and
**assert no NaNs before reporting a fit** — silent `₂F₁` overflow producing a plausible-
looking but wrong `α` is the classic failure of this model.

### 4.2 B1/B2 — SVI

- **Initialise from B0.** The closed-form posterior means from `btyd_classic` are the
  starting point for the guide, exactly as the Ariel solver uses a cheap `SimpleModel` grid
  search to find the basin before running the expensive inference. Free, and it should cut
  the step count materially. Assert the initialised ELBO is finite and better than a random
  init — if it is not, the two models disagree about parameterisation and something is wrong.
- NumPyro, `Trace_ELBO`, `Adam(lr=1e-2)` with a linear warmup, 20k steps.
- Minibatch the user plate at 4096; the block matrix `(4096, K)` is trivial.
- **Floor the scale parameters**: `σ_λ, σ_θ, σ_m ≥ 0.15`, enforced after fitting, with a
  logged warning whenever the floor binds. Variance components collapse to zero under
  maximum likelihood far more often than they should — the Ariel model carries an explicit
  `min_transit_scaling_factor = 0.2` for the same reason. A collapsed `σ` means total
  pooling: every user gets the population mean, the fit looks stable, and the score is bad.
- `AutoNormal` first. Move to `AutoLowRankNormal(rank=10)` only if posterior correlations
  between `μ_λ` and `μ_θ` visibly distort the alive probabilities — they are partially
  non-identifiable (a low rate and a high survival look like a high rate and a low
  survival), which is the known weak spot of this model family.
- 3 seeds, report ELBO spread. An SVI run that lands on a different ELBO plateau per seed
  is not converged, whatever the metric says.
- **Sanity fit with NUTS on a 5k-user subsample.** Not for production — as a check that the
  SVI posterior means are in the right place. If SVI and NUTS disagree on `μ_λ` by more
  than a posterior sd, the guide is wrong, not the sampler.

### 4.3 Diagnostics

- ELBO curve per seed.
- Posterior sd of each population parameter; anything with sd ≈ prior sd was not learned.
- Shrinkage plot: posterior mean λ_u against `x/T_u`, coloured by `x`. Should hug the
  identity line for high-`x` users and flatten toward the population mean for low-`x` users.
  If it is a straight line everywhere, `σ_λ` has run away and there is no pooling.

### 4.4 Prior predictive check

Before the first real fit: sample parameters from the priors, simulate `Y` for 250k
synthetic users, and plot `log1p(Y)` against the empirical `log1p(y)` at a held-out cutoff.
The prior should cover the empirical distribution generously without putting mass on
absurdities (a prior that generates 10⁸-rouble 30-day baskets for 5% of users is
mis-specified regardless of how well it fits afterwards). Tighten `σ_•` if it does.

---

## 5. Readout — from posterior to a submitted number

This section is the whole argument for the approach. Do not shortcut it.

### 5.1 Three point estimates, three purposes

| output | formula | scored by |
|---|---|---|
| `pred_rmsle` | `expm1(E[log1p Y])` | competition metric |
| `pred_mean` | `E[Y] = P(active) · λ·horizon·s_next · μ` | aggregate GMV, RMSPE tie-breaker |
| `rank_score` | `E[Y]`, or `P(alive)` for a pure engagement ranking | Gini tie-breaker |

`pred_rmsle` is the Bayes-optimal action under squared error in log space; `pred_mean` is
the Bayes-optimal action under squared error on the raw scale. They differ by a lot on
this target — for a user with `P(active) = 0.3` the first is dragged toward zero by the 70%
of predictive mass sitting exactly at zero, and the second is not. **Submit `pred_rmsle`,
report `pred_mean` in the write-up, ship both columns.**

### 5.2 Rao-Blackwellised `E[log1p Y]`

Do not estimate this by drawing `Y` samples and averaging `log1p`. Condition on `N`:

```
E[log1p Y] = Σ_{n≥1} P(N=n) · g(nφ, φ/μ),      g(a,b) = E[log1p(Gamma(a,b))]
```

`P(N=0)` contributes exactly 0 and already contains the churn mass. So:

- Truncate `n` at `n_max` = the 1−1e-8 quantile of the count distribution (per user; in
  practice `n_max ≤ 64` covers everyone, assert it).
- Compute `g(a, b)` by **64-node Gauss–Laguerre quadrature**, vectorised over the
  `(n_users, n_max)` grid. Cache the nodes.
- Marginalise over the parameter posterior with `S = 200` posterior draws of
  `(λ, θ, μ, φ)` per user; the inner sum is deterministic given a draw.

The Monte Carlo noise in a naive sample-average estimator is `O(1/√S)` on a quantity whose
differences between competing models are in the third decimal of RMSLE. Quadrature removes
that noise for roughly the same cost. If you skip this, your A/B comparisons measure your
random seed.

Cross-check once, not every run: for 1000 random users, assert the quadrature result and a
10⁶-sample MC estimate agree to 1e-3.

### 5.3 Calibration — a wrapper model, fitted on the metric

Even a well-specified model is not automatically calibrated for this metric under a
misspecified likelihood. Build calibration as a **first-class wrapper** around the model,
with its parameters fitted by direct numerical minimisation of RMSLE on OOF predictions —
not by least squares, not by a proxy loss. This is the pattern from the Ariel solutions
(`SigmaFudger` / `MeanBiasFitter` in 2024, `Fudger` in 2025), where the correction layer is
optimised against the competition score itself with `scipy.optimize`.

```
log1p(ŷ_final) = a_c · log1p(ŷ_rmsle) + b_c + γ · s_u
```

- `c` indexes **cohort** (buyer / browser / dormant) — the bias is not the same shape in
  each, and one global scalar averages three different problems together.
- `s_u` is the **posterior sd of `log1p Y`** for that user. This term is the equivalent of
  Cottaar's `adjust_based_on_u`: it lets the correction differ for users the model is
  uncertain about, which is exactly where retransformation bias concentrates. A GBDT cannot
  build this feature; we get it for free.
- Seven parameters, fitted on 200k+ OOF points by one `scipy.optimize.minimize` call
  against a vectorised RMSLE. No meaningful overfitting risk.

**Write the vectorised metric first.** It goes inside an optimiser loop, so a slow scorer
makes this whole section unusable — see `score_metric_fast` in the Ariel repo for the shape.

**Blocking rule.** If any `a_c` falls outside `[0.9, 1.1]`, **stop and find the missing
model component before submitting.** A calibration constant that will not sit near 1 is a
bug report about the prior, not a nuisance to be absorbed by the fit. Precedent: the 2024
Ariel solution carried an unexplained 1.0064 multiplier on the mean that was critical to the
score; it turned out to be a missed constant background signal, and fixing the model made
both fudge factors unnecessary. Log `a_c`, `b_c`, `γ` in `results.md` every run.

Isotonic regression instead of affine is tempting and usually wins ~0.002 on validation and
gives it back on the private LB. If you try it, fit it on one cutoff and evaluate on
another, never in-fold.

Calibrate `pred_mean` **separately and multiplicatively**: one scalar `c` chosen so
`Σ pred_mean · c = Σ y` on OOF. Do not reuse the RMSLE parameters — different scale,
different loss.

Final: `clip(ŷ, 0, None)`. Consider an upper clip at the 99.99th percentile of observed
30-day GMV; test it, don't assume it.

---

## 6. Splitting and validation

Two axes. Use both — this is where the design differs most from `CAUSAL_EXP.md` §4, which
deliberately took the user axis only and flagged the consequence.

**User axis.** Identical hash split, identical salt, so folds line up with the GRU runs and
the two families are comparable model-for-model:

```python
h = int(hashlib.md5(f"{salt}:{user_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
val = h < val_frac       # val_frac = 0.2, salt = "gmv-v1"
```

Hyperparameters and the §5.3 calibration are fitted on train users, evaluated on val users.

**Time axis — rolling origins.** Four cutoffs, each with a fully observed 30-day target:

| fold | cutoff `C` | target window | note |
|---|---|---|---|
| F1 | 2025-10-16 | 2025-10-17 → 2025-11-15 | neutral |
| F2 | 2025-11-15 | 2025-11-16 → 2025-12-15 | Black Friday / pre-NY ramp |
| F3 | 2025-12-15 | 2025-12-16 → 2026-01-14 | New Year peak then collapse |
| F4 | 2026-01-14 | 2026-01-15 → 2026-02-13 | most recent, closest to submission |
| — | 2026-02-13 | 2026-02-14 → 2026-03-15 | submission, target unobserved |

F3 is a stress test, not a representative fold — do not tune on it. **F4 is the primary
validation fold**: it is the only one whose feature window ends in the same regime the
submission fit will. Weight it accordingly and say so when reporting.

Refitting the whole model per fold is the point, not a cost — B0 is minutes, B1/B2 are
tens of minutes. The GRU spec could not afford this. Use the advantage.

Report **RMSLE mean ± range across F1/F2/F4** as the headline, F3 separately.

---

## 7. Evaluation

On val users, per fold, per cohort:

- **RMSLE** (primary).
- MAE in log space.
- **Aggregate bias** `Σ pred / Σ actual − 1`, for both `pred_rmsle` (will be very negative,
  by design) and `pred_mean` (should be near 0 after §5.3). Reporting only the first is
  what makes teams discover the RMSPE tie-breaker on results day.
- **Gini** over `rank_score` against actual `y`.
- **RMSPE on total GMV**, i.e. the tie-breaker as specified.
- Metrics sliced by activity decile and by cohort.
- **Predictive calibration** — the diagnostics only this family can produce:
  - PIT histogram of `y` under the predictive CDF (randomised PIT, since `Y` has an atom at
    0). Uniform is the goal; a U-shape means over-confidence, a hump means the opposite.
  - Empirical coverage of the 50% / 80% / 95% predictive intervals.
  - Mean CRPS in log space.
- **`P(alive)` reliability curve**: bucket users by predicted `P(alive)`, plot against the
  fraction that actually ordered in the target window. This is the single most
  interpretable plot in the whole project and belongs in the jury pitch.

Baseline rows, all reported, none skipped:

| baseline | definition |
|---|---|
| zero | `0` for everyone |
| persistence | `gmv` in the 30 days before `C` |
| log-persistence | `expm1(mean over users' last 3 blocks of log1p(gmv_block))` |
| shrunk rate | `x/T_u · horizon · m̄`, no churn term |
| GBDT | the tabular model on `log1p`, if it exists |

If a variant does not beat persistence, say so plainly. If B1 does not beat B0, the
covariates are not carrying their weight and that is worth more than another seed.

---

## 8. Test inference

1. Refit at `C = 2026-02-13` on **all** users, train and val, with the hyperparameters and
   the §5.3 calibration constants frozen from the F4 fold. Do not refit the calibration on
   data whose target you cannot see.

   Fitting the population hyperparameters on all 250k users at the submission cutoff is
   **legitimate and not leakage**: the test users *are* the train users, and their feature
   windows are fully observable. This is the same move as the 2024 Ariel solution refitting
   its PCA basis on the test set — a deliberate transductive step, taken because the test
   inputs are available and the test *labels* are not. Take it on purpose and say so in the
   write-up; it is a free win and a defensible one.
2. `s_next` for the 2026-02-14 → 2026-03-15 block: from the year-earlier estimate (§3.2),
   with the prior as specified. Log the posterior mean of `s_next`, it is a number you
   should be able to defend out loud.
3. Readout per §5.2, then §5.3, then clip.
4. Write:

```
predictions/{exp_name}.csv:
  user_id, predict, pred_mean, p_alive, e_orders, e_aov, pred_p10, pred_p50, pred_p90
```

`predict = pred_rmsle` post-calibration — that is the submission column. The rest are for
the write-up, the tie-breaker checks, and §10.

5. Also dump the full predictive distribution for 20 random users (`--dump-traces 20`) as a
   sanity check: a user with heavy recent activity should show a right-shifted, wide
   distribution; a user last seen in March 2025 should be a near-point-mass at zero. If
   they look the same, `P(alive)` is not doing anything.

---

## 9. Deliverables

```
cutoff.py      cutoff/block construction, sufficient statistics, cohorts
covariates.py  X at a cutoff + train-only standardisation, COVARIATE_SETS
model_btyd.py  B0: BG/NBD + Gamma-Gamma, MLE, closed-form posteriors
model_hier.py  B1/B2: NumPyro model + guide, death-time marginalisation
infer.py       SVI / MLE driver, seeds, diagnostics, checkpointing
readout.py     Gauss-Laguerre g(a,b), E[log1p Y], E[Y], quantiles, calibration
evaluate.py    all of §7, including PIT and the P(alive) reliability curve
predict.py     §8
results.md     appended after every run
```

CLI:

```bash
python infer.py --exp btyd_classic  --cutoff 2026-01-14 --data data/events.parquet
python infer.py --exp hier_cov      --cutoff 2026-01-14 --seeds 3
python infer.py --exp hier_seasonal --cutoff 2026-01-14 --seeds 3 --ablate no_seasonal
python evaluate.py --exp hier_seasonal --folds F1,F2,F3,F4
python predict.py  --exp hier_seasonal --ckpt runs/hier_seasonal/F4/best.pkl
```

Each run writes `runs/{exp}/{fold}/{seed}/` with config, posterior summary, metrics JSON,
ELBO curve, diagnostic plots, and appends one row to `results.md`: experiment, fold,
n_covariates, val RMSLE (mean ± std over seeds), aggregate bias (both readouts), Gini,
RMSPE, mean CRPS, 80% coverage, wall-clock.

### Compute budget

| variant | fit | readout (250k) | total per fold |
|---|---|---|---|
| B0 | ~2 min CPU | ~20 s | under 5 min |
| B1 | ~10 min GPU / 40 min CPU | ~1 min | under 15 min GPU |
| B2 | ~20 min GPU | ~1 min | under 30 min GPU |

Five folds × three seeds × three variants fits comfortably in an evening. State these
numbers in the repo — "resource efficiency of the solution" is an explicit jury criterion
and this family wins it outright against a GBDT ensemble.

---

## 10. Hybrid export

Expect B0 to lose to a tuned GBDT on raw RMSLE, and B1/B2 to be close. The ensemble is
where the value is, and it costs nothing extra:

Write `features/bayes_{fold}.parquet` with `user_id` plus `p_alive`, `log_lambda_post_mean`,
`log_lambda_post_sd`, `log_mu_post_mean`, `log_mu_post_sd`, `e_orders`, `e_gmv`,
`e_log1p_gmv`, `pred_p10`, `pred_p90`, `pred_p90 - pred_p10`, and the death-time posterior
entropy. Ten to fifteen columns, generated **at the fold's cutoff using only that fold's
feature window**, so they can be joined into the tabular pipeline without leakage.

The posterior *standard deviations* are the interesting ones. A GBDT can reconstruct
`e_orders` from RFM features given enough splits; it cannot manufacture "how much does the
history actually pin this user down", and that is exactly the quantity RMSLE cares about
when deciding how hard to shrink toward zero.

---

## 11. Checks to write as asserts, not as hope

- **No lookahead — structurally, not by discipline.** The feature builder must receive a
  dataframe already filtered to `event_date ≤ C`; the full raw table is never in its scope.
  Make peeking impossible rather than tested-for. (The Ariel `Model.infer()` does this by
  calling `unload_spectrum()` to physically delete the label from every test object before
  inference, unless a flag explicitly permits access.) Keep the recompute-and-compare assert
  from `CAUSAL_EXP.md` §9 as a second line of defence, not as the primary control.
- `t_x ≤ T_u` for every user; `k_last < K`; `Σ_k n_k == x`; `Σ_k g_k == total_gmv` to
  float32 tolerance.
- Target window is exactly `horizon` calendar days and shares no day with the feature
  window. Hand-check `y[u]` against `gmv[u, C+1 : C+31].sum()` for several random users.
- Every user in a fold has ≥1 row in `[start_date, C]`. Assert, don't filter silently.
- No train user id in the val set. Covariate standardisation uses train `mu`/`sigma` —
  assert the objects are identical, not merely equal-shaped.
- **Death-time marginalisation is a valid log-probability.** For a handful of users,
  enumerate all `K+1` branches by brute force in float64 and compare to the vectorised
  `logsumexp`. Assert to 1e-6.
- **`P(alive)` is a probability.** In `[0, 1]`, and monotonically decreasing in
  `days_since_order` when all other stats are held fixed. If it is not monotone, the
  marginalisation has an off-by-one in the admissible death blocks.
- **Shrinkage behaves.** For two users with the same empirical rate `x/T_u` but
  `x = 1` vs `x = 50`, assert the posterior mean λ of the first is strictly closer to the
  population mean. This is the one-line test that the hierarchy is real and not three
  independent per-user fits.
- **Readout consistency.** `expm1(E[log1p Y]) ≤ E[Y]` for every user (Jensen). Any
  violation is a bug in the quadrature grid or the truncation of `n`.
- `n_max` truncation covers ≥ 1−1e-8 of the count mass for every user; assert on the max.
- **Parameter recovery.** Simulate 20k users from the fitted generative model at known
  parameters, refit, assert every population parameter is recovered within 2 posterior sd.
  Run this once per model change, not per experiment.
- Covariate count matches what §3.3 claims; log `d` per run.

---

## 12. Sanity checks at submission time

Distinct from §11. Those check that the code is *correct*; these check that the *outputs are
plausible* on data whose ground truth we will never see. The Ariel repo calls
`sanity_check(f, value, name, code, limit)` roughly twenty times through preprocessing and
fitting, with limits recorded by running on visible data and violations raising a coded
exception on the private set. For a competition with a hidden private split this is the
highest-value practice available and costs an afternoon.

Record the min/max of each quantity across folds F1/F2/F4, widen by a margin, and **raise
rather than write a CSV** if the 2026-02-13 fit falls outside:

| quantity | why it catches things |
|---|---|
| mean and sd of `P(alive)` | churn latent collapsing to 0 or 1 |
| fraction of users with `predict < 0.01` | zero-inflation drifting between folds |
| mean and p99 of `predict` | scale errors, exploding `μ_u` |
| `Σ pred_mean / Σ (last-30-day GMV)` | aggregate calibration before we can see the target |
| posterior mean of `s_next` | the seasonal extrapolation going somewhere indefensible |
| tail mass beyond `n_max` | truncation in the §5.2 readout silently biting |
| share of users hitting the variance floor | §4.2 floors binding for most of the population |
| `a_c`, `b_c`, `γ` from §5.3 | the blocking rule, checked automatically |

Record every value seen even when it passes, so the limits tighten over the course of the
competition.

## 13. Regression tests

Three configurations — B0 baseline, B1 default, B1 with the covariate block disabled — run on
a fixed 5k-user subsample at fold F4, with the resulting metrics pickled as a reference and
compared at 1e-3 tolerance on every subsequent run. Fail loudly on mismatch. Add a loader
test that pickles the sufficient-statistic tables and compares them exactly.

This is what `ariel_test.py` does (three cases, three planets, `regression.pickle`, 1e-3), and
"repository quality is judged" is in our own notes from the rules.

## 14. Caveats to report, not to fix

- **Gamma-Gamma independence.** B0 assumes order value is independent of order frequency.
  Compute `corr(x, m̄)` on the buyer cohort and put the number in `results.md`. It will not
  be zero. B1/B2 relax it only partially — the latents are conditionally independent given
  covariates, so shared covariates induce correlation but a residual dependence is not
  modelled.
- **Constant hazard.** Geometric churn means a user who has been alive for 12 blocks is as
  likely to die next block as one alive for 2. B2's `δ` term tests this; if `δ` is
  significant, B0/B1 are mis-specified in a way that matters for long-tenure users.
- **Block discretisation** throws away within-block timing. A user who ordered on days 1
  and 2 of a block looks identical to one who ordered on days 1 and 30. Continuous-time
  BG/NBD (B0) does not have this problem, which makes the B0 vs B1 comparison
  not-quite-like-for-like. Note it rather than pretending the folds are clean.
- **Mean-field SVI understates posterior variance.** Point estimates are barely affected;
  the interval coverage in §7 will be optimistic. Report the coverage numbers as measured
  and attribute the gap rather than widening the intervals post hoc.
- **Latents shared across a grouping are dangerous.** The 2024 Ariel solution tried sharing
  the star spectrum across planets orbiting the same star: it gained on training and was
  disastrous on test, because the grouping structure differed between the two sets. Any
  latent we share across a user segment whose composition is not stable between folds and
  submission carries the same risk. If we try segment-level pooling, validate it on a fold
  whose segment mix differs from F4, not just on F4.
- **Nominal prices.** GMV is in nominal roubles across 14 months. `μ_u` estimated from
  early-2025 orders is not on the same scale as the 2026 forecast. Either deflate by an
  index or accept a systematic under-prediction of `E[Y]` and report it — do not
  half-correct it inside the model.
- **`s_next` is extrapolation.** It rests on 2026 February–March resembling 2025
  February–March. That is a real assumption about a real holiday calendar and it is the
  first thing to blame if the private LB diverges from F4.
- **Same caveat as `CAUSAL_EXP.md` §4, inverted.** The rolling-origin folds answer "does
  this generalise to next month?" but each fold's user population is the same 250k. Neither
  design tests generalisation to a genuinely new cohort. The competition's private split is
  by user on the same window, so the user axis is the one that matches the leaderboard —
  and the time axis is the one that matches reality.
