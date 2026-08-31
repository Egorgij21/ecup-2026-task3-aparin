# IDEAS.md — external research track

Purpose: find approaches that are **fundamentally different in kind** from anything in
`experiments.csv`, source them from the literature / other competitions, and cheaply test
feasibility. This file is the research log; confirmed ideas graduate to `BACKLOG.md`, refuted
ones graduate to the graveyard. Strategy rules in `CLAUDE.md` still bind — in particular
§4.1 isolation, §3.4 the noise floor, and §4.3 every run gets a row.

Started 2026-08-22. Prior art already reviewed by this project: `PAPERS.md` (28 papers),
`PAPERS_new.md`, `PAPERS_FEATURES_AND_IDEAS.md`. **Nothing below duplicates those** — each
entry states explicitly why it is not already killed.

---

## Status board — 2026-08-22

**Research round 2 (2026-08-23): I17 (TFM gate), I18 (pseudo-label), I19 (within-day Pearson GRU**
**loss) and I21 (feature neutralisation) all REFUTED; I20 (Heckman) parked. The loss-geometry axis**
**(I1/I2/I3), the TFM axis, and the entire post-processing axis are now shut.**
**Round 3 — the TOP-1 push (the gap is only +0.0005 rho). I25 (negative correlation learning /**
**manufacture a decorrelated member) REFUTED by an algebraic reduction — it is a slider between**
**two already-dead points. I22 (modern tabular NN, RealMLP) REFUTED 2026-08-25 — the run finished**
**on its third attempt and came in STRONG (rho_B 0.6716) and NOT decorrelated (r 0.9954), 0.10x the**
**bar; I23 (sequence pretraining, different representation) parked. The deeper truth (§1s): the blend**
**already captures ~all the learnable signal, so no member can disagree with it AND be right.**

| § | idea | status | headline |
|---|---|---|---|
| **I17** | **TFM scale-penalty gate** | **REFUTED (e0900–e0905, 5-fold)** | the TabPFN-2.5 50k-row context costs the LightGBM **+0.01718 RMSLE**, TabPFN-3's 100k **+0.00887**, even 1M rows **+0.00073**; the top-200 feature cut costs **+0.00020** (nil). Row-context is the binding limit, every arm is a weaker near-twin (r 0.987–0.999), so no TFM can reach §1f's +0.04 bar. Confirms BeyondArena/Closer-Look on our own data — do not install |
| **I18** | **Transductive pseudo-labelling** | **REFUTED (e0906–e0907, 5-fold)** | +0.00026 (w=1.0) / +0.00009 (w=0.3), both nil — the model already sits at the conditional mean of its own bins (§1q), so self-labels only reinforce it. r_vs_ref 0.9993 |
| **I19** | **Affine-invariant (within-day Pearson) loss on the GRU** | **REFUTED (e0290–e0293)** | every λ>0 worse on RMSLE (monotone in λ), and rho_partial vs the 9-member blend is ≈0/negative (best +0.00027 vs 0.024 bar); the arms correlate *more* with the family than a plain GRU. Port gate passed (λ=0 = e0101 byte-for-byte). Closes the loss-geometry axis from a 4th direction; I2/I3 now closed too. **Trap caught: rho_partial was +0.026 vs a single GRU, +0.0003 vs the blend — §E1 inflation on the frozen folds** |
| **I20** | **Heckman selection correction** (unlock the 3 contaminated months) | **parked ~10%** | the one item targeting the largest unused asset, but 2607.05806 shows it fails without an instrument — and we have none. Cheap OOF pre-test stated before any training run |
| **I25** | **Negative correlation learning / diversity-forced ensembling** (manufacture a decorrelated member) | **REFUTED by argument** | NCL with a FIXED blend reduces algebraically to fitting `y + 2λ(y−blend)` — a slider between "fit y" (λ→0, the r≥0.994 twin) and "fit the residual" (λ→0.5, the killed best_iteration=4 GBDT), unstable at λ=0.5. Not a new axis. Learner-collusion (Abe NeurIPS 2023) + the residual being unlearnable (§1q) confirm it. Snapshot/FGE give LESS decorrelation than reseeding. No Kaggle precedent |
| **I22** | **Modern tabular NN as a decorrelated blend member** | **REFUTED (e0913, fold 4, 2026-08-25)** | The TOP-1 push, resolved NO. **RealMLP_TD delivered the strength and not the decorrelation: rho_B 0.671601 at r 0.995356**, e −0.000284, rho_partial −0.003996; bar 0.674699 → **margin −0.003098 = 0.101x the requirement**, in-sample optimal weight **0.000**. The bet was that NN_TORCH's r=0.96 would survive at GBDT strength; it did not — **r=0.96 was a property of being WEAK (rho 0.647), not of being a neural net.** RealMLP is the best non-tree non-recurrent member ever built here (within 0.0021 of the gbdt half) and still lands on the same §1z-E frontier. NN_TORCH (e0912) −0.004; TabICLv2 (e0340) r 0.995, rho 0.659, −0.0005. **Function-class diversity at fixed features does not buy decorrelation, even across the tree/NN boundary.** |
| **I21** | **Targeted feature neutralisation** | **REFUTED (e0294)** | for every p>0 rho FALLS on every fold — including the most-shifted fold 4 — with no divergence; LOFO picks p=0 (Δ +0.00000), random-column control ≈0. The shift-protection the paper reports does not appear in the reachable shift range. **Closes the last non-monotone post-processing lever** — affine, monotone, segment and now feature-orthogonalisation are all nil |
| **I6** | **Measured bracket on the achievable rho** | **DONE, 250 k** | **The most useful result here. `[0.6871, 0.7254]` vs e0049's `0.66113` — the model sits BELOW even the persistent-trait-only end. Read §I6's caveat box before quoting it.** |
| I1 | HL-Gauss histogram loss (CE over binned `log1p y`) | **REFUTED (confirm)** | **+0.00124 calibrated** (not the +0.04382 raw CV — 97 % of that was calibration; see the correction in §E1-CONFIRM). `rho_partial` −0.00249 → zero blend value |
| **I14** | **Two-part hurdle retested at 665 features** (e0010 used 62) | **e0222: better standalone, worthless in the blend** | −0.00048 calibrated, **4/5 folds**, both tie-breakers improve — but `rho_partial` **−0.00500**, so it adds nothing to the 9-member blend |
| I9 | Discrete-time hazard for the buy flag | **REFUTED, replicated** | −0.00539 / −0.00530 at two anchors. The last unbuilt `P1` in `PAPERS.md`, now closed |
| I5 | `user_id` as a signup-cohort covariate | **REFUTED** | −0.00100 / +0.00141, opposite signs, inside the no-op band |
| I11 | Headroom map — *where* the unclaimed variance sits | **DONE, 250 k** | flat across segments (13.6–23.5 %) — **no routing pocket**; but relative capture runs 22 % → 63 % from sparse to dense users |
| **I13** | **Which term has the headroom — classification vs magnitude** | **CLOSED — gap real, closable, unspendable (e0243–e0247)** | measurement stands (80.2 % vs 89.6 %) and the gap is now **closed on demand**: buyers-only training reaches **89.8 %** (+0.0574, 5/5 folds, 574× seed spread). But **all three spending routes fail** — multiplicand +0.00462, feature = noise, weighted loss `w*=1`. Confirms §I15's prediction from the training side |
| **I15** | **Term profiles across 14 models; e0221's magnitude signal** | **DONE** | **13 of 14 models agree to 0.004 on the WHOLE decomposition. e0221's magnitude `rho_partial` is +0.0762 = 6x the best ever — and converts to +0.00011. Closes §I13.** |
| I16 | Latent price/catalogue structure in `gmv` values | **REFUTED** | `gmv` is anonymised off currency units — 0.01 % integers, 1 atom in 64,864 values. No catalogue to recover |
| I12 | Cross-user / peer features (closes Band D e0030) | **REFUTED** | every column negative at both anchors; `peer_rel` has corr 0.02 with the target |
| I10 | Inverse-variance weighting from measured label noise | **REFUTED** | shuffled-weight control scores as much as the real weights |
| I2 | Affine-invariant (correlation) training loss | designed, not run | follows from §1b: after calibration only rho matters, yet MSE pays for level and spread |
| I3 | Listwise / rank loss | parked behind I2 | metric is Pearson, not Spearman |
| I7 | Tabular foundation models (TabPFN-3 / TabICLv2 / TabFM) | parked behind a cheap gate | project's dismissal is stale; the 10 k-context constraint is not |
| I8 | Temporal-shift-aware weighted ERM | parked, low prior | e0070–e0073 found the response surface *flat*, not mis-tuned |

**Three methodological findings came out of the screens and are worth more than the arms that
produced them** — all three are now in `BACKLOG.md`'s graveyard:

1. **The 15 k local screen inflates `rho_partial` by ~an order of magnitude.** A reseeded bag
   of the *identical* model scores 0.086; one junk column scores 0.069; the real best-ever is
   0.0127. Any screen quoting `rho_partial` must ship the reseeded-bag control.
2. **Early-stopping and scoring on the same held-out half manufactured a whole result** —
   +0.00174 on a 2-way split, −0.01240 on a 3-way split, same arm.
3. **Uniform binning of a zero-inflated target creates an empty softmax class that silently
   corrupts every other class** via the shared normaliser. It presents as a port bug.

**Where I would look next, in order:**

1. ~~**Attack the magnitude term directly (§I13).**~~ — **DONE 2026-08-22, CLOSED. See §I13's
   results block and `EXPERIMENTS.md` §1u.** Run exactly as specified: buyers-only training,
   scored `corr(L,·|Z=1)` against the 0.6001 ceiling, recombination asked afterwards. The gap
   **closes on demand** — 80.2 % → **89.8 %**, +0.0574 on 5/5 folds at 574× the seed spread —
   and **none of the three spending routes works**: multiplicand +0.00462, feature =
   noise-equivalent, weighted loss `w* = 1` (0/9 at every weight). Confirms §I15 from the
   training side. **Do not reopen the magnitude term without a mechanism that is neither a
   decomposition, a stack, nor a re-weighting** — those three are now individually measured.
2. ~~**Loss geometry (I2, I3)**~~ — **downgraded.** I1's confirm has HL-Gauss losing by +0.044
   and +0.038 on the first two folds (~400σ each). That is evidence against §0's whole thesis,
   and I2/I3 share its mechanism. Run I2 only if something else revives the axis.
3. **I7's scale-penalty gate** — a single `lgb.train` on 10 k rows that decides the whole
   tabular-foundation-model direction without installing anything.

---

## 0. The axis nobody has varied

Read `EXPERIMENTS.md` §1b/§1c/§1f as one statement and a gap appears.

The project has varied, exhaustively:

| axis | how far it was pushed | outcome |
|---|---|---|
| information (features) | 1021 columns, tsfresh port, 19 causal candidates, funnel, year-lag | **closed** (`FEATURES.md`) |
| function class | LightGBM, XGBoost, CatBoost, Ridge, AutoGluon, GRU, LSTM, TCN, transformer×3, BTYD, hierarchical Bayes | **closed** (§1c, §1k) |
| target parametrisation | log1p, per-user residual, ratio-to-baseline | **closed** |
| CV protocol | anchor-split, user-split | **closed** |
| post-processing | affine calibration solved exactly from LB | **solved** (§1b) |
| blending | 21 predictors, optimal weights, LOFO-honest | **closed** (§1m) |

And has held **one** thing fixed the entire time:

> **Every one of the ten families minimises squared error on `L = log1p(y)`.**
> BTYD and the Bayes model are the only exceptions, and they lost outright because they
> optimise a likelihood for a *different estimand* (§1e: `log1p(E[y])` costs **+0.5626**).

That is not a small detail. It is the most likely single cause of the two facts that define
the endgame:

1. **rho ≈ 0.704 for everything.** Ten families, four feature regimes, two CV protocols.
2. **r ≈ 0.99 between blend members**, and `rho_partial` never above 0.0127 against a
   0.02383 bar (§1f). Members disagree about *fit*, never about *what to fit*.

Two estimators of the same functional, trained by minimising the same divergence on the same
data, are not independent views — they are the same estimator up to noise. The project has
been sampling one point in loss-space with ten different optimisers.

**So the thesis of this file:** vary the **loss geometry**, hold the **estimand**. Every idea
below keeps the prediction an estimate of `E[log1p y | x]` — the functional RMSLE elicits,
which is what `PAPERS_FEATURES_AND_IDEAS.md` §0.1 correctly used to kill ZILN/OptDist — and
changes only how that estimate is *learned*.

### The honest prior on this whole direction

Information is exhausted here; the literature agrees that what remains on this axis is
**optimisation, not information**. Wang et al. (JMLR 2026, [2402.13425](https://arxiv.org/abs/2402.13425))
investigate exactly this and conclude the Histogram Loss's gains "come from improvements in
optimization rather than modelling extra information". That is a *lower* expected gain than
a feature would have had — but it is the only axis with an untested mechanism, and it is the
axis most likely to produce a member that is decorrelated **for a principled reason**
rather than by being worse (§1c's corrected rule).

---

## I1. Histogram loss / HL-Gauss — cross-entropy over binned `log1p(y)`, read out as `E[L]`

**Status: TESTING (E-IDEA-01, see §E1 below).**

### The idea

Discretise `L = log1p(y)` into `K` bins. Train with categorical cross-entropy against a
target distribution over bins (one-hot, or Gaussian-smoothed = **HL-Gauss**). Predict

```
M(x) = Σ_k p_k(x) · c_k          c_k = bin centre
```

This is a *nonparametric conditional distribution* over `L` whose mean is read out. It is
not a distributional assumption, not a mixture head, and not a hurdle.

### Why it is not already killed

`PAPERS_FEATURES_AND_IDEAS.md` §0.1 killed ZILN and OptDist with a clean argument: they
output `p·exp(μ + σ²/2)`, an estimator of `E[Y|x]`, and correctly adapted to RMSLE they
collapse onto the e0010 hurdle model (−0.00012, no effect). **That argument does not reach
HL-Gauss.** `Σ p_k c_k` is an estimator of `E[L|x]` directly — the correct functional, by
construction, with no retransformation and no `p × value` factorisation. Estimand-correct
from the first line.

Nor is it "a classifier": e0160/e0161/e0162 built a *separate binary* `y>0` classifier and
stacked it (worth **+0.00007** over the no-op control, §1b). This is a single model, a single
loss, and no stacking — the zero atom is one bin among K, learned jointly with magnitude.

### Why it might work here specifically

`DATA.md` §6.1 measures the target as **two components**: a 44 % point mass at `L = 0` and a
bulk near `L ≈ 4.2`, with a nearly empty region between them (0.127 % of users). Squared
error on a bimodal conditional target is the textbook worst case: the conditional mean sits
in the empty region, and the gradient treats a `L=0` user and a `L=4.2` user as symmetric
deviations from a value neither of them can take. Cross-entropy over bins lets the model
represent the two modes and does the averaging in the *readout* instead of the *loss*.

That is exactly the regime the literature reports gains in:

- Imani & White, *Improving Regression Performance with Distributional Losses*, ICML 2018
  ([1806.04613](https://arxiv.org/abs/1806.04613)) — the original Histogram Loss.
- Farebrother et al., *Stop Regressing: Training Value Functions via Classification for
  Scalable Deep RL*, ICML 2024 ([2403.03950](https://arxiv.org/abs/2403.03950)) — HL-Gauss
  beats MSE consistently at scale; +67 % peak performance on robotic manipulation; the
  ordinal Gaussian smoothing is the component that matters, `σ / bin_width ≈ 0.75`.
- Wang et al., *Investigating the Histogram Loss in Regression*, JMLR 2026
  ([2402.13425](https://arxiv.org/abs/2402.13425)) — the caveat, recorded up front: the
  benefit is **optimisation**, not extra information, and it needs no expensive tuning.

### Second, independent reason to run it — the blend

For a GBDT the CE version is not merely a re-weighted loss, it is a **structurally different
predictor**: `K` tree ensembles instead of one, combined through a softmax and a nonlinear
readout. §1f says a member's entire value is `rho_partial = corr(L, B | M)`; the project's
best ever is 0.0127 and every candidate so far has been *the same estimator fitted
differently*. This is the first candidate that is a different estimator of the same quantity.

### Falsifier

`Δrho ≤ 0` at both screen anchors **and** `rho_partial < 0.013` (the best-ever member) →
kill, and record that the loss-geometry axis is closed for trees.

### Cost

Screen: 1 laptop run (`scripts/screen_loss.py`), no cluster. Confirm: `K`× a normal
LightGBM CV, so ~1 cluster job on `compute`.

---

## I6. How much is left? A measured upper bound on rho — `scripts/noise_ceiling.py`

**Status: RUN at full scale and confirmed. `reports/noise_ceiling_full.log` (all 250 k users,
job 24065016, `computeshort`, 69 s). The 15 k local run agreed to within 0.002 on every
number, which is itself a check on the estimator.**

### Why this is worth more than another model

`EXPERIMENTS.md` §1b concludes "model optimisation on this problem is finished". The evidence
is that ten families, four feature regimes and two CV protocols all land at rho ≈ 0.704. **That
is agreement between estimators, not a bound.** Ten estimators sharing a loss, a feature
vocabulary and a data-generating process can agree on a wrong answer. This project has never
measured how much of the target is predictable *at all* — it bounded the classification term
with an oracle (§1b), which is a different quantity, and the `rho_decomp` payoff curve was
explicitly flagged as measured along the oracle path.

### The measurement

Write `L_t = θ_t + ε_t`: `θ` is whatever a predictor could in principle know at the cut-off,
`ε` the realisation noise of one particular 30-day window. For two disjoint windows of equal
length at the same instant, `ε` is independent and `θ` is shared, so their correlation *is*
the reliability `r = Var(θ)/Var(L)`, and classical attenuation gives

```
rho(L, any predictor)  ≤  sqrt(r)
```

Two windows can never be at the same instant, so measure `corr(L_j, L_{j+m})` at lags
m = 1…6 windows and extrapolate the decay to lag 0. Doing it at three window lengths is the
consistency check: reliability must rise with length. Population is re-selected before every
window by the frozen-fold rule (active in the prior 30 days), all windows are kept inside the
clean region (end ≤ 2025-11-15), and no model is involved anywhere.

### Result — all 250 000 users

```
W=30  lag(d)     30     60     90    120    150    180
      corr    0.548  0.505  0.487  0.472  0.466  0.446     -> r0 = 0.5475 (0.5262 conservative)

reliability rises with window length:  W=7 0.313 | W=15 0.437 | W=30 0.548   MONOTONE
robustness, disjoint span from 2025-06-18:  r0 = 0.5638  ->  ceiling 0.7344 cons / 0.7401 drift
15k subset, same code:                      r0 = 0.5502  ->  ceiling 0.7268 cons / 0.7401 drift
```

⚠ **The comparison has to be like-for-like, and my first pass was not.** The ceiling is
measured on the clean fold region. The best *submission*'s rho of 0.70378 is on the **test
anchor**, whose window is measurably more predictable (`DATA.md` §9.4: the naive predictor
scores 0.577 there against 0.538 on the folds). Comparing the two mixes populations and
periods. The valid comparison is fold-region against fold-region:

| quantity, clean fold region | ceiling on corr | best achieved there | max remaining |
|---|---|---|---|
| **GMV** `L = log1p(y)` | **0.7254** cons · 0.7390 drift-corr | **0.66113** (e0049 pooled OOF) | **≤ 0.064 rho** |
| **buy flag** `1(y>0)` | **0.6623** cons · 0.6755 drift-corr | **0.59317** (e0160 classifier OOF) | ≤ 0.069 |

At the pooled OOF `sd_L = 2.3524` that is an **RMSLE floor of 1.6192 (conservative) / 1.5848
(drift-corrected) against e0049's 1.7655** — up to **0.146 RMSLE** unclaimed on the folds.

*(Internal check, and it validates both sides of the arithmetic: `sd_L·sqrt(1-rho²)` at
e0049's OOF rho gives **1.76495** against its actual logged CV of **1.76551** — agreement to
0.0006, i.e. the folds really are calibrated, exactly as `calibrate.py`'s `k* = 1.000` said.)*

**On the test anchor the ceiling cannot be measured at all** — the target is unobserved, which
is the whole point of the competition. Scaling the fold ceiling by the naive predictor's
fold→test ratio (0.577/0.538) would put it near 0.78, but that is an extrapolation and is not
claimed here.

### Sharpened: the ceiling is now BRACKETED, using the neuroscience noise-ceiling formalism

This exact problem is solved in sensory neuroscience, where N repeated presentations of one
stimulus give an unbiased split of total power into signal and noise — Sahani & Linden
(NIPS 2003) signal power, Schoppe et al. (2016) `CC_max`/`CC_norm`, and Pospisil & Bair
([PLOS Comp Biol 2021](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8367013/)) on the *bias*
of the naive version. Our seven disjoint 30-day windows are the repeats:

```
SP = (N·Var(mean over windows) − mean(Var of each window)) / (N − 1)
ceiling = sqrt(SP / TP)      SP = 2.5760, TP = 5.4563  ->  SP/TP = 0.4721  ->  0.6871
```

**The direction of its bias is the point, and it is opposite to the lag-0 extrapolation.** The
neuroscience setting assumes the signal is *identical* across repeats. Our windows span 210
days and `θ` drifts, so what survives here is only the component common to **all seven**
windows — the persistent trait — and it discards the recent state a real predictor also knows.
So the two estimators bracket rather than compete:

```
    0.6871              0.6611              0.7254
  persistent-trait     e0049 OOF        conservative lag-0
   ceiling (lower)      ACHIEVED       upper bound (upper)
```

**The model sits BELOW the lower end of the bracket.** It does not fully capture even the
component that is stably present across seven consecutive months — the part of the signal
least dependent on any assumption about drift. That is a stronger statement than "there is an
upper bound somewhere above us", and it is the sharpest thing in this file.

*(The lower end is a ceiling for a predictor that knows the 210-day-persistent trait. Whether
that trait is estimable from pre-anchor history is an inference, not a proof — supported by
the lag curve decaying only ~19 % over 150 days, i.e. it is a trait and not an episode.)*

### ⚠ What this does and does not say — read before quoting the number

**It is an UPPER bound on the ceiling, not a promise of headroom.** `θ` as measured is
everything shared between two *contemporaneous* windows, which includes common causes that
arise **after** the cut-off and are therefore unknowable to any predictor. So the true Bayes
ceiling is somewhere at or below 0.7254, possibly much closer to 0.661 than to 0.7254. Nothing
here exhibits a better predictor, and **an upper bound cannot prove a prize exists.**

What it *does* say, and this is new: **the entire remaining prize is bounded, for the first
time, at ≤ 0.038 rho ≈ ≤ 0.055 RMSLE** — and the bound applies to every competitor, not just
us. The leader's 0.000688 edge is **~1.3 % of the maximum conceivable remaining gain**. Both
statements were unavailable before; the project had a saturation *belief* and now has a number.

**Bias accounting, both directions:**

* **Bound too generous** (the main one): post-cut-off common causes count as `θ`. Mitigated
  three ways — the conservative variant refits on lags ≥ 2 so a burst straddling one window
  boundary cannot inflate it; the drift correction charges for `θ` moving between the cut-off
  and the window centre; and, decisively, **the lag curve is nearly flat** (0.548 → 0.446 over
  150 days, ~19 % decay). A shared component that persists for five months is a user *trait*,
  not a transient event — and a trait is exactly what 409 days of history can estimate.
* **Bound too tight:** if `ε` were negatively autocorrelated (satiation — a big month followed
  by a small one) then `c(1) < r`. No sign of it: the curve is convex at lag 1, i.e. `c(1)`
  sits *above* the trend, which is the opposite signature.
* The panel's guaranteed-activity conditioning lies in the future of every window used, so it
  applies uniformly rather than differentially.

### Validation — four independent checks, all pass

1. **Falsifier:** every ceiling must exceed the corresponding achieved value on the same
   population. GMV 0.7254 > 0.66113; buy flag 0.6623 > 0.59317. Pass. *(A ceiling below an achieved value would have meant
   the design was wrong, not the models.)*
2. **Known-quantity reproduction:** the lag-30 correlation, **0.548**, is the correlation between
   consecutive 30-day GMV windows — which `DATA.md` §7.1 measured independently at **0.557**.
   Different code, different population rule, agrees to 0.009.
3. **Monotone in window length** — 0.313 / 0.437 / 0.548 for W = 7 / 15 / 30. A sampling-noise
   argument demands this; an estimator picking up an artefact need not satisfy it.
4. **Robustness across calendar spans** — a disjoint span starting 2025-06-18 gives 0.7344
   conservative / 0.7401 drift-corrected against 0.7254 / 0.7390. And the 15 k subset
   reproduces the 250 k run to 0.002 on every headline number.

### What it changes about where to look

The bound is much *closer* to the current score than a fresh reader of `EXPERIMENTS.md` §1c
would guess from "ninety experiments moved rho by 0.006" — but it is not zero, and it is
distributed informatively: the buy-flag gap (≤ 0.069) is as large as the GMV gap (≤ 0.064), on a term §1b puts at 78.6 % of Cov(L, M).
§1b closed the classification lever on the grounds that four classifiers agreed within 0.002
AUC and that the AUC that *was* available converted at ~0. The conversion finding stands. The
"wall" finding does not — it is the same estimator-agreement inference this section was
written to question, and the bound leaves room above it.

**Follow-up worth one job, not more:** re-run `rho_decomp.py`'s BETWEEN/WITHIN split against
these ceilings instead of against the oracle, which will say how the ≤ 0.038 divides between
the two terms and therefore which one any future idea should target.

---

## I2. The affine-invariant (correlation) loss — the metric's own algebra applied to training

**Status: designed, not yet run. Blocked on I1's outcome — same axis, more expensive testbed.**

### The idea

`EXPERIMENTS.md` §1b established, and it is now standing protocol, that every submission is
affine-calibrated in log space, after which

```
RMSLE = sd_L · sqrt(1 - rho²)
```

So **the score depends on the model output only through `rho = corr(L, M)`**. Level and
spread are free. Now write down what the training loss is actually spending:

```
MSE(M) = (mu_L - mu_M)²  +  (sd_L - sd_M)²  +  2·sd_L·sd_M·(1 - rho)
         └── free ──┘      └──── free ────┘      └── the only real term ──┘
```

**Two of the three terms are removed for free at submission time, and the model is being
trained to minimise all three.** Not a rhetorical point: e0150 vs e0151 prices the spread
correction alone at **0.00078 LB** — i.e. the raw model is materially under-dispersed at the
test anchor, and MSE is what asks it to be.

The fix is a loss that is invariant to exactly the transform the calibration applies:

```python
loss = 1 - corr(L_batch, M_batch)          # or 1 - corr²; both affine-invariant
```

### Why the mechanism is real but bounded

For an unconstrained function class the two objectives have the same optimum (`E[L|x]`
maximises both), so this cannot help an unregularised, converged model. It bites where the
model is **regularised and early-stopped** — and this one is, hard: e0141 early-stops at
**13–25 epochs**, and e0106 prices 30 epochs at **+0.0204**. A model on that cliff spends
its handful of epochs partly on fitting a level and a spread that calibration will overwrite.
Same mechanism class as I1: optimisation, not information. State it that way, do not oversell.

### Testbed

The GRU (`src/seqnet.py`), not the GBDT — correlation is a batch-level statistic, which is
natural for SGD and awkward for boosting's per-sample Newton steps. Roughly a 20-line change.
Must ship with a **matched same-session control** (`BACKLOG.md`: cross-session drift is
+0.00027…+0.00046, larger than most effects here).

### Falsifier

No improvement in `rho` at matched control, and `rho_partial < 0.013` → kill, and I1+I2
together close the loss-geometry axis.

---

## I3. Listwise / rank loss — the extreme of I2

**Status: parked behind I2. Same axis, weaker theory.**

Push I2 further: train a differentiable rank objective (pairwise logistic, or a soft-Spearman
via `torchsort`). Maximally different loss geometry → the best a-priori `rho_partial`
candidate in the file.

**Why it is parked and not promoted:** the metric is Pearson `rho` on `L`, not Spearman. A
rank-perfect model with the wrong *shape* scores badly, and the fix (map ranks back through
the marginal of `L`) is a monotone transform whose Pearson `rho` is not guaranteed to be
better. Run only if I2 shows the affine-invariance mechanism is real; then the rank loss is
its natural extension.

---

## I4. Self-supervised sequence pretraining — the project's own recommendation, never built

**Status: parked. Documented so it is not re-derived a fourth time.**

`PAPERS_FEATURES_AND_IDEAS.md` §3 recommended a concrete order — TabM, then Abacus-style
count/sum pretraining, then CoLES — and `src/` contains **none of it**. This is the only
`P0`-rated recommendation in the project's own literature review that was never executed.

New evidence since that review was written, both 2026:

- [2607.09955](https://arxiv.org/abs/2607.09955) — *A Foundation Model for Multimodal Event
  Sequences in Financial Applications*. Next-event pretraining on unified user event
  sequences; reports **+1 % NPV over the GBDT baseline**, A/B tested, and shipped as the
  default production scorer. First result on this line with a deployed comparison rather than
  an offline benchmark.
- [2603.23032](https://arxiv.org/abs/2603.23032) — *Generative Event Pretraining with
  Foundation Model Alignment*.
- [2401.01641](https://arxiv.org/abs/2401.01641) — *Towards a Foundation Purchasing Model*,
  generative autoregression on transaction sequences.

**Honest prior, low:** this project's own `e0101` already shows a GRU on **13 raw daily
channels** reaches within 0.0017 rho of the 665-feature LightGBM, i.e. the raw sequence is
already being read about as well as the features are. Pretraining buys representation, and
representation is not what is binding. Also `EBES` (§4.2 of `PAPERS.md`) finds sequence order
matters less than expected, and our target is an order-invariant 30-day **sum**.

**Cheapest informative version** if it is ever run: pretrain once on data truncated to the
earliest anchor's train-end, freeze the encoder, reuse across all five frozen folds, feed the
embedding to the existing LightGBM. Leakage-safe and 5× cheaper than per-fold retraining.

---

## I9. Discrete-time hazard supervision for the buy/no-buy term — **REFUTED at screen tier**

**Status: RUN, negative, sign-replicated at both anchors. `scripts/screen_hazard.py`.**

### Why it was the best-motivated idea in this file

§I6 measures the largest gap in the project on the **buy flag**: ceiling 0.6623 against 0.59317
achieved, on the term `EXPERIMENTS.md` §1b puts at **78.6 % of Cov(L, M)**. And every classifier
ever built here — e0160 LightGBM-binary, e0161/e0162 GRU-BCE — was trained on **one label per
user**: did they buy in the 30-day window. *A user who bought on day 2 and a user who bought on
day 29 are the same training example.* Discrete-time survival uses the timing:

```
split the horizon into J intervals, one row per interval a user survives into,
fit the per-interval hazard h_j,  then  P(buy in 30d | x) = 1 - Π_j (1 - h_j(x))
```

Same estimand, strictly more supervision, no new information. Implemented as **survival
stacking** ([2107.13480](https://arxiv.org/abs/2107.13480)) so it runs on the existing feature
matrix with the existing LightGBM. Refs: Nnet-survival
([1805.00917](https://arxiv.org/abs/1805.00917)), Kvamme & Borgan
([1910.06724](https://arxiv.org/abs/1910.06724)), *Buy when?*
([2308.14343](https://arxiv.org/abs/2308.14343)). **`PAPERS.md` 6.2 rated this `P1` and it was
never built** — it was the last unbuilt `P1` in the project's own literature review.

### Result — worse, by the same amount, at both anchors

6 intervals × 5 days, three-way user split, `corr(Z, p)` on users no arm ever saw.

| arm | A2 (2025-10-16) | A1 (2025-06-18) |
|---|---|---|
| `binary_30d` (the installed design) | **0.60799** | **0.56176** |
| `hazard` (survival-stacked, J=6) | 0.60260 | 0.55646 |
| **Δ** | **−0.00539** | **−0.00530** |
| corr(hazard, binary) | 0.971 | 0.964 |

**Verdict: kill at screen tier.** Not a cluster confirm — the direction is negative, it
replicates, and the mechanism is legible.

### Why it loses, which is the part worth keeping

1. **The extra supervision answers a question the metric does not ask.** Knowing *when* inside
   the window a user buys is only useful insofar as it identifies *whether*. The hazard model
   spends capacity on the within-window timing distribution, which the 30-day sum integrates
   away — the same reason `EXPERIMENTS.md` §3b's dense per-day supervision was worth
   **+0.00012**, and the same reason the target is order-invariant by construction.
2. **The product form compounds error.** `1 − Π(1 − h_j)` multiplies six separately-estimated
   hazards; each interval's calibration error enters the product, and the per-row event rate
   falls to **0.14**, so each hazard is estimated from a scarcer signal than the single 0.57
   binary label was.
3. It lands in §1c's known-worthless quadrant — **decorrelated (r = 0.96–0.97) but weaker.**

### Honest limits on this kill

Screen tier, 6,718 training users, and §E1 measured this instrument's no-op band at ±0.005 —
so −0.0053 sits right at that edge on magnitude alone. What carries it is the **replication**:
−0.00539 and −0.00530 at two anchors whose target windows are four months apart. A noise draw
does not reproduce to 0.0001. (The two runs do share the 15 k user subset and the split seed,
so they are not fully independent — the outcome noise is, the population is not.)

**Do not read this as "the buy-flag gap is closed."** §I6's ≤ 0.069 stands; this is one
attempt on it, and it says the gap is not reachable by re-supervising the same features with
timing. It says nothing about a different information set.

---

## I10. Inverse-variance sample weighting from *measured* label noise — **REFUTED**

**Status: RUN, negative, and the no-op control is the reason. `scripts/screen_weights.py`.**

§I6 established that ~45 % of `Var(L)` is single-window realisation noise, and that noise is
violently heteroscedastic: a user buying 20× a month has a nearly deterministic
`log1p(30-day sum)`; a user at `p(buy) ≈ 0.5` has a target that flips between 0 and ≈ 4.2.
L2 weights both equally, so most of the gradient signal from the second group is noise.

Batch Inverse-Variance Weighting ([2107.04497](https://arxiv.org/abs/2107.04497)) is the
standard response — `w = 1/(σ²_noise + c)`. Its stated precondition is that "the labelling
process can estimate the variance of the noise distribution for each label", which is normally
the blocker. **Here it is free:** the within-user variance of `L` over the six 30-day windows
*before* the anchor estimates `σ²(u)` causally, from data the model already has. Weights depend
on `x` only, never on `y`, so the estimand stays `E[L|x]`.

| arm | A2 (2025-10-16) | A1 (2025-06-18) |
|---|---|---|
| unweighted | 0.66130 | 0.61912 |
| `inv_var` — `w = 1/(σ²+c)` | +0.00437 | +0.00301 |
| `prop_var` — `w = σ²+c` (opposite direction) | +0.00226 | −0.00071 |
| **`shuffled` — inv_var's weights permuted across users** | **+0.00355** | **+0.00387** |

**The shuffled control scores as much as the real weights, at both anchors.** Same weight
*marginal*, zero relationship to the user. So what moves rho here is *having non-uniform
weights at all* — a mild bagging effect at n = 6.7 k — not the information in them.

**And the premise is TRUE, which is what makes this worth writing down.**
`corr(historical sd(L), |residual|) = +0.30 / +0.32`: the measured noise estimate genuinely
predicts where the model will be wrong. It still buys nothing. *A correct premise is not a
working intervention* — without the shuffled control this reads as a clean +0.0044 win.

### Also killed here: nonlinear recalibration

rho is invariant to *affine* transforms but **not** to monotone ones, so `g(M) ≈ E[L|M]` was a
genuinely free lever the project had never pulled (§1b only ever fits an affine map). Leave-
one-fold-out on `oof/e0049.parquet`: isotonic **−0.00022**, cubic **−0.00014**. The diagnostic
explains it — `E[L|M] − M` is a **constant −0.017 across all 20 prediction bins**. The
miscalibration is a pure shift, which is affine, which calibration already removes. There is no
curvature for a monotone map to exploit.

---

## I11. The headroom map — *where* is the unclaimed variance? — **DONE**

**Status: RUN and confirmed at 250 k. `scripts/headroom_map.py`,
`reports/headroom_map_full.log` (job 24067890, 71 s). Numbers below are the full run; the 15 k
subset reproduced every prize share to within 1.2 points.**

§I6 gives one number for 250 k users. This splits it by segment so a future idea can be aimed
rather than guessed. Segments are defined causally (buy-days in the prior 90 days). **No model
is trained** — it is arithmetic on the panel plus an OOF file, so the ±0.004 screen band that
invalidated §I10 and §I1's screen does not apply here.

```
segment          share   Var_s(L)   reliability r_s   model R^2   gap      prize share
0 buy-days       0.276     2.316       0.3039          0.0668    +0.2371     23.1%
1                0.143     3.855       0.2679          0.0685    +0.1994     16.8%
2-3              0.186     4.507       0.2680          0.0846    +0.1834     23.5%
4-7              0.195     4.357       0.2788          0.1012    +0.1776     23.0%
8+               0.199     3.002       0.3987          0.2494   +0.1493      13.6%
```

**Consistency check:** total unclaimed within-segment variance is **0.6560** against
`Var(L) ≈ 5.49`, i.e. **12.0 %** — and §I6's global figure is `r − rho² = 0.550 − 0.661² =
0.113`, i.e. **11.3 %**. Two different constructions on different windows, agreeing to 1 point.

### Two readings, and they point opposite ways

1. **In absolute terms the headroom is FLAT** — 13.6 % to 23.5 % of the prize, essentially in
   proportion to segment size. **There is no pocket.** That is a direct negative for the whole
   "route users to a specialist model" line: `BACKLOG` B6 per-segment models, and
   `EXPERIMENTS.md` §10.4's one surviving variant of the branch finding. §1e already found BTYD
   loses in *every decile* by a flat +0.06…+0.10; this says the same thing about the ceiling
   rather than about one model. **If you are looking for a segment to attack, there isn't one.**
2. **In relative terms it is not flat at all.** Fraction of the *reliable* variance the model
   actually captures: **22 % for 0-buy-day users, 63 % for 8+**. The model is three times worse,
   proportionally, exactly where each user's own history is thinnest. That is the one asymmetry
   in the map, and it is the shape you would expect if what limits low-activity users is
   **how much evidence exists per user**, not what the model does with it.

Reading (2) is also, unfortunately, the reading that argues *against* the remaining ideas in
this file: more loss geometry, a better optimiser or a different function class cannot
manufacture evidence that is not in a sparse user's history. The intervention it points to is
**borrowing strength across users** — the one information axis this project has never
opened (`BACKLOG` Band D e0030 k-NN / e0031 cluster-id, both never run). Note the prior is
still poor: `DATA.md` §5.4 measured per-user *trait* reliability at 0.09 for gift response
while the plain spending level control reached 0.65, so cross-user structure beyond level may
simply not be there.

**Caveat, stated in the script's own output:** reliability is measured on windows from
2025-04-19 and the model R² comes from the five frozen fold anchors. Both are in the clean
region but they are not the same windows, so per-segment gaps are indicative, not paired.

---

## I13. Which term has the headroom? — classification vs magnitude, against *measured* ceilings

**Status: CLOSED 2026-08-22. Measurement: `scripts/noise_ceiling.py` (magnitude arm) + `oof/e0049.parquet`. Follow-up experiments e0243-e0247 — see the results block at the end of this section.**

`EXPERIMENTS.md` §1b splits `Cov(L, M)` into **78.6 % buy/no-buy** and **21.4 % magnitude among
buyers**, and closes the classification lever on two grounds: four classifiers agreed within
0.002 AUC, and the AUC that *was* available converted to RMSLE at ~0. **The magnitude term was
never measured against any ceiling at all** — it was simply the smaller share of covariance.

Now both terms have one. Same test–retest machinery; the magnitude arm conditions on the
**scored** window being positive, which is exactly what the achieved number conditions on
(`corr(L, M)` among fold users with `y > 0`). Getting that conditioning backwards leaves zeros
in the target and mixes the buy/no-buy decision back into a supposedly magnitude-only ceiling —
I made that mistake first and it moved the number by 0.015.

| term | ceiling (conservative) | achieved | gap | **fraction captured** |
|---|---|---|---|---|
| overall `corr(L, ·)` | 0.7254 | 0.6611 (e0049 OOF) | 0.064 | **91.1 %** |
| buy flag `corr(Z, ·)` | 0.6623 | 0.5932 (e0160 classifier) | 0.069 | **89.6 %** |
| **magnitude among buyers** `corr(L, ·\|Z=1)` | **0.6001** | **0.4814** | **0.119** | **80.2 %** |

**The model is proportionally weakest exactly where the project stopped looking.** Ten points
of relative capture separate the magnitude term from the other two.

### What this does and does not license

**Does:** it identifies the one term never evaluated against a reachable bound. §1b's closure
of the classification lever is untouched and still correct — 89.6 % captured, and its residual
converts at ~0. Nothing here reopens it.

**Does not:** *relative* headroom is not RMSLE. Converting 0.119 of within-buyer correlation
into overall rho needs the covariance weighting, and §1b's own history is the warning — its
`d(rho)/d(AUC) ≈ 1.2` was measured along the oracle path and the realised rate turned out to be
~0. A crude linear scaling off §1b's oracle experiments ("our split × oracle magnitudes" is
worth +0.054 rho) puts closing the magnitude gap at roughly **+0.012 rho**, but that
extrapolation is precisely the kind §1b retired. **Treat it as a direction, not a size.**

**And note the honest complication:** the magnitude term has been attacked once, by the e0010
hurdle model (`P(buy) × E[L|buy]`), which scored −0.00012. But that used the same features and
the same L2 loss on the buyer subset, and its magnitude half was never scored against a
ceiling — so what it showed is that *that* decomposition does not help, not that the term is
saturated.

**The concrete next experiment this implies:** train a magnitude model on buyers only, score it
as `corr(L, ·|Z=1)` against the 0.6001 ceiling rather than end-to-end, and only then ask
whether the improvement survives recombination. Scoring it end-to-end first is what hid the
term for the whole project.

### ✅ RUN — e0243–e0247 (2026-08-22). The gap closes on demand; nothing can spend it.

The experiment above was built and run exactly as specified, with the recombination question
asked *afterwards* rather than first. Every prediction was pre-registered in
`scratch_thoughts.md` before any code ran. Full write-up: `EXPERIMENTS.md` §1u.

**Step 1 — the gap is real and closable (e0243 vs same-harness control e0244):**

```
                       corr(L,·|Z=1)   per fold                                  % of ceiling
e0243 buyers-only  s0/s1/s2   0.5388   [0.5234 0.5219 0.5409 0.5502 0.5542]         89.8 %
e0244 control      s0/s1/s2   0.4814   [0.4701 0.4659 0.4807 0.4934 0.4986]         80.2 %
                             +0.0574   5/5 folds · 574× the within-arm seed spread
```

**80.2 % → 89.8 %, level with the buy flag's 89.6 %.** The control reproduces e0049 *exactly*
(cv 1.76551, identical per-fold, corr 0.4814), so this is not harness drift. Leak checklist
§3.3 all pass — Spearman rises too (0.5222 vs 0.4853, so genuine re-ranking); it predicts
**high** on zero-target users (3.763 vs 1.451), the opposite of a target leak; wins in all five
quintiles of L. Mechanism as predicted: the all-users model's prediction sd *among buyers* is
1.452 against the buyers-only model's 0.727 — most of its spread there is `P(buy)`, not amount.

**Step 2 — all three spending routes fail:**

| route | result |
|---|---|
| multiplicand `p̂·mag` (e0245) | **+0.00462 worse**; the hurdle *form* alone costs +0.0007…+0.0018 and the better half costs a **further** +0.0028…+0.0039 |
| stacked **feature** (e0246) | **−0.00002, 4/9** vs a noise column at +0.00021, 3/9 — indistinguishable |
| **weighted loss**, w ∈ {1,1.5,2,3,5} (e0247) | **`w* = 1`**, monotone worse (+0.031/+0.073/+0.158/+0.292), **0/9 at every weight** |

**Why each fails, measured:** the magnitude model is `corr = 0.875` with the baseline (0.896
among buyers) — ~88 % redundant. Among buyers its residual correlation is **−0.4152**, the
*wrong sign* for a stack, because there `base ≈ P(buy)·amount`, so a high base can mean
"certain to buy" rather than "will spend a lot". And weighting **changes the estimand**: L2
with weights elicits a *weighted* conditional mean, inflating predictions on the 43 % zero mass
(1.406 → 1.875 → 2.500 for w = 1/2/5). Per `EXPERIMENTS.md` §1q the model is already within
0.0010 of `E[L|M]`, so any re-weighting moves it away from the right estimand.

### This independently confirms §I15, from the training side

§I15 reached the same closure from the **blend** side and got there first with the sharper
argument: *a term defined by conditioning on the outcome cannot be exploited by routing,
because the conditioning variable is the thing being predicted.* These runs are the training-side
version of the same wall — and they sharpen it in one way: **§I15 showed the existing magnitude
signal cannot be routed; e0243 shows that even a purpose-built model that closes most of the gap
still cannot be spent.** The door is locked from both sides.

> ⚠ **The dissociation, now observed three times.** e0243: +0.0574 on `corr(L,·|Z=1)`, nothing
> end-to-end. e0245: a better magnitude half, a worse product. e0247 at w=5: the statistic
> **rises** +0.0085 while RMSLE **degrades** +0.29161 — opposite directions.
> **`corr(L,·|Z=1)` is a diagnostic, not an objective.** I13's own "a direction, not a size"
> caveat was right; the honest update is stronger — optimising this statistic is at times
> *anti*-correlated with the metric we are scored on.

---

## I15. The term-profile finding — real new information, measurably unexploitable

**Status: DONE. The most interesting negative in this file, and it closes §I13.**

### Every model this project has built has the same term profile — except one

Decomposing 14 OOF files into `corr(L,M)`, `corr(Z,M)` and `corr(L,M|Z=1)`:

```
                     corr(L,M)   corr(Z,M)   magnitude
13 non-BTYD models   0.65973..    0.57173..   0.47827..     spread <= 0.004 on EVERY term
                     0.66194      0.57511     0.48197
e0221 (HL-Gauss)     0.65872      0.56508     0.48687   <-- outside the pack on TWO terms
e0170 (BTYD)         0.62709      0.53425     0.47333
```

LightGBM, AutoGluon, GRU, TCN, tuned-seq, Bayes-cov — thirteen models from families this
project treats as distinct agree to **0.004 on the whole decomposition**, not just on overall
rho. §1c's "they all recover the same signal by different routes" is stronger than stated: they
recover it in the same *proportions*.

**e0221 is the only model in the project's history that breaks the pattern** — best magnitude
of anything built (0.48687, above the pack's 0.48197) and worst classification (0.56508, below
the pack's 0.57173). So §I1's mechanism was **right**: cross-entropy over bins does rank
magnitude better on a bimodal target. It simply loses more on the buy/no-buy split than it wins.

### And the magnitude information is enormous by this project's standards

`rho_partial` against the 9-member family, computed **on the magnitude term**:

| candidate | overall `rho_partial` | **magnitude-term `rho_partial`** |
|---|---|---|
| e0221 HL-Gauss | −0.00249 | **+0.07621** |
| e0170 BTYD | +0.01017 | **+0.06804** |
| e0064 (a blend member) | +0.00528 | +0.01507 |

§1f's bar is 0.02383 and the best value ever recorded on the overall target is 0.01269.
**e0221 is at 6× that, and it was killed.** Two models carry large magnitude signal the family
lacks, and the project never saw it because candidates are only ever scored against the overall
target — where the buy/no-buy term is 78.6 % of the covariance and drowns magnitude out.

### It converts to exactly zero, and both reasons are measured

Leave-one-fold-out stacks, **with §1b's mandatory no-op control**:

```
CONTROL: refit on M alone                  1.76245
+ e0221                                    1.76255   +0.00010
+ e0170 BTYD                               1.76236   -0.00009
+ e0221, gated on p(buy)                   1.76257   +0.00011
+ e0221 + e0170, gated                     1.76246   +0.00000
CONTROL: M gated on p(buy), no new member  1.76248   +0.00003
```

**Two independent reasons, both quantified:**

1. **The term is small.** Within-buyer variance is only **17.7 %** of `Var(L)`
   (`Var(L|Z=1) = 1.753`, `Var(L) = 5.534`, `P(Z=1) = 0.558`). So e0221's `dR² = 0.00446`
   *within buyers* is `0.00079` of the total — worth **−0.00124 RMSLE with a perfect gate**.
2. **The gate does not exist.** Exploiting a magnitude advantage means routing to buyers, and
   buyer status is unknowable at prediction time. The best available gate correlates **0.593**
   with `Z`. An oracle gate confirms the information is real — e0221 adds **−0.208** on top of
   an oracle-split control — but the oracle is exactly what the competition withholds.

> **The general statement, and it is the durable one:** a term defined by conditioning on the
> outcome cannot be exploited by routing, because the conditioning variable is the thing being
> predicted. This is why §1b's classifier stack returned +0.00007, and it is why §I13's
> "magnitude has the biggest relative gap" was true and still led nowhere.

### What this does to §I13

§I13 stands as a measurement and is now correctly priced. Closing the *entire* magnitude gap
(0.4814 → the 0.6001 ceiling) is worth **−0.036 RMSLE with a perfect gate** — real, but gated
on an oracle nobody has. **The magnitude direction is closed**, not because the signal is
absent, but because it is only reachable through a door the task locks.

---

## I16. Latent price / catalogue structure inside `gmv` — **REFUTED**

**Status: DONE, cheap. No model trained; a distributional probe on `train.parquet`.**

Every feature and every model in this project treats `gmv` as a **scalar to be summed**. But on
a day with exactly one order, `gmv` *is that order's price* — and prices repeat across users. If
a catalogue is recoverable, a user's "price fingerprint" (which price points they buy at) would
carry category/product-preference information that **no aggregate feature can express**, on a
dataset the organisers describe as having no item ids. That is the kind of thing a saturated
feature search misses by construction, because it never looks at `gmv` as a discrete symbol.

**3,181,713 single-order days (10.39 % of rows), 64,864 distinct 2-dp values.** The probe:

| diagnostic | result | what a real catalogue looks like |
|---|---|---|
| exact-integer values | **0.01 %** | prices are overwhelmingly round |
| values ending .00 / .50 / .99 | **0.02 %** | psychological pricing dominates |
| genuine atoms (count > 200 and > 3× neighbours) | **1 of 64,864** | hundreds of dominant SKUs |
| top-1000 values' share of rows | 41 % | but spread smoothly, ~1,700 each |

The top repeated values are 6.40, 6.62, 6.57, 6.36, 6.37, 6.41 … — a smooth band of adjacent
values with near-identical counts. That is a **continuous density binned at 2 dp**, not a price
list. **`gmv` has been rescaled/anonymised off currency units**, so there is no lattice to
recover and the idea is dead.

**The one real atom:** `0.03`, count **7,222** against a neighbourhood density of **13**
(555×), covering 0.227 % of single-order days. Almost certainly a floor or sentinel rather than
a product. Recorded, not pursued: 0.23 % of a 10 % row slice is far below what the feature
screen can resolve (±0.001, `FEATURES.md`).

### ⚠ My first atom test returned a FALSE NULL, and the mechanism generalises

The first pass used `scipy.ndimage.median_filter(counts, size=51, mode="nearest")` for the
local density and reported **zero atoms**. `mode="nearest"` pads the window by replicating the
edge value — and the 0.03 spike *is* the first element — so its own count filled 26 of the 51
window slots, the median became the spike itself, and its ratio came out **1.0** instead of
555. A 555× anomaly scored as perfectly ordinary.

I reported "zero atoms" before catching it. The fix is to compute the neighbourhood **excluding
self**, with one-sided windows at the edges. **General rule, and it is the second instance this
session** (the first was LightGBM's `K/(K-1)` multiclass hessian): *a smoothing/reference
statistic that can absorb the very observation it is meant to judge will silently return a
null.* Any local-outlier test must exclude the point under test from its own baseline.

---

## I12. Cross-user / peer information — **REFUTED**, and it closes `BACKLOG` Band D e0030

**Status: RUN at both anchors. `scripts/screen_peers.py`.**

§I11 says the model captures 63 % of the reliable variance for 8+-buy-day users but only 22 %
for 0-buy-day users — the shape you get when what binds is *how much evidence exists per user*.
Borrowing strength across users is the standard response, and it is **the one information axis
this project never opened**: all 1021 features are functions of a single user's own history.
`BACKLOG` Band D e0030 (k-NN, 5c) and e0031 (cluster-id, 2c) have been open since the start and
neither was ever run.

**The design constraint that matters:** at test time *no* user's target is known — all 250 k are
predicted at once. So a peer feature may only use neighbours' observed **past**, never their
outcome in the scored window. k = 50 neighbours in a 9-dim standardised behaviour space built
from data ≤ anchor.

| candidate (added alone) | A2 Δrho | A1 Δrho | `corr` with `L` |
|---|---|---|---|
| **noise control** (1 i.i.d. column) | **−0.00186** | **+0.00064** | — |
| `peer_gmv30` — mean neighbour log1p(GMV last 30 d) | −0.00311 | −0.00087 | +0.570 / +0.550 |
| `peer_rel` — own level *minus* peer level | −0.00338 | −0.00410 | **+0.026 / +0.019** |
| `peer_buyrate` — share of neighbours with a buy-day | −0.00143 | −0.00087 | +0.493 / +0.480 |
| `peer_dist` — how typical the user is | −0.00092 | −0.00300 | +0.122 / +0.201 |
| all four together *(bundle, not evidence per §4.1)* | +0.00226 | +0.00300 | — |

**Every individual column is negative at both anchors**, and none beats the noise control. The
4-column bundle is positive at both — but §4.1 forbids a bundle as a unit of evidence, it
carries 4× the control's capacity, and §I10 measured that *any* perturbation of this harness is
worth ~+0.004 (the shuffled-weight control). It is the same artefact.

### The mechanism, measured rather than argued

* **The peer level is a noisy copy of the user's own level.** `peer_gmv30` correlates +0.570
  with the target — and `DATA.md` §7.1 measured the user's *own* last-30-day GMV at **+0.557**.
  Neighbours are selected for behavioural similarity, so their average spend is the user's own
  spend with extra noise. There is nothing new in it, exactly the circularity the design risked.
* **The genuinely new quantity carries nothing.** `peer_rel` — where a user sits *inside* their
  behavioural neighbourhood, which no per-user feature can express (the `rank` block gives
  population-wide rank, never local rank) — has univariate `corr` with the target of
  **+0.026 / +0.019**. That is the cleanest possible statement that local peer structure is
  absent, and it needed no model to see.

This is consistent with `DATA.md` §5.4's split-half result from a completely different angle:
per-user *trait* reliability was 0.09 for gift response against 0.65 for the plain spending
level. **Users resemble each other in level and in almost nothing else** — so a neighbourhood
buys you a worse estimate of a number you already have.

**Consequence for §I11's reading (2).** The proportional weakness on sparse users is real, but
it cannot be fixed by borrowing from similar users, because the similarity metric has nothing to
transport. If it is fixable at all, it is not on this axis.

---

## I7. Tabular foundation models — the project's dismissal is stale, the constraint behind it is not

**Status: parked behind a cheap gate (below). Do not install anything before running the gate.**

`PAPERS_new.md` rejected TabPFN-2.5 with one line — "250k rows is outside the scale regime".
That was right when written and is now out of date on the first half only:

- As of mid-2026 the top single models by Elo on tabular benchmarks are **foundation models**
  (TabFM, TabPFN-3, TabPFN-2.6, RealTabPFN-2.5, TabICLv2), roughly **240 Elo ahead of the best
  single gradient-boosting configuration**.
- [TabICL](https://arxiv.org/abs/2502.05564) is explicitly *"a tabular foundation model for
  in-context learning on **large data**"* — the scale objection is the thing it targets.
- A TFM is the **only candidate in this file with a genuinely different inductive bias**: it
  does not fit parameters by gradient descent on our data at all, it conditions on the training
  set in-context under a prior learned from millions of synthetic datasets. §1f says a member's
  whole value is `rho_partial`; every candidate this project has tried was the same estimator
  fitted differently. This one is not.

**But the underlying constraint is still real, and the honest reading cuts against it:**

- TabPFN v2's native regime is **< 10⁴ samples with quadratic attention**; the standard
  large-data workaround is subsample-ensembling (e.g. four random 10 k support sets, averaged),
  or embedding-then-linear-head. Our folds carry **1–5 M training rows × 665 features**, so a
  10 k context is a 100–500× subsample.
- [TabFM's own headline is a 32-way ensemble](https://arxiv.org/abs/2502.05564); without
  ensembling multiple configurations, **CatBoost moves back ahead**. Our regime — large N, many
  engineered features — is where GBDT is strongest.
- §1c is decisive on what a weaker member is worth here: CatBoost and XGBoost at **+0.032**
  behind the GRU, and genuinely decorrelated at r = 0.974, moved a fitted blend by **+0.00001**.

### The gate — cheap, decisive, and it needs no TFM installed

**Measure the scale penalty first.** Train the existing LightGBM on a 10 k-row subsample of a
fold and score it on that fold's full population. If the 10 k-row GBDT is (say) +0.05 RMSLE
behind the full-data GBDT, then **any** 10 k-context model starts from that handicap and must
make all of it back from its inductive bias alone before it is even at the "decorrelated but
worthless" quadrant §1c mapped. That is a `lgb.train` call on data already on disk.

Run TabICLv2/TabPFN-3 **only if** the gate says the 10 k penalty is small, or if the model can
ingest ≥ 100 k rows × 665 features within one `apini` job. `apini` has **no outbound internet**
(CLUSTER.md §4), so weights must be pre-fetched on `compute` and the job run with
`HF_HUB_OFFLINE=1`.

---

## I8. Temporal-shift-aware training — a sharper version of a knob already turned

**Status: low prior, documented so it is not mistaken for untried.**

Every training anchor ends by 2025-10-16; the test anchor is 2026-02-13. `anchor_drift.py`
already found the test cut-off is **3.9× further** from the training cut-offs than they are
from each other. The project's response was `feature_exclude_patterns` (drop lifetime/365-day
features) and crude recency weighting (e0012, e0070–e0073: **flat above 10 anchors**).

Newer, more principled versions exist — RIDER-style *optimally weighted* ERM under evolving
regimes rather than an assumed exponential decay, and feature-aware modulation for temporal
tabular data ([2512.03678](https://arxiv.org/abs/2512.03678)); see also *Understanding the
Limits of Deep Tabular Methods with Temporal Shift*
([2502.20260](https://arxiv.org/abs/2502.20260)).

**Why the prior is low anyway:** e0070–e0073 did not merely fail to find a good decay rate,
they found the *response surface is flat* — N = 10/14/18/all anchors all within ±0.00005.
A better method for choosing weights cannot help when the objective is insensitive to them.
Fit the optimal weights only if some other result gives a reason to think that flatness was an
artefact.

---

## I5. `user_id` as a registration-order covariate

**Status: free rider on E-IDEA-01.**

`user_id` runs 2 … 918 481 over 250 000 users and the panel is sorted by it. If ids are
assigned at registration — the usual case — then `user_id` is a proxy for **signup date**,
which is *not* recoverable from the activity panel: `tenure_days` measures first *observed*
activity, and the panel starts 2025-01-01, so every user registered before then is censored
to the same value. No feature in `FEATURES.md` uses it.

Trivially cheap, legitimately available at test time, and the kind of thing that is either
worth nothing or worth noticing. Added as one column to the same screen; costs one extra fit.

---

## E1. Experiment E-IDEA-01 — does changing the loss change anything? (2026-08-22)

**Hypothesis.** The ten families agree at rho ≈ 0.704 partly because they all minimise the
same divergence; a cross-entropy-over-bins estimator of the same functional will either score
better (optimisation gain, I1) or disagree usefully (`rho_partial`, §1f).

**Single change.** The LightGBM objective, and only that: `regression` (L2 on `L`) →
`multiclass` CE over `K` bins of `L` with an `E[L] = Σ p_k c_k` readout. Same 1021 features,
same anchor, same 50/50 user split, same hyperparameters, same seed.

**Instrument.** `scripts/screen_loss.py`, the local tabular screen (30 k users, `Δrho` on the
held-out half). Two anchors (A1 = 2025-06-18, A2 = 2025-10-16) so a positive must replicate
in sign. Screen sensitivity is **±0.001 rho** (`FEATURES.md`); this is a `tier=screen`
instrument and cannot decide anything on its own (§4.2).

**Controls, all of which must be reported:**

* **hard-label CE vs the custom-objective port at σ→0** — the custom HL-Gauss objective is a
  hand-written port, and the graveyard is explicit that a wrong port trains, scores, and
  returns a plausible null. At `σ→0` it must reproduce LightGBM's built-in `multiclass`.
* **`rho_partial` against the L2 arm**, not just `Δrho` — §1f says that is the whole blend value.
* **the noise column**, so a positive `Δrho` can be sized against screen capacity artefacts.

### E1-RESULTS — the screen (2026-08-22)

Logs: `reports/screen_loss_A1_k6.log`, `reports/screen_loss_A2_k6.log`, `reports/screen_loss_A2.log`.
Code: `scripts/screen_loss.py` + `src/hlgauss.py`. 15 k users, 6,718 train / 3,373 early-stop /
3,400 score, K = 6, σ/width = 0.75, 1021 features, seed 0.

```
PORT GATE  custom objective vs built-in multiclass, matched zero init:
           max|d raw| = 0.000e+00   -> PASS at both anchors
```

| arm | A1 rho | Δrho | A2 rho | Δrho | trees A1/A2 |
|---|---|---|---|---|---|
| `l2_rmse_es` baseline | 0.61768 | — | 0.66444 | — | 58 / 51 |
| `l2_rho_es` *(no-op: same model, ES on rho)* | 0.62258 | **+0.00490** | 0.66382 | −0.00061 | 24 / 40 |
| `l2_bagK` *(no-op: 6 reseeded L2, averaged)* | 0.62927 | **+0.01159** | 0.66687 | **+0.00244** | 99 / 222 |
| `l2_uid` *(§I5, + `user_id`)* | 0.61909 | +0.00141 | 0.66344 | −0.00100 | 55 / 62 |
| `ce_hard` | 0.62288 | +0.00520 | 0.66275 | −0.00169 | 198 / 282 |
| `hlgauss_s0` *(σ→0)* | 0.62432 | +0.00664 | 0.66456 | +0.00012 | 168 / 354 |
| **`hlgauss`** | **0.62987** | **+0.01219** | **0.66760** | **+0.00316** | 96 / 192 |

**Verdict: neither confirmed nor refuted — escalated to a frozen-fold confirm.** Four things
were actually learned, and two of them are worth more than the arm they came from.

**1. The instrument is the finding, and it kills the screen for this question.**
`l2_bagK` and `l2_rho_es` are *no-ops*: the same estimator reseeded, and the same estimator
early-stopped on a different statistic. They move **+0.0116 / +0.0049** at A1. `hlgauss` moves
+0.0122 at A1 and +0.0032 at A2 — inside the band the no-ops span, at comparable tree cost.
Worse for the blend question: `rho_partial` against the baseline is **+0.086 for a reseeded
bag** and **+0.069 for adding one junk column**, against this project's best-ever real value of
**0.0127** and §1f's 0.02383 bar. **The screen inflates `rho_partial` by roughly an order of
magnitude** — the same failure mode `BACKLOG.md` records for the LightGBM causal proxy at 36×.
Any future screen that reports `rho_partial` must ship the reseeded-bag control, or its number
means nothing. *(This is why the arms list carries two no-ops. Without them `hlgauss` at
+0.0122, positive at both anchors, reads as a clear win.)*

**2. `hlgauss` is the only arm positive at both anchors** (the sign-replication bar
`FEATURES.md` sets) and is the best arm at A1. That is not evidence — see (1) — but it is the
reason to spend one cluster job rather than close the idea here.

**3. Bin count must scale with the sample, and under-populated classes hurt badly.** At K=16
(→14 after merging) on 6,718 training rows — about 480 rows per class — `ce_hard` scores
**−0.01240**. At K=6 it is −0.00169. The confirm uses K=12 against 1.6–5 M training rows.

**4. A new trap, and it generalises beyond this idea.** A uniform bin grid puts an **empty
bin** in the gap `DATA.md` §6.1 measures between the zero atom and the L≈4.2 bulk. An empty
softmax class has gradient `p` with no counterweight, its raw score runs away (measured: −34
after 3 rounds), and **because softmax shares a normaliser it corrupts every other class** —
non-empty classes drifted by 0.033 in raw score purely from that contamination. It presents as
a port bug, not a binning bug. `make_bins` now merges any bin below 0.2 % occupancy.
*Anyone discretising a zero-inflated target on this data will hit this.*

**5. §I5 `user_id` is dead.** −0.00100 / +0.00141, inside the no-op band with opposite signs at
the two anchors. Either ids are not assigned in registration order, or tenure already carries
it. Cheap, and now closed → `BACKLOG.md` graveyard.

**Two port bugs were found and fixed before any number above was believed** — recorded because
both are silent, both produce plausible results, and one of them is not in any documentation:

* LightGBM's built-in `multiclass` **boosts from the class prior**; a custom objective starts
  at 0. Round-1 raw scores are log-priors (≈ −3 for a rare class). Fixed with an explicit
  `init_score`, which `predict` does *not* include and which must be added back at readout.
* LightGBM's multiclass Newton step uses **`hess = K/(K−1) · p(1−p)`**, not the textbook
  `p(1−p)` and not XGBoost's `2·p(1−p)`. Getting it wrong is a pure step-size error that still
  trains and still scores. Established by bisection against the built-in objective, then
  verified to `max|d raw| = 0.0` for K ∈ {3, 5, 8, 16}. `port_exact_check` now runs this gate
  **inside every run**, screen and confirm, and aborts the job on failure.

### E1-CONFIRM — frozen folds (submitted 2026-08-22, job **24065665_[0-1]**, `compute`)

**Smoke-tested first** (job 24063949, `computeshort`), per CLUSTER.md gotcha 8, and it caught
one config error and validated three things worth stating:

* the port gate passes **at full scale**: `K=11 (requested 12) ... max|d raw| = 0.00e+00 PASS`
  — and note the merge fired, exactly as `DATA.md` §6.1's empty region predicts;
* the readout path is sound end-to-end: fold 3 scored **1.80177 against naive 1.91620**
  (Δ −0.11442) while capacity-capped at 60 boosting rounds by the smoke config;
* the first attempt failed on `max_train_anchors: 3` + `es_gap_days: 30` leaving no fit
  anchors — a config error, not a code one, and cheap to find at `computeshort` prices.

**Results are not in yet.** When they land: `scp` `runs/e0220.json`, `runs/e0221.json` and
`oof/e022*.parquet` back — CLUSTER.md §8, the cluster copy of `experiments.csv` is not
authoritative — and append the rows locally. `run.py` writes them itself.

`configs/e0220_l2_es.yaml` (control) and `configs/e0221_hlgauss.yaml` (treatment),
`scripts/e0220_e0221.slurm`, `src/run.py` gains an additive `model: hlgauss` branch (53 lines,
nothing existing modified — verified by diff against the cluster copy, so teammates' in-flight
e0210–e0217 are unaffected).

The control exists because **e0049's logged 1.76551 is the wrong comparator**: it uses
`fixed_rounds: 178` and therefore trains on ~4 more anchors per fold than any early-stopped
arm. A delta against it would confound the objective with the training set (§4.1). e0220 is
e0049 with early stopping and nothing else; e0221 is e0220 with the objective swapped.

**Decision rule, pre-registered:** `Δ = cv(e0221) − cv(e0220)`. Keep only if
`|Δ| > 2σ_noise = 0.00018` **or** it wins ≥ 4/5 folds (§3.4). Separately compute `rho_partial`
of e0221's OOF against the existing blend — the §1f bar is 0.02383, best-ever 0.0127. Note in
advance that the honest prior is low: the screen showed `hlgauss` ≈ a 6-seed bag of the L2
model, and multi-seed averaging is already a priced, known move here (`BACKLOG.md` C5).

---

# Research round 2 — 2026-08-22 (external literature + competition precedent)

Three parallel searches (arXiv 2025–2026, tabular-foundation-model status, competition
precedent incl. the E-CUP page itself). The full findings are in `reports/` implicitly via the
citations below; what changed the plan:

- **The E-CUP leaderboard is public** (ods.ai/competitions/e-cup-2026-search): top-20 spread is
  **0.0031** (#1 1.645444 … #20 1.648590), champion e0162 sits ~#5. No participant has published
  an approach or score (checked GitHub + Habr + Telegram). One public baseline repo
  (`github.com/k0ist/OZON-E-CUP-2026-3-track`) uses LightGBM-Tweedie(p=1.3) or classifier×log1p
  — both already in our graveyard.
- **Two cheap, genuinely-unblocked directions surfaced**, both now running (§I17, §I19). One
  medium-cost high-risk direction (§I18/Heckman) is partly testable for free and partly not.
- **Everything else was nil**: TFMs lose on our regime (BeyondArena 2606.30410, "Closer Look"
  2502.17361), label-noise-regression has nothing with measured gains at 44% noise, tabular TTA
  is classification-only, 2026 event-sequence FMs report no spend-regression gain on top of a
  feature-rich GBDT, and every 2025–26 LTV paper (DynaMoLTV, SHORE, AgentLTV, GRePO-LTV) is
  same-in-kind as the dead ZILN/OptDist/multi-task/price-lattice lines.

---

## I17. The scale-penalty gate for tabular foundation models — **RUNNING (e0900–e0905)**

**Status: submitted to the frozen folds (job 24098163, `compute`), chained behind a smoke.**

### Why the project's dismissal was stale and why the constraint is not

`PAPERS_new.md` skipped TabPFN-2.5 in one line ("250k rows outside the regime"). Six months on,
the frontier is TabPFN-3 (arXiv [2605.13986](https://arxiv.org/abs/2605.13986), May 2026),
TabICLv2 ([2602.11139](https://arxiv.org/abs/2602.11139)) and TabDPT-Turbo
([2608.01400](https://arxiv.org/abs/2608.01400)) — all support regression, and their own papers
claim wins over 8-hour-tuned GBDTs. **But the win is always inside their validated envelope**,
which is the problem: TabPFN-3 is validated to **1M rows × 200 features** or **100k × 2,000**,
TabICLv2 to **1M × 500**. Our folds are **1.6M–5.2M rows × 665 features**. Every ICL model
therefore sees a *subsample* of the evidence the LightGBM sees, and the independent benchmarks
in exactly our regime say GBDTs still win: **BeyondArena** ([2606.30410](https://arxiv.org/abs/2606.30410),
142 datasets to 1M rows, temporal/grouped splits) — "TFMs fail to compete … on non-IID,
large-scale, high-dimensional" — and **"A Closer Look at TabPFN v2"**
([2502.17361](https://arxiv.org/abs/2502.17361)) — avg rank 3.97 vs CatBoost 1.89 on 18 sets
with N×d > 1M. TabArena regression-only (v2 era) has CatBoost ahead of TabPFN.

### The gate — decisive, and it needs no TFM installed

`src/run_gate.py` measures the **scale penalty** directly: train the existing LightGBM on the
exact input regime a TFM would get — rows subsampled, features gain-cut — and score it on the
full validation population, against a same-session reference on all rows × all features. Arms
mirror the real context windows:

| arm | regime | which TFM |
|---|---|---|
| e0900 | 50k rows × 665 feat | TabPFN-2.5 |
| e0901 | 100k rows × 665 | TabPFN-3 |
| e0902 | 250k rows × 665 | — |
| e0903 | 1M rows × top-200 | TabPFN-3 (1M×200) / TabICLv2 |
| e0904 | all rows × top-200 | isolates the feature cut |
| e0905 | 100k rows, min_data_in_leaf 20 | small-N-fair config (brackets e0901) |

**Pre-registered reading.** If the 50k/100k penalty is (say) +0.02 RMSLE, any 50k-context TFM
starts −0.02 behind and must recover all of it from inductive bias *before* it reaches §1c's
"decorrelated but weaker" quadrant, and §1f still demands `rho_partial ≥ 0.04` against the
champion for any blend value. The gate also logs each arm's log-correlation with the full model,
so a "decorrelated because weaker" arm is visible immediately. Only if the penalty is *small*
does installing TabICLv2 (BSD-3, cleanest licence; weights `jingang/TabICL`, pre-cache for the
no-internet `apini`) earn a GPU job.

### The honest prior, stated before the numbers

Low. §1s bounds achievable rho at ≤ 0.745 (ICC) / ≤ 0.7254 (test-retest) and the LightGBM on
5M rows already sits near it; a 1–2 % subsample cannot recover an information limit. The gate's
value is a *number* that closes the parked TFM item either way — it converts "outside the
regime" (an assertion) into "the 50k-context handicap is X" (a measurement). Cost: one CPU array
sharing one feature build per fold, no install, no GPU.

### ⚠ Preliminary reading — smoke, one fold (fold 3), 424k-row training set (NOT the decision)

The smoke (`--screen --max-train-anchors 2`, so the pool is only 424,154 rows) already answers
the feasibility question directionally. Reference (all rows × 665) = 1.75106:

```
arm                              Δ vs ref    r_vs_ref
e0900  50k rows  × 665           +0.01239     0.9880     <- TabPFN-2.5 context
e0901  100k rows × 665           +0.00529     0.9932     <- TabPFN-3 context
e0902  250k rows × 665           +0.00070     0.9965
e0903  all(424k) × top-200       -0.00010     0.9978
e0904  all rows  × top-200       -0.00010     0.9978     <- the FEATURE cut alone
e0905  100k rows, min_data 20    +0.00789     0.9922     <- small-N config OVERFITS, worse
```

Two things are already clear and will only sharpen at full scale (1.6–5.2M rows, where 50k is a
**1–3 %** subsample rather than 12 %):

1. **The row-count (context) limit is the binding constraint; the feature limit is not.**
   Cutting to the top-200 features costs **nothing** (−0.00010), while cutting to a 50k-row
   context costs **+0.0124** — already ~22 % of the entire §I6 remaining-prize budget (≤0.055
   RMSLE), before the TFM's own inductive-bias deficit and before the full-scale subsample makes
   it worse. A model whose native limit is a *context* of 50k–100k rows starts far behind, and
   §1f still needs `rho_partial ≥ 0.04` on top of catching up. This corroborates the independent
   benchmarks (BeyondArena, Closer-Look) from our own data.
2. **`min_data_in_leaf=20` at 100k rows is worse, not better** (+0.00789 vs e0901's +0.00529) —
   so the penalty is not a fixable under-fit; small-N just overfits. The bracket closes that loophole.

The full 5-fold confirm will put an exact number on it, but the direction is not in doubt:
**installing a TFM is not worth a GPU job unless the full-scale 50k/100k penalty comes in far
smaller than the smoke — which the scale argument says it will not.** The clean-licence fallback
(TabICLv2) is the only one to consider if that surprise occurs.

### ✅ FULL 5-FOLD CONFIRM (e0900–e0907, job 24098163, 144 min) — REFUTED

Reference = same-session e0049 regime refit, **1.76551** (reproduces e0049 exactly). All arms
share the feature build; cv_mean over the 5 frozen folds:

```
arm                              cv_mean    Δ vs ref   wins   r_vs_ref
e0900  50k rows  × 665           1.78269    +0.01718   0/5    0.9872    <- TabPFN-2.5 context
e0901  100k rows × 665           1.77438    +0.00887   0/5    0.9922    <- TabPFN-3 context
e0902  250k rows × 665           1.77188    +0.00637   0/5    0.9954
e0903  1M rows   × top-200       1.76624    +0.00073   1/5    0.9983    <- TabPFN-3 1M×200 / TabICLv2
e0904  all rows  × top-200       1.76570    +0.00020   0/5    0.9992    <- the FEATURE cut alone
e0905  100k rows, min_data 20    1.77692    +0.01141   0/5    0.9907    <- small-N config: WORSE
```

**Verdict: REFUTED, decisively.** Three findings, each a number rather than an argument:

1. **The row-context limit is the whole penalty; the feature limit is nil.** Cutting to the
   top-200 features (e0904) costs **+0.00020** — inside noise — while a 50k-row context costs
   **+0.01718**, ~31% of the entire §I6 remaining-prize budget (≤0.055 RMSLE) *before* the TFM's
   own inductive-bias deficit. A model whose native limit is a 50k–100k-row context cannot get
   back to par, let alone clear §1f's `rho_partial ≥ 0.04`. This reproduces BeyondArena
   (2606.30410) and "Closer Look at TabPFN v2" (2502.17361) on our own data.
2. **The penalty is not a fixable under-fit.** `min_data_in_leaf 200→20` at 100k rows (e0905)
   is **worse** (+0.01141) than the default (e0901, +0.00887) — small-N just overfits. The
   bracket closes that loophole.
3. **Every arm is §1c's worthless quadrant** — weaker AND highly correlated (r 0.987–0.999),
   so even the decorrelation a TFM would bring (which, per the TFM-ensembling literature, is
   only +0.001 AUC at high dimension) has no member-quality behind it.

The parked TFM item is now closed with a measurement. If a TFM is ever revisited for the
*novelty* section of the jury write-up, use TabICLv2 (BSD-3) or TabDPT (Apache-2.0), never the
non-commercial TabPFN weights on an Ozon-hosted contest.

### ⚠ Licence flag for the write-up

TabPFN-3/2.5/2.6 weights are **non-commercial**; the licence permits "Data Science Competitions
… on established platforms (Kaggle, DrivenData, ChallengeData) or by academic/non-profit
institutions" — an Ozon-hosted contest is **not clearly covered**. TabICLv2 (BSD-3) and TabDPT
(Apache-2.0) are clean. If a TFM ever enters a final submission, use one of the latter.

---

## I18. Transductive pseudo-labelling under the measured test-anchor covariate shift — **RUNNING (e0906–e0907)**

**Status: submitted as two arms of `run_gate.py` (weights 1.0 and 0.3), same job.**

### The gap it addresses

`anchor_drift.py` measured the test cut-off as a **3.9× feature-space outlier** vs the training
cut-offs, and the GBDT is the cut-off-sensitive family (+0.00428 RMSLE per 100 days of gap,
§3b, vs the GRU's +0.00065). Self-training is the standard response to covariate shift: add the
*unlabelled test-regime rows* to the training set with the model's own prediction as a soft
label, so splits are pulled toward the test feature distribution. It is label-free — the
pseudo-label is `E[L|x]`, never `y` — so it cannot leak the target.

The CV analogue, which is what the arms run: add each fold's **validation-anchor rows** with the
reference model's own log prediction as label, weight `w`, refit, re-predict. At submission time
the exact analogue is the 250k test rows at 2026-02-13.

### The honest prior, and the competition precedent

Low-to-moderate. Kaggle precedent (Santander #1, Instant-Gratification) shows pseudo-labelling
helps mainly when it supplies **test-set features/covariance** unavailable at train time — which
is not our case (same users, same window). And §1q shows the model is already at the conditional
mean of its own prediction bins, so a self-consistent pseudo-label may simply reinforce what it
already predicts (a no-op) or over-confidently sharpen it (a small loss). The `w=0.3` arm exists
because a gentle weight is where self-training usually helps if it helps at all. **Decision:**
keep only on ≥4/5 folds and |Δ| > 2σ_noise against the same-session reference (§3.4).

### ✅ FULL 5-FOLD CONFIRM (e0906–e0907, same job) — REFUTED

```
arm                          cv_mean    Δ vs ref   wins   r_vs_ref
e0906  pseudo weight 1.0     1.76577    +0.00026   0/5    0.9993
e0907  pseudo weight 0.3     1.76560    +0.00009   2/5    0.9993
```

**Both nil**, exactly as the honest prior said. The precedent held: pseudo-labelling paid in
those competitions because it injected *test-set feature/covariance structure* the training set
lacked; here the test users and window are the same, so the model's own prediction as a label
adds no information — it reinforces the conditional mean it already fits (§1q). Closed.

**Not the same as the killed "train on contaminated recent anchors" (+0.0019):** that added
*real future labels* from the guard zone; this adds *the model's own predictions* on the
current anchor's feature rows, changing the feature distribution the trees see without adding
any label information. Different mechanism, hence worth the two arms it costs.

---

## I19. Affine-invariant (within-day Pearson) training loss on the GRU — **RUNNING (e0290–e0293)**

**Status: submitted to the frozen folds (job 24098193, `apini`), chained behind a 2-fold smoke.
This revives §I2, which was parked "run only if something else revives the axis" — a competition
win did.**

### The mechanism (unchanged from §I2, now with external evidence)

After the affine calibration §1b applies to every submission, `RMSLE = sd_L·√(1−rho²)` and the
score depends on the prediction **only through rho**. MSE spends on three terms —
`(μ_L−μ_M)² + (sd_L−sd_M)² + 2·sd_L·sd_M·(1−rho)` — and calibration discards the first two for
free. A model that is **regularised and early-stopped** — and the GRU is, hard (e0141 stops at
13–25 epochs; e0106 prices 30 epochs at +0.0204) — spends part of its handful of epochs fitting
a level and spread that will be overwritten. A correlation loss removes that waste. This is an
**optimisation** mechanism, not information (same class as I1), and it is NN-only: a batch-level
statistic is natural for SGD and impossible for boosting's per-sample Newton step.

### What revived it — real competition evidence, not theory

I2 was downgraded after I1's HL-Gauss confirm lost. Two 2025–26 results revive the axis:

- **DRW-Crypto 2025, 1st place** ([kaggle.com/…/drw-solution-1st](https://www.kaggle.com/competitions/drw-crypto-market-prediction/writeups/drw-solution-1st)):
  the winning MLP trained on **0.6·MSE + 0.4·Pearson**, with the explicit statement "solely
  using MSE to train MLP leads to poor correlation"; MSE was fine for their XGBoost (exactly our
  tree-vs-NN split). Metric was a correlation, as ours effectively is post-calibration.
- **CISIR** ([2509.16339](https://arxiv.org/abs/2509.16339), Dec 2025): `L = wMSE + λ·wPCC`,
  λ≈0.5–0.6, on imbalanced tabular regression; its MSE-decomposition ablation shows the PCC term
  **eliminates the sd-mismatch term** (1.199 → 0.000) — precisely the term §1b calibration
  removes for free, i.e. the term MSE wastes epochs on.
- **Ubiquant 3rd** (transformer, per-cross-section PCC loss, 5 seeds) is a third instance.

Counter-evidence, recorded up front: **Algonauts 2025** (metric = Pearson r) found MSE vs MSE−r
single-model nil (0.265 vs 0.264); the gain there was an ensemble effect (+2% over 100 models).
So the honest prior is that the *solo* rho gain is small and the real value, if any, is **blend
decorrelation for a principled reason** (§1c's corrected rule) — a member that disagrees because
it optimised a different geometry, not because it is worse.

### The one design decision that makes this not a naive port

**The correlation must be centred WITHIN each calendar day.** §1r proved that a pooled
correlation across days pays a model for knowing December outranks July, which the competition —
scoring one anchor — does not reward (`run_usercv.py` verified pooled 0.9424 vs within 0.5118 on
synthetic data). `within_day_corr()` centres each day-column on its own masked mean before
pooling the covariance, so the term is invariant to a per-day affine shift — the same freedom
the calibration has, and the correct target. A naive batch-Pearson would have re-introduced the
exact level-fitting §1b removes.

### Design and falsifier

`corr_lambda ∈ {0, 0.3, 0.5, 0.7}`, one change from e0101 (`arch=gru, d128, 2 layers, 12 epochs,
seed 0`), frozen folds. **e0290 (λ=0) is the byte-identical MSE control run this session** —
it must reproduce e0101, which is also the port gate (the additive branch must not perturb the
λ=0 path). Falsifier: no λ beats e0290 by ≥4/5 folds or 2σ_noise (nn_seq σ=0.00020) on CV,
**and** the best λ's OOF has `rho_partial < 0.013` against the champion → kill, and the
loss-geometry axis is closed for the GRU as it is for trees (I1).

### ✅ RESULT (e0290–e0293, job 24114985, 3.8 min on one H200) — REFUTED

**Port gate passed:** e0290 (λ=0) reproduces e0101 **byte-for-byte** — cv 1.76458, folds
`[1.77291, 1.79298, 1.77464, 1.74997, 1.73240]` identical — so the additive `within_day_corr`
branch does not perturb the MSE path.

```
arm        cv_mean   Δ vs control   wins   rho(L,M)   r vs 9-member blend   rho_partial(blend)
e0290 λ=0  1.76458      —            —      0.66162       0.99863            -0.00048
e0291 λ=.3 1.76490   +0.00032 (1.6σ) 0/5    0.66146       0.99842            -0.00082
e0292 λ=.5 1.76488   +0.00030 (1.5σ) 2/5    0.66148       0.99838            +0.00027
e0293 λ=.7 1.76515   +0.00057 (2.8σ) 0/5    0.66136       0.99832            -0.00157
```

**Both halves of the falsifier fired.** (1) Every λ>0 is *worse* on RMSLE, monotonically in λ,
none winning ≥3/5 folds. (2) Against the actual 9-member blend, every arm's `rho_partial` is
≈0 or negative — best is e0292's **+0.00027**, against the 0.02383 bar and the 0.01269 best-ever.
And the arms correlate *more* tightly with the family than a plain GRU (r 0.9983–0.9986 vs
e0101's 0.99827): the corr loss made a slightly worse GRU that is a **closer** copy of the blend,
not a structurally different one. `kill`.

### ⚠ The trap I nearly reported, and the rule it re-confirms

Measured against the **single** control GRU e0290, the arms' `rho_partial` is **+0.0256 to
+0.0268** — *above* §1f's bar and 2× the best member ever. Against the **9-member blend** it
collapses to +0.0003 — a ~90× drop. This is exactly §E1's inflation lesson (a reseeded bag scores
0.086 against one member, 0.0127 against the family), now reproduced on the frozen folds rather
than the 15k screen. **Always run admissibility against the actual family (`src/admissibility.py
<candidate>`), never against one member** — the seq family already sits at r≈0.9975 internally,
so any new GRU looks decorrelated against a single sibling and worthless against the blend.

### What this closes

The loss-geometry axis (IDEAS §0's thesis) is now refuted from **four independent directions**:
HL-Gauss on trees (I1, +0.00124 calibrated, rho_partial −0.00249), the eight-loss magnitude
sweep (§1r, "eight variants inside a band 10× smaller than their offset"), and now the
within-day Pearson loss on the early-stopped GRU (this) — the one testbed where §I2's
"regularised, early-stopped model wastes epochs on level/spread" mechanism should have bitten. It
does not: the GRU at 12 epochs is already near its rho optimum, and re-pointing its loss at the
correlation only adds gradient noise. I2 and I3 (rank loss) share this mechanism and are now
**closed** rather than parked — do not build them without a new reason.

---

## I20. Heckman selection correction to unlock the 3 contaminated months — **parked, high-risk**

**Status: not built. Documented because it is the one item that targets the largest unused asset
— three months of labelled data (2025-11-16 → 2026-02-13) the guard-zone rule forbids — and
because the research says it will most likely fail for a stateable reason.**

The panel is conditioned on activity in the label window (`DATA.md` §4), so the recent anchors'
inclusion depends on the outcome: MNAR selection. IPW is already refuted here (§I10's mechanism
is the same) and is *structurally* wrong for selection-on-the-outcome. The correct family is a
**Heckman correction** ([2607.05806](https://arxiv.org/abs/2607.05806), Jul 2026): a selection
head (probit on inclusion) jointly with the outcome head, coupled by the error correlation ρ, so
excluded units contribute the selection term only. The paper's own result is the warning:
**without a valid instrument** (a variable driving selection but not the outcome) the bias stays
~0.6 of the uncorrected level — and we have no obvious instrument, because everything predicting
"active in the window" also predicts spend. **Prior: ~10%.** The cheap first step, if ever taken,
is to test on OOF whether *any* pre-anchor variable predicts inclusion while being conditionally
independent of the target — if none does, the correction cannot work and the item is closed
without a training run. Not run this session; recorded so it is not re-derived as novel.

---

## I22. Modern tabular NN (RealMLP / TabM) as a decorrelated blend member — the TOP-1 bet — **REFUTED (e0913)**

**Status: RESOLVED NO, 2026-08-25. See the RESULT block at the end of this section.** The design
below is left as written so the pre-registered prediction can be read against the outcome.
`src/run_realmlp.py` (standalone pytabkit) + `scripts/e0917_realmlp.slurm`; the AutoGluon route
(`src/run_ag.py` + `scripts/e0330_tabnn.slurm`, jobs 24116247/24117750/24118751) never ran —
AG's torchvision is broken in `envs/ag`.

### Why this is the strongest untested lever for top-1

The gap to top-1 is ~0.0011 RMSLE ≈ **+0.0005 rho**, well inside the §I6 ceiling — so it is
reachable, and at ~3σ on the public split someone has a real edge. Every marginal axis is
closed. The ONE thing that ever paid here is a **strong model of a different function class that
decorrelates from the blend** (§1c): the gbdt+seq blend was +0.00143, the user-split GRU +0.00048.
The project tried the obvious different-class models and they failed *for known reasons*:
CatBoost/XGBoost are **twins of LightGBM** (r 0.998, §1c), Ridge is decorrelated but **too weak**
(rho 0.622). **What was never tried is a modern tabular *neural net*** — and it occupies a genuinely
empty point in model-space:

- it is **not a tree** (different inductive bias from the GBDT), and
- it is **not a recurrence on the raw sequence** (different input representation from the GRU) —
  it is a feedforward net on the 665 *aggregate* features.

So it can disagree with **both** halves of the blend for a structural reason, which is exactly
the §1f requirement. And it is not weak: **RealMLP** (Holzmüller et al., NeurIPS 2024) *beats*
tuned LightGBM/CatBoost on the TabArena regression benchmark (RealMLP t+e ≈1513 Elo vs LightGBM
t+e ≈1433, CatBoost ≈1417). TabM (ICLR 2025) is the parameter-efficient MLP-ensemble in the same
class. Both are in **AutoGluon 1.6.1**, already installed here.

### Why it is cheap and unblocked

The `envs/ag` torch is **CPU-only** (`2.13.0+cpu`), so RealMLP/TabM run on the `compute`
partition — no GPU, no `pilot_apini` quota fight (the session-long blocker). `run_ag.py` already
drives the frozen folds correctly (per-fold predictor, 30-day embargo on the tuning anchor,
`refit_full` to hand the anchors back, OOF written) — the same honest protocol e0064 used. The
only change is restricting the model set to `{"REALMLP":{},"TABM":{}}` (trees and the
already-refuted TFMs excluded). e0064's AutoGluon was **tree-dominated** (r 0.998 with the GBDT);
forcing NN-only is what makes this a genuinely different member rather than a re-skin of the trees.

### Design and decision rule

Feasibility first: **fold 4 only** (the most test-like anchor), one job, before spending five.
Measure on fold 4: (a) `rho_B = corr(L, RealMLP)` — is it competitive, ~0.66? and (b) the
single-fold partial correlation `corr(L, B | M)` against the 9-member blend M. **Escalate to all
5 folds only if fold-4 rho_partial > ~0.02 and rho_B > ~0.655** (a member that is both decorrelated
and strong). Then the honest 5-fold admissibility (`src/admissibility.py e0330`) decides against
the 0.02383 bar. Kill if it is a weak twin like every other member — but unlike the tree families,
its priors for *both* strength and decorrelation are backed by an external benchmark, so it is the
best-justified shot at the +0.0005 rho top-1 needs.

### Honest prior

Guarded-moderate — higher than anything else left. Against: nothing this project built ever cleared
rho_partial 0.013, the GRU already supplies the NN decorrelation, and RealMLP shares the 665
features with the GBDT so some correlation is guaranteed. For: it is the only strong model of a
third function class, its decorrelation-from-trees is architecturally expected (Grinsztajn: NN and
GBDT have different inductive biases — the whole reason trees usually win is also why they
disagree), and the top-1 gap is small enough that even a modest new `rho_partial` could close it.

### ✅ RESULT (e0913, fold 4, job 24152700, 20 h 35 m CPU) — REFUTED. Strength arrived; decorrelation did not.

**The run finished on its third attempt and nobody had read it.** e0910/e0911 FAILED on AG's
torchvision; e0913 (1.2 M rows, 6 h cap) and e0917 (500 k rows, 5 h cap) both TIMED OUT; the same
500 k config resubmitted with a **24 h cap completed in 20 h 35 m**, ending 2026-08-25T08:28:21.
§I29 below and `SESSION_2026-08-24.md` §8 recorded it as *"timed out twice / CLOSED"* because the
result landed after they were written. Both are corrected.

Fold 4, aligned 223,578-user intersection with the usercv OOF (joined on `user_id`). The champion's
fold-4 `rho_M` reproduces §1z-E's recorded **0.675020** exactly, so the harness is validated against
an independent number before use:

```
RealMLP_TD   rho_B = 0.671601      r vs the FULL champion = 0.995356
             excess e = -0.000284  rho_partial = -0.003996
bar for +0.000633 rho at that r    = 0.674699   ->  MARGIN -0.003098  (0.101x the requirement)
optimal in-sample weight inside the champion = 0.000
```

> **The pre-registered bet was right about strength and wrong about decorrelation.** This section
> argued that "a model at GBDT strength (rho ~0.67) *at* r ~0.96 gives e ≈ +0.022 ≫ the bar",
> extrapolating NN_TORCH's r = 0.96. The strength arrived — **0.6716 is the best non-tree,
> non-recurrent member the project has ever built, within 0.0021 of the gbdt half's 0.673636** — and
> `r` came in at **0.9954, not 0.96**. **NN_TORCH's decorrelation was a property of being WEAK
> (rho 0.647), not of being a neural net.** Extrapolating an `r` measured on a weak model to a strong
> one was the error, and it is the same shape as §1c's original mistake ("weak but decorrelated
> members help").

Slot-wise it is most redundant with the model that shares its feature matrix, exactly as expected:
`corr` 0.9965 with the gbdt half, 0.9949 with the seq half, 0.9929 with the usercv GRU.

**Screen tier, and the kill is still safe.** One fold and a 500 k-row subsample, so §E1 says this
`rho_partial` is *inflated* — and it is already negative at the inflated scale, so pooling can only
make it worse. No 5-fold confirm is needed to reject.

**The one honest caveat.** 500 k training rows against the GBDT's ~5.2 M. §I17's own scale-penalty
gate prices that handicap at roughly **+0.002 rho** if lifted to full data (a 250 k context cost
LightGBM +0.00637 RMSLE = −0.0030 rho; 1 M × top-200 cost +0.00073 = −0.0003), against the
**+0.0031 needed** — and more data normally raises `r` too, which raises the bar. So a full-data arm
most likely still misses, but it is the closest call the project has had. The run was **CPU-only**
(log: `GPU available: False`), which is the 20 h; RealMLP is Lightning/PyTorch, so an andrena port
would make a full-data 5-fold run a few GPU-hours. **Pre-registered falsifier if it is ever run:
does `rho_B` reach 0.6747 at `r ≤ 0.9954` on fold 4?** Prior ~10–15%, and there is no
`predict_realmlp.py`, so a submission needs extra engineering on top.

**What it closes.** The last untested *function class* for a blend member. Placed on the fold-4
frontier it sits 4th of 11 and squarely on the same monotone curve as everything else — a 12th point
on §1z-E's closure, now spanning trees, in-context learners, recurrences, TS foundation models and a
strong tabular neural net. ⚠ Every number here is **fold-4 scale** (`rho_M` 0.675020) and may not be
compared against §1z-E's **pooled** bar (`rho_M` 0.66342).

### The round-3 pipeline, sequenced behind I22's read (so compute is not spent on speculation)

I22 is the cheapest strong-NN test and its result gates the rest. If RealMLP shows *any*
decorrelation from the blend (even sub-bar), the NN-on-features direction is alive and these
follow; if it is a pure twin, only the *different-representation* bet (I23) survives.

- **I23 — CoLES / self-supervised sequence pretraining → frozen embedding → blend member —
  PARK, low prior (~0.005–0.015 rho_partial), researched 2026-08-23.** The project's own single
  unbuilt `P0`. Contrastive/generative encoder → frozen embedding → LightGBM, leakage-safe
  (pretrain on data ≤ earliest anchor, freeze, reuse). Cheap and offline-capable
  (`pip install pytorch-lifestream`, trains from scratch on our data, ~minutes–2h, CPU-feasible).
  **But the honest verdict is a correlated twin, not a decorrelated member,** for a measured
  mechanism: the supervised GRU already reads the ~13 channels to within **0.0017 rho** of the
  GBDT, so the sequence information is nearly exhausted; SSL only *re-represents* existing
  structure, and a CoLES-on-LGBM readout would correlate with **both** the GBDT (CoLES's inductive
  bias is aggregate/count content, which the 665 features already hold) **and** the banked GRU
  (shared sequence) — pushing it *more* correlated with the blend, not less. The relevant evidence
  cuts against us where it is closest: **NPPR** ([2401.01641](https://arxiv.org/abs/2401.01641))
  is literally our task (1-month forward spend, MSLE) and SSL embeddings cut MSLE ~2.7% *on top of
  features* — but that is standalone strength (never vs/with a GBDT), and ~2.7% is roughly the
  GRU's own already-banked contribution; **Abacus** ([2512.16581](https://arxiv.org/abs/2512.16581))
  wins with a *count-histogram* pretext, exactly what our features already encode; **EBES**
  confirms order matters little for an aggregate target. Expected rho_partial squarely in the
  GRU's already-banked 0.005–0.015 range, short of the 0.024 bar. Worth **one** cheap CoLES+LGBM
  graveyard run if a GPU is idle, but not ahead of confirming I22.
- **I24 — expand the NN member set (FT-Transformer, TabM tuned, a dedicated RealMLP *classifier*).**
  Cheap once I22's harness exists (all in AG 1.6.1, CPU). The classifier arm doubles as the
  **classification-ceiling probe**: is AUC 0.848 the data limit or the method limit? e0160–e0162
  hit 0.844 with LightGBM-binary / GRU-BCE; a RealMLP classifier is the one untested family, and
  §I6 says the buy-flag has 0.069 rho of headroom — the single biggest lever if it is real.
- **Standing note on the ceiling.** All of these are bounded by §I6/§1s: rho ≤ 0.725 (test-retest)
  / 0.745 (ICC), champion at 0.704, top-1 at ~0.7043. There is room, but it is thin, so the only
  thing that reaches it is genuine decorrelation at strength — which is exactly what I22–I24 test
  and nothing before them cleared.

---

## I28. Pretrained time-series foundation model as a frozen embedding → blend member — 2026-08-24

**Status: designed, not run. From a 5-paper batch the user supplied, all on TS foundation models
as feature extractors. This is the ONE remaining lever that injects EXTERNAL information.**

### The idea and why it differs from everything tried

Take a TS foundation model **pretrained on massive external corpora** (Timer 2402.02368 = GPT on
1B time points; Chronos/TimesFM/Moment/TiRex; TimelyGPT 2312.00817 = xPos for extrapolation),
run each user's 409-day daily GMV/activity series through the **frozen** encoder, extract the
embedding, and feed it to LightGBM as a blend member — the exact recipe of Auer et al.
(2510.26777): "pretrained forecasting models are strong zero-shot feature extractors," frozen FM
→ simple head, beats DTW/1NN on UCR/UEA. This is **different in kind** from:
- I17 (TabPFN/TabICL): those are *tabular* FMs on the 665-feature matrix; this is a *time-series*
  FM on the raw daily sequence.
- I23 (CoLES) and the supervised GRU: those learn *from scratch on our data* (no outside info);
  a pretrained TS-FM carries **temporal priors learned from a billion external points** — the only
  source of genuinely new (external) information the project has not tapped.

### Why the prior is nonetheless LOW — and the supplied papers say so themselves

- **Paper 3 (2507.02907)**: bigger pretrained TS-FMs do **not** reliably beat small specialised
  models. Our supervised GRU is already a strong specialist (within 0.0017 rho of the GBDT), so a
  TS-FM is unlikely to be more *accurate* standalone.
- **Domain gap is severe.** These FMs are pretrained on clean, dense, periodic series (energy,
  biosignals, traffic). Ours is **sparse (30% density), anonymised, aggregate-target (30-day sum),
  order-invariant** (EBES: order barely matters here). The pretrained temporal priors (trend,
  periodicity) may not align with an anonymised e-commerce spend panel.
- **Same collapse mechanism as I23/I19/TabICLv2/NN_TORCH:** fed to LightGBM to predict OUR target
  from the SAME sequence, the embedding's fit collapses onto the target-predictive subspace the
  GRU already spans → likely correlated with the blend, not decorrelated. The one way it escapes:
  the *external* priors extract temporal features our in-data models never learned. That is the
  bet, and it is the strongest conceptually-new one left.

### The cheapest decisive test (if run)

Frozen small FM (Chronos-bolt-small / Moment-small / TimesFM — pip-installable, weights on HF,
pre-cache on `compute` since `apini` has no internet), extract the last-hidden-state embedding per
user at the fold-4 anchor (data ≤ anchor, leakage-safe by construction — the FM never saw our
target), concat → LightGBM, score rho and **rho_partial vs the champion blend** (`admissibility.py`).
Escalate to 5 folds only if rho_partial > ~0.02 at rho ≳ 0.66. Build cost: one fresh env + weight
cache + an extraction pass over 250k×409 (GPU ideal, CPU-feasible on a subsample). Bigger than the
OOF tests, smaller than a from-scratch pretrain. **Recommendation: worth one feasibility run given
it is the last external-information lever, but with a clear-eyed low prior — most likely a
correlated twin like CoLES/TabICLv2, in which case the FM-embedding direction closes with a number.**

### ✅ RESULT (fold 4, MOMENT-1-small) — the FIRST positive-excess candidate of the session

Three heads on the frozen MOMENT embedding, all measured vs the 9-member blend AND the full
champion (e0162 = gbdt+seq+usercv-GRU):

```
member                           rho     r_vs_champ   rho_partial(9-blend)   rho_partial(CHAMPION)
e0914  MOMENT -> LightGBM        0.6489    0.9616        +0.0065               +0.0009
e0915  GRU over per-patch tokens 0.6513    0.9635        +0.0112               +0.0060   <- best
e0916  raw-GRU + FM channels     0.6484    0.9604        +0.0030               +0.0015
(ref) usercv GRU already in champ                        +0.0317               --
```

**This is the first thing all session with *positive* rho_partial against the FULL champion** —
TabICLv2 (−0.0005), NN_TORCH (−0.004), NCL, CoLES, causal-forces were all zero/negative. External
TS-FM temporal priors genuinely inject a *decorrelated* signal. **e0915 (learned per-patch GRU
aggregation) is the standout**, clearly beating the mean-pool (e0914) and FM-as-channels (e0916) —
the sequence of per-patch tokens carries structure the mean-pool discards.

**But three caveats keep it below top-1 relevance:** (1) **single fold** — rho_partial inflates
vs the pooled 5-fold (§E1); the honest number is lower. (2) The member is **weak** (rho 0.651 vs
0.675), so ~half its decorrelation is **already subsumed by the usercv GRU** (rho_partial falls
+0.0112 → +0.0060 when the champion's GRU is included). (3) Even at face value +0.006 is worth
**~−0.00003 RMSLE** in the blend — far below the **+0.0005 rho (rho_partial ≈ 0.04)** top-1 needs.

**Verdict: the one live lead.** Escalate e0915 to **5 folds + LOFO admissibility vs the champion**
(kills the single-fold inflation), and strengthen it (**MOMENT-base**, **multi-channel** not just
GMV) to see if rho_partial can climb toward relevance. Even if it confirms positive-but-small, it
is the first genuinely new (external-information) decorrelated member the project has produced.
RealMLP (e0913) TIMED OUT here (RealMLP_TD 256 epochs on 1.2M CPU rows > 6h) — needs a lighter config.
*(Superseded 2026-08-25: the 500k config finished on a 24h cap and is **REFUTED**, rho_B 0.671601 at
r 0.995356 = 0.101x the bar. See §I22's RESULT block.)*

### ✅ 5-FOLD CONFIRM (e0915, array 24145189) — REFUTED. Fold-4 was single-fold inflation.

Pooled admissibility (`src/admissibility.py e0915`, all 5 frozen folds, vs the 9-member family):
`rho_B 0.6354, r 0.9592, excess −0.00009, **rho_partial −0.00042**` — negative, blend gain ≈ 0.
Per-fold rho_partial vs the family: **[−0.0037, −0.0084, −0.0027, +0.0090, +0.0112]** — 3 of 5
NEGATIVE; the +0.0112 that launched the direction is fold 4 ALONE (the §E1 single-fold trap, now
confirmed on the frozen folds for the Nth time). Pooled, e0915 is a weak twin (r 0.96 at rho 0.635),
not a decorrelated member. **`kill`.** The frozen-MOMENT-per-patch-GRU lead is dead as instantiated.
This does NOT auto-close §I29 (OTHER FMs): 2510.26777 ranks MOMENT the *weakest* extractor and e0915
carries no abs-scale stats — but the bar is now exact: the frontier needs **rho_B ≥ 0.641 at r 0.96**
to buy +0.0005, and MOMENT hit 0.635, missing by 0.006. A stronger backbone + the scale-aug is the
only way §I29 clears it; prior lowered accordingly.

> ⚠ **CORRECTED 2026-08-24 (e0391).** The 0.641 above is measured against the **nine-member
> seq+gbdt family**, which EXCLUDES the usercv slot that is actually in the champion — the §E1
> substitution this file warns about everywhere else. Against the **full champion**
> (`rho_M = 0.66342` on the folds) the bar at r 0.96 is **0.6441**, and the 9-blend reading
> flatters a candidate by ~0.001. Judge every arm against the full champion, using this table.
>
> **THE EXACT BAR — `rho_B` needed at each `r` vs the FULL champion, on the fold scale**
> (`rho_M = 0.66342`; `scripts`-free, recomputable from `scratchpad/tsfm_bar.py` on any OOF):
>
> ```
>   r vs champion    for +0.0005 rho    for +0.0010 rho
>          0.900          0.6083             0.6130
>          0.940          0.6324             0.6360
>          0.960          0.6441             0.6471
>          0.970          0.6498             0.6524
>          0.980          0.6553             0.6574
>          0.990          0.6604             0.6619
>          0.995          0.6627             0.6637
> ```
>
> **Where the arms actually sit (pooled, all 5 folds, 1,062,003 common keys):**
>
> ```
>   candidate   rho_B     r vs CHAMP   bar@that r   short by   rho_partial
>   e0915      0.6355       0.9593       0.6437      -0.0082     -0.00415
>   e0919      0.6499       0.9799       0.6552      -0.0054     -0.00151   <- closest ever
> ```
>
> **A SECOND ERROR, mine, corrected here.** I earlier told the user these arms were "short by
> ~0.033" and that the direction was ~20× too small. That used §1z-A's frontier constant
> **0.70378**, which is the **TEST-ANCHOR** rho; on the frozen folds the champion sits at
> **0.66342**. The two scales differ by 0.040, so the test constant overstates a fold-measured
> candidate's requirement by roughly that much. **The real gap is 0.0053, not 0.033** — about 6×
> smaller, and it makes this direction materially more live than I claimed. *Never mix the
> fold-scale and test-scale rho constants; §1b's algebra is anchored on the test moments and
> §1f's admissibility on whichever population the OOF was measured in.*
>
> **Combining the two FM members does NOT clear it (e0391).** `corr(e0915, e0919) = 0.96942` —
> unusually decorrelated for this project — yet no weight clears: the margin peaks at
> `0.1·e0915 + 0.9·e0919` (−0.0053, i.e. e0919 alone) and falls monotonically toward e0915.
> Averaging raises `r` as fast as it raises `rho`, which is §1c's law and the mechanism e0303
> measured as a sign inversion on the leaderboard. **A single arm must clear on its own.**
>
> **AND THE PER-ANCHOR TREND, MEASURED (e0392) — it is real, it replicates, and it delivers about
> HALF of what is needed.** §I28's correction downgraded the e0915 kill to PARK because
> `rho_partial` rises with anchor date and the test anchor has the longest histories. Measured as
> the *margin* against the bar, per fold:
>
> ```
>   fold / anchor      0        1        2        3        4      corr(idx, margin)
>   e0915          -0.0080  -0.0092  -0.0083  -0.0064  -0.0058       +0.808
>   e0919          -0.0062  -0.0071  -0.0051  -0.0031  -0.0042       +0.808
> ```
>
> The trend is genuine and **replicates across two independent FM members**. But over the folds'
> full 120-day span it buys only **+0.0022 / +0.0020**. The test anchor is another ~120 days out,
> so §I28's own linear extrapolation yields ~+0.0020 more against a fold-4 deficit of −0.0042 →
> **extrapolated test margin ≈ −0.0022, still negative**; reaching zero needs ~250 further days,
> an anchor in mid-2026.
>
> **The mechanism deflates the story.** `rho_M` itself climbs 0.6553 → 0.6750 across the folds —
> *later anchors are more predictable for everyone* — and the FM's `rho_B` rises only marginally
> faster (+0.0215 vs the champion's +0.0197). The trend is mostly the task getting easier, not the
> FM getting relatively better. **PARK stands, but the "it will clear at the test anchor" hope does
> not.**
>
> **The concrete target for any LoRA / backbone arm:** at its current `r`, **+0.004 to +0.005 of
> `rho_B` over e0919** — or **−0.02 of `r` at constant `rho_B`**, which the bar table shows is the
> cheaper direction and is the one full fine-tuning destroys.

### ✅ TTM — the no-attention backbone, aimed at the cheap axis (e0394–e0396, fold 4)

The decorrelation axis was tested with the most architecturally orthogonal option on the shortlist:
**TinyTimeMixer, an MLP-Mixer with no attention at all** (385k params, via `transformers`'
`PatchTSMixerModel` — `granite-tsfm` is *not* required, see `src/ttm_gate.py`), against a field of
transformers (MOMENT = T5 encoder, Chronos-Bolt = T5 enc-dec).

```
arm                       rho_B   r vs CHAMP     bar    MARGIN   rho_partial
e0396 MOMENT (port gate) 0.65138    0.96369   0.65745  -0.00607    +0.00440
e0394 TTM                0.64230    0.95035   0.64959  -0.00729    +0.00345
e0395 TTM + scale-stats  0.65294    0.96717   0.65947  -0.00653    +0.00042
```

**The hypothesis worked mechanically and still did not pay.** TTM reached **r = 0.95035**, the most
decorrelated TS-FM member the project has produced. **The exchange rate is the finding:** dropping
`r` by 0.0133 lowered the bar by 0.00786, while `rho_B` fell 0.00908 — the strength loss *exceeds*
the bar relief (ratio ~0.87), so the margin does not improve. This is §1c's law measured on the
**architecture axis** for TS-FMs, and it says the frontier is close but on the wrong side.

**The scale augmentation splits, and §I29 above was half right.** It works on strength
(`rho_B` +0.01064 — recovering the absolute scale instance-normalisation strips *is* worth
something for a 30-day sum) but raises `r` more (+0.01682), collapsing `rho_partial` to +0.00042.
Mechanism: absolute scale is exactly what the GBDT and GRU halves already encode, so adding it
moves the FM member **toward** the family. Load-bearing for strength, counterproductive for the blend.

> ⚠ **THE HARNESS IS NOT DETERMINISTIC, and this bounds every fold-4 TS-FM comparison here.**
> The port gate (e0396) re-ran e0915's exact config: `rho` reproduced to **2e-05** (0.651283 vs
> 0.651262) but `corr(log preds) = 0.98997`, `max|dlog| = 3.28`. `run_tsfm_gru.py` set
> `torch.manual_seed` and never `cudnn.deterministic`. Run-to-run noise on `rho_partial` is
> **~0.0014** — the same size as the gaps between arms — so the TTM-vs-MOMENT margin difference
> (0.00122) is **inside run noise**. The honest claim is "neither clears", NOT "TTM is worse than
> MOMENT". A `--deterministic` flag now exists; it is OFF by default so e0394–e0397 stay
> reproducible, and should be ON for every new single-fold comparison.

**Also fixed:** the in-job champion reference assumed the usercv OOF shares a row order with the
seq/gbdt OOF. It does not (238,847 vs 225,431 fold-4 rows) — it now joins on `user_id`, and all
numbers above are on the aligned 223,578-user population.

**⚠ CORRECTION (per-anchor trend) — downgrade kill → PARK.** The per-fold rho_partial is NOT noise:
it rises MONOTONICALLY with anchor date = available history length (`scratchpad/e0915_by_anchor.py`):
fold0 −0.0037 (2025-06-18) → fold2 −0.0027 → fold3 +0.0090 → fold4 +0.0112 (2025-10-16); rho_B climbs
0.630→0.622→0.634→0.648→0.651. The FM reads history UP TO the anchor, so early folds (short histories)
starve it and late folds feed it — and the REAL test anchor (2026-02-13) has the LONGEST histories of
all. So the pooled −0.0004 AVERAGES the disagreement §3.2 says to FLAG, not a true saturation. A (shaky,
4-month) linear extrapolation puts test-anchor rho_B ≈ 0.68, excess ≈ +0.006, rho_partial ≈ +0.03 vs the
9-blend (≈half that vs the stronger champion). **Do not trust the extrapolation — MEASURE it:** a
STANDALONE affine-calibrated e0915 submission at the test anchor backs out its exact test-time rho via
§1b (a champion+e0915 blend submission is unreadable — tiny weight ≪ LB noise). Then strengthen (better
backbone + the abs-scale-stats aug e0915 lacks). The direction is alive; the pooled kill was a fold-mix
artefact.

---

## I27. Causal-forces damping (Armstrong & Collopy 1993) — REFUTED on OOF — 2026-08-24

**Status: DONE, `scripts/causal_forces.py`, no cluster (OOF only).** From the 5 causal/extrapolation
papers the user supplied; this was the one concrete, cheap, tabular-applicable rule.

**The idea.** The geo3 baseline `B = log1p(y_naive)` is the mean-reverting / no-trend ("regressing
force") prediction; the model's deviation `d = M − B` is the trend it extrapolates beyond
mean-reversion. Armstrong's rule: damp `d` where the force is contrary/regressing — and for a
regressing force, up-spikes (`d>0`) should be damped MORE than dips. Tested as LOFO-fitted damping
(params on 4 folds, applied to the 5th), scored on rho.

**Result — null across all three variants:**

```
rule                         mean rho    Δ vs M (LOFO)
no-op (M)                    0.66066       —
global damp   g*=1.010       0.66067      +0.00001
asymmetric    g+=1.00 g-=1.02 0.66066     -0.00000
tail damp    |d|>p90         0.66066      +0.00000
shuffled-dev (control)      0.57027      -0.09039   <- test has full power
```

The fitted global factor `g* = 1.010 ≈ 1` — the model's deviation from mean-reversion is *correctly
scaled, not over-extrapolated* (if anything, marginally under). Asymmetric `g+ ≈ g- ≈ 1`: no
spike-over-extrapolation for the regressing force to correct. The shuffled-dev control craters to
−0.090, so a real effect would have been visible. **Verdict: refuted** — and it is the same finding
as §1q (E[L|M] ≈ M to 0.0010) and §1t (per-segment slope ≈ 1) from a third angle: the model is
already conditionally calibrated, so there is no contrary trend left to damp.

**Caveat (why it does not fully close the causal-extrapolation *theme*):** this is measured on the
2025 fold anchors, where the model is calibrated. At the out-of-support test anchor it *could*
over-extrapolate in a way the folds do not reveal — but that is unmeasurable (no test ground
truth), and the only "force" that differs there (the Feb→Mar seasonal lift) is already measured to
*hurt* RMSLE (DATA.md §8.4, e0142). A panel-based version using the true recent-30d-vs-long spike
(rather than the `M−B` proxy) is available but low-value: the model already ingests those features
and §1q/§1t/this all agree it uses them calibratedly.

---

## I26. Domain / competition-specific search — no exploitable trick, one correction — 2026-08-23

**Status: DONE. A targeted search (EN+RU) for anything specific to E-CUP 2026 or near-identical
spend-regression tasks. Net: it confirms the project's thoroughness; no new exploitable lever.**

- **E-CUP 2026 Task 3 public material: none.** Competition is live (Aug 10–31 2026, finals
  Sep 12–13; ~572 teams, ~3,450 subs; ods.ai/tracks/e-cup-2026-competitions,
  habr.com/ru/companies/ozontech/news/1067838/). No Task-3 writeups/repos exist yet (only a
  Task-1 product-matching repo, irrelevant). The "tokenize behaviour + NN + BTYD" hint is not in
  any indexed organiser material — it came from the Telegram/webinar. Revisit after Sep 13.
- **Behaviour-tokenisation → transformer on an AGGREGATE-spend target: an unproven open bet.**
  No published recipe hits discrete-tokens + spend-regression + beats-GBDT-with-numbers. The
  strongest transferable precedents are *classification*: nuFormer (2507.23267) beat LightGBM
  +1.25 vs +0.97% AUC; Alfa-Bank ANNA 2.0 beat CatBoost +3–8 ppt AUC (habr). This CONFIRMS §I23:
  the representation direction is unproven for our order-invariant sum, not a known win.
- **Leak / metadata probes (Santander-style): closed by construction.** The agent flagged two —
  (a) lagged-copy feature columns, (b) row-order/user_id reconstructing the split — as "maybe
  unrun." Both are covered/inapplicable: our features are *semantic* (DATA.md §2 verified column
  identities), not raw anonymised columns, so the Santander feature=lagged-target leak cannot
  exist; the target is strictly future of every feature by construction; and `DATA.md` §10 already
  ran the leakage register (corr(user_id,target)=0.0001, rank corr −0.0158, ids non-contiguous,
  no post-cutoff rows). `user_id` as a covariate is separately dead (§I5). Nothing to probe.
- **Elo-style two-stage with a tunable spike count** (Elo 2019 winner: 1% artificial −33.2
  outlier + a binary is-outlier head, tune injection count): does not transfer. Our 45% mass is
  genuine non-buyers, not a 1% artificial spike; zeroing (every threshold) and the hurdle (e0010,
  e0222) are measured dead; and routing to a tail requires the gate §I15 proved does not exist.

### ⚠ Correction to the agent's Thread-4 "free tie-breaker lever" (it is NOT free)

The agent claimed `ŷ′ = e^c·(1+ŷ)^a − 1` (affine in log1p, so `rho`-invariant) gives two *free*
knobs to shape Gini (via `a`) and the total-GMV sum (via `c`) without touching RMSLE. **This is
wrong for the submitted metric.** RMSLE `= sd_L·√(1−rho²)` only *at the optimal affine
calibration* `(a*, c*)`; the champion already sits there, so moving `a` or `c` off it **increases
RMSLE** — it is a trade-off, not a free lever. `DATA.md` §8.4 measured it directly: forcing the
aggregate sum costs **+0.163 RMSLE**, and Gini is scale-invariant so only the RMSLE-costly
exponent moves it. Since the finals gate on private RMSLE rank (top-15 advance), shaping the
tie-breakers on the *submission* is a net loss. The only legitimate use is the **two-head
write-up** (PAPERS §2.2): report a separately Duan-corrected aggregate `× exp(σ²/2)` in the repo
to demonstrate understanding, leaving the submitted RMSLE vector untouched. Jury material, not score.

**Net: the domain search produced no new exploitable idea — it independently re-confirms the
saturation and closes the leak/tokenisation/two-stage threads a future session might otherwise
re-open.**

---

## I25. Negative correlation learning — manufacture a decorrelated member — **REFUTED by argument**

**Status: closed by an algebraic reduction (verified in sympy), no run needed.** The sharpest
idea of round 3 and the one that would, if it worked, directly break the plateau: instead of
*hoping* a new model decorrelates, *train* it to be anti-correlated with the frozen blend while
staying accurate.

### The reduction that kills it

Liu–Yao NCL adds a penalty `p_i = (f_i − f_ens)·Σ_{j≠i}(f_j − f_ens)`. Because `Σ_j(f_j − f_ens)
= 0`, this equals `−(f_i − f_ens)²`, so with the existing blend `M` **fixed** and one new member
`f`, the objective is `½(f − y)² − λ(f − M)²`. Its minimiser is exactly

```
f*  =  (y − 2λ·M) / (1 − 2λ)  ≈  y + 2λ·(y − M)      (verified: sympy, small-λ expansion)
```

**So NCL is not a new direction — it is a one-parameter slider along the line between two points
this project already measured as dead:**

- `λ → 0`: `f* → y` — just another model of the target → the **r ≥ 0.994 twin** (every GBDT/NN
  family tried).
- `λ → 0.5`: `f*` → dominated by the residual `(y − M)` — the **residual-GBDT** (§1q,
  best_iteration = 4, corr 0.0018 with the true residual, worse than zero), and the denominator
  blows up (unstable).

Between them it fits `y` plus an *amplified* residual. But §1q/§1s measured the blend's residual
as **essentially unlearnable** (autocorr 0.003, no feature clears incremental R² 0.0002), so
amplifying it just amplifies noise: small λ stays a twin, larger λ fits noise and rho collapses.
There is no λ that produces decorrelation-at-strength, because the strength (learnable residual)
is not there.

### Supporting evidence, all pointing the same way

- **Learner collusion** (Abe et al., *Joint Training of Deep Ensembles Fails Due to Learner
  Collusion*, NeurIPS 2023, [2301.11323](https://arxiv.org/abs/2301.11323)): the diversity term
  can be inflated arbitrarily with **no test benefit** — members scale/offset to game the
  covariance while individual quality degrades and the ensemble cancels out. Explicit diversity
  losses do not reliably beat independently-trained members, *especially in high-capacity models*.
- **Unified Theory of Diversity** (JMLR 24, 2023): diversity is not a free parameter — it trades
  against member accuracy; the ambiguity decomposition is an in-training identity with **no OOS
  guarantee**. Exactly the wall this project hit empirically.
- **Snapshot Ensembles / FGE / cSGLD**: produce members *more* correlated than independent
  reseeds (cyclic snapshots ≈ 2 independent models; FGE members lie on one low-loss path). Since
  plain reseeding is already worthless in this blend (§1m), these are strictly worse.
- **No competition precedent**: no Kaggle write-up trains a member with a decorrelation objective
  against a fixed model to break a plateau. Standard diversity comes from *independent*
  architectures/features/seeds — the r ≥ 0.994 route already exhausted here.

### Verdict

**Closed.** The one non-reducible experiment (NCL on the GRU with fixed blend, λ≈0.05–0.15,
scored as rho_partial) is not worth a GPU slot: the reduction is airtight and its two endpoints
are both measured dead. This is the strongest available statement that *manufacturing*
decorrelation does not work here — the constraint is not the training objective, it is that the
blend already captures essentially all the learnable signal (§1s), so no member can disagree with
it *and* be right.

---

## I21. Targeted feature neutralisation — the one non-monotone post-processing lever — **RUNNING (e0294)**

**Status: submitted to `compute` (job 24115005). No training — pure linear algebra on
`oof/e0049.parquet` + the per-fold feature matrix. `src/run_neutralize.py`.**

### Why it is not killed by the calibration algebra (the whole point)

§1b/§1q/§1t closed *calibration*: any function of the prediction `M` alone leaves rho unchanged,
because rho is invariant to monotone maps of `M` (isotonic scored −0.00006 vs a +0.000000 no-op
control, and §1r proved `E[L|M]` is linear so even a curve has nothing to fit). Feature
neutralisation is the first post-processing operation this project has considered that is **not a
function of `M` alone**:

```
M' = M − p · N · (N⁺ M),    N = [ z-scored selected feature columns | 1 ],    p ∈ [0, 1]
```

It subtracts `M`'s linear projection onto a chosen set of **feature** columns, re-ranking users
by information outside `M` — exactly the freedom a monotone map lacks. It uses **no labels** (the
projection is of the prediction onto features, computable at test time byte-identically), so it
is leakage-safe and applies unchanged to the 2026-02-13 submission.

### Why it might help HERE, and the falsifiable signature

The mechanism (Numerai; [arXiv 2303.16117](https://arxiv.org/abs/2303.16117), Table 3, OOS
2015–2022) **trades mean correlation for correlation stability under shift**: full projection
cost ~22% of mean corr but cut corr volatility ~35% and raised Sharpe ~20% out-of-sample. Our
test anchor is a measured **3.9× feature-space outlier** driven by the lifetime/365-day features
(`anchor_drift.py`, e0056/e0057), and the GBDT is the cut-off-sensitive family (+0.00428 RMSLE
per 100 days of gap, §3b). So neutralising `M` against **only those drifting features** should,
if the mechanism is real, **raise rho on the most-shifted fold (4, 2025-10-16) while lowering it
on the early folds** — a per-fold divergence no in-sample metric can fake. That divergence, not
the mean, is the test; the real test anchor is more shifted still, so a positive fold-4 signal is
the encouraging read.

### Design, controls, falsifier

Arms: feature set ∈ {**drift** (`_total$`/`^tenure`/`_365$`), **all 665**, **random matched-count
control**} × `p ∈ {0, 0.2, 0.35, 0.5, 0.7, 1.0}`. The random-column control is the e0214
discipline: if neutralising against random gaussians moves rho as much as against the drift
features, the effect is projection-removes-variance, not drift-specific (expected ~0, since
`lstsq(random, M) ≈ 0`). Each fold is a single anchor, so per-fold rho **is** the within-anchor
rho the metric is (§1r) — no pooling confound. **Falsifier:** if the LOFO-honest `p` (chosen on
4 folds, applied to the 5th) gives Δrho ≤ 0 on the held-out fold **and** fold 4 does not rise as
early folds fall, the lever is dead and the last non-monotone post-processing axis is closed.

### Honest prior

Low-to-moderate. The strong prior against: the GBDT already uses those features optimally
in-sample, so `M` is already "de-neutralised" and removing the projection removes signal — this
is why the mean effect will likely be negative. The prior *for*: the effect the paper reports is
purely out-of-sample, and this project has never separated in-sample rho from shift-robust rho
with a non-monotone operator; the CV−LB gap (+0.11) is precisely the room where it could live.
Cost is minutes of CPU on data already on disk, so it is worth the exact number either way.

### ✅ RESULT (e0294, 13.5 min CPU, 117 drift columns) — REFUTED

Per-fold rho, `M` = e0049 OOF prediction, by neutralisation proportion `p` (fold 4 = most
test-like; the `f4−f0` column is the within-run fold spread, NOT the effect):

```
[drift, 117 cols]   f0       f1       f2       f3       f4      mean
   p=0.00        0.65301  0.64774  0.65958  0.66991  0.67306  0.66066
   p=0.35        0.64980  0.64377  0.65466  0.66516  0.66847  0.65637   (all folds DOWN)
   p=0.70        0.60725  0.59502  0.59907  0.60987  0.61304  0.60485
   p=1.00        0.11759  0.12652  0.13643  0.13835  0.14243  0.13226
[random, 117 cols, control]  every p within 0.0002 of p=0  -> machinery is sound
LOFO-honest p: drift Δ +0.00000, all Δ +0.00000  (p=0 chosen on all 5 folds)
```

**The falsifier fired exactly as pre-registered.** The signature that would have kept the idea
alive — rho *rising* on the shifted fold 4 while early folds pay — did not appear: **fold 4
degrades in lockstep with fold 0** (0.67306 → 0.66847 → 0.61304 as `p` climbs, the same shape as
every other fold). There is no shift-protection gradient across the reachable anchors, so the
mechanism's out-of-sample benefit is simply absent in our shift range. The random control moving
≈0 confirms the drift/all degradation is signal-removal, not projection artefact.

**This closes the post-processing axis completely.** Affine (§1b), monotone/isotonic (§1q, §I10),
per-segment affine (§1t) and now non-monotone feature-orthogonalisation are all measured nil. The
one caveat is unfalsifiable by construction: the real test anchor (2026-02-13) is *more* shifted
than fold 4, and rho there is unobservable — but the flat fold-0→fold-4 gradient is the best
available evidence that neutralisation would not flip positive there either, and the mean effect
is so steeply negative that it is not worth a leaderboard slot to find out. `kill`.

### Also flagged (jury track, separate objective) — the two-head aggregate

Rescaling the whole submission to match an independent total-GMV forecast is **affine → monotone
→ rho-invariant** (does not touch the main score), yet it buys the total-GMV-RMSPE tie-breaker.
M5's top-down alignment ranked 2nd ([arXiv 2103.08250](https://arxiv.org/abs/2103.08250)).
`DATA.md` §8.4 measured that forcing the aggregate to the naive level costs +0.163 RMSLE, but
that was matching the *wrong* (unshifted) total; matching a seasonally-corrected ~24.4M total is
a smaller nudge, and RMSLE is flat near its optimum. Worth building for the finals write-up (a
cheap tie-breaker win at ~zero rho cost), not for the online leaderboard. Not run this session;
recorded as the one jury-specific lever with precedent.

---

## I29. The TS-FM extractor is a FAMILY, not one model — survey + shortlist (2026-08-24)

**Status: survey DONE (10 papers/leaderboards read via parallel agents), shortlist locked, fold-4
feasibility runs pending. Extends §I28 (frozen MOMENT gave the project's FIRST positive excess).
The question here: does a BETTER or MORE-DECORRELATED backbone — frozen, or LoRA-fine-tuned —
strengthen the one live lead? GPU extraction and fine-tuning are now IN SCOPE (user directive);
the CPU-only constraint of §I28 is lifted. Gated on the e0915 5-fold confirm (array 24145189).**

### The two findings that reframe §I28

1. **MOMENT is the WEAKEST extractor in the field.** Auer et al. (2510.26777) — the methodology
   paper for exactly our frozen recipe — benchmarks frozen forecasting FMs on UCR/UEA (RandomForest
   head): **TiRex 0.79 > Chronos-Bolt 0.78 ≈ Moirai 0.78 > TimesFM 0.77 > … > MOMENT 0.62–0.64 (LAST)**.
   So e0915's +0.006 sits on the weakest available backbone — switching is motivated on *quality*,
   not only decorrelation.
2. **Frozen is often these models' BEST regime** (TSFM-Bench 2410.11802: zero-shot ≥ few/full-shot on
   most datasets) and FMs specifically hold up on **daily** data (GIFT-Eval 2410.10393) — both
   favourable for us. BUT **no FM wins across datasets** → every candidate is a *decorrelation bet*,
   CV'd individually, never assumed from a leaderboard.

### The extraction recipe (durable, from 2510.26777 — replicate exactly)

- **Sequence axis → MEAN-pool** token/patch hidden states, masking inactive/padded days (our series
  are ~30% dense). Beats max and last-token.
- **Layer axis → CONCAT all layers**, normalising each before concat (deep layers over-specialise to
  forecasting). Exception: models with a built-in `.embed()` (Chronos) return one layer → seq-mean only.
- **Multivariate (≤13 channels):** run each channel through a univariate model and CONCAT across
  channels; or use a natively-multivariate model (TTM, Moirai) that ingests channels jointly.
- **⚠ ADD ABSOLUTE-SCALE STATS — load-bearing here.** Every FM instance-normalises internally, which
  *strips absolute scale*; our target is a 30-day **sum/level**. Recover it: split each series into
  k=8 patches, per-patch mean/std/min/max, concat to the embedding. **e0915 is currently scale-blind
  — this is a concrete strengthen even for the incumbent MOMENT.** Optional second aug: a first-order-
  difference embedding (push xₜ−xₜ₋₁ through the same frozen model, concat).
- **Head:** a STRONG non-linear head (our GBDT / per-patch GRU). A linear probe under-rates a good
  extractor (2510.26777) — do not judge a backbone with one.

### The shortlist (deduped across 10 sources; handles HF-verified by the agents)

Baseline: `AutonLab/MOMENT-1-small` (T5 masked-recon encoder, Time Series Pile, channel-independent).
Decorrelation ⇒ differ in architecture / objective / corpus.

| # | model | handle | params | arch / objective | corpus | multi? | licence | decorrelation vs MOMENT |
|---|---|---|---|---|---|---|---|---|
| 1 | **TTM / TinyTimeMixer** | `ibm-granite/granite-timeseries-ttm-r2` | 1–5M | **MLP-Mixer (no attn)** | ~1B public | native | Apache-2.0 | **max — diff family**; native 512 ctx + multichannel; cheapest |
| 2 | **Chronos-Bolt** | `amazon/chronos-bolt-small`/`-base` | 48/205M | T5 enc-dec, **value-tokenised** | ~100B, synthetic-heavy | uni | Apache-2.0 | high (objective+corpus); **`.embed()` API** |
| 3 | **TiRex** | `NX-AI/TiRex` | 35M | **xLSTM (recurrent)** | GiftEval+Chronos | uni | **NX-AI community — VERIFY contest use** | high (only non-transformer); **#1 extractor in 2510.26777** |
| 4 | **TimesFM 2.0** | `google/timesfm-2.0-500m-pytorch` | 500M | decoder-only, patched | **Google Trends/Wiki** | uni | Apache-2.0 | med-high; **on-domain (web traffic)**; GPU |
| 5 | **Toto** | `Datadog/Toto-Open-Base-1.0` | 151M | decoder + factorised space-time attn | **~1T observability telemetry** | native | Apache-2.0 | high (unique corpus); GPU (was CPU-blocked) |
| 6 | **Moirai-1.1-R** | `Salesforce/moirai-1.1-R-small`/`-base` | 14/91M | masked **any-variate encoder** | LOTSA (~27B) | **native any-variate** | Apache-2.0 | low arch (≈MOMENT) but native 13-ch joint + diff corpus |
| 7 | **VisionTS** | pip `visionts` (`facebook/vit-mae-base`) | ~112M | **ViT-MAE, TS-as-image** | **ImageNet (not TS!)** | uni | verify | **max corpus gap**; sparse→near-empty-image risk |
| 8 | **LPTM** | `kage08/lptm-large2` | 113M | encoder + **adaptive segmentation** | cross-domain (incl. stocks) | uni | verify | high (unique tokeniser); smaller corpus |

Further diversity, parked behind the top picks: Sundial `thuml/sundial-base-128m` (flow-matching),
Timer `thuml/timer-base-84m` (decoder), Chronos-2 `amazon/chronos-2` (native multivariate).

**Avoid / blocked:** Moirai-2.0-R (**cc-by-nc**), TabPFN-TS (non-commercial + not an embedding model),
TimeGPT (API-only), UniTS / ROSE (no public HF weights), LLM-reprogramming (GPT4TS/Time-LLM — lose to
specialists, TSFM-Bench). The newest FEV-Bench leaders (Chronos-2, TiRex-2, Toto-2.0, TimesFM-2.5) may
lack public weights — plan only around confirmed-downloadable checkpoints.

### Plan (two-tier, §4.2)

Screen = **fold-4 frozen feasibility** for the top picks, on GPU, per-patch-GRU + mean-pool heads,
each scored rho and `rho_partial` vs the **FULL champion** (never one member — §E1 inflation). Add the
abs-scale-stats aug to every arm. Cheapest-and-most-orthogonal first: **TTM** and **Chronos-Bolt**;
then **TiRex** (licence permitting), **Toto**, **TimesFM**; **VisionTS / LPTM** as pure-diversity
Phase-B. Escalate only a winner to **5-fold + LOFO admissibility vs the champion**.

### Fine-tuning axis (frozen → LoRA → full) — dedicated survey in flight

The frozen FM's entire value is external priors → decorrelation. **Full** fine-tune risks collapsing it
into a twin of the existing GRU (accurate, not decorrelated → useless). **Partial** (LoRA / adapters /
top-layers) is the regime that could gain strength while retaining priors → decorrelated AND strong,
the combination nothing here ever hit (rho_partial > 0.013). Per-fold FT with the embargo is mandatory
(no leakage).

**Survey done (agent).** Ships an official FT recipe *today*: Chronos-T5/Bolt/2 (AutoGluon, LoRA
default), TimesFM (PEFT/LoRA, official `examples/finetuning`), Moirai/-MoE (Hydra `conf/finetune`),
Timer, MOMENT (PEFT-ready), TTM (granite-tsfm), Toto-1.0. **Inference-only (no FT): TiRex, Sundial,
Toto-2.0** — so a fine-tune arm cannot use the frozen-#1 extractor (TiRex).

**Geometry answer to the tension — stay frozen→LoRA, never full FT.** Two weight-geometry papers:
*LoRA Learns Less and Forgets Less* (2405.09673 — full-FT perturbation rank 10–100× larger; LoRA
retains out-of-domain behaviour) and *LoRA vs Full FT: an Illusion of Equivalence* (2410.21228 —
full FT stays spectrally aligned to the pretrained weights, LoRA adds low-rank "intruder" dims but
forgets less). Mechanism: at 250k users we are NOT data-scarce, so full FT converges to the
*from-scratch* solution → a **GRU twin** (loses the external prior that decorrelates it); LoRA /
top-block-unfreeze stays near the prior → keeps the decorrelating info at a small accuracy cost.
**Decision variable = rho_partial / Δblend + corr(OOF, GRU) + CKA-vs-frozen, NEVER solo RMSLE, and
never one fold** (see the e0915 confirm above — that is exactly the trap).

**Turnkey scalar-regression path:** HF `PatchTSMixerForRegression` / `PatchTSTForRegression` (real
`num_targets` heads); TTM *is* PatchTSMixer, so pretrained-TTM-backbone + `…ForRegression` is the
least-engineering route. Two FT recipes: **R1 forecast-and-sum** (keep the native forecast head, sum
the 30 steps, backprop RMSLE → max prior preservation, best decorrelation — Chronos/TimesFM/Moirai/
Toto/TTM) and **R2 embedding+head** (MOMENT via `task_name='classification', num_class=1`).

**GPU cost trivial:** 100–500M full-FT fits A100-40GB; LoRA fits with room; ~10–30 min/fold, a few
GPU-hours for the whole spectrum. LoRA config: target attn `q,v` (T5) / `q_proj,v_proj` (decoder),
r=16 (sweep 8–32), α=32, LR 2e-4 head/adapters + 2e-5 backbone, bf16, 2–3 ep, batch 256, **fit
per-fold on that fold's train window only** (anti-leakage). Precedent: AutoGluon-TS ships Chronos-Bolt
LoRA-FT as a weighted-ensemble member.

**Spectrum to sweep as isolated experiments (§4.1):** head-only(=frozen) → +LoRA r∈{8,16,32} →
+top-2 blocks → full-FT (once, to confirm the twin). **⚠ Read against e0915:** frozen MOMENT is
already a pooled r=0.96 twin at rho 0.635, and §1s/the blend-bar findings say our data's signal
ceiling — not the model — is binding, so the honest prior that LoRA clears rho_B 0.641 @ r 0.96 is
low-to-moderate. Worth ONE clean LoRA test on the best-justified backbone, judged pooled vs champion.

### Honest prior

Low-to-moderate, unchanged from §I28. The realistic prize is a small *decorrelated blend member*
(~+0.006 rho_partial → ~−0.00003 RMSLE), not the +0.0005 rho / rho_partial≈0.04 top-1 needs — nothing
the project built ever exceeded 0.013. But this is the ONLY positive-excess direction ever found,
MOMENT is the weakest backbone in it, and a stronger / more-orthogonal / LoRA'd FM is the single
best-justified shot at a member that is decorrelated AND strong. **Gated on the e0915 5-fold confirm:**
if pooled rho_partial vs the champion holds positive, this whole shortlist is worth the feasibility
runs; if it collapses to ≈0, the bar for a *different* backbone rises sharply.

### ✅ RESULTS SO FAR (2026-08-24)

| member | pooled rho_partial vs family | test-time rho (LB) | notes |
|---|---|---|---|
| **e0915** frozen MOMENT per-patch GRU | **−0.0004** (fold-4 +0.011 was §E1 inflation) | **0.677** (LB 1.709235) | weak twin pooled; test rho = CV + the GENERIC +0.040 CV→test lift (GBDT e0049 got the same lift), so NOT an FM-specific long-context edge |
| **e0919** frozen Chronos-Bolt + abs-scale-stats | **+0.0054** (POSITIVE) | ~0.690 predicted (held — 2 slots left) | strengthen worked; first FM member positive pooled. ~42% of best-ever (0.0127), blend gain still ≈0 |

**Key lesson (the GBDT control):** the +0.040 CV→test rho lift is GENERIC — `e0049` (GBDT, no sequence)
lifted 0.661→0.702 = +0.041, identical to e0915's 0.637→0.677. So "the FM is stronger at test" is the
test window being more predictable for *everyone*, not a long-context edge. The FM member's blend VALUE
(rho_partial) is preserved CV→test, i.e. small. **Frozen FM = a real but modest decorrelated member.**

**Fine-tune axis REFUTED (e0920 r=16, e0921 r=8, `src/run_tsfm_lora.py`, A100).** LoRA MOMENT (q/v,
end-to-end) GAINS solo rho (0.651→0.664, matching frozen Chronos) but LOSES blend value: fold-4
rho_partial +0.0090 (r=16) / +0.0082 (r=8) — BELOW frozen MOMENT (+0.0112) and frozen Chronos (+0.0124).
Both ranks drift to **corr_vs_GRU 0.980** (r_vs_blend 0.983 vs frozen 0.962), i.e. adaptation pulls the
member into the supervised GRU's subspace regardless of rank → no minimal-adaptation sweet spot. The
geometry papers (2405.09673/2410.21228) + §1c confirmed: adaptation trades decorrelation faster than it
gains strength, net negative. **Frozen is the sweet spot; best FM member = frozen Chronos+scale-stats
(e0919, +0.0054 pooled).** Caveat: tested on MOMENT, but the drift-to-twin mechanism is backbone-general.
⚠ **CORRECTED 2026-08-25.** This line read *"RealMLP (e0917/e0913) CLOSED — timed out twice"*. It did
**not** stay closed on a timeout: the 500k config was resubmitted with a **24h cap (job 24152700)** and
**COMPLETED in 20h35m**, ending 08:28 on 2026-08-25 — after this section was written. **RealMLP is
REFUTED on a measurement, not a wall clock:** fold-4 `rho_B 0.671601` at `r 0.995356`, margin
**−0.003098 = 0.101x** the bar, in-sample optimal weight **0.000**. It is the *strongest* non-tree
non-recurrent member the project ever built and still lands on the same frontier — the tabular-NN
quadrant is not "decorrelated-but-too-weak" (that was NN_TORCH, rho 0.647); at full strength it is
**strong-but-not-decorrelated**. Full write-up: §I22's RESULT block; row in `experiments.csv`.

### Two user-proposed hypotheses (2026-08-24) — designs + honest priors

- **H1 — all channels → FM, concat (± PCA reduce).** `run_tsfm_chronos.py --multichannel`: mean-pool
  each of the ≤13 channels' Chronos embedding, concat [N, K·512], optional PCA→D, + scale-stats → head.
  Fold-4, then 5-fold. **Prior: LOW for rho_partial.** The champion's seq-GRU already reads all 13
  channels and §3b found extra channels didn't help — so more channels make the FM member MORE
  correlated with the blend (already r 0.979), raising solo rho but shrinking blend value. Cheap, worth
  the point, don't expect a wall-break.
- **H2 — FM embedding as EXTRA FEATURES to the strong models.** Sidesteps the weak-standalone-member
  wall entirely: the FM need only carry *some* orthogonal signal the model can pick up.
  - **H2a (→ GBDT), highest value:** save the Chronos mean-pooled embedding per (user, anchor), PCA→~32,
    add to the 665 features, retrain LightGBM, compare pooled CV vs e0049. `§1v` (a GBDT can't fit its own
    residual) does NOT cover this — the embedding is EXTERNAL info, not a recombination of existing
    features. The one route that could reveal genuinely new value. Cost: extract embeddings for the GBDT's
    fold rows (member-style subsample for feasibility) + LightGBM integration.
    **→ RESULT (e0922, fold-4): REFUTED, Δrho −0.00016.** BASELINE [665] rho 0.66967 → AUGMENTED
    [665 + Chronos-mean-pool PCA-32] rho 0.66951 (PCA kept 96% var, 400k train rows). The frozen FM
    embedding adds NOTHING to the GBDT — the 665 features already span its signal. §1v doesn't formally
    cover external info, but the empirical answer is identical. Closes the FM-as-GBDT-features route.
  - **H2b (→ CAUSAL_EXP GRU):** FM embedding as extra input to the seq GRU. Lower prior — the GRU already
    reads the raw sequence, so the FM's temporal summary overlaps more.
  **Order by value/cost: H2a → H1 → H2b.** All isolated (§4.1), judged on POOLED CV + `corr_vs_GRU`,
  never one fold (the e0915 confirm is why).

### ⛔ FM DIRECTION SATURATED (2026-08-24) — every lever tested and closed

| lever | fold-4 (rho / r / rho_partial) | verdict |
|---|---|---|
| frozen GMV per-patch GRU (e0919) | 0.664 / 0.979 / **+0.0124** (pooled +0.0054) | best member — but **earns weight 0 in the champion blend** (Task B) |
| LoRA MOMENT r=8/16 (e0920/e0921) | 0.664 / 0.983 / +0.008 | refuted — drifts to GRU twin (corr 0.98), no rank sweet spot |
| FM → GBDT features (H2a, e0922) | Δrho −0.00016 | refuted — 665 feats already span the FM signal |
| flat 13-channel (H1, e0923) | 0.615 / 0.911 / +0.0017 | decorrelated but too weak (mean-pool+PCA dilutes GMV) |
| hybrid GMV-GRU + 12-ch (e0924) | 0.646 / 0.957 / +0.00432 | strength-recovery FAILED — intermediate on a strict frontier |

**The frontier is strict and GMV-only sits at its top.** e0919 → e0924 → e0923 trace a line where every
step down in r costs MORE in rho: you cannot move to lower r (more decorrelated) without losing more
strength than you gain, so rho_partial only falls. §1c's law (decorrelation must be at comparable
quality) is unbeatable here — the other 12 channels' decorrelation is not at GMV's quality.

**Task B (blend):** the optimal non-negative champion+e0919 blend puts **weight 1.000 on the champion,
0.000 on e0919** → projected LB unchanged 1.6465. Even the best FM member earns zero weight; the
champion's usercv-GRU already subsumes it.

**CONCLUSION:** the FM's external temporal priors — the one genuinely-new information source — give at
best a small decorrelated member (e0919, +0.0054 pooled) that is worth **zero to the champion**, and NO
strengthening lever (LoRA, multi-channel, hybrid, feature-injection) helps. Same §1s wall the whole
project hit: the existing blend already captures essentially all the learnable signal from the ~13
channels; an off-domain frozen FM adds a decorrelated-but-weaker view, never a decorrelated-AND-strong
one. **FM direction closed.** Finals pick stands (e0300_cal/e0301 + e0162, per the robustness analysis).
