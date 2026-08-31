# EXPERIMENTS.md — running digest

Human-readable state, per `CLAUDE.md` §4.3. **Read this first in a new session.**
Full log: `experiments.csv`. Ranked ideas + graveyard: `BACKLOG.md`. Data facts: `DATA.md`.
Cluster how-to: `CLUSTER.md`. Literature: `PAPERS_FEATURES_AND_IDEAS.md`.
Local exploration: `explore.ipynb` (needs `reports/local_*` — see §8).

Last updated 2026-08-20. **§1b is the most important section in this file** — the metric is
solved from the leaderboard, rho is the only remaining quantity, and every route to raising
it has been measured and closed. §2's CV-based version of the same algebra is superseded by
§1b, which uses the true test moments instead of the folds'.

**State as of 2026-08-20 (CAUSAL_EXP feature line, e0193–e0197).** Nothing moved. Four
candidates measured on the user-split GRU, all inside the noise floor: `ds_order` −0.00010,
`cart_backlog` −0.00010, month-normalisation −0.00005 (2σ = 0.00040). e0193/e0194 are logged
`keep` **on user instruction, not on evidence** — the numbers in their rows are the record.
Two things from this round DO carry forward and are worth more than the features were:

1. **The LightGBM user-split proxy overstated `ds_order` by 36×** (−0.00360 screened vs
   −0.00010 confirmed). FEATURES_CAUSAL.md's central claim — that the recurrence cannot cheaply
   carry days-since-order — is refuted. Recalibrate that instrument before trusting it again.
2. **Cross-session drift on identical configs is +0.00027 to +0.00046**, larger than any effect
   size being tested. Every user-split candidate now ships with a matched same-session control
   (e0195, e0197). Do not compare against a logged baseline from another session.

---

## 1. Current best

| | exp | what | CV | LB |
|---|---|---|---|---|
| **best LB** | **e0301** | **e0300_cal with the usercv slot at GRU d48 instead of d128; weights unchanged** | n/a | **1.646456** |
| | e0303 | same, slot = mean(e0141, gru48, lstm48, xf48) | n/a | 1.646483 |
| | e0300_cal | e0162 with e0266 replacing e0049 in the gbdt slot | n/a | 1.646589 |
| | e0162 | optimal-weight blend (gbdt 0.20 / seq 0.38 / e0141 0.42), affine-calibrated | n/a | 1.646602 |
| | e0361 | e0301 with the seq half retrained THROUGH the guard zone (`--train-through 2026-01-14`) — **the guard-zone exclusion is correct**, §1z-D | no clean CV exists | 1.646806 |
| | e0302 | d48 added as a 4th component, weights FITTED | n/a | 1.647049 |
| | e0150 | **e0141+e0120 log blend, affine-calibrated to the probe-solved truth moments (§1b)** | n/a | **1.64670** |
| | e0141 | GRU on 85 causal features, user-split CV, 3 full-data seeds | 1.74341 (user-split) | 1.6488 |
| | e0193/4 | e0141 + `ds_order` / + `cart_backlog` (86 ch, one channel each) | 1.74377 both (user-split; vs matched control e0195 1.74387, Δ −0.00010, **not significant**) | — |
| | e0120 | 50/50 log-blend of the gbdt and seq families (9 members) | 1.76280 | 1.6553 |
| | e0101 | **GRU on 13 raw daily channels — no engineered features at all** | 1.76458 | — |
| previous best LB | e0064 | AutoGluon `WeightedEnsemble_L2_FULL` on e0060's 400 features | 1.76519 | 1.6559 |
| | e0049 | LightGBM, 665 features (core + `sbc` − `sbcmoment`) | 1.76551 | 1.6562 |
| | e0060 | LightGBM, 400 features (null-importance top400) | 1.76547 | 1.6567 |
| | e0020 | LightGBM, 185 features | 1.76638 | 1.6578 |
| naive floor | geo3 | — | 1.92862 | — |
| `sample_submit` | p30 | — | 2.25296 | 2.1200 |

**Top-1 has moved four times: 1.6485 -> 1.646480 -> 1.646180 -> 1.645914 (2026-08-18).
e0162 at 1.646602 is now 0.000688 behind (0.000300 of rho) -- the gap has widened 1.63x.**

> At the paired 50k-user noise sd of 0.00038 (DATA.md 8.3) that is **1.81 sigma**, up from 1.11.
> The 95% CI on the true gap is [-0.000057, +0.001433] -- zero is still inside, but only just.
> "We cannot show we are behind" was true at 1.11 sigma; at 1.81 it is no longer comfortable.
> What would close it, all measured: the §1m recombination ceiling covers **10%**; a new blend
> member would need `rho_partial` **0.02795** against a best-ever 0.01269; and a member-level
> gain would have to be **1.9x** (via e0141, w=0.42) to **3.9x** (via gbdt, w=0.20) the largest
> member gain the project has ever produced (e0090's +0.00038). Rival rho values are LOWER
> BOUNDS -- they invert `RMSLE = sd_L*sqrt(1-rho^2)`, which holds with equality only if the
> rival calibrated optimally; if they did not, their true rho is higher still. e0150 sits exactly on its rho ceiling, so no
post-processing can improve it further; raising the score now requires raising rho, and §1b
shows every route to that has been measured and closed.

> e0064's CV is now the full 5-fold 1.76519, assembled from `runs/ag/e0064_f*_oof.npy` by
> `src/ag_oof_to_parquet.py` and verified against each fold's logged score. The earlier
> 4-fold 1.76901 in this table was not comparable to anything and is superseded.

---

## 1b. ⚠ THE METRIC IS SOLVED FROM THE LEADERBOARD (2026-08-14) — read this first

Two probe submissions already in `cv_lb.csv` pin the **public test truth distribution exactly**:

```
probe_zeros    pred = 0   -> RMSLE^2 = E[L^2]               = 3.28^2
probe_const10  pred = 10  -> RMSLE^2 = E[(L - log1p 10)^2]  = 2.32^2
    =>   E[log1p y] = 2.3199      sd(log1p y) = 2.3187
```

Validated against something independent: `sample_submit` (= last-30d GMV, LB 2.12) comes back
at **rho = 0.5771** against DATA.md §7.1's CV-measured **0.557**.

With `sd_L` known, `RMSLE^2 = (sd_L - sd_M)^2 + 2 sd_L sd_M (1 - rho) + (mu_L - mu_M)^2` turns
**one LB score into that submission's rho**, and the optimal affine transform in log space
(`M' = mu_L + a(M - mu_M)`, `a = sd_L rho / sd_M`) collapses the score to `sd_L sqrt(1-rho^2)`.
**Level and spread are free to fix. Only rho is irreducible.** `src/calibrate_lb.py`.

### The rho history of the entire project

| submission | LB | rho | ceiling at that rho | unclaimed |
|---|---|---|---|---|
| `sample_submit` | 2.1200 | 0.57712 | 1.8936 | 0.2264 |
| e0001 (62 feat) | 1.6766 | 0.69804 | 1.6603 | 0.0163 |
| e0020 (185) | 1.6578 | 0.70216 | 1.6509 | 0.0069 |
| e0049 (665) | 1.6562 | 0.70235 | 1.6505 | 0.0057 |
| e0064 (AutoGluon) | 1.6559 | 0.70257 | 1.6500 | 0.0059 |
| e0120 (gbdt+seq blend) | 1.6553 | 0.70400 | 1.6467 | 0.0086 |
| **e0140 (GRU, 4 features)** | 1.6552 | **0.70068** | 1.6543 | 0.0009 |
| e0141 (GRU, 85 feat) | 1.6488 | 0.70345 | 1.6480 | 0.0008 |
| e0142 (GRU, 143 feat) | 1.6785 | 0.70083 | 1.6540 | 0.0245 |
| e0151 (shift-only) | 1.64748 | 0.70375 | 1.6467 | 0.0008 |
| e0152 (10-member NNLS) | 1.646697 | 0.70401 | 1.6467 | 0.0000 |
| e0090 (tuned LightGBM) | 1.65524 | 0.70209 | — | — |
| e0145 (tuned GRU, shift-cal) | 1.65323 | 0.70210 | 1.6513 | — |
| e0146g (gbdt half alone) | 1.651202 | 0.70177 | — | — |
| e0150 | 1.646700 | 0.70374 | 1.6467 | 0.0000 |
| e0161 (e0090 swapped in) | 1.646670 | 0.70375 | 1.6467 | 0.0000 |
| **e0162 (optimal weights)** | **1.646602** | **0.70378** | 1.6466 | **0.0000** |

> **The probe constants are only 3 significant figures and that mattered.** e0150 and e0151
> are affine transforms of the SAME blend, so they share rho exactly -- four precise equations
> in four unknowns. Re-solved from the 6-s.f. submission scores:
> **`mu_L = 2.3303`, `sd_L = 2.3178`** (probes said 2.3199 / 2.3187; both back-check inside
> their rounding). Under the refined constants:
>
> | model | rho | ceiling |
> |---|---|---|
> | e0120 (gbdt+seq) | 0.70326 | 1.64778 |
> | e0141 (CAUSAL_EXP GRU) | 0.70310 | 1.64814 |
> | **blend of the two** | **0.70375** | **1.64667** |
>
> **CORRECTION.** I earlier wrote that the blend with e0141 "added no correlation at all",
> from the probe-based constants. That is WRONG: the GRU contributes **+0.00048 rho =
> -0.00111 RMSLE**, and e0120 alone perfectly calibrated tops out at 1.64778 -- worse than
> e0150's actual 1.64670. e0141's solo rho is the LOWER of the two; its value is
> decorrelation, not accuracy.

**Ninety experiments moved rho by 0.006.** A GRU on *four* GMV features reaches 0.70068 —
within 0.0017 of the 665-feature LightGBM. Features, architectures, capacity and blending all
recover the same signal by different routes.

**Every submission before e0150 left 0.006–0.009 unclaimed to calibration.** That is larger
than everything modelling produced after e0020.

> **PROTOCOL RULE, effective now: shift every submission's log-mean to 2.3199 before sending
> it.** It needs no rho, costs nothing, and has been worth 0.006–0.009 every single time.

### The classification ceiling — the last lever, and it is closed

`src/rho_decomp.py` partitions `Cov(L, M)` exactly (law of total covariance, `Z = 1{y>0}`):

```
BETWEEN  separating buyers from non-buyers   78.6%
WITHIN   ranking magnitude among buyers      21.4%

oracle split x our magnitudes    rho 0.66247 -> 0.93260   (+0.271)
our split    x oracle magnitudes rho 0.66247 -> 0.71603   (+0.054)
```

Classification is worth **5x** magnitude in rho, **7.7x** in RMSLE. (Cross-check: "oracle split
x one constant" reproduces 0.98868 against DATA.md §8.2's independently measured 0.983.)

So we asked whether a **dedicated** classifier beats the regressor's implicit AUC of 0.84322:

| model | AUC on `y>0` |
|---|---|
| regression blend (never asked to classify) | 0.84322 |
| e0160 LightGBM, binary objective, 665 features | 0.84412 |
| e0161 seq GRU, BCE head, 13 raw channels | 0.84443 |
| e0162 seq GRU, BCE head, 40 channels | 0.84419 |
| average of the three classifiers | 0.84511 |

Four model classes, two objectives, two feature regimes — **all inside 0.002**. And the gain is
worth nothing end-to-end (leave-one-fold-out stack of `[M, clf, M*clf]`):

```
baseline blend M                       rho 0.66247   RMSLE 1.76280
+ any classifier                       rho 0.66227   RMSLE 1.76310   (+0.00031, 2/5)
stack on M alone (CONTROL)                           RMSLE 1.76304   (+0.00024)
```

**The control is the finding.** Refitting on `M` alone already costs +0.00024, so the
classifier's marginal contribution is **+0.00007** — zero. Without that control this reads as
"a small loss" and the reason is invisible. *Always run the no-op control in a stacking test.*

This also retires the payoff curve in `rho_decomp.py`: its `d(rho)/d(AUC) ~ 1.2` was measured
along the ORACLE path and flagged as an upper bound at the time. **The realised conversion rate
is ~0.** Extra AUC comes from reordering near-tied pairs, which moves rank statistics and
almost no squared log-error.

**Conclusion: model optimisation on this problem is finished.** The 0.271 of rho in the
classification term is real and unreachable — bounded from both sides, by the oracle above and
by four independent attempts to reach it.

---

## 1c. The ten-family sweep, and what decorrelation actually requires (2026-08-15)

The one channel that had ever paid was adding a different model FAMILY (+0.00143 for the first
gbdt+seq blend, +0.00048 for the user-split GRU).  So five more families were built in e0141's
exact setup -- same 85 features, same target, same `md5("gmv-v1")` folds, so every number is
directly comparable.

| model | unseen-user RMSLE | vs GRU | AUC on y>0 |
|---|---|---|---|
| **GRU (e0141)** | **1.74341** | — | 0.84647 |
| transformer (ALiBi, causal) | 1.74466 | +0.00125, **0/5 folds** | 0.84604 |
| GRU + residual target | 1.74618 | +0.00277 | 0.84573 |
| CatBoost (GPU) | 1.77543 | +0.03202 | 0.83766 |
| XGBoost (GPU) | 1.77606 | +0.03265 | 0.83755 |
| CatBoost + residual target | 1.78992 | +0.04651 | 0.83521 |
| Ridge (closed form) | 1.81596 | +0.07255 | 0.82818 |

### The surprise: they ARE decorrelated

Correlations of log-predictions at the five frozen anchors (all now blendable -- each user is
held out in exactly one user-fold, so stitching the folds gives complete OOF on the same
`(fold_id, user_id)` keys as `oof/e0049.parquet`):

```
                    vs gbdt    vs e0101    vs usercv GRU
usercv_catboost      0.9741     0.9701       0.9682
usercv_xgboost       0.9729     0.9691       0.9673
usercv_ridge         0.9433     0.9382       0.9348
usercv_full (GRU)    0.9949     0.9953       1.0000

reference: e0049<->e0064 0.9983 (twins, worth ~0)
           gbdt <->e0101 0.9951 (paid +0.00048)
           catboost<->xgboost 0.9983 (twins of each other -- ONE family, not two)
```

**Prediction on record, and it was wrong.** I expected CatBoost/XGBoost to be twins at >=0.998
because they scored within 0.0006 of each other. They are at **0.974** against the existing
family -- five times more decorrelated than anything we had. Near-identical accuracy does not
imply near-identical predictions, and inferring one from the other was an error.

### And the blend still does not move

```
equal weight, added to the 7-member blend:      delta
  + catboost                                  +0.00077
  + ridge                                     +0.00158
  + all three                                 +0.00433

leave-one-fold-out FITTED weights (honest, no LB):
  current 7 members                            1.76141
  + catboost                                   1.76142
  + catboost + ridge                           1.76142
  + all three                                  1.76142
```

Equal weight gets worse -- these members are 0.035-0.08 weaker and equal weighting overweights
them.  With weights fitted honestly the optimiser sees all that decorrelation and assigns it
**nothing**: unchanged to five decimals whether one new family is added or three.

> **THE CORRECTED RULE.  Decorrelation is necessary but not sufficient -- it must be
> decorrelation AT COMPARABLE QUALITY.**  The arithmetic `rho_B = rho*sqrt(2/(1+r))` assumes
> equally good members; the "find a partner at r <= 0.9885" target silently carried that
> condition.  A model at rho 0.622 correlating at 0.943 adds nothing, because its disagreement
> with the family is mostly its own error rather than a different view of the truth.

**Ten families now** -- LightGBM, AutoGluon, CatBoost, XGBoost, Ridge, TCN, GRU, transformer,
two CV protocols, two target parametrisations.  Best OOF blend **1.76141**, and three
genuinely decorrelated families move it by **+0.00001**.

---

## 1d. Mixup — synthetic users by interpolation (2026-08-15)

The first thing in many sessions to move anything, and it went the opposite way to my
prediction on both arms.

| run | unseen-user RMSLE | vs baseline | AUC on y>0 | best epochs |
|---|---|---|---|---|
| baseline (1 seed) | 1.74358 | — | 0.84647 | 13–25 |
| **naive mixup** | **1.74293** | **−0.00065, 5/5 folds** | **0.84673** | 17–41 |
| class-preserving mixup | 1.76216 | **+0.01858** | 0.84441 | 1–14 |

Per fold, naive vs baseline: `−0.00058 / −0.00053 / −0.00068 / −0.00062 / −0.00050` — spread
0.00018, against a seed noise of ~0.0002–0.0003.  The AUC also rose (+0.00026), the first
movement all session in the quantity §1b calls capped.

**Construction.** Interpolate a batch with a shuffled copy of itself, `lam ~ Beta(0.2, 0.2)`
(U-shaped, so most synthetic users stay near a real one).  Mixing is linear and standardisation
is affine, so mixing raw features then standardising is identical to the reverse.

### Two predictions, both wrong

**I argued naive mixup would HURT.**  DATA.md §6.1 measures the target as two components — a
44% spike at 0 and a bulk near 4.2 with an almost empty region between (0.127% of real users;
interpolated pairs land there 6.3% of the time, 50x over-represented).  That reasoning
conflated *generative plausibility* with *useful supervision*: mixup is not trying to sample
the marginal, it is a regulariser encouraging smooth behaviour between training points.
Whether a synthetic target resembles a real user is beside the point.

The mechanism is visible in the epoch counts: **13–25 without mixup, 17–41 with** — the model
trains 40–60% longer before overfitting.  That matters because this model is known to sit on a
cliff (e0106: 30 epochs cost +0.0204).

**I proposed class-preserving as the SAFE variant, and it collapsed.**  Supervising only where
both source users agree on buy/no-buy conditions the training set:

```
true P(buy)                      0.5577
P(both agree)                    0.5067
P(buy | supervised, i.e. agree)  0.6139   (+0.0562, a 10% inflated buy rate)
```

The model trains on an inflated buy rate and is validated on the true one, so validation
degrades from the first epoch and early stopping fires at epochs 1–6.  A conditioning bias I
should have checked before calling it the conservative option.

### Accounting

The gain sits inside the 0.00173 total variance available to any variance-reduction method
(measured by extrapolating the 4-seed averaging curve, `SSE = a + b/k`), and it is what remains
after 3-seed averaging has taken its share.  **Open question being measured now:** mixup and
seed averaging are both variance reductions and may not add.  The 3-seed comparison is against
the 3-seed baseline (1.74341), not the 1-seed one.

> Protocol note: these are user-split numbers and are NOT comparable to experiments.csv.

---

## 1e. BTYD — the last untried backlog item, and the functional it exposes (2026-08-16)

`src/btyd.py` · `src/run_btyd.py` · `src/btyd_blend.py` · `oof/e0170.parquet` · spec in `BTYD.md`

BG/NBD (timing) + Gamma-Gamma (value), maximum likelihood, refitted per anchor on that
anchor's population from days ≤ A only. Seven parameters over four numbers per user, against
the 665-feature LightGBM and the 13-channel GRU. Frozen folds, `src/metrics.py`, OOF written
in `run.py`'s schema. **8.6 minutes for all five folds on the laptop** — no cluster job.

### The headline is not the blend result. It is the functional.

Same fitted parameters, same data, two ways of turning the fitted process into a number:

| functional | CV | vs geo3 |
|---|---|---|
| `E[log1p y]` — simulate the process forward, average in **log** space | **1.83569 ± 0.02371** | **−0.09293** |
| `log1p(E[y])` — the textbook BTYD output, `E[X(30)] · E[M]` | 2.39829 ± 0.05281 | **+0.46968** |

**+0.5626 for the choice of estimand alone.** That is, by a factor of ~30, the largest single
effect this project has measured — larger than every feature, architecture and blend decision
combined. The cause is visible in one number: `E[y]` is dominated by the Gamma-Gamma right
tail, so `log1p(E[y])` has log-mean **3.819** against the truth's **2.397**. The naive output
is worse than the optimal global constant *and* worse than `sample_submit`.

> **This is why BTYD has a bad reputation on RMSLE tasks, and it is a reporting error rather
> than a model failure.** `BTYD.md` §3.3 predicted the direction; the size was not predicted
> by anyone. Generalisation for the write-up: a generative model's natural output is `E[y]`,
> and **every** metric in this family is minimised by a different functional of the same
> fitted distribution. Discriminative models never expose the choice because L2-on-`log1p`
> silently targets the right one.

### The blend decision — pre-registered, and it lands in Ridge's quadrant to three decimals

```
                         gap vs e0049    corr vs gbdt    LOFO fitted blend gain
usercv_ridge (§1c)         +0.07255         0.9433             0.00000
e0170 BTYD                 +0.07018         0.9423            -0.00006
```

Two model classes with nothing in common — a closed-form linear map on 665 features, and a
7-parameter generative process on 4 sufficient statistics — agree on **all three axes**. The
pre-registered rule (`BTYD.md` §6: keep only if the fitted leave-one-fold-out blend improves
by > 0.0005) returns **KILL** by a factor of eight; the optimiser does hand it weight 0.031,
and that weight is worth −0.00006.

Pre-registered prior was ~15%; recorded here as resolved **no**.

**No segment pocket either.** BTYD loses **0/5 folds** and **every decile** of the blend
prediction, by a strikingly flat +0.059 to +0.101. There is no subpopulation to route to it.

### BACKLOG e0033's actual staging, with the no-op control

`P(alive)`, `E[X(30)]`, `E[M]` and `log BTYD` stacked on the 9-member blend:

```
CONTROL: refit on the blend alone       1.76296
+ p_alive                               1.76296   -0.00000
+ log E[X(30)]                          1.76288   -0.00008
+ all four columns                      1.76288   -0.00008
```

The reason is that **the RFM inputs were already in the feature set** and nobody had noticed:
`buy_days_total` = `x + 1`, `recency_order_days` = `T − t_x`, `gmv_total / buy_days_total` =
`m_x`. Only `T` (measured from the first *buy*-day rather than the first active day) was
absent. e0033's own staging rule — "add raw `(x, t_x, T)` first; if those do not move CV, do
not fit BG/NBD" — was therefore already answered in the affirmative by e0001. **BG/NBD adds
functional form, not information**, and this problem has never once paid for functional form.

### What the fitted parameters say about the panel

`BTYD.md` §5.1 predicted the dropout half would collapse, because the population rule selects
on end-of-window activity. It collapsed harder than predicted:

```
a = 0.012 .. 0.020   (all five anchors)      mean P(alive) 0.978 .. 0.983
                                             P(alive) > 0.99 for 74% .. 85% of users
```

`a < 1` at every anchor, so the closed-form `E[X(t)]` (which requires `a > 1`) is **unusable
on this data** — the Monte-Carlo route is not a nicety here, it is the only one that works.
**BG/NBD has degenerated to an NBD**: the model cannot express churn on a panel that
excluded churned users by construction.

Gamma-Gamma's independence assumption is also violated: `corr(log x, log m_x) = +0.23` at
every anchor (raw-space `corr(x, m_x)` is a harmless +0.01…+0.03, which is what a naive check
would have reported). Both diagnostics were specified in advance and both came back bad —
the model is misspecified for this panel in exactly the two ways the spec nominated.

### Simulation noise is a real cost and it is measurable

`E[log1p y]` is a Monte-Carlo estimate, so its error enters the metric additively. Fitting
`RMSLE² = A + B/S` over S ∈ {20, 200, 1000} (the same extrapolation §1d uses for seed
averaging) gives a noise-free limit of **1.8348**:

```
S =   20   1.87579        S -> inf   1.83481  (from S = 200, 1000)
S =  200   1.83921                   1.83486  (from S =  20, 1000)
S = 1000   1.83569                   1.83510  (from S =  20,  200)
```

So 1000 draws sits 0.0009 above the limit, and the verdict is not a simulation artefact.

**Net: the backlog is now empty, and the one durable result is the +0.5626 functional gap.**

---

## 1f. The admissibility frontier — what a new blend member must actually deliver (2026-08-16)

`src/admissibility.py`. Written while reviewing `BAYES_EXP.md`, and it **supersedes the
correlation-threshold heuristic this project has used since §3b.**

§1c stated the rule qualitatively — "decorrelation must be at comparable quality". The exact
statement is one line of standard algebra. For truth `L`, family blend `M`, candidate `B`:

```
R² = rho_M² + (1 - rho_M²) · rho_partial² ,        rho_partial = corr(L, B | M)
```

> **A candidate's entire blend value is its partial correlation with the truth, controlling
> for the blend we already have.** Not its accuracy. Not its correlation with the family.
> One number that subsumes both, computable from any OOF file in a second.

Equivalently, with the *excess correlation* `e = rho_B − r·rho_M`: `ΔR² = e²/(1 − r²)`.
So `e = 0` is worth exactly zero **at any r** — a member can be wildly decorrelated and
worthless, or a near-twin and valuable.

**Validated before use.** Predicted vs measured two-member gains, all ten candidates, agree
to ≤0.0001. Each row uses a family blend that *excludes* the candidate:

| candidate | rho_B | r vs M | e | **rho_partial** | predicted | measured |
|---|---|---|---|---|---|---|
| e0064 | 0.66128 | 0.99708 | +0.00073 | **0.01269** | −0.00014 | −0.00006 |
| e0049 | 0.66113 | 0.99682 | +0.00075 | **0.01255** | −0.00014 | −0.00007 |
| **e0170 BTYD** | 0.62709 | 0.94266 | **+0.00254** | **0.01017** | −0.00009 | −0.00004 |
| e0100 | 0.66097 | 0.99710 | +0.00037 | 0.00647 | −0.00004 | −0.00003 |
| e0102 / e0101 | ~0.6615 | ~0.998 | ~0.0002 | 0.0053–0.0054 | −0.00003 | −0.00004 |
| e0101s1/s2/s3, e0108 | ~0.6615 | ~0.998 | ~0.0002 | 0.0027–0.0041 | −0.00001 | −0.00003 |

```
rho_partial required for -0.0005:  0.02383
best ever achieved:                0.01269  (e0064) = 53% of the bar
```

**Three findings, and the third is a correction to our own protocol.**

1. **BTYD has the largest excess correlation of any model ever built here** — `e = +0.00254`,
   **3.4× e0049's**. It is not a weak imitation of the family; it genuinely disagrees. Its
   problem is entirely the `1/(1−r²)` divisor: at r = 0.943 that divisor is 17× larger than
   at r = 0.997, and it eats the whole advantage.
2. **Decorrelation cuts both ways, and we only ever counted one side.** At fixed `e`, a *more*
   correlated candidate is worth *more*, because `1−r²` is smaller. "Find a partner at
   r ≤ 0.9885" is retired — `r` alone is not the currency and never was.
3. ⚠ **The 0.0005 bar was never achievable by a single member, including our own.** e0049 is
   worth −0.00007 marginally; e0064 −0.00006. By the standard `BTYD.md` §6 set, **the
   existing blend should never have been built** — its −0.00176 is the accumulation of nine
   sub-threshold contributions. BTYD at 80% of e0064's `rho_partial` is not an outlier; it is
   inside the same narrow band as everything else. The verdict on BTYD is unchanged (it is
   worth −0.00004) but the *reason* is now stated correctly: not "too weak to matter" but
   "nothing here clears a bar that no member has ever cleared".

---

## 1g. `BAYES_EXP.md` reviewed against the record (2026-08-16)

The spec proposes B0 (`btyd_classic`), B1 (`hier_cov`, SVI + covariates), B2
(`hier_seasonal`). Status after measurement:

**B0 is already built — it is e0170 (§1e).** Same model, same order-day counting, same
buyer/browser/dormant handling. CV 1.83569, killed. `BAYES_EXP` §5.2's Gauss–Laguerre
Rao-Blackwellisation would replace our 1000-draw MC, which §1e prices at **0.0009** above the
noise-free limit — real, worth having, and far too small to change any verdict.

**§10's uncertainty columns — the one genuinely new claim — measure zero (e0172).**

```
CONTROL: blend alone                     1.76296
+ sd_log1p                               1.76298   +0.00002
+ p_zero                                 1.76293   -0.00003
+ sd, p_zero, p_alive                    1.76297   +0.00001
+ all of those and the point estimates   1.76290   -0.00006
```

**The column is real and the argument for it is wrong.** `sd_log1p` correlates **+0.3187**
with the blend's absolute residual — it genuinely knows where we are wrong, and §10 is right
that a GBDT cannot manufacture it. But §10's reason ("exactly the quantity RMSLE cares about
when deciding how hard to shrink toward zero") does not hold:

> **Under squared error the Bayes action is the posterior MEAN. The posterior SD cannot
> change the optimal point prediction — the shrinkage is already inside `E[log1p Y]`.**
> Uncertainty columns pay for pinball, CRPS or an expectile. They cannot pay for RMSLE.

**§5.3's calibration wrapper adds nothing on CV (e0173)**, and its own blocking rule does not
fire: `a_c` = 0.9932…1.0005 over the five folds. That is §2's `k* = 1.000` again. §1b already
established that calibration pays at the **test anchor** and not on the folds, and shifting
every submission's log-mean is already standing protocol — so §5.3 is a re-derivation of
something we do, not an addition.

### ⚠ Two things in `BAYES_EXP.md` must not be built as written

1. **§6's folds are guard-zone contaminated, and it nominates the worst one as primary.**
   F2 (C=2025-11-15), F3 (C=2025-12-15) and F4 (C=2026-01-14) all have target windows lying
   **inside** `[2025-11-16, 2026-02-13]`, where all 250k users are active by construction.
   `DATA.md` §4.3 prices that bias at **+0.041 RMSLE** — 80× the effect being chased — and
   §5.1 calls it the single most important finding in the project. §6 says "**F4 is the
   primary validation fold**". Only F1 (C=2025-10-16) is clean, and it is already frozen
   fold 4. Rule 3 applies: use `data/folds.parquet`, unchanged.
2. **§3.2's `s_next` from the year-earlier block is in the graveyard twice.** The forecast
   window is 2026-02-14 → 2026-03-15 and §3.2 proposes estimating its multiplier from
   2025-02-14 → 2025-03-15. That is exactly what **e0142** did via day-of-year features:
   **1.6785, our worst model since e0001**, a −0.410 log-mean shift, predicted before
   submission and confirmed. Independently, `BACKLOG` Band A+ retracted the whole year-lag
   mechanism on CV — `corr(spike, residual) = +0.0001`, and **every** multiplier k > 1 makes
   CV worse, best k = **1.00**. §3.2 calls this "the single most falsifiable claim in the
   plan"; it has already been falsified twice, once with a submission spent on it.

Two things in the spec are right and worth keeping: **§8.1's transductive refit** (fitting
population hyperparameters on all 250k at the submission cutoff uses features, never targets
— legitimate, and worth saying out loud in the write-up), and **§7's `P(alive)` reliability
curve**, which is jury material rather than score.

### B1/B2: the bar, pre-registered

B1 must reach **`rho_partial` ≥ 0.02383** — **1.9× the best any model here has achieved**, and
2.3× B0's. Equivalently, `(rho_B, r)` pairs on the frontier:

```
if B1 lands at r =    it needs rho_B      = CV*      = B0 improved by
        0.9427            0.63051        1.82519         0.0065
        0.9600            0.64104        1.80479         0.0269
        0.9700            0.64700        1.79298         0.0387
        0.9900            0.65845        1.77150         0.0602
```

**The one live argument for building it**: B0 already carries `e = +0.00254`, the largest
excess on record. If B1 preserves that excess while raising its own quality — landing near
r ≈ 0.99 — it clears the bar. The usual pattern is that excess collapses as models converge,
which is why I put this at **~10%**, below `BTYD.md`'s 15% prior for B0. Recorded before any
decision to build.

---

## 1h. B1 built and measured — a better model and a worse blend member (2026-08-17)

`src/bayes_cov.py` · `src/bayes_model.py` · `src/run_bayes.py` · `oof/e0180.parquet`

`BAYES_EXP.md` B1 (`hier_cov`): 28 covariates on the rate, dropout and spend priors, frozen
folds, run.py's OOF schema. **Prior registered in §1g was ~10%. Resolved: no — and for
exactly the stated reason.**

**Departure from §4.2, and it is an upgrade.** §4.2 prescribes NumPyro mean-field SVI over
per-user random effects, then spends half the section defending that choice (floor the
scales, warn when the floor binds, check `mu_lam`/`mu_theta` posterior correlation, sanity-fit
NUTS). All of those are patches for approximating an integral this model does exactly:
BG/NBD's Gamma and Beta priors **are** the random effects. So the covariates go on the priors
and the closed form is kept. `beta = 0` reproduces e0170 exactly, and the JAX fit matches
`src/btyd.py`'s independent scipy fit to four decimals.

### The headline

| | raw CV | **CV\*** (optimally calibrated) | **rho** | r vs family | e | **rho_partial** |
|---|---|---|---|---|---|---|
| B0 `e0170` | 1.83569 | 1.83241 | 0.62709 | 0.94266 | +0.00254 | **0.01017** |
| **B1 `e0180`** | 1.87825 | **1.81822** | **0.63452** | 0.95638 | +0.00088 | **0.00400** |

**The covariates work.** rho rises **+0.0074, on all five folds**, and calibrated CV\* improves
**−0.0142**. Raw RMSLE gets 0.0426 *worse*, and per §1b that is the wrong comparison — it is a
level/spread artefact and calibration is free. Judging B1 on raw RMSLE would have been the
e0141 mistake in mirror image.

**And it is a worse blend member.** `r` rose 0.9427 → 0.9564, so the excess `e` **fell by 65%**
and `rho_partial` fell **0.01017 → 0.00400** — from 43% of §1f's bar to 17%. Measured blend
gain: **+0.00000**.

### Why, exactly — one regression settles it

```
log B1 = -0.7153 + 0.5601 * log B0 + 0.6419 * log family        R2 = 0.9382
```

**B1 is, to 94% of its variance, a 53/47 mixture of B0 and the models we already have.** The
28 covariates did not add a direction; they rotated B0 *toward* the family. Every bit of the
+0.0074 rho landed inside the span of the existing blend.

Adding it is not merely worthless, it is faintly harmful — and adding it *alongside* B0
dilutes B0's own contribution:

```
family alone           1.76274
+ B0                   1.76268    -0.00006
+ B1                   1.76290    +0.00016     <- worse than not adding it
+ B0 and B1            1.76280    +0.00006
```

> **This is the sharpest form of the project's oldest lesson.** Ten families, four model
> classes, a metric solved from the leaderboard, and now a generative model with covariates:
> everything that gets better gets better *in the same direction*. §1f's framework predicted
> precisely this risk before the run ("excess usually collapses as models converge") and the
> collapse is measurable in a single number.

### ⚠ The unconstrained fit diverges, and I called it wrong

`r` climbs 114 → 144 and is **still rising at 2000 iterations on every fold**; `a+b` → 194…484;
`p_gg` → 128…162 (B0: `r` ≈ 1.1, `a+b` ≈ 2.2, `p` ≈ 1.1). That is not slow convergence — the
optimum is on the boundary at `r → inf`. The Gamma prior on the rate is collapsing to a point
mass: **the per-user random effect is being optimised out of existence** and all heterogeneity
migrates into the covariates, destroying the credibility shrinkage that made B0 work.

This is `BAYES_EXP` §4.2's variance collapse, which prescribes a floor. **I argued the floor
was unnecessary here because marginalising the latents leaves "no free sigma to collapse".
That was wrong.** `1/r` and `1/(a+b)` play sigma's role and collapse identically — the hazard
was renamed, not removed. §4.2 was defending against something real.

### e0183 — the floor fixes the fit and changes nothing, which is the more interesting result

`--which loc` freezes every dispersion at its B0 value and lets covariates move only the
locations. The parameters become sane. The predictions do not move.

| | `r` | `a+b` | `p_gg` | rho | CV\* | r vs family | rho_partial |
|---|---|---|---|---|---|---|---|
| e0180 free | **113.96** | **389.2** | **162.1** | 0.63452 | 1.81822 | 0.95638 | 0.00400 |
| e0183 floored | **0.53** | **2.00** | **1.11** | 0.63435 | 1.81846 | 0.95641 | 0.00313 |

```
corr(log e0180, log e0183) = 0.98497
```

> **Two fits whose BTYD parameters differ by 200× produce the same predictions.** rho differs
> by 0.00017, correlation with the family by 0.00003. So the dispersion collapse was a genuine
> optimisation pathology, and it was **not** the cause of B1's blend failure.
>
> **Once 28 covariates sit on the locations, the generative process is just a link function.**
> The covariate regression determines the prediction; BG/NBD's parameters barely enter. And
> covariate regressions on this data all land in the same place — which is §1b's "features,
> architectures, capacity and blending all recover the same signal by different routes",
> now demonstrated inside a *generative* model as well as the discriminative ones.

Both variants decompose the same way against `{B0, family}`:

```
log e0180 = -0.715 + 0.560*log B0 + 0.642*log family     R2 = 0.9382
log e0183 = -0.888 + 0.678*log B0 + 0.515*log family     R2 = 0.9501
```

Neither adds a direction. `e0183` costs **19.4 min** for all five folds against `e0180`'s
unbounded search, so the floored variant is also the one to keep if this is ever revisited.

### Two bugs worth keeping

1. **A JAX where-trick failure that reported success.** `gammaln(p*n)` is evaluated at `n = 0`
   for Gamma-Gamma-masked users; `digamma(0)` is NaN, and reverse-mode AD multiplies it by the
   mask's zero cotangent to get NaN. The *objective stayed finite*, so L-BFGS-B stopped at
   iteration 1 and returned `success=True`. **Check the gradient for finiteness, not just the
   value** — the run looked like a converged fit at a terrible score.
2. **Covariates with |z| up to 249** (`ord_per_buyday`; `gmv_per_buyday` 31.5). Ratio features
   on this panel are violently heavy-tailed; a handful of users dominated the likelihood.
   Winsorised at ±8. §3.1's `Normal(0, 0.5)` prior on `beta` would not have caught this — the
   problem is the design matrix, not the prior.

**Cost: 8.5 h wall clock, ~45 min CPU** (the laptop slept between turns). No cluster job.

---

## 1i. The metric solver is now exact — and only when nothing is assumed

Five submissions were predicted in closed form before being sent:

```
        predicted     actual      error     note
e0150     1.64610   1.646700   +0.00060    probe constants (3 s.f.)
e0151     1.64694   1.647480   +0.00054    probe constants
e0152     1.64620   1.646697   +0.00050    probe constants
e0145     1.65130   1.653230   +0.00193    ASSUMED rho = 0.7030
e0161     1.64667   1.646670   +0.00000    all rho measured
e0162     1.64663   1.646602   -0.00003    all rho measured
```

Two changes explain the collapse in error from ~0.0005 to ~0.00003:

1. **Refined truth moments.** e0150 and e0151 are affine transforms of the *same* blend, so they
   share rho exactly — four precise equations in four unknowns. Re-solved from 6-significant-
   figure submission scores: **`mu_L = 2.3303`, `sd_L = 2.3178`** (probes said 2.3199 / 2.3187;
   both back-check inside the probes' rounding). `src/calibrate_lb.py` now uses these.
2. **No assumed rho.** The one submission that guessed a member's rho (e0145) missed by 0.0019 —
   60x worse than the two that measured everything.

**The strongest single validation:** `e0146g`'s rho was computed as **0.70219** purely from its
members' covariances, with no leaderboard score of its own, and solving from the actual score
gave **0.70219** — error 0.000000.

> **CONSEQUENCE. Submissions are no longer needed to evaluate a re-combination of measured
> members** — the blend's rho and calibrated score are computable to +/-0.00003 in advance.
> Spend slots only on members whose rho is unknown.

### Blend weights are already optimal

Solving the optimal non-negative weights over the three components:

```
component            Cov/sd_L      sd     pairwise corr
gbdt(e0090,e0064)     1.10354   1.5716   1.000 0.997 0.995
seq family            1.11633   1.5874   0.997 1.000 0.997
e0141 GRU             1.11398   1.5844   0.995 0.997 1.000

inherited weights 0.25/0.25/0.50  ->  rho 0.70375
optimal           0.20/0.38/0.42  ->  rho 0.70377   (worth -0.00004)
```

And a member's gain is diluted by roughly its weight, which is why nothing moves:

```
e0090's member-level gain          +0.00102 covariance  (+0.00038 rho)
     entering at weight 0.20   ->  +0.00001 at the blend
```

To close 0.000184 of rho now needs ~+0.00092 rho via gbdt, ~+0.00048 via the seq family, or
~+0.00044 via e0141 — against the +0.00038 that the single successful member-level tuning
produced.

---

## 1j. Tuning: one success, one instructive failure (2026-08-16/17)

Hyperparameter tuning was the last untouched large lever. Both halves were tried. **They gave
opposite answers, and the difference is entirely methodological.**

| | search protocol | objective | CV delta | rho delta on LB |
|---|---|---|---|---|
| **LightGBM** (e0090) | frozen date folds | fixed rounds, scored once | −0.00082, 5/5 | **+0.00038** |
| **GRU** (e0145) | user-split CV | **min of an early-stopped curve** | −0.00099, 5/5 | **−0.00101** |

The GRU's leaderboard loss almost exactly cancelled the gain its CV claimed.

### Two wrong explanations before the right one

Both were stated with more confidence than the evidence supported, and both are recorded
because the *refutations* are the useful part.

1. **"Wrong protocol — it optimised user-generalisation and broke temporal transfer."**
   **REFUTED.** Scored at all five frozen anchors spanning four months, the tuned model is
   better at *every* one, trend +0.00001/step (flat), extrapolating to **+0.00033** at the test
   anchor against an actual **−0.00101**.
2. **"It stopped mid-descent on a 150-epoch cosine, so the epoch count didn't transfer."**
   **REFUTED.** Re-run annealing over the epochs actually trained (`--sched-tmax 71`):
   **1.74356**, versus untuned 1.74351 and mid-cosine 1.74252. Proper annealing gives back
   exactly the untuned score.

### The actual cause: early-stopping selection bias

The validation curve is noise. Early stopping reports its **minimum**, and the minimum of `N`
noisy draws is biased low by about `sigma*sqrt(2*ln N)` — so a configuration that trains longer
gets more draws and reports a better number *for that reason alone*.

```
config                  evals   min       mean(last 10)   noise sd   min-selection bias
untuned (plateau)          21   1.74033      1.74109       0.00136        ~0.00336
tuned, cosine T=150       100   1.73875      1.74049       0.00273        ~0.00827
tuned, cosine T=71         71   1.74006      1.74071       0.00178        ~0.00519
```

On the honest statistic (`mean(last 10)`) the gap is 0.0006, not the 0.00099 the minima claimed
— and rho went the other way. **My Optuna objective was rewarding configurations for training
longer**, and I reported that back as the finding "better configurations train longer".
LightGBM with fixed rounds cannot suffer this, which is why the tabular search transferred.

> **RULE. Never score a tuning trial on the minimum of a variable-length early-stopped run.**
> Use a fixed budget scored once, or `mean(last k)`. Otherwise the search optimises the number
> of evaluations.

**RETRACTED:** "your long-training hypothesis is confirmed at 5 folds" — the confirm inherited
the same biased rule. Also retracted: "tuning cleared the ~0.845 AUC ceiling" (one trial on one
fold hit 0.84783; at five folds it is 0.84673, identical to mixup alone).

---

## 1k. The architecture search — 8 families, 31 trials, and a third methodological error

`src/tune_seq.py`, frozen fold 4, **fixed-epoch objective** so §1j's bias cannot arise.
`epochs` competed as an ordinary hyperparameter over 8–200 log-scaled.

| architecture | best on fold 4 | trials |
|---|---|---|
| **gru** | **1.73143** | 13 |
| cnngru | 1.73337 | 5 |
| xformer_rope | 1.73428 | 2 |
| lstm | 1.73430 | 2 |
| xformer_alibi | 1.73784 | 2 |
| tcn | 1.75180 | 2 |
| rnn | 1.75415 | 3 |
| xformer_learned | 1.75671 | 2 |

Three findings that hold regardless of the selection problem below, because they are measured
across all 31 trials rather than read off a winner:

* **The vanilla RNN is 0.023 behind the GRU** — the gating is doing real work, not decoration.
* **`xformer_learned` (learned absolute positions) is the worst of all eight**, 0.025 behind
  ALiBi and RoPE. The claim that absolute-time encodings extrapolate badly here — asserted
  since the first TCN and never tested — is now measured.
* **Winners chose 8–40 epochs, clustered at 13.** The unbiased objective *rejects* long
  training, confirming §1j's diagnosis from the opposite direction.

### The third error: selecting on one fold

I searched fold 4 alone ("most test-like per §3.2") and took the winner. §3.2 is a reason to
*report* that fold separately, not to *select* on it. Stage 2 confirmed all 8 leading configs on
all five folds:

```
exp      screen(f4)   cv_mean     delta  wins  f4 delta
e0180       1.73143   1.76417  -0.00041  2/5  -0.00103
e0181       1.73151   1.76549  +0.00091  2/5  -0.00095
e0182       1.73155   1.76543  +0.00085  2/5  -0.00085
e0183       1.73171   1.76559  +0.00101  2/5  -0.00061
e0184       1.73194   1.76607  +0.00149  2/5  -0.00054
e0185       1.73228   1.76724  +0.00266  2/5  -0.00009
e0186       1.73315   1.76881  +0.00423  1/5  -0.00006
e0187       1.73329   1.76533  +0.00075  1/5  +0.00150

corr(screen fold-4 score, 5-fold mean) = +0.596
```

**All eight beat e0101 on fold 4. Exactly one beats it on the 5-fold mean**, by −0.00041 with
2/5 folds — which fails §3.4 (needs >2 sigma AND ideally 4/5).

> **RULE. Screen on one fold if you must, but SELECT on the confirm.** A single-fold search with
> tens of trials produces single-fold hyperparameters. The screen/confirm split of §4.2 exists
> for exactly this and I skipped it.

**Verdict: e0101's hyperparameters stand.** The seq family is not improved by tuning as run.

> ⚠ **CORRECTION (2026-08-18).** I first wrote that "31 trials against one fold had room to fit
> that fold's noise, and did." **That explanation is wrong**, and §1l gives the measurement that
> refutes it: fold 3 was never selected on, yet 7 of 8 configs beat e0101 there — by more than
> they beat it on fold 4. The rule above still stands; the mechanism does not.

---

## 1l. The top-8 confirm read properly — a fold-index interaction, and zero blend value (2026-08-18)

§1k logged the eight confirms and drew the wrong lesson from them. Re-reading the same eight
runs against the noise floor and against the blend gives a different mechanism and the same
final verdict.

### The noise floor, measured on this family

Four seed replicates of the parent (e0101, e0101s1/s2/s3), frozen folds:

```
sigma_noise per fold : f0 0.00052  f1 0.00032  f2 0.00031  f3 0.00029  f4 0.00022
sigma_noise on the 5-fold MEAN : 0.00023      =>  2 sigma = 0.00047
```

e0180's mean delta is **−0.00041 = 1.8 sigma on 2/5 folds** → `no effect` by §3.4, confirmed.

### But the mean is a cancellation of two real effects, not a null

Per-fold delta vs e0101, all eight configs:

```
exp        f0        f1        f2        f3        f4   | early(f0-2)  late(f3-4)   spread
e0180  +0.00012  +0.00047  +0.00076  -0.00236  -0.00103 |   +0.00045    -0.00169   -0.00214
e0181  +0.00203  +0.00323  +0.00252  -0.00230  -0.00095 |   +0.00259    -0.00162   -0.00422
e0182  +0.00211  +0.00329  +0.00254  -0.00285  -0.00085 |   +0.00265    -0.00185   -0.00450
e0183  +0.00225  +0.00340  +0.00263  -0.00263  -0.00061 |   +0.00276    -0.00162   -0.00438
e0184  +0.00277  +0.00392  +0.00354  -0.00223  -0.00054 |   +0.00341    -0.00138   -0.00479
e0185  +0.00475  +0.00526  +0.00481  -0.00142  -0.00009 |   +0.00494    -0.00075   -0.00569
e0186  +0.01294  +0.00413  +0.00179  +0.00233  -0.00006 |   +0.00629    +0.00113   -0.00515
e0187  +0.00018  +0.00110  +0.00174  -0.00079  +0.00150 |   +0.00101    +0.00035   -0.00065
-------------------------------------------------------
wins        0/8       0/8       0/8       7/8       7/8
```

**8/8 have negative spread.** e0180 beats e0101 on fold 3 by −0.00236 against a per-fold sigma
of 0.00029 — **8.1 sigma** — and loses fold 2 by +0.00076 = 2.5 sigma. Both are real.

**Why this is not noise-fitting.** Two measurements kill that explanation:

1. **The search was run on fold 4 only. Fold 3 was never selected on** — and 7/8 configs beat
   e0101 there, by a *larger* mean margin (−0.00153) than on the selected fold 4 (−0.00033).
   Selection bias cannot produce a bigger effect on the held-out fold than on the fitted one.
2. **The confirm reproduces the search's fold-4 value to ≤0.00008 for ranks 1–6** (ranks 7–8
   drift up to 0.0008, the two configs with a different batch size). `run_seq.py` is
   deterministic at seed 0, so fold 4's confirm is the *same computation* as its search trial —
   it carries no independent information either way. All the evidence is in folds 0–3.

So the honest statement is a **hyperparameter × fold-index interaction**: the tuned regime
(d32–64, dropout ~0.35, 8–15 epochs) is genuinely better on the two most recent anchors and
genuinely worse on the three oldest. e0180 is the one config that keeps the late-fold gain
while staying inside noise early (+0.00045 = under 1 sigma).

> **The interaction is confounded and this run cannot separate it.** In an expanding-window
> scheme a later fold is *both* more recent *and* trained on more anchors (fold 0 gets 8, fold 4
> many more). Recency and train-size predict the same sign here. Separating them needs a
> fixed-window control, which is not worth running — see the blend result below.

### The blend says all of it is worth zero

`src/admissibility.py`, all eight against the 9-member family blend (§1f's frontier):

```
candidate   rho_B     r vs M    e          rho_partial   pred gain   measured
e0180       0.66194   0.99884   +0.00017   0.00465       -0.00002    -0.00004
e0186       0.65973   0.99499   +0.00051   0.00683       -0.00004    -0.00004
e0181..5    ~0.6615   ~0.998    ~+0.00001  0.0003-0.0006 -0.00000    -0.00001
e0187       0.66123   0.99814   -0.00008  -0.00169       -0.00000    +0.00000
                                     bar for -0.0005:    0.02383
```

Direct measurement on the calibrated 9-member blend, per fold:

```
variant                        rho        cal mean    delta
BASE (e0120 as built)          0.66254    1.76177        --
e0101 -> e0180                 0.66254    1.76176   -0.00001
e0101 -> e0186                 0.66255    1.76176   -0.00001
BASE + e0180 + e0186 (11)      0.66257    1.76171   -0.00006   <- best available
BASE + all 8 (17 members)      0.66246    1.76189   +0.00012   <- dilution, as §1i predicts
all 7 seq -> e0180..e0186      0.66231    1.76223   +0.00046   <- clearly worse
```

The best move in the table is **+0.00003 rho ≈ −0.00007 LB RMSLE** — the same size as adding
one more seed replicate of e0101, which §1f already prices at −0.00003. It is noise-averaging,
not information. And **replacing the seq half wholesale is +0.00046 worse**: the tuned configs
correlate more tightly with each other than the originals do, so they blend worse even where
they score better.

**Verdict: `kill` all eight as blend members; e0180 `park` as a standalone late-anchor model.**
The gap to top-1 is 0.000422 RMSLE; the entire top-8 exercise offers 0.00007 of it, inside the
blend's own measurement error. e0101's hyperparameters stand, now for the right reason: not
because the tuned configs failed to learn anything real, but because what they learned is
**already in the blend**.

---

## 1m. The recombination ceiling — everything we have ever built, blended optimally (2026-08-18)

`src/recombine_oof.py` · `reports/eda/recombine_all_oof.log`. **Closes the question "is there
something better hiding in the previous experiments?" with a number instead of an argument.**

Every OOF file on the frozen folds — 21 usable predictors, every family the project built —
searched three ways.

```
best single member       e0180                      rho 0.66194
e0120 as built (9 equal)                            rho 0.66254   cal 1.76177

greedy forward, equal weights, IN-SAMPLE (11)       rho 0.66258   cal 1.76168   -0.00009
greedy forward, subset chosen on 4 folds, scored on the 5th       mean          -0.00004
unconstrained OLS over all 21, leave-one-fold-out                 1.76181       +0.00004
```

**The honest ceiling is −0.00004 CV**, and greedy essentially *rediscovers e0120*: it keeps 7 of
the 9 existing members and adds e0180, e0186, e0110. Selection is stable — **e0180 is picked
first in all five LOFO folds**, so it is genuinely the best single member we own — and it still
buys nothing, because everything it knows is already in the blend.

**The unconstrained fit loses money out of sample** (+0.00004): 21 free weights over predictors
correlating at 0.998 is the same trap §1i flagged for the 10-member LB fit, now measured on CV
where a held-out check is possible.

### Why it is capped, two ways

* **Dilution, measured.** The largest member-level gain the project ever achieved (e0090,
  **+0.00038 rho**) moved the blend by **+0.00001**. Closing a 0.000184-rho gap this way needs
  ~18 such wins, or one ~18x larger.
* **Admissibility (§1f).** A new member needs `rho_partial ≈ 0.0219` to buy 0.000422 RMSLE. Best
  ever built is e0064's **0.01269**; best of the 8 tuned configs is **0.0068**. The top 19
  members sit inside 0.002 rho of each other at r ≈ 0.998 — there is no diversity left to
  harvest from this pool.

### ⚠ And the gap being chased is smaller than the instrument

Every large LB gain in this project came from **calibration, not rho**: e0141 scored 1.6488 on
the *level* term while carrying a LOWER rho than e0120 (0.70345 vs 0.70400), and e0150's jump
was the affine calibration itself. That lever is exhausted — e0162 sits exactly on its rho
ceiling — so the remaining 0.000422 is pure rho, the quantity that has moved 0.006 in ~90
experiments.

Against DATA.md §8.3's paired-delta noise on the 50 000-user public split:

```
gap to top-1                       0.000422
sd of a paired 50k delta           0.00038      =>  1.1 sigma
95% CI on the true gap    [-0.00033, +0.00117]  =>  zero is inside it
```

> **We cannot demonstrate that we are behind top-1**, and the private split is 200 000
> *different* users. A 1.1-sigma deficit on the sample everyone is tuning against is not
> evidence of a worse model. **The highest-value remaining decision is which two submissions to
> select as finals, judged on robustness rather than on public score** (§9 argues e0150 + e0151
> on exactly that basis) — not another point of rho.

---

## 1n. Multi-anchor test-time augmentation — killed, and the reason generalises (2026-08-18)

`src/run_seq_anchors.py` · `src/anchor_blend.py` · `oof/e0101_anchors.parquet` · exp **e0188**.

**The idea.** We emit one prediction, made at the final anchor A. But the same trained model can
be run at anchor `A−k` for any k, and at prediction time the realised GMV over `[A−k+1, A]` is
*observed*. So each k gives another view of the same user:

```
pred_k              ~= GMV over [A-k+1, A-k+30]      model output at anchor A-k
known_k              = GMV over [A-k+1, A]           OBSERVED, k days
pred_k - known_k    ~= GMV over [A+1, A-k+30]        the first 30-k target days
```

Weight them toward the final day (`w_k = (30−k)/30`), optionally rescale by `30/(30−k)`, average.

**Design.** Training is byte-identical to e0101 — same seed, same `t_hi = vai − 30`, only the
scoring points change — so this is one change (rule 2) and the baseline is *nested*: the k=0
column reproduced e0101's folds **exactly** (`[1.77291, 1.79298, 1.77464, 1.74997, 1.73240]`),
making every comparison paired. 5.9 min on one H200. Then 666 schemes searched offline:
3 estimators (raw / subtract / subtract+rescale) × 3 smoothings (none / MA3 / MA5) × 25
weightings (k0, linear, quad, sqrt, uniform_K, linear_K, exp_τ) × 3 combiners (log-space mean,
raw-space mean, extrapolate-to-k=0).

### Result: not one of 666 schemes beats k=0

```
BASELINE k=0                                cal 1.76358   rho 0.66162
best non-degenerate, no subtraction         cal 1.76374   rho 0.66154   +0.00015
best non-degenerate, with subtraction       cal 1.76526   rho 0.66081   +0.00168
THE PROPOSAL AS STATED (resc/linear/log)    cal 2.07297   rho 0.47243   +0.30939
worst of the family (resc/uniform_30/log)   cal 2.21800   rho 0.33258   +0.45442

LOFO scheme selection picks `k0` on all five folds.  Honest gain: +0.00000.
```

**Internal check:** `uniform_2 / extrap` scores *identically* to `k0` to 8 d.p. — a line through
two points read at the first point is that point, so the algebra demands it. The extrap combiner
is correct; the family simply degenerates toward "just use k=0".

### Two causes, and the second is the general lesson

1. **Unit mismatch in the subtraction.** `expm1(model output)` estimates roughly `E[log1p y]`
   (the loss is MSE on log1p), while `known_k` is an arithmetic realised sum. They are different
   functionals — **this is the +0.5626 gap §1e already measured as the largest effect in the
   project**. Measured here: `mean g_29 = 93.1` against `mean p_29 = 38.0`, so the correction is
   2.5x the thing it is subtracted from and **42.9% of users clip to zero**, destroying the
   ranking. A model trained to predict `E[y]` would be needed for the subtraction to be well
   posed.
2. ⚠ **Shifting the anchor is not label-preserving, so this is not augmentation.** Anchor `A−k`
   forecasts `[A−k+1, A−k+30]` — a *different target*. Image TTA works because flips and crops
   preserve the label; here every k changes it. So averaging buys bias growing linearly in k
   against variance reduction of essentially zero, because adjacent anchors of the same model on
   near-identical inputs correlate >0.99. **Pure averaging with no subtraction also loses**
   (+0.00015 best), which isolates the effect: it is not the subtraction alone that fails.

> **RULE. Before averaging predictions, check that the thing being averaged estimates the SAME
> quantity.** Seed replicates do (worth ~0.00003). Anchor shifts do not. "More predictions,
> averaged" is only variance reduction when the estimand is held fixed.

**Verdict `kill`.** Cost 5.9 GPU-min plus the offline search; both artefacts kept
(`reports/eda/e0101_anchors_schemes.csv` has all 666 rows).

---

## 1o. tsfresh — 60 validated statistics, worth nothing (2026-08-20)

`src/features.py::block_tsfresh` · `scripts/screen_features.py` · exps **e0192** (screen),
**e0191** (confirm) · `FEATURES.md` has the per-family table.

**The idea.** Port the [tsfresh](https://github.com/blue-yonder/tsfresh) statistics that the
installed `tsfeat`/`sbc`/`fcast` blocks do *not* already cover, over the per-user daily series.
The library is row-by-row Python/pandas — at 250k users × ~90 anchors × 3 windows it is hours
per fold — so ten extractors were hand-vectorised as matrix ops on the `(n_users, window)`
slice: 60 features = 10 statistics × {gmv, ord} × {30, 90, 365}.

### The one durable result: validate a reimplementation against the library

Each statistic was checked against tsfresh 0.21.2's own `feature_calculators` on random panels.
**Three of seven were wrong on the first pass**, and none would have been caught by CV — a
wrong statistic still trains, still scores, and still looks like a null result:

| statistic | first-pass error | cause | after fix |
|---|---|---|---|
| `c3` | rel **1e15** | bispectrum-style triple product instead of the lagged one | **0.0** |
| `arch7` | rel **0.41** | segment-wise Pearson; tsfresh uses a *global* mean/var with an `(n−lag)·var` denominator | 6e-08 |
| `trendt` | rel **0.123** | residual taken around `b·t`, omitting the intercept → t-stat deflated ~12% | 2e-15 |

`time_reversal_asymmetry` 3e-08, `longest_strike_above/below_mean` **0.0**, `lempel_ziv` 6e-08.

> **RULE. A "library-style" feature that is not the library's statistic is an untested new
> feature wearing a validated name.** The check cost about an hour and invalidated three of the
> ten columns. Applies to every future port — `tsfel`, `catch22`, anything from a paper.

### Result: nil at both tiers

```
e0192 screen (2 anchors, 30k users, paired)   bundle  +0.00019 (A1) / -0.00074 (A2)
                                       noise control  +0.00014 (A1) / -0.00097 (A2)
e0191 confirm (frozen folds, 250k users, 725 feat)   cv 1.76545 vs e0049 1.76551
      folds [1.77339, 1.79407, 1.77722, 1.75060, 1.73198]   Δ -0.00006   wins 2/5
```

Δ = **0.6× σ_noise**, 2/5 folds → `no effect` by §3.4, and 60 columns for 0.00006 is exactly
the within-noise accumulation §3.4 forbids. **`kill`.** 142.7 min on `compute`.

### A pattern that looks real and is not — worth the correction

The per-fold deltas are perfectly monotone in fold index (and so in training-set size):

```
fold          0        1        2        3        4
delta   +0.00027 +0.00007 +0.00007 -0.00028 -0.00041     spearman(delta, n_train) = -0.90
n_train    1.60M    2.42M    3.25M    4.10M    5.17M     P(perfect order | chance) = 1/120
```

Tempting story: high-variance shape statistics need rows, hurting early folds and helping late
ones — which is §1l's fold-index interaction and would make this `park`, not `kill`. **It does
not survive the right denominator.** σ_noise = 0.00009 is the sd of the *5-fold mean* over the
e0001 seed replicates; the sd of a *single fold* is 3× larger, `[0.00039, 0.00028, 0.00033,
0.00022, 0.00026]`. Against those the deltas are `[0.70, 0.25, 0.21, −1.28, −1.59]σ` — **not one
fold reaches 2σ.** A 1-in-120 ordering of five numbers that are individually noise is a 1-in-120
event, and we look at many such orderings. Verdict stands at `kill`.

> **Use the per-fold sd, not σ_noise, when reading per-fold deltas.** σ_noise is a
> mean-of-5 quantity and is ~3× too small for a single fold. §3.4's "≥4/5 folds" clause is the
> protocol's way of asking the same question without needing the per-fold sd.

### Lempel-Ziv: implemented, validated, deliberately not emitted

`_tf_lempelziv` matches the library (6e-08) but costs **~32 s per 250k users per window**
against ~0.3 s for every other statistic — an unvectorisable per-user prefix scan that ignores
all 16 cores, ~2 h added per fold build. **The first cluster attempt died on exactly this**
(job 23868236: 55 min of a 1 h `computeshort` cap without finishing one fold's features).
Dropping it took the block 66 → 60 features and 6.0 s → 0.8 s. It was also the worst screen
family (−0.00218 at A2). Function and its validation are kept in `src/features.py`; re-enable
only if someone writes a vectorised LZ.

---

## 1p. The XGBoost confirm harness — e0210–e0214, and what was verified before spending a slot (2026-08-21)

`src/run_xgb.py` · `configs/e0210_xgb_base.yaml` … `e0214_xgb_noisectl.yaml` ·
`scripts/e0210_xgb.slurm` · `scripts/e0210_smoke.slurm`. **Status: array 24055170 running.**

Built to answer FEATURES.md's three confirm-eligible candidates under XGBoost (user
instruction). The experiments and the decision rule are in FEATURES.md's "Confirm status"
section; what follows is the part that does not belong in a feature file.

### Why a fourth run exists that nobody asked for

e0214 adds a single i.i.d. normal column. It is not a hypothesis about the data — it measures
what |Δ| *this* protocol yields for a feature that cannot carry information. Every candidate
here scored ~1× the screen's own noise control, and a confirm inherits that problem: at 665
installed features the interesting quantity is not the sign of the delta but its size relative
to a null column. Reading e0211–e0213 against zero would repeat the error §3.4 exists to
prevent. This costs one run out of five and is what makes the other four legible.

### Why a fifth run exists that nobody asked for either

e0210 (XGB on e0049's exact features) is a pure family reference. Without it, "e0049 + XGB +
new feature" differs from its parent in two ways and neither delta is readable (§4.1). It also
closes a real gap in the record: §1c killed XGBoost/CatBoost/Ridge at *untuned* quality
(+0.00001 in the blend), but that was the user-split harness at lr=0.05/depth=8 — **XGBoost has
never been run on the frozen folds at LightGBM-comparable settings.** e0210 is therefore also
the CatBoost-shaped question from REVIEW_NOTES.md §D, asked of the other family.

### What was verified locally before submitting anything

The laptop cannot train (8.5 GB RAM, no lightgbm/xgboost, Python 3.9), but it *can* build
features on the 15k-user `data/_screen_subset.parquet`. Four checks, all on the laptop, zero
cluster time:

1. **Correctness against brute force.** Each of the four new blocks was re-implemented
   independently (dense daily matrices, no prefix sums) and compared. `age_bucket_gmv_share_3`
   agrees to 1.2e-07 (float32 epsilon); `cart_backlog_7`, `cohort_rel_buy_rate90` agree
   **exactly**. `noise_ctl` is deterministic across calls and has mean −0.006, sd 1.015.
2. **The leak guard passes** on all four blocks at both anchors 2025-06-18 and 2025-10-16.
3. **⚠ The guard was proved non-vacuous.** A guard that passes proves nothing until it has been
   shown to fail. The leaky first-draft `age_bucket` FEATURES.md describes — age window *not*
   capped at the anchor, so young users' `[90,120)` runs into the target — was reconstructed and
   run through `assert_no_lookahead`: **caught at both anchors** (`LOOK-AHEAD at anchor 168` /
   `288`, column 0). The guard is live on exactly the failure class these features risk.
4. **Cluster/laptop parity**: md5 of `src/{run,features,data,metrics}.py` + `fold_spec.json`
   identical on both sides before and after the push.

The smoke job then confirmed on real data: 665 → 666 features for every candidate (each block
adds exactly one column), guard passing at full scale, and a sane RMSLE from the parameter
mapping rather than a broken-objective number.

### Two operational failures, both mine, both cheap to repeat

* **A `computeshort` time limit survived the move to `compute`.** The smoke job was switched to
  the `compute` partition but kept `--time=1:00:00`; it TIMEOUT'd at 01:00:10 having validated
  4 of 5 configs. Nothing was lost (`--no-log`), but the chained array then sat on
  `DependencyNeverSatisfied` and had to be cancelled and resubmitted. **When changing a
  partition, change the wall clock with it** — the limit that was generous on the short queue is
  the one that kills the job on the long one.
* **`afterok` chaining turns a smoke timeout into a silently parked array.** The chain is still
  right — it stopped 5 full CV runs from starting behind a job that had not passed — but a
  parked array reports nothing until someone looks. Check the dependency state, not just the
  queue.

### The prior, stated before the numbers land

§1m puts the recombination ceiling at −0.00004 CV and the admissibility bar for a new blend
member at rho_partial ≈ 0.024 against a best-ever 0.01269. Ten consecutive feature nulls precede
these (§4). **The expected outcome is three nulls**, and that is a result: it closes the tabular
feature question rather than leaving it open with two "borderline" candidates in a screen file.
The value of e0210 is separate and does not depend on the candidates — it is the first honest
frozen-fold measurement of a second GBDT family at comparable settings.

### The numbers (2026-08-22) — the prior was right, and the noise column won

Array 24055170, five arms, 5 frozen folds each, ~2h15 per arm on `compute`:

```
arm                                   cv_mean      Δ vs e0210    sigma   folds won
e0210  XGB base, e0049's 665 feats    1.76588            --         --      --
e0214  + i.i.d. NOISE COLUMN          1.76595       +0.00006      +0.71     2/5
e0212  + cart_backlog_7      (cand B) 1.76596       +0.00007      +0.82     2/5
e0213  + cohort_rel_buy_rate90 (C)    1.76596       +0.00008      +0.89     2/5
e0211  + age_bucket_gmv_share_3 (A)   1.76600       +0.00011      +1.27     3/5

spread across all five arms = 0.00011 = 1.27 sigma
```

**Three nulls, as pre-registered — but the sharp form is that the noise column placed best of
the four.** Every candidate is positive (worse than omitting it) and every one is behind a
random number. No appeal to σ_noise or fold counts is needed: the thing that cannot carry
information scored better than the things that were supposed to. `kill` all three.

> **This is what the fourth run bought.** Read against zero, e0211–e0213 are "+0.00007 to
> +0.00011, small, 2–3/5 folds" — the exact shape of a borderline result someone argues about.
> Read against e0214 they are unarguable. **One extra arm in a five-arm array converted a
> judgement call into a fact.** Do this on every future feature confirm.

**And the screen's ranking was worthless — not merely small.** e0189 ranked candidate A first
(+0.00074 / +0.00112 at two anchors); A is the worst of the three here. A screen whose best
candidate ties its own noise control has no resolution left to order anything: it can say
"nothing here is large", never "this one is best". **Either build a screen with real separation
from its control, or confirm everything and skip the ranking step.** This is the second
instrument to fail this way — the causal proxy's 36× inflation (graveyard, e0193/e0195) was the
first, from completely different machinery.

### e0210: XGBoost is behind LightGBM on identical features

```
e0210 XGB  1.76588   folds [1.77368, 1.79506, 1.77738, 1.75070, 1.73260]
e0049 LGB  1.76551   folds [1.77312, 1.79400, 1.77715, 1.75088, 1.73239]
Δ = +0.00038 (+4.2 sigma), XGB wins 1/5 folds
```

A real difference by §3.4, and the first frozen-fold measurement of a second GBDT family at
LightGBM-comparable settings (§1c's kill was the user-split harness, untuned). It does **not**
say XGBoost is a bad model — hyperparameters were *mapped* from LightGBM's, not tuned for it,
so this is "XGBoost at LightGBM's settings". As a blend member it is a separate question
(§1f's bar is rho_partial 0.02383) and its OOF is on the cluster if anyone wants to price it.

⚠ **The smoke test's version of this comparison said +0.00177 (19.7σ) and was an artefact.**
The smoke trains on 2 anchors/fold (~424k rows) against the real 20–25 anchors (4.1–5.2M). At
1/10th the data the family gap inflates 4.7×. §5 already records that training-set size
dominates cut-off proximity in the transfer matrix; the same effect distorts any truncated
screen of a *model*, not just of a feature. **Never read a model-family delta off a smoke run.**

---

## 1q. Residual EDA — why no feature can work, measured directly (2026-08-22)

After 22 hand-designed candidates, 60 tsfresh statistics and a noise column beating all three
confirms, the right question stopped being "which feature next" and became **"is there anything
left for a feature to explain?"** This section answers it from the OOF, not from theory. All
numbers are `oof/e0049.parquet`, fold 4 (the most test-like anchor, 2025-10-16, n=225,431).

### 1. The model is already at the conditional mean of its own prediction bins

Bin users into 200 quantiles of the model's prediction — inside a bin, the model considers
users **identical**. Then:

```
var(L) total                                        5.4835
mean WITHIN-bin var(L), 200 bins                    2.9977    = 54.7% of total
best achievable RMSLE if each bin's mean were predicted perfectly   1.7314
ACTUAL RMSLE                                                        1.7324
```

**The gap between our model and a perfect predictor of its own bin means is 0.0010.** Over half
the remaining variance sits *inside* cells the model cannot distinguish, and we have already
captured essentially all of the between-cell signal. A new feature can only help by **splitting
users apart inside a prediction bin**. That is the concrete, measurable bar every candidate now
has to clear, and it is why bundle-level CV deltas keep landing at ±0.00006.

### 2. The task is 81% classification, and the classifier is the whole game

```
var decomposition of L:  between (buy vs not) 4.4555 = 81.3% | within-buyers 1.7963
rho(L, M) overall 0.67306 | rho(1{buy}, M) 0.57915 | rho(L,M) among BUYERS only 0.49863

ORACLES (fold 4):  perfect classifier + CONSTANT for buyers     1.01393
                   perfect classifier + our amount              1.30165
                   our classifier + PERFECT amount              1.14319
                   actual                                       1.73239
```

Two consequences. **(a)** A perfect classifier paired with a single constant (1.014) beats a
perfect *amount* model paired with our classifier (1.143) — DATA.md's 0.983 oracle, re-derived
on the frozen folds. **(b)** Among the bottom 8 prediction bins, `mean_L|buy` is **flat at
~3.5**: conditional on buying, the model barely predicts how much. Amount modelling is not
where the loss is.

### 3. Where the error physically lives, and why it is not addressable

| group | share of users | share of SSE |
|---|---|---|
| false positives (pred_L > 3, bought nothing) | 3.85% | 17.6% |
| false negatives (pred_L < 1, bought) | 3.51% | 11.3% |
| **both** | **7.35%** | **28.8%** |

- **FPs are not a model error.** They are ~30% *more* active than correctly-predicted users at
  **every** window (7/14/30/60/90/180/365 buy-days, active-days, carts). A uniform level shift
  with no timing tell — the model ranks them high because they genuinely are high-propensity;
  they just didn't buy. Irreducible.
- **FNs are active browsers who converted.** 100% active within 30 days, 59.7% within 7, but
  52.5% have **never ordered** in their entire history and the rest last ordered a median 124
  days ago. Restricting to the honest comparison population — 33,916 never-bought users active
  in the last 30d, conversion rate 12.9% — **nothing separates the converters**: carts d=−0.08,
  searches d=−0.12, active-days d=−0.12, tenure d=−0.13, recency d=−0.18, **every effect tiny
  and pointing the wrong way** (converters are *less* active). This is a coin-flip population.

### 4. Fifteen structurally-new candidates, screened by residual correlation

The efficient test is not CV — it is `corr(candidate, model residual)`. A feature the model
already knows correlates with its *prediction*, not its *error*. Kill bar: the year-lag died at
incremental R² = 0.000135 (Band A+).

| candidate | corr(x, resid) | corr(x, pred) | incr. R² |
|---|---|---|---|
| inter-purchase `gap_mean` / `gap_sd` / `n` | −0.002 / −0.002 / +0.002 | −0.78 / −0.59 / **+0.90** | — |
| `overdue` = recency / own mean gap | −0.006 | −0.37 | — |
| `overdue_z` = (recency − gap)/gap_sd | −0.006 | −0.35 | — |
| *(all 5 BTYD-style gap features jointly)* | | | **0.00009** |
| `srch_per_day` | −0.011 | +0.44 | 0.00008 |
| `burst` (activity clustering) | +0.009 | +0.04 | 0.00005 |
| `cart_accel` (30d vs prior-30d carts) | +0.008 | +0.24 | 0.00004 |
| `cart2ord` (lifetime conversion) | +0.008 | +0.49 | 0.00004 |
| `aov` (GMV per order, not per day) | +0.003 | +0.60 | 0.00001 |
| `wknd_ord_share`, `cat_ord_share`, `browse_share`, `gmv_top1_share` | ≤0.002 | — | 0.00000 |
| 5 PCs of level-free 26-fortnight buying **shape** | ≤0.012 | — | 0.00018 |

**Every one is at or below the year-lag kill bar.** The `overdue`/`overdue_z` result is the
most informative: recency-relative-to-own-cadence is the single best-motivated unbuilt feature
in the backlog (it is the BTYD mechanism), it shows d=−0.30 raw separation, **and it carries
0.006 residual correlation** — the model reconstructs it from recency + frequency, which it has.
This closes §1e's BTYD thread from the feature side as well as the model side.

### 5. Per-cell bias correction — fails out of sample, as calibration already predicted

If the model were systematically wrong in identifiable regions, a cell-id feature would fix it.
Split-half correlation of cell-mean residual over 123 cells (8×8×8 on buy-days × recency × GMV,
≥30 users per half): **+0.225** — mostly sampling noise. Out of sample it *hurts*:

```
held-out RMSLE   1.73525 -> 1.73786   (+0.00261)   raw cell means
                          shrink k=10  +0.00200
                          shrink k=50  +0.00086
                          shrink k=200 -0.00012   (shrunk to ~nothing, does ~nothing)
```

Consistent with `calibrate.py`'s global k\* = 1.000 and with §1n. **There is no mispriced region.**

### What this section is for

It converts "we keep failing to find features" into **"there is measurably nothing left for a
feature to find"**, with a reusable test. Any future candidate should be screened by
`corr(candidate, residual)` on the OOF *before* it costs a CV run — it takes seconds, needs no
cluster, and would have killed all 22 prior candidates. **A candidate that does not clear
incremental R² ≈ 0.0002 against the residual cannot move CV.**

> **The remaining levers are not features.** Per §1b rho is the only quantity and RMSLE is
> dominated by classification (81% of variance) — so what is left is the *objective* (e0221's
> HL-Gauss over binned log1p is exactly this shape), and blend diversity, which §1m prices at a
> −0.00004 ceiling. Feature engineering on this dataset is finished.

---

## 1r. The classification head — right target, saturated instrument (2026-08-22)

§1q showed the task is 81.3% classification, so the head is where the leverage is. This section
asks whether we can actually *move* it. Local proxy, 15k users, 3 anchors × 3 seeds, LightGBM
4.6.0. Screen tier — nothing here decides (§4.2) — but the controls are what matter.

### The exchange rate: AUC is worth a lot

Mixing the model score toward the truth and re-measuring both quantities on fold 4:

```
AUC 0.84832 -> best-achievable RMSLE 1.73139     (our actual ranking)
AUC 0.86983 -> 1.67393        AUC 0.89045 -> 1.61241        AUC 0.92885 -> 1.47632
```

**≈ −0.0027 RMSLE per +0.001 AUC.** Against σ_noise 0.00009 that is a 30× payoff. If any lever
in this project is worth pushing, it is the ranking.

### The ceiling: every model we own lands in a 0.0007-wide AUC band

AUC on fold 4 of **all 21 OOFs in `oof/`** — GBDT, XGBoost, AutoGluon, GRU, LSTM, CNN-GRU,
three transformer variants, tuned and untuned:

```
0.84806 ... 0.84876     (e0110 lowest, e0181 highest)   band width 0.0007
BTYD e0170 0.82443 / e0170ey 0.79514  -- the only real outliers, and both WORSE
```

Twenty-one independent attempts, four model families, and nothing separates them by more than
8× σ_noise-equivalent. **The ranking is saturated at AUC ≈ 0.848**, which is the same statement
§1q made from the residual side, measured a completely different way.

### Three interventions, measured

| intervention | result | verdict |
|---|---|---|
| **dedicated `binary` objective vs `regression`** (AUC) | **+0.00113**, 8/9 runs, t = 2.79 | real, tiny |
| **hurdle `P(buy) × E[L∣buy]`** (RMSLE) | **+0.00534, 0/9 runs** | `kill` |
| **classifier score as a feature** (RMSLE) | −0.00851, 9/9 → **−0.00058 under controls** | `kill` |

**The hurdle independently reconfirms e0010** at a modern baseline. e0010 was run on base
features at 1.77836 and could have been dismissed as an artefact of that era; it is not. The
parts multiply in log space, and the decomposition does not make estimation easier.

### ⚠ The stacking result, and why it did not survive

`clf-score-as-feature` first measured **−0.00851, better in 9/9 runs** — ~95× the noise floor
and the largest single-change effect this project has seen. Four controls dismantled it:

```
clf OOF (honest)          -0.00581     <- the claim
NOISE column control      -0.00194     <- a RANDOM column buys a third of it
reg-OOF stack             -0.01295     <- BETTER than the classifier stack
clf IN-SAMPLE (leaky)     +0.24947     <- leak control fires; OOF construction verified
```

The reg-OOF stack winning is decisive: **the gain is stacking, not classification.** And the
round sweep shows it is not capacity either — plain regression gets monotonically *worse* with
more rounds (300→1500: 1.77174→1.84041), because the screen has **no early stopping**. That is
precisely the condition under which an OOF column acts as free regularisation. Re-run with
pipeline parameters (`min_data_in_leaf=200`, `lambda_l2=1.0`) and a real ES anchor:

```
-0.00058  (sd 0.00124, better in 7/9)      <- 15x smaller
and on the most test-like anchor 10-16 it FLIPS POSITIVE in 2/3 seeds
```

> **RULE. A screen without early stopping systematically overvalues anything that regularises.**
> The measured inflation here is 15×. §1p already recorded that a truncated smoke inflates a
> *model-family* gap 4.7×; this is the same disease in the *regularisation* direction. Any
> proxy result should be re-read under the real stopping rule before it is believed.

### Verdict

**No classification-head change earns a confirm slot.** The head is the right place to look —
the exchange rate proves that — but it is not the bottleneck: the *ranking* is, and 21 models
across 4 families agree on it to within 0.0007 AUC. The dedicated binary objective is the only
survivor at +0.00113 AUC, worth ≈ −0.003 RMSLE if it transferred intact, but it optimises a
different loss than the one we are scored on and the hurdle result shows that conversion is not
free. Worth one confirm **only** if a cheap route to the amount head is kept alongside it.

---

## 1r. The magnitude + regime sweep — 15 arms, and the arithmetic of the 5 % target (2026-08-22)

`src/run_magnitude.py` · `src/combine_magnitude.py` · `src/run_regime.py` ·
`configs/e0250_magnitude.yaml` · `configs/e0260_regime.yaml`. Arms **e0250–e0257** and
**e0260–e0266**.

### The target, stated as arithmetic before any result

The brief was "improve tabular models by 5 % from the best tabular submission". Best tabular is
**e0090, LB 1.655247**; −5 % is **1.5725**. Via `RMSLE = sd_L·√(1−ρ²)` with `sd_L = 2.3178`
(§1b), that requires **ρ 0.70209 → 0.7346, i.e. +0.0325**. For scale:

| quantity | value | ratio to the target |
|---|---|---|
| largest member-level gain ever measured here (e0090) | +0.00038 ρ | **86×** |
| §1m recombination ceiling over all 21 predictors | −0.00004 CV | — |
| §I6's measured *upper bracket* on achievable ρ | 0.7254 | **below the target** |

The last row is the binding one and it is not an opinion. §I6's test–retest estimate brackets
achievable ρ at `[0.6871, 0.7254]`, and **0.7254 is an UPPER bound on the ceiling** — it counts
as predictable everything two contemporaneous windows share, including common causes arising
*after* the cut-off that no predictor can know. So:

* the target ρ = 0.7346 sits **above the upper bound on what the data itself contains**;
* even *attaining* that upper bound exactly — a perfect predictor of every knowable trait —
  gives RMSLE 1.5954, a **3.6 %** improvement, still short of 5 %;
* the bound applies to **every competitor**, so no rival submission has reached it either. The
  leader's 0.000688 edge is ~1.3 % of the maximum conceivable remaining gain.

**A 5 % tabular improvement is therefore not reachable by any model on this data.** Stated here
before the results, so the arms below are read as what they are — the best available use of the
compute against the one term with measured headroom — and not as a route to a number the data
does not contain. If the target is a hard requirement, the honest answer is that it cannot be
met, and the reason is a property of the data rather than of the modelling.

### What the sweep does instead, and why these arms

The information axis is closed by direct measurement (§1q: nothing clears incremental R² 0.0002
against the OOF residual; 22 candidates have failed; a noise column beat the last three
confirms). So **no arm here changes a feature.** All 15 run e0049's exact 665 columns.

**Batch 1 — the magnitude term (e0250–e0257).** §I13 is the only place in the record with a
*measured double-digit relative gap*:

| term | ceiling | achieved | captured |
|---|---|---|---|
| buy flag `corr(Z,·)` | 0.6623 | 0.5932 | 89.6 % |
| **magnitude `corr(L,·|Z=1)`** | **0.6001** | **0.4814** | **80.2 %** |

and it names the experiment nobody ran: train on buyers only, score `corr(L,·|Z=1)` *against the
ceiling* rather than end-to-end. Scoring end-to-end is what hid this term for the whole project.
The eight arms vary the loss geometry and capacity on that restricted population — L2 (the
isolation alone), Huber, L1, rank-Gaussian, capacity up, capacity down, e0093's tuned optimum,
and the tuned×Huber interaction. Baseline reproduced locally to five decimals (0.48141 vs
§I13's 0.4814) before anything was submitted.

**Batch 2 — training regime (e0260–e0266).** Two arms are items this file's own notes mark as
never run: **GBDT multi-seed averaging** (§4 note 3 — "done for `nn_seq` … **never run on the
GBDT side** … nearly free") and **anchor recency decay** (`BACKLOG` B3). e0070–e0073 tested
anchor *truncation* and found it flat, but a step function on anchor age and a smooth decay are
different interventions. Plus buyer up/down-weighting as a bracket, and two combinations.

### Three protocol points, each fixing a way this could have produced a fake result

1. **Same-session reference, refitted in every fold.** No arm is compared to e0049's logged
   1.76551. Cross-session drift is measured at +0.00027…+0.00046 for configs with *zero*
   changes — larger than several effects being tested.
2. **`rho|Z=1` is a diagnostic, not a score.** A buyers-only model is not a submission: at test
   time we do not know who buys. `combine_magnitude.py` converts it through the existing
   classifier OOF with a LOO-fitted weight, against two controls (the same combination using
   e0049's own predictions, and classifier × constant). Feeding e0049 in as its own magnitude
   arm reproduces its per-fold scores exactly with the hurdle weight collapsing to ~0 — the
   combiner declines to double-count signal e0049 already has, which is the check that it is
   measuring the arm rather than itself.
3. **Multi-arm ≠ bundled.** Arms share one feature build per fold because building dominates
   cost (~25 min/fold vs ~2 min/fit), but each differs from the reference by exactly one
   declared thing and writes its own OOF and its own `runs/` row.

### ⚠ Caught mid-flight: `rho|Z=1` and RMSLE move in OPPOSITE directions

The smoke returned the buyers-only arms at **+0.0517 on `rho|Z=1`** — which would be, by two
orders of magnitude, the largest effect in this project's history. Per §3.3 a jump larger than
any plausible modelling gain is a leak until proven otherwise, so it was investigated before
the confirm was allowed to matter. It is not a leak; it is worse than that for the diagnostic.

**Isotonic recalibration of e0049's EXISTING predictions — no new model, no new information,
a monotone map fitted leave-one-fold-out — supplies +0.0285 of that +0.0517 by itself.**

The mechanism is already in §1q and was not connected to §I13 until now: among buyers the
all-rows model's bottom 8 prediction bins run 0.61 → 2.95 while actual `mean L` is **flat at
~3.5–3.96**. Pearson correlation is penalised by precisely that curvature. Training on buyers
removes the zero-inflation squash, so most of the "magnitude gain" is a rescaling that costs
nothing and teaches nothing.

Then the decisive measurement, on real OOFs through `combine_magnitude.py`:

```
CONTROL 1   clf x e0049                    RMSLE 1.76582   rho|Z=1 0.48141
CONTROL 2b  clf x e0049-ISOTONIC           RMSLE 1.77106   rho|Z=1 0.50990
            recalibration alone:           RMSLE +0.00524  rho|Z=1 +0.02849
```

**A monotone map cannot add information, and it raises the diagnostic by +0.028 while making
the metric worse by +0.005.** That is direct proof, on this data, that `corr(L,·|Z=1)` is not
a monotone proxy for RMSLE — the conversion failure §I13 flagged as a risk, now measured
rather than extrapolated. §I13's "roughly +0.012 rho" estimate for closing the magnitude gap
should be treated as retired on the same grounds §1b retired its own `d(rho)/d(AUC)`.

**Consequence, applied before any arm was scored:** the combiner now carries two bars, because
they are different claims — `info` (beats the isotonic control on `rho|Z=1` ⇒ genuinely new
information) and `metric` (beats control 1 on LOO RMSLE ⇒ actually improves the score). Only
both together is a keep. An arm that is `info` alone has walked into the §I13 trap.

### Pre-registered bars

Keep on ≥4/5 folds **and** better, or |Δ| > 2σ_noise (0.00009). A sub-2σ delta is `no effect`
regardless of sign — e0060 flipped sign on LB and cost 0.0005 by being promoted on 0.4σ.
For the magnitude arms, additionally: the RMSLE bar is control 1 and the information bar is
control 2b, and a `keep` requires clearing both.

### RESULT — the magnitude confirm: real information, zero score. All 8 arms `kill`.

Full 5 folds, all anchors. The reference reproduces `oof/e0049.parquet`'s ρ|Z=1 **exactly** on
every fold (0.47012 / 0.46594 / 0.48066 / 0.49339 / 0.49863), so the harness is validated
end-to-end against an independently computed baseline.

```
reference (all-rows L2)   rho|Z=1  0.48175 ± 0.01270
arms                               0.53675 .. 0.53956      all 5/5 folds, ~89.7% of ceiling
```

Every arm gains **+0.055 ρ|Z=1 on 5/5 folds** — and the isotonic control (§1r) accounts for only
+0.0285 of it, so **+0.028 is genuinely new information, not the free rescaling.** §I13 was
right that the magnitude term has unclaimed signal.

Then the conversion, through the existing classifier with a LOO-fitted hurdle weight:

| arm | ρ\|Z=1 | Δ vs isotonic ctrl | LOO RMSLE | Δ vs control 1 | verdict |
|---|---|---|---|---|---|
| e0253 rank-Gaussian | 0.53873 | **+0.02883** | 1.76607 | **+0.00025** | kill |
| e0256 tuned | 0.53956 | **+0.02967** | 1.76650 | **+0.00068** | kill |
| e0250 L2 (isolation) | 0.53883 | **+0.02893** | 1.76660 | **+0.00079** | kill |
| e0255 low capacity | 0.53862 | +0.02872 | 1.76666 | +0.00084 | kill |
| e0257 tuned×Huber | 0.53850 | +0.02861 | 1.76673 | +0.00091 | kill |
| e0251 Huber | 0.53770 | +0.02781 | 1.76690 | +0.00108 | kill |
| e0254 high capacity | 0.53763 | +0.02773 | 1.76694 | +0.00112 | kill |
| e0252 L1 | 0.53730 | +0.02740 | 1.76715 | +0.00133 | kill |

**All eight are `info`. None is `metric`. Not one earns a keep.** Every arm carries real new
information about how much a buyer spends, and every arm makes the competition score *worse*.

**This is the sharpest result the sweep produced**, and it needed the two-bar design to be
visible at all: with only the ρ|Z=1 column these would read as the largest wins in the project's
history (+0.055 on 5/5 folds, 8 for 8). §I13's warning — that its own "+0.012 ρ" estimate came
from an oracle path §1b had already retired — is now confirmed with a measurement rather than
inherited as a caution.

**Why it cannot convert.** §1q measured the task as 81 % classification variance, and among the
bottom 8 prediction bins `mean L|buy` is flat at ~3.5 — the model barely distinguishes *how
much* a buyer spends, and improving that further does not move a score dominated by *whether*
they buy. Better magnitude estimates get multiplied by the same classifier and the composition
loses more than the magnitude term gains.

**Closed:** the loss-geometry axis on the buyer subset (L2 / Huber / L1 / rank-Gaussian, three
capacities, the tuned optimum, and one interaction), and with it IDEAS.md §I13's proposed next
experiment — run exactly as specified, with the answer that the term is real and unreachable.

#### And they are the cleanest instance of §1c's rule the project has produced

The buyers-only arms estimate a **different functional** (`E[L|x, buyer]`), which §1s argued is
the property that ought to decorrelate. It does — dramatically:

| family | r vs the 9-member blend | own ρ_B | ρ_partial | % of the 0.04067 bar |
|---|---|---|---|---|
| **magnitude arms** | **0.87** (most decorrelated we own) | **0.58** | +0.0068 … +0.0076 | ≤ 18.6 % |
| tabular members | 0.997 (least decorrelated) | 0.661 | ≤ +0.0127 | ≤ 31 % |

LOFO-honest, adding the best of them (e0256) to the blend: **−0.00001, 3/5 folds.**

**Neither family clears the bar, and they fail for opposite reasons — the magnitude arms lack
quality, the tabular arms lack difference.** §1c says decorrelation is only worth something *at
comparable quality*; here is that sentence as a measurement, with both failure modes exhibited
side by side by artefacts built in the same sweep. Dropping ρ_B by 0.08 to buy a 0.13 drop in
correlation is a losing trade, and the algebra of §1f says exactly why: the gain goes as the
square of the excess `e = ρ_B − r·ρ_M`, which stays near zero in both rows.

### Screen-tier readings (2 folds, 2 anchors — NOT decisions, §4.2)

**Magnitude smoke — the eight arms are indistinguishable from each other.**

```
reference (all-rows L2)   rho|Z=1 0.49590
arms                      0.54583 .. 0.55110      spread 0.00527
common shift off ref                              +0.05378   = 10x the spread
```

Changing the loss (L2 → Huber → L1), the target parametrisation (rank-Gaussian), the capacity
(63 / 255 / 31 leaves) and the full tuned optimum move `rho|Z=1` by **less than 0.005
combined**, while the bare act of restricting to buyers moves it **+0.054**. Every arm is
measuring one thing — the population restriction — and §1r's isotonic control shows +0.0285 of
that is free recalibration which makes RMSLE *worse*. **This is independent evidence against
IDEAS.md §0's loss-geometry thesis**, from a different direction than I1's HL-Gauss confirm
(which lost by +0.044/+0.038 on its first two folds): when eight loss variants land inside a
band ten times smaller than their common offset, the loss is not the active ingredient.

**Regime smoke — two arms move, three are flat, two are badly worse.**

```
reference (e0049 regime, refit)   1.75106
e0266  tuned + 3-seed averaging   1.74952   -0.00154   <- best
e0260  5-seed averaging           1.74973   -0.00133
e0261  anchor decay hl=60d        1.75104   -0.00002
e0262  anchor decay hl=180d       1.75116   +0.00010
e0264  buyer weight 0.5           1.80233   +0.05127
e0265  seeds + buyer weight 2.0   1.80638   +0.05532
e0263  buyer weight 2.0           1.81015   +0.05909
```

Three results worth keeping even at screen tier, because two of them are *brackets* and a
bracket closes a question rather than leaving it open:

* **Buyer re-weighting is closed.** Both directions hurt, and hurt symmetrically (+0.059 up,
  +0.051 down). L2's implicit row-count weighting between the classification and magnitude
  terms is already at its optimum; this was the cheapest available test of §1q's 81 %/19 %
  variance split and the answer is that the split is not mis-weighted. e0265 additionally
  shows a bad component dominates a good one in combination — deltas do not add (§4.1).
* **Anchor recency decay is closed**, at two half-lives spanning 3×. The decay surface is as
  flat as e0070–e0073's truncation surface, so `BACKLOG` B3 can be retired rather than left
  open as "never run".
* **Seed averaging is the live arm**, and it is larger at the tuned low-learning-rate setting
  (e0266, 1162 rounds) than at the default (e0260, 178 rounds) — which is the direction
  predicted in the config before the run, since a slower learner has more seed-to-seed path
  variance to average away.

### ⚠ And seed averaging cannot reach the champion — measured, not assumed

Variance reduction on one member is worth nothing to a blend that already averages nine. On
`oof/e0120`'s members, shrinking e0049 toward the consensus of the other eight (an upper bound
on what any variance-reduction move can do, since it is the limit of averaging infinitely many
seeds):

| shrink | member ρ | Δ member | blend ρ | Δ blend | Δ blend RMSLE |
|---|---|---|---|---|---|
| 10 % | 0.66146 | +0.00033 | 0.66254 | −0.00000 | **+0.00001** |
| 20 % | 0.66174 | +0.00061 | 0.66253 | −0.00001 | **+0.00002** |
| 30 % | 0.66198 | +0.00085 | 0.66253 | −0.00001 | **+0.00002** |

**The member gains up to +0.00085 and the blend moves by +0.00002 — the wrong way.** The
mechanism is the same one §1m priced for e0090 (+0.00038 member ρ → +0.00001 blend, ~40×
dilution), but sharper here: averaging seeds removes exactly the idiosyncratic component that
blending *already* removes, so the two are substitutes rather than complements.

**Consequence:** e0260/e0266 are real gains for a **standalone tabular submission** and are
worth ~0.14–0.17 % on LB at the measured 1.5–1.8× transfer rate. They do **not** improve the
champion blend, and a seed-averaged member should not be added to it expecting one.

### The mechanism, checked in closed form rather than asserted

Averaging `n` seeds cuts the seed-variance component of `E[(L−M)²]` by `(1−1/n)`, so

    ΔRMSLE ≈ −(1 − 1/n) · s² / (2·RMSLE)

for a per-prediction seed sd of `s` in log space. Against the 2-fold screen:

| arm | n | observed Δ | implied `s` |
|---|---|---|---|
| e0260 | 5 | −0.00170 | ≈ 0.085 |
| e0266 | 3 | −0.00167 | ≈ 0.095 |

Both imply `s ≈ 0.08–0.09`, which is an ordinary amount of seed-to-seed disagreement for a
GBDT at `feature_fraction 0.8` / `bagging_fraction 0.8`. **The gain is therefore textbook
variance reduction and carries no new information** — which is simultaneously why its sign was
guaranteed in advance (worth running despite §1m) and why it cannot survive into the blend
(§1m's dilution table above). A result that is both real and structurally unable to reach the
champion is worth stating plainly rather than filing as a win.

---

## 1s. The data's own ceiling — an ICC bound on rho, from repeated measurements (2026-08-22)

§1q and §1r asked "what is left for a feature / a head". This section asks the prior question:
**how much of the target is predictable by anything at all?** The folds make this measurable
without a single training run, because 176,982 users are scored in **all five** folds — five
repeated 30-day measurements of the same person. That is a classic variance-components design
and nobody in this project had used it.

### The target is 56% stable user + 44% month-to-month randomness

```
var(user mean log-GMV, observed)   3.2523        var(within user, across windows)  2.2416
var(TRUE user effect) = 3.2523 - 2.2416/5      = 2.8040    (correcting for a 5-window mean)

ICC (naive)      0.5920   ->  rho ceiling  0.7694
ICC (corrected)  0.5557   ->  rho ceiling  0.7455      <- the honest bound
our fold-4 rho (these users)      0.6619
```

**No model of any kind can exceed rho ≈ 0.745 on this task**, because 44% of the variance in
a user's 30-day log-GMV is *not a property of the user* — it is which month they happened to
buy in. Priced in RMSLE on this population: ceiling **1.554**, ours **1.748**, headroom **0.194**.

### The within-user component is genuinely unpredictable

Before trusting that bound, the within-user part must be noise and not drift we could model:

- lag-1 autocorrelation of within-user deviation **−0.192**, lag-2 **−0.280** — a 5-window mean
  mechanically induces ≈ −0.25, so this is **pure noise with no persistence**.
- The one positive signal — a per-user **trend** over windows 0–3 predicting the deviation at
  window 4 at **+0.0943** — is already inside the model: `corr(trend, residual) = +0.0008`,
  `corr(trend, prediction) = +0.1269`, **incremental R² = 0.00000**.

### The model is already past the raw persistence of the target

```
adjacent-window target autocorrelation      0.520      (the user's own last window)
user's own mean of 4 past windows -> fold 4 rho 0.6176
OUR MODEL                                   0.6619
```

We beat both. And adding the 4-window user mean to the model gives **incremental R² = 0.00000**
— the 365-day features already extract the stable component optimally. Shrinking the fold-4
prediction toward the model's own past predictions is worst at every w>0 (best `w=0`), which is
§1n's anchor-averaging kill arriving again from a different direction.

### Two structural checks that came back clean

1. **The CV is statistically honest.** Per-user residuals across the five folds are
   **uncorrelated** (mean off-diagonal −0.0037), so the folds are five genuinely independent
   measurements — `σ_noise` from seeds is not understating uncertainty, and the effective n is
   the full 1.07M rows.
2. **The fold population matches the test's.** Recency distribution at fold 4 vs test:
   0d 37.9% / 36.5%, 1d 54.2% / 52.6%, 7d 84.0% / 82.8%, 30d 100% / 100%. The §4.4 re-selection
   rule reproduces the test population closely; 12.4% of scored fold-4 users would fail the
   test's 3-block rule, but DATA.md §9.4 already measured that filter as **harmful** (corr
   0.528, RMSLE 2.279) — it over-selects. Nothing to fix.

> **One caveat on the bound.** The ICC ceiling assumes a model could know each user's *true*
> stable level exactly. It is an upper bound on what feature and model work can ever buy, not a
> promise that 0.194 is reachable — the reachable part is limited by how well history estimates
> that level, and §1q shows we are already within 0.0010 of the conditional mean of our own
> prediction bins. **Read §1s as the reason the search keeps returning nulls, not as headroom.**

### Why this matters more than another feature test

It converts the project's central puzzle — "21 models across 4 families all land at AUC 0.848
and rho 0.67" — from a coincidence into a **consequence**. They agree because they are all near
the information limit of a target that is 44% coin-flip. The remaining levers do not raise
`rho` against this bound; they exploit `RMSLE = sd_L·√(1−rho²)` at fixed rho, which is precisely
what §1b's calibration route already does.

---

## 1t. Segment-varying calibration — pre-registered, and it failed as predicted (2026-08-22)

Written down in `scratch_thoughts.md` **before** any code ran, including the expected outcome
(70/30 against) and a falsifiable diagnostic. Exp **e0239**, screen tier, no training.

### The hypothesis

Minimising `RMSLE² = (sd_L − sd_M)² + 2·sd_L·sd_M·(1−rho) + (E[L]−E[M])²` over `sd_M` gives the
shrinkage optimum `sd_M* = rho · sd_L`. On fold 4: `sd_L = 2.342`, `rho = 0.673` → `sd_M* =
1.576`, and the model's actual `sd_M = 1.577`. **The GBDT sits exactly on the global optimum**,
which is *why* `calibrate.py` returns `k* = 1.000` and every monotone map in e0235 loses.

But rho is **not global** — §1s measured 0.66 for users present in all five folds against
0.47–0.52 for intermittent users. If reliability varies by segment, so should the optimal
shrinkage: `sd_M*(g) = rho(g)·sd_L(g)`. A globally-optimal model could still be over-confident
where it is unreliable and under-confident where it is not.

**Not covered by prior work:** `calibrate.py` is one global parameter; e0235 fits monotone
`g(M)` — any function of M *alone*, which cannot express "same M, different shrinkage because
this user is intermittent"; §1q corrected per-cell **means** (intercept), this is the **slope**.

### The pre-registered diagnostic, and the answer

> *"If the model is conditionally calibrated, the per-segment regression `L = a_g + b_g·M` will
> give b_g ≈ 1 for every segment."*

```
segment (active 30d-blocks of last 12)   n(f4)     b_g     rho_g    sd_M   rho*sd_L
 1 (most intermittent)                    4032   0.9975   0.4717   0.896   0.894
 8                                       15541   0.9991   0.5374   1.111   1.110
10 (always active)                      138438   1.0030   0.6558   1.508   1.513
                                    GLOBAL      0.9995
```

**b_g ∈ [0.9976, 1.0252], sd 0.0093, every one within 0.025 of 1.0** — and `sd_M` already tracks
`rho·sd_L` *inside every segment*. The model is conditionally calibrated, exactly as
"L2 estimates E[L|x]" implies. The hypothesis was dead before the correction was even applied.

### Applied anyway, with the control that matters

| | mean Δ | folds better |
|---|---|---|
| **real segments** | **+0.00024** | 2/5 |
| global affine (1 segment) | +0.00023 | 1/5 |
| **RANDOM segments, matched sizes** | **+0.00025** | 1/5 |

**Real is indistinguishable from random** — the e0214 lesson reproducing exactly, on a different
question. Strictly-forward (fit folds 0–3 → apply fold 4): **−0.00001**, nil. `kill`.

### Why the null is worth the section

It closes the **segment-calibration direction**, which was the last one I believed open after
§1q/§1r/§1s. And it extends §1q's central result: `E[L|M] ≈ M` to within 0.0010 globally is not
just a marginal property — it holds **conditionally**, inside every reliability segment. There
is no sub-population where the model's confidence is mis-set.

> **The pre-registration is the transferable part.** The prediction ("this fails, ~70/30") and
> the diagnostic ("alive only if b_g departs from 1") were both recorded before the numbers
> existed, so neither could be fitted to the outcome afterwards. Cost: 7 minutes on existing
> OOF, zero cluster time. **Do this for every remaining hypothesis** — most of what is left is
> cheap to settle on OOF, and a hypothesis with a stated kill condition dies in one run
> instead of becoming a "borderline" that needs a confirm slot.

---

## 1s. Can a better TABULAR member reach LB 1.6450? — the requirement, priced (2026-08-22)

Asked directly: reach **1.6450** from the champion's 1.646602 by improving the tabular part of
the blend. Unlike the −5 % target of §1r this is **not** excluded by the §I6 ceiling — it needs
blend ρ 0.70378 → 0.70448, `+0.00070`, comfortably inside the `[0.6871, 0.7254]` bracket, and
it is a 4.2σ move on the paired 50k-user LB noise. So it was priced rather than dismissed.

### 1. Weight tuning is exhausted (as §1m said)

Sweeping the gbdt weight in the `0.2·gbdt + 0.8·seq` structure on the frozen folds: the
optimum is w = 0.30 at ρ 0.66255 against w = 0.20 at 0.66253. **Worth +0.00002**, reproducing
§1m's −0.00004 from a different direction. Nothing to collect here.

### 2. What the tabular member would have to deliver

Holding the weight at 0.20 and raising only the gbdt member's ρ:

| member ρ gain | blend ρ gain |
|---|---|
| +0.001 | +0.00020 |
| +0.005 | +0.00099 |
| +0.010 | +0.00198 |

So `+0.00070` blend ρ needs roughly **+0.0035 member ρ** — about **9× e0090's +0.00038**, the
largest member-level gain in the project's history.

### 3. Stated properly, as a partial correlation — and this is the binding number

The weight-and-ρ framing understates it, because a better member that is also more correlated
with the blend buys less. The admissibility algebra (§1f) gives the honest requirement:

```
required rho_partial for +0.00070 blend rho:   0.04067   (CV scale)
                                               0.04410   (LB scale, solved test moments)
```

Two independent scales, same answer. Against it, **every tabular member ever built**:

| member | ρ_B | r vs blend | ρ_partial | % of requirement |
|---|---|---|---|---|
| e0064 (AutoGluon) | 0.66128 | 0.99708 | +0.01269 | **31.2 %** |
| e0049 (LightGBM 665) | 0.66113 | 0.99682 | +0.01255 | 30.9 % |
| **e0210 (XGBoost, new)** | 0.66093 | 0.99727 | +0.00358 | **8.8 %** |
| e0191 (tsfresh) | 0.66115 | 0.99737 | +0.00657 | 16.1 % |
| e0020 (185 feat) | 0.66069 | 0.99701 | +0.00224 | 5.5 % |

**The target needs 3.2–3.5× the most decorrelated member this project has ever produced.**

And because the gain is *quadratic* in ρ_partial, being 3× short is being ~10× short in value:

```
member at 100 % of the required rho_partial -> LB 1.645000
member at  50 %                             -> LB 1.646202
member at  31.2 % (= the best ever built)   -> LB 1.646446
```

**Even a new member matching e0064's best-ever decorrelation lands at 1.646446 — 0.00016 better
than the champion, and 0.00145 short of the target.**

### 4. A genuinely different family does not help, measured

e0210 is XGBoost — a different library, different split-finding, different regularisation, on
the same 665 features. It correlates with e0049 at **0.99896**. That is the §1c finding again
and it is worth restating in its sharpest form: **function-class diversity at fixed features
and fixed loss does not produce decorrelation.** Members disagree about fit, never about what
to fit.

Honest LOFO-NNLS over **all 16 frozen-fold predictors we own** (weights fitted on 4 folds,
scored on the 5th): **−0.00001** against the e0162 structure, per-fold
`[−0.00010, 0.00000, −0.00002, +0.00006, −0.00000]`. An independent reproduction of §1m's
recombination ceiling, now including the two members built since it was written.

### 5. What DOES clear the bar — and why it is already spent

Screening every predictor we own against the 0.04067 requirement produces exactly one class of
hit, and it is not a tabular model:

| candidate | ρ_partial vs the 9-member blend | % of requirement |
|---|---|---|
| **usercv_tuned** (user-split GRU) | **+0.04444** | **109 %** |
| usercv_full (user-split GRU) | +0.03353 | 82 % |
| logit of the buy classifier | +0.00583 | 14 % |
| every frozen-fold tabular member | ≤ +0.01269 | ≤ 31 % |

Measured on the honest intersection: 1,062,003 shared `(fold_id, user_id)` keys, `y_true`
agreeing on **100.000 %** of them, no imputation (the naive full-frame version inflates this,
which is why §C of REVIEW_NOTES exists). LOFO-honest — weight fitted on four folds, scored on
the fifth — adding it gives **+0.00061 blend ρ, winning 5/5 folds**, chosen weights a stable
0.66–0.72.

**That clears the +0.00070 bar almost exactly. And it is already banked.** e0162 *is*
`gbdt 0.20 / seq 0.38 / e0141 0.42`, and e0141 is the user-split GRU family. The +0.00061
measured here is the gain that produced the champion in the first place — measured against the
9-member CV blend (e0120's structure), not against e0162. `cv_lb.csv` records e0162 at
ρ 0.70378, "exactly on its ceiling".

This is the useful form of the result: **the one thing that ever cleared the admissibility bar
was a different CV protocol, not a different model.** The user split changes the inductive bias
(unseen users rather than a future window), which is a structural reason to disagree —
precisely the property §1c says decorrelation requires and that XGBoost-vs-LightGBM does not
have.

### 6. The user-split TABULAR slot is negative — the last structural idea, closed

If a different CV protocol is what buys decorrelation (§5), the obvious follow-up is a
*tabular* model under that protocol. We own three, built untuned for §1c. Scored against the
**e0162 structure** (`0.20 gbdt / 0.38 seq / 0.42 usercv`), where the requirement is
ρ_partial ≥ 0.04074:

| member | ρ_B | r vs champion | ρ_partial |
|---|---|---|---|
| usercv_catboost | 0.64407 | 0.97250 | **−0.00601** |
| usercv_xgboost | 0.64353 | 0.97145 | **−0.00502** |
| usercv_ridge | 0.62209 | 0.94044 | **−0.00692** |
| usercv_full_mixnaive | 0.66326 | 0.99812 | +0.02508 |

**All three user-split tabular members are negative** — adding any of them makes the champion
worse. The protocol is not transferable to the tabular family: it helps the GRU (already in at
weight 0.42) and hurts the trees. That closes the last structural slot this analysis
identified, and it also disposes of the standing "CatBoost tuned on the frozen folds" idea from
REVIEW_NOTES §D — the family's problem is not tuning, it is that it has nothing to add.

### 7. The best honest recombination of everything we own

LOFO-NNLS (weights fitted on four folds, scored on the fifth) over **all 22 predictors**,
aligned on the 1,062,003-key intersection, against the e0162 structure:

```
best LOFO blend rho   0.66311      e0162 structure   0.66286      d +0.00025
per-fold d            [+0.00027, +0.00033, +0.00012, +0.00033, +0.00021]   wins 5/5
implied LB at 1:1 transfer   1.646031     (champion 1.646602, target 1.6450)
```

**+0.00025, 5/5 folds** — a real result, honestly measured, worth about **−0.00057 LB** if it
transfers 1:1. It would beat the champion and land near public top-1 (1.645914). It does not
reach 1.6450, and it is a *recombination* gain rather than a tabular-member gain, so it is not
what this goal asked for either.

### Conclusion

**1.6450 is not reachable through the tabular member.** Not because the score is at a physical
ceiling — §I6 leaves room, and the target is inside the bracket — but because the tabular
family is at *its* ceiling: it enters at weight 0.20, correlates 0.997+ with the blend, and
would need a decorrelation 3.2× beyond anything the project has produced in ~100 experiments.
The only artefact that has ever met the bar is the user-split GRU, whose contribution is
already inside the champion; the user-split *tabular* members are all negative against it.

What is actually available, measured and LOFO-honest, is **+0.00025 blend ρ ≈ −0.00057 LB**
from re-solving the weights over all 22 predictors — champion 1.646602 → ~1.64603. That is
worth submitting on its own merits. It is **not** 1.6450, and no arrangement of the tabular
members gets there.

### 8. The projection identity, validated on the champion itself

Before trusting any of the above, the identity was checked against a known answer:

```
champion measured test rho 0.70378  ->  sd_L*sqrt(1-rho^2) = 1.646607
champion actual LB                                          = 1.646602
```

**Agreement to 5e-6.** So applying a CV ρ delta to the champion's measured test ρ and
re-inverting is sound arithmetic, not an extrapolation, which is what lets §7's +0.00025 be
quoted as ~1.64603 rather than as a hope. (It also re-confirms §1b: after affine calibration
the score *is* one number.)

The recombination therefore delivers **36 % of the +0.00070 the target needs.**

### 8b. ⚠ The §7 recombination was contaminated by e0145, and catching it cost 24 % of the gain

Building the submission forced a check §7 had not done: **which OOF corresponds to which
submitted file.** The answer disqualified the largest-weighted member.

`oof/usercv_tuned.parquet` is **e0145** — the tuned GRU that FAILED on the leaderboard. Its
record, side by side:

| member | sub | CV ρ | measured LB |
|---|---|---|---|
| usercv_full | e0141 | 0.66301 | **1.6488** |
| **usercv_tuned** | **e0145** | **0.66331 (highest)** | **1.65323 (worst)** |
| usercv_full_mixnaive | e0143 | 0.66326 | not submitted |

**It has the best CV ρ of the three and the worst LB**, because its Optuna objective was
min-of-N early-stopped evaluations (§1j) — its CV is inflated by exactly the selection bias
that section documents, and `cv_lb.csv` records the failure as "ρ 0.70311 → 0.70210, almost
exactly cancelling the −0.00099 its CV claimed".

Fitting blend weights on CV therefore *over-weights it precisely because it is biased*, and
§7's LOFO gave it **0.330 — the largest weight in the blend.** The +0.00025 was contaminated.

Re-fitted over only members that have both an aligned OOF and a real test file, with e0145
excluded:

```
LOFO rho 0.66301  vs e0162 structure 0.66282   d +0.00019   wins 5/5
per-fold [+0.00021, +0.00025, +0.00008, +0.00021, +0.00019]
projected LB 1.646172   (champion 1.646602, target 1.645000)
```

**+0.00019 rather than +0.00025** — the contamination was worth 24 % of the apparent gain.

**The generalisable rule:** a CV-fitted blend weight is only as trustworthy as the CV of its
members, and one member with an inflated CV silently redistributes weight toward itself. Any
future weight fit must exclude members whose CV is known to be optimistic **before** fitting,
not after — the fit cannot detect it, because the inflation looks exactly like skill.

### 8c. The reweighting ceiling — 41 % of the target, and that figure is deliberately optimistic

The submission was built, so the remaining question is whether *any* rearrangement of what we
own could reach 1.6450. Unconstrained in-sample OLS over the 11 submittable members — no
non-negativity, no LOFO, fitted and scored on the same rows, i.e. the most optimistic number
this data can produce:

```
unconstrained OLS rho   0.66362      e0162 structure   0.66333      d +0.00029
honest LOFO-NNLS                                                    d +0.00019
required for LB 1.6450                                              d +0.00070
```

**Even the optimistic ceiling reaches 41 % of the requirement** (implied LB 1.645944); the
honest figure reaches 27 %. Reweighting is therefore closed as a route to the target, not by
argument but by an upper bound: no weight vector over these members exists that gets there.

1.6450 requires a **new member**, and §1s prices that at ρ_partial ≥ 0.0407 against a best-ever
0.01269.

### 8d. RESULT — e0270 submitted, and it LOST. CV +0.00019 → LB −0.00010 (2026-08-22)

Both files were scored:

```
e0270_blend (uncalibrated)   LB 1.647898   -> solved rho 0.70369
e0270_cal   (calibrated)     LB 1.646836   -> solved rho 0.70368
champion e0162                  1.646602   ->        rho 0.70378
```

**The two files solve to ρ within 6e-6 of each other** — an independent confirmation of §1b's
algebra from two submissions of the same predictor at different calibrations, and it means the
measurement below is trustworthy rather than a noisy LB draw. Calibration behaved exactly as
predicted (−0.001062, and e0270_cal sits at `sd_L·√(1−ρ²)` to six decimals — **at its ceiling**).

**The blend is 0.00010 ρ WORSE than the champion.** CV predicted +0.00019. The covariance
*lower bound* of §8 (ρ ≥ 0.70348) held; the CV-delta point estimate did not.

#### Why — and it is the same failure I caught once and missed twice

`usercv_full_mixnaive` (**e0143**) carries the **second-largest weight, 0.3656**, and has
**never been scored on the leaderboard.** Its CV ρ was the only evidence for it, and CV ranked
it *above* `usercv_full`/e0141 (0.66326 vs 0.66301, +0.00025). e0141's LB is measured (1.6488);
e0143's is not. The blend that leans 37 % on the unscored sibling lost to the one that does not.

That makes **two independent demonstrations that user-split GRU CV does not rank these models
correctly on test**:

| model | CV ρ rank | test evidence |
|---|---|---|
| e0145 (usercv_tuned) | **best** (0.66331) | **worst LB** — 1.65323 vs 1.6488 (§8b, early-stopping bias) |
| e0143 (usercv_full_mixnaive) | 2nd (0.66326) | never scored; the blend leaning on it loses 0.00010 ρ |
| e0141 (usercv_full) | 3rd (0.66301) | **the only measured one**, and the champion's choice |

I excluded e0145 for exactly this reason and then let its sibling in at weight 0.37 on the same
kind of evidence. **The rule that generalises: a blend weight fitted on CV is only as good as
the weakest CV among the members it up-weights, and an unscored member is not evidence of
quality — it is an untested hypothesis carrying 37 % of the file.**

#### The transfer record, now three for three in the same direction

```
e0090   member CV +0.00038  ->  blend LB +0.00001    dilution (§1m)
e0152   9 LB-fitted weights ->  tie, no gain         LB overfitting
e0270   blend CV +0.00019   ->  blend LB -0.00010    CV-fitted weights on unscored members
```

**Every attempt to improve the champion by re-solving weights has now failed, by three
different mechanisms.** e0162 remains champion at 1.646602. Combined with §8c's ceiling — even
unconstrained in-sample OLS reaches only 41 % of the target — the reweighting axis is closed
with a measured result rather than an argument.

### 9c. ⚠ CORRECTION — §9/§9b's prediction was WRONG IN SIGN on the real artefact (2026-08-23)

§9 predicted, and §9b "confirmed" via the `e0101` seed replicates, that a seed-averaged member
would have **more negative** ρ_partial. When e0260/e0266's actual OOFs arrived that was
measured directly, and it is false:

| member | ρ_B | r vs blend | ρ_partial | % of the 0.04067 bar |
|---|---|---|---|---|
| e0049 | 0.66113 | 0.99749 | +0.00478 | 11.8 % |
| e0210 (XGB) | 0.66093 | 0.99727 | +0.00358 | 8.8 % |
| e0260 (5-seed) | 0.66129 | 0.99772 | +0.00519 | 12.8 % |
| **e0266 (tuned + 3-seed)** | **0.66166** | 0.99790 | **+0.01052** | **25.9 %** |

**e0266's ρ_partial is POSITIVE and 2.2× e0049's** — the best new tabular member the project
has built. And in the champion structure with weights untouched, e0266 in the gbdt slot wins
**5/5 folds**.

**Why the proxy misled.** §9b averaged four *independent seeds of one config* (`e0101s1-3`),
which is pure variance reduction — nothing but noise removed. e0266 changes **two** things at
once relative to e0049: seeds *and* e0093's tuned hyperparameters. The tuning is what moves
ρ_partial; the averaging alone (e0260, +0.00519 vs e0049's +0.00478) barely does. Reasoning
from a proxy that isolated only one of the two components produced a confident wrong sign.
**The lesson is the one this file keeps relearning: a proxy is a hypothesis, and §9b should
have been labelled `predicted` rather than `CONFIRMED` until the real OOF existed.**

**But the gain still does not reach the goal.** 5/5 folds and +0.00001 blend ρ:

```
champion structure (e0049+e0064 in the gbdt slot)   rho 0.66282   -> LB 1.646607
e0266 + e0064                                       rho 0.66284   -> LB 1.646578   5/5
e0049 + e0064 + e0266                               rho 0.66284   -> LB 1.646581   5/5
needed for 1.6450                                   d +0.00070
```

A **+0.00018 measured member ρ gain dilutes to +0.00001 at weight 0.20** — an 18× dilution,
the same mechanism §1m priced for e0090 (+0.00038 → +0.00001, ~40×). Consistent, directional,
and two orders of magnitude short of the target.

### 9e. NEW CHAMPION e0300_cal = 1.646589, and the usercv slot rejected before submission

**e0300_cal (submitted as `e0271_cal`; renamed — e0271-e0276 were concurrently claimed by a
usercv_andrena sweep, the FILE is unchanged) scored 1.646589 — a new best**, +0.000013 over
e0162, solved ρ 0.70379.

**The projection was correct for the first time this session:** predicted 1.646578 from a CV
delta of +0.000012, actual 1.646589, error **+1.1e-5**. The reason is structural — **no weights
were fitted**, only the gbdt slot's *contents* changed (e0266 for e0049), so there was no
fitting risk to mis-project. That is exactly what e0270 got wrong.

Variant selection was done on OOF before any test file was touched: `e0266+e0064` won **5/5
folds including the last** with min-fold +0.000005, and was preferred over `e0266` alone, which
had a larger mean but a **negative** fold.

⚠ **Honest size:** +0.000013 is **0.03σ** on the 0.00038 paired-50k noise. A new best on the
board, *not* statistically distinguishable from e0162.

#### The usercv slot looked 8× better — and it is a CV artefact

Having exhausted the gbdt slot, the other two slots were tested (weights fixed, one slot at a
time):

| change | Δ ρ | folds won |
|---|---|---|
| **usercv: e0141 → mean(e0141, e0143)** | **+0.00010** | **5/5** |
| usercv: e0141 → e0143 alone | +0.00006 | 4/5 |
| seq: drop the 3 seed duplicates | −0.00002 | 0/5 |

+0.00010 is **8× the gbdt-slot gain**. e0143 is unscored, so — per §8d's lesson — the mechanism
was separated from the member's quality by averaging in **e0145 instead, whose LB is measured
and *worse*** (1.65323 vs e0141's 1.6488):

```
e0141 + e0143 (unscored)          CV d +0.00010   5/5
e0141 + e0145 (measured WORSE)    CV d +0.00009   5/5
```

Nearly identical, which said the gain was variance reduction across GRU runs rather than
e0143's quality — a mechanism that would not need an unscored CV. **Then the same question was
put to the test-measured ρ's, where nothing is inferred:**

```
e0141  rho 0.70311 (LB 1.6488)      e0145  rho 0.70210 (LB 1.65323)     corr 0.99747
rho(mean of the two) = 0.70306   ->  -0.00005 vs e0141 alone
```

**CV says +0.00009, the measured-ρ algebra says −0.00005. Sign inversion.** The user-split GRU
CV is now demonstrably wrong in **ranking** (e0145: best CV, worst LB), in **level** (e0143,
inferred from e0270's loss), and in **the value of averaging them**.

**The +0.00010 was not submitted.** It is the same evidence class that cost e0270, caught this
time before spending a submission rather than after. The only usercv fact that survives contact
with the leaderboard is e0141's own 1.6488, already in the champion at weight 0.42.

### 9g. TabICLv2 — a different FUNCTION CLASS breaks the correlation floor, and still falls short

`src/run_tabicl.py` · `configs/e0340_tabicl.yaml` · exp **e0340** (job 24115504, A100, 171 min).

**Why this was run after §9d declared the axis shut.** §I17's scale-penalty gate (e0281–e0286)
priced what a TFM's row-context constraint costs in *quality* and concluded "weaker near-twin".
That is correct about quality and **structurally silent about disagreement** — every gate arm
was still LightGBM, so the gate could not measure the one thing the blend pays for. The
requirement made the gap explicit: at the champion's ρ 0.66335, a member of TabICL's expected
quality clears the bar **iff r ≤ 0.9866**, i.e. a member *weaker* than our own handicapped
LightGBM qualifies if its errors are built differently. In-context learning — no split-finding,
no greedy loss descent — was the only candidate with a mechanism to get there.

**Result.**

```
cv_mean 1.77013 ± 0.02135   folds [1.77764, 1.79847, 1.78176, 1.75524, 1.73752]
member rho_B    0.65915     (-0.0042 vs e0266; the gate predicted a handicap 3x larger)
r vs champion   0.99433     <-- needed <= 0.9866
rho_partial    -0.00545     = -13.4 % of the bar
LOFO blend      +0.000000   0/5 folds, out-of-fold weights all 0.00
```

**Verdict `kill`.** But the informative part is not the null:

| family | r vs blend | 1−r² (unexplained share) |
|---|---|---|
| every GBDT (LGB / XGB / CatBoost / AutoGluon) | ~0.998 | 0.0040 |
| **TabICLv2 (in-context learning)** | **0.99433** | **0.0113 — 2.7× larger** |
| seq family (GRU, different architecture) | 0.9968 | 0.0064 |

**The hypothesis was right in mechanism and wrong in magnitude.** A genuinely different function
class *does* break the GBDT correlation floor — 0.99433 against 0.998 is a real move, and it
even beats the seq family's decorrelation. It reaches roughly a third of the distance to the
0.9866 the bar requires, on the `√(1−r²)` scale.

**What it closes.** The tabular axis is now shut on *evidence* rather than inference. §9d
argued from six GBDT variants that nothing tabular decorrelates below 0.998; the obvious
objection was that all six shared one function class. That objection has now been tested with
the strongest available alternative, at near-GBDT quality, and the answer is that the floor
moves but not far enough. Together with §9f (the best construction from all 15 owned artefacts
delivers +0.000003 LOFO) there is no remaining tabular route to +0.00070 blend ρ.

**Reusable:** the gate answers "what does the constraint cost?"; it cannot answer "are the
errors different?". Those need separate experiments, and only the second one decides a blend
member. Quality gates are necessary and never sufficient.

### 9f. CLOSED FORM: the best member constructible from everything we own = 50.9 % of the bar

The final question worth asking is not "which member should we build" but **"what is the best
member that could possibly be built from our artefacts"** — and that has a closed-form answer,
no search required. Residualise the truth `L` and every predictor on the blend `M`, then
regress: the multiple correlation of the residualised predictors against the residualised truth
**is** the maximum attainable ρ_partial.

Over all 15 frozen-fold predictors (three GBDT libraries, the tuned optima, the seed-averaged
arms, the buyers-only magnitude arms, BTYD, the seq variants):

```
max achievable rho_partial = 0.02073        bar for LB 1.6450 = 0.04068     -> 50.9 %
dominant: e0266 +0.39, e0260 -0.28, e0180 +0.17, e0092_cat +0.16, e0256 +0.16
implied best-possible blend rho 0.66283  ->  implied best-possible LB 1.646166
```

**Fitted in-sample, over 15 predictors, with no non-negativity constraint — every choice made
in the optimistic direction — and it reaches half the requirement.** The honest LOFO value is
lower still.

Two things this settles that the per-member table could not:

* **It is not a selection problem.** No subset, weighting, or mixture of what we own reaches
  1.6450; the bound covers all of them simultaneously.
* **The mixture intuition was worth testing and is worth retiring.** A blend of a strong
  correlated member with a weak decorrelated one *does* beat both endpoints — `0.90·e0266 +
  0.10·e0250` peaks at ρ_partial 0.01169 (28.7 %) against 26.4 % and 15.5 % for the endpoints —
  but the peak is far below the bar, and the closed-form optimum over all 15 only reaches
  50.9 %.

**1.6450 requires an artefact that does not exist yet, and cannot be assembled from the ones
that do.** Per §9d's table the missing property is *disagreement*, not quality: a member at
r = 0.95 could be **0.023 ρ weaker** than e0266 and still clear the bar, while at the tabular
floor of r = 0.998 it must be 0.0015 *stronger* than anything built in ~110 experiments.

#### ⚠ CORRECTION: "50.9 % of the bar" is an in-sample artefact, not a buildable target

The 0.02073 above and its implied LB of 1.646166 were quoted as an upper bound, which is
correct, but the phrase "best possible from all owned artefacts" implied something buildable.
**It is not.** Tested honestly — construct the member on four folds, apply it to the fifth,
choose its blend weight on the four as well:

```
in-sample promise      +0.00018 rho (50.9 % of bar)
LOFO-honest delivery   +0.000003     wins 2/5 folds
per-fold: [-5.3e-05, +2.6e-05, -1.4e-05, +6.7e-05, -1.2e-05]
```

**A 60× collapse, three folds negative.** The weights show why:

| member | mean w | min | max | sign flips |
|---|---|---|---|---|
| e0093_lgb3 | −0.019 | −0.090 | +0.014 | **yes** |
| e0064 | −0.053 | −0.117 | +0.014 | **yes** |
| e0250 | +0.015 | −0.022 | +0.063 | **yes** |
| e0260 | −0.186 | −0.219 | −0.140 | |

Three of fifteen members switch sign across folds, and the fit leans on large negative
coefficients (e0260 −0.19, e0210 −0.10) applied to predictors correlating 0.998 with each
other. **Differencing near-identical predictors extracts fold-specific noise, not signal** —
the exact failure §1m recorded when unconstrained OLS over 21 predictors *lost* (+0.00004
LOFO).

**The lesson, and it generalises past this project:** a residual-regression bound measures the
fit's capacity to memorise, and at r ≈ 0.998 that capacity is almost entirely noise. Such a
bound is only meaningful alongside its out-of-sample counterpart. Quote it as "in-sample
ceiling, likely unattainable", never as an achievable target.

Also worth recording: **9 of the 15 predictors have no test-side file**, so the construction
was not even directly buildable — the two largest coefficients were e0266 (+0.39) and e0260
(−0.28), and e0260 has never been predicted at the test anchor. Restricting to the 6 that do
have test files drops the in-sample bound to 34.3 % of the bar before any honesty correction.

### 9d. What 1.6450 requires of a tabular member, stated as a falsifiable bound (2026-08-23)

The correlation matrix over every tabular member the project owns, after varying everything
that can be varied:

```
           e0049    e0210    e0260    e0266    e0191    e0020
e0049    1.00000  0.99896  0.99949  0.99915  0.99916  0.99819
e0210    0.99896  1.00000  0.99928  0.99893  0.99891  0.99808
e0266    0.99915  0.99893  0.99948  1.00000  0.99911  0.99845
e0020    0.99819  0.99808  0.99849  0.99845  0.99813  1.00000
```

`e0049` and `e0266` differ by **6.5× in learning rate, 2.4× in leaves, 8× in min_data_in_leaf,
2.8× in feature_fraction, 2× in max_bin, 6.5× in rounds, plus 3-seed averaging** — and still
correlate **0.99915**. `e0210` is a different library. `e0020` uses 185 features against 665,
the largest information gap available, and reaches only 0.99819. **Nothing tabular has ever
gone below r ≈ 0.998**; the seq family, a genuinely different architecture, sits at 0.9968.

Taking r = 0.998 as the empirical floor and inverting the admissibility identity:

| r vs blend | required ρ_B | i.e. a member scoring |
|---|---|---|
| 0.9980 | 0.66314 | 1.7349 |
| 0.9990 | 0.66324 | 1.7347 |
| 0.9995 | 0.66317 | 1.7348 |

Against the best tabular member ever built (e0266, ρ_B **0.66166**), the target needs
**+0.00148 member ρ — 3.9× e0090's +0.00038, the largest member-level gain in the project's
history.** The requirement is nearly flat in r, because at this correlation the `√(1−r²)`
term is doing almost nothing: **decorrelation is not available on this axis at any price, so
the entire burden falls on raw quality.**

**This is the goal's answer in falsifiable form.** 1.6450 via the tabular member requires a
GBDT scoring ρ_B ≥ 0.66314 on the frozen folds. Ten families, four feature regimes, tuning,
seed averaging and a second library have produced a best of 0.66166 across ~110 experiments.
The claim is refutable by exhibiting such a model — and nothing in the measured record suggests
where it would come from.

### 9b. PREDICTION CONFIRMED (ON A PROXY — see §9c, which corrects the sign)

§9's prediction was registered before any measurement. It can be tested *without* e0260's OOF,
because the project already owns four seed replicates of one config — `e0101` and `e0101s1/2/3`
— which is the identical operation ("average N seeds of one model") on the family where §4
note 3 says it was already done.

```
e0101      member rho 0.66162        blend containing ONE seed        rho 0.66249
e0101s1               0.66151        blend containing the 4-SEED AVG  rho 0.66249   d +0.00001
e0101s2               0.66138        blend listing all 4 AS MEMBERS   rho 0.66254   d +0.00005
e0101s3               0.66151
4-seed average        0.66211   (+0.00061 over the mean single seed)
```

**+0.00061 member ρ → +0.00001 blend ρ. A 60× dilution.** And averaging the seeds *inside* one
member is equivalent to listing them *as* members (+0.00005 apart): **the blend was already
doing the averaging.**

The admissibility view shows the mechanism, and matches the registered wording exactly:

| | member ρ | r vs blend | ρ_partial |
|---|---|---|---|
| e0101, single seed | 0.66162 | 0.99863 | −0.00048 |
| e0101, 4-seed average | 0.66211 (**+0.00050**) | 0.99947 (**+0.00084**) | **−0.00330** |

Seed averaging raises the member's quality **and** its correlation with the blend, and the
second effect dominates: ρ_partial goes from −0.00048 to −0.00330, i.e. **more negative**,
precisely as §9 predicted. Removing a member's idiosyncratic noise makes it a *closer copy of
the consensus* — better alone, worth less in company.

**Consequence for e0260 (confirmed keep, −0.00034 on 5/5 folds) and e0266 (−0.00094):** both
are genuine improvements to a **standalone tabular submission** and neither should be expected
to improve the champion blend. §1c's rule again: quality without difference buys nothing.

### 9. ⚠ Pre-registered prediction: the seed-averaged arms will NOT close the remaining 64 %

e0260 / e0266 (§1r) are seed-averaged tabular members and are the only new tabular artefacts
still in flight, so it is worth stating *before* their confirms land what they can do for the
blend. Modelling seed averaging as shrinkage of e0049 toward its own expectation:

| shrink toward consensus | member ρ | ρ_partial vs champion |
|---|---|---|
| 0.0 (raw e0049) | 0.66125 | −0.00827 |
| 0.2 | 0.66136 | −0.00855 |
| 0.4 | 0.66145 | −0.00879 |
| 0.6 | 0.66152 | −0.00899 |

**Member ρ rises monotonically while ρ_partial gets monotonically *more negative*.** The
component seed averaging removes is precisely the idiosyncratic noise the blend was already
averaging away, so removing it makes the member a *closer* copy of the consensus — better
alone, worth less than nothing in company. This is the sharpest available statement of §1c's
corrected rule, and it is falsifiable: if the e0260/e0266 confirms produce a member that
improves the blend, this prediction is wrong and §1c needs revisiting.

The remaining room is real but it is **not on this axis**. §I13's magnitude term and §1b's
classification term are both scored end-to-end inside the *same* estimand and the same loss;
what §1s adds is the measurement that no rearrangement of the tabular members reaches the
target, so a route to 1.6450 must come from a member that is decorrelated **for a structural
reason** — a different estimand or a different supervision signal — not from a better GBDT.

---

---

> ⚠ **SECTION-NUMBER COLLISIONS (2026-08-24).** This file is being edited by two sessions at
> once, and the single-letter `§1x` space is exhausted: `1r`, `1s`, `1v` and `1w` each appear
> twice. I moved my two sections into a `§1z-*` namespace that cannot collide rather than keep
> chasing free letters; the other session's duplicates are left untouched because they are not
> mine to renumber. **Read this file by SECTION TITLE, not by number, until it is reconciled.**
>
> Mine: **`§1z-A`** (twelve routes to +0.00070 rho) and **`§1z-B`** (causal capacity optimum +
> the submission transfer rule).


## 1z-A. Twelve routes to +0.00070 rho, all measured and closed (2026-08-22)

**The bar, derived exactly.** `RMSLE = sd_L*sqrt(1-rho^2)`, `sd_L = 2.3178` (1i). Champion
e0162 rho 0.70378 -> 1.646602. Target **1.6450 needs rho 0.704479 = +0.00070**, and by 1f's
identity `rho_partial^2 = (0.704479^2 - 0.70378^2)/(1 - 0.70378^2) = 0.0019437`: anything new
must explain **0.194% of the blend's residual variance** (partial r >= 0.0441).

### Pass-through, measured on the test-anchor covariances (this governs everything)

```
improve e0141 ONLY (+0.00070)  ->  +0.00036 blend   0.52x
improve seq   ONLY             ->  +0.00037         0.53x
improve gbdt  ONLY             ->  +0.00018         0.26x
improve ALL THREE              ->  +0.00070         1.00x
    ...and if the common signal raises inter-member r by +0.001, 1.00x -> 0.69x
to reach target: all three at +0.00070 each, or e0141 alone at +0.00120
```

### The twelve

| # | route | result vs bar |
|---|---|---|
| 1 | monotone (isotonic/binned) recalibration | -0.000064 vs a **+0.000000** no-op control |
| 2 | GBDT stack on blend + 123 features | -0.00005 vs stack-on-M control |
| 3 | seed/SGD variance elimination | **+0.00018** = 25% of need |
| 4 | calendar-aware forward CV + `extra` decomposition | non-calendar half nil |
| 5 | `extra_nodoy` (period-safe calendar only) | +0.00006 within-anchor |
| 6 | population regime index (`pop_gmv_30`) | -0.00009 within-anchor; **+0.00247 (1/5)** on frozen folds |
| 7 | distributional loss at matched tree count | +0.00004 vs properly-budgeted L2 |
| 8 | per-user random effect from OOF residuals | 0.01x bar |
| 9 | raw daily panel (360 un-aggregated columns) | 0.26x bar, dRho negative |
| 10 | local/nonparametric member (cell means, kNN) | **negative** rho_partial |
| 11 | clip-aware calibration | the clip never binds (0.00%) |
| 12 | per-user equal loss weighting | effective sample already 94.1% |

### Three results worth keeping regardless of the target

1. **`E[L|M]` is already linear.** `corr(L,g(M)) >= corr(L,M)` with equality iff linear, so
   measuring it at zero PROVES linearity. 1b's "only rho is irreducible" therefore holds for
   **any monotone map**, not just affine -- which also kills quantile-matching the submission
   onto the probe-solved truth distribution.
2. **No persistent per-user bias exists.** Residual autocorrelation across anchors (same user,
   disjoint 30d windows) is 0.00312 and does NOT grow with averaging. Since any feature built
   from a user's own history is persistent, a residual with no autocorrelation cannot hide one.
   With #9 this closes the information axis for user-own-history.
3. **The admissibility frontier, solved against the champion:** a new member needs
   `rho_B >= 0.70378*r + 0.03133*sqrt(1-r^2)` -- r=0.998 -> 0.70435, r=0.97 -> 0.69029,
   r=0.90 -> 0.64706. A far WEAKER model suffices if decorrelated enough; everything ever built
   here sits at r >= 0.94, and kNN reaches r=0.926 but only rho 0.628 where 0.666 is needed.

### The methodological lesson of the session: level hides inside two statistics

The same error class produced a false positive twice, and both were caught by asking for a
different statistic rather than by re-running anything.

* **Raw RMSLE hides level.** `extra` beat `full` by -0.00326 raw at the forward anchor; only
  -0.00107 survived perfect calibration and its rho gain was +0.00049. **67% was level.**
* **Pooled rho across anchors ALSO hides level.** e0234 read **+0.00079 pooled** over 91
  forward anchors -- above the bar -- and **-0.00009 within-anchor**. Pooling credits a model
  for knowing December outranks July; the competition scores ONE anchor. `run_usercv.py` now
  reports `rho_within()` (each day centred on its own mean), verified on synthetic data where
  pooled reads 0.9424 against within 0.5118.

> **RULE. On any multi-anchor protocol, report WITHIN-anchor rho. Raw RMSLE and pooled rho
> both pay a model for the calendar level, and 1b proves level is free at submission time.**

### B7, and a structural limit that cannot be fixed

`--t-cut` adds a calendar split to the user-split CV (train + in-calendar val on anchors
<= cut, 30-day embargo, forward val on anchors >= cut+30). It works, and it measures the
calendar block at **+0.00049 rho** forward-within-year.

**But the axis that killed e0142 is provably untestable on this panel.** Training anchors start
at index 14 = **doy 15** (burn-in eats doy 1-14 of 2025) and the last usable anchor anywhere is
2026-01-14 = **doy 14**. The two day-of-year sets are disjoint by construction, so no cut can
ever put a forward anchor on a previously-trained doy. Only the leaderboard can see that failure
mode. **Consequence: never ship an absolute-calendar feature -- the prohibition has to be
policy, because no internal instrument can validate it.**

**Did B7 cost leaderboard score? No.** e0142 was pre-registered with the failure predicted
before submission, and the champion never used it. The abandoned `extra` bundle -- the plausible
hidden prize -- measures **+0.00001 rho / -0.00001 RMSLE** against `full` on the five frozen
anchors, and `extra_nocal`/`extra_nodoy` confirm the non-calendar half is nil.

---

## 1u. The magnitude term — closable, and not recombinable (2026-08-22)

`IDEAS.md` §I13 measured that the magnitude term captures only **80.2%** of its ceiling against
**89.6%** for the buy flag, and was never attacked with a ceiling-scored experiment. Exps
**e0243** (buyers-only), **e0244** (all-users control), **e0245** (recombination). Both
predictions were written into `scratch_thoughts.md` *before* any code ran.

### P1 — confirmed, and it is the largest measured effect in the project

```
                          corr(L,·|Z=1)   per fold                                    % ceiling
e0243 buyers-only   s0/s2     0.5388      [0.5234 0.5219 0.5409 0.5502 0.5542]          89.8%
e0244 control    s0/s1/s2     0.4813      [0.4703 0.4658 0.4806 0.4931 0.4984]          80.2%
                             +0.0575      5/5 folds, min fold +0.0533
```

**The magnitude term goes 80.2% → 89.8% of ceiling — level with the buy flag's 89.6%.** The
kill threshold was 0.002; this exceeds it **28×**. Seed spread is **0.0001** across three
control seeds, so the effect is **574× the reseeded-bag control** `IDEAS.md` §E1 requires.

**The control validates the harness:** `e0244s0` reproduces e0049 *exactly* — cv 1.76551, every
per-fold score identical, `corr(L,·|Z=1)` 0.4814. So the +0.0575 is attributable to the one
change and not to cross-session drift (e0195: +0.00046 for a config with zero changes).

**Leak checklist §3.3, all pass** — a jump this size demands it:

| check | result |
|---|---|
| rescaling, not ranking? | **no** — Spearman also rises, 0.5222 vs 0.4853 |
| target leak? | **no** — predicts **high** on zero-target users (3.763 vs 1.451); it never saw a zero, so it cannot identify buyers. A leak shows the opposite |
| population drift? | **no** — 444,224 rows, identical |
| one lucky subgroup? | **no** — wins in **all five** quintiles of L |
| genuinely new variance? | **yes** — corr(e0243 pred, control residual among buyers) = **+0.123** |

Mechanism, exactly as pre-registered: the all-users model spends capacity encoding `P(buy)` —
its prediction sd *among buyers* is **1.452** against the buyers-only model's **0.727**.

### P2 — refuted. The better ranker makes the product worse.

```
baseline (control, end-to-end)                          1.76515
E[L] = p_hat(M_base) · M_magnitude  (LOFO, 200 bins)    1.76977    +0.00462 WORSE
same recombination using the CONTROL's magnitude        1.76589    +0.00073
```

The second line is the control that isolates it: **the hurdle *form* costs +0.0007…+0.0018 on
its own, and swapping in the better magnitude half costs a further +0.0028…+0.0039.** A model
574× the seed spread better at ranking buyers converts to *nothing*, then to *worse*.

**Why**, and this is the part worth keeping: RMSLE elicits `E[L|x]`, and the hurdle identity
`E[L|x] = P(buy|x)·E[L|x,buy]` multiplies two separately-estimated halves, so their errors
compound. Worse, the magnitude half is trained on a population (buyers) that is **not the
scoring population**, and its behaviour on the 43% zeros is undefined by construction — it
predicts 3.763 there.

This independently reconfirms **e0010** (−0.00012) and **e0237**'s hurdle screen (+0.00534,
0/9) at a *third* baseline — and now it is explained rather than merely observed: **the
decomposition is not estimation-friendly even when one half is genuinely better.**

### What this changes

I13 was right that the magnitude term is the weakest and right that scoring end-to-end hid it.
It is **real and closable** — 89.8% of ceiling, on demand. What is now measured is that the
gap is **not recombinable by the hurdle form**, which is a *different and stronger* closure
than "the term is saturated": we know the information exists and that `p·mag` cannot spend it.

### The feature route — also dead, and it explains the whole thing (e0246)

`p·mag` failing left one cheap route: give the magnitude prediction to the model as a **feature**.
Two independent tests, both null:

```
RESIDUAL SCREEN (real 5-fold OOF, no training)
  corr(mag_pred, baseline residual), full population   +0.0047     incremental R2  0.00003
  LOFO refit of L on [1, base, mag]  (linear upper bound)   1.76538 vs 1.76515   +0.00023 WORSE
  non-parametric 20x20 map of (base, mag)                   1.77080              +0.00564 WORSE

LIGHTGBM STACK (pipeline params + early stopping, 3 anchors x 3 seeds)
  +mag feature    -0.00002   better in 4/9
  +NOISE column   +0.00021   better in 3/9      <- indistinguishable
```

**Why the +0.0575 does not transfer — the durable part:**

- `corr(mag, base) = +0.8747` overall, **+0.8957 among buyers**. The magnitude model is ~88%
  redundant with the baseline it would be stacked onto.
- Among buyers `corr(mag, baseline residual) = −0.4152` — the **wrong sign** for a stack. The
  baseline *over*-predicts exactly where the magnitude model says "high", because among buyers
  `base ≈ P(buy)·amount`, so a high base can mean *"certain to buy"* rather than *"will spend a
  lot"*.
- So the **5.87% incremental R² the magnitude prediction shows among buyers is real but is
  entirely the buy-probability confound being removed** — it is not new information about the
  amount. That is the same quantity `corr(L,·|Z=1)` rewards, which is why e0243 wins so large
  on that statistic and nothing anywhere else.

### The weighted-loss route — also dead, and now the mechanism is general (e0247)

The last form that avoids both previous failure modes: one estimator, one population, no
decomposition — just up-weight buyer rows so the L2 gradient spends more budget on magnitude.
Pre-registered at **80/20 against**, with the failure mode named in advance.

```
   w      RMSLE Δ    better    corr(L,·|Z=1)
 1.0     +0.00000     0/9         0.4623
 1.5     +0.03053     0/9         0.4616
 2.0     +0.07255     0/9         0.4623
 3.0     +0.15796     0/9         0.4649
 5.0     +0.29161     0/9         0.4708
```

**`w* = 1`.** Monotone increasing from baseline, no interior optimum, 0/9 at every weight.

**Why, measured rather than argued:** L2-with-weights elicits a *weighted* conditional mean, so
re-weighting does not reallocate capacity — it **changes the estimand**. Predictions on
zero-target users inflate `1.4059 → 1.8752 → 2.5002` for w = 1/2/5, and overall bias runs
`+0.1402 → +0.5300 → +0.9909`. RMSLE pays for that immediately. The hoped-for capacity effect
is not visible at *any* weight. This is the direct consequence of §1q: the model is already
within 0.0010 of `E[L|M]`, so **any** re-weighting moves it away from the right estimand.

> **All three routes out of I13 are closed** — multiplicand (e0245), feature (e0246), weighted
> loss (e0247). The magnitude gap is real, closable on demand, and **not spendable**.

> ⚠ **The dissociation, now seen three times.** At w=5 `corr(L,·|Z=1)` **rises** +0.0085 while
> RMSLE **degrades** +0.29161 — the two move in opposite directions. Same as e0243 (+0.0575 on
> the magnitude statistic, nothing end-to-end) and e0245 (a better magnitude half, a worse
> product). **`corr(L,·|Z=1)` is a diagnostic, not an objective**, and I13's "direction, not a
> size" caveat should be read as the stronger claim: optimising this statistic is not merely
> unreliable, it is at times *anti*-correlated with the metric we are scored on.

> ⚠ **And it reframes `corr(L,·|Z=1)` as a target.** e0243 moves it 80.2% → 89.8% while moving
> end-to-end RMSLE by nothing (as a feature) or backwards (as a multiplicand). **A metric that
> can be moved 28× its own threshold without moving the competition metric is a diagnostic, not
> an objective.** I13's own caveat — "treat it as a direction, not a size" — was right, and the
> honest update is stronger: the direction exists and both routes out of it are closed.

> **Method note.** Both outcomes were pre-registered with stated odds (P1 60/40 for, P2 75/25
> against) and a numeric kill condition. Both came in as written. The pre-registration is what
> made a 28×-threshold result safe to believe within minutes instead of arguable for days —
> and it is why the "+0.056 vs e0049" first read (a **fold-subset artefact**: folds 3–4 score
> 0.4962, not the pooled 0.4814) never became the headline.

---

---

## 1v. "Is any feature left?" — answered with a bound, not a candidate list (2026-08-22)

The recurring question after §1q/§1u is whether something remains in the **raw data**, or in a
**complex combination** of existing features, that no hand-designed candidate happened to hit.
Exp **e0248** answers it three ways.

### 1. Raw data does hold structure the 665 features miss — and we already spend it

`e0101` is a GRU on **13 raw daily channels with zero engineered features**. Against e0049:

```
corr(seq_pred, gbdt residual)  +0.0070      incremental R2  0.00153   <- 8x the kill bar,
corr(seq_pred, gbdt pred)      +0.9946                                   the largest ever screened
```

That is a genuine positive — the raw panel carries something the feature set does not. But it
is **not unexploited**: LOFO-blending the two gives `1.76500 -> 1.76296`, **-0.00204**, which is
precisely the blend the champion already banks (e0120/e0150/e0162). The answer to "is there
signal in the raw data" is *yes, and it is already in the submission.*

### 2. Seven exotic combinations, screened against what is actually left

Screened against the **blend** residual (the only unexploited quantity), full 250k, all folds:
day-of-week spend entropy, spike-vs-habit, funnel slope, buy-lifespan density, channel
divergence, max dormancy, log intensity. **All <= 0.00012 incremental R2 within fold, every one
sign-flipping across folds.**

> ⚠ **My error, caught and corrected — the most useful part of this run.** The *pooled* version
> of that screen returned `corr +0.1522`, `incR2 0.02316` — **100x the kill bar and the largest
> "signal" in project history.** It was entirely a **between-fold offset**: pooling five folds
> without removing per-fold means lets a fold-level mean difference masquerade as user-level
> signal. The same feature scores **-0.0006** within fold 4. **Always screen within fold.**
> Added to the graveyard; `scripts/residual_screen.py` takes a `--fold` argument for this reason.

### 3. The bound: a GBDT cannot fit its own residual

Instead of guessing more combinations, train a **second GBDT on the first's residual**, same
features, pipeline params, early stopping:

```
second model best_iteration = 4          (it finds essentially nothing to fit)
corr(residual-model prediction, true test residual) = +0.0018
test RMSE on the residual  1.72126 -> 1.72151     +0.00025 WORSE than predicting zero
```

**A full gradient-boosted search over these inputs cannot fit the residual out of sample, so no
hand-crafted combination of the same inputs can either.** Boosting *is* an automated search over
combinations — that is what the trees do. This is a bound on the feature space rather than one
more null candidate, and it is the honest answer to "are you sure there is nothing left".

**What it does not cover, stated plainly:** it bounds combinations *of these inputs*. It cannot
bound a genuinely new information source, and §1s already fixes the ceiling that any such source
would run into (ICC 0.5557 => rho <= 0.7455; 44% of the target is month-to-month coin flip).

---

---

## 1w. The seq side, audited the same way — also bounded (2026-08-22)

§1v bounded the tabular side. Exp **e0249** is its mirror: what, if anything, explains the
**GRU's** residual? Four tests, all screened **within fold** (the §1v pooling lesson).

### 1. Cross-family — real, and already spent

```
corr(gbdt_pred, seq residual)  and incremental R2 over the seq prediction, per fold:
  f0 0.00107   f1 0.00088   f2 0.00053   f3 0.00064   f4 0.00073     consistent, all 5 folds
```

The GBDT explains part of the seq residual in **every** fold — the exact symmetric counterpart
of §1v's finding that the seq model explains part of the GBDT's. This is the `gbdt+seq`
disagreement, and blending it is worth **−0.00204**, which the champion already banks.

### 2. Within the seq family — ten models, nearly one model

Ten seq OOFs (GRU, TCN, transformer, CNN-GRU, tuned variants, three seed replicates) correlate
**0.9924–0.9979** pairwise. LOFO-stacking **all ten** beats the best single member (e0180,
1.76365) by **−0.00072**. There is no hidden diversity inside the family.

### 3. The bound — features cannot fit what the sequence model misses

A GBDT with 53 tabular features trained **directly on the seq residual**, scored out of fold:

```
corr(prediction, true seq residual)  +0.0067
applying the correction:  RMSLE 1.72887 -> 1.76872     +0.03985 WORSE
```

### 4. Seq-specific temporal quantities — two survive sign, neither survives pricing

Screened last-3d GMV, last-7d GMV, last-3d buy count, sequence length, recency, and activity at
the 30-day boundary. Only **`seqlen`** (−0.0081) and **`edge`** (−0.0053) hold their sign across
all five folds; everything else flips. Priced together: **incremental R² = 0.00009** — below the
0.0002 bar and **10× smaller than what the GBDT already supplies** (0.00088). And `seqlen` is
tenure, which the tabular model already carries as a feature.

> **Both paths are now bounded by the same argument.** The only genuine cross-model signal in
> this project is the gbdt↔seq disagreement, and the champion blend already spends it. §1s says
> why: with ICC 0.5557 the target is 44% month-to-month coin flip, so models built on the same
> history converge — 21 of them to within 0.0007 AUC (§1r), ten seq models to within 0.008
> correlation here.

---

---

## 1z-B. The causal path's capacity optimum, and the transfer rule settled by three submissions (2026-08-23/24)

**Champion moved 1.646589 -> 1.646456.** The gain came from the CAUSAL_EXP path, from the axis
§1j's invalidated search never reached: **capacity**.

### The width curve, and why nobody had seen it

`e0141` runs at `hidden=128`. Sweeping width on the user-split CV (within-anchor rho, matched
same-session controls) gives a clean unimodal curve:

```
d192 0.66462 < d128 0.66486 < d96 0.66497 < d64 0.66513 < d48 0.66525 > d32 0.66519
```

**+0.00039 at d48, 14x anything else ever measured on this path.** It survives three checks that
kill most results here:

1. **§1j's fixed-epoch control.** A 2x2 of {d128, d64} x {14, 22} epochs, scored ONCE at the final
   epoch with no min-of-N selection: width is worth +0.00023 at 14 epochs and **+0.00080 at 22**.
   The effect GROWS with budget, because d128 gets *worse* with more epochs (0.66467 -> 0.66424)
   while the small model keeps learning (0.66490 -> 0.66504). A capacity x epochs interaction, not
   an evaluation-count artefact.
2. **An independent protocol.** The frozen-fold `seq` path (date folds, not a user split) gives the
   same peak: d32 1.76460, **d48 1.76359**, d64 1.76380, d96 1.76436, d128 1.76458 -- -0.00099 vs
   e0101, 5 sigma, **5/5 folds**.
3. **The leaderboard.** Predicted 1.646454 from OOF, actual **1.646456**.

> **Why the axis was invisible:** §1j's tuning of this path scored trials on the MIN of a
> variable-length early-stopped curve, which §1j itself then proved invalid; §1k redid the search
> properly but on the *frozen-fold seq* path, not here. So the causal path had never had a valid
> capacity measurement. `--fixed-epochs` now exists and prices the protocol's own bias at
> **+0.00124 RMSLE / +0.00019 rho** -- larger than nearly every effect this path has reported.

### The transfer rule, settled by three submissions in one day

```
slot contents -> a SINGLE better member   e0301  realised 0.88x   GAINED -0.000133
slot contents -> an AVERAGE of members    e0303  SIGN INVERTED    +0.000027
weights FITTED on CV                      e0302  SIGN INVERTED    +0.000593
```

> **CV over-values every form of variance reduction on the usercv slot, and is reliable only for a
> single-member quality swap.** §9e reached the same conclusion from seed-averaging; e0303 extends
> it to ARCHITECTURE averaging, which is the same mechanism despite looking different.

**Two guards that did NOT save the losing arms, and this is the useful part:**

* **Leave-one-fold-out does not protect fitted weights.** e0302's LOFO said +0.000217 winning 5/5
  folds *including the most test-like*, and the measured d rho was **-0.000259**. LOFO guards
  against variance from overfitting the OOF; it is blind to CV<->LB **ranking** shift, which is the
  actual failure mode. This is strictly stronger than §1m's caution.
* **Solving the component rhos from measured LB scores diagnoses it exactly.** With the component
  correlations taken from the submitted files and three measured blend rhos, the solo rhos are
  gbdt 0.70194, seq 0.70344, e0141 0.70311, **d48 0.70269**. d48 is *worse* standalone than e0141 --
  it helps by DECORRELATION at lower quality -- so CV handing it weight 0.4818 over-weighted a
  member CV over-rated. (Caveat: 4 equations, 4 unknowns, so the fit is exact by construction and
  inherits the LB's ~0.0002 rho noise with no redundancy. Do not reweight on it.)

### Architectures: measured properly, and both open questions closed

`--model {gru,lstm,transformer}` on the causal path, all at matched widths:

```
GRU  d48   0.66525  <- best      LSTM d128   0.66471
LSTM d32   0.66489               xformer d48 0.66456
LSTM d48   0.66487               xformer d64 0.66436
GRU  d128  0.66486               xformer d128 0.66420
```

* **"Was the transformer worse because of too few epochs?" NO.** Raising the cap 60 -> 150 gives
  **byte-identical** results (e0330 vs e0313: same rho, same best-epoch vector). Early stopping
  fired naturally; nothing was ever truncated. It also proves andrena runs are bit-deterministic.
* **"Was it handicapped by width?" PARTLY, and it matters.** §1c tested it at d128 -- the width now
  known to be ~2.7x too wide -- and §1k gave `xformer_rope` **2 search trials against the GRU's 13**.
  Fixing width is worth +0.00036 (d128 -> d48), but GRU d48 is still 0.00069 ahead. **§1c's verdict
  survives a proper search: attention loses to recurrence on this data.**
* **LSTM is level with the OLD GRU and below the tuned one.** §1k's 0.0029 gap was a best-of-2 vs
  best-of-13 comparison; at matched width the true gap to GRU d48 is 0.00038.

**The blend value inverts the solo ranking**, which is why the losers were worth running:

```
member          rho_B     r vs blend   rho_partial
GRU d48        0.66308     0.99836      0.02345     <- 1.8x the project best-ever (e0064 0.01269)
xformer d64    0.66244     0.99730      0.01952
LSTM d48       0.66278     0.99805      0.01948
GRU d128 (in)  0.66276     0.99900      0.00797
```

The transformer and LSTM carry **2.4x the incumbent's blend value while being worse standalone**.
That is §1c's law working in our favour -- but e0303 shows it cannot be harvested by averaging.

### The correlation loss (IDEAS.md §I2): run, and refuted

§1b says only rho scores after calibration, yet MSE also pays for a level and a spread calibration
discards. Implemented as `--loss {corr,mix}` using §1r's WITHIN-anchor estimator.

```
mix (MSE + 1-rho)  d128  rho 0.66487 (+0.00001)   |  on top of d64: +0.00001
pure corr          d128  rho 0.66428 (-0.00058)
```

**Nil in both forms.** IDEAS.md downgraded §I2 as "sharing HL-Gauss's mechanism" -- it does not
(HL-Gauss changes the estimator of the conditional mean; this is affine-invariant) -- but the
conclusion holds. *Training directly on the scored quantity buys nothing, which is evidence the
model is signal-limited rather than mis-optimised.*

**The one genuinely new fact it produced:** pure-corr training reaches **r = 0.879 with the blend**,
the most decorrelated model this project has ever built (BTYD 0.9427, Ridge 0.9433, every neural
variant >= 0.997). It is short of the admissibility frontier by only 0.016 against BTYD's 0.047 --
**3x closer than anything on record.** It contributes -0.000001 today because decorrelation only
pays at comparable quality. The obvious fix, that its in-batch rho estimate over 256 users is too
noisy, is **refuted and monotone in the wrong direction**: d128 b256 0.66428 > b1024 0.66201;
d48 b1024 0.66273 > b4096 0.65831. (Confound named: batch size also changes steps per epoch.)
Batch >= 4096 at d128, or >= 16384 at d48, OOMs a 40 GB A100 with the 17.4 GB panel resident.

### Everything else measured on the causal path, all nil on within-anchor rho

```
dropout 0.2 -0.00001 | dropout 0.3 -0.00002 | weight-decay 1e-3 +0.00002
mixup +0.00001       | --pop-train  -0.00002 | layers 3 -0.00008
```

* **Mixup overturns §1d's `keep` on the statistic that matters.** d RMSLE -0.00016 (0.8 sigma),
  **d rho +0.00001**. The mechanism replicates exactly (mean best-epoch 13.5 -> 18.2, ~35% longer
  training) but the gain lands in the level/spread term §1b proves is free. §1d's -0.00065 was
  measured against a ONE-SEED baseline; §1d's own note said the honest comparison is the 3-seed one.
  **Do not add mixup to the submitted e0141.**
* **`--pop-train` fixes a real code inconsistency for zero gain.** `run_seq.py:160` trains on
  in-population days only; `run_usercv.py` trained on every masked day, having computed the same
  mask and used it for reporting only. Measured on the panel: **6.2% of the causal path's
  80,899,560 training user-days are DORMANT** while **100.0%** of the 250,000 test users are
  in-population by construction. The mismatch is real and the model does not care.
* **Explicit regularisation is closed.** Neither dropout nor weight decay helps; only having fewer
  parameters does.

---

## 1z-C. Measurements made this session that were never written up (2026-08-24)

Everything below was run and used to make decisions, but only existed in the session transcript.
Recorded here so none of it has to be re-derived.

### 1. Premise checks on the target — all clean, none previously verified in this file

```
gmv == gmv_search + gmv_cat        0 violating rows in 1,824,348   (exact)
to_ord == search_to_ord + cat_to_ord  0 violating rows              (exact)
gmv > 0  <=>  to_ord > 0            0 exceptions
fold i's y_true == fold i+1's gmv_sum_30   100.00% exact match, corr 1.000000
```

The last identity matters: **the previous window's realised outcome IS an installed feature at
the next anchor.** So the 0.535 adjacent-anchor target correlation (below) is information the
model already has, not headroom.

### 2. A self-corrected error worth keeping: "test-retest reliability" is NOT a ceiling

Adjacent frozen anchors give the same users **disjoint** 30-day windows:

```
gap  30d  corr 0.53538      gap  90d  0.47882
gap  60d  0.49718           gap 120d  0.46762      linear extrapolation to gap 0: 0.55016
```

I initially read `sqrt(0.535) = 0.732` as a lower bound on achievable rho — i.e. +0.028 of
headroom over the champion. **That is wrong**, and the identity in §1 above is why: window *i*'s
realised outcome is literally `gmv_sum_30` at anchor *i+1*, so `Cov(e_i, s_{i+1}) != 0` and the
correlation is contaminated by information the model already uses. Caught before it changed any
decision. *Any repeated-measures bound on this panel has to exclude the earlier window from the
later anchor's feature set, which is impossible here.*

### 3. The Bayes-floor probe by feature cells — a lower bound that never binds

Quantising the top-k features into cells and taking within-cell `Var(L)` bounds what any model
can extract from those features:

```
top-4 x 8 bins   R2_max >= 0.41638      top-7 x 4 bins  R2_max >= 0.40812
top-5 x 6 bins   R2_max >= 0.41314      blend's actual R2 = 0.45553
NOISE-CELL CONTROL (4 random features): R2_max >= +0.00048   (must be ~0)
```

**The blend already beats a nonparametric fit on the top features**, so the cells are too coarse
to bind. Recorded so nobody re-runs it expecting a tight ceiling.

### 4. Pass-through, verified against the measured test-anchor covariances

Perturbing one component's `Cov(L, M_i)` in §1i's solved system:

```
improve e0141 ONLY (+0.00070)  ->  +0.00036 blend   0.52x
improve seq   ONLY             ->  +0.00037         0.53x
improve gbdt  ONLY             ->  +0.00018         0.26x
improve ALL THREE              ->  +0.00070         1.00x
   ...and if the common signal raises inter-member r by +0.001, 1.00x -> 0.69x
```

**Correction to a figure quoted earlier in this project:** reaching 1.6450 via e0141 alone needs
**+0.00120** member rho, not the +0.00167 that comes from using the weight 0.42 as the
pass-through — the optimiser re-weights, giving 0.52x. *And the measured OOF pass-through for a
real member swap turned out to be 0.19, lower than either.*

### 5. The train/metric weighting mismatch — priced out without a run

The causal loss weights each user by its number of scored days; the metric weights every user
once. Measured on the panel:

```
scored days per user: median 360, mean 324, min 17, max 365
top 50% longest-history users carry 56.2% of the loss (vs 50% of the metric)
effective number of equally-weighted users: 235,220 of 250,000 = 94.1%
```

**94.1% effective, so per-user equal weighting cannot be worth much.** Closed without spending a
run.

### 6. The three-block population rule, measured at a training anchor

The test population satisfies "active in each of the last three 30-day blocks" by construction.
At anchor 2025-10-16, on the 15k screen subset:

```
group                 n      share   zero%     rho
3 blocks (TEST-like) 11,831  88.4%   39.0%   0.66724
2 blocks              1,155   8.6%   72.2%   0.47664
1 block                 405   3.0%   72.6%   0.46162
ALL                  13,391          42.8%   0.67421
```

**The test-like subpopulation is HARDER than the mixed one** (0.667 vs 0.674) — the mixed
population's higher rho partly comes from easy "sporadic user -> zero" separations that do not
exist at the test anchor.

### 7. Component reconstruction identities — the basis for every blend built this session

```
0.5*log(e0090) + 0.5*log(e0064)              vs e0146g      corr 0.999999
0.20*gbdt + 0.38*seq + 0.42*e0141            vs e0162       corr 0.999997
0.5*log(e0266) + 0.5*log(e0064) in the slot  vs e0300_cal   corr 0.999997
```

`src/blend_ext.py` assembles from component submission files and **self-tests against a known
answer** before touching an unknown one: with today's components it reproduces e0300_cal's
1.646589 to 5e-6 and e0301's 1.646456 to 1e-6. Any future blend should be built this way.

### 8. Component solo rhos, solved from measured LB scores

Three measured blend rhos + e0141's own, with the component correlations taken exactly from the
submitted files:

```
gbdt half 0.70194 | seq half 0.70344 | e0141 0.70311 | d48 0.70269
```

**d48 is WORSE standalone than e0141** — it helps by decorrelation at lower quality. That is the
diagnosis of e0302's failure, and it is why any CV-fitted weighting over-weights it.

⚠ **Do not reweight on these.** Four equations, four unknowns, so the fit is exact by
construction (the "back-check" reproducing all three blends to 0.00e+00 is NOT a validation), and
it inherits the LB's ~0.0002 rho noise with no redundancy. Solving the optimal weights from them
gives +0.000053 — inside the noise of its own inputs.

### 9. The correlation-loss axis, closed ANALYTICALLY

A mix-weight sweep interpolates between MSE and pure-correlation. Four measured points show the
shortfall to the admissibility frontier is **monotone in r**:

```
arm                r vs blend   rho_B    needed   shortfall
MSE only (d48)      0.99836    0.66308   0.66387   -0.00079
mix w=1.0 (d64)     0.99774    0.66297   0.66377   -0.00080
mix w=1.0 (d128)    0.99766    0.66236   0.66375   -0.00139
PURE corr (d128)    0.87947    0.58216   0.59814   -0.01598
```

The endpoints bracket the axis below the frontier and the gap **grows** as the model decorrelates,
so **no interior mix weight can clear it.** Closed without running the sweep.

### 10. Operational events

* **`train.parquet` was deleted from the project root mid-session** (180,148,911 bytes, present at
  12:57, gone by 14:30) and restored from the cluster: md5 `8c955f99e60a6b0407f08e938c32557a`
  matches, and it re-reads as 30,631,006 rows / 250,000 users / 2025-01-01..2026-02-13 / 18 cols.
* **Two stray SLURM jobs** were submitted by malformed command lines and cancelled after
  confirming `WorkDir` — one no-argument `usercv` job, one duplicate `seqd32`.
* **`e0230`/`e0231` were cancelled before running**, not failed: the year-boundary forward CV they
  were meant to measure is **provably impossible on this panel** (training anchors start at
  doy 15 because of burn-in; the last usable anchor anywhere is doy 14 of 2026; the two
  day-of-year sets are disjoint by construction). See §1z-A.

---

## 1z-D. The guard-zone submission, and the control that stopped it (2026-08-24)

**Status: built, held, not submitted.** `subs/e0361_seqext_cal.csv` exists and is assembled
correctly; it is not readable yet, for a reason found by a control that was nearly skipped.

### What was built

`e0361` in BACKLOG's "Open" section is the seq family retrained through the guard zone. The naive
version — dropping `subs/e0253_seqext.csv` into the seq slot — would have changed **two** things:
the guard-zone extension *and* collapsing a 7-member ensemble to one model. So all seven members
(tcn, gru seeds 0–3, xformer, gru-deep) were rebuilt in submit mode with
`--train-through 2026-01-14`, and averaged in log space to form an extended seq half:

```
extended half  mu 2.3534  sd 1.5988        e0120s  mu 2.5123  sd 1.5874
corr(seq_ext, e0120s) = 0.998336
assembled blend corr with e0301 = 0.9997545
```

### The control failed, and that is the finding

An **unextended** rebuild of the same `e0101` config was queued alongside, to verify the pipeline
reproduces a known file before being trusted on an unknown one. It did not:

```
corr(regenerated e0101, subs/e0101.csv) = 0.998287     max|d log| = 2.125
```

**Cause: the original was built on apini H200s; the rebuild ran on andrena A100s.**
`cudnn.deterministic` fixes kernel choice for a given device and does not make two GPU
architectures agree. So the rebuilt seq half carries hardware drift *as well as* the intervention.

Separating the two on a single member — same code, same hardware, only the flag differing:

```
regeneration / hardware   corr(ctl, orig) = 0.998287    1-corr 0.001713
GUARD-ZONE EXTENSION      corr(ext, ctl)  = 0.996073    1-corr 0.003927
                                                        ratio  2.29x
```

**2.29× is not enough separation to spend the slot.** Submitting this against `e0301` would have
conflated *"does guard-zone training help?"* with *"were the models rebuilt on a different GPU?"* —
and the entire value of this slot is that it is the one question CV cannot answer. A confounded
answer would have burned it.

> **RULE. Rebuild the CONTROL on the same partition as the treatment.** Never compare a model
> rebuilt on one partition against a submission file built on another. This is now also in
> `CLUSTER.md` §10, where it corrects an earlier claim of mine that "andrena is bit-deterministic"
> — true within a GPU architecture, misleading across one.

### The fix, and the prediction it was built to test

Six matched **unextended** rebuilds were run on the same partition. The expectation was specific
and falsifiable, and written down before the files existed: **hardware drift is random per member
and averages down over seven; the extension is systematic across all seven and does not.** So the
ratio should widen substantially at blend level. The test is `corr(matched-unextended blend,
e0301)`:

* ≳0.99995 → drift is negligible once averaged, and the extended blend can be read against `e0301`
  in **one** slot;
* lower → the honest design needs **two** slots (matched-unextended *and* extended), and that
  should be stated rather than papered over.

### ✅ RESULT (e0382) — the prediction held, and e0361 is unblocked

```
                              1-corr        vs the single-member figures above
seq half   hardware drift    0.000200       was 0.001713   ->  8.6x smaller
seq half   guard-zone extn   0.001727       was 0.003927   ->  2.3x smaller
                             ratio 8.64x    was 2.29x
```

**Drift fell 8.6× across the seven-member average while the intervention fell only 2.3×** — the
exact asymmetry predicted, and the reason is the one stated: averaging seven independent hardware
perturbations shrinks them like `1/√7`, while a change applied identically to all seven survives it.

At blend level, which is what a slot actually reads:

```
self-test  corr(rebuilt ORIGINAL blend, e0301)      0.9999979   <- pipeline reproduces a known file
DECISION   corr(matched-UNEXTENDED blend, e0301)    0.9999687   1-corr 3.13e-05   PASSES >=0.99995
           corr(EXTENDED blend, e0301)              0.9997545   1-corr 2.455e-04
           blend-level separation                   7.84x
```

**87 % of the extended blend's perturbation is the intervention, 13 % is hardware**, and the
hardware share is worth ~**5e-05 RMSLE** — an order of magnitude below the effect being measured
and below the 6th decimal the leaderboard reports. **One slot suffices.**

Note the self-test line is doing real work: it confirms the assembly path reproduces `e0301` from
components to 2e-6 *before* the same path is trusted on an unknown blend. Without it, a bug in the
averaging would be indistinguishable from the hardware finding.

### The pre-registered read for the submission

`subs/e0361_seqext_cal.csv` carries `mu 2.330766, sd 1.630546` against `e0301`'s
`2.330728 / 1.630602`. That mismatch is worth **1e-07 RMSLE at any rho** (the sd terms cancel to
second order, §1i), so the comparison is paired for practical purposes and no rebuild is needed.
`e0301`'s own rho, solved from its measured 1.646456, is **0.703846**.

```
rho 0.703246 -> 1.647833   guard-zone HURTS, roughly as 5.2's +0.00189 predicts
rho 0.703646 -> 1.646915
rho 0.703846 -> 1.646456   no change
rho 0.704046 -> 1.645997
rho 0.704306 -> 1.645399   top-1 (1.6452459)
rho 0.704479 -> 1.645002   the 1.6450 target
```

**This is the only open question on the board that a slot can still resolve**, because §5.2's
+0.00189 was measured by validating *at* 2026-01-14 — an anchor inside the guaranteed-activity
zone — and no clean anchor exists to re-test it (BACKLOG's "Open" section derives why). The project
already contradicts itself here: `e0141`, at weight 0.42 of the champion, trains through
2026-01-14 while the gbdt and seq halves stop at 2025-10-16.

### ✅ SUBMITTED — training the seq half through the guard zone LOSES

> ⚠ **Read the e0386 diagnostic below before concluding "guard-zone data is bad".** The loss is
> real and replicated in direction, but it is at least partly a **decorrelation** loss: extending
> moved the seq half selectively *toward* the guard-trained usercv slot. I first wrote this section
> as "the guard-zone exclusion is CORRECT"; that is stronger than the evidence supports.

```
e0301             1.646456   rho 0.703846
e0361_seqext_cal  1.646806   rho 0.703693      d rho -0.000153   d RMSLE +0.000350
```

**Training the seq half through the guard zone LOSES**, in the same direction as §5.2's
independent, contaminated-anchor CV measurement. This is the one question on the board that no
internal fold could ever answer, and it is now answered.

**The magnitude, and my pre-registered number was wrong.** I wrote "if guard-zone hurts as §5.2
suggests → 1.647708", applying §5.2's +0.00189 to the blend directly. That was **~3× too
pessimistic**, because I did not apply the pass-through I had already measured and written down two
sections above (§1z-C item 4: the realised OOF pass-through for a real member swap is **0.19**):

```
0.00189  (5.2's full-model penalty)  x  0.19  (measured pass-through)  =  +0.000359
                                                          observed        +0.000350
```

3 % agreement — but state plainly that **this product is post-hoc**. What was pre-registered and
held is the *direction*; the magnitude is a retrospective reconciliation, and it is only worth
recording because both of its inputs were measured before the submission, not fitted to it.

> **Lesson, and it is a repeat.** §1z-A's pass-through table exists precisely so that a
> component-level effect is not quoted at blend level. I built that table and then failed to use
> it on my own prediction eight sections later. **Any component-level number must be multiplied by
> its slot's measured pass-through before it is compared to a leaderboard delta.**

**⚠ Statistical honesty.** +0.000350 is **0.92σ** on DATA.md §8.3's paired-50k sd of 0.00038
(1.40σ on `robustness.py`'s 0.00025). *On the leaderboard alone this is not significant.* What
makes it a `kill` is the **concordance of two independent instruments** — a CV measurement at a
contaminated anchor and a leaderboard measurement at the real one — agreeing in sign and, after the
correct pass-through, in magnitude. Neither alone would carry it.

**The control earned its keep.** The measured effect is **6.9× the 5.1e-05 hardware residue**
`e0382` quantified, so this read is not confounded by the apini→andrena rebuild. Had the control
been skipped, a +0.00035 delta against a hardware drift of unknown size would have been
uninterpretable — which is exactly the outcome the hold was protecting against.

### What it opens: the contradiction now points the other way

The project's internal inconsistency is unchanged in fact but reversed in implication. `e0295`
(the d48 GRU in the usercv slot, **weight 0.42 — the largest**) trains through **2026-01-14**;
`predict_usercv.py` sets `last_anchor = max_anchor(raw)` and its own docstring says so. The gbdt
and seq halves stop at 2025-10-16.

If the guard-zone penalty is a property of the *data* rather than of the seq architecture, then the
champion's largest slot is carrying it. **Retracting `e0295` to 2025-10-16 is the inverse of what
was just measured**, and it sits in the only category that has transferred this session — a
single-member contents swap with weights untouched (4-for-4, ~0.88×).

Two honest caveats before it is built:

* **It costs training data on a path where that matters.** §5.2's finding was that guard-zone
  anchors lose *despite* 35 % more rows. The causal path supervises per-day across the whole panel,
  so retracting three months is a larger relative cut there than on the anchor-grid paths.
* **It needs its own matched control**, and the usercv member is only a 3-seed average — drift
  shrinks like `1/√3 ≈ 0.58`, far less protection than the seq half's `1/√7`. Build the retracted
  version *and* a matched unretracted control on the same partition, and check
  `corr(matched control blend, e0301) ≥ 0.99995` locally before spending a slot, exactly as `e0382`
  did.

### ✅ BUILT (e0383–e0385) — and the diagnostic says do NOT submit it

**The control worry was misplaced, in the best way.** `e0295` was itself built on andrena
(job 24116151), so the control is a *same-architecture* rebuild, and it reproduces **exactly**:

```
corr(e0383_ctl48, e0295_usercv48) = 1.0000000     mu/sd identical (2.2425 / 1.5485)
blend-level residue                 0.0000021     <- float noise in the calibration
```

So CLUSTER.md §10's "bit-identical within a GPU architecture" holds for the full-data 3-seed
submit path too. **There is no drift confound here at all** — one slot, no averaging argument.

```
arm                   member r vs e0295   blend 1-corr vs e0301   vs drift floor
e0383 control            1.000000              0.0000021              1.0x
e0384 --guard-clean e32  0.996364              0.0006370            300.2x
e0385 --guard-clean e24  0.996814              0.0005578            262.9x
```

The perturbation is **2.59× e0361's** and 300× the floor, so it is easily readable. The epoch
bracket closes its objection: `e0385` sits at member-corr 0.999513 with `e0384`, so the epoch count
is not the binding variable.

### ⚠ But the retraction moves the member the WRONG WAY, and it reinterprets e0361

The usercv slot is paid for **decorrelation**, not accuracy — §1b is explicit that e0141's solo rho
is the *lower* of the two and "its value is decorrelation". Retracting makes it a **closer** copy of
the other two slots:

```
r vs (gbdt+seq)/2      e0295 incumbent  0.99559
                       e0384 gc32       0.99764     (+0.00205)
                       e0385 gc24       0.99807     (+0.00248)
```

**Why: the gbdt and seq halves are BOTH capped at 2025-10-16** (`predict.py:77`,
`run_seq.py`), while the usercv slot runs to 2026-01-14. Part of its decorrelation is simply that
it trains on a *different window*. Retracting aligns the windows and spends that, while 27 % less
data also lowers `rho_B`. Both terms point the same way.

**The discriminating test on e0361 (e0386), which costs nothing and changes its meaning.** Two
rival explanations of that +0.000350: (A) guard-zone data is intrinsically harmful, or (B) it merely
aligned the seq half's window with the guard-trained usercv slot. Extending the seq half moved it:

```
                              r vs usercv (guard-trained)   r vs gbdt (clean)
unextended (clean)                     0.99561                  0.99670
extended (guard-trained)               0.99819                  0.99595
change                                +0.00257                 -0.00075
```

A pure quality degradation lowers correlation with *everything*. This move is **selective** —
toward the one member sharing the new training window, away from the clean one. So **e0361's loss
is at least partly a decorrelation loss, not proof that guard-zone data is bad.** The two are not
exclusive: the −0.00075 against gbdt shows some genuine degradation as well.

> **The unifying statement, and it is the useful output.** The champion's three slots are
> **training-window-diverse** — gbdt and seq to 2025-10-16, usercv to 2026-01-14 — and that
> diversity is **load-bearing**. Any move that aligns their windows, **in either direction**,
> destroys blend value. The project's long-standing "contradiction" is not a bug to be fixed; it is
> a decorrelation source. e0361 aligned them from one side and lost; e0384 aligns them from the
> other and is predicted to lose.

**⚠ The caveat that keeps this a prediction rather than a result.** `r` is not the currency —
§1f/e0174 retired exactly that reasoning ("a MORE correlated candidate is worth MORE at fixed
excess"). A more-correlated member still pays if `rho_B` rises enough, and `rho_B` here is
unmeasurable without OOF or a slot. What makes the prediction worth acting on is that **both**
terms — higher `r` *and* 27 % less training data — move against it, and one independent
leaderboard observation (e0361) already went the predicted way.

**Recommendation: do not spend a slot on `e0384`.** It is built, logged and reproducible if the
call goes the other way.

### ✅ CV PRE-CHECK (e0387 vs e0388) — the "unanswerable" question, answered without a slot

**The project recorded this question as unanswerable by CV. That is true on the DATE-fold paths
and false here**, and the distinction is worth keeping: training through the guard zone and
validating *earlier* is incoherent when folds are dates, which is why §5.2 had to validate at
2026-01-14 and inherit the contamination. This path splits by **user**, so a model trained through
the guard zone can be scored at an **earlier clean anchor on users it never saw**. Users are
disjoint, the architecture is causal per-user, and within-anchor rho is invariant to the shared
calendar level — so training past the scoring anchor cannot leak into the statistic.

`run_usercv.py` gains `--train-cap`, which decouples the **training** mask from the **scoring**
mask. Those were tied together (`Mg` served both), which is precisely why this looked impossible;
`--t-cut` always scores *after* its cut and cannot express the comparison.

```
ARM A  --train-cap 2026-01-14   train 80,899,560 user-days (22,069,330 = 27.3% guard-contaminated)
ARM B  --train-cap 2025-10-16   train 58,830,230 user-days (all guard-clean)
BOTH   score held-out users on the SAME 58,830,230 clean user-days
```

**Two port gates passed without being engineered to.** Arm A's training mask is **80,899,560** —
the exact figure §1z-C item 5 quotes for this path from the unrelated `--pop-train` analysis, so
Arm A reproduces the champion's grid byte-for-byte. And **58,830,230** matches
`predict_usercv.py --guard-clean`'s independently-written count exactly.

```
                          rho_in      per fold (within-anchor rho, clean block)
A guard grid (incumbent) 0.659190   [0.662626 0.659705 0.660248 0.654924 0.658448]
B retracted (clean)      0.659062   [0.662495 0.659564 0.660089 0.654834 0.658329]
B - A                   -0.000128   [-0.000131 -0.000141 -0.000160 -0.000090 -0.000119]
                                     B wins 0/5 folds · 3.97 sigma
```

Noise floor measured from **this instrument's own** within-arm seed sd (0.000072 → sd of the
5-fold mean 0.000032), not borrowed.

**In the currency, not in `r`.** I argued against `e0384` from `r` alone — which is exactly the
reasoning e0174 retired — so here it is done properly, both arms against the same family on the
same 1,062,003 common keys:

```
arm                        rho_B     r vs fam        e    rho_partial   pred gain
A guard grid (incumbent) 0.66339    0.99724   +0.00258      0.04632      -0.00189
B retracted (clean)      0.66311    0.99757   +0.00207      0.03970      -0.00139
(ref) e0141 d128         0.66301    0.99675   +0.00252      0.04173      -0.00153
```

**Retracting costs 14.3 % of the member's blend value.** And my argument was only *half* right:
both terms move against B by similar amounts — quality `rho_B` −0.00029, and correlation
`r` +0.00033 (worth −0.00022 of `e`). The member is not merely more redundant; it is also simply
worse. Note also that **A beats e0141 d128 on `rho_partial` too** (0.04632 vs 0.04173), which is
the width finding showing up a fourth time.

**⚠ Two caveats, both pre-registered.**

1. **The confound stands.** Arm B trains on 27 % fewer user-days, so "guard-zone anchors help" is
   *not* separated from "more data helps". A win for B would have been clean; this loss is not.
   Resolving it needs a volume-matched control (A's grid, 27 % of anchors dropped at random).
2. **The absolute `rho_partial` values are inflated.** The family here is the 9-member seq+gbdt
   blend, which **excludes** the usercv slot — the §E1 trap. Only the paired A-vs-B difference is
   trustworthy; 0.046 is not a number to quote against §1f's bar.

### The unified reading of e0361 + e0387/e0388

The two results look contradictory — extending the seq half *into* the zone lost, and retracting
the usercv slot *out of* it also loses — but they are one statement:

> **Guard-zone anchors improve a member's own quality (they are more data, and more recent data),
> and they simultaneously make members that share the window more alike.** The champion sits at a
> configuration where the windows *differ*, so it collects both the quality on the slot that has
> them and the decorrelation from the slots that do not. Moving either component toward the
> other's window trades one for the other, and both trades measured negative.

**The "contradiction" is retired.** The usercv slot training to 2026-01-14 while the gbdt and seq
halves stop at 2025-10-16 is not an inconsistency to fix — it is load-bearing, and it is now
measured from both sides rather than asserted.

### Seq-half architecture members at d48 — the width finding replicates a third time

The seq half's three non-duplicate architecture members had never been width-tuned. Unlike the four
GRU seed duplicates they carry no variance-reduction role, so narrowing them cannot destroy seed
diversity (which is what made the earlier all-GRU→d48 swap *worse*, at −0.000024):

```
TCN       d48 1.76735  vs e0100 1.76775   -0.00040
xformer   d48 1.76496  vs e0102 1.76575   -0.00079
GRU-deep  d48 1.76390  vs e0108 1.76479   -0.00089
```

**All three improve.** The width result now holds for GRU, GRU-deep, TCN and transformer, across
both the user-split and frozen-fold protocols. **`d_model = 128` is the wrong default for every
sequence architecture in this project**, and that is the single most transferable thing this
session produced.

## 1z-E. The closure, stated as geometry rather than as a list of nulls (2026-08-25)

Every candidate this project has produced has a coordinate `(r, rho_B)` — its correlation with the
champion and its own accuracy. §1f gives the bar as a *curve* over `r`. Plotting all of them
against it, **pooled over the 5 frozen folds, against the FULL champion** (`rho_M = 0.66342`):

```
member                                  r    rho_B      bar       GAP
corr-loss pure (bs1024)           0.78441  0.52066  0.53637  -0.01571
behavioural 6ch (gmv-blind)       0.88103  0.58269  0.59668  -0.01399
MOMENT frozen (e0915)             0.95920  0.63550  0.64364  -0.00814
Chronos-Bolt (e0919)              0.97990  0.64990  0.65522  -0.00532
gmv-only 1ch (e0402)              0.98566  0.65337  0.65825  -0.00488
monetary 7ch (e0401)              0.98671  0.65420  0.65879  -0.00459
cigru w16 (e0403)                 0.99766  0.66149  0.66363  -0.00214
joint GRU e0101 (INCUMBENT)       0.99796  0.66172  0.66371  -0.00199
```

**The gap is negative everywhere and monotone increasing in `r`.** The achievable frontier lies
strictly below the admissibility bar at every correlation this project has ever reached, and
converges to it **only as `r → 1`** — where the candidate *is* the incumbent and adds nothing by
construction. **There is no interior sweet spot**, so there is no `r` worth aiming at.

That is the closure. Not "we tried many things and they failed" but: the two curves do not cross
anywhere in the reachable region, and the direction of approach is away from any usable point.

These eight members span correlation-loss training, gmv-blind channel restriction, three frozen
time-series foundation models, two channel subsets, a channel-independent recurrence, and the
incumbent — i.e. different losses, different information sets, different pretraining corpora,
different architectures, and different orderings of the same computation. They trace one curve.

> **The empirical exchange rate, which is the transferable number:** buying decorrelation costs
> accuracy faster than the bar falls. From the incumbent to the gmv-blind model, `r` drops 0.117
> and `rho_B` drops 0.079, while the bar only drops 0.067. Every route pays ~1.18 units of `rho_B`
> for every unit of bar relief, so every route loses.

### ⚠ CORRECTION to the table above — the sign-correct test, and it is stronger

The "GAP" column implicitly assumes a **positive** excess. That is wrong: `R² = rho_M² + e²/(1−r²)`
with `e = rho_B − r·rho_M`, and `e²` does not care about the sign — a **negative** excess is
exploited with a **negative weight** on that member. So the admissibility test is `|e|` against
`sqrt(dR² · (1−r²))`, not `rho_B` against a one-sided bar. Redone properly, for the true target of
+0.000633 rho:

```
member                              r          e       |e|  need@+633u   ratio
corr-loss pure                0.78441   +0.00027   0.00027     0.01798   0.01x
behavioural 6ch               0.88103   -0.00180   0.00180     0.01371   0.13x
MOMENT frozen e0915           0.95920   -0.00085   0.00085     0.00820   0.10x
Chronos-Bolt e0919            0.97990   -0.00019   0.00019     0.00578   0.03x
gmv-only 1ch                  0.98566   -0.00054   0.00054     0.00489   0.11x
monetary 7ch                  0.98671   -0.00040   0.00040     0.00471   0.09x
cigru w16                     0.99766   -0.00038   0.00038     0.00198   0.19x
joint GRU e0101 (in champ)    0.99796   -0.00035   0.00035     0.00185   0.19x
```

**Every member is inadmissible by a factor of 5× to 100×**, and the conclusion survives the
correction rather than depending on it. The sharpest single number:

```
largest |e| ever measured : behavioural 6ch, |e| = 0.00180 at r = 0.88103
best achievable blend gain: +0.000011 rho, at the OPTIMAL (signed) weight
                            = 1.7% of the +0.000633 the target needs
```

**Granting every candidate its optimal weight, including negative weights, the best single
addition available to this project is worth 1.7 % of the remaining gap.** That is the closure, and
it no longer rests on the one-sided framing.

### ⚠ I nearly published a false positive here, from an error I had corrected 30 minutes earlier

The first version of this table included the fold-4-only TS-FM data-scale arms (`e0394`–`e0399`)
alongside the pooled members, and they came out with **positive** gaps of +0.0038 to +0.0058 —
reading as "TTM and MOMENT clear the bar". They do not. Their `rho_B` is measured on **fold 4**,
where the champion sits at `rho_M = 0.67502`, not on the pooled folds where it sits at 0.66342.
Scoring a fold-4 `rho_B` against a pooled bar credits the candidate with the 0.012 the champion
loses by being averaged over harder anchors.

**This is exactly the scale error §1z-D/e0391 was written to correct**, committed again within the
hour, on the same axis, by the same person. It survives only because the table was recomputed
before being reported. *A fold-4 number and a pooled number may never appear in the same
comparison — put the population in the variable name if that is what it takes.*

### The 12th point — RealMLP, and the frontier redrawn entirely on the FOLD-4 scale (2026-08-25)

`IDEAS.md` §I22's TOP-1 bet finally returned a number. It had been recorded as "CLOSED — timed out
twice"; in fact a 24 h resubmit (job 24152700) **completed in 20 h 35 m** and the result sat unread.
Full write-up in §I22's RESULT block and `experiments.csv:e0913`.

⚠ **This table is entirely fold-4** (`rho_M = 0.675020`, aligned 223,578-user intersection, joined on
`user_id`) and must not be read against the pooled table above (`rho_M = 0.66342`). It is drawn here
*separately* rather than merged, precisely because of the error the previous subsection records. The
harness was validated first by reproducing §1z-E's own `rho_M` to six decimals from OOF components.

```
member                            r    rho_B      bar    MARGIN  |e|/need
cigru16 e0403               0.99793  0.67361  0.67550  -0.00189     0.01x
joint GRU e0101 (incumbent) 0.99840  0.67370  0.67559  -0.00189     0.14x
XGBoost e0210               0.99737  0.67298  0.67536  -0.00238     0.12x
RealMLP e0913 (NEW)         0.99536  0.67160  0.67470  -0.00310     0.10x
TabICLv2 e0340              0.99457  0.67103  0.67440  -0.00337     0.11x
gmv-only 1ch e0402          0.98703  0.66667  0.67096  -0.00429     0.09x
Chronos-Bolt e0919          0.98314  0.66423  0.66899  -0.00476     0.11x
MOMENT e0915                0.96323  0.65137  0.65805  -0.00668     0.15x
BTYD e0170                  0.94932  0.64120  0.65000  -0.00879     0.04x
corr-loss pure              0.79278  0.53654  0.55296  -0.01642     0.08x
behavioural 6ch e0400       0.88006  0.59007  0.60794  -0.01787     0.29x
```

**Same shape, independently on a second population scale: negative everywhere, monotone in `r`,
converging to the bar only as `r → 1`.** The pooled closure was not an artefact of pooling.

**What RealMLP adds that the other eleven could not.** Every prior member on this curve either shared
the GBDT's *function class* or was *weaker*. RealMLP is neither: a modern tabular neural net at
essentially GBDT strength (`rho_B` 0.6716 against the gbdt half's 0.673636 — the best non-tree,
non-recurrent member the project has built). §I22 predicted, from NN_TORCH's `r = 0.96`, that such a
model would land near `e ≈ +0.022`, far above the bar. It landed at **`r = 0.995356`, `e = −0.000284`**.

> **The generalisation, and it is stronger than §1c's original statement.** §1c said decorrelation
> must come *at comparable quality*. This says the two are not independent axes at all: **`r` measured
> on a weak model does not survive being made strong.** NN_TORCH's `r = 0.96` was a property of being
> wrong (rho 0.647), not of being a neural net; when the same architecture family is trained to
> GBDT strength on the same 665 features, it converges into the blend's subspace like everything else.
> §1s's "function-class diversity at fixed features does not produce decorrelation" now holds
> **across the tree/NN boundary**, which was the one cell it had never been tested in.

**Caveat kept in view:** 500 k training rows against ~5.2 M. §I17's gate prices the handicap at
~+0.002 rho against the +0.0031 needed, and more data normally raises `r` as well. A GPU port
(the run was CPU-only) would settle it in a few GPU-hours; falsifier: **`rho_B ≥ 0.6747` at
`r ≤ 0.9954`**. Prior ~10–15%.

---

## 2. The metric on CV (superseded by §1b, kept for the fold-level numbers)

> §1b solves the same algebra against the **true public-test moments** rather than the folds'.
> Where the two disagree, §1b wins: in particular "calibration cannot help" below was true
> ON THE FOLDS (`k*=1.000`) and **false at the test anchor**, where every model was mis-levelled
> by 0.14–0.41 in log space. That error cost us 0.006–0.009 on every submission up to e0120.

At the optimal prediction scale, `RMSLE = sd_L · sqrt(1 − ρ²)`, where ρ is the log-space
correlation between truth and prediction. Measured on e0049's OOF:

```
sd_L 2.3524   sd_M 1.5524   ρ 0.6611
optimal sd_M = sd_L·ρ = 1.5553   -> we are 0.18% off. Nothing to win by rescaling.
RMSLE^2 terms: (sdL-sdM)^2 0.6401 | 2·sL·sM·(1-ρ) 2.4750 | mean 0.0002 (0.01%)
```

**The entire score is one number, ρ.** To reach top-1: **ρ 0.6611 → 0.6647, i.e. +0.0036.**

Three doors are therefore shut, and each has been tested rather than assumed:

1. **Calibration / rescaling** — we sit 0.18% off the optimum. `k* = 1.000` independently.
   Applying the *real* measured +15.4% YoY growth costs **+0.00474**.
2. **Custom objectives** — L2 on `log1p` **is** RMSLE². The training loss already equals the
   metric, so no quantile/Tweedie/LambdaRank reformulation can help.
3. **Blending our current models** — `corr(log e0049, log e0064) = 0.9989`. AutoGluon-vs-
   LightGBM on the same features is not diversity. A blend buys ~nothing.

**Where the error lives (it is diffuse, there is no pocket to attack):**

| segment | share of rows | share of SSE | RMSE in segment |
|---|---|---|---|
| target = 0 | 44.2% | 42.4% | 1.7288 |
| target > 0 | 55.8% | 57.6% | 1.7932 |

Oracles: perfect on zeros → 1.339; perfect on non-zeros → 1.150. Both enormous, neither reachable.

---

## 3. CV → LB transfer: the rule

| step | ΔCV | σ | significant? | ΔLB | ratio |
|---|---|---|---|---|---|
| p30 → e0001 | −0.46833 | — | yes | −0.44340 | 0.95× |
| e0001 → e0020 | −0.01209 | 134σ | yes | −0.01880 | 1.56× |
| e0020 → e0049 | −0.00087 | 9.7σ | yes | −0.00160 | 1.84× |
| **e0049 → e0060** | **−0.00004** | **0.4σ** | **no** | **+0.00050** | **−12.5×** |
| e0060 → e0064 | −0.00018 (4-fold) | 2σ | marginal | −0.00080 | **4.4×** |
| **e0064 → e0120** | **−0.00239** | **12σ** | **yes** | **−0.00060** | **0.25×** |

* **Significant CV deltas transfer amplified (1.5–1.8×)** — *for single-model changes inside
  one family.* This no longer holds in general; see the e0120 row.
* **Sub-2σ deltas do not transfer at all** — e0060 flipped sign and cost 0.0005. `CLAUDE.md`
  §3.4 vindicated with a price tag; do not promote within-noise winners.
* ~~**A model-FAMILY change transferred at 4.4×**~~ — **over-read, and it cost us.** That
  ratio came from a −0.00018 CV delta, itself only 2σ. Reading a transfer *rate* off a delta
  that small was never sound, and propagating it into e0120's projection was the mistake.

### ⚠ e0120: the first significant delta that transferred at LESS than 1× (2026-08-13)

**Projected 1.6516–1.6523, scored 1.6553.** Still our best LB, but the CV gain arrived at a
quarter size. The diagnosis, all measured rather than argued:

* **Not sampling.** Bootstrapping a paired 50 000-user delta on our OOF gives sd **0.00038**
  (2σ ±0.00076). The shortfall is **4.6σ**. "Paired LB deltas are exact" is a good shorthand:
  they are precise to about ±0.0008.
* **Not cut-off distance** — and the answer is the *opposite* of the hypothesis.
  `src/seq_transfer.py` builds the seq analogue of `reports/eda/transfer_test.json`:
  degradation per 100 days of cut-off gap is **+0.00065 for `seq` against +0.00428 for the
  GBDT** (corr with gap +0.07 vs +0.71). **The sequence model is 6.6× more robust to cut-off
  distance than the GBDT.** Its calendar-invariance claim survives contact with measurement.
* **Not sequence-length extrapolation.** Cropping the input to the model's own trained length
  costs **+0.004 to +0.018** at every gap. The model genuinely uses the long history; feeding
  the full 409 days at the test anchor is correct.
* **Not family convergence or spread collapse at the test anchor.** `corr(log gbdt, log seq)`
  0.9968 (CV) → **0.9964** (test); `sd_seq/sd_gbdt` 1.013 → **1.011**. The decorrelation that
  the blend monetises is fully intact at the test cut-off.

**What is actually going on, and my error.** The per-fold deltas are monotone —
−0.00365 / −0.00288 / −0.00255 / −0.00144 / −0.00145 — and §3.2 of this file says to report
the last fold separately and flag it when it disagrees with the mean. **I projected from the
mean.** Decomposing the −0.00240: **−0.00042 is gbdt-side averaging** (folding e0049 into
e0064) and **−0.00198 is the seq family**; on fold 4 alone the seq part is already down to
−0.00119. The observed −0.00060 is almost exactly the gbdt-side part alone at 1.5×
(−0.00062), i.e. **the seq family's contribution at the test cut-off is consistent with zero**
— even though the family is demonstrably more cut-off-robust and demonstrably decorrelated.

Those two facts are in tension and **one submission resolves it**: `subs/e0120s.csv`, the pure
seq-family blend, is already built. It is the only measurement that separates *"blend gains
decay toward the test period"* from *"the seq family does not transfer"*, and the two answers
point at opposite next moves.

**Rule change, effective now: project transfer from the LAST FOLD, not the fold mean, and
never from a delta smaller than 2σ.**

CV−LB gap is stable: 0.1019 → 0.1086 → 0.1093 → 0.1088 (drifting up very slowly).
**Paired LB deltas are exact** (same 50k public users); only unpaired comparisons carry the
±0.0118 noise.

---

## 3b. The `seq` approach — a sequence model on the raw daily panel (NEW, 2026-08-13)

`src/seqdata.py` · `src/seqnet.py` · `src/run_seq.py` · `configs/e01*.yaml` · `scripts/seq.slurm`

**What it is.** The 250k × 409 panel as a dense `(user, channel, day)` tensor — 13 channels,
2.66 GB fp16, resident on one H200, so there is no dataloader and a batch is a slice. A causal
network emits a prediction at *every* day in one forward pass; the head estimates
`log1p(sum gmv over [t+1, t+30])` from days ≤ t only.

**Two things it has that the tabular pipeline structurally cannot — one of which turned out to
be worth nothing.**

1. ~~**Dense supervision.**~~ **Measured at ZERO (e0115).** Supervising every 7th day instead of
   every day — 5.2M user-days instead of 35.1M, exactly the GBDT's anchor grid — costs
   **+0.00012 (0.6σ, wins 2/5)**. The targets overlap 29 of 30 days, so the extra positions
   carry almost no independent information. This was the headline claim when the approach was
   proposed and it does not survive contact with a measurement. **Consequence: any design that
   emits one prediction per forward pass instead of 409 starts from no handicap at all.**
2. **Calendar translation invariance — this one is real.** No feature references absolute time,
   so none of the cut-off drift `anchor_drift.py` measured (test cut-off = 3.92× feature-space
   outlier). Confirmed by `src/seq_transfer.py`: **+0.00065 RMSLE per 100 days of cut-off gap
   against the GBDT's +0.00428**, 6.6× more robust. This is now the approach's only structural
   edge.

**The headline.** A 2-layer GRU on 13 raw channels — **zero engineered features** — beats the
665-feature LightGBM that fifty experiments produced:

| exp | model | CV | vs e0049 | folds won |
|---|---|---|---|---|
| e0101 | **GRU d128 ×2, 12 epochs** | **1.76458** | **−0.00093** | 4/5 + 1 tie |
| e0108 | GRU ×3 layers | 1.76479 | −0.00072 | — |
| e0102 | causal transformer, ALiBi | 1.76575 | +0.00024 | tie |
| e0100 | dilated TCN d128 ×8 (RF 511 d) | 1.76775 | +0.00224 | — |
| e0049 | LightGBM, 665 features | 1.76551 | — | — |

**But the accuracy is not the point — the decorrelation is.** §2 closed the blending door on
the grounds that `corr(log e0049, log e0064) = 0.9989`. Measured log-prediction correlations:

```
gbdt vs gbdt   e0049–e0064   0.9983      <- the door that was shut
seq  vs seq    e0101–e0108   0.9976
seq  vs gbdt   e0101–e0049   0.9946      <- a genuinely different function
```

That 0.0037 drop in correlation is the whole gain. Blend ladder, all leave-one-fold-out:

| blend | CV | gain vs best member |
|---|---|---|
| e0049 + e0064 (gbdt only) | 1.76481 | −0.00039 |
| seq family only (7 members) | 1.76342 | −0.00116 |
| e0049 + e0064 + e0101 | 1.76302 | −0.00156 |
| **all 9 members, equal weight** | **1.76274** | **−0.00176** |

**Family weighting is a free choice.** `src/blend_fixed.py` sweeps stated (not fitted) weights
between the two families; the curve is flat from 0.4 to 0.8:

```
w(seq)   0.00     0.40     0.50     0.60     0.70     0.80     1.00
cv      1.76478  1.76300  1.76280  1.76269  1.76268  1.76277  1.76323
```

**e0120 takes w(seq) = 0.50**, costing +0.00012 over the CV optimum — well inside the seq
noise floor — in exchange for keeping half the mass on the only family the leaderboard has
ever scored. `subs/e0120.csv` is built and verified; not yet submitted.

**σ_noise for `nn_seq` = 0.00020** (e0101 seeds 0–3: 1.76458 / 1.76479 / 1.76514 / 1.76477).
That is **2.2× the GBDT's 0.00009** — SGD order, dropout masks and init all move here and none
of them move a boosted tree. Use 0.0002, not 0.00009, when judging any seq delta.

**Correctness.** `assert_causal` perturbs every day after t and requires outputs at ≤ t not to
move, *and* requires the perturbation to change something downstream so the probe cannot pass
vacuously; it runs on the real model, on every fold, before the first gradient step. The
tensor itself is cross-checked against `src/data.py`'s independently-written prefix panel —
targets, population masks, channel window sums and `geo3` all match, the last one bit-for-bit
so `delta` stays comparable across approaches (`scripts/seq_selftest.slurm`).

**Cost.** e0101 is **3.7 minutes** for all 5 folds on one H200. The entire twelve-experiment
sweep above cost less wall-clock than one AutoGluon fold.

### 3b.1 Can the GRU use the tabular features, better features, or a CNN? — measured, no

Five experiments (σ_noise 0.00020, so 2σ = 0.00040). `derived_channels` = 27 per-day channels
holding the tabular set's top-ranked family (7/30/90/365-day sums of gmv/ord/cart/srch,
active-day and buy-day counts, both recencies, geo3 itself), each a prefix-sum difference so
all 409 days cost one subtraction and dense supervision is preserved. `rank_channels` = 5
cross-sectional ranks within the day's population.

| exp | change | CV | vs parent | folds won |
|---|---|---|---|---|
| e0110 | GRU **+27 derived channels** | 1.76595 | **+0.00137** vs e0101 | **0/5** |
| e0111 | **cnngru** (conv front-end → GRU) | 1.76620 | **+0.00162** vs e0101 | **0/5** |
| e0112 | cnngru + derived (combination) | 1.76579 | +0.00121 vs e0101 | 0/5 |
| **e0113** | **TCN +27 derived channels** | **1.76610** | **−0.00165** vs e0100 | **5/5** |
| e0114 | GRU **+5 cross-sectional rank channels** | 1.76563 | **+0.00105** vs e0101 | **0/5** |

**The control experiment is what makes this readable.** e0113 was run precisely to separate
"the channels carry information" from "the GRU was bad at long-window integration", and it
settles it:

> **The derived channels are not information. They are a substitute for long-range
> integration.** They help exactly the architectures that lack it — TCN **−0.00165 (5/5)**,
> cnngru **−0.00041** vs cnngru-alone — and hurt the one that already has it: the GRU, whose
> recurrence computes these windows for itself, loses **+0.00137 (0/5)** to the dilution.

That is the third independent confirmation of this repo's oldest lesson — `sbcmoment` (dropping
the highest-gain feature family cost −0.00008), `funnel` (10 unused raw columns, zero
incremental value) — now in a second model family: **redundant inputs are not free.**

**And nothing else the GRU is missing helps either.** e0114 tested the one family a per-user
sequence model provably *cannot* build for itself — cross-sectional rank, strictly outside the
hypothesis space at any depth or width, and a keep on the GBDT side (e0004). It also loses
5/5. With day-of-week (e0107, +0.00051, 0/5) that is **four independent input additions, all
negative, all losing every fold**. e0101's 13-channel input is at a sharp optimum, and the
binding constraint is not information — it is overfitting (e0106: 30 epochs cost +0.0204).

**Blend impact: none.** Adding all four new members moves the best family-weighted blend from
1.76268 to **1.76262** (−0.00006). Even swapping the upgraded TCN in for the old one — a member
that is individually better by 0.00165 — moves it to 1.76262, a **3% pass-through**. The seq
family is internally correlated at 0.995–0.998, so member quality barely reaches the blend.
Per §3.4 and the e0060 precedent, a sub-2σ blend delta is not promoted.

**Net: five experiments, one real mechanism, zero change to what we would submit.**

### 3b.2 Should CV be split by user instead of by date? — no, but the holdout has one use

**A user-split CV cannot be the reported metric, and the reason is in the task, not in taste.**
The public/private split is by customer (`TASK.md`), but **both leaderboards cover the same
future window and every one of the 250k test users is a user we already hold history for**.
The train→test relationship is therefore *same users, later date* — pure temporal
extrapolation, with no user-generalisation gap in it at all. §3.1 requires CV to reproduce that
relationship; rule 3 freezes the folds that do.

**But it does have one job**, and it fills a real hole: the seq models train for 12 epochs
chosen by fiat, with no honest signal behind it, while e0106 measured 30 epochs at +0.0204.
`src/seq_usersplit.py` trains fold 4 on 80% of users and scores the fold anchor on both halves:

| epochs | seen users | held-out users | gap | Δ held-out vs best |
|---|---|---|---|---|
| 6 | 1.73363 | 1.73065 | −0.00299 | +0.00042 |
| **12** | **1.73351** | **1.73023** | −0.00328 | **0.00000** |
| 20 | 1.73640 | 1.73345 | −0.00295 | +0.00322 |
| 30 | 1.74713 | 1.74886 | **+0.00173** | +0.01863 |

Two findings, and the second matters more than the first:

* **The held-out-user curve picks 12 epochs — exactly the frozen-fold optimum** — and prices
  30 epochs at +0.0186 against the folds' +0.0204. So it is a **valid, free epoch selector**
  that costs no training anchor, unlike a temporal early-stopping split (e0017/e0020: the ES
  holdout cost ~4 anchors, and removing it was worth −0.0038). **12 epochs is now validated
  rather than assumed.**
* **User memorisation is only a quarter of the overfitting.** The baseline gap is ~−0.003
  (a fixed population difference between the two random slices); at 30 epochs it moves
  +0.005. But the *seen* users degrade by +0.0136 over the same sweep. Most of the failure is
  temporal — the model fits the training days and generalises worse to a later anchor — and a
  user split is blind to it. It ranks 12 vs 30 correctly here while systematically
  under-reporting the cost of overtraining.

---

## 4. The GBDT approach was saturated — and that is what §3b was for

Everything since e0049 sits inside ±0.0002:

| exp | change | CV | Δ | verdict |
|---|---|---|---|---|
| e0058–e0062 | null-importance filter, 5 cut-offs | 1.76547–1.76911 | best −0.00004 | no effect |
| e0063 | core + `sbcnomoment` + `fcast` | 1.76557 | +0.00006 | `fcast` absorbed by `sbc` |
| e0065 | AutoGluon, RF/XT excluded | 1.76554 | +0.00008 | no effect |
| e0066 | AutoGluon on e0049's 665 features (RF excluded) | 1.76563 | +0.00012 | no effect, confounded |
| e0070–e0073 | recency: train on last 6/10/14/18 anchors | 1.76546–1.76574 | ±0.00005 | flat above 10 anchors |
| **e0080** | **+`funnel` block, the 10 unused raw columns (819 feats)** | **1.76547** | **−0.00004, wins 2/5** | **no effect** |

More than ten consecutive nulls. **GBDT-on-engineered-tabular-features has converged.**

### What was genuinely untouched — status after 2026-08-13

1. **Hyperparameter tuning — still never done, not once.** `lr=0.05, num_leaves=63,
   min_data_in_leaf=200, feature_fraction=0.8, 178 rounds` have been fixed since e0001 as
   deliberate "fast honest defaults" (§5: don't tune a losing feature set). The feature set is
   now emphatically settled. Optuna is already installed. **Still open.** Caveat: do NOT let it
   tune `num_boost_round` against a holdout — the ES set is gone by design and the loss curve
   is flat 80–740 (`reports/eda/diag_es.json`).
2. ~~**A genuinely different model class**~~ — **DONE, and it worked.** See §3b. This was the
   right call: it was the only route to real blend decorrelation, and the measured correlation
   drop 0.9983 → 0.9946 is where the entire −0.00239 comes from.
3. **Multi-seed averaging** — done for `nn_seq` (4 seeds of e0101, −0.00104 for the seq family
   alone, and all four are members of e0120). **Never run on the GBDT side**; σ_noise 0.00009
   there so the ceiling is small, but it is nearly free.

---

## 5. The results that matter most

Ordered by size of effect.

1. **The panel is conditioned on future activity.** All 250k users are active in each of the
   three 30-day blocks ending 2026-02-13. Validating on an overlapping anchor is optimistic by
   **+0.041**. Frozen folds stop at 2025-10-16 (`DATA.md` §4).
2. **Guard-zone anchors are also unusable for TRAINING** — measured 2026-08-13, not assumed.
   Validating at 2026-01-14: clean-only (29 anchors, 6.07M rows) **1.68068** vs +guard
   (38 anchors, 8.18M rows) **1.68258**, **Δ +0.00189**. Contamination beats 35% more data,
   and beats our own "training-set size wins" finding. The marginal target distribution is
   nearly identical between zones (P(y=0) 44.4% vs 43.6%) — **matching marginals told us
   nothing about the conditional relationship.**
3. **The early-stopping holdout was contaminated**, −0.0038 to fix; largest single gain.
4. **RMSLE ≈ 2·sd_L·sd_M·(1−ρ)** — see §2. Everything is correlation.
5. **Feature engineering has hit diminishing returns.** Base 62 features bought −0.15 vs naive;
   everything since (≈50 experiments, 800+ features) bought −0.0092 combined.
6. **Gain importance cannot see redundancy.** Dropping the whole `sbcmoment` family — which
   contains `sbc_dutycycle_ord_180`, the single highest-gain feature at 2.4× the next — changed
   CV by −0.00008. Twins carried the same information. Same story for the `funnel` block.

---

## 6. Hypotheses tested and killed (2026-08-13, from `my_hypothesis.txt`)

| claim | measurement | verdict |
|---|---|---|
| YoY growth is ~×2 and can be exploited | naive ratio ×1.460 is mostly the guard zone forcing 76.3%→100% activity. **Intensive margin (positive in both windows, n=69,884) = ×1.154.** Optimal multiplier k=0.990; applying +15.4% costs **+0.00474** | **dead** — real but unusable |
| Predict 0 for inactive users | worse at *every* threshold. At t=0.25, 95.7% of zeroed users truly are zero, and it still loses (+0.00001). Break-even needs >96% precision | **dead** |
| The two branches in the pair grid | the split is exactly `no orders in 180d` (18.4% of users, 85.3% zero targets). `ord_last30`/`buy_days_180`/`gmv_last30` are all identically 0 there — already visible to the model | **dead** |
| Weekly instead of monthly horizon | e0024 already tested 7d/14d targets as extra rows: 1.76678 vs 1.76638, **lost**. (e0023, short-horizon predictions as *stacked features*, won and is in the model.) Main motivation was reaching 2026 targets — now measured harmful (§5.2) | **dead** |

---

## 6b. Operational incidents worth not repeating (2026-08-17)

**⚠ `oof/e0111.parquet` is corrupt (found 2026-08-18).** 2.8 MB against ~13.9 MB for every other
seq OOF; pyarrow reports "Parquet magic bytes not found in footer" — a truncated write, almost
certainly an interrupted transfer from the cluster. e0111 (cnngru backbone, CV 1.76620) is
therefore **excluded from every blend analysis** and rule 10's graveyard has a hole in it. The
cluster copy may still be intact and is worth re-pulling. **Check file size against its siblings
after every OOF transfer** — nothing in the pipeline validates a downloaded parquet, and this one
sat unnoticed since 2026-08-13.

**⚠ `experiments.csv`'s `delta` and `significant` columns do not mean what CLAUDE.md §4.3 says
(found 2026-08-18, affects 97 of 98 rows).** §4.3 defines `delta = cv_mean − parent.cv_mean`.
In fact `cv_mean − delta = 1.92862` for **97 rows regardless of `parent_id`** — every delta is
measured against the fixed naive floor, not against the declared parent. `significant` follows
it, which is why **every row in the log says `yes`**: beating the naive floor is not the question
any experiment was asking. The two columns that exist to answer "did this change help?" have
been answering "is this better than predicting nothing?" for the whole project.

Nothing downstream was misled — every promotion decision in this log was made from the per-fold
list and an explicit parent comparison, not from these columns — but the log cannot be read at a
glance and `significant` should not be trusted by anyone resuming from it. **Not fixed
unilaterally:** rewriting 97 rows of a frozen shared log is the user's call, and the
non-destructive repair is to add `delta_parent` / `significant_parent` alongside rather than
overwrite. Flagged, not actioned.

**I overwrote a teammate's `scripts/tune.slurm`** by writing a heredoc to that path without
checking whether it existed. Their running array survived (SLURM captures the batch script at
submit time) but a resubmit would silently have launched my GRU tuner instead of their LightGBM
one. Recovered with `scontrol write batch_script <jobid> -`, which dumps the submitted script —
worth knowing, since this repo is not a git repository and there is no other history. My version
now lives at `scripts/tune_seq.slurm`.

**OOF filename collisions.** `run_usercv.py` builds its output tag from variant + mixup only, so
three different models (plain mixup, tuned mid-cosine, tuned annealed) all wrote to
`oof/usercv_full_mixnaive.parquet` in turn. No measurement was corrupted — the local copies were
pulled under distinct names between runs — but any blend read from the cluster copy would have
silently compared the wrong model. **A tag that does not include the hyperparameters is not a
name.** `--tag` is still not implemented; do it before the next OOF-writing run.

> **2026-08-20: this note came true, destructively, and is now fixed.** The `--month-norm`
> screen inherited tag `extra` and **overwrote the `usercv_extra` 5×3 baseline's own
> `oof/usercv_extra.parquet` and `reports/eda/usercv_extra.json`** with a single-fold
> month-norm run — a CLAUDE.md rule 10 violation (never delete an OOF). The report was
> restored from a local copy; **the Aug-13 baseline OOF parquet was not recoverable** and was
> regenerated by re-running the control (e0197). Fixes now in `run_usercv.py`: the tag is
> computed once as `ck_tag` **before** the fold loop, includes `_mnorm`, and now also names the
> **checkpoints** (`runs/usercv/{ck_tag}_f{k}_s{seed}.pt`), which had the same collision. The
> run also logs its standardisation mode and artefact tag on startup — added because a
> mis-submitted run was indistinguishable from its control in the logs (below).

**A flag omitted from an `sbatch` line is invisible.** The first `--month-norm` confirm was
submitted as `sbatch scripts/usercv_monthnorm.slurm --variant extra ...` **without
`--month-norm`** — the slurm script only forwards `"$@"` and bakes in nothing, so the job ran as
a second global-norm control. Nothing in the log said which normalisation was active, so it read
as a valid experiment whose curve merely happened to match the control. **A run must print the
thing it is varying**; `run_usercv.py` now prints `standardisation: ... | artefact tag '...'`.

**Scratch quota exhaustion masquerades as a torch bug.** Two confirms died at ~8 min inside
`torch.save` with `RuntimeError: [enforce fail at inline_container.cc:672] . unexpected pos 7168
vs 7062`. It looks like a corrupt checkpoint or two jobs racing the same path; it was neither —
the jobs ran sequentially, and `df` showed 63 TB free **filesystem-wide**, which is not the same
question as the **per-user quota**. GPFS hits the quota mid-write, silently truncates, and
torch's zip finalizer reports an offset mismatch. Diagnostic that settles it in one step:
serialise to `io.BytesIO` and write the bytes yourself — the error becomes the real one,
`[Errno 122] Disk quota exceeded`. Symptom to recognise: saves below ~2 KB succeed, everything
larger fails at a **constant** byte offset.

**I ran a Panel build on the login node**, which `CLUSTER.md` rule 5 explicitly forbids; the node
killed it, exactly as documented. Verification scripts go through SLURM like everything else
(`scripts/chk_resid.slurm` is the pattern).

**⚠ I OOM'd the user's laptop with a screening script (2026-08-19).** `run_causal_lgb` did
`Xs = (X.astype(np.float32) - mu) / sd` on the full `(users × days × channels)` tensor: a 2.09 GB
float32 copy plus a same-size transient, alongside the 1.04 GB float16 source and a concatenated
candidate tensor — ~6 GB of peaks on a **8.6 GB machine that often has ~2.4 GB free**. Fixed by
gathering only the *scored rows* before promoting to float32 (`_gather`), passing candidates as
`extra=` instead of concatenating, and `del raw` once shapes are captured. **Measured peak after
the fix: 2.00 GB**, and the baseline reproduced to 1.75022 vs 1.75027 recorded. **A screen is
allowed to be slow; it is not allowed to take the machine down.** Size the peak *before* running
anything that touches the full panel locally — the cluster exists for the rest.

**`resource.setrlimit` is not a memory guard on macOS.** The obvious fix for the above —
`RLIMIT_AS`/`RLIMIT_DATA` — silently does nothing: both start at `RLIM_INFINITY` and the kernel
refuses to lower them ("current limit exceeds maximum limit"). My `cap_memory()` was a **no-op
that looked like a safety net**, which is worse than no guard. Replaced with a psutil RSS
watchdog thread that `os._exit(137)`s on breach. ⚠ The first watchdog test appeared to *fail*
because macOS compresses `np.ones` pages — RSS actually fell while the array grew; retested with
`standard_normal` and it aborted correctly at 1.24 GB. **Test the guard with incompressible
data, and test that the guard itself fires.**

**A screen is not a small confirm — it has a different sensitivity, and bundle size is the
confound.** The local proxy trains on 13.5k rows against the real pipeline's ~66k/fold, so it is
overfit-starved and penalises an N-column bundle roughly N× harder than the 1-column noise
control it is judged against. A 60-column block scoring "nil" there therefore does **not** mean
its parts are nil — that is why §1o decomposed into 11 families **at both anchors** before
concluding. See the graveyard's `ds_order` row for the same instrument overstating by 36×.
**Never report a screen delta without the control at the same column count.**

---

## 7. Operational

**Partitions** (`CLUSTER.md` rule 5b, admin 2026-08-13): **never occupy a GPU partition for
CPU-only work.** LightGBM / EDA / plotting / exports → `compute` (385 GB) or `highmem`
(772 GB), no `--account`, no `--gres`. Only AutoGluon (`run_ag.py`) needs `apini`
(`--account=pilot_apini --gres=gpu:nvidia_h200_nvl:1`); it trains NeuralNetTorch/FastAI.

**Scratch quota**: 3.26 TB against a 3 TB soft limit, 6 days grace, 2.74 TB to the hard limit.
Our own footprint is ~3 GB + the feature cache. The overage is the co-tenant's.

**Feature cache** (`feature_cache: true`): verified bit-identical, **87× on feature building**
(101.5 s → 1.2 s per anchor). Key hashes `features.py` + `data.py`, so any code edit starts a
new generation — `data.py` now has 16 value columns, so the old generation is stale.
`scripts/clear_cache.slurm` prunes. Budget 40 GB is *not* honoured under concurrency (six
writers each track their own counter; observed 53 GB, plateaus ~70 GB).
`assert_no_lookahead` disables the cache while it runs — a hit there would silently defeat the
guard.

**Two AutoGluon design points that must not be lost** (`src/run_ag.py`):
* `num_bag_folds=0` — the same user appears at ~20 anchors, so AG's default random KFold would
  put near-duplicate rows on both sides and tune ensemble weights on leaked validation;
* `refit_full('best', train_data_extra=<embargoed anchors>)` — `fit()` may only see anchors
  ≥30 d from the tuning anchor (that is what makes selection honest), which costs 5 of 25
  anchors; `train_data_extra` hands them back at refit so AG trains on the **same anchor count
  as LightGBM** (verified 8/12/16/20/25 per fold).
* **Do NOT exclude RandomForest.** It is slow (13–16k s) but carries real ensemble diversity:
  e0064 (with RF) beat e0060; e0065/e0066 (without) are ties. My exclusion was an error.

---

## 8. Local exploration

`explore.ipynb` — 41 cells, runs on the laptop with numpy/pandas/matplotlib/pyarrow only.
Needs three files from the cluster (`sbatch scripts/export_local.slurm`, then scp):
`reports/local_extract.parquet` (225k users × 120 features + target), `local_importance.csv`,
`local_meta.json`.

Sections: raw events · calendar with guard zone and gifting dates · **model predictions on the
calendar** (fold predicted-vs-actual, ratio 0.38–0.44, deliberate) · per-user sparsity · target ·
live metric decomposition · fold timeline · `feature_vs_target()` · **`pair_grid()`** ·
correlation · single-user explorer · **§12 hypothesis testing** (all of §6 above, with plots).

High-resolution pair grids: `reports/eda/hi_pairgrid_{scatter,binned}.png`, 39,476 × 42,622 px.

---

## 9. Next actions — the endgame (2026-08-14)

Modelling is done (§1b). What remains is not score, it is *keeping* the score and the second
scoreboard.

1. **Final-2 selection: `e0150` + `e0151`.** Measured, not projected:
   * `e0150` **1.64670** — affine-calibrated; leans on two probe constants plus one LB-derived
     rho. The better score, and the primary.
   * `e0151` **1.64748** — shift-only; leans on the probe constants ALONE, no rho. Costs
     +0.00078, which is the measured value of the spread correction.
   **NOT `e0152`.** It ties e0150 (1.646697, −0.000003) while carrying nine blend weights
   fitted directly on leaderboard scores with no validation set — strictly dominated, and the
   private set is 200k different users. I earlier proposed e0150+e0152 on the grounds they were
   "built differently"; that was wrong reasoning. Differing in construction is not hedging when
   one of the two concentrates all the leaderboard-fitting risk.
2. **Verify the calibration transfers to private.** `mu_L` is estimated on the 50k public
   users; its sampling sd is `2.32/sqrt(50000) = 0.0104`, and the penalty is quadratic, so a
   0.010 error costs 0.0001 RMSLE^2 (~0.00003 RMSLE). Low risk, but write it down rather than
   trust it.
3. **Do not spend submissions on differences below 0.0006.** That is the demonstrated error of
   the projection method itself (e0150: predicted 1.64610, actual 1.64670). The non-negative
   10-member blend projects -0.0005 -- inside its own error bar. The *unconstrained* 10-member
   blend projects 1.6263 and is a trap: ten predictors correlating at 0.99, nine free weights,
   ten observations, three of them negative. That is the shape of a number that wins on public
   and collapses on private.
4. **The jury track**, which is the scoreboard still in play. Top-15 private buys entry, then
   the repo is judged on quality of work with data and models, resource efficiency, and novelty
   (TASK.md). Assets, to be presented deliberately rather than left implicit:
   * the metric solved exactly from two probe submissions (§1b) — and used to *predict* a
     submission's score in closed form before sending it, to within 0.0006;
   * frozen folds, a guarded population rule, and a graveyard of measured negatives;
   * machine-checked causality (`assert_causal`, `assert_no_lookahead`, `assert_causal_features`)
     with a vacuity check, which caught a real look-ahead in `tenure`;
   * a 5-fold GPU model in **3.7 minutes** against AutoGluon's 8 hours — the resource-efficiency
     criterion, met by a factor of ~130;
   * negative results reported as results: the classification ceiling, the dense-supervision
     null, the calendar-extrapolation failure predicted *before* submission and confirmed.
   * Gini improved with calibration (0.7876 vs e0120's 0.7685, truth 0.818). The RMSPE
     tie-breaker stays structurally unwinnable (+0.163 to fix, DATA.md §8.4) and is documented
     as a deliberate trade-off rather than an oversight.

## 10. Open hypotheses still worth testing

0. **The hyperparameter × fold-index interaction of §1l — does the fold set have a *late regime*
   the mean is averaging away?** Eight tuned configs, 8/8, are better on folds 3–4 and worse on
   folds 0–2, at up to 8 sigma per fold. The test anchor sits later than every fold, so the
   5-fold mean may be the wrong selection statistic for *any* hyperparameter choice — including
   e0101's own, which was never tuned. Cheapest test that is not confounded by train-size:
   re-run e0101 and e0180 with a **fixed-length training window** on all five folds; if the
   spread survives it is recency, if it collapses it is data volume. **Do not chase this for the
   blend** — §1l prices the whole family at 0.00007 RMSLE. It is worth running only if it would
   change how *every* future experiment is scored, which is a much bigger claim and the reason
   to keep it on the list.
1. **Normalise the long-window features by history actually available.** `gmv_365` covers 93
   days at the first training anchor, 289 at the last, and a full 365 at test — no training
   anchor ever gets a full year. *Dropping* them lost (e0056 1.76844, e0057 1.76658);
   *normalising* is untested and is a different intervention.
2. ~~**catch22** (`PAPERS_new` §8.3) — 22 pre-vetted, redundancy-minimised features.~~
   **Downgraded to `park` 2026-08-20 by §1o.** tsfresh's 60 statistics — which include catch22's
   own family (autocorrelation structure, level strikes, trend, complexity/LZ) — scored
   **−0.00006, 2/5 folds** on the frozen folds, and no statistic family cleared the noise
   control at *both* screen anchors. catch22's selling point is redundancy-minimisation within
   the shape-statistic vocabulary; §1o says that whole vocabulary is already spanned by
   `sbc`+`tsfeat`, so a smaller basis for it is not a new direction. Run only if something
   *outside* that vocabulary motivates it. **If it is run anyway: validate against the
   `pycatch22` reference implementation first** — §1o found 3 of 7 hand-ports wrong.
3. ~~**BTYD** (`BACKLOG` e0033)~~ — **DONE 2026-08-16, killed. See §1e.** +0.0702 behind
   e0049, blend gain −0.00006 against a −0.00050 threshold, and its latent columns are worth
   −0.00008 over the no-op control because the RFM inputs were already in the feature set.
   The durable result is the **+0.5626 functional gap** between `E[log1p y]` and `log1p(E[y])`.
4. **Per-segment models** (the one live variant of the branch finding). **Note §1e closes the
   obvious version of this:** BTYD loses in every decile of the blend prediction by a flat
   +0.06…+0.10, so routing users to a second model by prediction level has nothing to route.
