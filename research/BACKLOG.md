# BACKLOG.md — ranked hack list + graveyard

Scoring per `CLAUDE.md` §6: **priority = (expected gain × probability) / cost**. Cheapest-first
within a band. Every entry cites the measurement that motivates it — an idea with no evidence
behind it is not ranked higher for being clever, but it is still allowed in (§1: cost of
testing is the only filter for cheap ideas).

Cost unit: one confirm-tier run = 5 folds ≈ 12 min on `compute`. "1c" = one such run.

---

## Current state

| | |
|---|---|
| best CV | **e0001 = 1.77847 ± 0.02175** (5 frozen folds) |
| best LB | **1.6766** |
| CV−LB offset | +0.102, deltas transfer at 93% |
| naive floor | geo3 = 1.92862 · p30 = 2.25296 · zero = 3.31 |
| oracle ceiling | split+geo3 = 1.349 · split+constant = **0.983** |

**The metric is `≈ 2·sd_L·sd_M·(1 − corr)` and the sds barely move (DATA.md §9.4). Every point
comes from raising log-space correlation between prediction and target.**

---

## Band A — run now (high evidence, ≤ 1c each)

| id | hack | why (measured) | cost | prio |
|---|---|---|---|---|
| e0002 | **`counts` block**: buy-day counts per window, buy-day rate, orders/buy-day, order counts at 3/21/45/120/270 d, max orders in a day | `ord_sum_180` carries **4× the gain** of the next feature and 4 of the top 6 are counts. Counts correlate 0.586 vs GMV's 0.557 and are far less heavy-tailed | 1c | **A1** |
| e0003 | **`trend` block**: EWM of gmv/orders at half-lives 7/30/90; activity and GMV centre-of-mass within 30/90/365 d; consecutive-block ratios | lag-block corr decays slowly and monotonically (0.557 → 0.416 over a year) — direction of travel is not yet encoded, only levels | 1c | **A2** |
| e0004 | **`rank` block**: cross-sectional percentile of gmv_30/90, ord_90, geo3, recency **within the anchor** | fold level drifts hard (E[L] 2.13 → 2.44 across anchors, DATA.md §6.2) and the test sits outside that range. Ranks are level-invariant, so they should transfer where raw levels cannot | 1c | **A3** |
| e0005 | **`visit` block**: empty-day vs non-empty-day counts per window, streaks | 14.85 % of rows are visits with zero search/catalog activity; they carry independent signal (corr 0.147) that `days_W` currently blends away | 1c | A4 |
| e0006 | **`channel` block**: search/catalog GMV and orders at more windows, catalog share | catalog is 7.35 % of GMV and nearly uncorrelated with search (row-level 0.054) — decorrelated signal is worth more than its volume | 1c | A5 |

## Band A+ — the seasonal year-lag (NEW, biggest measured signal outside infrastructure)

| id | finding | number | file |
|---|---|---|---|
| — | **Spikes ARE repeatable.** Same calendar window (15 Jan–13 Feb) measured in 2025 and 2026, level removed via a common reference and the shared-denominator artefact partialled out | corr **+0.137** (raw +0.273 was inflated) | `reports/spikes.log` |
| — | **Spring-2025 spikers stay elevated.** Top 10 % / 5 % / 1 % of spring spikers, matched on reference level | **+0.176 / +0.187 / +0.198** log-points in Jan–Feb 2026 | `reports/spikes.log` |
| — | ~~Year-lag survives all our controls~~ **— RETRACTED.** partial corr +0.0896 and implied −0.0069 were measured against a **linear** baseline. Re-measured against the **actual GBDT residual** at anchor 2026-01-14: incremental R² = **0.000135**, implied RMSLE **−0.00011** (≈1σ_noise). The GBDT had already absorbed **98.3 %** of it via the 365-day window and its interactions | `reports/yearlag_coef.log` |
| — | The apparent "−0.0126 gain" in that test is an **intercept** effect, not the feature: all five candidate lags land within 0.00006 of each other. A per-anchor bias shift, and `calibrate.py` already showed a global shift does not help on clean folds (k\* = 1.000) | | |
| — | **The mechanism itself is dead, CV-validated.** Does the model under-predict spikers? Pooled over all 5 frozen folds: `corr(spike, residual) = +0.0001`, top-decile spiker residual +0.0123 vs −0.0045 elsewhere (difference **+0.017**, versus the **+0.18** the hypothesis needs). **Every** multiplier k>1 makes CV worse; best k = **1.00** | `reports/spiker_residual.log` |
| — | Max achievable prize even if the hypothesis were true: **−0.00078 RMSLE** on 8.1 % of users — 15× below LB 2σ noise (0.0118), so no submission could ever have verified it | arithmetic on `subs/e0020.csv` |
| e0025 | **Verdict: do not build the year-lag feature, do not apply a gifter multiplier.** Narrow pre-holiday variants (last-2w, spike-vs-own-reference) score no better than the full month — coefficients 0.004–0.021, corr with residual 0.005–0.010 | — | **kill** |

**Note on the earlier gifter null:** it is not contradicted. That test used 7–17 day holiday
run-up windows, where a per-user lift is mostly noise (reliability 0.09). The spike tests use
30-day and 3.5-month windows against a 6.5-month reference. Narrow gift-response is not a
trait; broad seasonal position is.

## Band B — structural, run after Band A settles

| id | hack | why | cost | prio |
|---|---|---|---|---|
| e0010 | **Two-part model** `P(buy) × E[GMV∣buy]` | oracle split + a *single constant* scores **0.983** vs 1.834 for the best naive. Classification is the dominant lever and is completely untested | 3c | **B1** |
| e0011 | Train on the dirty anchors too (adds 3 months of recent data) | open question DATA.md §11.2; CV↔LB now trustworthy enough to believe the answer | 1c | B2 |
| e0012 | Recency weighting of training anchors (several decay rates) | test-period predictability is higher than fold-period (corr 0.577 vs 0.538) — recent anchors are more representative | 1c | B3 |
| e0013 | Per-anchor target normalisation: predict `log1p(y) − mean_log1p(anchor)`, add back | removes the level drift that rank features only sidestep | 1c | B4 |
| e0014 | Tweedie / Poisson objective on raw GMV; Huber on log1p | zero-inflated non-negative target is textbook Tweedie; may beat L2-on-log1p | 1c | B5 |
| e0015 | Per-segment models by prev-30d order count (0 / 1 / 2-4 / 5+) | P(y>0) ranges 0.279 → 0.909 across these segments with very different E[y∣y>0] | 2c | B6 |

## Band C — post-processing and blending (cheap, do once the model is settled)

| id | hack | why | cost | prio |
|---|---|---|---|---|
| e0020 | Global multiplier fitted on OOF; retransformation-bias correction after `expm1` | e0001 under-shoots total GMV by 43 % (11.98 M vs ~24.4 M expected) | 0.1c | C1 |
| e0021 | Per-segment bias correction from OOF residuals | same, but conditional | 0.2c | C2 |
| e0022 | Seasonal multiplier ~1.16 for the test window | measured on the 2025 calendar analogue (DATA.md §5.4); helps RMSPE, unknown effect on RMSLE | 0.1c | C3 |
| e0023 | Blend with a linear/ridge model on log-lags; rank-mean fallback | `CLAUDE.md` §6 requires a non-GBDT alive into Phase B; near-zero cost | 1c | C4 |
| e0024 | Multi-seed averaging | cheap variance reduction, but only worth it once the feature set is frozen | 2c | C5 |

## Band D — expensive or speculative

| id | hack | why | cost |
|---|---|---|---|
| e0030 | Similar-user aggregates (k-NN on activity shape) | classic LTV lift; O(n²) unless approximated | 5c |
| e0031 | Cluster id from series shape as a categorical | cheap once clustering exists | 2c |
| e0032 | Sequence NN (GRU / N-BEATS) on tokenised daily behaviour | organisers hint at it; only if GBDT plateaus (`CLAUDE.md` §6) | 20c |
| ~~e0033~~ | ~~BTYD (Pareto/NBD, BG/NBD) as a feature source~~ | **DONE 2026-08-16 → graveyard.** Both routes killed: as a blend member −0.00006 (threshold −0.00050), as columns −0.00008 vs the no-op control. Ran in 8.6 min on the laptop. See EXPERIMENTS.md §1e | 0.7c spent |
| e0034 | AutoML (AutoGluon) stacking on our folds | `CLAUDE.md` §5 step 5 — confirmation, not exploration | 10c |

---

## Explicitly deferred

* **Hyperparameter tuning** — deferred to an AutoML pass once the feature set is settled
  (`CLAUDE.md` §5: tuning a losing feature set is wasted compute). Note that all e0001 folds
  hit the 3000-tree cap, so absolute gains are measured on an undertrained model; paired
  comparisons remain valid.
* **Feature-importance filtering** (§5 steps 3–4) — pointless at 62 features; revisit past ~300.

---

## In flight

*Added 2026-08-21. Derivation in EXPERIMENTS.md §1p; decision rule in FEATURES.md.*

| id | what | status |
|---|---|---|
| **e0210** | XGBoost on e0049's exact 665 features, frozen folds, hyperparameters mapped from LightGBM. The first frozen-fold measurement of a second GBDT family at comparable settings — §1c's XGB/CatBoost/Ridge kills were all user-split AND untuned, so −0.032 was never that family's ceiling | array 24055170 |
| **e0211–e0213** | FEATURES.md candidates A/B/C (`age_bucket_gmv_share_3`, `cart_backlog_7`, `cohort_rel_buy_rate90`), one feature each, parent e0210 | same array |
| **e0214** | **noise control** — one i.i.d. normal column. The yardstick e0211–e0213 are read against; not a hypothesis about the data | same array |

Pre-registered: keep only if Δ vs e0210 beats 2σ_noise or wins ≥4/5 folds **and** exceeds
\|Δ(e0214)\|. Prior (§1m, and ten consecutive feature nulls): three nulls.

## Operational lessons — cheap to repeat, so written down

*Added 2026-08-21, from the e0210 submission (EXPERIMENTS.md §1p).*

* **Changing a SLURM partition means changing `--time` too.** A `computeshort` 1 h limit rode
  along into a `compute` job and TIMEOUT'd it at 01:00:10 with 4 of 5 configs validated.
  Nothing was lost (the smoke ran `--no-log`), but it cost a resubmission.
* **`--dependency=afterok` on a job that fails leaves the dependent array parked silently**
  (`DependencyNeverSatisfied`), not failed. The chain is still the right pattern — it stopped
  5 full CV runs launching behind an unvalidated harness — but the dependency state has to be
  read, not just the queue.
* **Prove the leak guard fails before trusting that it passed.** Write the leaky version of a
  new feature, confirm `assert_no_lookahead` catches it, then write the correct one. Two
  minutes on `data/_screen_subset.parquet`, no cluster time. REVIEW_NOTES.md §B0.
* **Feature blocks can be verified on the laptop even though training cannot.** The 15k-user
  screen subset builds a real `Panel`; every new block was checked against an independent
  brute-force implementation (agreement 0.0 to 1.2e-07) before any job was submitted.


## Open — built, unfalsified, never submitted

*Updated 2026-08-24 after e0361 came back. The guard-zone question is now CLOSED (see the
graveyard); what it opened is the inverse operation on the largest slot.*

| id | what | status |
|---|---|---|
| ~~e0361~~ | ~~Seq family retrained through the guard zone~~ | **SUBMITTED, LOST +0.000350 → graveyard** |
| ~~e0384~~ | ~~RETRACT the usercv slot out of the guard zone~~ (was "e0362") | **CLOSED ON CV (e0387/e0388) — measured to lose 0/5 folds, no slot spent** |
| ~~e0370–e0372~~ | ~~Seq-half architecture members at d48~~ | **CLOSED on OOF (e0390) — all three gains reach the blend as +0.000000** |

**e0390 — the "ready to fold in" item, closed for zero submissions.** Swapping the three
architecture members for their d48 versions (leaving the 4 GRU seeds alone, since BACKLOG records
that narrowing *those* was worse by destroying seed diversity) gives, on 1,062,003 common OOF keys:

```
member rho   TCN +0.000057 · xformer +0.000195 · GRU-deep +0.000364   (all positive)
seq half     +0.000038      corr(old, new) 0.999911
FULL BLEND   +0.000000      2/5 folds; fold 4 (most test-like) -0.000002
projected    1.646454 vs e0301's 1.646456, at BOTH 1.00x and 0.88x pass-through
```

**The dilution chain, measured:** a member gain of +0.00036 becomes +0.000038 at the half (4 of 7
members are unchanged *and* the members correlate at 0.9999, so one member's gain is averaged away
~9×), then +0.000000 at the blend (weight 0.38, then pass-through). This **replicates §3b's
"improving a seq member to improve the blend = 3 % pass-through"** — e0113 was a genuinely better
TCN at −0.00165 on 5/5 folds and moved the blend by −0.00006. The width finding is real for the
fourth time and is still worth nothing here.

**Why e0362 follows directly from e0361's loss.** `predict_usercv.py` sets
`last_anchor = max_anchor(raw)` = **2026-01-14**, so the champion's largest component trains
through the guard zone while the gbdt and seq halves stop at 2025-10-16. e0361 just measured that
extending a half *into* the guard zone costs +0.00035 at weight 0.38. If the penalty is a property
of the data rather than of the seq architecture, the 0.42 slot is carrying it now. It is also the
transfer-safe category — single-member contents swap, weights untouched, 4-for-4 this session.

**Two things to do before spending a slot on it**, both learned this session:

1. **Build a matched unretracted control on the same partition** and check
   `corr(matched control blend, e0301) ≥ 0.99995` locally, as `e0382` did. The usercv member is
   only a **3-seed** average, so hardware drift shrinks like `1/√3 ≈ 0.58` — much weaker
   protection than the seq half's `1/√7`, and the single-member ratio was only 2.29×.
2. **Price it through the slot's pass-through before predicting**, not after. e0361's
   pre-registered magnitude was 3× too pessimistic for exactly this omission.

**✅ BUILT (e0383–e0385) AND THE PRIOR RESOLVED DOWNWARD — do not spend a slot.** The control worry
was misplaced in the best way: `e0295` was itself built on andrena, so `e0383` is a same-architecture
rebuild and reproduces it **exactly** (`corr 1.0000000`, blend residue 2.1e-06). One slot would
suffice. But the diagnostic argues against using it:

* **The retraction moves the member the wrong way.** `r vs (gbdt+seq)/2` goes **0.99559 → 0.99764**.
  The usercv slot is paid for *decorrelation* (§1b: its solo rho is the lower of the two), and part
  of that decorrelation is simply that it trains on a **different window** — `predict.py:77` and
  `run_seq.py` cap the other two halves at 2025-10-16. Retracting aligns the windows and spends it,
  while 27 % less data lowers `rho_B` too. Both terms point the same way.
* **e0361 re-read (e0386) says the same thing from the other side.** Extending the seq half moved it
  **+0.00257 toward the guard-trained usercv member** and **−0.00075 away from the clean gbdt half**
  — a *selective* move, which a pure degradation cannot produce.

> **The finding both experiments share:** the champion's three slots are **training-window-diverse**
> and that diversity is **load-bearing**. Aligning their windows in *either* direction destroys blend
> value. The long-standing "contradiction" is a decorrelation source, not a bug.

✅ **NO LONGER A PREDICTION — measured on CV (e0387/e0388), no slot spent.** The question the
project recorded as unanswerable by CV *is* answerable on this path, because it splits by **user**:
train through the guard zone, score held-out users at an **earlier clean anchor**. New additive flag
`--train-cap` decouples the training mask from the scoring mask (they were tied via `Mg`, which is
why it looked impossible; `--t-cut` always scores *after* its cut).

```
                          rho_in     B wins    rho_partial (paired, same family + keys)
A guard grid (incumbent) 0.659190      —            0.04632
B retracted (clean)      0.659062     0/5           0.03970      -14.3% of blend value
                         -0.000128  3.97 sigma
```

**Retracting makes the member worse on both terms** — quality `rho_B` −0.00029 *and* correlation
with the family +0.00033. So the r-only argument was half the story. Port gates: arm A's training
mask is 80,899,560 user-days, the exact figure §1z-C quotes independently for this path; 58,830,230
matches `predict_usercv.py --guard-clean` exactly.

⚠ **Confound, pre-registered and unresolved:** B trains on 27 % fewer user-days, so "guard-zone
anchors help" is not separated from "more data helps". A win for B would have been clean; this loss
is not. ⚠ The absolute 0.046/0.040 are against the 9-member family which *excludes* the usercv slot
(§E1 inflation) — only the paired difference is trustworthy.

> **Unified reading of e0361 + e0387/e0388, and it retires the "contradiction".** Guard-zone anchors
> improve a member's own quality *and* make window-sharing members more alike. The champion sits
> where the windows **differ**, collecting quality on the slot that has them and decorrelation from
> the slots that do not. Extending the seq half in traded one away; retracting the usercv slot trades
> the other. Both measured negative. **The differing training windows are load-bearing, not a bug.**

**Open follow-up this creates:** if window diversity is load-bearing, *widening* it is the untested
direction (a member trained on a deliberately different window). Prior is poor — BACKLOG's
lookback-restricted ensemble found that diversity-by-removing-information yields "a strictly degraded
copy" — but that restricted the FEATURE lookback, not the training ANCHOR window, so it is not the
same experiment.

**Why it is still open when everything else is closed.** The +0.00189 that justified excluding
guard-zone anchors from TRAINING was measured by validating **at 2026-01-14 — an anchor inside the
guaranteed-activity zone**. No clean anchor exists to re-test it, because "clean" *means* the target
window ends before 2025-11-16 and the last such anchor **is** 2025-10-16. So the exclusion rests on
a contaminated measurement that no internal fold can check.

Meanwhile the project already contradicts itself across components: **e0141 (weight 0.42 of the
champion) trains through 2026-01-14**, while the gbdt and seq halves stop at 2025-10-16.

**Why the seq arm and not the gbdt one.** The gbdt version (e0360) was built and measured first:
the extension is real (`corr(extended, e0090) = 0.997797`, *more* different than
`corr(e0090, e0064) = 0.998972`), but only `e0090` is extended and it enters at effective weight
0.10, so the resulting blend correlates **0.999976** with the champion — inside the 0.00038
paired-50k noise, i.e. an unreadable slot. **The seq slot carries 0.38 fully**; scaling that
perturbation 3.8x puts the blend correlation near 0.9996, comparable to e0150-vs-e0162, which *is*
measurable.

**It is also the transfer-safe operation.** It changes member CONTENTS with weights untouched — the
only category that transferred this session (4-for-4 on exact projections), versus fitted weights
and averaging, which both inverted sign.


⚠ **e0361 was held, and is now released (2026-08-24).** The reconstruction control failed first:
rebuilding the *unextended* `e0101` on andrena reproduces the apini-built `subs/e0101.csv` only to
`corr 0.998287`, because `cudnn.deterministic` does not hold across GPU architectures. On a single
member the intervention (`1-corr 0.003927`) beat that hardware drift (`0.001713`) by only **2.29x**
— not enough separation to spend the slot.

✅ **The fix worked and the prediction behind it held (e0382).** Seven matched *unextended* rebuilds
on the same partition confirm that drift is random per member and averages down while the extension
is systematic and does not: at seq-half level drift fell 8.6× (`1-corr 0.000200`) against the
intervention's 2.3× (`0.001727`), widening the ratio **2.29x → 8.64x**. At blend level
`corr(matched-unextended, e0301) = 0.9999687`, passing the pre-registered ≥0.99995 bar, so **87 % of
the extended blend's perturbation is the intervention and 13 % is hardware (~5e-05 RMSLE, an order
of magnitude below the effect).** **ONE slot suffices.** Pre-registered rho map and full detail in
EXPERIMENTS.md §1z-D.

**Honest prior: uncertain, and that is the point.** §5.2's +0.00189 may well be right and this may
lose. But it is the one remaining question where CV *cannot* answer and a submission *can* — which
is precisely what a slot is for. Everything else on this board is closed by measurement.

## ⛔ The entry criterion — read before proposing any new blend member

*Added 2026-08-25 (`EXPERIMENTS.md` §1z-E, full record `SESSION_2026-08-25.md`).*

A candidate is admissible **iff** its pooled 5-fold `rho_B` lands **above** its row here, measured
against the FULL champion (`rho_M = 0.66342`), never against the 9-member family:

```
r vs champion   0.90     0.94     0.96     0.97     0.98     0.99     0.995
rho_B needed  0.6083   0.6324   0.6441   0.6498   0.6553   0.6604   0.6627
```

**Eight members spanning every axis this project has — different losses, information sets,
pretraining corpora, architectures, and orderings of the same computation — trace one curve that
lies strictly BELOW this bar everywhere**, converging to it only as `r → 1`, where the candidate is
the incumbent. There is no interior sweet spot. The exchange rate is ~**1.18 units of `rho_B` per
unit of bar relief**, so buying decorrelation always costs more than it returns.

**The best member ever measured is the joint 13-channel GRU already inside the champion**
(margin −0.00199). Do not spend a submission on a new member that has not cleared this table.

**✅ Replicated on a second population scale (2026-08-25).** The same frontier redrawn entirely on
**fold 4** (`rho_M = 0.675020`) with **RealMLP (e0913) as a 12th point** has the identical shape —
negative everywhere, monotone in `r`, converging only as `r → 1`. So the closure is not an artefact
of pooling. The 12th point matters because it is the first member that is *neither* a tree *nor*
weak: a tabular neural net at essentially GBDT strength, which still lands at `r = 0.9954`. Table in
`EXPERIMENTS.md` §1z-E, "The 12th point". ⚠ Those are fold-4 numbers and may **not** be checked
against the pooled bar above.

⚠ Two measurement rules this table depends on: a **fold-4** `rho_B` may never be scored against a
**pooled** bar (it flatters by ~0.012 and made TS-FM arms appear to clear), and `run_tsfm_gru.py`
needs `--deterministic` for any single-fold comparison (run-to-run `rho_partial` noise is ~0.0014,
the size of the gaps being compared).

## Graveyard — do not re-run

*Added 2026-08-24 — the causal-path capacity sweep and four submissions. Derivation in
EXPERIMENTS.md §1z-B/§1z-D; the one KEEP is `hidden=48`, now in the champion.*

| idea | result | evidence |
|---|---|---|
| **Training on the guard zone — the last question CV could not answer** | **REFUTED ON THE LEADERBOARD, +0.000350 (rho −0.000153).** The seq half (w 0.38) retrained with `--train-through 2026-01-14` scores **1.646806** against e0301's 1.646456. This is the ONE question no internal fold could ever settle: §5.2's +0.00189 was measured by validating *at* 2026-01-14, inside the guaranteed-activity zone, and the last clean anchor **is** 2025-10-16. Direction confirms §5.2 independently. ⚠ **On the LB alone it is 0.92σ** (paired-50k sd 0.00038) — what makes it a kill is the *concordance* of two independent instruments, not the delta. Magnitude reconciles post-hoc at `0.00189 × 0.19 (measured pass-through) = +0.000359` vs +0.000350; **my pre-registered figure was 3× too pessimistic because I applied the component penalty without the pass-through I had already measured.** The e0382 control earned its keep — the effect is **6.9×** the 5.1e-05 hardware residue, so the read is not confounded | e0361, e0380–e0382 |
| **Comparing an ensemble rebuilt on one GPU partition against a submission built on another** | **not fatal, but it must be measured, and the size depends on member count.** `cudnn.deterministic` fixes kernel choice per device, not across architectures: one member rebuilt apini→andrena reproduces only to `corr 0.998287`. But drift is **random per member and averages down** while a config change is **systematic and does not** — across 7 members drift fell 8.6× (1-corr 0.001713 → 0.000200) against the intervention's 2.3×, widening separation **2.29× → 8.64×**, and the residue at blend level was 3.13e-05. **Rebuild the control on the treatment's partition; then check the ratio before spending a slot** | e0381, e0382, CLUSTER.md §10 |
| **GRU `hidden=128` on the causal path** — the spec default, never measured | **WRONG, and it was the last real gain available.** Width curve is unimodal on within-anchor rho: d192 .66462 < d128 .66486 < d96 .66497 < d64 .66513 < **d48 .66525** > d32 .66519. +0.00039 rho, 14× anything else on this path, and it moved the LB **1.646589 → 1.646456** (predicted from OOF to 2e-6). Survives §1j's fixed-epoch control (+0.00023 at 14 ep, **+0.00080 at 22** — the effect GROWS with budget because d128 overfits while the small model keeps learning) and replicates on the independent frozen-fold seq protocol (d48 −0.00099, 5σ, 5/5). **Invisible until now because §1j's search here scored trials on the MIN of an early-stopped curve — the statistic §1j itself invalidated — and §1k redid it properly on the OTHER path** | e0264–e0272, e0280–e0283, e0301 |
| **Blend weights FITTED on CV OOF, LOFO-validated** | **SIGN INVERSION, +0.000593 on the LB.** LOFO said +0.000217 winning **5/5 folds including the most test-like**; measured Δrho was **−0.000259**. ***LEAVE-ONE-FOLD-OUT DOES NOT PROTECT FITTED WEIGHTS*** — it guards variance from overfitting the OOF and is blind to CV↔LB **ranking** shift, which is the actual failure. Strictly stronger than §1m's caution | e0302 |
| **Averaging several models into the usercv slot** (4 architectures, weights untouched) | **SIGN INVERSION, +0.000027.** The averaging step alone: OOF +0.000047, measured −0.000012. §9e had already measured this for *seed* averaging ("variance reduction across GRU runs rather than quality"); **architecture averaging is the same mechanism.** Realisation 0.40× vs the single-swap's 0.88× | e0303 |
| **The affine-invariant correlation loss (IDEAS.md §I2)** | **RUN AND REFUTED, both forms.** mix (MSE+1−ρ) +0.00001 at d128 and +0.00001 on top of d64; pure corr −0.00058. Larger batch is monotonically WORSE at both widths (d128 b256 .66428 > b1024 .66201; d48 b1024 .66273 > b4096 .65831), so the "noisy in-batch ρ estimate" fix is dead too. IDEAS.md downgraded I2 as "sharing HL-Gauss's mechanism" — it does NOT (HL-Gauss changes the estimator of the conditional mean, this is affine-invariant) — but the verdict holds. **Training directly on the scored quantity buying nothing is evidence the model is signal-limited, not mis-optimised.** Keeper: pure-corr reaches **r = 0.879** with the blend, the most decorrelated model this project has built (BTYD .9427, Ridge .9433, every net ≥.997) and 3× closer to the frontier than anything on record — still worth ~0, because decorrelation only pays at comparable quality | e0274–e0275, e0320–e0325 |
| **"The transformer lost because of too few epochs"** | **REFUTED, byte-identically.** Raising the cap 60 → 150 reproduces the SAME rho and the SAME best-epoch vector; early stopping fired naturally, nothing was ever truncated. (Also proves andrena runs are bit-deterministic.) Width WAS a real handicap — §1c tested at d128 and §1k gave `xformer_rope` 2 trials vs the GRU's 13 — and fixing it is worth +0.00036, but GRU d48 is still 0.00069 ahead. **§1c's verdict survives a proper search: attention loses to recurrence here** | e0310–e0314, e0330–e0332 |
| **LSTM as a different recurrent cell** | Level with the OLD GRU (d48 .66487 vs d128 .66486), below the tuned one (.66525). §1k's 0.0029 gap was best-of-2 vs best-of-13; the true gap at matched width is 0.00038. **But rho_partial 0.0195 = 2.4× the incumbent's** — worth more to a blend than standalone, though e0303 shows it cannot be harvested by averaging | e0313, e0314 |
| **§1d's mixup `keep`** | **OVERTURNED ON THE STATISTIC THAT MATTERS.** ΔRMSLE −0.00016 (0.8σ), **Δrho +0.00001**. Mechanism replicates exactly (best-epoch 13.5 → 18.2) but the gain lands in the level/spread term §1b proves calibration discards for free. §1d's −0.00065 was measured against a **one-seed** baseline; §1d's own note said the honest comparison is the 3-seed one. **Do not add mixup to the submitted e0141** | e0265 |
| **Population-matched training** (`--pop-train`) | **Fixes a real code inconsistency for zero gain (−0.00002).** `run_seq.py:160` trains on in-population days only; `run_usercv.py` trained on every masked day, having computed the same mask and used it for reporting only. Measured on the panel: **6.2% of the causal path's 80,899,560 training user-days are DORMANT**, while **100.0%** of the 250,000 test users are in-population by construction. The mismatch is real and the model does not care | e0263 |
| **Explicit regularisation on the causal GRU** | **CLOSED.** dropout 0.2 −0.00001, dropout 0.3 −0.00002, weight-decay 1e-3 +0.00002, layers 3 −0.00008. §1k's dropout≈0.35 finding does NOT transfer from the seq path. This model is helped only by having **fewer parameters**, not by penalising them | e0261, e0262, e0268, e0269 |


*Added 2026-08-24 — research round 3. Full record: `SESSION_2026-08-24.md`, derivations `IDEAS.md` §I25–I28.*

| idea | result | evidence |
|---|---|---|
| **Negative correlation learning (train a member anti-correlated with a fixed blend)** | **refuted by argument.** NCL with the blend fixed minimises `½(f−y)² − λ(f−M)²`, optimum `f* = (y−2λM)/(1−2λ) ≈ y + 2λ(y−M)` (sympy-verified) — a slider between "fit y" (λ→0, r≥0.994 twin) and "fit the residual" (λ→0.5, killed best_iter=4). Not a new axis. Learner-collusion (Abe NeurIPS 2023) + unlearnable residual (§1q) + snapshot/FGE < reseeding all agree. **Manufacturing decorrelation fails; the blend already captures ~all learnable signal (§1s).** | `IDEAS.md` §I25 |
| **Causal-forces trend damping (Armstrong & Collopy 1993)** | **refuted on OOF.** LOFO global damp `g*=1.010≈1`, asymmetric `g+≈g−≈1`, tail nil; shuffled-dev control **−0.090** (full power). Third confirmation (§1q, §1t) the model is conditionally calibrated — no contrary trend to damp | `scripts/causal_forces.py`, `IDEAS.md` §I27 |
| **TS-FM tabular regime (TabICLv2/NN_TORCH) as a member** | e0340 (teammate's TabICLv2, 100k×top-200): r=0.995, rho 0.659 → **rho_partial −0.0005** (decorrelated-but-weak, confirms the I17 gate). NN_TORCH (e0912): r=0.96, rho 0.647, rho_partial −0.004. Strong different-class model does not clear the bar | e0340, e0912, `IDEAS.md` §I22 |
| **RealMLP — a modern tabular NEURAL NET at GBDT strength** (`IDEAS.md` §I22's TOP-1 bet, the last untested function class) | **REFUTED, and it is the sharpest statement of §1c's law the project has.** Fold 4, 500k rows, aligned 223,578 users vs the FULL champion (`rho_M` 0.675020, reproducing §1z-E exactly): **`rho_B` 0.671601 at `r` 0.995356**, e −0.000284, `rho_partial` −0.003996; bar 0.674699 → **margin −0.003098 = 0.101x the requirement**, and the **in-sample optimal weight is 0.000** — fitted on the very fold it was measured on, it earns nothing. **The bet was right about strength and wrong about decorrelation:** §I22 extrapolated NN_TORCH's `r`=0.96 to GBDT strength and predicted e≈+0.022; the strength arrived (0.6716 — the best non-tree, non-recurrent member ever built here, within 0.0021 of the gbdt half) and `r` came in at **0.9954**. **NN_TORCH's r=0.96 was a property of being WEAK (rho 0.647), not of being a neural net** — do not extrapolate an `r` measured on a weak model to a strong one. Slot-wise redundancy is exactly as expected: 0.9965 vs the gbdt half, 0.9949 vs seq, 0.9929 vs the usercv GRU. ⚠ **Do not re-run on a wall-clock excuse:** e0910/e0911 FAILED (AG torchvision), e0913 (1.2M/6h) and e0917 (500k/5h) TIMED OUT, and `IDEAS.md` §I29 logged it "CLOSED — timed out twice" — but a **24h resubmit (job 24152700) COMPLETED in 20h35m** and produced the number above. Screen tier (1 fold, 500k rows) and the kill is still safe, because §E1 says single-fold `rho_partial` *inflates* and this one is already negative. Caveat: 500k rows vs the GBDT's ~5.2M — §I17's gate prices that at ~+0.002 rho against the +0.0031 needed, and more data raises `r` too. Falsifier if ever revisited on GPU: **does `rho_B` reach 0.6747 at `r ≤ 0.9954` on fold 4?** ⚠ Fold-4 scale throughout — NOT comparable to the pooled entry-criterion table above | **e0913** (job 24152700), `IDEAS.md` §I22 RESULT |

> **NOT graveyard — the one live lead (`IDEAS.md` §I28):** a **GRU over a frozen MOMENT-1 TS-foundation
> model's per-patch tokens (e0915)** is the FIRST candidate with *positive* excess vs the FULL champion
> (rho_partial **+0.0060** on fold 4) — external temporal priors inject genuinely new decorrelated signal.
> Small, single-fold, half-subsumed by the usercv GRU (~−0.00003 RMSLE). **5-fold confirm running (array
> 24145189).** Do NOT blend until the pooled LOFO-vs-champion number confirms.

*Added 2026-08-23 — research round 2, the tabular-foundation-model and pseudo-label directions.
Full derivations in `IDEAS.md` §I17/§I18. Both closed with a full 5-fold confirm (e0900–e0907),
not a screen.*

| idea | result | evidence |
|---|---|---|
| **Tabular foundation models (TabPFN-2.5/3, TabICLv2, TabDPT) as a model or blend member** | **refuted by the scale-penalty gate, no install needed.** Training the existing LightGBM on the *input regime a TFM would get* and scoring on the full population: a **50k-row context costs +0.01718 RMSLE** (TabPFN-2.5), 100k **+0.00887** (TabPFN-3), 1M×top-200 **+0.00073**; the **feature cut to top-200 costs +0.00020** (nil). Row-context is the binding limit, not feature count. Every arm is a weaker near-twin (r 0.987–0.999) — §1c's worthless quadrant — so no TFM can clear §1f's `rho_partial ≥ 0.04`. `min_data_in_leaf 20` at 100k is *worse* (+0.01141), so the penalty is not a fixable under-fit. Reproduces BeyondArena (2606.30410) + "Closer Look at TabPFN v2" (2502.17361) on our data. **Licence note for the write-up:** TabPFN weights are non-commercial and do not clearly cover an Ozon-hosted contest; TabICLv2 (BSD-3) / TabDPT (Apache-2.0) are clean | e0900–e0905, `IDEAS.md` §I17 |
| **Transductive pseudo-labelling under the test-anchor covariate shift** | **refuted, nil.** Adding the validation-anchor rows with the model's own log-prediction as a soft label (weight 1.0 / 0.3), refitting: **+0.00026 / +0.00009, r_vs_ref 0.9993.** The model already sits at the conditional mean of its own bins (§1q), so a self-consistent label reinforces what it predicts and adds nothing. Kaggle precedent (Santander #1, Instant-Gratification) paid only because it injected *test-set feature/covariance* the train set lacked — not our case (same users, same window). Distinct from the killed "train on contaminated recent anchors" (+0.0019, real future labels); this is label-free and still nil | e0906–e0907, `IDEAS.md` §I18 |
| **The scale-penalty gate as a method** | **keep the technique.** Measuring what a proposed model's *input constraint* (row/feature budget) costs the incumbent, before installing the new model, converted "TabPFN is outside our regime" (an assertion carried for 6 months) into "the 50k-context handicap is +0.0172" (a number that closes the item). Any future "should we try model X that only ingests N rows / K features" question is answered this way first, on the existing model, for one CPU array and zero installs | `src/run_gate.py`, e0900–e0905 |
| **Affine-invariant (within-day Pearson) loss on the GRU — IDEAS.md §I2/§I19** | **refuted, and it closes the loss-geometry axis from a 4th direction.** `(1−λ)·MSE + λ·(1−corr)`, corr centred within each calendar day (§1r). Every λ∈{.3,.5,.7} is *worse* on RMSLE (monotone: +0.00032/+0.00030/+0.00057, 0–2/5 folds) and `rho_partial` vs the 9-member blend is ≈0/negative (best e0292 +0.00027 vs the 0.02383 bar); the arms correlate *more* with the family than a plain GRU (r 0.9983–0.9986 vs 0.99827). Port gate passed: λ=0 = e0101 byte-for-byte. The GRU at 12 epochs is already at its rho optimum; DRW-Crypto's corr-loss win (0.6·MSE+0.4·Pearson) does not transfer. **⚠ Trap: rho_partial was +0.0256–0.0268 vs the single control GRU, +0.0003 vs the blend — §E1's ~90× inflation, now on the frozen folds not just the 15k screen. Always run `admissibility.py` against the family, never one member.** I2/I3 now closed, not parked | e0290–e0293, `IDEAS.md` §I19 |
| **Feature neutralisation (Numerai-style orthogonalisation of the prediction against features)** | **refuted, and it closes the post-processing axis.** The one NON-monotone post-processing lever: `M' = M − p·N(N⁺M)`, N = drift features (`_total$`/`^tenure`/`_365$`, 117 cols). For every p>0 rho FALLS on every fold; the shift-protection signature (fold 4 rising as early folds pay) never appears — **fold 4 degrades in lockstep with fold 0** (0.67306→0.61304 at p=0.7). LOFO picks p=0 (Δ +0.00000); the random-column control moves ≈0, so the loss is signal-removal not artefact. Numerai's +20% Sharpe (2303.16117) is cross-era variance reduction that a single-anchor score cannot bank. **Affine (§1b) + monotone (§1q) + segment (§1t) + feature-orthogonalisation (this) all nil → post-processing is closed.** Only caveat: the real 2026-02-13 anchor is more shifted and unobservable, but the flat fold-0→4 gradient argues it would not flip there | e0294, `src/run_neutralize.py`, `IDEAS.md` §I21 |

*Added 2026-08-22 — the "is any feature left" bound, and a screening error worth more than the arms.*

| idea | result | evidence |
|---|---|---|
| ⚠ **Screening a candidate on POOLED folds without removing per-fold means** | **manufactured a 100×-kill-bar result out of nothing.** Seven exotic raw-panel combinations scored `corr +0.1522` / `incR2 0.02316` against the blend residual when pooled across 5 folds — the largest "signal" in project history. **Within fold the same feature scores −0.0006** and all seven land ≤ 0.00012, sign-flipping across folds. A fold-level *mean* difference masquerades as user-level signal. **Always screen within fold**; `scripts/residual_screen.py` takes `--fold` for this | e0248 |
| **Any further hand-crafted combination of the existing features** | **bounded, not merely untested.** A second GBDT trained directly on the first's residual (same features, pipeline params, early stopping) stops at **best_iteration = 4**, scores `corr +0.0018` with the true test residual, and is **+0.00025 worse than predicting zero**. Boosting *is* an automated search over feature combinations — if it cannot fit the residual out of sample, no hand-designed combination of the same inputs can | e0248 |
| **"Is there signal in the raw daily panel the 665 features miss?"** | **YES — and it is already spent.** The GRU on 13 raw channels with zero engineered features has `incR2 = 0.00153` over e0049, 8× the kill bar and the largest ever screened. But LOFO-blending it gives **−0.00204**, which is exactly the gbdt+seq blend the champion already banks. The raw-data question is answered *affirmatively* and is *already in the submission* | e0248, e0120 |


*Added 2026-08-22 — the FEATURES.md tabular candidates, confirmed with a noise-column arm.*

| idea | result | evidence |
|---|---|---|
| **`age_bucket_gmv_share_3`, `cart_backlog_7`, `cohort_rel_buy_rate90`** (the three confirm-eligible FEATURES.md candidates) | **all three lose to a random number.** Full 5-fold XGBoost arms vs matched base e0210 (1.76588): noise column **+0.00006**, `cart_backlog_7` +0.00007, `cohort_rel_buy_rate90` +0.00008, `age_bucket_gmv_share_3` +0.00011. **The i.i.d. noise control has the best cv_mean of the four**; total 5-arm spread 0.00011 = 1.27σ. All positive (worse than omitting them), none ≥3/5 folds against the control. `kill` | e0211, e0212, e0213 vs **e0214** |
| **Ranking candidates by a screen that sits at ~1× its own noise control** | **the ordering carried zero information.** The screen ranked `age_bucket_gmv_share_3` first (+0.00074/+0.00112 across two anchors); on the confirm it is the **worst** of the three. A screen whose best candidate ties its own random-column control can say "nothing here is huge" — it cannot rank what is left. **Either build a screen with real separation, or confirm everything and skip the ranking.** Second instrument to fail this way after the 36× causal proxy | e0189 vs e0211–e0214 |
| **Including a noise column as a fifth arm** | **the opposite of a dead end — do this every time.** It converted four indistinguishable numbers into an unambiguous kill at a glance, with no appeal to σ_noise, fold counts, or seed replicates. Costs one extra arm in an array that was already running | e0214 |

*Added 2026-08-22 — the external-research track. Full derivations in `IDEAS.md`.*

| idea | result | evidence |
|---|---|---|
| **Recovering latent item/price structure from `gmv` values** (treat `gmv` as a discrete symbol, not a scalar) | **refuted for ~0 cost: `gmv` is anonymised off currency units.** On 3.18M single-order days (where `gmv` IS one order's price), 64,864 distinct 2-dp values: **0.01 % are exact integers**, **0.02 %** end .00/.50/.99, and exactly **1 value of 64,864** is a genuine atom. The top repeats (6.40, 6.62, 6.57, 6.36 …) are a smooth band with near-identical counts — a continuous density binned at 2 dp, not a price list. The lone atom is **0.03** (7,222 rows, 555x its neighbours, 0.227 % of single-order days), a floor/sentinel rather than a product; below what the ±0.001 feature screen can resolve | `IDEAS.md` §I16 |
| **Local-outlier tests whose baseline can absorb the point being tested** | **returned a FALSE NULL on the above, and I reported it before catching it.** `median_filter(counts, size=51, mode="nearest")` replicates the edge value to pad; the 0.03 spike is element 0, so 26 of 51 window slots were filled with the spike itself, the median became the spike, and a **555x** anomaly scored **1.0x**. Second instance this session of a reference statistic silently swallowing the effect (the first: LightGBM's `K/(K-1)` multiclass hessian). **Compute any neighbourhood baseline EXCLUDING self, one-sided at the edges** | `IDEAS.md` §I16 |
| **Routing a better MAGNITUDE model to buyers (any two-stage / gated architecture)** | **the information is real and the gate does not exist.** e0221's `rho_partial` on the magnitude term is **+0.07621** — 6x the best value ever recorded on the overall target (0.01269) and 3.2x §1f's bar — and BTYD's is +0.06804. LOFO stacks convert both to **+0.00011 / −0.00009 against the no-op control**. Two measured reasons: within-buyer variance is only **17.7%** of Var(L), capping e0221 at **−0.00124 with a PERFECT gate**; and the best available gate correlates **0.593** with buyer status. An oracle gate shows e0221 is worth −0.208 on top of an oracle-split control, so the signal is genuine. **General rule: a term defined by conditioning on the outcome cannot be exploited by routing, because the conditioning variable is what you are predicting.** Same mechanism as §1b's classifier stack (+0.00007) | `IDEAS.md` §I15 |
| **Judging a candidate only against the OVERALL target** | **it hides term-level structure completely.** e0221 scores `rho_partial` −0.00249 overall and **+0.07621** on the magnitude term; BTYD +0.01017 and +0.06804. Both were killed on the overall number alone. Decomposing is one line and should be standard for any candidate with a different output geometry — even though, per the row above, the decomposition did NOT produce an exploitable gain here | `IDEAS.md` §I15 |
| **"The families are distinct"** | **13 of 14 OOF files agree to <= 0.004 on EVERY term** — overall rho, `corr(Z,M)` and magnitude alike (LightGBM, AutoGluon, GRU, TCN, tuned-seq, Bayes-cov). §1c said they recover the same signal by different routes; the sharper statement is that they recover it in the same PROPORTIONS. Only e0221 (best magnitude, worst classification) and e0170 BTYD sit outside the pack | `IDEAS.md` §I15 |
| **Comparing models of DIFFERENT output parametrisation on raw CV RMSLE** | **inflated one verdict 35x, in this session.** e0221 (HL-Gauss, binned readout) scores **+0.04382** vs its matched L2 control on raw CV and **+0.00124** after the optimal per-fold affine map — the map §1b already applies to every submission. 97% of the "loss" was level and spread the leaderboard never charges for. The per-fold pattern also REVERSES: raw says 5/5 losses, calibrated says 3/5, with the two most test-like folds favouring it. **For L2-family models the calibration gain is a near-constant ~0.0019 that cancels in a paired delta — which is why this never bit before — but it does NOT cancel across output geometries.** Same class of error as §1e's +0.5626 functional gap. **Rule: any candidate whose OUTPUT TRANSFORM differs from the parent must be compared after affine calibration, not on raw CV** (`src/admissibility.opt_affine_rmsle`) | `IDEAS.md` §I1 |
| **HL-Gauss / histogram loss as a blend member** | **rho_partial −0.00249, i.e. negative excess correlation.** Worth nothing at any weight regardless of the accuracy question — `e = rho_B − r*rho_M = −0.00019`. §1f: a candidate's entire blend value is its partial correlation with the truth controlling for the family, and a negative excess is worth zero, not a little | `e0221`, `src/admissibility.py` |
| **Cross-user / peer (k-NN) features** — `BACKLOG` Band D **e0030**, open since the start, costed 5c | **refuted for 0.1c, and the univariate number is the whole story.** k=50 neighbours in a 9-dim causal behaviour space; every column added alone is NEGATIVE at both anchors (`peer_gmv30` −0.00311/−0.00087, `peer_rel` −0.00338/−0.00410, `peer_buyrate` −0.00143/−0.00087, `peer_dist` −0.00092/−0.00300) against a noise control of −0.00186/+0.00064. **Mechanism:** the peer level correlates +0.570 with the target while the user's OWN last-30d GMV correlates +0.557 (`DATA.md` §7.1) — neighbours are a noisy copy of the user. And `peer_rel`, the one quantity per-user features cannot express (local rather than population rank), has univariate corr **+0.026 / +0.019**. Users resemble each other in LEVEL and almost nothing else, which is `DATA.md` §5.4's trait-reliability result (0.09 vs 0.65) reached from another direction. **e0031 cluster-id inherits this kill** — same similarity metric, same absent structure | `IDEAS.md` §I12 |
| **Inverse-variance (BIV) sample weighting from measured per-user label noise** — 2107.04497 | **indistinguishable from a random reweighting.** `w = 1/(sigma^2+c)` with sigma^2 = within-user variance of log1p(30d GMV) over the 6 past windows (causal): **+0.00437 / +0.00301** at A2/A1 — but the **shuffled-weight control** (identical weight marginal, permuted across users) scores **+0.00355 / +0.00387**, i.e. as much or more. The premise is *true* — `corr(historical sd(L), |residual|) = +0.30 / +0.32` — the intervention still buys nothing. The opposite direction (`w = sigma^2+c`) gives +0.00226 / −0.00071 | `IDEAS.md` §I10 |
| **Nonlinear (monotone) recalibration of the prediction — isotonic or cubic `g(M) ~ E[L|M]`** | **dead, and the diagnostic is cleaner than the result.** rho is invariant to *affine* transforms but NOT to monotone ones, so this was a genuinely free lever the project had never pulled. Leave-one-fold-out isotonic **−0.00022**, cubic **−0.00014** on `oof/e0049.parquet`. Reason: `E[L|M] − M` is a **constant −0.017 across all 20 prediction bins** — the miscalibration is a pure shift, which is affine and which §1b's calibration already removes. Nothing is left for a curve to fit | `IDEAS.md` §I10 |
| **Screening MODEL-level changes on the 15k local harness at all** | **the band is ~±0.004 rho, ~50x the frozen-fold sigma_noise of 0.00009.** Three separate no-op controls now agree: a reseeded 6-model bag (+0.0116/+0.0024), the same model early-stopped on a different metric (+0.0049/−0.0006), and a *shuffled* weight vector (+0.0036/+0.0039). `FEATURES.md` calibrated this harness at ±0.001 for adding a COLUMN; a change to the model, the loss or the sample weights rides a band four times wider. **Feature screens locally, model changes on the frozen folds** | `IDEAS.md` §I10, §E1-RESULTS |
| **Discrete-time hazard / survival supervision for P(buy in 30d)** — `PAPERS.md` 6.2, the last unbuilt `P1` in the project's own literature review | **worse, and it replicates.** Survival-stacked (2107.13480) LightGBM, 6 intervals x 5 days, vs the installed one-label-per-user binary classifier on identical features and split: `corr(Z, p)` **0.60260 vs 0.60799 at A2 (−0.00539)** and **0.55646 vs 0.56176 at A1 (−0.00530)**. Two reasons: the timing supervision answers a question the 30-day SUM integrates away (same mechanism as §3b's dense-supervision null, +0.00012), and `1 − prod(1 − h_j)` compounds six separately-estimated hazards whose per-row event rate is only 0.14. Lands in §1c's decorrelated-but-weaker quadrant (r = 0.96–0.97) | `IDEAS.md` §I9 |
| **`user_id` as a registration-order / signup-cohort covariate** | **no effect, and now closed.** Δrho **−0.00100 at A2, +0.00141 at A1** — opposite signs, both inside the band spanned by the screen's own no-op controls. `user_id` runs 2…918,481 over 250k users and was never in `FEATURES.md`, so it was worth the one fit it cost. Either ids are not assigned in registration order, or `tenure_days` already carries what they encode | `IDEAS.md` §I5, §E1-RESULTS |
| **Reporting `rho_partial` from the 15k local screen** | **the screen inflates it by roughly an order of magnitude.** A *reseeded 6-model bag of the identical L2 model* scores `rho_partial` **+0.086**; adding *one i.i.d. junk column* scores **+0.069** — against this project's best-ever real value of **0.0127** and §1f's 0.02383 bar. Same failure mode as the 36× causal-proxy inflation. **Any screen quoting `rho_partial` must ship the reseeded-bag control or the number is meaningless** | `IDEAS.md` §E1-RESULTS |
| **Early-stopping and scoring on the same held-out half, when comparing models of different evaluation-noise profiles** | **manufactured the entire first result.** On a 2-way split the CE arm read **+0.00174**; on a 3-way split (train / early-stop / score, disjoint) the same arm reads **−0.01240**. §1j priced this bias at ~`sigma*sqrt(2 ln N)` for hyperparameter search; it applies just as hard to a model-class comparison, because a K-class model's validation curve is noisier than a regressor's and therefore gains more from being allowed to pick its own maximum | `IDEAS.md` §E1-RESULTS |
| **Uniform bins over a zero-inflated target (any discretisation method, not just HL-Gauss)** | **produces an EMPTY class that silently corrupts every other class.** `DATA.md` §6.1's measured gap between the zero atom and the L≈4.2 bulk means a uniform grid reliably lands a bin with no training mass. Its softmax gradient is `p` with no counterweight, its raw score runs away (**−34 after 3 rounds**), and the shared normaliser then shifts every other class by ~0.033 in raw score. **It presents as a port bug, not a binning bug.** `src/hlgauss.make_bins` merges any bin under 0.2% occupancy | `IDEAS.md` §E1-RESULTS |
| **LightGBM's multiclass Newton step is `hess = K/(K-1)*p(1-p)`** | not a kill — a **fact that costs hours if unknown**. It is neither the textbook `p(1-p)` nor XGBoost's `2*p(1-p)`, and it is not in the documentation. A wrong coefficient is a pure step-size error that still trains and still scores. Also: the built-in `multiclass` **boosts from the class prior** while a custom objective starts at 0. Established by bisection, verified to `max\|d raw\| = 0.0` for K ∈ {3,5,8,16}; `src/hlgauss.port_exact_check` runs the gate inside every run | `IDEAS.md` §E1-RESULTS |


*Added 2026-08-20 — the tsfresh port. Full derivation in EXPERIMENTS.md §1o, per-family table in FEATURES.md.*

| idea | result | evidence |
|---|---|---|
| **tsfresh — 60 hand-vectorised statistics (c3, time-reversal asymmetry, ARCH, Haar wavelet energy, level strikes, trend t-stat, lag-1 autocorr) × {gmv,ord} × {30,90,365}** | **nil at both tiers.** Confirm: cv **1.76545 vs e0049's 1.76551 = Δ −0.00006 (0.6σ), 2/5 folds** → `no effect` by §3.4, and 60 columns for 0.00006 is the within-noise accumulation §3.4 forbids. Screen: bundle +0.00019 (A1) / −0.00074 (A2) against a +0.00014 / −0.00097 noise control, and **no statistic family positive at both anchors** (decomposed into 11 families at *both*, since a 60-col bundle is penalised ~60× harder than the 1-col control on that budget). The shape-statistic vocabulary is already spanned by `sbc`+`tsfeat` | e0192, e0191, §1o |
| **The "monotone fold trend" inside that null** | **my own near-miss, corrected before it changed the verdict.** e0191's per-fold deltas are perfectly ordered in fold index (+0.00027, +0.00007, +0.00007, −0.00028, −0.00041; spearman −0.90 vs train rows, P = 1/120), which reads as §1l's fold-index interaction and would argue `park`. **It dies on the right denominator:** σ_noise = 0.00009 is the sd of the *5-fold mean*; a *single* fold's sd is 3× larger (`[0.00039, 0.00028, 0.00033, 0.00022, 0.00026]` from the e0001 seed replicates), so the deltas are `[0.70, 0.25, 0.21, −1.28, −1.59]σ` — **not one fold reaches 2σ**. A 1-in-120 ordering of five noise draws, and we look at many orderings. **Use the per-fold sd for per-fold deltas; σ_noise is ~3× too small** | §1o |
| **Porting a feature library by hand without checking it against the library** | **3 of 7 statistics were WRONG on the first pass, and CV cannot detect this** — a wrong statistic trains, scores, and returns a plausible null. `c3` had a bispectrum-style triple product (rel err **1e15**); `arch7` used a segment-wise Pearson where tsfresh uses a global mean/var with an `(n−lag)·var` denominator (**0.41**); `trendt` omitted the intercept from the residual, deflating the t-stat ~**12%**. After fixes, max rel err ≤ 6e-08 against `feature_calculators`. **A "library-style" feature that is not the library's statistic is an untested new feature wearing a validated name.** Applies to catch22/`tsfel`/any paper port | §1o |
| **`lempel_ziv_complexity` as a feature** | **implemented, validated (6e-08), deliberately not emitted — a cost kill, not an evidence kill.** ~**32 s per 250k users per window** vs ~0.3 s for every other statistic (unvectorisable per-user prefix scan, single-threaded, ~2 h per fold build). **It killed the first cluster attempt outright** — job 23868236 spent 55 min of a 1 h cap without finishing one fold's features. Dropping it: 66 → 60 features, 6.0 s → 0.8 s. Also the worst screen family (−0.00218 at A2). Code kept in `src/features.py`; re-enable only with a vectorised LZ | §1o, job 23868236 |

*Added 2026-08-20 — the FEATURES_CAUSAL.md causal-feature candidates, and month-normalisation.*

| idea | result | evidence |
|---|---|---|
| **The LightGBM user-split proxy as a screening instrument for GRU features** | **overstates by ~36×, measured.** It screened `ds_order` at **−0.00360**; the real GRU confirm gives **−0.00010**. Mechanism: the proxy is a tabular model at n=15k, where a long-memory scalar is worth a great deal; the GRU accumulates the same quantity in its hidden state for free. FEATURES_CAUSAL.md built its whole case on the asymmetry "recency-of-order informative (−0.00360), recency-of-activity/cart nil" — that asymmetry exists **only on the proxy**. On the GRU all three are nil. **Any future proxy screen needs recalibrating against a confirm before its numbers are believed** | e0193, e0195 |
| **`ds_order` / `cart_backlog` as GRU channels** | **both −0.00010 = 0.5σ, 3/5 folds, paired t = −1.02 / −0.74 over 15 matched cells.** Retained as `keep` **on user instruction, not on evidence** (see the rows' notes) — recorded here so the *measurement* is not re-litigated. Two structurally unrelated channels returning the identical delta is the signature of a noise distribution centred at zero. AUC flat (+0.00008 / +0.00013), and per §1b AUC is the gate on rho | e0193, e0194 |
| **Month-normalisation (trailing-30d per-day standardisation stats)** | **−0.00005, 3/5 folds, AUC flat.** The GRU sees the whole 409-day sequence, so a slowly-varying level is already learnable; re-centring per day re-parametrises what it had. Measured on `extra`, whose CV↔LB link is known broken (e0142), so this is a null on the *mechanism* | e0196, e0197 |
| **Comparing a user-split run against a logged baseline from another session** | **cross-session drift is +0.00027 to +0.00046 for configs with ZERO changes** (`extra` 1.74280→1.74307; `full`/e0141 1.74341→1.74387). That exceeds every effect size in this section. **Every user-split candidate must ship with a matched same-session control** — the deltas above are all measured that way | e0195, e0197 |

*Added 2026-08-17/18 — tuning and the architecture search. Derivations in EXPERIMENTS.md §1i, §1j, §1k, §1l.*

| idea | result | evidence |
|---|---|---|
| **Tuning the GRU on the user-split CV with an early-stopping objective** | **actively harmful: rho 0.70311 -> 0.70210**, cancelling the −0.00099 its CV claimed. Cause is neither the protocol nor the stopping point (both tested, both refuted) but **early-stopping selection bias**: min-of-N noisy evaluations is biased by ~`sigma*sqrt(2 ln N)`, and the winner got 100 evaluations vs the baseline's 21. On `mean(last 10)` the gap is 0.0006, not 0.00099. **Never score a trial on the minimum of a variable-length run** | e0145, `tuned_confirm`, `tuned_tmax` |
| ~~"Better-regularised configs train longer, so train for thousands of epochs"~~ | **an artefact of the above, refuted twice.** (a) annealing the cosine over the epochs actually trained gives back the untuned score exactly (1.74356 vs 1.74351); (b) the unbiased frozen-fold search, with `epochs` free over 8–200 log-scaled, chose **8–40, clustered at 13**. Long training was buying evaluation draws, not quality | §1j, §1k |
| **Selecting hyperparameters on a single fold** | **31 trials against fold 4 produced late-fold hyperparameters.** All 8 leading configs beat e0101 on fold 4; exactly one beat it on the 5-fold mean (−0.00041, 2/5 folds, fails §3.4). corr(screen, confirm) = +0.60. **Screen on one fold if you must; SELECT on the confirm.** ⚠ The cause is *not* noise-fitting — see §1l: fold 3 was never selected on and 7/8 configs beat e0101 there by 5–8σ. It is a real hyperparameter × fold-index interaction | e0180–e0187 |
| **LSTM, vanilla RNN, TCN, and three transformer position schemes as seq backbones** | **GRU wins all of them** on fold 4: gru 1.73143, cnngru 1.73337, xformer_rope 1.73428, lstm 1.73430, xformer_alibi 1.73784, tcn 1.75180, rnn 1.75415, xformer_learned 1.75671. Two results worth keeping: the **vanilla RNN is 0.023 behind the GRU**, so the gating is load-bearing; and **learned absolute positional embeddings are the worst of all eight**, 0.025 behind ALiBi/RoPE — the absolute-time extrapolation risk asserted since the first TCN, finally measured | `reports/eda/seqarch_fold4.json` |
| **Re-weighting the blend** | **worth −0.00004.** Optimal non-negative weights are 0.20/0.38/0.42 against the inherited 0.25/0.25/0.50; rho 0.70375 → 0.70377. The inherited weights were already at the optimum, and this also kills gated/per-segment weighting (measured separately at −0.00001, in-sample upper bound) | e0162 |
| **Swapping a better member into the blend** | **e0090 is genuinely better (+0.00038 rho, the project's largest member-level gain) and moves the blend by +0.00001**, because it enters at weight 0.20 and correlates 0.998 with the member it replaces. Member gains are diluted by weight and then absorbed by the recalibration | e0161, e0162 |

| **Multi-anchor TTA (predict at A-0..A-29, subtract observed spend, weight toward the final day)** | **killed, and badly.** Not one of **666** weighting/smoothing/combiner schemes beats k=0; LOFO scheme selection picks `k0` on 5/5 folds. The proposal as stated costs **+0.30939** (rho 0.66162 → 0.47243). Two causes: the subtraction mixes functionals (`expm1(model)` ≈ E[log1p y] vs an arithmetic realised sum — mean g_29 = 93.1 vs mean p_29 = 38.0, 42.9% of users clip to zero), and **anchor shift is not label-preserving**, so it is not augmentation at all. Pure averaging without subtraction also loses (+0.00015) | e0188, §1n |
| **Recombining every OOF we own into a better blend** | **honest ceiling −0.00004 CV.** 21 predictors, greedy equal-weight: in-sample −0.00009, but selection-on-4-folds-scored-on-the-5th gives **−0.00004**, and unconstrained OLS over all 21 **loses** (+0.00004 LOFO). Greedy rediscovers e0120 — 7 of its 9 members plus e0180/e0186/e0110. e0180 is picked first in all 5 folds and still buys nothing | §1m, `src/recombine_oof.py` |
| **The 8 tuned seq configs as blend members** | **all worth zero.** Best `rho_partial` 0.00683 (e0186) against §1f's 0.02383 bar; best measurable move is `BASE + e0180 + e0186` at **−0.00006 CV / +0.00003 rho ≈ −0.00007 LB**, the same size as adding one more seed of e0101. Replacing the whole seq half with the tuned configs is **+0.00046 worse** — they correlate more tightly with each other than the originals do, so they blend worse even where they score better | §1l, `src/admissibility.py` |

*Added 2026-08-17 — BAYES_EXP B1 built and measured. Full derivation in EXPERIMENTS.md §1h.*

| idea | result | evidence |
|---|---|---|
| **B1 `hier_cov` — 28 covariates on the BTYD priors** | **A BETTER MODEL AND A WORSE BLEND MEMBER.** rho 0.62709 → **0.63452** (+0.0074, up on 5/5 folds), calibrated CV\* 1.83241 → **1.81822** (−0.0142) — the covariates genuinely work. But `r` vs the family rose 0.9427 → 0.9564, so excess `e` fell 65% and **`rho_partial` fell 0.01017 → 0.00400** (43% → 17% of the bar). Blend gain **+0.00000**; adding it is faintly *harmful* (family+B1 = 1.76290 vs family alone 1.76274) and it dilutes B0 when both are in. **Judge on rho/CV\*, never raw RMSLE** — raw got 0.043 worse purely on level/spread, which §1b says is free | `e0180` |
| **...and the mechanism, in one regression** | `log B1 = −0.715 + 0.560·log B0 + 0.642·log family`, **R² = 0.938**. B1 is a 53/47 mixture of B0 and the models we already have. **The covariates did not add a direction; they rotated B0 toward the family.** Keep this line — it is the sharpest statement of why nothing blends here | `e0180` |
| ~~**"marginalising the latents removes §4.2's need for a variance floor"**~~ | **MY ERROR, and it broke the fit.** Unconstrained, `r` climbs 114 → 144 and is still rising at 2000 iterations on every fold (`a+b` → 194…484, `p_gg` → 128…162, against B0's 1.1 / 2.2 / 1.1). The optimum is on the boundary at `r → inf`: the rate prior collapses to a point mass and the per-user random effect is optimised out of existence. `1/r` and `1/(a+b)` play sigma's role — **the hazard was renamed, not removed** | `e0180d` |
| **B1 with §4.2's variance floor** (`--which loc`) | **The floor fixes the fit and changes nothing — which is the sharper finding.** Parameters go sane (`r` 114 → 0.53, `a+b` 389 → 2.0, `p_gg` 162 → 1.11, a **200× difference**) and the predictions correlate at **0.98497**: rho 0.63435 vs 0.63452, `r` vs family 0.95641 vs 0.95638, blend +0.00000 both. So the dispersion collapse was a real optimisation pathology but **not** the cause of the blend failure. **Once 28 covariates sit on the locations, the BTYD process is just a link function** — the covariate regression decides the prediction, and covariate regressions here all land in the family's span. 19.4 min for 5 folds; use this variant if ever revisited | `e0183` |
| **A NaN gradient that reported `success=True`** | `gammaln(p*n)` at `n=0` for masked users: `digamma(0)` is NaN, reverse-mode AD multiplies it by the mask's zero cotangent, the **objective stays finite**, and L-BFGS-B stops at iteration 1 claiming success. **Check gradients for finiteness, not just the objective.** Also: covariates reached \|z\| = 249 (`ord_per_buyday`) — ratio features here are violently heavy-tailed and need winsorising; a `Normal(0, 0.5)` prior on beta would not have caught it | `src/bayes_model.py` |

*Added 2026-08-16 — the `BAYES_EXP.md` review. Full derivation in EXPERIMENTS.md §1f/§1g.*

| idea | result | evidence |
|---|---|---|
| **Posterior-uncertainty columns as features** (`sd(log1p Y)`, `P(Y=0)`) — `BAYES_EXP` §10's headline claim | **+0.00001 against the no-op control.** The column is real — `corr(sd_log1p, |blend residual|) = +0.3187`, and a GBDT genuinely cannot build it — but the *argument* is wrong: **under squared error the Bayes action is the posterior MEAN, so the posterior SD cannot move the optimal point prediction.** The shrinkage §10 wants it for is already inside `E[log1p Y]`. Uncertainty columns pay for pinball / CRPS / expectiles, never for RMSLE | `e0172` |
| **`BAYES_EXP` §5.3's calibration wrapper** (per-cohort affine + `γ·s_u`) | **−0.00001, noise.** Its own blocking rule never fires: `a_c` = 0.9932…1.0005 across the five folds. This is §2's `k*=1.000` again — calibration pays at the TEST anchor, not on the folds, and shifting the submission log-mean is already standing protocol | `e0173` |
| ~~**"find a blend partner at r ≤ 0.9885"**~~ | **RETIRED — `r` alone was never the currency.** Exactly: `R² = rho_M² + (1−rho_M²)·rho_partial²`. A candidate's whole value is `corr(L, B | M)`. At fixed excess, a MORE correlated candidate is worth MORE (the `1/(1−r²)` divisor shrinks), so decorrelation cuts both ways and we only ever counted one side. Validated to ≤0.0001 against measured gains on all ten candidates | `src/admissibility.py`, `e0174` |
| **The `0.0005` blend-gain threshold itself** | **Mis-set: no single member has ever cleared it, ours included.** Bar is `rho_partial ≥ 0.02383`; e0064 achieves 0.01269, e0049 0.01255, BTYD 0.01017, the seq members 0.0027–0.0065. The 9-member blend's −0.00176 is nine sub-threshold contributions accumulating. **Judge a new family against `rho_partial ≈ 0.013` (the best ever), not against 0.0005** | `e0174` |
| **`BAYES_EXP` §6's fold design** — DO NOT BUILD | **Guard-zone contaminated, and it nominates the worst fold as primary.** F2/F3/F4 all have target windows inside `[2025-11-16, 2026-02-13]`, where all 250k users are active by construction; `DATA.md` §4.3 prices the bias at **+0.041**, 80× the effect being chased. §6 says "F4 is the primary validation fold". Only F1 (2025-10-16) is clean — and it is already frozen fold 4. Rule 3: use `data/folds.parquet` | `DATA.md` §4.3, `folds.py` |
| **`BAYES_EXP` §3.2's `s_next` seasonal extrapolation** — DO NOT BUILD | **Already falsified twice, once with a submission spent on it.** Estimating the 2026-02-14→03-15 multiplier from 2025-02-14→03-15 is exactly what **e0142** did: **1.6785, worst model since e0001**, a −0.410 log-mean shift. Independently, Band A+ retracted the year-lag mechanism on CV (`corr(spike, residual) = +0.0001`; every k > 1 worse; best k = **1.00**) | e0142, `reports/spiker_residual.log` |

*Added 2026-08-16 — BTYD, the last untried backlog entry. Full derivation in EXPERIMENTS.md §1e.*

| idea | result | evidence |
|---|---|---|
| **BTYD (BG/NBD + Gamma-Gamma) as a blend member** — e0033 | **KILL, and it replicates Ridge almost exactly.** CV **1.83569** (beats geo3 by −0.0929, loses to e0049 by **+0.0702**); `corr` vs gbdt **0.9423**, vs e0101 0.9404; fitted LOFO blend gain **−0.00006** against the pre-registered −0.00050. Ridge was +0.07255 / 0.9433 / 0.00000. Two model classes with nothing in common land in the same quadrant on all three axes — **the third and cleanest confirmation that decorrelation must come at comparable quality**. Loses 0/5 folds and every decile of the blend prediction (+0.059…+0.101), so there is no segment to route to it either | `e0170`, `src/btyd_blend.py` |
| **BTYD latent columns (`P(alive)`, `E[X(30)]`, `E[M]`) on the blend** — e0033's actual staging | **−0.00008 against the no-op control** (`p_alive` alone: −0.00000). Cause found: **the RFM inputs were already in the 665-feature set** — `buy_days_total` = `x+1`, `recency_order_days` = `T − t_x`, `gmv_total/buy_days_total` = `m_x`. e0033's own gate ("add raw `(x,t_x,T)` first; if those do not move CV, do not fit BG/NBD") was answered back at e0001 and nobody checked. **Check whether a proposed feature is an algebraic rearrangement of existing ones before building the model that generates it** | `e0171` |
| ~~"a generative model gives a new capability: `E[log1p y]` by simulation"~~ | **Half wrong, and the wrong half is the interesting one.** It is not a new capability *relative to the family* — L2-on-`log1p` already targets `E[log1p y]` directly, so every model here had it for free. It IS an enormous correction to **BTYD's own** output: `log1p(E[y])` scores **2.39829** against `E[log1p y]`'s **1.83569**, a **+0.5626** gap from the estimand alone. Worse than the optimal global constant and worse than `sample_submit`. **Keep this for the write-up — it is the largest single effect in the project and it is a reporting error, not a model failure** | `e0170` vs `e0170ey` |
| **BG/NBD's dropout component on this panel** | **Degenerate, as `BTYD.md` §5.1 predicted and worse.** Fitted `a` = 0.012–0.020 at all five anchors, mean `P(alive)` 0.978–0.983, `P(alive) > 0.99` for 74–85% of users. `a < 1` everywhere means the closed-form `E[X(t)]` **diverges and cannot be used** — only the Monte-Carlo route works. The population rule selects on end-of-window activity, so churned users were excluded by construction and the model collapses to an NBD. Gamma-Gamma is misspecified too: `corr(log x, log m_x)` = **+0.23** at every anchor (raw-space `corr(x, m_x)` is +0.01…+0.03 — **the naive check would have passed it**) | `reports/e0170_btyd.json` |

*Added 2026-08-15 — mixup. Full derivation in EXPERIMENTS.md §1d.*

| idea | result | evidence |
|---|---|---|
| **Class-preserving mixup** (mix only where both source users agree on buy/no-buy) | **catastrophic: +0.01858**, early-stops at epochs 1–6. Masking on agreement conditions the training set: P(buy) goes 0.5577 → **0.6139**, a 10% inflated buy rate, while validation uses the true one. I proposed this as the SAFE variant of mixup; it is the broken one. **Check for conditioning bias before masking a loss on any function of the label** | `usercv_full_mixclass` |
| ~~"Interpolation fabricates targets in the empty region, so mixup will hurt"~~ | **WRONG, and naive mixup is a KEEP: −0.00065, 5/5 folds, AUC +0.00026.** The empty-region argument (0.127% of real users vs 6.3% of interpolated pairs) conflates generative plausibility with useful supervision — mixup is a regulariser over the function, not a sampler of the marginal. Mechanism confirmed by best epochs moving 13–25 → 17–41 | `usercv_full_mixnaive` |

*Added 2026-08-15 — the ten-family sweep. Full derivation in EXPERIMENTS.md §1c.*

| idea | result | evidence |
|---|---|---|
| **CatBoost / XGBoost as new families** | **+0.03202 / +0.03265 vs the GRU**, and worth **+0.00001** in a fitted blend. They ARE decorrelated (0.974 vs the gbdt family, against 0.9983 for the e0049/e0064 twins) — the decorrelation just does not pay, because they are 0.035 weaker. Also twins of EACH OTHER at 0.9983: two "structurally distant" tree algorithms are one family, not two | `usercv_catboost`, `usercv_xgboost` |
| **Ridge as a maximally-different function class** | **+0.07255**, correlation **0.9433** — by far the most decorrelated model ever built here, and it contributes **nothing** to a fitted blend. This is the cleanest demonstration that decorrelation alone is not the currency | `usercv_ridge` |
| **"Weak but decorrelated members help the blend"** | **FALSE as stated, and it was my error.** `rho_B = rho*sqrt(2/(1+r))` assumes members of EQUAL quality; the "find a partner at r <= 0.9885" target silently carried that condition. Three families at r = 0.94–0.97 moved a leave-one-fold-out blend from 1.76141 to 1.76142. **Decorrelation must be at comparable quality to pay** | EXPERIMENTS.md §1c |
| **Causal transformer in the e0141 setup** | **+0.00125, 0/5 folds** vs the GRU, AUC 0.84604 vs 0.84647. Consistent with e0102 on the frozen folds (third of three). Attention buys nothing over recurrence on this data | `usercv_full_transformer` |
| **Per-user residual target** (`L - log1p(geo3)`, add back) | **worse on both families**: CatBoost 1.77543 → **1.78992**, GRU 1.74341 → 1.74618. The reason was already measured and I missed it: `regress M on base` has **slope 0.749**, so the optimal loading on the baseline is 0.749, not 1 — an offset forces the coefficient to exactly 1, imposing a constraint the data rejects. A *shrunk* offset (0.75×base) is the defensible variant and is untested | `usercv_catboost_resid`, `usercv_full_resid` |
| **42 interpretable behavioural features** (cart-minus-order level, cart backlog stock, FIFO cart→purchase delay, conversion rate, AOV, basket size, GMV/buy-day, cart-without-order days, search↔cart ratios; local + 7/30/60/90) | **no effect: −0.00014 RMSLE (3/5 folds, inside seed noise), AUC −0.00002.** The FIFO delay measures something the project had never modelled — `to_ord <= to_cart` holds exactly, so cumulative curves nest and the wait is exact without item ids. Still nothing. Fifth independent probe of the ~0.845 AUC ceiling, now spanning 13 → 143 features | e0142-behav, `usercv_behav` |
| **Zeroing predictions for confident non-buyers** (re-tested with the CALIBRATED classifier, not the regression's implied p) | **still dead.** Only `p < 0.05` helps at all — 1,128 users of 1.07M, −0.00002. Break-even is `M < 2(1-q)*E[L|buy]` with `E[L|buy] = 4.298`: at 97% precision it pays only where our prediction is already below 0.258. And the model is near-optimal in every confidence bin (lowest bin: ideal 0.319, we predict 0.351) — **the right answer for a 90%-unlikely buyer is a small positive number, not zero** | `rho_decomp`, calibrated `e0160` |
| **e0152 — blend weights fitted on leaderboard scores** | **no validation set of any kind.** Nine free parameters on a simplex fitted to ten LB observations, over predictors correlating at 0.99. Scored 1.646697 — a tie with e0150 to the sixth decimal. More parameters, zero gain: the signature of fitting noise. Violates rule 7 and must not be a final submission | e0152 |

*Added 2026-08-14 — the endgame batch. See EXPERIMENTS.md §1b for the full derivation.*

| idea | result | evidence |
|---|---|---|
| **A dedicated `y>0` classifier to break the rho plateau** | **dead, and it closes the last lever.** The classification term holds 78.6% of Cov(L,M) and a perfect split is worth +0.271 rho (7.7x magnitude), so this was by far the largest prize left. Measured AUC: regression blend 0.84322, LightGBM-binary 0.84412, seq-GRU-BCE 0.84443, seq-GRU-BCE+feats 0.84419, 3-classifier average 0.84511 — **four model classes and two objectives inside 0.002**. ~0.845 is the DATA's ceiling, not ours | e0160, e0161, e0162 |
| **...and the AUC that IS available converts to zero** | LOFO stack of `[M, clf, M*clf]`: +0.00031 (2/5 folds). **The control is the finding** — refitting on `M` alone costs +0.00024, so the classifier's marginal contribution is **+0.00007**. Retires `rho_decomp.py`'s `d(rho)/d(AUC) ~ 1.2`, which was measured along the ORACLE path and flagged as an upper bound; the realised rate is ~0. **Methodological rule: always run the no-op control in a stacking test** | same |
| **Believing "calibration cannot help"** | **true on CV, false on the leaderboard, and it cost 0.006–0.009 per submission for the whole project.** `calibrate.py` found `k*=1.000` on OOF because the folds' level matched the model's. At the TEST anchor every model is mis-levelled by 0.14–0.41 in log space, because they are trained on 2025 anchors whose target level differs from the test window's. e0120 alone left 0.0086 | EXPERIMENTS.md §1b |
| **Variant C's calendar block (day-of-year)** | **predicted before submission, then confirmed: 1.6785, our worst model since e0001.** `moy_sin/cos` maps the test anchor (doy 44) onto Feb–Mar 2025, the lowest-GMV stretch in the series, so the model applies 2025's February level to 2026 — a −0.410 log-mean shift. The user-split CV ranked C **best**, because within the observed range the feature is interpolation. This is CAUSAL_EXP.md §4's own "calendar artefact is scored as skill" caveat, measured | e0142 |
| **Fitting blend weights on LB scores** | The unconstrained optimum over 10 scored submissions projects **1.6263** and is an artefact: ten predictors correlating at 0.99, nine free weights, ten observations, three weights negative. The non-negative version projects −0.0005, which is **inside the demonstrated error of the projection method itself** (e0150: predicted 1.64610, actual 1.64670). Do not spend submissions on projected gains below 0.0006 | EXPERIMENTS.md §9 |

*Added 2026-08-13 (evening) — the `seq` batch. σ_noise for `nn_seq` is **0.00020**, 2.2× the
GBDT's 0.00009 (e0101 seeds 0–3); judge these against that, not against 0.00009.*

| idea | result | evidence |
|---|---|---|
| **Train the sequence model longer** (12 → 30 epochs) | **worse on both backbones, and badly.** GRU 1.76458 → **1.78500** (+0.0204, 100σ, 0/5 folds); TCN 1.76775 → 1.76833. Training loss falls monotonically throughout, so the curve tells you nothing — the model overfits the ~30 M highly-overlapping supervised user-days. 12 epochs is not a lower bound to be relaxed, it is near the optimum. This also kills the boring explanation of e0100's gap to e0049 ("it was undertrained") | e0103, e0106 |
| **More sequence-model capacity** | **flat to worse.** GRU d128 → d256: 1.76458 → 1.76627 (+0.0017, 8σ, 0/5). TCN d128 → d256: 1.76775 → 1.76790 (no effect). GRU 2 → 3 layers: 1.76479 (no effect, −0.00003 vs the 4-seed mean). Capacity does not bind; the signal ceiling does | e0104, e0105, e0108 |
| **Day-of-week channels for the sequence model** | **no effect, slightly negative.** 1.76458 → 1.76509 (+0.00051, 2.5σ) and it loses **5/5** folds. Consistent with DATA.md §5.2 — day-of-week medians span only 92.5–104.8% of the overall median. The one calendar feature with no extrapolation risk is also the one with nothing to extract | e0107 |
| **Dilated TCN as the seq backbone** | Not dead, but third of three: 1.76775 vs GRU 1.76458 and transformer 1.76575. Still carries weight in the blend (its correlation to the GRU family is 0.9955, the lowest inside the seq family), so keep it as a member — just do not develop it | e0100 |
| **Tabular features as extra GRU input channels** (27 per-day window aggregates: 7/30/90/365d sums of gmv/ord/cart/srch, active/buy days, recencies, geo3) | **worse: +0.00137 (7σ), 0/5 folds.** But the control says why, and it is not "the features are bad": the SAME channels help the TCN by **−0.00165 (5/5)** and the cnngru by −0.00041. They are a substitute for long-range integration, not information — so they help architectures that lack it and dilute the one that has it. Third confirmation of the `sbcmoment`/`funnel` lesson, now in a second model family | e0110, e0113, e0112 |
| **Cross-sectional rank channels for the GRU** (rank within the day's population of gmv30/ord30/days30/recency/geo3) | **worse: +0.00105 (5σ), 0/5 folds.** This was the strongest remaining candidate — the one family a per-user sequence model provably cannot compute for itself, and a keep on the GBDT side (e0004). It still loses. With `dow` (+0.00051, 0/5) and the 27 derived channels, that is **four independent input additions, every one negative, every one losing 5/5**. e0101's 13 channels are a sharp optimum; the binding constraint is overfitting (e0106: 30 epochs = +0.0204), not missing information | e0114 |
| **CNN + GRU hybrid** (4 dilated causal conv blocks feeding a 2-layer GRU) | **worse than the GRU alone: +0.00162 (8σ), 0/5.** The conv front-end pre-summarises away detail the recurrence was using. Consistent with the row above: adding the derived channels *recovers* part of it (e0112 beats e0111 by −0.00041) precisely because the front-end degraded the raw signal | e0111, e0112 |
| **Improving a seq member to improve the blend** | **3% pass-through.** e0113 is a genuinely better TCN (−0.00165, 5/5) and swapping it in for e0100 moves the best family-weighted blend by **−0.00006** — inside σ_noise. Adding all four new members: also −0.00006. The seq family is internally correlated at 0.995–0.998, so member quality barely reaches the ensemble. Blend diversity has to come from a different *kind* of model, not a better one | e0110–e0114 |
| **Dense per-day supervision** (the seq approach's headline claim) | **worth ZERO: +0.00012 (0.6σ), wins 2/5.** Supervising every 7th day instead of every day drops 35.1M user-days to 5.2M -- exactly the GBDT's anchor grid -- and changes nothing. Targets overlap 29/30 days, so the extra positions carry almost no independent information. **Consequence: a one-target-per-forward-pass design (windowed / bidirectional encoder) starts from no handicap.** The approach's only surviving structural edge is calendar translation invariance | e0115 |
| **CV split by user_id instead of by date** | **wrong instrument, and the reason is in the task.** The public/private split is by customer, but both leaderboards cover the SAME future window and every test user is one we already hold history for -- so the train->test relationship is *same users, later date*, pure temporal extrapolation with no user-generalisation gap. CLAUDE.md §3.1 requires CV to reproduce that relationship. Measured cost of the blind spot: over a 6->30 epoch sweep the held-out-user gap moves +0.005 while the SEEN users degrade +0.0136, so a user split sees about a quarter of the overfitting | `reports/eda/seq_usersplit.json` |
| ~~**Epochs chosen by fiat**~~ -- resolved | A held-out-user slice at the fold's own anchor is a valid, FREE epoch selector: it picks 12 (the frozen-fold optimum) and prices 30 epochs at +0.0186 against the folds' +0.0204. Costs no training anchor, unlike a temporal ES split (e0017/e0020: ~4 anchors, -0.0038). **This is the one legitimate use of a user split here** | `src/seq_usersplit.py` |
| **Random left-crop augmentation for the seq model** (and any other fix for "visible-history length differs between train and test") | **Do not build it.** The premise is false in both directions: cut-off gap barely moves the seq model (**+0.00065 RMSLE per 100 days, corr +0.07**, against the GBDT's **+0.00428, corr +0.71** — the sequence model is 6.6× MORE cut-off-robust), and cropping the input to the model's own trained length makes it **worse at every gap tested** (+0.00413 … +0.01767). The model uses the long history and should be given all 409 days at the test anchor | `reports/eda/seq_transfer.json`, `src/seq_transfer.py` |
| **"Paired LB deltas are exact"** | Good shorthand, now quantified: bootstrapping a paired 50 000-user delta on our OOF gives **sd 0.00038**, so ±0.00076 at 2σ. Precise, not exact. A 0.0006 LB move is a ~1.6σ observation; a 0.0024 one is not attributable to sampling | bootstrap on `oof/*.parquet` |
| **Projecting LB transfer from the CV fold MEAN** | **My error, and it cost the e0120 projection.** The per-fold deltas were monotone (−0.00365 … −0.00145) and EXPERIMENTS.md §3.2 already said to report the last fold separately and flag disagreement. Projecting from the mean predicted 1.6516–1.6523; the result was 1.6553. Also retired: the "4.4× model-family transfer rate", which was read off a −0.00018 CV delta (2σ) and was never sound | e0120 |

*Added 2026-08-13 — the saturation batch. All measured, none argued.*

| idea | result | evidence |
|---|---|---|
| **Guard-zone anchors as TRAINING data** (not validation) | **worse by +0.00189.** Validating at 2026-01-14: clean-only 29 anchors/6.07M rows = 1.68068; +guard 38 anchors/8.18M rows = 1.68258. Beats 35% more data AND our "training size wins" finding. Note the marginal target distributions are nearly identical (P(y=0) 44.4% vs 43.6%) — matching marginals says nothing about the conditional | `reports/eda/guard_test.json` |
| **`funnel` block — the 10 raw columns never ingested** | `search_to_ord` has genuine univariate signal (Spearman 0.5668, level with `to_ord` 0.5731) and +0.00489 incremental linear R², but **zero incremental value to the GBDT**: 819 features, CV 1.76547 vs 665-feature 1.76551, wins 2/5 folds. The existing features had already reconstructed it. Columns remain in `VALUE_COLS` — harmless, and needed if a different model class is ever tried | e0080 |
| **Recency: train on the most recent N anchors** | flat above 10 anchors. N=6 **worse** (+0.00023); N=10/14/18 all within ±0.00005 of using all ~25. Older anchors are neither stale nor valuable | e0070–e0073 |
| **Excluding RandomForest from AutoGluon** | **my error.** RF is slow (13–16k s) but carries real ensemble diversity. e0064 (with RF) beat e0060 by −0.00018; e0065 (without) = tie; e0066 (without, 665 feats) = tie. Never exclude it for speed | e0065, e0066 |
| **Global growth / seasonal multiplier** | optimal k = 0.990 on OOF (Δ −0.00003, noise). Applying the **real** measured +15.4% YoY growth costs **+0.00474**. The level term is 0.01% of RMSLE². Growth is real (intensive margin ×1.154) but structurally unusable | `explore.ipynb` §12.2 |
| **Predicting 0 for likely-inactive users** | worse at every threshold. At t=0.25, **95.7%** of the zeroed users truly are zero and it *still* loses. Break-even needs >96% precision. Same asymmetry that killed the hurdle: a true zero mispriced at 0.5 costs `log1p(0.5)=0.405`, a real buyer wrongly zeroed costs the full `log1p(y)` | `explore.ipynb` §12.3 |
| **The "two branches" in the pair grid** | the split is exactly `no orders in 180 d` — 18.4% of users, 85.3% zero targets, and `ord_last30`/`buy_days_180`/`gmv_last30` are all identically 0 there. Already fully visible to the model | `explore.ipynb` §12.4 |
| **Weekly instead of 30-day horizon** | e0024 tested 7d/14d targets as extra training rows: 1.76678 vs 1.76638, **lost**. (e0023 — short-horizon predictions as *stacked features* — won and is in the model.) Its main motivation, reaching 2026 targets, is independently dead per the guard-zone row above | e0023 / e0024 |



| what | result | evidence |
|---|---|---|
| Three-block population rule (users active in each of the last three 30-d blocks) | **worse**: RMSLE 2.279 vs 2.247, corr 0.528 vs 0.538 | `reports/eda_joint.log` §2–3 |
| Scoring past anchors on all 250 k users | optimistic by ~0.10; `E[L]` 2.014 vs the LB-measured 2.320 | `reports/eda_joint.log` §2 |
| Validating on anchors ≥ 2025-11-15 | optimistic by +0.041 — target window sits in the guaranteed-activity zone | DATA.md §4.3 |
| Target = `gmv_search` only, or ≠ 30-day horizon | both rejected as explanations of the CV−LB gap | `reports/eda_lbgap.log` H3, H4 |
| Reweighting folds to the test's feature cells | moves CV only 2.247 → 2.226; not a covariate-shift artefact | `reports/eda_match.log` §2 |
| `com` — centre-of-mass of activity/GMV in a window | **+0.00100 (11σ, 0/5 folds)** — the poison inside the e0003 bundle | e0014 |
| `cumshare` — cumsum-derived lifetime shares, expanding means, days-to-half-of-lifetime | **+0.00206 (23σ, 1/5)** — normalising by lifetime destroys the level information the model needs | e0012 |
| `diff` — 30-d windows at lags 0–6, first and second log-differences | −0.00015 (1.6σ, 3/5) = **no effect**; growth/acceleration adds nothing over levels | e0011 |
| Two-part hurdle model `P(buy)·E[log1p∣buy]` | −0.00012 (1.3σ) = **no effect**. Both forms estimate the same `E[log1p(y)∣x]`; the decomposition does not make estimation easier. The 0.983 oracle bound assumed *perfect* classification and is not reachable this way | e0010 |
| **Per-user gift-holiday responsiveness feature** ("gifters") | **Retested properly with 7 celebrations** (23 Feb, 8 Mar, 9 May, 1 Sep, 11.11, Black Friday, New Year) and a split-half reliability design. Gifter-score split-half = **+0.043**; Spearman-Brown reliability of the full 7-holiday composite = **+0.09**. Control on the same users with the same machinery: plain spending level splits at **+0.652** — so the method works, the trait does not exist. Restricting to the 22 042 users who actually buy in every window does not help (+0.043). Max true correlation consistent with the data ≈ 0.02 | `reports/gifters.log` |
| **Lookback-restricted ensemble** (members at 15/30/60/90/120/180/270/360-day lookbacks, blended in log space) | **Blend = 1.76719 vs best single member 1.76569 — WORSE by +0.00150.** The top members correlate at **0.99+** in log-prediction space, so they add nothing; the only decorrelated member (15 d, r = 0.82) is far too weak (1.977) to carry weight. Fitted weights collapsed onto the three best members. Equal-weight blend was far worse (1.784) | `logs/blend_23056182.out` |
| **Diversity by REMOVING information** (the general lesson) | Restricting a model's lookback does not create useful diversity — it creates a strictly degraded copy that makes the same errors plus noise. Useful diversity needs a different *inductive bias* over the **same** information (a linear model, an NN, a different objective), not less information | same |
| Extending training anchors earlier (min history 90 d → 60 d / 30 d) | **worse: +0.00026 / +0.00073.** The 90-day history requirement is load-bearing; early anchors with truncated windows are distributionally different enough to hurt | e0025, e0026 |
| **Dropping calendar-drifting features** (lifetime totals, 365-day windows) | **Pure cost.** The test cut-off IS a feature-space outlier (Wasserstein ratio 3.92 vs the training cut-offs' spread), but the transfer test shows **distance does not predict transfer loss**. Dropping 8 lifetime totals costs +0.00019 CV and cuts the ratio only 3.92 → 3.62; dropping all 37 costs +0.00205 for 3.92 → 2.69. Measured cost, unmeasured benefit | e0056, e0057, `transfer_test.json` |
| **My reasoning error worth remembering** | I measured a distance and treated it as a cost without testing the link — the same pattern as the +0.917 gifter excess and the 0.008 year-lag R², both of which evaporated when measured against the actual model | — |
| **What the transfer matrix actually showed** | The diagonal is NOT the best entry in 4 of 5 columns: models trained on later folds (more anchors) beat a fold's own model on that fold. **Training-set size dominates cut-off proximity** — consistent with the ES-holdout fix (−0.0038 from recovering ~4 anchors) and the min-history result | `transfer_test.json` |
| Killing `trend` as a bundle | **my error** — it was `ewm` (−0.00042, a real keep) + `com` (+0.00100) + `diff` (−0.00015). A negative bundle hid a positive part | e0003 vs e0011/13/14 |

---

*Added 2026-08-25 — the TRAINING-REGIME axis on the causal path (e0930–e0938, array 24333391,
9 arms × 5 folds × 3 seeds × 32 fixed epochs, ~91 min each). Pre-registration and the schedule
derivation: `scratch_thoughts.md` pre-registration #4. All arms scored on **within-anchor rho**
against a **matched same-session control** (e0195: cross-session drift on identical configs is
+0.00027…+0.00046, larger than every effect here).*

**Measured noise floor for this instrument:** control seed sd within fold **0.000068** → sd of
the 5-fold × 3-seed mean **0.000017**, so 2σ = **0.000035**. The pre-registered bar was
0.000064 and the verdicts below use the pre-registered, more conservative one.

| idea | result | evidence |
|---|---|---|
| **Day-recency weighting of the loss** (`w = 0.5**((last_anchor − t)/H)`) | **CLOSED across a 9× half-life bracket, monotone declining.** hl 270 −0.000013 (2/5, nil) · hl 90 −0.000300 (0/5) · hl 30 −0.002736 (0/5). The control **is** the `hl → ∞` limit, so the surface only falls as the half-life shortens — there is no interior optimum on the tested side. A flat-to-declining surface across 9× means the axis is **closed, not untuned**; the GBDT anchor-decay test that preceded it spanned only 3× (60d/180d). hl 30 keeps an effective 11.6% of days and reproduces e0025/e0026 (cutting data hurts) on the causal path | e0931, e0932, e0933 |
| **Curriculum, expanding window** (add fresh data each epoch, never remove old) | **KILL −0.000270, 0/5**, 4.2× the pre-registered bar. Notable *because* it should have been the most nil arm: every epoch ends at the full anchor range, so this is the control plus a warm-up. Losing means **the warm-up itself costs** — 32 fixed epochs spent partly on a restricted window are 32 epochs not spent on the whole one | e0934 |
| **Curriculum, sliding window** (drop old as fresh arrives, constant work/epoch) | **KILL −0.001162, 0/5**, 18× the bar. Consistent with e0388 (retracting the usercv window loses 0/5, −0.000128): a schedule that **ends** on a narrow recent window discards data exactly when the low-LR epochs consolidate. Per-epoch supervised cells were **measured** constant (32.43/32.43/32.49/32.39/32.20M, drift 0.9%), so it is not confounded with per-epoch work | e0935 |
| **The curriculum ORDER carries anything** (`--curr-shuffle` ordering null) | **REFUTED, and this is the informative arm.** Identical per-epoch window volumes with the epoch order permuted recovers **+0.000694** of `slide`'s −0.001162, winning **5/5 folds** against it (+0.00077/+0.00054/+0.00058/+0.00075/+0.00083). So **~60% of the curriculum's damage is the recency *progression* itself**, not the reduced per-epoch data — a random order of the same windows beats the "principled" increasing-recency one. **e0214's lesson again: when the ordering null beats the ordering, the schedule was never carrying information.** Still a kill in absolute terms (−0.000467, 0/5 vs control) | e0936 vs e0935 |
| **(user, date) sampling** — Bernoulli day mask resampled **every step** | **NO EFFECT, and the null is worth more than a win would have been.** keep-prob 0.5 → **+0.000010 (3/5)**; keep-prob 0.25 → **−0.000025 (2/5)**. Both inside the bar. **Three quarters of the supervised positions can be discarded at every step at no measurable cost** — which prices e0115's redundancy claim from the other direction (30-day target windows overlap 29/30 days, so positions are near-redundant). This was the one arm with a mechanism worth believing (a stochastic regulariser on a model known to sit on an overfitting cliff, e0106) and it is nil | e0937, e0938 |
| **Reading a training-regime delta off raw RMSLE** | **e0931 is the counter-example to keep.** It is nominally **−0.000061 better on raw UNSEEN RMSLE** and **−0.000013 on within-anchor rho** — the apparent gain was entirely calendar level, which §1b proves calibration removes for free. Report rho (§1z-A); never promote an arm on raw RMSLE | e0931 |
| **A schedule stated in the units the code is written in** | **The first `slide` draft held the *day width* constant (190 days) and still drifted +26% in supervised cells per epoch** (14.31M → 17.13M → 18.10M), because the mask is `t ≥ first_active + 14` and users enter the panel at different dates — an early 190-day window holds far fewer valid user-days than a late one. That would have confounded the arm with exactly what it exists to isolate. Fixed by stepping in **cell mass** (`cells_per_day`/`cum_cells`/`_hi_for_mass`), drift 0.9%. **Print the quantity the claim is about and read it off the log** — the +26% was invisible in the config and obvious in one counter | smoke 24324738 vs 24326235 |
