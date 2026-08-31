# FEATURES.md — tabular (`anchor_cv`) feature screens

Status of the feature screen for the **anchor_cv (tabular LightGBM, e0049-family)** path.
Every entry below was **measured locally**, not speculated: a single candidate added alone to
the installed 1021-feature block set, training a Golden LightGBM (lgb 4.6.0, lr 0.05,
num_leaves 63, min_data_in_leaf 40, ff 0.8, bf 0.8, early-stop 80) on a frozen 30k-user
subsample, random user 50/50 split, scoring Delta-rho against baseline rho. Metric is the
log-space correlation `rho` (the CV driver; RMSLE σ_noise for GBDT = 0.00009, §3.4).

> ## ⚠ CONFIRMED 2026-08-22 — all three candidates are dead, and a noise column beat them
>
> Every confirm-eligible candidate in this document went to full 5-fold CV as XGBoost arms
> (e0211–e0213) against a matched base (**e0210**, cv 1.76588) **and a pure i.i.d. normal
> column as a fifth arm** (e0214). Including that control is what makes the result readable:
>
> | arm | cv_mean | Δ vs e0210 | σ | folds won |
> |---|---|---|---|---|
> | **e0214 — i.i.d. NOISE COLUMN** | **1.76595** | **+0.00006** | +0.71 | 2/5 |
> | e0212 `cart_backlog_7` (cand. B) | 1.76596 | +0.00007 | +0.82 | 2/5 |
> | e0213 `cohort_rel_buy_rate90` (cand. C) | 1.76596 | +0.00008 | +0.89 | 2/5 |
> | e0211 `age_bucket_gmv_share_3` (cand. A) | 1.76600 | +0.00011 | +1.27 | 3/5 |
>
> **The noise control has the best cv_mean of the four, and the whole 5-arm spread is 0.00011
> = 1.27σ.** Every candidate is (a) positive — worse than not adding it — and (b) *behind a
> random number*. Nothing here is distinguishable from the measurement. **All three: `kill`.**
>
> **What this document got wrong.** The screen ranked A at +0.00074/+0.00112 and B at
> +0.00090/+0.00052 and I called them "borderline, mechanism-sound, worth one confirm each."
> The mechanism arguments were fine and the confirms were the right call — but the screen's
> ordering carried no information: it put A first and A is now the *worst* of the three, behind
> noise. **A screen at ~1× its own noise control ranks nothing.** Same lesson the causal path
> learned at 36× inflation (FEATURES_CAUSAL.md), from a different instrument.
>
> Tabular feature search is now closed on 22 hand-designed candidates + 60 tsfresh statistics,
> all nil. Do not open it again without a mechanism the installed 665 features provably cannot
> represent.

**TL;DR (2026-08-20, superseded above): the tabular feature screen is CLOSED.** 19 candidates
(e0189), the tsfresh port (e0192 screen + **e0191 confirm: Δ −0.00006, 2/5 folds, no
effect**) — nothing survived.

**Two independent anchors were run** so a positive result must replicate in sign:
A1 = 2025-06-18, A2 = 2025-10-16 (both on the *same* 30k user subset = clean paired delta).

## Screen sensitivity (the noise control)

A pure i.i.d. normal column added alone moved rho by:

| anchor | noise-control d_rho |
|---|---|
| A1 (06-18) | +0.00014 |
| A2 (10-16) | −0.00097 |

The screen's own sensitivity is thus **~±0.001**. On this budget (13.5k train rows vs the
~66k/fold of the real pipeline) the installed 1021-feature baseline is mildly
overfit-starved, so *any* screen delta under ~0.001 is indistinguishable from noise. Only
candidates that are (a) positive at **both** anchors AND (b) at or above the noise control
are worth a confirm run. **Screen tier results never enter a final decision** (§4.2).

---

## Result: no candidate clears 2σ. Three are borderline-consistent and worth ONE cheap confirm.

The installed set (base, counts, trend, rank, visit, channel, diff, cumshare, ewm, com,
dispersion, sbcnomoment, tsfeat, fcast, funnel) already encodes level, window, recency-of-
activity/order, ranks, funnel and dispersion. **Every one of the 19 candidate families —
cohort-normalised levels, weekend/dow shape, recency-of-cart/search, cart backlog, age-aligned
lifecycle buckets — is within ~1× the noise control.** Names and both-anchor deltas:

| candidate | A1 Δrho | A2 Δrho | verdict |
|---|---|---|---|
| **age_bucket_gmv_share_3** | +0.00074 | +0.00112 | borderline — **test one** |
| **cart_backlog_7** | +0.00090 | +0.00052 | borderline — **test one** |
| cohort_rel_buy_rate90 | +0.00011 | +0.00042 | borderline (smallest) |
| cohort_rel_gmv90 | +0.00068 | −0.00042 | ✗ sign flip |
| cohort_rel_gmv30 | −0.00013 | −0.00063 | ✗ |
| weekend_gmv_share_90 | +0.00072 | −0.00040 | ✗ sign flip |
| distinct_dow_with_buy_90 | +0.00011 | −0.00235 | ✗ |
| recency_cart | −0.00038 | +0.00065 | ✗ sign flip |
| recency_srch | −0.00025 | −0.00181 | ✗ |
| cart_to_ord_ratio_l7 | −0.00049 | −0.00148 | ✗ |
| age_bucket_gmv_0 | +0.00109 | −0.00072 | ✗ sign flip |
| age_bucket_gmv_share_0 | −0.00134 | −0.00096 | ✗ (negative) |
| age_bucket_gmv_1 | −0.00016 | −0.00019 | ✗ |
| age_bucket_gmv_share_1 | −0.00046 | +0.00059 | ✗ |
| age_bucket_gmv_2 | +0.00058 | −0.00002 | ✗ |
| age_bucket_gmv_share_2 | −0.00131 | −0.00044 | ✗ (negative) |
| age_bucket_gmv_3 | −0.00039 | +0.00022 | ✗ |
| cohort_rel_ord90 | +0.00019 | −0.00515 | ✗ |
| cohort_rel_buy_days90 | −0.00055 | −0.00284 | ✗ |

**Recommendation: do not add tabular features on this evidence.** The strongest two
(age_bucket_gmv_share_3, cart_backlog_7) are inside screen noise at 30k users and would need
a full-data confirm against e0049 to be believed; **the correct action is a single confirm
run each (one change, frozen folds), and if either does not beat 2× noise on ≥4/5 folds, kill
it and treat tabular features as a closed chapter.** Both candidates below are mechanism-sound
(gaps the installed set actually has), so confirm is cheap and worth it.

---

## Candidate A — `age_bucket_gmv_share_3` *(confirm-eligible)*

- **The change (one block):** GMV share in the user's *own* age-bucket `[90,120)` days after
  first active day: `gk = Σ gmv(t)·[age(t)∈[90,120) ∧ t≤ai]`, `life = Σ gmv(t≤ai)`,
  feature = `gk / life`. Causal by construction — both numerator and denominator are killed at
  the anchor `ai` (**do not** sum over the whole panel; that leaks the target window).
- **Gap it fills:** installed `gmv_blk` are *calendar*-anchored months, so for a 6-month-old
  user "month 6" is their 6th calendar month (a cohort-blend), not their own 6th month-of-life.
  `age_bucket_gmv_share_3` removes that confound — the same quantity DATA.md calls out as the
  difference between cohort and age alignment.
- **Why it might work:** age-aligned late-life (months 3-4) GMV share isolates the "survivor
  spenders" trajectory independent of join date; the third bucket is when the join-date
  transient has washed out and the long-run level emerges.
- **Measured informativeness (screen, n=30k):** winning bucket share-3, Δrho **+0.00112 (A2)**
  and **+0.00074 (A1)** — the largest consistent positive in the table, but only ~1× noise
  control. Screen verdict: *borderline; validate with one full-data confirm vs e0049*.
- **Suggested confirm parent:** e0049 (665-feature LightGBM, frozen folds, `data/folds.parquet`).

## Candidate B — `cart_backlog_7` *(confirm-eligible)*

- **The change (one block):** `max(to_cart − to_ord, 0)` summed over the trailing 7 days
  (window 7, anchor ≤ ai): how many items are sitting unconverted in the basket.
- **Gap it fills:** no installed feature measures *undelivered* selection volume — cart is
  present as input flow, order as outflow, but the *stock between them* is never formed. This
  is the classic purchase-intent leading indicator (add-to-cart streak without conversion).
- **Why it might work:** a user who has loaded & not emptied a cart in the last week is in a
  different near-buy state than one who carts-and-buys immediately; RMSLE cares about the
  buy-flag far more than the amount (classification dominates, §1b).
- **Measured informativeness (screen, n=30k):** Δrho **+0.00090 (A1)**, **+0.00052 (A2)** —
  consistent positive, small, within ~1× noise control. Screen verdict: *borderline; validate
  with one full-data confirm vs e0049*.

## Candidate C — `cohort_rel_buy_rate90` *(optional third confirm)*

- **The change (one block):** the trailing-90d buy rate minus its **tenure-cohort median**
  (cohort = `first_act // 14`, i.e. 14-day join buckets). Separates a user's buy-rate from the
  baseline rate of their join cohort (recent joiners naturally buy less often than veterans).
- **Measured:** Δrho **+0.00011 (A1)**, **+0.00042 (A2)** — consistently positive but the
  smallest; lowest priority.

---

## Dead-on-arrival, do not re-test

Rejected by the screen — these are near-noise or sign-flipped, and three are the exact
nominal "informative" features that the leak audit killed:

- **recency_cart / recency_srch** — *leaky in the first draft:* scanning last-cart over the
  whole panel read post-anchor days into a "*days since*" that was counted negative; the fake
  signal was **+0.00605 rho**, which collapsed to nil (+0.00065) once the scan was capped at
  `t ≤ ai`. **Never scan recency past the anchor.**
- **age_bucket_gmv_share_{0,1,2}** — same leak class: buckets that extend past `ai` read the
  target window. Bucket-3 is the only one that survives the causal cap (because for old-enough
  users `[90,120)` still lies ≤ `ai`) — which is *why* only share-3 looks alive, a leak-by-
  construction tell worth remembering.
- **All cohort_normalised LEVELS (gmv30/gmv90/ord90/buy_days90)** — redundant with existing
  level + rank features; log1p(level − cohort-mean) is a rank almost identically captured.
- **weekend_gmv_share_90** — weekday/weekend split adds nothing once level and rate are in.

---

## How to confirm (the only way these enter the pipeline)

For each confirm-eligible candidate: add the block to `src/features.py` as an independently
toggleable block, train **e0049 + block** on the frozen folds, report against e0049 parent.
Δ>2×σ_noise (0.00009) **or** ≥4/5-fold win → keep; else kill and log (§4.1, §3.4). One
candidate per run. Bundling A+B is a separate experiment and must be re-validated.

### Confirm status (2026-08-21) — RUNNING under XGBoost as e0210–e0214 (EXPERIMENTS.md §1p)

All three candidates were built as single-feature blocks and submitted. **The confirm is run
under XGBoost, not LightGBM**, on user instruction ("validate those new features via XGB CV").
That forces one extra run: a candidate confirm needs a parent differing by exactly one thing
(§4.1), so an XGB baseline on the *identical* 665 features must exist first, or every candidate
delta confounds feature-change with family-change.

| exp | parent | change | blocks |
|---|---|---|---|
| e0210 | e0049 | XGBoost instead of LightGBM, matched hyperparameters | 665 feat |
| e0211 | e0210 | + `age_bucket_gmv_share_3` (Candidate A) | `agebucket` |
| e0212 | e0210 | + `cart_backlog_7` (Candidate B) | `cartbacklog` |
| e0213 | e0210 | + `cohort_rel_buy_rate90` (Candidate C) | `cohortrel` |
| e0214 | e0210 | + one i.i.d. normal column — **noise control** | `noisectl` |

**e0214 is the load-bearing run.** These candidates sat at ~1× the *screen's* noise control, and
a confirm inherits the same problem one level down: with 665 features already installed, the
question is not "is the delta positive" but "is it larger than what a definitionally worthless
column produces under this exact protocol". Read e0211–e0213 against e0214, never against zero.
FEATURES.md's screen could only say "nothing clears noise" *because* it carried such a control;
a confirm without one is a confirm you cannot interpret.

Parameter mapping (what makes e0210 a fair family reference rather than a different-sized model
wearing the same features): `num_leaves 63` → `grow_policy lossguide` + `max_leaves 63`,
`max_depth 0`; `min_data_in_leaf 200` → `min_child_weight 200` (squared error ⇒ hessian 1 per
row, so sum-of-hessians per leaf *is* the row count); `feature_fraction`/`bagging_fraction 0.8`
→ `colsample_bytree`/`subsample 0.8`; `lambda_l2 1.0` → `lambda 1.0`; `max_bin 255` → `256`
(xgb counts bin edges); 178 fixed rounds at lr 0.05, no early stopping (§1j).

**Pre-registered decision rule.** Keep only on Δ vs **e0210** beating 2σ_noise or winning ≥4/5
folds, *and* exceeding |Δ(e0214)|. Anything else is killed and logged. Given §1m — the
recombination ceiling is −0.00004 CV and a new blend member needs rho_partial ≈ 0.024 against a
best-ever 0.01269 — the honest prior is that all three land inside noise. The point of running
them is that they are the last unmeasured tabular candidates, and the cost is one array.

---

## tsfresh block (e0192 screen) — nil at screen resolution, confirm running as e0191

A port of the [tsfresh](https://github.com/blue-yonder/tsfresh) statistics that the installed
`tsfeat`/`sbc`/`fcast` blocks do **not** already cover. The library itself is row-by-row
Python/pandas and would take hours over 250k users × ~90 anchors, so the selected extractors
were hand-vectorised as matrix ops over the `(n_users, window)` slice: `block_tsfresh` in
`src/features.py`, 60 features = 10 statistics × {gmv, ord} × {30, 90, 365}.

**Every statistic was validated against tsfresh 0.21.2 itself** (max relative error):

| statistic | vs library | | statistic | vs library |
|---|---|---|---|---|
| `c3` | **0.0** | | `autocorrelation` (pacf1, arch7) | 4e-08 |
| `time_reversal_asymmetry` | 3e-08 | | `linear_trend` slope/stderr (trendt) | 2e-15 |
| `longest_strike_above/below_mean` | **0.0** | | `lempel_ziv_complexity` | 6e-08 |

**Three of the seven were wrong on the first pass and only the library check caught it:**
`c3` had a bispectrum-style formula (rel err 1e15), `arch7` used a segment-wise Pearson
instead of tsfresh's global-mean/`(n−lag)·var` convention (0.41 off), and `trendt` omitted the
intercept from the residual, deflating the t-stat ~12%. A "tsfresh-style" feature that is not
the tsfresh statistic is an untested new feature wearing a validated name — worth the hour.

### Screen result: the bundle is nil, and no family survives both anchors

| | A1 (06-18) | A2 (10-16) |
|---|---|---|
| noise control | +0.00014 | −0.00097 |
| **tsfresh bundle (66 cols)** | **+0.00019** | **−0.00074** |

Because a 66-column bundle is penalised ~66× harder than the 1-column control on this
overfit-starved screen, a null bundle does **not** clear the 6-column families — so the
decomposition was forced at **both** anchors (§4.1: a bundle is never evidence as a unit):

| family | A1 Δrho | A2 Δrho | verdict |
|---|---|---|---|
| tf_wavenergy2 | +0.00039 | −0.00003 | ✗ best at A1, dies at A2 |
| tf_wavenergy3 | +0.00010 | +0.00009 | ✗ positive but ~10× under noise |
| tf_pacf1 | −0.00086 | +0.00018 | ✗ sign flip |
| tf_longstrike_below | −0.00035 | +0.00004 | ✗ sign flip |
| tf_lz | +0.00002 | −0.00218 | ✗ |
| tf_trendt | −0.00028 | −0.00081 | ✗ |
| tf_c3 | −0.00043 | −0.00054 | ✗ |
| tf_wavenergy1 | −0.00078 | −0.00053 | ✗ |
| tf_longstrike_above | −0.00086 | −0.00198 | ✗ |
| tf_arch7 | −0.00171 | −0.00110 | ✗ |
| tf_timerev | −0.00175 | −0.00054 | ✗ |

**No family is positive at both anchors, and none reaches the ±0.001 noise control.** This is
the same verdict as the 19 hand-designed candidates above: the tabular stack is saturated.

`lempel_ziv` is implemented and validated but **deliberately not emitted** — it costs ~32 s
per 250k users per window against ~0.3 s for every other statistic (an unvectorisable
per-user prefix scan that ignores all 16 cores, ~2 h added to a fold build), and it was the
worst family at A2. The first cluster attempt died on exactly this: 55 min without finishing
one fold's features. Re-enable only if a vectorised LZ is written.

**Confirm-tier run:** e0191 (e0049's 7 blocks + `tsfresh`, frozen folds, full 250k users)
settled it: **cv 1.76545 vs e0049's 1.76551 = Δ −0.00006 (0.6× σ_noise), wins only 2/5
folds.** The null is real information — the block has no signal at any resolution. **Kill;
tsfresh is a closed chapter** (e0192 screen + e0191 confirm, both logged). The three
implementation bugs the library check caught are the lasting value of this line of work:
they document how easy it is to ship a "tsfresh-style" statistic that is not the tsfresh
statistic.

Per §3.4 the block does **not** fail by the book: Δ = −0.00006 (RMSLE, negative is better)
is *below* 2×σ_noise = 0.00018 in the improving direction and it wins 2/5 folds — yet at
0.6× noise the verdict is `no effect` by the |Δ| < σ_noise bar, and a 0.00006 gain on 60
columns is the definition of overfit-CV accumulation §3.4 warns about. **Killed on merit,
not on the letter of the rule.**
