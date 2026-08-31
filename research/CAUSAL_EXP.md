# Experiment: GRU for 30-day forward GMV

Sequence model over per-user daily panels. One sample = one user's full history.
At every timestamp the model emits a prediction; the target at timestamp `t` is the
**mean daily GMV over the next `horizon` days**.

Implement this as a runnable pipeline. Three feature-set variants (A/B/C) share
everything else — data loading, target construction, model, training loop, evaluation,
inference — so build those once and switch feature sets by config.

---

## 1. Data

Raw table, one row per `(user_id, event_date)` where the user was active. Days with no
activity are simply absent from the table.

| column | meaning |
|---|---|
| `event_date` | date, 2025-01-01 … 2026-02-13 |
| `user_id` | user identifier |
| `search` | used Search that day (0/1) |
| `cat` | used Catalog that day (0/1) |
| `has_search_to_cart` | added to cart via Search (0/1) |
| `has_search_to_ord` | purchased via Search (0/1) |
| `has_cat_to_cart` | added to cart via Catalog (0/1) |
| `has_cat_to_ord` | purchased via Catalog (0/1) |
| `search_to_cart` | # items added to cart via Search |
| `search_to_ord` | # items purchased via Search |
| `cat_to_cart` | # items added to cart via Catalog |
| `cat_to_ord` | # items purchased via Catalog |
| `gmv_search` | GMV of items purchased via Search |
| `gmv_cat` | GMV of items purchased via Catalog |
| `to_cart` | total # items added to cart |
| `to_ord` | total # items purchased |
| `gmv` | total GMV |
| `searches` | total # search queries |

Groups: `FLAG_COLS` = the six `has_*` plus `search`, `cat`; `COUNT_COLS` = the ten numeric
columns. Flags get no log/scaling; counts get `log1p` then standardisation.

### 1.1 Panel construction

1. Parse dates, clip to `[start_date, end_date]`.
2. `groupby(["user_id", "event_date"]).sum()` to collapse accidental duplicate rows;
   re-binarise the flags with `> 0`.
3. Reindex onto the **full calendar** `pd.date_range(start_date, end_date, freq="D")`
   (T = 409 days). Missing days become all-zero rows.
4. Result: dense `float32` array `(n_users, T, n_raw)` plus an `active` mask
   `(n_users, T)` — 1 where the user had a row in the source table. `active` is itself a
   feature: an all-zero row means "no visit", and the model must be able to tell that from
   a visit with no purchase.

Keep the raw (pre-transform) `gmv` slice around — the target is built from it before any
`log1p`/scaling is applied to features.

---

## 2. Target

```
horizon = 30
y[u, t] = mean(gmv[u, t+1 : t+1+horizon])         # strictly future, day t excluded
```

- Compute with a cumsum over the time axis, not a Python loop.
- `include_today_in_target = False` by default. Setting it to `True` (window `t … t+29`)
  leaks: `gmv[t]` is an input feature at step `t`, so the model reads part of its own
  answer off the input.
- Apply `log1p` to the target (`log1p_target = True`). The reporting metric is RMSE in log
  space, i.e. RMSLE on the mean-daily-GMV scale. Keep the inverse (`expm1`) in one place so
  mean-scale reporting is a one-liner.
- Config flag `target_agg ∈ {"mean", "sum"}`. Default `"mean"`. `sum = mean * horizon`, but
  because of the `log1p` the two are *not* a rescaling of each other in loss space — pick
  one and keep it fixed across A/B/C or the runs aren't comparable.

### 2.1 Which timestamps are scored — `tmask`

A boolean `(n_users, T)` mask, 1 where the loss and the metrics are computed.

- **Horizon tail.** `y[t]` needs day `t + horizon` to exist. Last valid index is
  `T - horizon - 1` (= 378, i.e. 2026-01-14). Beyond that the cumsum returns a *partial*
  sum, which looks like a genuinely low-spending user rather than a truncated window. Zero
  the mask there. Also expose `truncate: bool` which physically slices the arrays to
  `T - horizon`; it's an optimisation and a guard against forgetting the mask elsewhere,
  not a correctness requirement.
- **Burn-in.** `burn_in = 14`: at `t = 3` the GRU has three days of context and its
  prediction is noise that still contributes gradient. Zero the first `burn_in` steps. If
  `trim_to_first_seen = True`, measure burn-in from the user's first active day rather than
  from 2025-01-01 — it matters for users who first appear in October.

Loss is `((pred - y)**2 * tmask).sum() / tmask.sum()`. Every metric must be mask-aware; a
plain `.mean()` anywhere is a bug.

---

## 3. Feature engineering

Every feature is **causal**: at timestamp `t` it may only use days `≤ t`. Window
aggregates are inclusive of day `t`.

Base helper — rolling mean over the past `W` days, vectorised via cumsum, with the
denominator clipped at the start of the series so early timestamps average over however
many days exist rather than dividing by `W`:

```python
def rolling_mean(x, W):          # x: (N, T) -> (N, T)
    cs = np.concatenate([np.zeros((x.shape[0], 1)), np.cumsum(x, 1, dtype=np.float64)], 1)
    idx = np.arange(x.shape[1])
    lo  = np.maximum(0, idx + 1 - W)
    return ((cs[:, idx + 1] - cs[:, lo]) / (idx + 1 - lo)).astype(np.float32)
```

`rolling_std` from `E[x²] − E[x]²` using the same helper.

Windows: `WINDOWS = [7, 30, 60, 90]` (day, week, month, 2 months, 3 months).

The output of this stage is, for every `(user, timestamp)`, a flat vector of `n_features`.

### 3.1 Three variants

**Experiment A — `gmv_only`.** Minimal baseline, isolates how much of the signal is
autoregressive in GMV alone.
- `gmv` on day `t`
- `rolling_mean(gmv, 7)`
- `rolling_mean(gmv, 30)`
- `active` flag
- (≈4 features)

**Experiment B — `full`.** All raw columns plus their window aggregates.
- all 18 raw columns on day `t` (flags as-is, counts `log1p`-ed)
- `rolling_mean(c, W)` for every `c` in `COUNT_COLS` and every `W` in `WINDOWS`
- `rolling_mean(f, W)` for every flag `f` — this is the usage *rate* over the window,
  which is more informative than the flag itself
- `active` + `rolling_mean(active, W)`
- (≈100 features)

**Experiment C — `extra`.** B plus the blocks that carried weight in the tabular
experiments. Add them as named blocks so individual blocks can be ablated.
- **Recency:** `days_since_active`, `days_since_order` (`to_ord > 0`),
  `days_since_cart` (`to_cart > 0`). Vectorise with `np.maximum.accumulate` over
  `where(active, t, -1)`; use `T` as the sentinel for "never". Feed as `log1p(days)`.
- **Dispersion of activity timing** — the block that mattered most in the tabular runs:
  `rolling_std(gmv, W)`, `rolling_std(active, W)`, `rolling_std(days_since_active, W)`,
  and coefficient of variation `rolling_std / (rolling_mean + eps)` for `gmv` and `to_ord`.
- **Ratios / conversion:** `to_ord / (to_cart + eps)`, AOV `gmv / (to_ord + eps)`,
  `to_cart / (searches + eps)`, `search_to_ord / (search_to_cart + eps)`,
  `cat_to_ord / (cat_to_cart + eps)`, Search share `gmv_search / (gmv + eps)`. Compute
  these **on the rolling sums, not per-day** — per-day ratios are 0/0 on most days. Clip
  to a sane range.
- **Trend:** `rolling_mean(gmv, 7) / (rolling_mean(gmv, 30) + eps)` and
  `rolling_mean(gmv, 30) / (rolling_mean(gmv, 90) + eps)` — short-vs-long momentum.
- **Expanding:** tenure (days since first active day), cumulative active-day count,
  expanding mean GMV, expanding max GMV.
- **Calendar:** `sin/cos(2π·dow/7)`, `day_of_month / 31`, month one-hot or `sin/cos`.
  Shared across users, broadcast.

### 3.2 Scaling

Fit `mu`/`sigma` on the **train users only** and pass them into the val/test datasets
(`stats=train_ds.stats`). Leave binary flags unscaled (`mu=0, sigma=1`). Computing stats
over all users leaks the validation population's spending distribution into the features.

---

## 4. Splitting

**Split by users, not by dates.** Deterministic hash split so it's stable across runs and
dataframe orderings:

```python
h = int(hashlib.md5(f"{salt}:{user_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
val = h < val_frac       # val_frac = 0.2, salt = "gmv-v1"
```

Apply any `min_active_days` filter to the user list *before* splitting so the proportions
hold exactly.

Every kept user contributes their whole 409-day series. Since the sets are user-disjoint
there is no target overlap between train and val, so no date cutoff is needed — the
horizon-tail mask from §2.1 is the only temporal restriction.

**Caveat to report in the results, not to fix:** a user split answers "does this
generalise to a new user?", not "does it generalise to next month?". Both splits span the
same calendar, so any calendar artefact (December spike, promo week) is scored as skill.
Optionally log a second validation number with `burn_in` pushed to 2025-11-15 on the val
users, which scores only the tail of their series and gives a forward-in-time reading too.

---

## 5. Model

```
input (B, L, n_features)
  → optional Linear(n_features, hidden) + LayerNorm + GELU     # input projection
  → GRU(hidden, num_layers=2, batch_first=True, dropout=0.1)
  → LayerNorm
  → Linear(hidden, 1)
  → squeeze                                                    # (B, L)
```

Defaults: `hidden = 128`, `num_layers = 2`, `dropout = 0.1`.

The GRU is causal, so the output at step `t` already conditions on the entire history up
to `t` — no windowing, no packing needed if every user has the same length (they do,
unless `trim_to_first_seen = True`, in which case right-pad and use `lengths` +
`pack_padded_sequence`, and make sure padding steps are zero in `tmask`).

Training:
- Adam, `lr = 1e-3`, `weight_decay = 1e-5`
- `batch_size = 256` users
- gradient clipping at `1.0`
- up to 60 epochs, early stopping on val loss with `patience = 8`, restore best weights
- masked MSE on the `log1p` target; expose `--loss huber` as an alternative given the
  heavy tail
- `ReduceLROnPlateau` on val loss
- fixed seed; `--seeds 3` to average the final metric, since run-to-run variance on this
  kind of data is easily the size of the A→B gap

Sequence length is fixed at 409, so memory is `batch × 409 × hidden` — trivial. If the
date range is ever extended, add truncated BPTT: chunk the series and carry the hidden
state with `h.detach()` between chunks.

---

## 6. Evaluation

On val users, masked over `tmask`:
- **RMSE in log space** (primary — this is RMSLE on the target scale)
- MAE in log space
- mean-scale metrics after `expm1`: MAE, and aggregate-level bias
  `sum(pred) / sum(actual) − 1` (log-space training systematically under-predicts the
  mean; report it rather than correcting it, since the primary metric is the log-space one)
- the same metrics sliced by activity decile, so it's visible whether a variant only helps
  on heavy users
- a baseline row: predict `rolling_mean(gmv, 30)` at every timestamp. If a variant doesn't
  beat this, say so plainly.

---

## 7. Test inference

The last `horizon` days have no observable target, but they are perfectly valid **inputs**.

1. Build the inference dataset with `truncate = False` and `stats = train_ds.stats`, over
   **all** users (train and val), so the model sees every user's full history including the
   most recent 30 days that never appeared in any training target.
2. One forward pass over the entire series.
3. Take the prediction at the **last timestamp only** — index `T - 1` = 2026-02-13, the
   step at which the model has been given the complete context. Do not average over the
   last few days; the earlier steps have strictly less information.
4. `expm1` to get mean daily GMV; multiply by `horizon` for the total. Write both columns.

```
predictions/{exp_name}.csv:  user_id, pred_mean_daily_gmv, pred_gmv_30d
```

Also dump the per-timestamp predictions for a handful of users (`--dump-traces 20`) as a
sanity check — the trace should track the GMV series with a lag, not be flat.

---

## 8. Deliverables

```
data.py        panel construction, target, tmask, splits, Dataset, collate
features.py    rolling helpers + FEATURE_SETS = {"gmv_only": ..., "full": ..., "extra": ...}
model.py       GRUForecaster
train.py       train / eval loop, CLI
predict.py     test inference (§7)
results.md     appended after every run
```

CLI:

```bash
python train.py --exp gmv_only --data data/events.parquet --seeds 3
python train.py --exp full     --data data/events.parquet --seeds 3
python train.py --exp extra    --data data/events.parquet --seeds 3
python predict.py --exp extra  --ckpt runs/extra/best.pt
```

Each run writes `runs/{exp}/{seed}/` with config, metrics JSON, loss curves, checkpoint,
and appends one row to `results.md`: experiment, n_features, val RMSLE (mean ± std over
seeds), mean-scale MAE, aggregate bias, epochs to best, wall-clock.

## 9. Checks to write as asserts, not as hope

- No feature at index `t` reads any day `> t`. Test: zero out the panel from day `t+1`
  onward for a random user, recompute features, assert columns at `t` are unchanged.
- `y[u, t]` equals a hand-computed mean of `gmv[u, t+1 : t+31]` for several random `(u, t)`.
- `tmask` is 0 for every `t > T - horizon - 1` and for every `t < burn_in`.
- No train user id appears in the val set.
- Val normalisation uses train `mu`/`sigma` — assert the objects are identical, not merely
  equal-shaped.
- Feature counts match what §3.1 claims for each variant; log `n_features` per run.

---

# RESULTS — what this design actually produced (updated 2026-08-24)

> This spec described an experiment. Below is what measurement did to it. Full derivations in
> `EXPERIMENTS.md` §1z-B; every run has a row in `experiments.csv`.

## The one setting in this document that was wrong: `hidden = 128`

§5's defaults (`hidden=128, num_layers=2, dropout=0.1`) were stated, not measured. Sweeping
width on the user-split CV — matched same-session controls, **within-anchor** rho — gives a
clean unimodal curve with the optimum at **d = 48**:

```
d192 0.66462 < d128 0.66486 < d96 0.66497 < d64 0.66513 < d48 0.66525 > d32 0.66519
```

**+0.00039 rho, 14× anything else ever measured on this path**, and it is the change that moved
the leaderboard: **1.646589 → 1.646456**, predicted from OOF to within 2e-6.

It survives the three checks that kill most results here: §1j's **fixed-epoch control** (width is
worth +0.00023 at 14 epochs and +0.00080 at 22 — the effect GROWS with budget because d128
overfits while the small model keeps learning); an **independent protocol** (the frozen-fold seq
path peaks at d48 too, −0.00099, 5σ, 5/5 folds); and the **leaderboard**.

Why it stayed hidden: §1j's tuning of this path scored trials on the MIN of a variable-length
early-stopped curve — the statistic §1j itself then proved invalid — and §1k redid the search
properly but on the *frozen-fold* path, not this one. `--fixed-epochs` now exists and prices this
protocol's own bias at **+0.00124 RMSLE / +0.00019 rho**.

## §5's architecture question, answered

`--model {gru,lstm,transformer}`, all at matched widths:

```
GRU  d48   0.66525  <- best      LSTM d128   0.66471
LSTM d32   0.66489               xformer d48 0.66456
LSTM d48   0.66487               xformer d64 0.66436
GRU  d128  0.66486               xformer d128 0.66420
```

**GRU > LSTM > transformer at every width.** The transformer was genuinely handicapped by being
tested at d128 (fixing width is worth +0.00036) but that closes only half the gap. Raising the
epoch cap 60 → 150 gives **byte-identical** results, so nothing was ever under-trained — and that
also proves andrena runs are bit-deterministic.

## §5's `--loss huber` slot: built as something better, and refuted

The doc reserved an alternative loss. What was built instead is the one §1b actually motivates —
an **affine-invariant correlation loss**, since after calibration only rho scores while MSE also
pays for a level and spread that calibration discards. Implemented with §1r's WITHIN-anchor
estimator as `--loss {corr,mix}`.

**Nil**: mix +0.00001 at d128 and +0.00001 on top of d64; pure corr −0.00058. Larger batches make
it monotonically worse. *Training directly on the scored quantity buying nothing is evidence this
model is signal-limited, not mis-optimised.*

One genuinely new fact fell out: pure-corr training reaches **r = 0.879 with the blend** — the most
decorrelated model this project has built (BTYD 0.9427, Ridge 0.9433, every neural variant ≥0.997),
and 3× closer to the admissibility frontier than anything on record. It is still worth ~0, because
decorrelation only pays at comparable quality.

## Everything else tried on this design — all nil on within-anchor rho

```
dropout 0.2 -0.00001 | dropout 0.3 -0.00002 | weight-decay 1e-3 +0.00002
mixup +0.00001       | --pop-train  -0.00002 | layers 3 -0.00008
```

* **§5's mixup is NOT a keep on the statistic that matters.** ΔRMSLE −0.00016 (0.8σ) but
  **Δrho +0.00001**. The mechanism replicates (best-epoch 13.5 → 18.2) but the gain lands in the
  level/spread term §1b proves is free. §1d's −0.00065 was against a *one-seed* baseline.
* **§2.1's tmask has a population mismatch, and fixing it changes nothing.** `run_seq.py` trains on
  in-population days only; this path trained on every masked day. Measured: **6.2% of the
  80,899,560 training user-days are dormant**, while **100%** of the 250,000 test users are
  in-population by construction. `--pop-train` aligns them for −0.00002.

## ⚠ Reading this path's numbers: what CV can and cannot tell you

Three submissions in one day settle how far the user-split CV can be trusted:

```
slot contents -> a SINGLE better member   realised 0.88x   GAINED -0.000133
slot contents -> an AVERAGE of members    SIGN INVERTED    +0.000027
blend weights FITTED on CV                SIGN INVERTED    +0.000593
```

> **The user-split CV over-values every form of variance reduction, and is reliable only for a
> single-member quality swap.** §9e reached this from seed-averaging; architecture-averaging is the
> same mechanism. And **leave-one-fold-out does not protect fitted weights** — it said +0.000217 on
> 5/5 folds while the measured Δrho was −0.000259, because LOFO guards variance from overfitting
> the OOF and is blind to CV↔LB *ranking* shift.

Solving component rhos against the measured LB scores shows why: **d48's solo rho is 0.70269,
BELOW e0141's 0.70311.** It helps the blend by *decorrelation at lower quality*, so any procedure
that reads CV as "d48 is the better model" will over-weight it.

## Current production recipe

```
src/run_usercv.py    --variant full --hidden 48 --folds 0 1 2 3 4 --seeds 3 --epochs 60
src/predict_usercv.py --variant full --hidden 48 --epochs 32 --seeds 3   # 32 = CV median best-epoch
```

`--epochs` is architecture-specific (GRU 32, LSTM 43, transformer 21) and must match the CV run the
median came from.
