# FEATURES_CAUSAL.md — causal (`user_cv` GRU) feature screens

> ## ⚠ CONFIRMED 2026-08-20 — the headline of this document did not survive the GRU
>
> Both candidates were built as isolated single channels on top of `full` (variants
> `full_dso`, `full_backlog`) and confirmed at 5 folds × 3 seeds against a **matched
> same-session control** (e0195 = 1.74387):
>
> | candidate | this doc's proxy Δ | **confirmed GRU Δ** | folds | paired t (15 cells) |
> |---|---|---|---|---|
> | `ds_order` | −0.00360 | **−0.00010** | 3/5 | −1.02 (p≈0.32) |
> | `cart_backlog` | −0.00033 | **−0.00010** | 3/5 | −0.74 (p≈0.47) |
>
> Threshold was Δ < −0.00040 (2σ) or ≥4/5 folds. Neither reaches it; AUC is flat for both
> (+0.00008 / +0.00013), and §1b makes AUC the gate on rho. Logged `keep` at the user's
> instruction (e0193, e0194) — the evidence itself says no effect.
>
> **What this document got wrong, and it is the useful part.** The §16 mechanism argument —
> "the GRU cannot cheaply carry how long since the user last converted" — is now measured and
> **false**: it carries it fine, at 1/36th of the screened value. The asymmetry this file is
> built on (order-recency informative, activity/cart-recency nil) is a property of the
> **LightGBM n=15k proxy**, not of the GRU; on the GRU all three are nil. The proxy is a
> tabular model, and a long-memory scalar is worth far more to a tree ensemble than to a
> recurrence that accumulates one for free.
>
> **Consequence for future screens: this proxy has a measured inflation factor of ~36× on
> long-memory channels.** Its rankings may still be usable; its magnitudes are not. Do not
> quote a proxy Δ as a projected GRU Δ again — the "18σ" projection below was 36× optimistic.
>
> Causality verified on real data before both runs (§9 assert, t = 120/250/348, job 23871687).
> The redundancy penalty now has a third and fourth data point: e0110 (+27 ch, +0.00137, 0/5),
> e0114 (+5 ch, +0.00105, 0/5), and these two at +1 ch each. Isolated single channels do not
> *hurt* the way bundles do — they simply buy nothing.

Status of the feature screen for the **user_cv (causal GRU / seq) path**. The GRU runs the
`full` 85-channel variant (16 raw + 4-window rolling means + flag-rates + active) = e0141 base.
Its sharp optimum is documented: **surplus inputs are not free** — e0110 (+27 derived channels,
+0.00137, 0/5), e0114 (+5 rank channels, +0.00105, 0/5), e0142 (143-feature bundle incl.
calendar, worst since e0001). σ_noise for `nn_seq` = 0.00020 (2σ = 0.00040).

Because GRU runs are expensive, informativeness was measured on the **user-split LightGBM
proxy**: the same per-user-day `full` features, LightGBM under a user split (hash_fold),
burn-in 14, anchors cap at max_anchor, unseen-user RMSLE (baseline **1.75027 on n=15k**; the
real e0141 GRU = 1.74341 on full data — the proxy is a touch weaker, the sign is what matters).
A pure-noise channel moved the proxy by **−0.00005** (the causal proxy is not
overfit-starved), so a candidate is real on the proxy at ~5× that.

## Result: ONE candidate is clearly informative — `ds_order`

| candidate | ΔRMSLE (n=15k, user-split proxy) | verdict |
|---|---|---|
| **ds_order (days since last order)** | **−0.00360** | **informative (~72× proxy noise)** |
| cart_backlog_running | −0.00033 | ~6× proxy noise, marginal |
| ds_cart | −0.00005 | no effect |
| ds_active | +0.00008 | no effect |
| aov_daily | +0.00042 | hurts |
| conv_daily | +0.00005 | no effect |
| cart_per_srch_d | +0.00014 | no effect |

Everything else the `full` variant lacks (per-day compositions aov / conv / cart-per-search,
recency-of-cart / recency-of-activity, running cart-balance) is **nil or negative** — these
are the derived/composition channels the recurrence already tracks, consistent with the
e0110/e0114 redundancy-penalty evidence. **Do not add them.**

The asymmetry is the informative part: **recency-of-ACTIVITY and recency-of-CART are nil, but
recency-of-ORDER is −0.00360.** The GRU already carries "when did this user last do anything"
inside its hidden state (which is why dsa/dsc add nothing), but it cannot cheaply carry "how
long since the user last converted to an order" — that is the exact long-memory scalar the
recurrence would otherwise have to compress from a long sparse history. Lean signal, coherent
mechanism.

---

## Candidate (confirm-eligible) — `ds_order`: days since last order (`to_ord > 0`)

- **The change (one channel):** `ds_order[t] = t − last{t' ≤ t : to_ord[t'] > 0}` — capped at
  `T` (never) for users with no prior order; `log1p`-scaled. Causal by construction
  (`np.maximum.accumulate` running last-order day; never reads a future day).
- **Why the GRU lacks it:** the `full` variant has no recency channel of any kind; the only
  place `dso` has ever appeared is the `behav`/`extra` variants as part of a **bundled**
  feature set (e0142, 143 features) — and that bundle's failure was dominated by its
  calendar/day-of-year block, so **`dso` was never isolated.** That is the open question
  this screen answers in isolation: alone, it is the strongest single causal feature measured.
- **Measured informativeness (proxy, n=15k):** ΔRMSLE **−0.00360**. Placeholder for GRU σ:
  the proxy's own noise is ~−0.00005/control; against the GRU's 0.00020 σ_noise the delta
  would be **18σ** — but the proxy is not the GRU, and the redundancy penalty is real.
- **Prior against, stated plainly:** e0110/e0114 say tuned GRUs reject added channels. The
  difference here is that ds_order is (a) single-channel, (b) a long-memory scalar rather
  than a window/rank redundancy, and (c) isolated rather than bundled. It could still lose on
  the real GRU. That is precisely why it needs the confirm.
- **Suggested confirm parent:** e0141 (GRU, `full`, user hash_fold split, report unseen-user
  RMSLE **1.74341** + the in-population/AUC triplet). Add `dsc`-style channel to
  `src/usercv_features.py` as a toggleable block, train, compare. Δ<−0.00040 (2σ) or ≥4/5-fold
  win → keep; else graveyard it and note the redundancy penalty won again.

## Secondary (only if ds_order confirms clean) — `cart_backlog_running`

- **The change (one channel):** `cumsum(to_cart) − cumsum(to_ord)` = running stock of items
  selected but not yet converted, per day. Causal (cumulative over days ≤ t).
- **Measured (proxy):** ΔRMSLE **−0.00033**, ~6× proxy noise but one order of magnitude weaker
  than ds_order. Marginal; test only as a *separate* single-channel run after ds_order, and
  only if the ds_order confirm already won — a tuned GRU gives one channel per experiment is
  cheap, two is where e0110 started to die.

---

## Rejected on the causal proxy — do not re-test

- **aov_daily (+0.00042), conv_daily (+0.00005), cart_per_srch_d (+0.00014)** — per-day
  compositions: `0/0` for most users on most days, and the quantities are already present
  arithmetically inside the raw channels. This is the e0110 cluster re-incarnated; adding the
  computed form forces the GRU to fit redundant columns.
- **ds_cart (−0.00005), ds_active (+0.00008)** — recency-of-activity and recency-of-cart are
  already recoverable by the hidden state (hence nil), and cart is a weak conversion signal
  compared to orders. Confirms the asymmetry: **order-conversion recency is the signal;
  activity and cart recency are not.**
- **Calendar features** — re-confirmed dead by this line of thinking: any absolute-time input
  breaks the GRU's calendar-translation invariance (e0142). Never a candidate here.

---

## Why the GRU path gets one feature while the tabular path gets none

Tabular (FEATURES.md) measures *incremental rho on top of a saturated 665-feature GBDT* —
the margin for any single feature is genuinely ~0 and the screen found nothing above noise.
The GRU proxy measures *incremental unseen-user RMSLE on an 85-channel recurrent stack* —
a structurally leaner baseline with one obvious long-memory gap (days-since-order) that the
recurrence cannot cheaply reconstruct. Different baselines, different remaining signal: the
causal stack still has slack; the tabular stack does not. This asymmetry is measured, not
assumed — it is the whole reason the two files have different verdicts.
