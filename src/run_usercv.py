#!/usr/bin/env python
"""
CAUSAL_EXP.md, run as 5-fold CV split by user_id.

    python src/run_usercv.py --variant extra --folds 0 1 2 3 4 --seeds 3

Protocol, per the doc and the decisions of 2026-08-13:
  * split by user_id via the doc's deterministic md5 hash (salt "gmv-v1"), generalised from
    one 80/20 split to 5 disjoint folds: fold = floor(h * 5).
  * target = log1p(SUM of gmv over [t+1, t+30]) -- so the training loss IS RMSLE^2 and the
    validation number is directly readable as RMSLE.
  * last train/val anchor = 2026-01-14, the horizon-tail bound.
  * causal GRU emitting a prediction at every timestep (doc §5).
  * scaling stats fit on TRAIN USERS ONLY and passed to val (doc §3.2), flags left unscaled.
  * no fixed epoch count: up to 60 epochs, ReduceLROnPlateau, early stopping patience 8 on the
    unseen-user loss, best weights restored.

WHAT THIS NUMBER IS.  Unseen-user RMSLE at anchors whose calendar window the model trained on
through other users.  It is NOT comparable to experiments.csv, which scores the same users at a
FUTURE anchor.  Two known upward biases, both flagged and neither fixed:
  * every anchor from 2025-11-16 on has its target window inside the guaranteed-activity zone,
    where 100% of users are active by construction (DATA.md §4; measured +0.041 on the date
    split);
  * the doc's tmask has no population rule, so dormant user-days are scored too -- DATA.md §9
    measured that choice as ~0.10 optimistic. Reported here BOTH ways.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rho_decomp import auc                                                            # noqa
from usercv_features import (HORIZON, Raw, build_features, build_target, build_tmask,  # noqa
                             flag_channels, geo3_log, hash_fold, max_anchor, rolling_mean)


FROZEN_ANCHORS = {0: "2025-06-18", 1: "2025-07-18", 2: "2025-08-17",
                  3: "2025-09-16", 4: "2025-10-16"}

# DATA.md 4: every user is active in each of the three 30-day blocks ending 2026-02-13, so a
# target window touching [2025-11-16, ...] is scored on a population that is 100% active by
# construction (+0.041 optimistic on the date split).  An anchor is "guard-clean" iff its whole
# target window [a+1, a+30] lands before that date.
GUARD_START = date(2025, 11, 16)


def log(m: str) -> None:
    print(m, flush=True)


class CausalXformer(nn.Module):
    """Same contract as GRUForecaster: (B, L, F) -> (B, L), strictly causal.

    ALiBi, no absolute positional embedding: the test anchor sits 120 days past the last clean
    training anchor, which is where an absolute encoding extrapolates worst. Reuses seqnet's
    backbone so both sequence families share one audited causal implementation.
    """

    def __init__(self, n_feat: int, hidden: int = 128, layers: int = 4, heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        import seqnet
        self.inp = nn.Sequential(nn.Linear(n_feat, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.bb = seqnet.XformerBackbone(hidden, n_layers=layers, n_heads=heads,
                                         dropout=dropout, max_len=512)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(x).transpose(1, 2)          # (B, d, L) for the backbone
        return self.head(self.bb(h).transpose(1, 2)).squeeze(-1)


class GRUForecaster(nn.Module):
    """Doc §5, verbatim.  `cell="lstm"` swaps the recurrent cell and nothing else.

    §1k ranked LSTM 0.0029 behind the GRU on the frozen-fold path -- but on TWO search trials
    against the GRU's THIRTEEN, so that gap is a best-of-2 vs best-of-13 comparison, not a
    measured architecture difference.  Cheap to settle now that the width axis is understood.
    """

    def __init__(self, n_feat: int, hidden: int = 128, layers: int = 2, dropout: float = 0.1,
                 cell: str = "gru"):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_feat, hidden), nn.LayerNorm(hidden), nn.GELU())
        RNN = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = RNN(hidden, hidden, num_layers=layers, batch_first=True,
                       dropout=dropout if layers > 1 else 0.0)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # (B, L, F) -> (B, L)
        h, _ = self.rnn(self.inp(x))
        return self.head(self.norm(h)).squeeze(-1)


def masked_rmse(pred: torch.Tensor, y: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(((pred - y) ** 2 * m).sum() / m.sum())


def within_day_corr(pred: torch.Tensor, targ: torch.Tensor, m: torch.Tensor,
                    eps: float = 1e-6) -> torch.Tensor:
    """Mask-aware Pearson correlation centred WITHIN each calendar day (column).

    Same estimator as run_seq.py's `within_day_corr` (IDEAS.md I2/I19), reproduced here rather
    than imported to avoid pulling seqdata/seqnet into the causal path. Layouts match: both
    are (users, days).

    WHY THIS IS THE RIGHT TRAINING SIGNAL. After the affine calibration 1b applies to every
    submission, the score depends on the prediction only through rho -- MSE additionally pays
    for a level and a spread that calibration then throws away. And it must be the WITHIN-anchor
    correlation, which 1r proves: pooling across days credits a model for knowing December
    outranks July, and the competition scores a single anchor. Centring each day on its own
    masked mean makes the term invariant to a per-day affine shift -- exactly the freedom
    calibration has.
    """
    cnt = m.sum(0)
    denom = cnt.clamp(min=1.0)
    pmean = (pred * m).sum(0) / denom
    ymean = (targ * m).sum(0) / denom
    vcol = (cnt > 1).to(pred.dtype)[None, :]
    pc = (pred - pmean) * m * vcol
    yc = (targ - ymean) * m * vcol
    return (pc * yc).sum() / torch.sqrt((pc * pc).sum() * (yc * yc).sum() + eps)


class Corr:
    """Streaming Pearson correlation over a mask, so rho can be reported per block.

    WHY THIS EXISTS. The first version of the forward reading reported raw RMSLE only, and
    the e0215/e0216 pair showed that is misleading: `extra` beat `full` by -0.00326 raw at
    the forward anchor while its rho gain was only +0.00049 -- 67% of the apparent gain was
    LEVEL, which EXPERIMENTS.md 1b proves is free to fix at submission time. A protocol that
    scores a calendar block on raw RMSLE will keep promoting level artefacts. rho, and the
    perfectly-calibrated score sd_L*sqrt(1-rho^2), are the honest statistics.
    """

    # WARNING: POOLED rho ACROSS ANCHORS IS THE WRONG STATISTIC, measured 2026-08-22.
    # e0234 scored +0.00079 pooled over 91 forward anchors and -0.00009 WITHIN the single
    # forward anchor the frozen folds score. The difference is between-anchor level: pooling
    # rewards a model for knowing that December outranks July, and 1b proves level is free at
    # submission time. The competition scores ONE anchor, so only the within-anchor component
    # is real. `rho_within()` below removes the between-anchor part by centring per day.

    def __init__(self, n_days: int = 0):
        self.n = self.sx = self.sy = self.sxx = self.syy = self.sxy = 0.0
        # per-day accumulators for the within-anchor (pooled-within-group) correlation
        self.T = n_days
        if n_days:
            z = lambda: np.zeros(n_days, np.float64)                        # noqa: E731
            self.dn, self.dx, self.dy = z(), z(), z()
            self.dxx, self.dyy, self.dxy = z(), z(), z()

    def add(self, x: torch.Tensor, y: torch.Tensor, m: torch.Tensor) -> None:
        # float64 deliberately: the reduction runs over ~2.4e8 masked user-days with a mean
        # near 2.3, so sxx ~ 2.6e9 while sxx - sx^2/n ~ 1.3e9. In float32 (7 digits) that
        # cancellation eats the answer; in float64 it is exact to 15 digits. The cost is a
        # rounding error next to the forward pass.
        mf = m.double()
        xf, yf = x.double() * mf, y.double() * mf
        self.n += float(mf.sum())
        self.sx += float(xf.sum()); self.sy += float(yf.sum())
        self.sxx += float((xf * xf).sum()); self.syy += float((yf * yf).sum())
        self.sxy += float((xf * yf).sum())
        if self.T:                                   # per-DAY sums, for the within-anchor rho
            self.dn += mf.sum(0).double().cpu().numpy()
            self.dx += xf.sum(0).cpu().numpy(); self.dy += yf.sum(0).cpu().numpy()
            self.dxx += (xf * xf).sum(0).cpu().numpy()
            self.dyy += (yf * yf).sum(0).cpu().numpy()
            self.dxy += (xf * yf).sum(0).cpu().numpy()

    def rho_within(self) -> float:
        """Pooled-WITHIN-anchor correlation: every day centred on its own mean first.

        This is the statistic the competition actually scores -- it predicts ONE anchor, so a
        model gets no credit for knowing that one calendar date outranks another. `rho()`
        above pools across days and does give that credit, which is how e0234 read +0.00079
        pooled while being -0.00009 within-anchor.
        """
        if not self.T:
            return float("nan")
        ok = self.dn >= 2
        if not ok.any():
            return float("nan")
        n, sx, sy = self.dn[ok], self.dx[ok], self.dy[ok]
        cxx = (self.dxx[ok] - sx ** 2 / n).sum()
        cyy = (self.dyy[ok] - sy ** 2 / n).sum()
        cxy = (self.dxy[ok] - sx * sy / n).sum()
        if cxx <= 0 or cyy <= 0:
            return float("nan")
        return float(cxy / np.sqrt(cxx * cyy))

    def star_within(self) -> float:
        r = self.rho_within()
        if not np.isfinite(r):
            return float("nan")
        ok = self.dn >= 2
        n, sy = self.dn[ok], self.dy[ok]
        sd = np.sqrt(max(0.0, (self.dyy[ok] - sy ** 2 / n).sum()) / n.sum())
        return float(sd * np.sqrt(1 - r * r))

    def rho(self) -> float:
        if self.n < 2:
            return float("nan")
        cx = self.sxx - self.sx ** 2 / self.n
        cy = self.syy - self.sy ** 2 / self.n
        if cx <= 0 or cy <= 0:
            return float("nan")
        return (self.sxy - self.sx * self.sy / self.n) / np.sqrt(cx * cy)

    def sd_y(self) -> float:
        return float(np.sqrt(max(0.0, self.syy - self.sy ** 2 / self.n) / self.n))

    def star(self) -> float:
        """RMSLE at the optimal affine calibration = sd_L*sqrt(1-rho^2)."""
        r = self.rho()
        return float("nan") if not np.isfinite(r) else self.sd_y() * float(np.sqrt(1 - r * r))


def calendar_masks(M: np.ndarray, T: int, cut: int, clean_max: int):
    """Split a tmask along the CALENDAR axis (see the block comment in main()).

    Returns (in_calendar, forward, forward_guard_clean, oof).

      in_calendar  t <= cut                 trained on, and validated on unseen users
      forward      t >= cut + HORIZON       unseen users AND unseen calendar
      fwd_clean    forward, and the target window ends before the guaranteed-activity zone
      oof          in_calendar | forward    everything except the embargo

    The embargo is the whole point: a training anchor t <= cut supervises calendar days up
    to cut + HORIZON, so an anchor inside (cut, cut + HORIZON) has a target window whose days
    were already supervised through OTHER users and is not forward at all.
    """
    t = np.arange(T)[None, :]
    m_in = M & (t <= cut)
    m_fw = M & (t >= cut + HORIZON)
    m_fc = m_fw & (t <= clean_max)
    return m_in, m_fw, m_fc, m_in | m_fw


def month_stats(X: np.ndarray, tr_u: np.ndarray, is_flag: np.ndarray, win: int = 30,
                floor: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel standardise stats from the TRAILING `win` days, pooled over train users.

    Contrast with the global path (`sub.mean(axis=(0, 1))`): that pools a user's whole 409-day
    history into one scalar per channel, so a GMV of 30 in January and 30 in November are
    normalised identically even though the fold-level target level drifts ~15% over the calendar
    (DATA.md §6.2).  Here each position t standardises by the last `win` days' pooled moments
    over train users -- "how much is this user spending relative to the recent cohort" instead
    of "how much in absolute terms."  Causal: the window is `[t-win+1, t]`, all days <= t, and
    uses the same clipped-denominator rolling_mean the feature builder does, so the head of
    each user's history degrades gracefully to the shorter available window.

    Returns mu, sd shaped (T, F) for the month fit (broadcast against (B, L, F): trailing F lines
    up, then T lines up with L) or (F,) for a global fit.  Flags stay unscaled (mu=0, sd=1)
    exactly as in the global path.
    """
    sub = X[tr_u[::37]].astype(np.float32)              # same subsample as the global path
    day1 = sub.mean(axis=0).T                            # (F, T) pooled mean over sampled users
    day2 = (sub ** 2).mean(axis=0).T                     # (F, T) pooled second moment
    sd_global = np.maximum(sub.std(axis=(0, 1)), floor)  # (F,) for reference
    m = rolling_mean(day1, win)
    sd = np.sqrt(np.maximum(rolling_mean(day2, win) - m ** 2, 0.0))
    # a degenerate trailing window (e.g. a mostly-zero channel right after a dormant stretch)
    # would inflate sd toward the floor; clamp so no position amplifies more than the pooled
    # global spread of the channel.
    sd = np.maximum(sd, floor)
    sd = np.minimum(sd, sd_global[:, None] * 100.0)
    mu = m
    mu[is_flag] = 0.0; sd[is_flag] = 1.0
    return mu.T.astype(np.float32), sd.T.astype(np.float32)   # (T, F)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    choices=["gmv_only", "full", "full_dso", "full_backlog", "full_popidx",
                             "extra", "extra_nocal", "extra_nodoy", "behav"])
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--sched", default="plateau", choices=["plateau", "cosine"])
    ap.add_argument("--feat-drop", type=float, default=0.0,
                    help="probability of zeroing a whole feature CHANNEL per batch")
    ap.add_argument("--model", default="gru", choices=["gru", "transformer", "lstm"])
    ap.add_argument("--mixup", default="none", choices=["none", "naive", "class"],
                    help="synthetic users by interpolating pairs; 'class' only mixes "
                         "timesteps where both sources agree on buy/no-buy")
    ap.add_argument("--mixup-alpha", type=float, default=0.2)
    ap.add_argument("--residual", action="store_true",
                    help="predict L - log1p(geo3) and add the baseline back (per-user offset)")
    ap.add_argument("--burn-in", type=int, default=14)
    ap.add_argument("--fixed-epochs", action="store_true",
                    help="§1j's PRESCRIBED FIX: disable early stopping, train exactly "
                         "--epochs, and score ONCE at the final epoch. The default protocol "
                         "reports the MIN of a variable-length early-stopped curve, which is "
                         "biased low by ~sigma*sqrt(2 ln N) -- so a config that trains longer "
                         "wins on evaluation count alone. e0266 (d64) averaged 22.4 best-epochs "
                         "against the d128 control's 13.5, so its -0.00079 needs this control.")
    ap.add_argument("--loss", default="mse", choices=["mse", "corr", "mix"],
                    help="mse = the historical masked MSE on log1p (default, unchanged). "
                         "corr = 1 - within-anchor Pearson rho, the quantity 1b proves is the "
                         "ONLY thing that scores after calibration. mix = MSE + w*(1-rho), "
                         "which keeps MSE's scale anchor so RMSLE stays interpretable.")
    ap.add_argument("--corr-weight", type=float, default=1.0,
                    help="w in the `mix` loss")
    ap.add_argument("--tag-suffix", default=None,
                    help="append this to the artefact tag (oof/, runs/, reports/). "
                         "REVIEW_NOTES A3 asked for this and it was never implemented; use it "
                         "to give every run its own exp_id-named artefacts so a re-run of an "
                         "identical config cannot overwrite the reference it is compared to.")
    ap.add_argument("--pop-train", action="store_true",
                    help="restrict TRAINING to in-population user-days (>=1 active day in "
                         "[t-29, t]), matching run_seq.py and the test population. "
                         "Validation is unchanged, so numbers stay comparable.")
    ap.add_argument("--train-cap", default=None, metavar="YYYY-MM-DD",
                    help="Cap the TRAINING anchor grid at this date while scoring unseen users "
                         "ONLY on guard-clean anchors (t <= 288 = 2025-10-16). This is the one "
                         "question the project recorded as unanswerable by CV, and it is "
                         "answerable HERE and not on the date-fold paths: the split is by USER, "
                         "so a model trained through the guard zone can be scored at an EARLIER "
                         "clean anchor on users it never saw. Users are disjoint, the "
                         "architecture is causal per-user, and within-anchor rho is invariant to "
                         "the shared calendar level, so training past the scoring anchor cannot "
                         "leak into the statistic. Pass it to BOTH arms (2026-01-14 = the "
                         "current default grid, 2025-10-16 = retracted) so both score the same "
                         "block and the only difference is the training cap. Distinct from "
                         "--t-cut, which always scores AFTER the cut and therefore cannot "
                         "express this comparison.")
    ap.add_argument("--t-cut", default=None, metavar="YYYY-MM-DD",
                    help="add a CALENDAR split on top of the user split. Training and the "
                         "existing in-calendar validation use anchors t <= t_cut; a second "
                         "FORWARD score is reported on the same unseen users at anchors "
                         "t >= t_cut + 30, whose target windows no training target ever "
                         "covered. Omit for the historical behaviour (identical numbers).")
    ap.add_argument("--month-norm", action="store_true",
                    help="standardize each position with per-channel stats from the TRAILING 30 "
                         "days (pooled over train users) instead of one scalar over the full "
                         "panel. Causal: only days <= t are used.")
    ap.add_argument("--no-save-weights", action="store_true",
                    help="skip checkpointing; weights are saved by default")
    # ---------------------------------------------------------------- training-regime axis
    # Everything below defaults to the historical behaviour, so every logged run reproduces.
    # The axis exists because of a structural fact nobody had written down: this loop batches
    # over USERS and supervises EVERY valid day of the sampled users in one masked mean, so
    # (a) there is no curriculum and no date sampling of any kind, and (b) every calendar day
    # carries equal gradient weight -- the position that actually matters at submission (the
    # final anchor) gets ~1/365th of the loss, the same as a day in April 2025.
    ap.add_argument("--day-weight-halflife", type=float, default=0.0, metavar="DAYS",
                    help="exponential RECENCY weighting of the per-day loss: a day t gets "
                         "w = 0.5 ** ((last_anchor - t)/H). 0 (default) = uniform = unchanged. "
                         "The weighted mean is scale-invariant in w, so the loss stays on the "
                         "same scale and lr/epoch settings remain comparable. NOT the killed "
                         "e0247 buyer re-weighting: that weighted rows by a quantity correlated "
                         "with the target and so changed the elicited functional (w*=1, "
                         "monotone worse). A weight that depends only on the DATE leaves "
                         "E[L|x] unchanged for fixed x; it reallocates finite capacity. "
                         "Nearest prior evidence is the GBDT anchor decay (hl 60d/180d, flat) "
                         "-- but that grid spans 6.5 months against this path's 12, so it is "
                         "not a direct test.")
    ap.add_argument("--curriculum", default="none", choices=["none", "expand", "slide"],
                    help="epoch-dependent day window, oldest-first. "
                         "expand = left edge FIXED, right edge grows -- fresh data is ADDED "
                         "each epoch and old data is never removed (supervised cells grow). "
                         "slide = CONSTANT-WIDTH window whose both edges move right together, "
                         "so old data is dropped as fresh arrives and the supervised-cell "
                         "count per batch is constant across epochs. "
                         "NOTE: the number of BATCHES per epoch is n_train_users/batch and is "
                         "already constant in every arm, because the batch axis is users; what "
                         "`slide` additionally holds constant is the WORK PER BATCH.")
    ap.add_argument("--curr-start", type=float, default=0.5, metavar="FRAC",
                    help="initial window width as a fraction of the full day range "
                         "(also the constant width for --curriculum slide)")
    ap.add_argument("--curr-epochs", type=int, default=0, metavar="E",
                    help="epochs over which the schedule completes; 0 (default) = --epochs, "
                         "i.e. the schedule spans the whole run")
    ap.add_argument("--curr-shuffle", action="store_true",
                    help="ORDERING NULL for the curriculum arms (the e0214 discipline): keep "
                         "the exact same multiset of per-epoch windows but present them in a "
                         "fixed random order, destroying the oldest->freshest progression "
                         "while holding the per-epoch data volume identical. If a curriculum "
                         "arm and its shuffled twin score the same, the ORDER carries nothing "
                         "and only the volume schedule matters -- which is the difference "
                         "between 'curriculum learning works here' and 'we trained on less "
                         "data for a while'. Deterministic (seed 12345), so the shuffled "
                         "schedule is identical across folds and seeds.")
    ap.add_argument("--day-sample", type=float, default=1.0, metavar="Q",
                    help="Bernoulli keep-probability applied to the day mask, RESAMPLED EVERY "
                         "STEP -- i.e. sample (user, date) cells rather than whole users, so "
                         "the gradient is no longer a deterministic function of the user "
                         "permutation. 1.0 (default) = keep all days = unchanged. "
                         "⚠ Why not a flat (user,date) index: a causal GRU needs the whole "
                         "prefix to emit a prediction at day t, so supervising ONE cell still "
                         "costs a full-sequence forward pass. Drawing `batch` flat pairs would "
                         "cost the same forward as today's `batch` users while supervising "
                         "~365x fewer cells -- strictly worse signal per unit compute. "
                         "Sampling users and then thinning their days gives (user,date)-level "
                         "randomness at IDENTICAL cost, which is what this flag does.")
    args = ap.parse_args()

    if args.day_weight_halflife > 0:
        assert args.loss == "mse", ("--day-weight-halflife folds a weight into the MSE mask; "
                                    "combining it with --loss would be two changes (§4.1)")
    if args.curriculum != "none":
        assert 0 < args.curr_start <= 1.0, "--curr-start is a fraction in (0, 1]"
    assert 0 < args.day_sample <= 1.0, "--day-sample is a probability in (0, 1]"

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"\n=== CAUSAL_EXP user-split CV : variant={args.variant} model={args.model}"
        f"{' mixup=' + args.mixup if args.mixup != 'none' else ''} ===")
    log(f"    target=sum(30d) log1p | up to {args.epochs} epochs, lr {args.lr}, "
        f"patience {args.patience}, {args.seeds} seed(s), device={device}")

    raw = Raw()
    last_anchor = max_anchor(raw)
    log(f"    last train/val anchor {raw.day(last_anchor)} "
        f"(target window {raw.day(last_anchor + 1)} .. {raw.day(last_anchor + HORIZON)})")
    Y = build_target(raw, "sum")
    M = build_tmask(raw, last_anchor, burn_in=args.burn_in, trim_to_first_seen=True)
    Xn, names = build_features(raw, args.variant)
    is_flag = flag_channels(names)

    # secondary mask: the frozen population rule (>=1 active day in [t-29, t]), so the same run
    # also reports the number on the population the competition actually scores.
    csa = np.concatenate([np.zeros((raw.n, 1), np.int32),
                          np.cumsum(raw.active > 0, 1, dtype=np.int32)], 1)
    lo = np.maximum(np.arange(raw.T) - 29, 0)
    POP = (csa[:, 1:] - csa[:, lo]) > 0
    del csa

    # ---------------------------------------------------------------- the calendar split
    # A pure user split cannot see calendar extrapolation: val users share the training
    # users' calendar, so a day-of-year feature is INTERPOLATION at validation time and is
    # scored as skill.  Measured, on the leaderboard: `extra` beats `full` by -0.00080 (5/5
    # folds) on this CV and is -0.00262 rho WORSE on the test anchor (e0142 1.6785 vs e0141
    # 1.6488).  The instrument returned the wrong SIGN, not merely a null.
    #
    # `--t-cut A` fixes that by adding a calendar dimension to the split:
    #   train + in-calendar val : anchors t <= A
    #   embargo                 : anchors (A, A+30)  -- a training anchor t <= A supervises
    #                             calendar days up to A+30, so anything below that is not
    #                             forward at all
    #   FORWARD val             : anchors t >= A + 30, on the SAME unseen users
    # Both readings come out of one run, so their divergence is a paired measurement of
    # exactly the failure mode above.
    cut = last_anchor if args.t_cut is None else raw.idx(date.fromisoformat(args.t_cut))
    clean_max = raw.idx(GUARD_START) - 1 - HORIZON      # last guard-clean anchor (= 288)
    M_in, M_fw, M_fc, M_oof = calendar_masks(M, raw.T, cut, clean_max)
    if args.t_cut is None:
        assert M_in.sum() == M.sum() and not M_fw.any(), "t-cut off must be a no-op"
    else:
        log(f"    CALENDAR SPLIT --t-cut {raw.day(cut)} (anchor {cut})")
        log(f"      train + in-calendar val : anchors <= {raw.day(cut)}   "
            f"({M_in.sum():,} user-days)")
        log(f"      embargo (not scored)    : anchors {raw.day(cut + 1)} .. "
            f"{raw.day(cut + HORIZON - 1)}")
        log(f"      FORWARD val             : anchors >= {raw.day(cut + HORIZON)}   "
            f"({M_fw.sum():,} user-days)")
        log(f"      forward AND guard-clean : anchors <= {raw.day(clean_max)}   "
            f"({M_fc.sum():,} user-days)"
            + ("  <- EMPTY, forward score is guard-zone only" if not M_fc.any() else ""))
        assert not (M_in & M_fw).any(), "in-calendar and forward blocks overlap"

    # --train-cap: decouple the TRAINING mask from the SCORING mask. Everything else in this
    # file ties them together (Mg serves both), which is precisely why the guard-zone training
    # question looked unanswerable. Scoring is pinned to the guard-clean block for BOTH arms,
    # so the comparison is paired and the only difference is how far training runs.
    if args.train_cap is not None:
        assert args.t_cut is None, "--train-cap and --t-cut are different splits; do not combine"
        assert not args.pop_train, ("--pop-train derives its training mask from the SCORING mask "
                                    "(Pg = M_in & POP); combining it with --train-cap would "
                                    "silently train on the clean block in both arms")
        tcap = raw.idx(date.fromisoformat(args.train_cap))
        tt = np.arange(raw.T)[None, :]
        M_train = M & (tt <= tcap)
        M_score = M & (tt <= clean_max)
        assert M_score.any(), "guard-clean scoring block is empty"
        n_dirty = int((M_train & (tt > clean_max)).sum())
        log(f"    --train-cap {raw.day(tcap)} (anchor {tcap})")
        log(f"      TRAIN on anchors <= {raw.day(tcap)}   ({M_train.sum():,} user-days"
            f"{f', {n_dirty:,} of them GUARD-ZONE contaminated' if n_dirty else ', all guard-clean'})")
        log(f"      SCORE unseen users on anchors <= {raw.day(clean_max)}   "
            f"({M_score.sum():,} user-days)  <- identical in both arms")
        M_in, M_oof = M_score, M_score          # scoring/ES/OOF all on the clean block
    else:
        M_train = M_in

    fold_of = hash_fold(raw.users)
    log(f"    user folds: {np.bincount(fold_of).tolist()}  (md5 salt 'gmv-v1')")
    # One tag for every artefact this run writes (checkpoints, oof, report). Computed BEFORE
    # the fold loop because the checkpoints are written inside it.
    ck_tag = args.variant if args.model == "gru" else f"{args.variant}_{args.model}"
    if args.residual:
        ck_tag += "_resid"
    if args.mixup != "none":
        ck_tag += f"_mix{args.mixup}"
    if args.month_norm:
        ck_tag += "_mnorm"
    if args.pop_train:
        ck_tag += "_poptrain"
    if args.fixed_epochs:
        ck_tag += "_fix"
    if args.loss != "mse":
        ck_tag += f"_{args.loss}" + (f"{args.corr_weight:g}" if args.loss == "mix" else "")
    # ⚠ HYPERPARAMETERS MUST BE IN THE TAG. REVIEW_NOTES A3 and the 2026-08-20 incident that
    # destroyed the usercv_extra OOF both say so, and on 2026-08-22 three dropout arms were
    # launched concurrently that would ALL have written oof/usercv_full.parquet -- racing each
    # other and clobbering the e0195 baseline (rule 10). Only non-default values are appended,
    # so every previously-logged tag is unchanged and reproducible.
    for flag, val, dflt in (("h", args.hidden, 128), ("l", args.layers, 2),
                            ("do", args.dropout, 0.1), ("lr", args.lr, 1e-3),
                            ("wd", args.wd, 1e-5), ("bs", args.batch, 256),
                            ("bi", args.burn_in, 14), ("ep", args.epochs, 60)):
        if val != dflt:
            ck_tag += f"_{flag}{val:g}".replace("-", "m")
    if args.t_cut is not None:
        # REVIEW_NOTES A3: a tag that does not name the thing being varied is not a name --
        # the `--month-norm` screen overwrote its own baseline's OOF for exactly this reason.
        ck_tag += f"_fwd{args.t_cut.replace('-', '')}"
    if args.train_cap is not None:
        ck_tag += f"_tcap{args.train_cap.replace('-', '')}"
    # Training-regime axis: same rule as above -- only non-default values are appended, so
    # every previously-logged tag is byte-identical, and no two arms of the sweep can collide.
    if args.day_weight_halflife > 0:
        ck_tag += f"_dwhl{args.day_weight_halflife:g}"
    if args.curriculum != "none":
        ck_tag += f"_curr{args.curriculum}{args.curr_start:g}"
        if args.curr_epochs:
            ck_tag += f"e{args.curr_epochs}"
        if args.curr_shuffle:
            ck_tag += "shuf"
    if args.day_sample < 1.0:
        ck_tag += f"_ds{args.day_sample:g}"
    if args.tag_suffix:
        ck_tag += f"_{args.tag_suffix}"
    log(f"    standardisation: "
        f"{'TRAILING-30d per-day stats (--month-norm)' if args.month_norm else 'GLOBAL pooled stats'}"
        f" | artefact tag '{ck_tag}'")

    Bnp = geo3_log(raw) if args.residual else np.zeros_like(Y)
    Xg = torch.from_numpy(Xn).to(device)
    Yg = torch.from_numpy(Y).to(device)          # absolute log target, kept for the AUC label
    Bg = torch.from_numpy(Bnp).to(device)
    Tg = Yg - Bg                                 # what the network actually fits
    Mg = torch.from_numpy(M_in).to(device)       # train + in-calendar val (== M when off)
    Pg = torch.from_numpy(M_in & POP).to(device)
    # `--pop-train` restricts the TRAINING signal to in-population user-days, leaving
    # validation untouched so every reported number stays comparable to e0141/e0195.
    #
    # WHY. run_seq.py:160 trains on `popg` -- in-population days only. run_usercv.py trains on
    # every masked day. The two sequence paths therefore disagree about their own training
    # population, and the causal one is the odd path out despite carrying the largest blend
    # weight (0.42). Measured on the real panel: 6.2% of the causal path's 80,899,560
    # training user-days are DORMANT (no active day in [t-29, t]), while at the test anchor
    # 100.0% of the 250,000 scored users are in-population by construction. So a slice of the
    # training signal describes a population that does not exist at scoring time.
    TRg = torch.from_numpy(M_train).to(device)   # == M_in unless --train-cap is set
    TRAINg = Pg if args.pop_train else TRg
    if args.train_cap is not None:
        log(f"    TRAINING mask decoupled from the scoring mask: "
            f"{int(M_train.sum()):,} train user-days vs {int(M_in.sum()):,} scored")
    if args.pop_train:
        log(f"    POPULATION-MATCHED TRAINING: {int((M_in & POP).sum()):,} of "
            f"{int(M_in.sum()):,} user-days kept "
            f"({100 * (M_in & POP).sum() / max(1, M_in.sum()):.1f}%); "
            f"validation mask UNCHANGED")
    Fg = torch.from_numpy(M_fw).to(device)       # forward block
    Cg = torch.from_numpy(M_fc).to(device)       # forward block, guard-clean subset

    # ------------------------------------------------------------- training-regime schedule
    # All three knobs act on the DAY axis of the training mask only. Validation, the scoring
    # mask, the OOF block and the population rule are untouched, so every reported number
    # stays paired with the control.
    dw_g = None
    if args.day_weight_halflife > 0:
        tt_f = np.arange(raw.T, dtype=np.float32)
        w_np = np.power(0.5, (last_anchor - tt_f) / args.day_weight_halflife).astype(np.float32)
        w_np[tt_f > last_anchor] = 0.0           # never supervised anyway; keep it explicit
        dw_g = torch.from_numpy(w_np).to(device)[None, :]
        eff = float(w_np[: last_anchor + 1].sum())
        log(f"    DAY RECENCY WEIGHTING: half-life {args.day_weight_halflife:g}d, "
            f"w({raw.day(last_anchor)})=1.000 -> w({raw.day(0)})={w_np[0]:.2e}; "
            f"effective day count {eff:.1f} of {last_anchor + 1} "
            f"({100 * eff / (last_anchor + 1):.1f}%)")
    else:
        log("    day weighting: UNIFORM (historical)")

    curr_E = args.curr_epochs or args.epochs
    tt_g = torch.arange(raw.T, device=device)[None, :]

    # ⚠ THE SCHEDULE IS IN SUPERVISED-CELL MASS, NOT IN DAYS, AND THE SMOKE IS WHY.
    # A first version stepped the window by calendar days. The e0930 smoke measured the
    # per-epoch cell counter it also added and found `slide` -- a CONSTANT 190-day window --
    # still drifting 14.31M -> 18.10M cells (+26%) across three epochs. The mask is
    # `t >= first_active + 14`, and users enter the panel at different dates, so a 190-day
    # window early in the calendar holds far fewer valid user-days than a late one. A
    # curriculum arm whose supervision volume grows 26% is confounded with exactly the thing
    # it is supposed to isolate. Stepping in mass fixes it: `slide` now holds the number of
    # supervised cells per epoch constant BY CONSTRUCTION, which is what "the number of
    # batches stays constant across epochs" has to mean once the day axis is not uniform.
    cells_per_day = M_train.sum(axis=0).astype(np.float64)
    cells_per_day[last_anchor + 1:] = 0.0
    cum_cells = np.concatenate([[0.0], np.cumsum(cells_per_day)])   # cum[t] = cells in [0, t)
    total_cells = float(cum_cells[last_anchor + 1])

    def _hi_for_mass(lo: int, mass: float) -> int:
        """Smallest hi with cells in [lo, hi] >= mass (clamped to last_anchor)."""
        j = int(np.searchsorted(cum_cells, cum_cells[lo] + mass, side="left"))
        return int(min(max(j - 1, lo), last_anchor))

    def curr_window(ep: int) -> tuple[int, int]:
        """Inclusive day-index bounds of the training window at epoch `ep`.

        Both schedules start on the OLDEST data and move toward the freshest -- the direction
        that makes `expand` "add increasingly fresh data". `expand` pins the left edge so old
        data is never removed; `slide` moves both edges while holding the supervised-cell mass
        constant, so old data is dropped as fresh arrives at constant work per epoch.
        """
        if args.curriculum == "none":
            return 0, last_anchor
        f = 1.0 if curr_E <= 1 else min(1.0, ep / (curr_E - 1))   # 0 at ep 0, 1 at ep E-1
        m0 = args.curr_start * total_cells                        # initial (and slide) mass
        if args.curriculum == "expand":
            return 0, _hi_for_mass(0, m0 + f * (total_cells - m0))    # left edge FIXED
        # slide: the last window must END at last_anchor, so lo_max is where the remaining
        # mass is exactly m0; interpolate lo between 0 and lo_max in mass, not in days.
        lo_max = int(np.searchsorted(cum_cells, total_cells - m0, side="left"))
        lo_max = int(min(max(lo_max, 0), last_anchor))
        lo = int(np.searchsorted(cum_cells, f * cum_cells[lo_max], side="left"))
        lo = int(min(max(lo, 0), lo_max))
        return lo, (last_anchor if f >= 1.0 else _hi_for_mass(lo, m0))

    # Ordering null: permute WHICH schedule position each epoch sees, leaving the multiset of
    # windows (and therefore the total data volume over the run) exactly unchanged.
    curr_order = list(range(args.epochs))
    if args.curr_shuffle:
        assert args.curriculum != "none", "--curr-shuffle needs a curriculum to shuffle"
        np.random.default_rng(12345).shuffle(curr_order)
    _curr_window_raw = curr_window

    def curr_window(ep: int) -> tuple[int, int]:            # noqa: F811 -- deliberate wrapper
        return _curr_window_raw(curr_order[ep] if ep < len(curr_order) else ep)

    if args.curriculum != "none":
        if args.curr_shuffle:
            log(f"    ⚠ ORDERING NULL (--curr-shuffle): same windows, order permuted "
                f"{curr_order[:6]}... -- per-epoch volume identical, progression destroyed")
        log(f"    CURRICULUM '{args.curriculum}': start-mass {args.curr_start:g} of the "
            f"{total_cells / 1e6:.1f}M supervised cells, schedule completes at epoch {curr_E}")
        for ep_ in sorted({0, (curr_E - 1) // 2, curr_E - 1}):
            lo_, hi_ = curr_window(ep_)
            mass = (cum_cells[hi_ + 1] - cum_cells[lo_]) / 1e6
            log(f"      epoch {ep_:<4d}: days [{lo_:>3d}..{hi_:>3d}] = {hi_ - lo_ + 1:>3d} days, "
                f"{mass:6.2f}M cells ({100 * mass * 1e6 / total_cells:5.1f}%)  "
                f"{raw.day(lo_)} .. {raw.day(hi_)}")
        log(f"      batches/epoch is CONSTANT in every arm (the batch axis is users); "
            f"'slide' additionally holds the CELL MASS constant -- the day width does NOT, "
            f"because users enter the panel at different dates (measured in the e0930 smoke: "
            f"a fixed 190-day window drifted +26% in cells)")
    else:
        log("    curriculum: NONE (historical -- every valid day supervised every epoch)")

    if args.day_sample < 1.0:
        log(f"    (USER,DATE) SAMPLING: Bernoulli keep-prob {args.day_sample:g} on the day "
            f"mask, resampled EVERY STEP (not per epoch)")
    else:
        log("    day sampling: OFF (historical -- all valid days of the sampled users)")
    Og = torch.from_numpy(M_oof).to(device)      # OOF selection (== M when off)
    has_fwd = bool(M_fw.any())
    has_fc = bool(M_fc.any())
    if device == "cuda":
        log(f"    gpu resident {torch.cuda.memory_allocated() / 1e9:.1f} GB  "
            f"({len(names)} features)")

    frozen = {k: raw.idx(date.fromisoformat(v)) for k, v in FROZEN_ANCHORS.items()}
    oof_rows = []
    results = []
    for k in args.folds:
        tr_u = np.flatnonzero(fold_of != k)
        va_u = np.flatnonzero(fold_of == k)
        assert not set(tr_u) & set(va_u), "user appears in both splits"
        # §3.2: mu/sigma from TRAIN USERS only; flags unscaled.
        if args.month_norm:
            mu, sd = month_stats(Xn, tr_u, is_flag)          # (T, F) -- trailing 30d, causal
        else:
            sub = Xn[tr_u[::37]].astype(np.float32)
            mu = sub.mean(axis=(0, 1)); sd = np.maximum(sub.std(axis=(0, 1)), 1e-3)
            mu[is_flag] = 0.0; sd[is_flag] = 1.0
            del sub
        mu_g = torch.from_numpy(mu).to(device)
        sd_g = torch.from_numpy(sd).to(device)
        tr_g = torch.from_numpy(tr_u).to(device)
        va_g = torch.from_numpy(va_u).to(device)

        for seed in range(args.seeds):
            torch.manual_seed(1000 * seed + k)
            model = (CausalXformer(len(names), args.hidden, args.layers, 4, args.dropout)
                     if args.model == "transformer" else
                     GRUForecaster(len(names), args.hidden, args.layers, args.dropout,
                                   cell=args.model)).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                    weight_decay=args.wd)
            sched = (torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
                     if args.sched == 'plateau' else
                     torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs))
            g = torch.Generator().manual_seed(1000 * seed + k)
            best, best_state, bad, best_ep = float("inf"), None, 0, -1
            best_sel = float("inf")
            best_pop = best_auc = best_fwd = best_fwdc = best_fauc = float("nan")
            best_rho = best_star = (float("nan"),) * 3
            log(f"\n    fold {k} seed {seed}: {tr_u.size:,} train users | "
                f"{va_u.size:,} unseen users | {int(M_in[va_u].sum()):,} scored user-days"
                + (f" | {int(M_fw[va_u].sum()):,} forward" if has_fwd else ""))

            def batches(idx: torch.Tensor, bs: int, shuffle: bool):
                order = (idx[torch.randperm(idx.numel(), generator=g).to(device)]
                         if shuffle else idx)
                for i in range(0, order.numel(), bs):
                    yield order[i:i + bs]

            def amp():
                return (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                        if device == "cuda" else torch.autocast(device_type="cpu", enabled=False))

            # Separate RNG for the day sampler so it cannot perturb the user permutation
            # stream `g`: with --day-sample off, every draw from `g` is byte-identical to the
            # historical path, which is what makes the port gate a real gate.
            g_day = torch.Generator(device=device).manual_seed(1000 * seed + k + 977)

            for ep in range(args.epochs):
                model.train()
                lo_ep, hi_ep = curr_window(ep)
                curr_g = (((tt_g >= lo_ep) & (tt_g <= hi_ep))
                          if args.curriculum != "none" else None)
                n_cell = 0.0
                num = den = 0.0
                for b in batches(tr_g, args.batch, True):
                    if args.mixup == "none":
                        xb, tb, msk = Xg[b].float(), Tg[b], TRAINg[b]
                    else:
                        # Synthetic users: interpolate a batch with a shuffled copy of itself.
                        # Beta(0.2,0.2) is U-shaped, so most samples stay close to a real user.
                        # Mixing is linear and standardisation is affine, so mixing the raw
                        # features and standardising after is identical to the reverse.
                        b2 = b[torch.randperm(b.numel(), generator=g).to(device)]
                        lam = float(np.random.beta(args.mixup_alpha, args.mixup_alpha))
                        xb = lam * Xg[b].float() + (1 - lam) * Xg[b2].float()
                        tb = lam * Tg[b] + (1 - lam) * Tg[b2]
                        msk = TRAINg[b] & TRAINg[b2]
                        if args.mixup == "class":
                            # DATA.md 6.1: the target is a spike at 0 plus a bulk near 4.2 with
                            # an almost empty region between (0.127% of real users). Mixing a
                            # buyer with a non-buyer manufactures targets in that gap -- 50x
                            # over-represented. Supervise only where both sources agree.
                            msk = msk & ((Yg[b] > 0) == (Yg[b2] > 0))
                    if args.feat_drop > 0:
                        keep = (torch.rand(1, 1, len(names), device=device)
                                > args.feat_drop).float()
                        xb = xb * keep / max(1e-6, 1 - args.feat_drop)
                    # --- training-regime axis: day window, then (user,date) thinning ---
                    if curr_g is not None:
                        msk = msk & curr_g
                    if args.day_sample < 1.0:
                        msk = msk & (torch.rand(msk.shape, generator=g_day, device=device)
                                     < args.day_sample)
                    m = msk.float()
                    n_cell += float(m.sum())
                    if dw_g is not None:
                        # Fold the recency weight into the mask: the existing expression is
                        # already a weighted mean, so this makes it the RIGHT weighted mean
                        # without changing its scale (it is invariant to a common factor on w).
                        m = m * dw_g
                    if float(m.sum()) == 0:
                        continue
                    with amp():
                        out = model(((xb - mu_g) / sd_g))
                    of_ = out.float()
                    mse = ((of_ - tb) ** 2 * m).sum() / m.sum()
                    if args.loss == "mse":
                        loss = mse
                    else:
                        rho_t = within_day_corr(of_, tb, m)
                        loss = ((1.0 - rho_t) if args.loss == "corr"
                                else mse + args.corr_weight * (1.0 - rho_t))
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    num += float(loss.item()) * float(m.sum()); den += float(m.sum())
                model.eval()
                sn = sd_ = pn = pd_ = fn = fd_ = cn = cd_ = 0.0
                ps, zs, fps, fzs = [], [], [], []
                c_in, c_fw, c_fc = Corr(raw.T), Corr(raw.T), Corr(raw.T)
                with torch.no_grad():
                    for b in batches(va_g, 1024, False):
                        with amp():
                            out = model(((Xg[b].float() - mu_g) / sd_g)).float()
                        e2 = (out - Tg[b]) ** 2      # == (L - (out+B))^2, RMSLE unchanged
                        pred_abs = out + Bg[b]       # the ACTUAL log-prediction
                        c_in.add(pred_abs, Yg[b], Mg[b])
                        if has_fwd:
                            c_fw.add(pred_abs, Yg[b], Fg[b])
                            if has_fc:
                                c_fc.add(pred_abs, Yg[b], Cg[b])
                        m1, m2 = Mg[b].float(), Pg[b].float()
                        sn += float((e2 * m1).sum()); sd_ += float(m1.sum())
                        pn += float((e2 * m2).sum()); pd_ += float(m2.sum())
                        # AUC on the y>0 event -- the quantity EXPERIMENTS.md 1b shows is
                        # capped at ~0.845 across every model class tried. If a feature block
                        # cannot move this, it cannot move rho, and rho is the whole score.
                        sel = Mg[b]
                        # AUC ranks the ACTUAL prediction, so the offset must go back on
                        ps.append((out + Bg[b])[sel].cpu().numpy())
                        zs.append((Yg[b][sel] > 0).cpu().numpy())
                        if has_fwd:
                            m3 = Fg[b].float()
                            fn += float((e2 * m3).sum()); fd_ += float(m3.sum())
                            if has_fc:
                                m4 = Cg[b].float()
                                cn += float((e2 * m4).sum()); cd_ += float(m4.sum())
                            fsel = Fg[b]
                            fps.append((out + Bg[b])[fsel].cpu().numpy())
                            fzs.append((Yg[b][fsel] > 0).cpu().numpy())
                v = (sn / sd_) ** 0.5
                vpop = (pn / pd_) ** 0.5
                a = auc(np.concatenate(ps), np.concatenate(zs).astype(float))
                # Forward metrics are REPORTED, never selected on: early stopping stays on the
                # in-calendar number so the forward reading is a genuine held-out statistic and
                # the protocol is unchanged for the config being compared against.
                vf = (fn / fd_) ** 0.5 if has_fwd else float("nan")
                vfc = (cn / cd_) ** 0.5 if has_fc else float("nan")
                af = (auc(np.concatenate(fps), np.concatenate(fzs).astype(float))
                      if has_fwd else float("nan"))
                rho_in, rho_fw, rho_fc = (c_in.rho_within(), c_fw.rho_within(),
                                          c_fc.rho_within())
                star_in, star_fw, star_fc = (c_in.star_within(), c_fw.star_within(),
                                             c_fc.star_within())
                sched.step(v) if args.sched == 'plateau' else sched.step()
                log(f"      epoch {ep + 1:>3d}  train {(num / den) ** 0.5:.5f}  "
                    f"UNSEEN {v:.5f}  (in-pop {vpop:.5f})  AUC {a:.5f}  "
                    + (f"FWD {vf:.5f}  fAUC {af:.5f}  " if has_fwd else "")
                    # Supervised cells actually consumed this epoch. Logged for EVERY arm so
                    # "expand grows the work, slide holds it constant" is a measurement in the
                    # log rather than a property of the schedule I asserted in a docstring.
                    + (f"cells {n_cell / 1e6:.2f}M  "
                       if (args.curriculum != "none" or args.day_sample < 1.0) else "")
                    + f"lr {opt.param_groups[0]['lr']:.2e}  [{(time.time() - t0) / 60:.1f}m]")
                # The early-stopping statistic MUST match the objective: under a
                # correlation loss the RMSLE of an unscaled prediction is meaningless, so
                # selecting on it would pick an arbitrary epoch. `best` still HOLDS the RMSLE
                # so every reported number keeps its meaning; only the decision changes.
                sel = v if args.loss == "mse" else (1.0 - rho_in)
                if args.fixed_epochs:
                    # no selection at all: the LAST epoch is the answer, so the number carries
                    # no min-of-N bias and configs with different curve lengths are comparable.
                    sel = -float(ep)
                if sel < best_sel - 1e-6:
                    best_sel = sel
                    best, best_ep, bad = v, ep + 1, 0
                    best_state = {q: w.detach().clone() for q, w in model.state_dict().items()}
                    best_pop, best_auc = vpop, a
                    best_fwd, best_fwdc, best_fauc = vf, vfc, af
                    best_rho = (rho_in, rho_fw, rho_fc)
                    best_star = (star_in, star_fw, star_fc)
                else:
                    bad += 1
                    if bad >= args.patience and not args.fixed_epochs:
                        log(f"      early stop at epoch {ep + 1} (best {best:.5f} @ {best_ep})")
                        break
            model.load_state_dict(best_state)
            if seed == 0:
                # OOF at the frozen anchors for this fold's unseen users (seed 0 only, so the
                # file is one model per user rather than a seed average)
                model.eval()
                with torch.no_grad():
                    for b in batches(va_g, 1024, False):
                        with amp():
                            o_ = model(((Xg[b].float() - mu_g) / sd_g)).float() + Bg[b]
                        for fk, ta in frozen.items():
                            sel = Og[b][:, ta]          # == Mg when --t-cut is off
                            if not bool(sel.any()):
                                continue
                            oof_rows.append((
                                fk, raw.users[b[sel].cpu().numpy()],
                                Yg[b][:, ta][sel].cpu().numpy(),
                                o_[:, ta][sel].cpu().numpy()))
            if not args.no_save_weights:
                # Checkpoint every fold x seed.  Without this the only way to reuse a CV model
                # is to retrain it, which is what forced the first submission build to fit a
                # fresh full-data model instead of reusing what CV had already produced.
                cd = ROOT / "runs" / "usercv"; cd.mkdir(parents=True, exist_ok=True)
                ck = cd / f"{ck_tag}_f{k}_s{seed}.pt"
                torch.save({"state_dict": best_state, "variant": args.variant,
                            "n_features": len(names), "feature_names": names,
                            "fold": k, "seed": seed, "best_epoch": best_ep,
                            "unseen_rmsle": best, "hidden": args.hidden,
                            "mu": mu, "sd": sd, "burn_in": args.burn_in,
                            "last_anchor": int(last_anchor)}, ck)
                log(f"    saved runs/usercv/{ck.name}")
            results.append({"fold": k, "seed": seed, "unseen_rmsle": best,
                            "unseen_rmsle_in_population": best_pop, "auc": best_auc,
                            "fwd_rmsle": best_fwd, "fwd_clean_rmsle": best_fwdc,
                            "fwd_auc": best_fauc, "best_epoch": best_ep,
                            "rho_in": best_rho[0], "rho_fwd": best_rho[1],
                            "rho_fwd_clean": best_rho[2],
                            "star_in": best_star[0], "star_fwd": best_star[1],
                            "star_fwd_clean": best_star[2]})
            log(f"    -> fold {k} seed {seed}: UNSEEN {best:.5f} "
                f"(in-population {best_pop:.5f}) at epoch {best_ep}"
                + (f" | FORWARD {best_fwd:.5f} (guard-clean {best_fwdc:.5f}) "
                   f"fAUC {best_fauc:.5f}" if has_fwd else ""))

    r = np.array([x["unseen_rmsle"] for x in results])
    rp = np.array([x["unseen_rmsle_in_population"] for x in results])
    per_fold = [float(np.mean([x["unseen_rmsle"] for x in results if x["fold"] == k]))
                for k in args.folds]
    runtime = (time.time() - t0) / 60
    log(f"\n  === variant {args.variant} : {len(names)} features ===")
    log(f"  UNSEEN-USER RMSLE = {r.mean():.5f} +/- {r.std():.5f}   "
        f"(over {len(results)} fold x seed runs)")
    log(f"  per fold (seed-averaged): {[round(x, 5) for x in per_fold]}")
    log(f"  same, restricted to the in-population user-days = {rp.mean():.5f} +/- {rp.std():.5f}")
    au = np.array([x["auc"] for x in results])
    log(f"  AUC on y>0 (the capped quantity) = {au.mean():.5f} +/- {au.std():.5f}")
    fwd_fold = fwd = fwdc = fau = None
    if has_fwd:
        fwd = np.array([x["fwd_rmsle"] for x in results])
        fwdc = np.array([x["fwd_clean_rmsle"] for x in results])
        fau = np.array([x["fwd_auc"] for x in results])
        fwd_fold = [float(np.mean([x["fwd_rmsle"] for x in results if x["fold"] == k]))
                    for k in args.folds]
        log(f"\n  --- FORWARD-IN-CALENDAR (the B7 reading; anchors >= {raw.day(cut + HORIZON)}) ---")
        log(f"  FORWARD unseen-user RMSLE = {fwd.mean():.5f} +/- {fwd.std():.5f}")
        log(f"  per fold (seed-averaged): {[round(x, 5) for x in fwd_fold]}")
        if has_fc:
            log(f"  forward AND guard-clean   = {np.nanmean(fwdc):.5f} +/- {np.nanstd(fwdc):.5f}"
                f"   (anchors <= {raw.day(clean_max)}; the only unbiased slice)")
        else:
            log("  forward AND guard-clean   = n/a (no clean anchor at this t-cut) -- the")
            log("    forward score sits entirely in the guaranteed-activity zone and is")
            log("    optimistic in LEVEL by ~0.041 (DATA.md 4.3). Paired deltas still read.")
        log(f"  FORWARD AUC on y>0        = {fau.mean():.5f} +/- {fau.std():.5f}")
        log("  Selection was on the IN-CALENDAR number; forward is held out.")
    # rho and the perfectly-calibrated score, for every block. THE HONEST STATISTICS:
    # raw RMSLE mixes level (free, 1b) with rho (irreducible). Read these, not the raw ones.
    log("\n  --- WITHIN-ANCHOR rho and calibrated score (RMSLE* = sd_L*sqrt(1-rho^2)) ---")
    log("      (within-anchor: each day centred on its own mean. Pooling across days")
    log("       credits between-anchor LEVEL, which 1b proves is free -- e0234 read")
    log("       +0.00079 pooled and -0.00009 within-anchor.)")
    for key, lab in (("in", "in-calendar"), ("fwd", "FORWARD"), ("fwd_clean", "fwd guard-clean")):
        rr = np.array([x[f"rho_{key}"] for x in results], float)
        ss = np.array([x[f"star_{key}"] for x in results], float)
        if not np.isfinite(rr).any():
            continue
        log(f"  {lab:16s} rho = {np.nanmean(rr):.5f} +/- {np.nanstd(rr):.5f}   "
            f"RMSLE* = {np.nanmean(ss):.5f} +/- {np.nanstd(ss):.5f}")
    log(f"  best epoch per run: {[x['best_epoch'] for x in results]}")
    log(f"  runtime {runtime:.1f} min")
    log("\n  NOT comparable to experiments.csv (same users at a FUTURE anchor). Two upward")
    log("  biases are live here by design: guard-zone target windows, and no population rule.")

    tag = ck_tag
    if oof_rows:
        import pyarrow as pa, pyarrow.parquet as pq
        fk = np.concatenate([np.full(u.size, kk, np.int8) for kk, u, _, _ in oof_rows])
        uu = np.concatenate([u for _, u, _, _ in oof_rows])
        yy = np.concatenate([y for _, _, y, _ in oof_rows])
        pp = np.concatenate([p for _, _, _, p in oof_rows])
        o = np.lexsort((uu, fk))
        (ROOT / "oof").mkdir(exist_ok=True)
        pq.write_table(pa.table({"fold_id": fk[o], "user_id": uu[o],
                                 "y_true": np.expm1(yy[o]).astype(np.float64),
                                 "y_pred": np.maximum(np.expm1(pp[o]), 0.0).astype(np.float64)}),
                       ROOT / "oof" / f"usercv_{tag}.parquet")
        log(f"  wrote oof/usercv_{tag}.parquet ({fk.size:,} rows at the 5 frozen anchors)")

    out = ROOT / "reports" / "eda" / f"usercv_{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "variant": args.variant, "n_features": len(names),
        "date": datetime.now().isoformat(timespec="seconds"),
        "protocol": "5-fold split by user_id (md5 salt gmv-v1); target log1p(sum 30d)",
        "last_anchor": str(raw.day(last_anchor)),
        "unseen_rmsle_mean": float(r.mean()), "unseen_rmsle_std": float(r.std()),
        "in_population_mean": float(rp.mean()), "auc_mean": float(au.mean()),
        "per_fold": per_fold,
        "t_cut": args.t_cut,
        "t_cut_anchor": (int(cut) if args.t_cut is not None else None),
        # `last_anchor` above is the PANEL's last usable anchor, not the training cap -- under
        # --train-cap the two differ and only the tag distinguished them. Recorded explicitly.
        "train_cap": args.train_cap,
        "scored_block": ("guard-clean (t <= %d)" % clean_max) if args.train_cap else "all masked",
        "fwd_rmsle_mean": (float(fwd.mean()) if has_fwd else None),
        "fwd_rmsle_std": (float(fwd.std()) if has_fwd else None),
        "fwd_clean_rmsle_mean": (float(np.nanmean(fwdc)) if has_fc else None),
        "fwd_auc_mean": (float(fau.mean()) if has_fwd else None),
        "fwd_per_fold": fwd_fold,
        **{f"{p}_{k}_mean": (float(np.nanmean([x[f"{p}_{k}"] for x in results]))
                             if np.isfinite([x[f"{p}_{k}"] for x in results]).any() else None)
           for p in ("rho", "star") for k in ("in", "fwd", "fwd_clean")},
        "runs": results, "runtime_min": round(runtime, 1)}, indent=2))
    log(f"  wrote reports/eda/usercv_{tag}.json")


if __name__ == "__main__":
    main()
