#!/usr/bin/env python
"""
20x20 pair-plot matrix of the strongest features, coloured by the target.

Cell (i, j) = feature j on x, feature i on y, one point per user, colour = log1p(target).
Diagonal cells would be a useless y=x line, so they show feature-vs-target instead.

Three decisions the plot depends on, all of which would hide the structure if taken naively:

  * COLOUR ON log1p(target). GMV spans five orders of magnitude; on a linear colour scale
    every user below the 99th percentile is the same shade and the grid shows nothing.
  * RANK BY GAIN AMONG FEATURES THAT BEAT THEIR NULL. Pure null-importance score is a ratio,
    so a feature the model barely splits on can top the list purely because its null gain is
    also ~0 (ord_sum_270: score 8.4, gain 4,437 against 12,120,254 for the leader). That is
    "cleanest evidence of signal", not "most predictive". Pure gain, meanwhile, is what
    CLAUDE.md §5.3 warns is biased toward high-cardinality continuous columns. Requiring both
    is the only ranking that means "predictive AND verified".
  * CLIP AXES TO [q0.5, q99.5]. These are heavy-tailed counts and sums; one user with 400
    orders otherwise compresses everyone else into a single pixel column.

Points are drawn in SHUFFLED order. Drawing high-target users last would put them on top of
every cluster and make the target look better separated than it is.

Outputs (reports/eda/):
  pairgrid_scatter.png   the grid as asked -- subsampled scatter, zoomable
  pairgrid_binned.png    same layout, 2D bins coloured by MEAN log1p(target); scatter of
                         200k users overplots badly and the binned version is what actually
                         shows where the target lives
  pairgrid_features.txt  index -> feature name, so a zoomed cell can be identified
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
import numpy as np                                   # noqa: E402
import yaml                                          # noqa: E402
from matplotlib.colors import Normalize              # noqa: E402

ROOT = Path("/path/to/ecup")
sys.path.insert(0, str(ROOT / "src"))
from data import Panel                               # noqa: E402
from features import build                           # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/e0049_nomoment.yaml")
ap.add_argument("--k", type=int, default=20, help="grid size (k x k)")
ap.add_argument("--fold", type=int, default=-1, help="which validation anchor (-1 = last)")
ap.add_argument("--n-points", type=int, default=25000, help="users sampled per scatter panel")
ap.add_argument("--panel-in", type=float, default=2.6)
ap.add_argument("--dpi", type=int, default=180)
ap.add_argument("--bins", type=int, default=110, help="grid resolution for the binned figure")
ap.add_argument("--min-bin-n", type=int, default=12,
                help="bins with fewer users than this are left blank, not drawn as noise")
ap.add_argument("--marker", type=float, default=0.6, help="scatter marker size in points^2")
args = ap.parse_args()

OUT = ROOT / "reports" / "eda"
OUT.mkdir(parents=True, exist_ok=True)

# Agg refuses to render a canvas wider than 2^16 px in either direction, and bbox_inches
# ="tight" re-renders slightly larger than the nominal figure. Clamp before matplotlib
# throws after the (expensive) feature build has already happened.
_px = args.k * args.panel_in * args.dpi
if _px > 60000:
    _new = int(60000 / (args.k * args.panel_in))
    print(f"  dpi {args.dpi} would render {_px:.0f} px/side; Agg's limit is 65536. "
          f"Clamping to {_new}.")
    args.dpi = _new
print(f"  target canvas ~{args.k * args.panel_in * args.dpi:.0f} px/side "
      f"({args.k}x{args.k} panels @ {args.panel_in}in, {args.dpi} dpi)")
cfg = yaml.safe_load((ROOT / args.config).read_text())
spec = json.loads((ROOT / "data" / "fold_spec.json").read_text())


def hdr(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}", flush=True)


# --------------------------------------------------------------------- pick the features
hdr("1 -- RANK THE FEATURES")
imp = json.loads((ROOT / "reports" / "importance.json").read_text())
names_all = imp["names"]
gain = np.asarray(imp["actual"], float)
score = np.asarray(imp["score"], float)

eligible = np.flatnonzero(score > 0)                 # beat its own shuffled-target null
ranked = [int(i) for i in eligible[np.argsort(-gain[eligible])]]
print(f"  {len(eligible)} of {len(names_all)} features beat their null")

# --------------------------------------------------------------------- build the matrix
hdr("2 -- BUILD FEATURES AT THE VALIDATION ANCHOR")
p = Panel()
fs = spec["folds"][args.fold]
va = date.fromisoformat(fs["valid_anchor"])
ai = p.idx(va)
keep = p.active_in(ai - 29, ai)
X, names = build(p, ai, keep, cfg["feature_blocks"])
y = p.target(ai)[keep]
print(f"  anchor {va} (fold {args.fold}), {int(keep.sum()):,} users, {X.shape[1]} features")

# Rank AMONG THE FEATURES THIS CONFIG ACTUALLY BUILDS, then take the top k -- otherwise a
# config narrower than the importance run (e0049 drops the whole sbcmoment family, which
# happens to hold the single highest-gain feature) silently yields a grid smaller than asked.
have = set(names)
sel_names = [names_all[i] for i in ranked if names_all[i] in have][: args.k]
skipped = [names_all[i] for i in ranked[: args.k] if names_all[i] not in have]
if skipped:
    print(f"  {len(skipped)} top-ranked features are not in this config, backfilled from "
          f"further down the ranking: {skipped}")
print(f"  {'#':>3} {'feature':42s} {'gain':>13} {'null-score':>11}")
for r, n in enumerate(sel_names, 1):
    i = names_all.index(n)
    print(f"  {r:>3} {n:42s} {gain[i]:>13,.0f} {score[i]:>11.3f}")
idx = [names.index(n) for n in sel_names]
F = np.nan_to_num(X[:, idx].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
del X
K = len(sel_names)
L = np.log1p(np.maximum(y, 0.0))                     # colour axis
print(f"  grid will be {K}x{K}; target log1p range {L.min():.2f} .. {L.max():.2f}, "
      f"mean {L.mean():.3f}, zeros {100 * (y <= 0).mean():.1f}%")

(OUT / "pairgrid_features.txt").write_text(
    "\n".join(f"{r:>3}  {n}" for r, n in enumerate(sel_names, 1)) + "\n")

# display range per feature -- heavy tails would otherwise squash everything to one pixel
lo = np.percentile(F, 0.5, axis=0)
hi = np.percentile(F, 99.5, axis=0)
hi = np.where(hi <= lo, lo + 1e-9, hi)
norm = Normalize(vmin=float(np.percentile(L, 1)), vmax=float(np.percentile(L, 99)))
CMAP = "viridis"

rng = np.random.default_rng(0)
n_pts = min(args.n_points, F.shape[0])
samp = rng.choice(F.shape[0], n_pts, replace=False)   # one shared sample for all panels
samp = rng.permutation(samp)                          # ...and shuffled draw order
Fs, Ls = F[samp], L[samp]


def label_axes(ax, i, j, K):
    """Only the outer edge carries feature names; every panel carries its y/x index so a
    zoomed-in cell can still be located in the grid."""
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.02, 0.97, f"y{i + 1} x{j + 1}", transform=ax.transAxes, fontsize=5,
            va="top", ha="left", color="0.35")
    if j == 0:
        ax.set_ylabel(f"{i + 1}. {sel_names[i]}", fontsize=6, rotation=0,
                      ha="right", va="center", labelpad=6)
    if i == K - 1:
        ax.set_xlabel(f"{j + 1}. {sel_names[j]}", fontsize=6, rotation=45,
                      ha="right", va="top")


def finish(fig, path, title):
    cax = fig.add_axes([0.92, 0.35, 0.008, 0.30])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax,
                 label="log1p(target GMV, next 30d)")
    fig.suptitle(title, fontsize=13, y=0.995)
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path}  ({path.stat().st_size / 1e6:.1f} MB)")


# --------------------------------------------------------------------- scatter grid
hdr("3 -- SCATTER GRID")
fig, axes = plt.subplots(K, K, figsize=(K * args.panel_in, K * args.panel_in))
for i in range(K):
    for j in range(K):
        ax = axes[i, j]
        if i == j:
            ax.scatter(Fs[:, j], Ls, s=args.marker, c=Ls, cmap=CMAP, norm=norm,
                       linewidths=0, alpha=0.45, rasterized=True)
            ax.set_xlim(lo[j], hi[j])
            ax.set_facecolor("#f2f2f2")               # mark the diagonal: y is the TARGET
        else:
            ax.scatter(Fs[:, j], Fs[:, i], s=args.marker, c=Ls, cmap=CMAP, norm=norm,
                       linewidths=0, alpha=0.45, rasterized=True)
            ax.set_xlim(lo[j], hi[j]); ax.set_ylim(lo[i], hi[i])
        label_axes(ax, i, j, K)
    print(f"  row {i + 1}/{K} done", flush=True)
finish(fig, OUT / "pairgrid_scatter.png",
       f"Top-{K} features (gain among null-beating), anchor {va} — "
       f"{n_pts:,} users, colour = log1p(target). Diagonal (grey) = feature vs target.")

# --------------------------------------------------------------------- binned grid
hdr("4 -- BINNED GRID (mean target per cell)")
# With 200k users the scatter overplots: whichever point is drawn last wins the pixel, so
# dense regions show one arbitrary user's target rather than the local average. Binning by
# MEAN log1p(target) is what actually answers "where does the target live in this plane".
MIN_N = args.min_bin_n                                # bins thinner than this are noise
occ = args.bins ** 2
print(f"  {args.bins}x{args.bins} = {occ:,} bins over {F.shape[0]:,} users "
      f"-> {F.shape[0] / occ:.1f} users/bin on average; bins under {MIN_N} stay blank")
fig, axes = plt.subplots(K, K, figsize=(K * args.panel_in, K * args.panel_in))
for i in range(K):
    for j in range(K):
        ax = axes[i, j]
        yv = L if i == j else F[:, i]
        ylo, yhi = (norm.vmin, norm.vmax) if i == j else (lo[i], hi[i])
        xe = np.linspace(lo[j], hi[j], args.bins + 1)
        ye = np.linspace(ylo, yhi, args.bins + 1)
        cnt, _, _ = np.histogram2d(F[:, j], yv, bins=[xe, ye])
        tot, _, _ = np.histogram2d(F[:, j], yv, bins=[xe, ye], weights=L)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(cnt >= MIN_N, tot / np.maximum(cnt, 1), np.nan)
        ax.imshow(mean.T, origin="lower", aspect="auto", cmap=CMAP, norm=norm,
                  extent=[lo[j], hi[j], ylo, yhi], interpolation="nearest")
        if i == j:
            ax.set_facecolor("#f2f2f2")
        label_axes(ax, i, j, K)
    print(f"  row {i + 1}/{K} done", flush=True)
finish(fig, OUT / "pairgrid_binned.png",
       f"Top-{K} features, anchor {va} — colour = MEAN log1p(target) per bin "
       f"(all {F.shape[0]:,} users, bins with <{MIN_N} users left blank).")

print("\n  done")
