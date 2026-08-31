#!/usr/bin/env python
"""
E-IDEA-05 -- WHERE is the unclaimed variance?   IDEAS.md §I11.

`scripts/noise_ceiling.py` bounds the total: reliability r ~ 0.55, so any predictor's
corr(L, .) <= ~0.725, against e0049's 0.661.  That is one number for 250k users.  This splits
it by user segment, so a future idea can be aimed instead of guessed.

Per segment s, defined CAUSALLY from activity before each window:

    r_s      reliability inside the segment  (test-retest, lag curve extrapolated to 0)
    rho_s^2  the model's within-segment R^2  (affine-optimal, from oof/e0049.parquet)
    gap_s    r_s - rho_s^2                   = reliable variance the model does not capture
    share_s  w_s * Var_s(L) * gap_s          = that segment's contribution to the total gap

The accounting: a predictor knowing the segment gets the BETWEEN-segment variance for free, so
the contestable quantity is the within-segment part, and sum_s share_s is the whole prize.

NO MODEL IS TRAINED HERE.  That matters: BACKLOG.md now records that the 15k local harness has
a ~±0.004 rho band for model-level changes, which makes trained screens useless at this scale.
This is arithmetic on the panel plus an OOF file, so that band does not apply.

Run:  python3.11 scripts/headroom_map.py            # 15k subset
      python3.11 scripts/headroom_map.py --full     # all 250k (cluster)
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

try:
    from screen_features import cap_memory, log, make_subset, rss_gb    # noqa: E402
except ModuleNotFoundError:                                             # cluster
    def log(m): print(m, flush=True)
    def cap_memory(gb): pass
    def make_subset(n): raise SystemExit("use --full on the cluster")
    def rss_gb():
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 1e9

CLEAN_END = date(2025, 11, 15)
POP_ANCHOR = date(2025, 4, 19)
LABELS = ["0 buy-days", "1", "2-3", "4-7", "8+"]
EDGES = [1, 2, 4, 8]                                   # np.digitize cut points on buy-days/90d


def segment_of(p, at: int) -> np.ndarray:
    """Segment index from buy-days in the 90 days ending at day `at` -- known at the cut-off."""
    bd = p.cs_buy[:, at + 1] - p.cs_buy[:, at - 89]
    return np.digitize(bd, EDGES)


def main(full: bool, max_gb: float):
    import polars as pl
    from data import Panel
    cap_memory(max_gb)
    p = Panel() if full else Panel(path=make_subset(15000))

    ai, end_i = p.idx(POP_ANCHOR), p.idx(CLEAN_END)
    n_win = (end_i - ai) // 30
    Ls, pops, segs = [], [], []
    for j in range(n_win):
        a = ai + 1 + j * 30
        Ls.append(np.log1p(p.wsum("gmv", a, a + 29)))
        pops.append(p.active_in(a - 30, a - 1))
        segs.append(segment_of(p, a - 1))
    Ls, pops, segs = np.array(Ls), np.array(pops), np.array(segs)
    log(f"  {p.n_users:,} users, {n_win} clean 30-day windows from {POP_ANCHOR}")

    # ---- reliability per segment: lag curve inside the segment, extrapolated to lag 0 ----
    rel, varL, wgt = {}, {}, {}
    for s in range(len(LABELS)):
        lags, vals = [], []
        for m in range(1, n_win):
            cs = []
            for j in range(n_win - m):
                sel = pops[j] & (segs[j] == s)
                if sel.sum() > 200 and Ls[j][sel].std() > 1e-6 and Ls[j + m][sel].std() > 1e-6:
                    cs.append(float(np.corrcoef(Ls[j][sel], Ls[j + m][sel])[0, 1]))
            if cs:
                lags.append(m * 30); vals.append(float(np.mean(cs)))
        if len(lags) < 3:
            rel[s] = float("nan"); continue
        rel[s] = float(np.polyval(np.polyfit(np.asarray(lags, float),
                                             np.asarray(vals, float), 1), 0.0))
        inseg = [Ls[j][pops[j] & (segs[j] == s)] for j in range(n_win)]
        varL[s] = float(np.mean([v.var() for v in inseg if v.size > 200]))
        wgt[s] = float(np.mean([(pops[j] & (segs[j] == s)).sum() / pops[j].sum()
                                for j in range(n_win)]))

    # ---- achieved per segment: e0049 OOF, segment recomputed at each fold's own anchor ----
    spec = __import__("json").loads((ROOT / "data" / "fold_spec.json").read_text())
    oof = pl.read_parquet(ROOT / "oof" / "e0049.parquet")
    ach_num, ach_den = {s: 0.0 for s in rel}, {s: 0.0 for s in rel}
    for fs in spec["folds"]:
        k, va = fs["fold_id"], date.fromisoformat(fs["valid_anchor"])
        vi = p.idx(va)
        seg_u = dict(zip(p.users.tolist(), segment_of(p, vi).tolist()))
        f = oof.filter(pl.col("fold_id") == k)
        uid = f["user_id"].to_numpy()
        m = np.array([u in seg_u for u in uid])
        if not m.any():
            continue
        sg = np.array([seg_u[u] for u in uid[m]])
        L = np.log1p(f["y_true"].to_numpy()[m]); M = np.log1p(f["y_pred"].to_numpy()[m])
        for s in rel:
            q = sg == s
            if q.sum() > 200 and L[q].std() > 1e-6 and M[q].std() > 1e-6:
                ach_num[s] += float(np.corrcoef(L[q], M[q])[0, 1]) ** 2 * q.sum()
                ach_den[s] += q.sum()

    log(f"\n  segment          share   Var_s(L)   reliability r_s   model R^2   gap    prize share")
    tot = 0.0
    rows = []
    for s, lab in enumerate(LABELS):
        if s not in varL or not np.isfinite(rel.get(s, np.nan)) or ach_den[s] == 0:
            log(f"  {lab:14s}  (too few users to estimate)"); continue
        r2 = ach_num[s] / ach_den[s]
        gap = rel[s] - r2
        prize = wgt[s] * varL[s] * gap
        tot += max(prize, 0.0)
        rows.append((lab, wgt[s], varL[s], rel[s], r2, gap, prize))
    for lab, w, v, r, r2, gap, prize in rows:
        log(f"  {lab:14s}  {w:5.3f}   {v:7.3f}    {r:8.4f}       {r2:7.4f}   "
            f"{gap:+7.4f}   {100*max(prize,0)/max(tot,1e-9):5.1f}%")
    log(f"\n  total unclaimed within-segment variance: {tot:.4f}"
        f"   (Var(L) overall ~ {np.mean([Ls[j][pops[j]].var() for j in range(n_win)]):.3f})")
    log(f"\n  NOTE: reliability is measured on windows from {POP_ANCHOR}; the model R^2 comes from"
        f"\n  the five frozen fold anchors. Both sit in the clean region but they are not the same"
        f"\n  windows, so treat the per-segment gap as indicative, not as a paired delta.")
    log(f"  peak RSS {rss_gb():.2f} GB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--max-gb", type=float, default=5.5)
    a = ap.parse_args()
    main(a.full, a.max_gb)
