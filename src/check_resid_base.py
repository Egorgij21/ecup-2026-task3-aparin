#!/usr/bin/env python
"""Guard for the residual parametrisation: geo3_log must equal the geo3 already stored as
`y_naive` in the frozen-fold OOF.  If it does not, the offset is not the baseline every
experiment is scored against and the residual runs are not comparable to anything."""
import sys, datetime as dt
import numpy as np, pyarrow.parquet as pq
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from usercv_features import Raw, geo3_log
raw = Raw(verbose=False); B = geo3_log(raw)
oof = pq.read_table(ROOT / "oof" / "e0049.parquet").to_pydict()
o = np.lexsort((np.array(oof["user_id"]), np.array(oof["fold_id"])))
fid, uid = np.array(oof["fold_id"])[o], np.array(oof["user_id"])[o]
nv, yt = np.log1p(np.array(oof["y_naive"])[o]), np.log1p(np.array(oof["y_true"])[o])
A = {0: "2025-06-18", 1: "2025-07-18", 2: "2025-08-17", 3: "2025-09-16", 4: "2025-10-16"}
worst = 0.0
for k in range(5):
    m = fid == k; t = raw.idx(dt.date.fromisoformat(A[k]))
    d = np.abs(B[np.searchsorted(raw.users, uid[m]), t] - nv[m]).max()
    worst = max(worst, d)
    print(f"    fold {k}: max abs diff = {d:.3e}   n={m.sum():,}", flush=True)
assert worst < 1e-3, f"geo3_log disagrees with the frozen OOF baseline ({worst:.3e})"
print(f"\n  residual target var {np.var(yt-nv):.4f} vs absolute {np.var(yt):.4f}"
      f"  -> {np.var(yt)/np.var(yt-nv):.2f}x smaller learning problem")
print("  check_resid_base: PASSED")
