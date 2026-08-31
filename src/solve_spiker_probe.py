#!/usr/bin/env python
"""Turn the returned LB score for subs/probe_spikers.csv into E[log1p(y) | spikers].

    python3 src/solve_spiker_probe.py <lb_score>
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
M = json.loads((ROOT / "reports" / "eda" / "probe_spikers_meta.json").read_text())
R = float(sys.argv[1].replace(",", "."))
lc, nS, N, EL2, ELa = np.log1p(M["c"]), M["n_S_pub_expected"], 50_000, M["EL2"], M["EL_all"]
EL_S = (N * EL2 + nS * lc ** 2 - N * R ** 2) / (2 * lc * nS)
d = M["precision"]
EL_rest = (N * ELa - nS * EL_S) / (N - nS)
print(f"\n  LB score for the spiker probe = {R}")
print(f"  => E[log1p(y) | pre-holiday spikers] = {EL_S:.4f}  +-{d:.4f}")
print(f"     E[log1p(y) | everyone else]       = {EL_rest:.4f}")
print(f"     population mean                   = {ELa:.4f}")
print(f"\n  effect = {EL_S - EL_rest:+.4f} log-points  ({(EL_S - EL_rest) / d:+.1f} sigma)")
print(f"  H0 (no effect) predicted {ELa:.2f};  H1 (+0.18) predicted {ELa + 0.18:.2f}")
if abs(EL_S - ELa) < 2 * d:
    print("  -> consistent with H0: pre-holiday spikers are NOT distinguishable. Kill the idea.")
elif EL_S > ELa:
    print("  -> spikers DO have higher 2026 Feb-March GMV than the population.")
    print("     Next question is whether the MODEL already knows: compare against")
    print("     mean(log1p(e0020 prediction)) over the same subset, printed below.")
else:
    print("  -> spikers have LOWER outcomes than the population (opposite of the hypothesis).")
try:
    import pandas as pd
    s = pd.read_csv(ROOT / "subs" / "probe_spikers.csv")
    e = pd.read_csv(ROOT / "subs" / "e0020.csv")
    m = s.predict.values > 0
    print(f"\n  our model's own mean log1p(pred) on S      = {np.log1p(e.predict.values[m]).mean():.4f}")
    print(f"  our model's own mean log1p(pred) elsewhere = {np.log1p(e.predict.values[~m]).mean():.4f}")
    print("  (if the model's gap already matches the measured gap, it knows -- no feature needed)")
except Exception as ex:
    print(f"  [model comparison skipped: {ex}]")
