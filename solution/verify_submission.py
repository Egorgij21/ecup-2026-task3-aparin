#!/usr/bin/env python
"""
Проверка готового файла сабмита: формальная валидность + метрики, по которым его будет
смотреть жюри.

    python solution/verify_submission.py subs/e0303_arch4_cal.csv
    python solution/verify_submission.py --finals

Проверяется:
  1. ФОРМАТ — 250 000 строк, ровно то множество user_id, что в sample_submit, без NaN,
     без отрицательных, порядок совпадает.
  2. МОМЕНТЫ log1p(prediction) против решённых из probe-сабмитов моментов тестовой правды
     (mu_L = 2.3303, sd_L = 2.3178) и стоимость промаха по уровню в единицах RMSLE.
  3. ТАЙ-БРЕЙКЕРЫ ЖЮРИ (TASK.md, «Определение призёров»): коэффициент Джини по предсказаниям
     и суммарный предсказанный GMV. Оба считаются на самом файле; истинный тотал
     заякорен по OOF fold 4 и по probe-моментам — см. research/finals/finals_tiebreak.py.
  4. ПОПАРНАЯ БЛИЗОСТЬ двух финальных файлов — то, ради чего второй слот и держат.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
MU_L, SD_L = 2.3303, 2.3178          # решены из probe_zeros (3.28) и probe_const10 (2.32)
FINALS = ["subs/e0303_arch4_cal.csv"]
TRUE_TOTAL_ANCHOR = 22_190_727        # оценка истинного тотала теста, finals_tiebreak.py (b)


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x)
    if x.sum() <= 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2.0 * (idx * x).sum()) / (n * x.sum()) - (n + 1.0) / n)


def check(path: Path, ids_ref: np.ndarray) -> np.ndarray:
    d = pl.read_csv(path)
    cols = d.columns
    print(f"\n=== {path.relative_to(ROOT) if path.is_absolute() else path}")
    print(f"  колонки {cols}")
    ids = d[cols[0]].to_numpy()
    p = d[cols[1]].to_numpy().astype(np.float64)

    problems = []
    if len(ids) != 250_000:
        problems.append(f"{len(ids)} строк вместо 250000")
    if ids_ref is None:
        pass
    elif not np.array_equal(np.sort(ids), np.sort(ids_ref)):
        problems.append("множество user_id не совпадает с sample_submit")
    elif not np.array_equal(ids, ids_ref):
        problems.append("порядок user_id отличается (множество то же) — скорер обычно "
                        "переупорядочит, но проверьте требования площадки")
    if np.isnan(p).sum():
        problems.append(f"{int(np.isnan(p).sum())} NaN")
    if (p < 0).sum():
        problems.append(f"{int((p < 0).sum())} отрицательных")
    ok_msg = ("OK — 250000 строк, id совпадают, без NaN и отрицательных" if ids_ref is not None
              else "OK — 250000 строк, без NaN и отрицательных (сверка id пропущена)")
    print("  формат: " + (ok_msg if not problems else "ПРОБЛЕМЫ: " + "; ".join(problems)))

    L = np.log1p(np.maximum(p, 0.0))
    cost = (L.mean() - MU_L) ** 2 / (2 * 1.6465)
    print(f"  log1p: mu {L.mean():.4f} (цель {MU_L:.4f}, промах {L.mean()-MU_L:+.4f} "
          f"= {cost:+.6f} RMSLE)   sd {L.std():.4f}")
    tot = p.sum()
    print(f"  тай-брейкеры жюри: Gini {gini(p):.5f}   тотал GMV {tot:,.0f} "
          f"({tot/TRUE_TOTAL_ANCHOR-1:+.1%} к оценке истинного)")
    print(f"  доля нулей {100*(p <= 1e-9).mean():.3f}%   p99 {np.quantile(p,0.99):,.0f}   "
          f"max {p.max():,.0f}")
    return L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="csv для проверки")
    ap.add_argument("--finals", action="store_true", help="проверить финальный файл")
    a = ap.parse_args()
    files = [ROOT / f for f in (FINALS if a.finals or not a.files else a.files)]

    ref = next((ROOT / "data" / n for n in ("sample_submit.csv", "sample_submission.csv")
                if (ROOT / "data" / n).exists()), None)
    if ref is None:
        print("  ⚠ data/sample_submit.csv не найден — сверка множества user_id пропущена.\n"
              "    Положите sample_submission.csv площадки в data/, чтобы включить её.")
        ids_ref = None
    else:
        ids_ref = pl.read_csv(ref)["user_id"].to_numpy()
    logs = {}
    for f in files:
        if not f.exists():
            print(f"\n=== {f}: ФАЙЛ ОТСУТСТВУЕТ")
            continue
        logs[f.name] = check(f, ids_ref)

    if len(logs) == 2:
        (n1, a1), (n2, a2) = logs.items()
        r = float(np.corrcoef(a1, a2)[0, 1])
        rms = float(np.sqrt(((a1 - a2) ** 2).mean()))
        print(f"\n=== два финальных файла между собой")
        print(f"  corr {r:.6f}   rms(dlog) {rms:.4f}   1-r^2 {1-r*r:.5f}")
        print(f"  ожидаемая разница на привате: ~{1.10*rms/200:.6f} RMSLE (sd), "
              f"т.е. второй слот стоит порядка 0.00005 — хеджа по разнообразию тут нет")


if __name__ == "__main__":
    main()
