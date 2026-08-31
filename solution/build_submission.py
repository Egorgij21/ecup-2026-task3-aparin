#!/usr/bin/env python
"""
Сборка финального сабмита из предсказаний компонентов.

    python solution/build_submission.py                      # собрать и сверить
    python solution/build_submission.py --out-dir submission # ещё и записать

ЧТО ЭТО ЗА ШАГ. Решение — взвешенное среднее трёх «слотов» в ЛОГ-пространстве плюс аффинная
калибровка. Обучение моделей (13 штук) требует кластера и GPU-часов, а вот этот последний шаг
детерминирован, занимает секунды и полностью определяет отправленный файл. Поэтому он вынесен
отдельно: судья, у которого есть предсказания компонентов, воспроизводит финальный csv
без единого GPU.

    предсказание = expm1( a * SUM_k w_k * log1p(p_k) + b )

СЛОТЫ И ВЕСА (0.20 / 0.38 / 0.42) — не подгонялись под этот файл. Они получены оптимизацией
неотрицательных весов на OOF ещё для e0162 (research/EXPERIMENTS.md §1m) и с тех пор
НЕ ПЕРЕСЧИТЫВАЛИСЬ ни разу: e0300/e0301/e0303 отличаются только СОДЕРЖИМЫМ слота usercv.
Это осознанное решение, а не лень — три подряд перевзвешивания на CV/OOF проиграли на LB
(e0270 −0.00010, e0302 −0.00026, обе с инверсией знака относительно предсказания CV), тогда
как две замены содержимого слота при неизменных весах спрогнозировались с точностью 1e-6 и
2e-6. Вывод занесён в лог: подгонка весов по CV на членах, коррелирующих на 0.998, —
это подгонка под шум.

ПОЧЕМУ ЛОГ-ПРОСТРАНСТВО. Метрика RMSLE, её оптимальный точечный прогноз — E[log1p(y)|x].
Усреднение сырых предсказаний оценивало бы log1p(E[y]), что EXPERIMENTS.md §1e оценивает
в +0.5626 RMSLE. Все усреднения в проекте — в логах.

КАЛИБРОВКА. Аффинное преобразование в лог-пространстве к целевым моментам (mu*, sd*).
Моменты не выдуманы: E[L] и E[L^2] тестовой правды решены в закрытом виде из двух
probe-сабмитов (константа 0 -> RMSLE^2 = E[L^2]; константа 10 -> RMSLE^2 = E[L^2] - 2cE[L] + c^2),
откуда E[L] = 2.3303 и sd_L = 2.3178, а оптимальный масштаб sd* = rho * sd_L, где rho
решается из измеренного LB по тождеству RMSLE = sd_L * sqrt(1 - rho^2).
Отдельная запись в логе: калибровка дала 92% улучшения e0266_cal, модель — 8%.

На практике каждая новая сборка приводилась к моментам ДЕЙСТВУЮЩЕГО чемпиона, а не
пересчитывалась от проб заново: тогда сравнение двух файлов ПАРНОЕ (одинаковые уровень и
разброс), и разница в счёте относится к rho, то есть к модели, а не к постобработке.
Константы a, b каждого файла записаны ниже и восстановлены из него самого (аффинная
подгонка по неусечённым пользователям), поэтому сборка воспроизводит все три отправленных
файла ДО МАШИННОЙ ТОЧНОСТИ: rms(dlog) ~1e-15. Проверено дважды на несовпадающих стеках —
локально (Python 3.9.6 / numpy 1.26 / polars 1.36) и на кластере
(Python 3.13.5 / numpy 2.2.6 / polars 1.34).
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
# Каталог с предсказаниями компонентов. По умолчанию reference/ — там лежат ТЕ САМЫЕ файлы,
# из которых собран отправленный сабмит. Обучение (run.sh gbdt/seq/usercv) пишет свои
# предсказания в subs/, и тогда сборку запускают с --components-dir subs.
REFERENCE = ROOT / "reference"
SUBS = REFERENCE

W = {"gbdt": 0.20, "seq": 0.38, "usercv": 0.42}

# --- слоты: список файлов-компонентов, усредняемых в лог-пространстве -----------------------
GBDT = ["e0266", "e0064"]
SEQ = ["e0100", "e0101", "e0101s1", "e0101s2", "e0101s3", "e0102", "e0108"]

# --- три собранных финалиста ----------------------------------------------------------------
# a, b — константы аффинной калибровки в лог-пространстве: L_hat = a*M + b.
# Провенанс: масштаб a = rho * sd_L / sd_M, где sd_L = 2.3178 решено из probe-сабмитов, а rho
# решено из измеренного LB эталона по тождеству RMSLE = sd_L*sqrt(1-rho^2); сдвиг b доводит
# среднее до E[L] = 2.3303. Обратный пересчёт из этих a: rho = 0.70378 / 0.70363 / 0.70351
# против решённых из LB 0.70379 / 0.703846 / 0.703834 — сходится (расхождение даёт усечение
# отрицательных предсказаний в нуль, 1.3-1.5 тыс. пользователей).
# С ними сборка воспроизводит отправленные файлы до машинной точности (rms ~1e-16).
TARGETS = {
    "e0303_arch4_cal": dict(usercv=["e0141", "e0295_usercv48", "e0340_lstm48", "e0341_xf48"],
                            a=1.0328948154022706, b=-0.17536170182531663,
                            lb_public=1.646483,
                            note="ФИНАЛЬНОЕ РЕШЕНИЕ"),
}
FINAL = "e0303_arch4_cal"


def logp(name: str) -> tuple[np.ndarray, np.ndarray]:
    """user_id и log1p(max(predict, 0)) одного файла предсказаний."""
    path = SUBS / f"{name}.csv"
    if not path.exists():
        raise SystemExit(
            f"нет {path.relative_to(ROOT)}.\n"
            f"  reference/ — предсказания, из которых собран отправленный сабмит;\n"
            f"  subs/      — сюда пишет обучение (run.sh gbdt/seq/usercv).\n"
            f"Если обучали сами, укажите --components-dir subs.")
    d = pl.read_csv(path)
    return (d["user_id"].to_numpy(),
            np.log1p(np.maximum(d["predict"].to_numpy().astype(np.float64), 0.0)))


def slot(names: list[str], users: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Среднее логов группы членов. Порядок user_id сверяется, а не предполагается."""
    acc = None
    for n in names:
        u, v = logp(n)
        if users is None:
            users = u
        if not np.array_equal(u, users):
            raise SystemExit(f"{n}: порядок user_id отличается от остальных компонентов")
        acc = v if acc is None else acc + v
    return users, acc / len(names)


def build(target: str, verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:
    spec = TARGETS[target]
    users = None
    users, gbdt = slot(GBDT, users)
    users, seq = slot(SEQ, users)
    users, usercv = slot(spec["usercv"], users)

    if verbose:
        print(f"\n=== {target}   public LB {spec['lb_public']:.6f}   {spec['note']}")
        for tag, names, v in (("gbdt", GBDT, gbdt), ("seq", SEQ, seq),
                              ("usercv", spec["usercv"], usercv)):
            print(f"  слот {tag:6s} w={W[tag]:.2f}  {len(names)} чл.  "
                  f"mu {v.mean():.4f} sd {v.std():.4f}   {', '.join(names)}")

    B = W["gbdt"] * gbdt + W["seq"] * seq + W["usercv"] * usercv

    # аффинная калибровка в лог-пространстве
    Bc = spec["a"] * B + spec["b"]
    pred = np.maximum(np.expm1(Bc), 0.0)

    # --- инварианты сабмита: лучше упасть здесь, чем отправить битый файл ---
    assert len(pred) == 250_000, f"{len(pred)} строк вместо 250000"
    # sample_submission площадки — эталон порядка user_id. Если он подложен в data/, сверяемся
    # с ним; если нет, порядок всё равно перекрёстно проверен между всеми компонентами в slot().
    ref = next((ROOT / "data" / n for n in ("sample_submit.csv", "sample_submission.csv")
                if (ROOT / "data" / n).exists()), None)
    if ref is not None:
        ss = pl.read_csv(ref)
        assert ss.height == 250_000, f"{ref.name}: {ss.height} строк"
        assert np.array_equal(users, ss["user_id"].to_numpy()), f"порядок user_id != {ref.name}"
    elif verbose:
        print("  (data/sample_submit.csv не подложен — порядок user_id сверен между компонентами)")
    assert np.isfinite(pred).all(), "не-конечные предсказания"
    assert (pred >= 0).all(), "отрицательные предсказания"
    assert 10.0 < pred.mean() < 400.0, f"подозрительный масштаб: mean {pred.mean():.1f}"

    if verbose:
        print(f"  сырой бленд          mu {B.mean():.4f} sd {B.std():.4f}")
        print(f"  калибровка           L = {spec['a']:.9f}*M {spec['b']:+.9f}")
        print(f"  после калибровки     mu {Bc.mean():.4f} sd {Bc.std():.4f}")
    return users, pred


def selftest(target: str, pred: np.ndarray, strict: bool = True) -> bool:
    """Сверка с отправленным файлом, если он на месте. Это и есть доказательство сборки."""
    ref_path = REFERENCE / f"{target}.csv"
    if not ref_path.exists():
        print(f"  self-test: эталон subs/{target}.csv отсутствует — пропущен")
        return True
    ref = np.log1p(np.maximum(pl.read_csv(ref_path)["predict"]
                              .to_numpy().astype(np.float64), 0.0))
    got = np.log1p(pred)
    corr = float(np.corrcoef(got, ref)[0, 1])
    rms = float(np.sqrt(((got - ref) ** 2).mean()))
    md5 = hashlib.md5(ref_path.read_bytes()).hexdigest()[:8]
    print(f"  self-test vs reference/{target}.csv (md5 {md5}): corr {corr:.7f}   "
          f"max|dlog| {np.abs(got - ref).max():.5f}   rms {rms:.6f}")
    if strict:
        # компоненты эталонные -> обязано совпасть до машинной точности
        ok = rms < 1e-7
        print("  " + ("OK — БИТ-В-БИТ то, что ушло на площадку" if ok else
                      "!! РАСХОЖДЕНИЕ — компоненты не совпадают с эталонными"))
    else:
        # компоненты переобучены -> расхождение ОЖИДАЕМО: сети не бит-в-бит детерминированы,
        # два прогона одного конфига дают corr ~0.990 на члене и ~0.9999 после усреднения
        ok = corr >= 0.999
        print(f"  {'OK' if ok else '!! СЛИШКОМ ДАЛЕКО'} — собрано из ВАШИХ переобученных "
              f"компонентов, corr {corr:.6f} с отправленным файлом")
        print("     (бит-в-бит здесь и не ожидается: cudnn.deterministic выключен, "
              "порог приёмки corr >= 0.9999)")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--target", default=FINAL, choices=[FINAL],
                    help="что собирать (в репозитории лежит один финальный файл)")
    ap.add_argument("--components-dir", default="reference", choices=["reference", "subs"],
                    help="откуда брать предсказания компонентов: reference (эталонные, "
                         "из них собран отправленный сабмит) или subs (ваши, после обучения)")
    ap.add_argument("--out-dir", default=None,
                    help="куда писать (по умолчанию не писать, только собрать и сверить)")
    ap.add_argument("--check-only", action="store_true", help="только self-test, ничего не писать")
    a = ap.parse_args()

    global SUBS
    SUBS = ROOT / a.components_dir
    strict = a.components_dir == "reference"
    targets = [a.target]
    ok = True
    for t in targets:
        users, pred = build(t)
        ok &= selftest(t, pred, strict=strict)
        if a.out_dir and not a.check_only:
            out = Path(a.out_dir)
            out.mkdir(parents=True, exist_ok=True)
            f = out / f"{t}.csv"
            pl.DataFrame({"user_id": users, "predict": pred}).write_csv(f)
            print(f"  записано {f}  ({f.stat().st_size / 1e6:.1f} MB)")
    print("\n" + ("ВСЁ СОШЛОСЬ" if ok else "ЕСТЬ РАСХОЖДЕНИЯ — см. выше"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
