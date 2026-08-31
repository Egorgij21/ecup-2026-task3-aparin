#!/usr/bin/env bash
#
# Полный прогон решения: сырые данные площадки -> файл сабмита.
#
#   bash run.sh quick     собрать сабмит из готовых предсказаний компонентов
#                         секунды, без GPU, данные площадки не нужны
#   bash run.sh all       обучить все 13 моделей с нуля и собрать сабмит
#                         нужен GPU; ~15 GPU-часов + ~8 CPU-часов
#                         разбиение на фолды берётся готовым из data/
#
#   bash run.sh folds     пересоздать разбиение из train.parquet (в 'all' не входит:
#                         готовое лежит в data/folds.parquet)
#   bash run.sh gbdt      только слот GBDT (2 модели, CPU, ~7.5 ч)
#   bash run.sh seq       только слот SEQ (7 моделей, 1 GPU, ~40 мин)
#   bash run.sh usercv    только слот USERCV (4 модели, 1 GPU, 1-3 ч на модель)
#   bash run.sh blend     только сборка из ваших предсказаний в subs/ (после обучения)
#
# ГДЕ ЧТО ЛЕЖИТ
#   reference/  предсказания 13 компонентов, из которых собран ОТПРАВЛЕННЫЙ сабмит.
#               Только чтение; `quick` берёт их и получает файл бит-в-бит.
#   subs/       сюда пишут этапы обучения. `blend` собирает из них.
#               Совпадения бит-в-бит здесь НЕ будет: сети не детерминированы,
#               порог приёмки — corr >= 0.9999 с отправленным файлом.
#
# ДАННЫЕ. Положите файлы площадки в data/ перед запуском `all`:
#   data/train.parquet          30.6 млн строк, 250 000 user_id, 2025-01-01..2026-02-13
#   data/sample_submit.csv      эталон порядка user_id (sample_submission.csv площадки)
#
# ОКРУЖЕНИЕ. `quick` требует solution/requirements.txt, `all` — requirements-full.txt.
# Python задаётся переменной PY, по умолчанию python3:
#   PY=./.venv/bin/python bash run.sh quick

set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-python3}"

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
need() { [ -f "$1" ] || { echo "НЕТ ФАЙЛА: $1"; echo "  $2"; exit 1; }; }

stage_folds() {
  say "фолды — ПЕРЕСОЗДАНИЕ поверх того, что лежит в data/"
  need data/train.parquet "положите train.parquet площадки в data/ (см. шапку run.sh)"
  # Готовое разбиение уже лежит в data/folds.parquet и едет вместе с репозиторием.
  # Этот этап нужен, только если вы хотите убедиться, что оно воспроизводится из
  # train.parquet. Он ПЕРЕЗАПИШЕТ файл; содержимое должно получиться идентичным.
  echo "  (в 'all' этот этап НЕ входит: разбиение поставляется готовым)"
  $PY src/folds.py
}

stage_gbdt() {
  say "слот GBDT (вес 0.20): e0266 + e0064 — CPU, ~7.5 ч, 250-300 ГБ RAM"
  need data/train.parquet "положите train.parquet площадки в data/"
  $PY src/run_regime.py   --config configs/e0260_regime.yaml
  $PY src/predict_seeds.py --config configs/e0260_regime.yaml --arm e0266 --out e0266
  $PY src/run_ag.py --config configs/e0060_filt_top400.yaml --exp-id e0064 --mode full \
      --presets medium_quality --time-limit 9000 --cache
}

stage_seq() {
  say "слот SEQ (вес 0.38): 7 моделей на сырых каналах — 1 GPU, ~40 мин суммарно"
  need data/train.parquet "положите train.parquet площадки в data/"
  for c in e0100_seq_tcn e0101_seq_gru e0101s1_seed e0101s2_seed e0101s3_seed \
           e0102_seq_xformer e0108_gru_deep; do
    echo "--- $c"
    $PY src/run_seq.py --config "configs/$c.yaml"                # CV + OOF
    $PY src/run_seq.py --config "configs/$c.yaml" --mode submit  # предсказание на тесте
  done
}

stage_usercv() {
  say "слот USERCV (вес 0.42): 4 модели, user-split — 1 GPU, 1-3 ч на модель"
  need data/train.parquet "положите train.parquet площадки в data/"
  # CV пишет json с медианной лучшей эпохой, его читает предсказатель при --epochs 0
  $PY src/run_usercv.py --variant full --folds 0 1 2 3 4 --seeds 3
  $PY src/predict_usercv.py --variant full --exp-id e0141          --model gru         --hidden 128 --seeds 3 --epochs 0
  $PY src/predict_usercv.py --variant full --exp-id e0295_usercv48 --model gru         --hidden 48  --seeds 3 --epochs 32
  $PY src/predict_usercv.py --variant full --exp-id e0340_lstm48   --model lstm        --hidden 48  --seeds 3 --epochs 32
  $PY src/predict_usercv.py --variant full --exp-id e0341_xf48     --model transformer --hidden 48  --seeds 3 --epochs 32
}

stage_quick() {
  say "сборка сабмита из ЭТАЛОННЫХ предсказаний компонентов (reference/)"
  $PY solution/build_submission.py --components-dir reference --out-dir submission
  $PY solution/verify_submission.py submission/e0303_arch4_cal.csv
  echo
  echo "готово: submission/e0303_arch4_cal.csv"
}

stage_blend() {
  say "сборка сабмита из ВАШИХ переобученных предсказаний (subs/)"
  $PY solution/build_submission.py --components-dir subs --out-dir submission
  $PY solution/verify_submission.py submission/e0303_arch4_cal.csv
  echo
  echo "готово: submission/e0303_arch4_cal.csv"
  echo "сравнение с отправленным файлом — в строке self-test выше"
}

case "${1:-quick}" in
  quick)  stage_quick ;;
  folds)  stage_folds ;;
  gbdt)   stage_gbdt ;;
  seq)    stage_seq ;;
  usercv) stage_usercv ;;
  blend)  stage_blend ;;
  all)    stage_gbdt; stage_seq; stage_usercv; stage_blend ;;
  *)      sed -n '3,20p' "$0"; exit 1 ;;
esac
