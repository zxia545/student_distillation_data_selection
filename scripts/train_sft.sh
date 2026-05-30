#!/usr/bin/env bash
set -euo pipefail

DATASET_DIR="${DATASET_DIR:?Set DATASET_DIR to a grouped SCAS directory containing dataset_info.json.}"
DATASET_NAME="${DATASET_NAME:-selected_group_1}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:?Set MODEL_NAME_OR_PATH to the base student checkpoint.}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR for the trained checkpoint.}"
LLAMAFACTORY_TRAIN="${LLAMAFACTORY_TRAIN:-src/train.py}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-examples/deepspeed/ds_z3_config.json}"
NUM_GPUS="${NUM_GPUS:-1}"

args=(
  --dataset-dir "$DATASET_DIR"
  --dataset-name "$DATASET_NAME"
  --model-name-or-path "$MODEL_NAME_OR_PATH"
  --output-dir "$OUTPUT_DIR"
  --llamafactory-train "$LLAMAFACTORY_TRAIN"
  --deepspeed-config "$DEEPSPEED_CONFIG"
  --num-gpus "$NUM_GPUS"
  --template "${TEMPLATE:-qwen}"
  --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
  --per-device-eval-batch-size "${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-2}"
  --learning-rate "${LEARNING_RATE:-1.0e-5}"
  --num-train-epochs "${NUM_TRAIN_EPOCHS:-3}"
  --cutoff-len "${CUTOFF_LEN:-4096}"
  --val-size "${VAL_SIZE:-0.001}"
  --warmup-ratio "${WARMUP_RATIO:-0.05}"
  --logging-steps "${LOGGING_STEPS:-10}"
  --num-checkpoints "${NUM_CHECKPOINTS:-4}"
  --preprocessing-num-workers "${PREPROCESSING_NUM_WORKERS:-16}"
  --seed "${SEED:-42}"
  --report-to "${REPORT_TO:-tensorboard}"
)

if [[ -n "${SAVE_STEPS:-}" ]]; then
  args+=(--save-steps "$SAVE_STEPS")
fi
if [[ "${RESUME:-0}" == "1" || "${RESUME:-0}" == "true" ]]; then
  args+=(--resume)
fi
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  args+=(--resume-from-checkpoint "$RESUME_FROM_CHECKPOINT")
fi
if [[ "${OVERWRITE_OUTPUT_DIR:-0}" == "1" || "${OVERWRITE_OUTPUT_DIR:-0}" == "true" ]]; then
  args+=(--overwrite-output-dir)
fi
if [[ "${SKIP_IF_COMPLETE:-0}" == "1" || "${SKIP_IF_COMPLETE:-0}" == "true" ]]; then
  args+=(--skip-if-complete)
fi
if [[ "${DRY_RUN:-0}" == "1" || "${DRY_RUN:-0}" == "true" ]]; then
  args+=(--dry-run)
fi

python -m scas.training.llamafactory "${args[@]}" "$@"
