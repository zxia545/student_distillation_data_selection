#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATASET_DIR="${DATASET_DIR:?Set DATASET_DIR to grouped SCAS training data.}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:?Set MODEL_NAME_OR_PATH to the base student checkpoint.}"
VALIDATION_JSONL="${VALIDATION_JSONL:?Set VALIDATION_JSONL to validation prompts with references.}"
RUN_ROOT="${RUN_ROOT:-outputs/scas_run}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-$RUN_ROOT/model}"
GENERATED_JSONL="${GENERATED_JSONL:-$RUN_ROOT/generated_responses.jsonl}"
EVAL_OUTPUT_JSONL="${EVAL_OUTPUT_JSONL:-$RUN_ROOT/evaluated_responses.jsonl}"

DATASET_DIR="$DATASET_DIR" \
DATASET_NAME="${DATASET_NAME:-selected_group_1}" \
MODEL_NAME_OR_PATH="$MODEL_NAME_OR_PATH" \
OUTPUT_DIR="$TRAIN_OUTPUT_DIR" \
bash "$REPO_ROOT/scripts/train_sft.sh"

if [[ "${DRY_RUN:-0}" == "1" || "${DRY_RUN:-0}" == "true" ]]; then
  echo "[DRY_RUN] training command printed; generation and evaluation were not started"
  exit 0
fi

INPUT_JSONL="$VALIDATION_JSONL" \
OUTPUT_JSONL="$GENERATED_JSONL" \
MODEL_PATH="${EVAL_MODEL_PATH:-$TRAIN_OUTPUT_DIR}" \
MODEL_NAME="${MODEL_NAME:-$(basename "${EVAL_MODEL_PATH:-$TRAIN_OUTPUT_DIR}")}" \
bash "$REPO_ROOT/scripts/generate_responses.sh"

MODEL_OUTPUT_JSONL="$GENERATED_JSONL" \
EVAL_OUTPUT_JSONL="$EVAL_OUTPUT_JSONL" \
bash "$REPO_ROOT/scripts/evaluate_math.sh"

echo "SCAS train/generate/evaluate outputs written under $RUN_ROOT"
