#!/usr/bin/env bash
set -euo pipefail

MODEL_OUTPUT_JSONL="${MODEL_OUTPUT_JSONL:?Set MODEL_OUTPUT_JSONL to generated responses.}"
EVAL_OUTPUT_JSONL="${EVAL_OUTPUT_JSONL:?Set EVAL_OUTPUT_JSONL for evaluated responses.}"

args=(
  --input-jsonl "$MODEL_OUTPUT_JSONL"
  --output-jsonl "$EVAL_OUTPUT_JSONL"
  --method "${EVAL_METHOD:-rule}"
  --reference-field "${REFERENCE_FIELD:-reference_answer}"
  --model-output-field "${MODEL_OUTPUT_FIELD:-model_output}"
  --port "${JUDGE_PORT:-8001}"
  --tensor-parallel-size "${JUDGE_TENSOR_PARALLEL_SIZE:-${JUDGE_NUM_GPUS:-1}}"
  --num-workers "${JUDGE_NUM_WORKERS:-16}"
)

if [[ -n "${JUDGE_API_BASE:-}" ]]; then
  args+=(--api-base "$JUDGE_API_BASE")
fi
if [[ -n "${JUDGE_MODEL:-}" ]]; then
  args+=(--judge-model "$JUDGE_MODEL")
fi
if [[ -n "${JUDGE_MODEL_PATH:-}" ]]; then
  args+=(--judge-model-path "$JUDGE_MODEL_PATH")
fi
if [[ "${JUDGE_START_VLLM:-0}" == "1" || "${JUDGE_START_VLLM:-0}" == "true" ]]; then
  args+=(--start-vllm)
fi
if [[ -n "${JUDGE_CUDA_VISIBLE_DEVICES:-}" ]]; then
  args+=(--gpus "$JUDGE_CUDA_VISIBLE_DEVICES")
fi
if [[ "${DRY_RUN:-0}" == "1" || "${DRY_RUN:-0}" == "true" ]]; then
  printf '[DRY_RUN]'
  printf ' %q' python -m scas.evaluation.judge_math "${args[@]}" "$@"
  printf '\n'
  exit 0
fi

python -m scas.evaluation.judge_math "${args[@]}" "$@"
