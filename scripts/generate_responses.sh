#!/usr/bin/env bash
set -euo pipefail

INPUT_JSONL="${INPUT_JSONL:?Set INPUT_JSONL to a validation JSONL file.}"
OUTPUT_JSONL="${OUTPUT_JSONL:?Set OUTPUT_JSONL for generated responses.}"
MODEL_NAME="${MODEL_NAME:-}"
MODEL_PATH="${MODEL_PATH:-}"
API_BASE="${API_BASE:-}"

args=(
  --input-jsonl "$INPUT_JSONL"
  --output-jsonl "$OUTPUT_JSONL"
  --port "${PORT:-8000}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-${NUM_GPUS:-1}}"
  --max-tokens "${MAX_TOKENS:-2048}"
  --temperature "${TEMPERATURE:-0.7}"
  --num-workers "${NUM_WORKERS:-32}"
)

if [[ -n "$API_BASE" ]]; then
  args+=(--api-base "$API_BASE")
fi
if [[ -n "$MODEL_NAME" ]]; then
  args+=(--model "$MODEL_NAME")
fi
if [[ -n "$MODEL_PATH" ]]; then
  args+=(--model-path "$MODEL_PATH")
fi
if [[ "${START_VLLM:-0}" == "1" || "${START_VLLM:-0}" == "true" ]]; then
  args+=(--start-vllm)
fi
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  args+=(--gpus "$CUDA_VISIBLE_DEVICES")
fi
if [[ "${DRY_RUN:-0}" == "1" || "${DRY_RUN:-0}" == "true" ]]; then
  printf '[DRY_RUN]'
  printf ' %q' python -m scas.generation.generate_responses "${args[@]}" "$@"
  printf '\n'
  exit 0
fi

python -m scas.generation.generate_responses "${args[@]}" "$@"
