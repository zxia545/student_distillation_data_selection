#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/data/scas_verified_teacher_pool}"
SOURCE_DATASET="${SOURCE_DATASET:-deepscaler}"
TEACHER_NAME="${TEACHER_NAME:-qwen3-32b}"
OUTPUT_DATASET_DIR="${OUTPUT_DATASET_DIR:-$REPO_ROOT/outputs/verified_teacher_pool/fixed_teacher/${SOURCE_DATASET}_${TEACHER_NAME}}"
DATASET_NAME="${DATASET_NAME:-fixed_teacher}"

INPUT_JSONL="$DATASET_ROOT/data/$SOURCE_DATASET/$TEACHER_NAME.jsonl"
if [[ ! -f "$INPUT_JSONL" ]]; then
  echo "Missing teacher file: $INPUT_JSONL" >&2
  echo "Download the dataset with:" >&2
  echo "  hf download Student-Centric-Answer-Sampling/scas_verified_teacher_pool --repo-type dataset --local-dir data/scas_verified_teacher_pool" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DATASET_DIR"

INPUT_JSONL="$INPUT_JSONL" \
OUTPUT_JSONL="$OUTPUT_DATASET_DIR/train.jsonl" \
DATASET_INFO_JSON="$OUTPUT_DATASET_DIR/dataset_info.json" \
DATASET_NAME="$DATASET_NAME" \
python - <<'PY'
import json
import os
from pathlib import Path

input_jsonl = Path(os.environ["INPUT_JSONL"])
output_jsonl = Path(os.environ["OUTPUT_JSONL"])
dataset_info_json = Path(os.environ["DATASET_INFO_JSON"])
dataset_name = os.environ["DATASET_NAME"]

count = 0
with input_jsonl.open("r", encoding="utf-8") as src, output_jsonl.open(
    "w", encoding="utf-8"
) as dst:
    for line in src:
        row = json.loads(line)
        dst.write(
            json.dumps(
                {
                    "system": row.get("system", ""),
                    "instruction": row["instruction"],
                    "output": row["teacher_output"],
                    "teacher_name": row["teacher_name"],
                    "source_dataset": row["source_dataset"],
                    "id": row["id"],
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        count += 1

dataset_info_json.write_text(
    json.dumps(
        {
            dataset_name: {
                "file_name": output_jsonl.name,
                "columns": {
                    "system": "system",
                    "prompt": "instruction",
                    "response": "output",
                },
            }
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
print(f"Prepared {count} rows at {output_jsonl}")
PY

DATASET_DIR="$OUTPUT_DATASET_DIR" \
DATASET_NAME="$DATASET_NAME" \
bash "$REPO_ROOT/scripts/train_sft.sh"
