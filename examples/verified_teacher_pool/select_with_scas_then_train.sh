#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/data/scas_verified_teacher_pool}"
SOURCE_DATASET="${SOURCE_DATASET:-deepscaler}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/outputs/verified_teacher_pool/$SOURCE_DATASET/scas}"
STUDENT_MODEL="${STUDENT_MODEL:?Set STUDENT_MODEL to the current student checkpoint for SCAS scoring.}"
LAMBDA_SCAS="${LAMBDA_SCAS:-0.5}"
SCORE_KEY="${SCORE_KEY:-scas_score}"
NUM_GROUPS="${NUM_GROUPS:-5}"
SELECTED_GROUP="${SELECTED_GROUP:-1}"
INSTALL_SCAS="${INSTALL_SCAS:-1}"

CANDIDATE_DIR="$DATASET_ROOT/data/$SOURCE_DATASET"
if [[ ! -d "$CANDIDATE_DIR" ]]; then
  echo "Missing candidate directory: $CANDIDATE_DIR" >&2
  echo "Download the dataset with:" >&2
  echo "  hf download Student-Centric-Answer-Sampling/scas_verified_teacher_pool --repo-type dataset --local-dir data/scas_verified_teacher_pool" >&2
  exit 1
fi

if [[ "$INSTALL_SCAS" == "1" || "$INSTALL_SCAS" == "true" ]]; then
  python -m pip install -e "$REPO_ROOT[model]"
fi

mkdir -p "$RUN_ROOT"

python -m scas.scoring.model_candidates \
  --student-model "$STUDENT_MODEL" \
  --candidate-dir "$CANDIDATE_DIR" \
  --output-dir "$RUN_ROOT/scored" \
  --lambda-scas "$LAMBDA_SCAS"

python -m scas.selection.group_by_score \
  --input-dir "$RUN_ROOT/scored" \
  --output-dir "$RUN_ROOT/grouped" \
  --score-key "$SCORE_KEY" \
  --num-groups "$NUM_GROUPS" \
  --selected-group "$SELECTED_GROUP"

DATASET_DIR="$RUN_ROOT/grouped/$SCORE_KEY/${NUM_GROUPS}_groups" \
DATASET_NAME="selected_group_$SELECTED_GROUP" \
bash "$REPO_ROOT/scripts/train_sft.sh"
