#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/demo}"
NUM_GROUPS="${NUM_GROUPS:-3}"
SELECTED_GROUP="${SELECTED_GROUP:-1}"
SCORE_KEY="${SCORE_KEY:-scas_score}"
LAMBDA_SCAS="${LAMBDA_SCAS:-0.5}"

python -m scas.scoring.score_demo_candidates \
  --input-dir "$REPO_ROOT/examples/candidates" \
  --output-dir "$OUTPUT_ROOT/scored" \
  --lambda-scas "$LAMBDA_SCAS"

python -m scas.selection.group_by_score \
  --input-dir "$OUTPUT_ROOT/scored" \
  --output-dir "$OUTPUT_ROOT/grouped" \
  --score-key "$SCORE_KEY" \
  --num-groups "$NUM_GROUPS" \
  --selected-group "$SELECTED_GROUP"

echo "Demo outputs written under $OUTPUT_ROOT"
