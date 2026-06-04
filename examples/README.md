# Examples

`examples/candidates/` contains a tiny three-teacher candidate pool used by the
public smoke test. It is intentionally synthetic and only checks file contracts.

`examples/validation/toy_generations.jsonl` contains generated-answer examples
for the rule-based math evaluator.

`examples/verified_teacher_pool/` contains scripts for using the released
verified teacher-answer pool either as fixed-teacher SFT data or as SCAS
candidate answers before training.

Run:

```bash
bash examples/run_demo.sh
```

Set `OUTPUT_ROOT`, `NUM_GROUPS`, `SELECTED_GROUP`, or `LAMBDA_SCAS` to override
the defaults.

Run a training dry-run after the demo:

```bash
DATASET_DIR=outputs/demo/grouped/scas_score/3_groups \
DATASET_NAME=selected_group_1 \
MODEL_NAME_OR_PATH=/path/to/base-student \
OUTPUT_DIR=outputs/demo/train_dry_run \
DRY_RUN=1 \
bash scripts/train_sft.sh
```

Run rule-based evaluation:

```bash
MODEL_OUTPUT_JSONL=examples/validation/toy_generations.jsonl \
EVAL_OUTPUT_JSONL=outputs/demo/eval_math.jsonl \
bash scripts/evaluate_math.sh
```
