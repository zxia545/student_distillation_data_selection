# Training and Evaluation

This repository provides the method-side training and evaluation utilities, but
does not vendor heavyweight training frameworks or benchmark suites.

## Train Selected Supervision

SCAS grouping writes LLaMA-Factory compatible files:

```text
outputs/grouped/scas_score/5_groups/
├── selected_group_1.jsonl
└── dataset_info.json
```

Launch SFT through the provided command builder:

```bash
DATASET_DIR=outputs/grouped/scas_score/5_groups \
DATASET_NAME=selected_group_1 \
MODEL_NAME_OR_PATH=/path/to/base-student \
OUTPUT_DIR=outputs/train/student_scas \
LLAMAFACTORY_TRAIN=/path/to/LLaMA-Factory/src/train.py \
DEEPSPEED_CONFIG=/path/to/LLaMA-Factory/examples/deepspeed/ds_z3_config.json \
NUM_GPUS=4 \
bash scripts/train_sft.sh
```

Use `DRY_RUN=1` to print the exact command without launching DeepSpeed.

## Generate Student Responses

Use an existing OpenAI-compatible endpoint:

```bash
INPUT_JSONL=data/validation.jsonl \
OUTPUT_JSONL=outputs/eval/generated.jsonl \
API_BASE=http://127.0.0.1:8000/v1 \
MODEL_NAME=student-scas \
bash scripts/generate_responses.sh
```

Or let the script start a local vLLM server:

```bash
INPUT_JSONL=data/validation.jsonl \
OUTPUT_JSONL=outputs/eval/generated.jsonl \
MODEL_PATH=outputs/train/student_scas \
START_VLLM=1 \
NUM_GPUS=1 \
bash scripts/generate_responses.sh
```

## Evaluate Math Answers

Rule-based evaluation works with the base installation:

```bash
MODEL_OUTPUT_JSONL=outputs/eval/generated.jsonl \
EVAL_OUTPUT_JSONL=outputs/eval/judged.jsonl \
REFERENCE_FIELD=reference_answer \
MODEL_OUTPUT_FIELD=model_output \
bash scripts/evaluate_math.sh
```

For ambiguous answers, enable an LLM judge with `EVAL_METHOD=llm` and provide a
judge endpoint through `JUDGE_API_BASE` and `JUDGE_MODEL`, or a local
`JUDGE_MODEL_PATH` with `JUDGE_START_VLLM=1`.

## End-to-End Wrapper

After selection, run train, generation, and evaluation as one workflow:

```bash
DATASET_DIR=outputs/grouped/scas_score/5_groups \
MODEL_NAME_OR_PATH=/path/to/base-student \
VALIDATION_JSONL=data/validation.jsonl \
RUN_ROOT=outputs/run_scas \
bash scripts/run_train_generate_eval.sh
```
