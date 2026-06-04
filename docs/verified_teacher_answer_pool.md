# Verified Teacher Answer Pool

This repository provides utilities for working with the SCAS verified
teacher-answer pool hosted on Hugging Face:

```text
Student-Centric-Answer-Sampling/scas_verified_teacher_pool
```

The dataset contains aligned, correctness-verified teacher-generated
mathematical reasoning solutions for Hendrycks MATH and DeepScaleR. For each
source corpus, all released teacher files share the same retained question ids:

| Source corpus | Aligned questions | Teachers | Teacher responses |
|---|---:|---:|---:|
| Hendrycks MATH | 4,256 | 9 | 38,304 |
| DeepScaleR | 9,551 | 9 | 85,959 |
| **Total** | 13,807 | 9 | 124,263 |

This alignment makes the data useful for two complementary workflows:

1. Fixed-teacher distillation, where a student is trained on one teacher's
   verified solutions.
2. Student-centric answer selection, where SCAS scores multiple verified
   teacher solutions for each question and trains on the selected supervision.

## Download

Install the Hugging Face CLI if needed, then download the JSONL layout:

```bash
hf download Student-Centric-Answer-Sampling/scas_verified_teacher_pool \
  --repo-type dataset \
  --local-dir data/scas_verified_teacher_pool
```

The downloaded structure is:

```text
data/scas_verified_teacher_pool/
├── README.md
├── data/
│   ├── deepscaler/
│   │   └── <teacher_name>.jsonl
│   └── hendrycks_math/
│       └── <teacher_name>.jsonl
└── metadata/
    ├── manifest.json
    ├── stats.json
    └── teacher_models.json
```

Each source-corpus directory contains one JSONL file per teacher. Pass a single
source-corpus directory, such as `data/scas_verified_teacher_pool/data/deepscaler`,
to the SCAS scorer.

## Schema

Each JSONL row contains:

| Field | Meaning |
|---|---|
| `id` | Source question id, aligned across all teachers within a source corpus. |
| `source_dataset` | `deepscaler` or `hendrycks_math`. |
| `teacher_name` | Teacher model that produced the solution. |
| `system` | System prompt used during generation, if available. |
| `instruction` | Math question / user prompt. |
| `teacher_output` | Verified teacher-generated reasoning solution. |
| `reference_answer` | Short answer when available. |
| `reference_solution` | Reference solution or source-provided answer text when available. |
| `is_common_correct` | Always `true` for released rows. |

Additional provenance fields are preserved when available. See the dataset card
for the full schema.

## Fixed-Teacher Distillation

Use this workflow to train a student directly on one teacher's verified
solutions.

```bash
DATASET_ROOT=data/scas_verified_teacher_pool \
SOURCE_DATASET=deepscaler \
TEACHER_NAME=qwen3-32b \
OUTPUT_DATASET_DIR=outputs/fixed_teacher/deepscaler_qwen3_32b \
bash examples/verified_teacher_pool/train_fixed_teacher.sh
```

The script converts `teacher_output` to the SFT `output` field, writes a
LLaMA-Factory-compatible `dataset_info.json`, and calls `scripts/train_sft.sh`.
Set the standard training environment variables before launching actual
training:

```bash
MODEL_NAME_OR_PATH=/path/to/base-student \
OUTPUT_DIR=outputs/train/deepscaler_qwen3_32b \
LLAMAFACTORY_TRAIN=/path/to/LLaMA-Factory/src/train.py \
DEEPSPEED_CONFIG=/path/to/LLaMA-Factory/examples/deepspeed/ds_z3_config.json \
NUM_GPUS=4 \
bash examples/verified_teacher_pool/train_fixed_teacher.sh
```

Set `DRY_RUN=1` to print the training command without starting DeepSpeed.

## SCAS Selection Then Training

Use this workflow to score all teacher solutions with a student checkpoint,
select low-cost supervision with SCAS, and train on the selected group.

```bash
DATASET_ROOT=data/scas_verified_teacher_pool \
SOURCE_DATASET=deepscaler \
STUDENT_MODEL=/path/to/student-checkpoint \
MODEL_NAME_OR_PATH=/path/to/base-student \
OUTPUT_DIR=outputs/train/deepscaler_scas \
LLAMAFACTORY_TRAIN=/path/to/LLaMA-Factory/src/train.py \
DEEPSPEED_CONFIG=/path/to/LLaMA-Factory/examples/deepspeed/ds_z3_config.json \
NUM_GPUS=4 \
bash examples/verified_teacher_pool/select_with_scas_then_train.sh
```

The script:

1. Optionally installs this package with model-scoring dependencies.
2. Scores all teacher JSONL files for the selected source corpus.
3. Groups candidates by ascending SCAS score.
4. Trains on `selected_group_1`, the lowest-cost selected group.

Important environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `DATASET_ROOT` | `data/scas_verified_teacher_pool` | Downloaded dataset root. |
| `SOURCE_DATASET` | `deepscaler` | Source corpus to use. |
| `STUDENT_MODEL` | required | Student checkpoint used for SCAS scoring. |
| `LAMBDA_SCAS` | `0.5` | Weight on the answer-question SCAS component. |
| `NUM_GROUPS` | `5` | Number of rank groups. |
| `SELECTED_GROUP` | `1` | Group used for training. |
| `RUN_ROOT` | `outputs/verified_teacher_pool/<source>` | Output root for scoring and grouping. |
| `INSTALL_SCAS` | `1` | Install this package before scoring. |
| `DRY_RUN` | `0` | Pass through to `scripts/train_sft.sh`. |

## Notes

- Candidate answers are already correctness-filtered in the released pool.
- The SCAS scorer expects one JSONL file per teacher in the candidate directory.
- Grouping uses ascending score; `group_1` is the lowest-cost group.
