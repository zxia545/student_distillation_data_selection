# Verified Teacher Pool Examples

These scripts show how to use the released verified teacher-answer pool with
the SCAS codebase.

Download the dataset first:

```bash
hf download Student-Centric-Answer-Sampling/scas_verified_teacher_pool \
  --repo-type dataset \
  --local-dir data/scas_verified_teacher_pool
```

Run a fixed-teacher training dry run:

```bash
DATASET_ROOT=data/scas_verified_teacher_pool \
SOURCE_DATASET=deepscaler \
TEACHER_NAME=qwen3-32b \
MODEL_NAME_OR_PATH=/path/to/base-student \
OUTPUT_DIR=outputs/train/deepscaler_qwen3_32b \
DRY_RUN=1 \
bash examples/verified_teacher_pool/train_fixed_teacher.sh
```

Run SCAS selection followed by a training dry run:

```bash
DATASET_ROOT=data/scas_verified_teacher_pool \
SOURCE_DATASET=deepscaler \
STUDENT_MODEL=/path/to/student-checkpoint \
MODEL_NAME_OR_PATH=/path/to/base-student \
OUTPUT_DIR=outputs/train/deepscaler_scas \
DRY_RUN=1 \
bash examples/verified_teacher_pool/select_with_scas_then_train.sh
```

For actual training, set `LLAMAFACTORY_TRAIN`, `DEEPSPEED_CONFIG`, `NUM_GPUS`,
and remove `DRY_RUN=1`.
