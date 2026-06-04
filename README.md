<h1 align="center">Student-Centric Answer Sampling (SCAS)</h1>

<p align="center">
  <strong>Open-source implementation for student-centric answer selection in LLM distillation</strong>
</p>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2605.26872-b31b1b.svg)](https://arxiv.org/abs/2605.26872)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Package](https://img.shields.io/badge/package-pyproject.toml-5B7FFF.svg)](pyproject.toml)

</div>

## Overview

This repository accompanies the paper
[The Strongest Teacher Is Not Always the Best Teacher: Student-Centric Answer Selection](https://arxiv.org/abs/2605.26872).
The paper studies teacher-generated supervision for LLM distillation and shows
that the strongest teacher by benchmark accuracy does not necessarily provide
the best training answer for a particular student model.

<div align="center">
  <img src="assets/scas_workflow.jpg" alt="SCAS workflow" width="88%" />
</div>

## Installation

Install the base package for JSONL processing, candidate selection, command
construction, and rule-based evaluation:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Install optional dependencies for model scoring and runtime integrations:

```bash
# Student-model SCAS scoring
pip install -e ".[model]"

# OpenAI-compatible generation, LLM judging, and vLLM helpers
pip install -e ".[runtime]"

# Full local setup
pip install -e ".[full]"
```

LLaMA-Factory is not vendored. Pass local paths through
`LLAMAFACTORY_TRAIN` and `DEEPSPEED_CONFIG` when launching SFT.

## Candidate Data

SCAS expects one JSONL file per teacher. Each prompt id should be shared across
teacher files so candidates can be ranked against one another:

```text
candidate_answers/
├── teacher_a_student_train.jsonl
├── teacher_b_student_train.jsonl
└── teacher_c_student_train.jsonl
```

Each row must contain a prompt id, an instruction, and a candidate answer:

```json
{"id": "item-1", "instruction": "Question text", "teacher_output": "Answer text"}
```

Candidate answers should be verified before SCAS ranking. For tasks with
reference answers, this repository includes lightweight filtering utilities;
for other tasks, use a task-appropriate verifier and pass only valid candidate
answers to the scorer.

See [docs/data_format.md](docs/data_format.md) for the full schema.

## Verified Teacher Answer Pool

We also provide the SCAS verified teacher-answer pool on Hugging Face:

```text
Student-Centric-Answer-Sampling/scas_verified_teacher_pool
```

The release contains aligned, correctness-verified teacher-generated solutions
for Hendrycks MATH and DeepScaleR. It can be used either as fixed-teacher
distillation data or as the candidate-answer pool for SCAS selection.

Download:

```bash
hf download Student-Centric-Answer-Sampling/scas_verified_teacher_pool \
  --repo-type dataset \
  --local-dir data/scas_verified_teacher_pool
```

See [docs/verified_teacher_answer_pool.md](docs/verified_teacher_answer_pool.md)
for the full data description and
[examples/verified_teacher_pool](examples/verified_teacher_pool/) for runnable
fixed-teacher and SCAS-selection training workflows.

## SCAS Workflow

### 1. Score Candidate Answers

Run the model-based scorer with the current student checkpoint:

```bash
python -m scas.scoring.model_candidates \
  --student-model /path/to/student-checkpoint \
  --candidate-dir candidate_answers \
  --output-dir outputs/scored \
  --lambda-scas 0.5
```

By default, the scorer uses the final `model.layers.*.mlp.up_proj` layer for
Llama/Qwen-style decoder models. For other architectures, pass
`--target-layer` explicitly.

The scorer writes one file per teacher:

```text
outputs/scored/
├── teacher_a_scas_scores.jsonl
├── teacher_b_scas_scores.jsonl
├── teacher_c_scas_scores.jsonl
└── manifest.json
```

### 2. Select Supervision

Group candidates by ascending score and export the selected group:

```bash
python -m scas.selection.group_by_score \
  --input-dir outputs/scored \
  --output-dir outputs/grouped \
  --score-key scas_score \
  --num-groups 5 \
  --selected-group 1
```

`group_1` is the lowest-cost group. The selected SFT file is:

```text
outputs/grouped/scas_score/5_groups/selected_group_1.jsonl
```

For deterministic minimum-score selection instead of group sampling:

```bash
python -m scas.selection.select_min_score \
  --input-dir outputs/scored \
  --output-dir outputs/min_selected \
  --score-key scas_score
```

### 3. Train the Student

Launch SFT through LLaMA-Factory:

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

Set `DRY_RUN=1` to print the DeepSpeed command without starting training.

### 4. Generate and Evaluate

Generate validation responses with an OpenAI-compatible endpoint:

```bash
INPUT_JSONL=data/validation.jsonl \
OUTPUT_JSONL=outputs/eval/generated.jsonl \
API_BASE=http://127.0.0.1:8000/v1 \
MODEL_NAME=student-scas \
bash scripts/generate_responses.sh
```

Evaluate generated math-style answers:

```bash
MODEL_OUTPUT_JSONL=outputs/eval/generated.jsonl \
EVAL_OUTPUT_JSONL=outputs/eval/judged.jsonl \
REFERENCE_FIELD=reference_answer \
MODEL_OUTPUT_FIELD=model_output \
bash scripts/evaluate_math.sh
```

For local vLLM serving, set `MODEL_PATH=/path/to/checkpoint` and
`START_VLLM=1`. For LLM-based judging, set `EVAL_METHOD=llm`,
`JUDGE_API_BASE`, and `JUDGE_MODEL`.

## Examples

The [examples](examples/) directory contains small synthetic JSONL fixtures for
smoke tests and runnable workflows for the verified teacher-answer pool. The
synthetic fixtures are only for file-contract checks and should not be
interpreted as evidence for method performance.

## Project Structure

```text
student_distillation_data_selection/
├── scas/
│   ├── scoring/                    # SCAS metric and candidate scoring
│   ├── selection/                  # Group and minimum-score selection
│   ├── training/                   # LLaMA-Factory command builder
│   ├── generation/                 # OpenAI-compatible generation
│   ├── evaluation/                 # Teacher filtering and answer judging
│   ├── runtime/                    # vLLM server helpers
│   └── io.py                       # JSONL utilities
├── scripts/                        # Workflow entry points
├── examples/                       # Synthetic fixtures and dataset workflows
├── docs/                           # Schemas, train/eval, and dataset guides
├── assets/                         # Workflow figure
└── tests/                          # Public smoke tests
```

## Citation

```bibtex
@article{hu2026strongest,
  title={The Strongest Teacher Is Not Always the Best Teacher: Student-Centric Answer Selection},
  author={Hu, Zhengyu and Xiao, Zheyuan and Song, Linxin and Jiang, Fengqing and Li, Yutai and Chen, Zhengyu and Xiong, Zhihan and Liu, Yue and Lin, Junhao and Su, Yao and Hu, Lijie and Ding, Kaize and Teng, Xiao and Poovendran, Radha},
  journal={arXiv preprint arXiv:2605.26872},
  year={2026}
}
```
