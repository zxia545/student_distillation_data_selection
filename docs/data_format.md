# Data Format

SCAS tools use JSONL files. Each line is one prompt-answer item.

## Candidate Answer Inputs

Place one file per teacher in a candidate directory. Supported file names:

```text
<teacher>_student_train.jsonl
<teacher>_teacher_responses.jsonl
<teacher>_teacher_responses_evaluated.jsonl
```

Required fields:

| Field | Meaning |
|---|---|
| `id` | Prompt id shared across teacher files. |
| `instruction` | User prompt. |
| `teacher_output` or `output` | Candidate teacher answer. |

Optional fields such as `system`, `dataset`, `split`, `reference_answer`,
`reference_output`, and `teacher` are preserved when possible.

## Score Outputs

`scas.scoring.model_candidates` and `scas.scoring.score_demo_candidates` write
one file per teacher:

```text
<teacher>_scas_scores.jsonl
```

Important score fields:

| Field | Meaning |
|---|---|
| `scas_score` | Default score for ranking answers. Lower is selected earlier. |
| `answer_answer_block` | Answer-answer learning-cost component. |
| `answer_question_block` | Answer-question learning-cost component. |
| `answer_mean_nll` | Mean student NLL on answer tokens. |
| `question_mean_nll` | Mean student NLL on prompt tokens. |
| `lambda_scas` | Weight on the answer-question block. |
| `scas_scoring_error` | Optional row-level error marker. |

## Grouped Selection Outputs

`scas.selection.group_by_score` writes:

```text
<output>/<score_key>/<num_groups>_groups/group_1.jsonl
<output>/<score_key>/<num_groups>_groups/group_2.jsonl
...
<output>/<score_key>/<num_groups>_groups/selected_group_1.jsonl
<output>/<score_key>/<num_groups>_groups/dataset_info.json
<output>/<score_key>/<num_groups>_groups/manifest.json
```

Rows in `selected_group_*.jsonl` are SFT-ready:

| Field | Meaning |
|---|---|
| `system` | System prompt, if available. |
| `instruction` | User prompt. |
| `output` | Selected teacher answer. |
| `teacher_name` | Teacher that produced the selected answer. |
| `score_key` | Score used for ranking. |
| `score_value` | Selected answer score. |
| `score_rank` | Rank among candidates for the same prompt. |
| `group` | Selected group id. |

`group_1` is the lowest-cost group after ascending score sorting.

## Training Inputs

The grouped selection directory includes `dataset_info.json`, which maps
dataset names such as `selected_group_1` to JSONL files. This is the format
expected by `scas.training.llamafactory` and `scripts/train_sft.sh`.

## Generation Inputs

`scas.generation.generate_responses` accepts JSONL rows with any of:

| Field | Meaning |
|---|---|
| `instruction` | Preferred prompt field. |
| `question` | Fallback prompt field. |
| `prompt` | Fallback prompt field. |
| `system` | Optional system prompt. |

It writes `model_output` and, on request failure, `generation_error`.

## Evaluation Inputs

`scas.evaluation.judge_math` compares generated outputs against references.
The default fields are:

| Field | Meaning |
|---|---|
| `model_output` | Generated student answer. |
| `reference_answer` | Ground-truth answer or solution. |

The evaluator writes `judge_correct`, `judge_reason`, and optionally
`judge_extracted_answer`.
