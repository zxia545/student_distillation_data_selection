#!/usr/bin/env python3
"""Create deterministic demo scores for teacher-answer samples.

This is not the model-based SCAS metric. It exists so the public repository has
a small no-GPU smoke path that exercises the same downstream file contracts as
the real scorer. For real scoring, use ``scas.scoring.model_candidates`` with a
student model checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
from typing import Any

from scas.io import read_jsonl, write_json, write_jsonl


def stable_unit(*parts: Any) -> float:
    text = "||".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12 - 1)


def token_count(text: str | None) -> int:
    if not text:
        return 0
    return len(text.split())


def score_row(row: dict[str, Any], teacher: str, lambda_scas: float) -> dict[str, Any]:
    out = dict(row)
    answer_text = str(row.get("teacher_output") or row.get("output") or "")
    question_text = str(row.get("instruction") or "")
    answer_tokens = max(1, token_count(answer_text))
    question_tokens = max(1, token_count(question_text))
    jitter = stable_unit(row.get("id"), teacher) * 0.01

    answer_mean_nll = 0.8 + min(answer_tokens, 1200) / 900.0 + jitter
    question_mean_nll = 0.6 + min(question_tokens, 800) / 1200.0
    answer_answer_similarity = 1.0 / math.sqrt(answer_tokens) + jitter
    answer_question_similarity = 1.0 / math.sqrt(answer_tokens + question_tokens)
    question_question_similarity = 1.0 / math.sqrt(question_tokens)

    answer_nll_weight = answer_mean_nll * answer_mean_nll
    answer_question_nll_weight = answer_mean_nll * question_mean_nll
    aa = answer_nll_weight * answer_answer_similarity
    aq = answer_question_nll_weight * answer_question_similarity
    full = (1.0 - lambda_scas) * aa + lambda_scas * aq

    out["teacher_name"] = row.get("teacher") or teacher
    out["answer_mean_nll"] = answer_mean_nll
    out["question_mean_nll"] = question_mean_nll
    out["answer_nll_weight"] = answer_nll_weight
    out["answer_question_nll_weight"] = answer_question_nll_weight
    out["answer_answer_weight"] = 1.0 - lambda_scas
    out["answer_answer_similarity"] = answer_answer_similarity
    out["answer_answer_similarity_no_diag"] = (
        answer_answer_similarity + (0.001 / answer_tokens)
    )
    out["answer_question_similarity"] = answer_question_similarity
    out["question_question_similarity"] = question_question_similarity
    out["question_question_similarity_no_diag"] = (
        question_question_similarity + (0.001 / question_tokens)
    )
    out["answer_answer_block"] = aa
    out["answer_answer_block_no_diag"] = (
        answer_nll_weight * out["answer_answer_similarity_no_diag"]
    )
    out["answer_question_block"] = aq
    out["question_question_block"] = (
        question_mean_nll * question_mean_nll * question_question_similarity
    )
    out["scas_score"] = full
    out["scas_score_no_diag"] = (
        (1.0 - lambda_scas) * out["answer_answer_block_no_diag"]
        + lambda_scas * aq
    )
    out["nll_weight_score"] = (
        (1.0 - lambda_scas) * answer_nll_weight
        + lambda_scas * answer_question_nll_weight
    )
    out["similarity_score"] = (
        (1.0 - lambda_scas) * answer_answer_similarity
        + lambda_scas * answer_question_similarity
    )
    out["similarity_score_no_diag"] = (
        (1.0 - lambda_scas) * out["answer_answer_similarity_no_diag"]
        + lambda_scas * answer_question_similarity
    )
    out["answer_mean_nll_score"] = answer_mean_nll + jitter
    out["lambda_scas"] = lambda_scas
    out["scoring_backend"] = "demo_length_proxy"
    return out


def teacher_from_path(path: Path) -> str:
    for suffix in (
        "_teacher_responses_evaluated.jsonl",
        "_teacher_responses.jsonl",
        "_student_train.jsonl",
        ".jsonl",
    ):
        if path.name.endswith(suffix):
            return path.name[: -len(suffix)]
    return path.stem


def discover_inputs(input_dir: Path) -> list[Path]:
    patterns = (
        "*_teacher_responses.jsonl",
        "*_teacher_responses_evaluated.jsonl",
        "*_student_train.jsonl",
    )
    files: dict[str, Path] = {}
    for pattern in patterns:
        for path in sorted(input_dir.glob(pattern)):
            files.setdefault(teacher_from_path(path), path)
    return [files[key] for key in sorted(files)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--lambda-scas", dest="lambda_scas", default=0.5, type=float)
    parser.add_argument("--limit", default=None, type=int)
    args = parser.parse_args()

    if not (0.0 <= args.lambda_scas <= 1.0):
        raise ValueError("--lambda-scas must be in [0, 1]")

    files = discover_inputs(args.input_dir)
    if not files:
        raise FileNotFoundError(
            f"no candidate answer JSONL files found in {args.input_dir}"
        )

    manifest: dict[str, Any] = {
        "backend": "demo_length_proxy",
        "input_dir": str(args.input_dir),
        "lambda_scas": args.lambda_scas,
        "files": [],
    }

    for source in files:
        teacher = teacher_from_path(source)
        rows = []
        for idx, row in enumerate(read_jsonl(source)):
            if args.limit is not None and idx >= args.limit:
                break
            rows.append(score_row(row, teacher, args.lambda_scas))

        rel_out = f"{teacher}_scas_scores.jsonl"
        out_path = args.output_dir / rel_out
        count = write_jsonl(out_path, rows)
        manifest["files"].append({"teacher": teacher, "rows": count, "path": rel_out})
        print(f"{teacher}: {count} rows -> {out_path}")

    write_json(args.output_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
