#!/usr/bin/env python3
"""Score teacher-answer candidates with a student model using SCAS."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from tqdm import tqdm

from scas.io import read_jsonl, write_json, write_jsonl

_MODEL_IMPORT_ERROR: ModuleNotFoundError | None = None

try:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from scas.scoring.metric_utils import (
        calculate_scas_metrics_on_answer,
        zero_scas_metrics,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - exercised without extras
    _MODEL_IMPORT_ERROR = exc
    torch = None  # type: ignore[assignment]
    AutoConfig = None  # type: ignore[assignment]
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]
    calculate_scas_metrics_on_answer = None  # type: ignore[assignment]
    zero_scas_metrics = None  # type: ignore[assignment]


CANDIDATE_SUFFIXES = (
    "_teacher_responses_evaluated.jsonl",
    "_teacher_responses.jsonl",
    "_student_train.jsonl",
    ".jsonl",
)


def require_model_dependencies() -> None:
    if _MODEL_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Model scoring requires the optional model dependencies. "
            'Install them with: pip install -e ".[model]"'
        ) from _MODEL_IMPORT_ERROR


def extract_teacher_name(path: Path) -> str:
    for suffix in CANDIDATE_SUFFIXES:
        if path.name.endswith(suffix):
            return path.name[: -len(suffix)]
    return path.stem


def discover_candidate_files(input_dir: Path) -> list[Path]:
    files: dict[str, Path] = {}
    for suffix in CANDIDATE_SUFFIXES:
        pattern = f"*{suffix}" if suffix != ".jsonl" else "*.jsonl"
        for path in sorted(input_dir.glob(pattern)):
            if path.name.endswith("_scas_scores.jsonl"):
                continue
            files.setdefault(extract_teacher_name(path), path)
    return [files[key] for key in sorted(files)]


def resolve_device(device: str) -> str:
    require_model_dependencies()
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return device


def resolve_dtype(dtype: str):
    require_model_dependencies()
    mapping = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[dtype]


def infer_target_layer(model_name_or_path: str) -> str:
    require_model_dependencies()
    config = AutoConfig.from_pretrained(model_name_or_path)
    num_layers = getattr(config, "num_hidden_layers", None)
    if num_layers is None:
        raise ValueError(
            "Could not infer a target layer from the model config. "
            "Pass --target-layer explicitly."
        )
    return f"model.layers.{num_layers - 1}.mlp.up_proj"


def load_student_model(
    model_name_or_path: str,
    *,
    device: str,
    dtype: str,
    device_map: str | None,
) -> tuple[Any, Any, Any]:
    require_model_dependencies()
    model_kwargs: dict[str, Any] = {"torch_dtype": resolve_dtype(dtype)}
    if device_map:
        model_kwargs["device_map"] = device_map

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    model.eval()

    if device_map:
        model_device = getattr(model, "device", None)
        if model_device is None:
            model_device = next(model.parameters()).device
        return model, tokenizer, model_device

    resolved = resolve_device(device)
    model.to(resolved)
    return model, tokenizer, torch.device(resolved)


def build_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    system = str(row.get("system") or "You are a helpful assistant.")
    instruction = str(row.get("instruction") or "")
    teacher_output = str(row.get("teacher_output") or row.get("output") or "")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": teacher_output},
    ]


def score_row(
    row: dict[str, Any],
    *,
    teacher_name: str,
    model: Any,
    tokenizer: Any,
    target_layer: str,
    device: Any,
    lambda_scas: float,
) -> dict[str, Any]:
    result = dict(row)
    result["teacher_name"] = row.get("teacher_name") or row.get("teacher") or teacher_name

    if not row.get("instruction") or not (row.get("teacher_output") or row.get("output")):
        result.update(zero_scas_metrics(lambda_scas=lambda_scas))
        result["scas_scoring_error"] = "Missing instruction or teacher_output/output."
        return result

    try:
        scores = calculate_scas_metrics_on_answer(
            model=model,
            tokenizer=tokenizer,
            messages_list=[build_messages(row)],
            target_layer=target_layer,
            device=device,
            lambda_scas=lambda_scas,
            show_progress=False,
        )
        result.update(scores[0])
    except Exception as exc:  # keep row-level failures visible in JSONL outputs
        result.update(zero_scas_metrics(lambda_scas=lambda_scas))
        result["scas_scoring_error"] = str(exc)

    return result


def score_file(
    input_path: Path,
    output_path: Path,
    *,
    model: Any,
    tokenizer: Any,
    target_layer: str,
    device: Any,
    lambda_scas: float,
    limit: int | None,
) -> dict[str, Any]:
    teacher_name = extract_teacher_name(input_path)
    rows = list(read_jsonl(input_path))
    if limit is not None:
        rows = rows[:limit]

    scored = [
        score_row(
            row,
            teacher_name=teacher_name,
            model=model,
            tokenizer=tokenizer,
            target_layer=target_layer,
            device=device,
            lambda_scas=lambda_scas,
        )
        for row in tqdm(rows, desc=f"Scoring {teacher_name}")
    ]
    count = write_jsonl(output_path, scored)
    error_count = sum(1 for row in scored if row.get("scas_scoring_error"))
    return {
        "teacher": teacher_name,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "rows": count,
        "errors": error_count,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--student-model",
        "--student_model_path",
        dest="student_model_path",
        required=True,
        help="Hugging Face model name or local student checkpoint path.",
    )
    parser.add_argument(
        "--candidate-dir",
        "--teacher_responses_dir",
        dest="candidate_dir",
        required=True,
        type=Path,
        help="Directory containing one JSONL file per teacher.",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        required=True,
        type=Path,
        help="Directory for *_scas_scores.jsonl outputs.",
    )
    parser.add_argument("--target-layer", "--target_layer", dest="target_layer")
    parser.add_argument(
        "--lambda-scas",
        "--lambda_scas",
        dest="lambda_scas",
        default=0.5,
        type=float,
        help="Weight on the answer-question block. Must be in [0, 1].",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--device-map",
        dest="device_map",
        default=None,
        help="Optional Transformers device_map value, for example 'auto'.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--skip-existing", "--skip_existing", action="store_true")
    parser.add_argument("--specific-teacher", "--specific_teacher", dest="specific_teacher")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 <= args.lambda_scas <= 1.0:
        raise ValueError("--lambda-scas must be in [0, 1].")
    if not args.candidate_dir.exists():
        raise FileNotFoundError(args.candidate_dir)

    files = discover_candidate_files(args.candidate_dir)
    if args.specific_teacher:
        files = [path for path in files if args.specific_teacher in extract_teacher_name(path)]
    if not files:
        raise FileNotFoundError(f"No candidate JSONL files found in {args.candidate_dir}")

    target_layer = args.target_layer or infer_target_layer(args.student_model_path)
    model, tokenizer, device = load_student_model(
        args.student_model_path,
        device=args.device,
        dtype=args.dtype,
        device_map=args.device_map,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "student_model": args.student_model_path,
        "candidate_dir": str(args.candidate_dir),
        "target_layer": target_layer,
        "lambda_scas": args.lambda_scas,
        "files": [],
    }

    for input_path in files:
        teacher_name = extract_teacher_name(input_path)
        output_path = args.output_dir / f"{teacher_name}_scas_scores.jsonl"
        if args.skip_existing and output_path.exists():
            manifest["files"].append(
                {
                    "teacher": teacher_name,
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "skipped": True,
                }
            )
            continue
        stats = score_file(
            input_path,
            output_path,
            model=model,
            tokenizer=tokenizer,
            target_layer=target_layer,
            device=device,
            lambda_scas=args.lambda_scas,
            limit=args.limit,
        )
        manifest["files"].append(stats)
        print(
            f"{teacher_name}: {stats['rows']} rows, "
            f"{stats['errors']} errors -> {output_path}"
        )

    write_json(args.output_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
