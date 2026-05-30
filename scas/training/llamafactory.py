#!/usr/bin/env python3
"""Run or print an SFT command for LLaMA-Factory."""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
from pathlib import Path


def dataset_file(args: argparse.Namespace) -> Path:
    info_path = args.dataset_dir / "dataset_info.json"
    with info_path.open("r", encoding="utf-8") as handle:
        info = json.load(handle)
    if args.dataset_name not in info:
        raise KeyError(f"{args.dataset_name!r} not found in {info_path}")
    file_name = info[args.dataset_name].get("file_name")
    if not file_name:
        raise KeyError(f"{args.dataset_name!r} in {info_path} has no file_name")
    return args.dataset_dir / file_name


def count_jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def compute_save_steps(args: argparse.Namespace) -> int:
    if args.save_steps is not None:
        return max(1, args.save_steps)
    rows = count_jsonl_rows(dataset_file(args))
    samples_per_step = (
        args.num_gpus
        * args.per_device_train_batch_size
        * args.gradient_accumulation_steps
    )
    steps_per_epoch = max(1, math.ceil(rows / max(1, samples_per_step)))
    total_steps = max(1, math.ceil(steps_per_epoch * args.num_train_epochs))
    return max(1, total_steps // max(1, args.num_checkpoints))


def latest_complete_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda path: path.name)
    for checkpoint in reversed(checkpoints):
        if (checkpoint / "trainer_state.json").is_file():
            return checkpoint
    return None


def build_command(args: argparse.Namespace, save_steps: int, resume_from: Path | None) -> list[str]:
    command = [
        "deepspeed",
        "--num_gpus",
        str(args.num_gpus),
        str(args.llamafactory_train),
        "--deepspeed",
        str(args.deepspeed_config),
        "--stage",
        "sft",
        "--do_train",
        "--model_name_or_path",
        args.model_name_or_path,
        "--dataset",
        args.dataset_name,
        "--dataset_dir",
        str(args.dataset_dir),
        "--template",
        args.template,
        "--finetuning_type",
        "full",
        "--output_dir",
        str(args.output_dir),
        "--overwrite_cache",
        "--report_to",
        args.report_to,
        "--cutoff_len",
        str(args.cutoff_len),
        "--per_device_train_batch_size",
        str(args.per_device_train_batch_size),
        "--per_device_eval_batch_size",
        str(args.per_device_eval_batch_size),
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--preprocessing_num_workers",
        str(args.preprocessing_num_workers),
        "--lr_scheduler_type",
        args.lr_scheduler_type,
        "--logging_steps",
        str(args.logging_steps),
        "--warmup_ratio",
        str(args.warmup_ratio),
        "--save_strategy",
        "steps",
        "--save_steps",
        str(save_steps),
        "--learning_rate",
        str(args.learning_rate),
        "--num_train_epochs",
        str(args.num_train_epochs),
        "--val_size",
        str(args.val_size),
        "--ddp_timeout",
        "1800000",
        "--plot_loss",
        "--seed",
        str(args.seed),
        "--fp16",
    ]
    if args.overwrite_output_dir:
        command.append("--overwrite_output_dir")
    if resume_from is not None:
        command.extend(["--resume_from_checkpoint", str(resume_from)])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--student-short-name", default=None)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--llamafactory-train", required=True, type=Path)
    parser.add_argument("--deepspeed-config", required=True, type=Path)
    parser.add_argument("--num-gpus", default=4, type=int)
    parser.add_argument("--template", default="qwen")
    parser.add_argument("--per-device-train-batch-size", default=1, type=int)
    parser.add_argument("--per-device-eval-batch-size", default=1, type=int)
    parser.add_argument("--gradient-accumulation-steps", default=2, type=int)
    parser.add_argument("--learning-rate", default="1.0e-5")
    parser.add_argument("--num-train-epochs", default=3, type=float)
    parser.add_argument("--cutoff-len", default=4096, type=int)
    parser.add_argument("--val-size", default=0.001, type=float)
    parser.add_argument("--warmup-ratio", default=0.05, type=float)
    parser.add_argument("--logging-steps", default=10, type=int)
    parser.add_argument("--save-steps", default=None, type=int)
    parser.add_argument("--num-checkpoints", default=4, type=int)
    parser.add_argument("--preprocessing-num-workers", default=16, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--report-to", default="tensorboard")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default=None, type=Path)
    parser.add_argument("--overwrite-output-dir", action="store_true")
    parser.add_argument("--skip-if-complete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not (args.dataset_dir / "dataset_info.json").is_file():
        raise FileNotFoundError(args.dataset_dir / "dataset_info.json")
    if not args.dry_run:
        if not args.llamafactory_train.is_file():
            raise FileNotFoundError(args.llamafactory_train)
        if not args.deepspeed_config.is_file():
            raise FileNotFoundError(args.deepspeed_config)

    if args.skip_if_complete and (args.output_dir / "config.json").is_file():
        print(f"[SKIP] trained checkpoint already exists: {args.output_dir}")
        return

    save_steps = compute_save_steps(args)
    resume_from = args.resume_from_checkpoint
    if resume_from is None and args.resume:
        resume_from = latest_complete_checkpoint(args.output_dir)

    command = build_command(args, save_steps, resume_from)
    if args.dry_run:
        dataset_path = dataset_file(args)
        print(
            "[DRY_RUN_CONFIG] "
            f"dataset={dataset_path} rows={count_jsonl_rows(dataset_path)} "
            f"save_steps={save_steps} student={args.student_short_name or ''} "
            f"resume_from={resume_from or ''}"
        )
        print("[DRY_RUN] " + " ".join(shlex.quote(part) for part in command))
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
