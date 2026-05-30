#!/usr/bin/env python3
"""Evaluate teacher outputs against references."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from scas.evaluation.judge_math import evaluate_row
from scas.io import read_jsonl, write_json, write_jsonl


def teacher_output_field(row: dict[str, Any]) -> str:
    return "teacher_output" if "teacher_output" in row else "output"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--single-file", default=None)
    parser.add_argument("--method", choices=["rule"], default="rule")
    parser.add_argument("--reference-field", default="reference_answer")
    parser.add_argument("--limit", default=None, type=int)
    args = parser.parse_args()

    files = [args.input_dir / args.single_file] if args.single_file else sorted(args.input_dir.glob("*.jsonl"))
    manifest = {"input_dir": str(args.input_dir), "files": []}
    for source in files:
        rows = []
        for idx, row in enumerate(read_jsonl(source)):
            if args.limit is not None and idx >= args.limit:
                break
            out = evaluate_row(
                row,
                method="rule",
                output_field=teacher_output_field(row),
                reference_field=args.reference_field,
                client=None,
                judge_model=None,
                retries=1,
                judge_max_tokens=0,
            )
            out["teacher_correct"] = out.pop("judge_correct")
            out["eval_reason"] = out.pop("judge_reason")
            rows.append(out)
        out_path = args.output_dir / source.name
        count = write_jsonl(out_path, rows)
        manifest["files"].append({"path": source.name, "rows": count})
        print(f"{source.name}: {count} rows -> {out_path}")
    write_json(args.output_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
