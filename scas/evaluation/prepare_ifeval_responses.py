#!/usr/bin/env python3
"""Convert generated SCAS responses to IFEval's prompt/response JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from scas.io import read_jsonl, write_jsonl


def first_text(row: dict[str, Any], fields: list[str]) -> str:
    for field in fields:
        value = row.get(field)
        if value is not None:
            return str(value)
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--response-field", default="model_output")
    parser.add_argument("--skip-errors", action="store_true")
    args = parser.parse_args()

    rows = []
    for row in read_jsonl(args.input_jsonl):
        if args.skip_errors and row.get("generation_error"):
            continue
        prompt = first_text(row, [args.prompt_field, "instruction", "question"])
        response = first_text(row, [args.response_field, "response", "output", "model_output"])
        rows.append({"prompt": prompt, "response": response})

    count = write_jsonl(args.output_jsonl, rows)
    print(f"wrote {count} rows -> {args.output_jsonl}")


if __name__ == "__main__":
    main()
