#!/usr/bin/env python3
"""Filter evaluated teacher-response JSONL files to correct rows."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from scas.io import read_jsonl, write_json
from scas.io import write_jsonl


def truthy(value: Any) -> bool:
    if value is True or value == 1:
        return True
    if isinstance(value, str) and value.lower() in {"true", "1", "yes"}:
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pattern", default="*.jsonl")
    parser.add_argument("--correct-field", default="teacher_correct")
    args = parser.parse_args()

    manifest = {"input_dir": str(args.input_dir), "files": []}
    for source in sorted(args.input_dir.glob(args.pattern)):
        rows = [row for row in read_jsonl(source) if truthy(row.get(args.correct_field))]
        out_path = args.output_dir / source.name
        count = write_jsonl(out_path, rows)
        manifest["files"].append({"path": source.name, "rows": count})
        print(f"{source.name}: {count} correct rows -> {out_path}")
    write_json(args.output_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
