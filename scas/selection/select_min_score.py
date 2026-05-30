#!/usr/bin/env python3
"""Select the single minimum-score teacher answer for each item."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from scas.io import canonical_id, read_jsonl, write_json, write_jsonl
from scas.selection.group_by_score import (
    candidate_score,
    discover_inputs,
    output_row,
    teacher_from_path,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--score-key", default="scas_score")
    args = parser.parse_args()

    files = discover_inputs(args.input_dir)
    if not files:
        raise FileNotFoundError(f"no *_scas_scores.jsonl files found in {args.input_dir}")

    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in files:
        teacher = teacher_from_path(path)
        for row in read_jsonl(path):
            score = candidate_score(row, args.score_key)
            if (
                row.get("id") is None
                or score is None
                or row.get("scas_scoring_error")
            ):
                continue
            candidate = dict(row)
            candidate["teacher_name"] = row.get("teacher_name") or row.get("teacher") or teacher
            candidate["_score_value"] = score
            by_id[canonical_id(row["id"])].append(candidate)

    selected = []
    for item_key in sorted(by_id):
        ranked = sorted(by_id[item_key], key=lambda row: (row["_score_value"], str(row.get("teacher_name"))))
        selected.append(output_row(ranked[0], score_key=args.score_key, group=0, rank=1))

    out_root = args.output_dir / args.score_key
    count = write_jsonl(out_root / "min_score_selected.jsonl", selected)
    write_json(
        out_root / "dataset_info.json",
        {
            "min_score_selected": {
                "file_name": "min_score_selected.jsonl",
                "columns": {"system": "system", "prompt": "instruction", "response": "output"},
            }
        },
    )
    write_json(
        out_root / "manifest.json",
        {"score_key": args.score_key, "input_dir": str(args.input_dir), "items": count},
    )
    print(f"items={count} output={out_root}")


if __name__ == "__main__":
    main()
