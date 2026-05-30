#!/usr/bin/env python3
"""Group candidate answers by ascending SCAS score.

Candidates are sorted from low cost to high cost, and group 1 is the
lowest-score group.
"""

from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from scas.io import canonical_id, read_jsonl, write_json, write_jsonl

SCORED_SUFFIX = "_scas_scores.jsonl"


def discover_inputs(input_dir: Path) -> list[Path]:
    return sorted(input_dir.glob(f"*{SCORED_SUFFIX}"))


def teacher_from_path(path: Path) -> str:
    if path.name.endswith(SCORED_SUFFIX):
        return path.name[: -len(SCORED_SUFFIX)]
    return path.stem


def as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:
        return None
    return out


def candidate_score(row: dict[str, Any], score_key: str) -> float | None:
    return as_float(row.get(score_key))


def load_candidates(paths: list[Path], score_key: str) -> dict[str, list[dict[str, Any]]]:
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        teacher = teacher_from_path(path)
        for row in read_jsonl(path):
            item_id = row.get("id")
            score = candidate_score(row, score_key)
            if (
                item_id is None
                or score is None
                or row.get("scas_scoring_error")
            ):
                continue
            candidate = dict(row)
            candidate["teacher_name"] = row.get("teacher_name") or row.get("teacher") or teacher
            candidate["_score_value"] = score
            by_id[canonical_id(item_id)].append(candidate)
    return dict(by_id)


def group_for_rank(rank_zero_based: int, size: int, num_groups: int) -> int:
    """Map an ascending rank to the SCAS group id."""
    rank_one_based = rank_zero_based + 1
    return min(num_groups, max(1, math.ceil(rank_one_based * num_groups / size)))


def output_row(row: dict[str, Any], *, score_key: str, group: int, rank: int) -> dict[str, Any]:
    answer = row.get("teacher_output") or row.get("output") or ""
    return {
        "id": row.get("id"),
        "dataset": row.get("dataset"),
        "split": row.get("split"),
        "system": row.get("system", ""),
        "instruction": row.get("instruction", ""),
        "output": answer,
        "teacher_output": answer,
        "reference_answer": row.get("reference_answer") or row.get("answer"),
        "reference_output": row.get("reference_output"),
        "category": row.get("category", ""),
        "teacher_name": row.get("teacher_name"),
        "score_key": score_key,
        "score_value": row["_score_value"],
        "score_rank": rank,
        "group": group,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--score-key", default="scas_score")
    parser.add_argument("--num-groups", default=5, type=int)
    parser.add_argument("--selected-group", default=1, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    if args.num_groups < 1:
        raise ValueError("--num-groups must be >= 1")
    if not (1 <= args.selected_group <= args.num_groups):
        raise ValueError("--selected-group must be between 1 and --num-groups")

    files = discover_inputs(args.input_dir)
    if not files:
        raise FileNotFoundError(
            f"no *_scas_scores.jsonl files found in {args.input_dir}"
        )

    rng = random.Random(args.seed)
    by_id = load_candidates(files, args.score_key)
    if not by_id:
        raise RuntimeError(f"no candidates with score key {args.score_key!r}")

    out_root = args.output_dir / args.score_key / f"{args.num_groups}_groups"
    grouped_rows: dict[int, list[dict[str, Any]]] = {g: [] for g in range(1, args.num_groups + 1)}
    selected_rows: list[dict[str, Any]] = []
    missing_groups: dict[int, int] = {g: 0 for g in range(1, args.num_groups + 1)}

    for item_key in sorted(by_id):
        ranked = sorted(by_id[item_key], key=lambda row: (row["_score_value"], str(row.get("teacher_name"))))
        ranked_by_group: dict[int, list[tuple[int, dict[str, Any]]]] = {
            g: [] for g in range(1, args.num_groups + 1)
        }
        for rank_zero, candidate in enumerate(ranked):
            ranked_by_group[group_for_rank(rank_zero, len(ranked), args.num_groups)].append(
                (rank_zero + 1, candidate)
            )
        for group_idx in range(1, args.num_groups + 1):
            candidates = ranked_by_group[group_idx]
            if not candidates:
                missing_groups[group_idx] += 1
                continue
            rank, chosen = rng.choice(candidates)
            row = output_row(chosen, score_key=args.score_key, group=group_idx, rank=rank)
            grouped_rows[group_idx].append(row)
            if group_idx == args.selected_group:
                selected_rows.append(row)

    dataset_info: dict[str, Any] = {}
    counts: dict[str, int] = {}
    for group_idx, rows in grouped_rows.items():
        name = f"group_{group_idx}.jsonl"
        counts[name] = write_jsonl(out_root / name, rows)
        dataset_info[f"group_{group_idx}"] = {
            "file_name": name,
            "columns": {"system": "system", "prompt": "instruction", "response": "output"},
        }

    selected_name = f"selected_group_{args.selected_group}.jsonl"
    selected_count = write_jsonl(out_root / selected_name, selected_rows)
    dataset_info[f"selected_group_{args.selected_group}"] = {
        "file_name": selected_name,
        "columns": {"system": "system", "prompt": "instruction", "response": "output"},
    }
    write_json(out_root / "dataset_info.json", dataset_info)
    write_json(
        out_root / "manifest.json",
        {
            "score_key": args.score_key,
            "num_groups": args.num_groups,
            "selected_group": args.selected_group,
            "seed": args.seed,
            "input_dir": str(args.input_dir),
            "input_files": [str(path) for path in files],
            "items": len(by_id),
            "group_counts": counts,
            "selected_count": selected_count,
            "missing_groups": missing_groups,
            "grouping_convention": "ascending score; group_1 is the lowest-cost group",
        },
    )
    print(f"items={len(by_id)} selected={selected_count} output={out_root}")


if __name__ == "__main__":
    main()
