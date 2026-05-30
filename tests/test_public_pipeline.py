from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class PublicPipelineTest(unittest.TestCase):
    def run_module(self, *args: str) -> None:
        subprocess.run(
            [sys.executable, "-m", *args],
            cwd=REPO_ROOT,
            check=True,
            text=True,
        )

    def test_demo_scoring_and_grouping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            scored_dir = tmp_path / "scored"
            grouped_dir = tmp_path / "grouped"

            self.run_module(
                "scas.scoring.score_demo_candidates",
                "--input-dir",
                "examples/candidates",
                "--output-dir",
                str(scored_dir),
                "--lambda-scas",
                "0.5",
            )
            self.run_module(
                "scas.selection.group_by_score",
                "--input-dir",
                str(scored_dir),
                "--output-dir",
                str(grouped_dir),
                "--num-groups",
                "3",
                "--selected-group",
                "1",
            )

            selected = grouped_dir / "scas_score" / "3_groups" / "selected_group_1.jsonl"
            rows = [json.loads(line) for line in selected.read_text().splitlines()]
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(row["group"] == 1 for row in rows))
            self.assertTrue((scored_dir / "manifest.json").exists())

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scas.training.llamafactory",
                    "--dataset-dir",
                    str(grouped_dir / "scas_score" / "3_groups"),
                    "--dataset-name",
                    "selected_group_1",
                    "--model-name-or-path",
                    "/tmp/base-student",
                    "--output-dir",
                    str(tmp_path / "train_dry_run"),
                    "--llamafactory-train",
                    "/tmp/LLaMA-Factory/src/train.py",
                    "--deepspeed-config",
                    "/tmp/LLaMA-Factory/examples/deepspeed/ds_z3_config.json",
                    "--num-gpus",
                    "1",
                    "--dry-run",
                ],
                cwd=REPO_ROOT,
                check=True,
                text=True,
            )

            eval_path = tmp_path / "eval_math.jsonl"
            self.run_module(
                "scas.evaluation.judge_math",
                "--input-jsonl",
                "examples/validation/toy_generations.jsonl",
                "--output-jsonl",
                str(eval_path),
            )
            judged = [json.loads(line) for line in eval_path.read_text().splitlines()]
            self.assertEqual(len(judged), 3)
            self.assertTrue(all(row["judge_correct"] is True for row in judged))


if __name__ == "__main__":
    unittest.main()
