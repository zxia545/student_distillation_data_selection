#!/usr/bin/env python3
"""Evaluate generated math answers with rule-based or LLM judging."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import time
from pathlib import Path
from typing import Any

from scas.io import read_jsonl, write_jsonl


def require_openai_client():
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "LLM judging requires the OpenAI SDK. Install optional runtime "
            'dependencies with: pip install -e ".[runtime]"'
        ) from exc
    return OpenAI


def require_vllm_helpers():
    try:
        from scas.runtime.vllm import (
            default_served_model_name,
            normalize_api_base,
            start_vllm_server,
            stop_process,
        )
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Starting a local vLLM judge requires runtime dependencies. "
            'Install them with: pip install -e ".[runtime]"'
        ) from exc
    return default_served_model_name, normalize_api_base, start_vllm_server, stop_process


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    out = str(text).strip()
    out = re.sub(r"\\boxed\{([^{}]*)\}", r"\1", out)
    out = re.sub(r"\\(?:text|mathrm|mathbf|textbf)\{([^{}]*)\}", r"\1", out)
    out = out.strip("$ ")
    out = re.sub(r"\s+", " ", out)
    return out


def extract_number(text: Any) -> float | None:
    text = normalize_text(text)
    if "####" in text:
        text = text.rsplit("####", 1)[-1]
    frac = re.search(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", text)
    if frac and int(frac.group(2)) != 0:
        return int(frac.group(1)) / int(frac.group(2))
    simple = re.search(r"(-?\d+)\s*/\s*(-?\d+)", text)
    if simple and int(simple.group(2)) != 0:
        return int(simple.group(1)) / int(simple.group(2))
    matches = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if matches:
        return float(matches[-1].replace(",", ""))
    return None


def rule_judge(reference: Any, output: Any) -> tuple[bool | None, str]:
    ref = normalize_text(reference)
    pred = normalize_text(output)
    if ref and pred and ref.lower() in pred.lower():
        return True, "reference string found in model output"
    ref_num = extract_number(ref)
    pred_num = extract_number(pred)
    if ref_num is not None and pred_num is not None:
        return abs(ref_num - pred_num) < 1e-6, f"numeric comparison: reference={ref_num}, prediction={pred_num}"
    return None, "rule judge could not determine correctness"


def llm_judge(
    client: Any,
    *,
    model: str,
    row: dict[str, Any],
    output_field: str,
    reference_field: str,
    retries: int,
    max_tokens: int,
) -> tuple[bool | None, str, str]:
    prompt = {
        "instruction": row.get("instruction") or row.get("question") or "",
        "reference_answer": row.get(reference_field),
        "model_output": row.get(output_field),
    }
    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict math answer judge. Return JSON only with keys "
                "correct, extracted_answer, and reason. Mark correct only when "
                "the final answer is mathematically equivalent to the reference."
            ),
        },
        {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
    ]
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0,
            )
            text = response.choices[0].message.content or "{}"
            parsed = json.loads(text.strip().strip("`"))
            return bool(parsed.get("correct")), str(parsed.get("reason", "")), str(parsed.get("extracted_answer", ""))
        except Exception as exc:  # pragma: no cover - depends on external service
            if attempt == retries:
                return None, f"LLM judge error: {exc}", ""
            time.sleep(min(2 ** (attempt - 1), 8))
    return None, "LLM judge failed", ""


def evaluate_row(
    row: dict[str, Any],
    *,
    method: str,
    output_field: str,
    reference_field: str,
    client: Any | None,
    judge_model: str | None,
    retries: int,
    judge_max_tokens: int,
) -> dict[str, Any]:
    out = dict(row)
    correct, reason = rule_judge(row.get(reference_field), row.get(output_field))
    extracted = ""
    if method == "llm" and client is not None and judge_model and correct is None:
        correct, reason, extracted = llm_judge(
            client,
            model=judge_model,
            row=row,
            output_field=output_field,
            reference_field=reference_field,
            retries=retries,
            max_tokens=judge_max_tokens,
        )
    out["judge_correct"] = correct
    out["judge_reason"] = reason
    if extracted:
        out["judge_extracted_answer"] = extracted
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--method", choices=["rule", "llm"], default="rule")
    parser.add_argument("--model-output-field", default="model_output")
    parser.add_argument("--reference-field", default="reference_answer")
    parser.add_argument("--api-base", default=None, help="Existing OpenAI-compatible judge endpoint.")
    parser.add_argument("--api-key", default="dummy")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--judge-model-path", default=None, type=Path, help="judge model path to serve with local vLLM")
    parser.add_argument("--served-judge-model-name", default=None)
    parser.add_argument("--start-vllm", action="store_true", help="start a local judge vLLM server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8001, type=int)
    parser.add_argument("--tensor-parallel-size", default=1, type=int)
    parser.add_argument("--gpus", default=None, help="comma-separated CUDA_VISIBLE_DEVICES for local judge vLLM")
    parser.add_argument("--gpu-memory-utilization", default=0.80, type=float)
    parser.add_argument("--max-model-len", default=4096, type=int)
    parser.add_argument("--max-num-seqs", default=64, type=int)
    parser.add_argument("--server-timeout", default=1200, type=int)
    parser.add_argument("--request-timeout", default=120.0, type=float)
    parser.add_argument("--sdk-max-retries", default=0, type=int)
    parser.add_argument("--judge-max-tokens", default=512, type=int)
    parser.add_argument("--num-workers", default=16, type=int)
    parser.add_argument("--retries", default=3, type=int)
    args = parser.parse_args()

    process = None
    stop_process = lambda _process: None
    should_start_vllm = args.method == "llm" and (
        args.start_vllm or (args.judge_model_path is not None and args.api_base is None)
    )
    judge_model = args.judge_model or args.served_judge_model_name
    if args.method == "llm" and not judge_model and args.judge_model_path is not None:
        default_served_model_name, _normalize, _start, _stop = require_vllm_helpers()
        judge_model = default_served_model_name(args.judge_model_path)
    if args.method == "llm" and not judge_model:
        raise ValueError("LLM judging requires --judge-model or --judge-model-path")

    api_base = args.api_base
    if args.method == "llm" and should_start_vllm:
        if args.judge_model_path is None:
            raise ValueError("--start-vllm requires --judge-model-path")
        _default, _normalize, start_vllm_server, stop_process = require_vllm_helpers()
        process, api_base = start_vllm_server(
            model_path=args.judge_model_path,
            served_model_name=judge_model,
            host=args.host,
            port=args.port,
            tensor_parallel_size=args.tensor_parallel_size,
            gpus=args.gpus,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            wait_timeout_seconds=args.server_timeout,
        )
    elif args.method == "llm":
        _default, normalize_api_base, _start, _stop = require_vllm_helpers()
        api_base = normalize_api_base(api_base or f"http://127.0.0.1:{args.port}")

    OpenAI = require_openai_client() if args.method == "llm" else None
    client = (
        OpenAI(
            base_url=api_base,
            api_key=args.api_key,
            timeout=args.request_timeout,
            max_retries=args.sdk_max_retries,
        )
        if args.method == "llm"
        else None
    )
    rows = list(read_jsonl(args.input_jsonl))
    results: list[dict[str, Any] | None] = [None] * len(rows)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as pool:
            futures = {
                pool.submit(
                    evaluate_row,
                    row,
                    method=args.method,
                    output_field=args.model_output_field,
                    reference_field=args.reference_field,
                    client=client,
                    judge_model=judge_model,
                    retries=args.retries,
                    judge_max_tokens=args.judge_max_tokens,
                ): idx
                for idx, row in enumerate(rows)
            }
            for future in concurrent.futures.as_completed(futures):
                results[futures[future]] = future.result()
        count = write_jsonl(args.output_jsonl, [row for row in results if row is not None])
        print(f"wrote {count} rows -> {args.output_jsonl}")
    finally:
        if should_start_vllm:
            stop_process(process)


if __name__ == "__main__":
    main()
