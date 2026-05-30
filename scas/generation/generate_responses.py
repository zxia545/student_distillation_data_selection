#!/usr/bin/env python3
"""Generate JSONL responses through an OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import concurrent.futures
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from scas.io import read_jsonl, write_jsonl
from scas.runtime.vllm import (
    default_served_model_name,
    normalize_api_base,
    start_vllm_server,
    stop_process,
)


def prompt_from_row(row: dict[str, Any]) -> tuple[str, str]:
    system = str(row.get("system") or "You are a helpful assistant.")
    instruction = str(row.get("instruction") or row.get("question") or row.get("prompt") or "")
    return system, instruction


def call_model(
    client: OpenAI,
    *,
    model: str,
    row: dict[str, Any],
    max_tokens: int,
    temperature: float,
    retries: int,
) -> dict[str, Any]:
    system, instruction = prompt_from_row(row)
    out = dict(row)
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": instruction},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            out["model_output"] = response.choices[0].message.content or ""
            out["generation_error"] = None
            return out
        except Exception as exc:  # pragma: no cover - depends on external service
            if attempt == retries:
                out["model_output"] = ""
                out["generation_error"] = str(exc)
                return out
            time.sleep(min(2 ** (attempt - 1), 8))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--api-base", default=None, help="Existing OpenAI-compatible endpoint.")
    parser.add_argument("--api-key", default="dummy")
    parser.add_argument("--model", default=None, help="served model name for an existing endpoint")
    parser.add_argument("--model-path", default=None, type=Path, help="model path to serve with local vLLM")
    parser.add_argument("--served-model-name", default=None, help="served name when starting local vLLM")
    parser.add_argument("--start-vllm", action="store_true", help="start a local vLLM server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--tensor-parallel-size", default=1, type=int)
    parser.add_argument("--gpus", default=None, help="comma-separated CUDA_VISIBLE_DEVICES for local vLLM")
    parser.add_argument("--gpu-memory-utilization", default=0.80, type=float)
    parser.add_argument("--max-model-len", default=4096, type=int)
    parser.add_argument("--max-num-seqs", default=64, type=int)
    parser.add_argument("--server-timeout", default=1200, type=int)
    parser.add_argument("--request-timeout", default=120.0, type=float)
    parser.add_argument("--sdk-max-retries", default=0, type=int)
    parser.add_argument("--max-tokens", default=2048, type=int)
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--num-workers", default=32, type=int)
    parser.add_argument("--retries", default=3, type=int)
    parser.add_argument("--limit", default=None, type=int)
    args = parser.parse_args()

    should_start_vllm = args.start_vllm or (args.model_path is not None and args.api_base is None)
    if should_start_vllm and args.model_path is None:
        raise ValueError("--start-vllm requires --model-path")
    model_name = args.model or args.served_model_name
    if not model_name and args.model_path is not None:
        model_name = default_served_model_name(args.model_path)
    if not model_name:
        raise ValueError("Set --model for an existing endpoint, or pass --model-path to start local vLLM")

    rows = list(read_jsonl(args.input_jsonl))
    if args.limit is not None:
        rows = rows[: args.limit]

    process = None
    api_base = args.api_base
    try:
        if should_start_vllm:
            process, api_base = start_vllm_server(
                model_path=args.model_path,
                served_model_name=model_name,
                host=args.host,
                port=args.port,
                tensor_parallel_size=args.tensor_parallel_size,
                gpus=args.gpus,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
                max_num_seqs=args.max_num_seqs,
                wait_timeout_seconds=args.server_timeout,
            )
        else:
            api_base = normalize_api_base(api_base or f"http://127.0.0.1:{args.port}")

        client = OpenAI(
            base_url=api_base,
            api_key=args.api_key,
            timeout=args.request_timeout,
            max_retries=args.sdk_max_retries,
        )
        results: list[dict[str, Any] | None] = [None] * len(rows)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as pool:
            futures = {
                pool.submit(
                    call_model,
                    client,
                    model=model_name,
                    row=row,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    retries=args.retries,
                ): idx
                for idx, row in enumerate(rows)
            }
            for future in concurrent.futures.as_completed(futures):
                results[futures[future]] = future.result()
    finally:
        if should_start_vllm:
            stop_process(process)

    count = write_jsonl(args.output_jsonl, [row for row in results if row is not None])
    print(f"wrote {count} rows -> {args.output_jsonl}")


if __name__ == "__main__":
    main()
