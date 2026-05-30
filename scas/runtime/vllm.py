"""Helpers for OpenAI-compatible vLLM servers."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import requests


def normalize_api_base(api_base: str) -> str:
    """Return an OpenAI-compatible base URL ending in /v1."""
    api_base = api_base.rstrip("/")
    if not api_base.endswith("/v1"):
        api_base = f"{api_base}/v1"
    return api_base


def default_served_model_name(model_path: str | Path) -> str:
    return Path(model_path).name.rstrip("/") or "scas-model"


def _build_vllm_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    env.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")

    nvsmi = shutil.which("nvidia-smi")
    non_nvlink = False
    if nvsmi is not None:
        try:
            topo = subprocess.run(
                [nvsmi, "topo", "-m"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            ).stdout
            bad_tokens = {"SYS", "NODE", "PHB", "PXB"}
            for line in topo.splitlines():
                stripped = line.lstrip()
                if not stripped.startswith("GPU"):
                    continue
                tokens = stripped.split()
                for tok in tokens[1:]:
                    if tok.startswith("NIC") or tok.startswith("mlx"):
                        break
                    if tok in bad_tokens:
                        non_nvlink = True
                        break
                if non_nvlink:
                    break
        except Exception:
            non_nvlink = False

    if non_nvlink:
        env.setdefault("NCCL_P2P_DISABLE", "1")
        env.setdefault("NCCL_IB_DISABLE", "1")
        env.setdefault("NCCL_SHM_DISABLE", "0")

    if extra:
        env.update(extra)
    return env


def wait_for_server(api_base: str, timeout_seconds: int = 1200, poll_seconds: float = 2.0) -> None:
    """Wait until an OpenAI-compatible server answers /models."""
    api_base = normalize_api_base(api_base)
    deadline = time.time() + timeout_seconds
    url = f"{api_base}/models"
    last_error = ""
    while time.time() < deadline:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code < 500:
                return
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
        except Exception as exc:
            last_error = str(exc)
        time.sleep(poll_seconds)
    raise TimeoutError(f"vLLM server did not become ready at {url}: {last_error}")


def start_vllm_server(
    *,
    model_path: str | Path,
    served_model_name: str,
    port: int,
    host: str = "0.0.0.0",
    tensor_parallel_size: int = 1,
    gpus: str | Iterable[int] | None = None,
    gpu_memory_utilization: float = 0.80,
    max_model_len: int = 4096,
    max_num_seqs: int = 64,
    wait_timeout_seconds: int = 1200,
    extra_args: list[str] | None = None,
) -> tuple[subprocess.Popen, str]:
    """Start a local vLLM OpenAI API server and return (process, client_base)."""
    env_extra: dict[str, str] = {}
    if gpus:
        gpu_list = ",".join(str(gpu) for gpu in gpus) if not isinstance(gpus, str) else gpus
        env_extra["CUDA_VISIBLE_DEVICES"] = gpu_list

    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        f"--model={model_path}",
        f"--served-model-name={served_model_name}",
        f"--tensor-parallel-size={tensor_parallel_size}",
        f"--gpu-memory-utilization={gpu_memory_utilization}",
        f"--max-model-len={max_model_len}",
        f"--max-num-seqs={max_num_seqs}",
        "--enforce-eager",
        f"--host={host}",
        f"--port={port}",
        "--trust-remote-code",
    ]
    if extra_args:
        command.extend(extra_args)

    process = subprocess.Popen(command, env=_build_vllm_env(env_extra))
    client_base = normalize_api_base(f"http://127.0.0.1:{port}")
    try:
        wait_for_server(client_base, timeout_seconds=wait_timeout_seconds)
    except Exception:
        stop_process(process)
        raise
    return process, client_base


def stop_process(process: subprocess.Popen | None, timeout_seconds: int = 30) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=timeout_seconds)
