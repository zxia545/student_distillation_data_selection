"""Forward-only SCAS scoring utilities."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tqdm import tqdm


def mean_pairwise_cos(mat: torch.Tensor, exclude_diag: bool = False) -> torch.Tensor:
    if not exclude_diag:
        return mat.mean()

    n = mat.size(0)
    if n <= 1:
        return mat.new_tensor(0.0)
    return (mat.sum() - mat.diag().sum()) / (n * n - n)


def nll_per_token_from_logits(
    logits: torch.Tensor, input_ids: torch.Tensor
) -> torch.Tensor:
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    nll = -log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    if nll.shape[0] == 1:
        return nll.squeeze(0)
    return nll


def common_prefix_len(a: torch.Tensor, b: torch.Tensor) -> int:
    if a.dim() != 1:
        a = a.view(-1)
    if b.dim() != 1:
        b = b.view(-1)

    n = min(a.numel(), b.numel())
    if n == 0:
        return 0

    eq = (a[:n] == b[:n]).to(torch.int32)
    if int(eq.all().item()) == 1:
        return n

    mismatch = (eq == 0).nonzero(as_tuple=False)
    if mismatch.numel() == 0:
        return n
    return int(mismatch[0].item())


def avg_nll_by_pos_mask(
    nll_per_pos: torch.Tensor, pos_mask: torch.Tensor
) -> torch.Tensor:
    if pos_mask.dim() != 1:
        pos_mask = pos_mask.view(-1)
    if pos_mask.numel() == 0:
        return nll_per_pos.new_tensor(0.0)

    safe_mask = pos_mask.clone().bool()
    safe_mask[0] = False
    nll_mask = safe_mask[1:]
    if int(nll_mask.any().item()) == 0:
        return nll_per_pos.new_tensor(0.0)
    return nll_per_pos[nll_mask].mean()


def compute_scas_scores(
    *,
    answer_answer_similarity: float,
    answer_answer_similarity_no_diag: float,
    answer_question_similarity: float,
    question_mean_nll: float,
    answer_mean_nll: float,
    lambda_scas: float = 0.5,
) -> dict[str, float]:
    """Compute the public SCAS score fields.

    The SCAS score is:

        S = (1 - lambda) * AA + lambda * AQ

    where AA is the answer-answer block and AQ is the answer-question block.
    """
    if not 0.0 <= lambda_scas <= 1.0:
        raise ValueError("lambda_scas must be in [0, 1]")

    lambda_scas = float(lambda_scas)
    answer_weight = 1.0 - lambda_scas
    answer_mean_nll = float(answer_mean_nll)
    question_mean_nll = float(question_mean_nll)
    answer_answer_similarity = float(answer_answer_similarity)
    answer_answer_similarity_no_diag = float(answer_answer_similarity_no_diag)
    answer_question_similarity = float(answer_question_similarity)

    answer_nll_weight = answer_mean_nll * answer_mean_nll
    answer_question_nll_weight = answer_mean_nll * question_mean_nll
    answer_answer_block = answer_nll_weight * answer_answer_similarity
    answer_answer_block_no_diag = (
        answer_nll_weight * answer_answer_similarity_no_diag
    )
    answer_question_block = answer_question_nll_weight * answer_question_similarity

    scas_score = (
        answer_weight * answer_answer_block
        + lambda_scas * answer_question_block
    )
    scas_score_no_diag = (
        answer_weight * answer_answer_block_no_diag
        + lambda_scas * answer_question_block
    )

    nll_weight_score = (
        answer_weight * answer_nll_weight
        + lambda_scas * answer_question_nll_weight
    )
    similarity_score = (
        answer_weight * answer_answer_similarity
        + lambda_scas * answer_question_similarity
    )
    similarity_score_no_diag = (
        answer_weight * answer_answer_similarity_no_diag
        + lambda_scas * answer_question_similarity
    )

    return {
        "lambda_scas": lambda_scas,
        "answer_answer_weight": answer_weight,
        "answer_mean_nll": answer_mean_nll,
        "question_mean_nll": question_mean_nll,
        "answer_nll_weight": answer_nll_weight,
        "answer_question_nll_weight": answer_question_nll_weight,
        "answer_answer_block": answer_answer_block,
        "answer_answer_block_no_diag": answer_answer_block_no_diag,
        "answer_question_block": answer_question_block,
        "question_question_block": 0.0,
        "scas_score": scas_score,
        "scas_score_no_diag": scas_score_no_diag,
        "nll_weight_score": nll_weight_score,
        "similarity_score": similarity_score,
        "similarity_score_no_diag": similarity_score_no_diag,
        "answer_mean_nll_score": answer_mean_nll,
    }


def zero_scas_metrics(lambda_scas: float = 0.5) -> dict[str, float]:
    return {
        "answer_answer_similarity": 0.0,
        "answer_answer_similarity_no_diag": 0.0,
        "answer_question_similarity": 0.0,
        "question_question_similarity": 0.0,
        "question_question_similarity_no_diag": 0.0,
        **compute_scas_scores(
            answer_answer_similarity=0.0,
            answer_answer_similarity_no_diag=0.0,
            answer_question_similarity=0.0,
            question_mean_nll=0.0,
            answer_mean_nll=0.0,
            lambda_scas=lambda_scas,
        ),
    }


def build_special_token_mask(tokenizer, input_ids_1d: torch.Tensor) -> torch.Tensor:
    special_ids: set[int] = set()

    raw = getattr(tokenizer, "all_special_ids", None)
    if raw:
        special_ids.update(int(x) for x in raw)

    additional = getattr(tokenizer, "additional_special_tokens", None)
    if additional and hasattr(tokenizer, "convert_tokens_to_ids"):
        try:
            special_ids.update(
                int(x) for x in tokenizer.convert_tokens_to_ids(additional)
            )
        except Exception:
            pass

    if hasattr(tokenizer, "convert_tokens_to_ids"):
        unk_id = getattr(tokenizer, "unk_token_id", None)
        for token in ["<|im_start|>", "<|im_end|>"]:
            try:
                token_id = tokenizer.convert_tokens_to_ids(token)
                if token_id is not None and token_id != unk_id:
                    special_ids.add(int(token_id))
            except Exception:
                pass

    is_special = torch.zeros_like(input_ids_1d, dtype=torch.bool)
    for special_id in special_ids:
        is_special |= input_ids_1d == special_id
    return is_special


def register_act_hooks(model, target_layer_name: str, store_dict: dict):
    handles = []
    for name, module in model.named_modules():
        if name == target_layer_name:
            handle = module.register_forward_hook(
                lambda _module, _inputs, output: store_dict.update({name: output})
            )
            handles.append(handle)
            break
    return handles


def remove_hooks(handles) -> None:
    for handle in handles:
        handle.remove()


def calculate_scas_metrics_on_answer(
    model,
    tokenizer,
    messages_list,
    target_layer,
    device=None,
    *,
    lambda_scas: float = 0.5,
    q_nll_scope: str = "prompt_with_gen",
    debug_token_boundaries: bool = False,
    boundary_window: int = 5,
    show_progress: bool = False,
):
    """Calculate SCAS metrics for assistant-answer tokens.

    `messages_list` contains chat messages ending in the teacher answer. The
    answer tokens define A; the system and user prompt define Q.
    """
    if device is None:
        device = model.device
    if q_nll_scope not in {"prompt_with_gen", "prompt_no_gen"}:
        raise ValueError(
            f"Unknown q_nll_scope={q_nll_scope}. "
            "Use 'prompt_with_gen' or 'prompt_no_gen'."
        )

    metric_results = []
    model.eval()

    for messages in tqdm(
        messages_list,
        desc="Calculating SCAS metrics",
        disable=not show_progress,
    ):
        prompt_messages = messages[:-1]
        prompt = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        prompt_nogen = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=False
        )
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        prompt_tokens = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_nogen_tokens = tokenizer(prompt_nogen, return_tensors="pt").to(device)
        full_tokens = tokenizer(full_text, return_tensors="pt").to(device)

        prompt_ids = prompt_tokens["input_ids"].squeeze(0)
        prompt_nogen_ids = prompt_nogen_tokens["input_ids"].squeeze(0)
        full_ids = full_tokens["input_ids"].squeeze(0)

        n_prompt = int(prompt_ids.shape[0])
        n_prompt_nogen = int(prompt_nogen_ids.shape[0])
        n_full = int(full_ids.shape[0])

        if n_prompt > 0 and n_full >= n_prompt:
            if not torch.equal(full_ids[:n_prompt], prompt_ids):
                n_prompt = common_prefix_len(prompt_ids, full_ids)
        if n_prompt_nogen > 0 and n_full >= n_prompt_nogen:
            if not torch.equal(full_ids[:n_prompt_nogen], prompt_nogen_ids):
                n_prompt_nogen = common_prefix_len(prompt_nogen_ids, full_ids)

        n_q_nll_end = n_prompt if q_nll_scope == "prompt_with_gen" else n_prompt_nogen
        n_answer = n_full - n_prompt

        if debug_token_boundaries:
            left = max(0, n_prompt - boundary_window)
            right = min(n_full, n_prompt + boundary_window)
            window_ids = full_ids[left:right]
            if hasattr(tokenizer, "convert_ids_to_tokens"):
                window_tokens = tokenizer.convert_ids_to_tokens(window_ids.tolist())
            else:
                window_tokens = [
                    tokenizer.decode([token_id]) for token_id in window_ids.tolist()
                ]
            print(
                "[DEBUG] token_boundary",
                {
                    "n_full": n_full,
                    "n_prompt": n_prompt,
                    "n_prompt_nogen": n_prompt_nogen,
                    "n_q_nll_end": n_q_nll_end,
                    "n_answer": n_answer,
                    "q_nll_scope": q_nll_scope,
                    "window_left": left,
                    "window_right": right,
                    "window_ids": window_ids.tolist(),
                    "window_tokens": window_tokens,
                },
            )

        if n_answer <= 0:
            raise ValueError(
                f"Answer length is non-positive (n_answer={n_answer}). "
                "Check chat-template token boundaries."
            )

        store: dict = {}
        hooks = register_act_hooks(model, target_layer, store)
        with torch.no_grad():
            outputs = model(**full_tokens)
        remove_hooks(hooks)

        if target_layer not in store:
            raise ValueError(f"Target layer '{target_layer}' was not executed.")

        hidden_states = store[target_layer]
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        hidden_states = hidden_states.squeeze(0).float()
        normalized = F.normalize(hidden_states, p=2, dim=1)

        full_ids_1d = full_tokens["input_ids"].squeeze(0)
        is_special = build_special_token_mask(tokenizer, full_ids_1d)
        pos = torch.arange(n_full, device=normalized.device)
        question_mask = (pos < n_prompt) & (~is_special)
        answer_mask = (pos >= n_prompt) & (pos < n_full) & (~is_special)

        question_pos = question_mask.nonzero(as_tuple=False).squeeze(1)
        answer_pos = answer_mask.nonzero(as_tuple=False).squeeze(1)
        if answer_pos.numel() == 0:
            metric_results.append(zero_scas_metrics(lambda_scas=lambda_scas))
            continue

        answer_states = normalized[answer_pos]
        answer_answer_cos = answer_states @ answer_states.T
        answer_answer_similarity = float(
            mean_pairwise_cos(answer_answer_cos, exclude_diag=False).item()
        )
        answer_answer_similarity_no_diag = float(
            mean_pairwise_cos(answer_answer_cos, exclude_diag=True).item()
        )

        if question_pos.numel() > 0:
            question_states = normalized[question_pos]
            answer_question_similarity = float((answer_states @ question_states.T).mean().item())
            question_question_cos = question_states @ question_states.T
            question_question_similarity = float(
                mean_pairwise_cos(question_question_cos, exclude_diag=False).item()
            )
            question_question_similarity_no_diag = float(
                mean_pairwise_cos(question_question_cos, exclude_diag=True).item()
            )
        else:
            answer_question_similarity = 0.0
            question_question_similarity = 0.0
            question_question_similarity_no_diag = 0.0

        nll_per_pos = nll_per_token_from_logits(
            outputs.logits, full_tokens["input_ids"]
        )
        question_nll_mask = (pos < n_q_nll_end) & (~is_special)
        question_mean_nll = avg_nll_by_pos_mask(nll_per_pos, question_nll_mask)
        answer_mean_nll = avg_nll_by_pos_mask(nll_per_pos, answer_mask)

        score_parts = compute_scas_scores(
            answer_answer_similarity=answer_answer_similarity,
            answer_answer_similarity_no_diag=answer_answer_similarity_no_diag,
            answer_question_similarity=answer_question_similarity,
            question_mean_nll=float(question_mean_nll.item()),
            answer_mean_nll=float(answer_mean_nll.item()),
            lambda_scas=lambda_scas,
        )
        question_question_block = (
            float(question_mean_nll.item())
            * float(question_mean_nll.item())
            * question_question_similarity
        )
        score_parts["question_question_block"] = question_question_block

        metric_results.append(
            {
                "answer_answer_similarity": answer_answer_similarity,
                "answer_answer_similarity_no_diag": answer_answer_similarity_no_diag,
                "answer_question_similarity": answer_question_similarity,
                "question_question_similarity": question_question_similarity,
                "question_question_similarity_no_diag": question_question_similarity_no_diag,
                **score_parts,
            }
        )

    return metric_results
