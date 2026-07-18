"""Chat-template wrapping for instruct models (train/inference alignment)."""

from __future__ import annotations

from typing import Any, Optional

from core.layer_config import LAYER_CONFIG


def uses_chat_template(model_name: str) -> bool:
    cfg = LAYER_CONFIG.get(model_name) or {}
    return bool(cfg.get("use_chat_template", False))


def format_user_text(
    tokenizer: Any,
    text: str,
    *,
    model_name: Optional[str] = None,
    enable: Optional[bool] = None,
) -> str:
    """
    Wrap raw user text with the tokenizer chat template when the model family
    expects instruct formatting. Base/completion models pass text through unchanged.
    """
    if enable is None:
        if model_name is None:
            return text
        enable = uses_chat_template(model_name)
    if not enable:
        return text
    apply = getattr(tokenizer, "apply_chat_template", None)
    if apply is None:
        return text
    messages = [{"role": "user", "content": text}]
    try:
        return apply(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Some tokenizers reject roles without a system message — try with a no-op system.
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": text},
            ]
            return apply(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return text


def format_qa_claim(
    tokenizer: Any,
    question: str,
    answer: str,
    *,
    model_name: Optional[str] = None,
) -> str:
    """Format a question+answer claim for encode-path guard scoring."""
    body = f"Question: {question.strip()}\nAnswer: {answer.strip()}"
    return format_user_text(tokenizer, body, model_name=model_name)
