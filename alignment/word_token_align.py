"""
Map stimulus words (with character order in story text) to last tokenizer positions.

Handles subword tokenization by using HF offset mappings when available.
"""

from __future__ import annotations

from typing import List


def _char_spans_for_words(story_text: str, words: List[str]) -> List[tuple[int, int]]:
    spans: List[tuple[int, int]] = []
    pos = 0
    for w in words:
        if not w:
            spans.append((-1, -1))
            continue
        idx = story_text.find(w, pos)
        if idx == -1:
            idx = story_text.find(w.strip(), pos)
        if idx == -1:
            spans.append((-1, -1))
            continue
        spans.append((idx, idx + len(w)))
        pos = idx + len(w)
    return spans


def _last_overlapping_token(offset_mapping, c0: int, c1: int) -> int:
    last = -1
    for ti, (s, e) in enumerate(offset_mapping):
        if e <= c0:
            continue
        if s >= c1:
            break
        if s < c1 and e > c0:
            last = ti
    return last


def word_last_token_indices(tokenizer, story_text: str, words_json: list) -> List[int]:
    words = [str(w.get("word", "")) for w in words_json]
    spans = _char_spans_for_words(story_text, words)

    batch = tokenizer(
        story_text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=False,
    )
    offset_mapping = batch["offset_mapping"]
    if offset_mapping is None:
        raise ValueError(
            "Tokenizer did not return offset_mapping; use a fast tokenizer or upgrade transformers."
        )

    indices: List[int] = []
    for (c0, c1) in spans:
        if c0 < 0:
            indices.append(-1)
            continue
        indices.append(_last_overlapping_token(offset_mapping, c0, c1))
    return indices
