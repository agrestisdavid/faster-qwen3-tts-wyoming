from __future__ import annotations

from qwen3_tts_worker.text import estimate_token_budget, split_tts_text


def test_split_tts_text_is_bounded_and_preserves_content() -> None:
    segments = split_tts_text(
        "This is a normal sentence. "
        "The intentionally longer second sentence should be split at a natural boundary.",
        max_chars=80,
    )
    assert segments
    assert all(len(segment) <= 80 for segment in segments)
    assert "normal sentence" in " ".join(segments)


def test_budget_is_aligned_and_capped() -> None:
    budget = estimate_token_budget(
        "A short synthesis test sentence.",
        chunk_size=12,
        configured_cap=300,
        safety_margin=3.0,
    )
    assert budget.requested % 12 == 0
    assert budget.resolved == 300
    assert budget.capped
