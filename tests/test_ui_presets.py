from __future__ import annotations

from qwen3_tts_worker.config import PRESET_FIELDS, Settings, production_preset


def test_production_preset_uses_worker_settings() -> None:
    config = Settings()
    preset = production_preset(config)

    assert tuple(preset) == PRESET_FIELDS
    assert preset["temperature"] == config.temperature
    assert preset["top_p"] == config.top_p
    assert preset["top_k"] == config.top_k
    assert preset["repetition_penalty"] == config.repetition_penalty
    assert preset["do_sample"] == config.do_sample
    assert preset["seed"] == config.seed
    assert preset["max_new_tokens"] == config.max_new_tokens
    assert preset["non_streaming_mode"] == config.non_streaming_mode
    assert preset["streaming_chunk_size"] == config.streaming_chunk_size
    assert preset["token_safety_margin"] == config.token_safety_margin
    assert preset["max_segment_chars"] == config.max_segment_chars
    assert preset["segment_pause_ms"] == config.segment_pause_ms
    assert preset["instruct"] == config.style
