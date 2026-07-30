from __future__ import annotations

import pytest
from pydantic import ValidationError
from qwen3_tts_worker.config import QUALITY_STYLE, settings
from qwen3_tts_worker.schemas import SpeechRequest


def test_request_rejects_unknown_and_unsafe_ranges() -> None:
    with pytest.raises(ValidationError):
        SpeechRequest(input="Hello", temperature=9)
    with pytest.raises(ValidationError):
        SpeechRequest(input="Hello", file_path="/tmp/audio.wav")


def test_request_normalizes_text_and_instruction() -> None:
    request = SpeechRequest(
        input="  Hello\n  world  ",
        instruct="  calm\n and clear ",
    )
    assert request.input == "Hello world"
    assert request.instruct == "calm and clear"
    assert SpeechRequest(input="Hello", instruct="   ").instruct == ""


def test_request_defaults_follow_worker_settings() -> None:
    request = SpeechRequest(input="Hello")

    assert settings.default_speaker == "sohee"
    assert settings.default_language == "english"
    assert settings.style == QUALITY_STYLE
    assert request.temperature == settings.temperature == 0.5
    assert request.top_p == settings.top_p == 0.3
    assert request.top_k == settings.top_k == 150
    assert request.repetition_penalty == settings.repetition_penalty == 1.3
    assert request.do_sample is settings.do_sample is True
    assert request.seed == settings.seed == 0
    assert request.max_new_tokens == settings.max_new_tokens == 4096
    assert request.non_streaming_mode is settings.non_streaming_mode is False
    assert request.streaming_chunk_size == settings.streaming_chunk_size == 120
    assert request.token_safety_margin == settings.token_safety_margin == 8
    assert request.max_segment_chars == settings.max_segment_chars == 500
    assert request.segment_pause_ms == settings.segment_pause_ms == 120


def test_explicit_request_values_override_defaults() -> None:
    request = SpeechRequest(
        input="Hello",
        temperature=0.7,
        top_p=0.6,
        top_k=30,
        repetition_penalty=1.1,
        do_sample=False,
        seed=42,
        max_new_tokens=1024,
        non_streaming_mode=True,
        streaming_chunk_size=24,
        token_safety_margin=2,
        max_segment_chars=180,
        segment_pause_ms=50,
    )

    assert request.temperature == 0.7
    assert request.top_p == 0.6
    assert request.top_k == 30
    assert request.repetition_penalty == 1.1
    assert request.do_sample is False
    assert request.seed == 42
    assert request.max_new_tokens == 1024
    assert request.non_streaming_mode is True
    assert request.streaming_chunk_size == 24
    assert request.token_safety_margin == 2
    assert request.max_segment_chars == 180
    assert request.segment_pause_ms == 50


def test_openai_response_formats_are_accepted() -> None:
    for response_format in ("wav", "pcm", "pcm_s16le", "mp3", "opus", "flac"):
        request = SpeechRequest(input="Hello", response_format=response_format)
        assert request.response_format == response_format

    with pytest.raises(ValidationError):
        SpeechRequest(input="Hello", response_format="aac")
