from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .config import settings


class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    input: str = Field(min_length=1, max_length=4000)
    voice: str | None = None
    language: str | None = Field(default=None, max_length=40)
    instruct: str | None = Field(default=None, max_length=1000)
    response_format: Literal[
        "wav",
        "pcm",
        "pcm_s16le",
        "mp3",
        "opus",
        "flac",
    ] = "wav"
    stream: bool = False
    temperature: float = Field(
        default_factory=lambda: settings.temperature, ge=0.05, le=2.0
    )
    top_p: float = Field(default_factory=lambda: settings.top_p, gt=0.0, le=1.0)
    top_k: int = Field(default_factory=lambda: settings.top_k, ge=1, le=200)
    repetition_penalty: float = Field(
        default_factory=lambda: settings.repetition_penalty, ge=0.8, le=2.0
    )
    do_sample: bool = Field(default_factory=lambda: settings.do_sample)
    seed: int = Field(
        default_factory=lambda: settings.seed, ge=0, le=2_147_483_647
    )
    max_new_tokens: int = Field(
        default_factory=lambda: settings.max_new_tokens, ge=64, le=4096
    )
    non_streaming_mode: bool = Field(
        default_factory=lambda: settings.non_streaming_mode
    )
    streaming_chunk_size: int = Field(
        default_factory=lambda: settings.streaming_chunk_size, ge=1, le=120
    )
    token_safety_margin: float = Field(
        default_factory=lambda: settings.token_safety_margin, ge=1.0, le=8.0
    )
    max_segment_chars: int = Field(
        default_factory=lambda: settings.max_segment_chars, ge=80, le=500
    )
    segment_pause_ms: int = Field(
        default_factory=lambda: settings.segment_pause_ms, ge=0, le=2000
    )

    @field_validator("input")
    @classmethod
    def clean_input(cls, value: str) -> str:
        value = " ".join(value.replace("\x00", "").split())
        if not value:
            raise ValueError("input must contain visible text")
        return value

    @field_validator("instruct")
    @classmethod
    def clean_instruction(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = " ".join(value.replace("\x00", "").split())
        return value


class VoiceList(BaseModel):
    voices: list[str]
    default: str
    model: str
    backend: str
    sample_rate: int


class SynthesisMetrics(BaseModel):
    request_id: str
    queue_wait_ms: float
    ttfa_ms: float | None
    total_ms: float
    audio_seconds: float
    rtf: float | None
    sample_rate: int
    samples: int
    segments: int
    seed: int
    voice: str
    language: str
