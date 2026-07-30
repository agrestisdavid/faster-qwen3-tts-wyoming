from __future__ import annotations

import os
from dataclasses import dataclass

QUALITY_STYLE = (
    "Speak naturally, clearly, and warmly in English. Use a calm, fluent pace, "
    "lively but restrained prosody, and natural sentence pauses. Articulate "
    "sibilants and numbers clearly. Avoid monotonous intonation, exaggerated "
    "emotion, and dramatic pitch changes."
)

DEFAULT_VOICES = [
    "aiden",
    "dylan",
    "eric",
    "ono_anna",
    "ryan",
    "serena",
    "sohee",
    "uncle_fu",
    "vivian",
]

OPENAI_MODEL_ALIASES = {
    "tts-1",
    "tts-1-hd",
    "qwen3-tts",
}

PRESET_SETTING_FIELDS = (
    ("temperature", "temperature"),
    ("top_p", "top_p"),
    ("top_k", "top_k"),
    ("repetition_penalty", "repetition_penalty"),
    ("do_sample", "do_sample"),
    ("seed", "seed"),
    ("max_new_tokens", "max_new_tokens"),
    ("non_streaming_mode", "non_streaming_mode"),
    ("streaming_chunk_size", "streaming_chunk_size"),
    ("token_safety_margin", "token_safety_margin"),
    ("max_segment_chars", "max_segment_chars"),
    ("segment_pause_ms", "segment_pause_ms"),
    ("instruct", "style"),
)
PRESET_FIELDS = tuple(field_name for field_name, _setting_name in PRESET_SETTING_FIELDS)


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_value(name: str, legacy_name: str, default: str) -> str:
    value = os.getenv(name)
    if value is not None:
        return value
    return os.getenv(legacy_name, default)


@dataclass(frozen=True)
class Settings:
    model: str = os.getenv(
        "QWEN3_TTS_MODEL",
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    )
    backend: str = os.getenv("QWEN3_TTS_BACKEND", "ggml")
    device: str = os.getenv("QWEN3_TTS_DEVICE", "cuda")
    dtype: str = os.getenv("QWEN3_TTS_DTYPE", "bfloat16")
    attn_implementation: str = os.getenv("QWEN3_TTS_ATTN_IMPLEMENTATION", "sdpa")
    quant: str = os.getenv("QWEN3_TTS_QUANT", "BF16")
    default_speaker: str = env_value(
        "QWEN3_TTS_DEFAULT_VOICE",
        "QWEN3_TTS_SPEAKER",
        "sohee",
    )
    default_language: str = os.getenv("QWEN3_TTS_LANGUAGE", "english")
    force_language: bool = env_bool("QWEN3_TTS_FORCE_LANGUAGE", True)
    sample_rate: int = int(os.getenv("QWEN3_TTS_OUTPUT_SAMPLE_RATE", "24000"))
    temperature: float = float(os.getenv("QWEN3_TTS_TEMPERATURE", "0.5"))
    top_p: float = float(os.getenv("QWEN3_TTS_TOP_P", "0.3"))
    top_k: int = int(os.getenv("QWEN3_TTS_TOP_K", "150"))
    repetition_penalty: float = float(os.getenv("QWEN3_TTS_REPETITION_PENALTY", "1.3"))
    do_sample: bool = env_bool("QWEN3_TTS_DO_SAMPLE", True)
    seed: int = int(os.getenv("QWEN3_TTS_SEED", "0"))
    max_new_tokens: int = int(os.getenv("QWEN3_TTS_MAX_NEW_TOKENS", "4096"))
    non_streaming_mode: bool = env_bool("QWEN3_TTS_NON_STREAMING_MODE", False)
    streaming_chunk_size: int = int(os.getenv("QWEN3_TTS_STREAMING_CHUNK_SIZE", "120"))
    token_safety_margin: float = float(os.getenv("QWEN3_TTS_TOKEN_SAFETY_MARGIN", "8"))
    max_segment_chars: int = int(
        env_value(
            "QWEN3_TTS_MAX_SEGMENT_CHARS",
            "QWEN3_TTS_SEGMENT_MAX_CHARS",
            "500",
        )
    )
    segment_pause_ms: int = int(os.getenv("QWEN3_TTS_SEGMENT_PAUSE_MS", "120"))
    style: str = os.getenv("QWEN3_TTS_STYLE_INSTRUCTION", QUALITY_STYLE)
    queue_max_pending: int = int(os.getenv("QWEN3_TTS_QUEUE_MAX_PENDING", "8"))
    warmup: bool = env_bool("QWEN3_TTS_WARMUP", True)


settings = Settings()


def production_preset(config: Settings) -> dict[str, object]:
    return {
        field_name: getattr(config, setting_name)
        for field_name, setting_name in PRESET_SETTING_FIELDS
    }


def build_presets(config: Settings) -> dict[str, dict[str, object]]:
    production = production_preset(config)
    hf_demo = {
        **production,
        "temperature": 0.9,
        "top_p": 1.0,
        "top_k": 50,
        "max_new_tokens": 2048,
        "instruct": "",
    }
    return {
        "Production": production,
        "HF Demo": hf_demo,
        "Quality Style": {**hf_demo, "instruct": QUALITY_STYLE},
    }
