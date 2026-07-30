from __future__ import annotations

import io
import wave
from types import SimpleNamespace

import numpy as np
from qwen3_tts_worker.audio import (
    TrailingSilenceLimiter,
    encode_pcm16,
    float_to_pcm16,
    pcm16_to_wav,
)


def test_wav_is_native_24khz_mono_pcm16() -> None:
    pcm = float_to_pcm16(np.array([-1.0, 0.0, 1.0], dtype=np.float32))
    payload = pcm16_to_wav(pcm, 24000)
    with wave.open(io.BytesIO(payload), "rb") as wav:
        assert wav.getframerate() == 24000
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getnframes() == 3


def test_trailing_silence_is_capped_without_dropping_internal_pause() -> None:
    limiter = TrailingSilenceLimiter(
        sample_rate=1000,
        threshold=10,
        max_internal_ms=600,
        max_trailing_ms=250,
    )
    silence = np.zeros(1000, dtype="<i2").tobytes()
    speech = np.full(100, 1000, dtype="<i2").tobytes()
    assert limiter.push(silence) == []
    initial = limiter.push(speech)
    assert sum(len(item) for item in initial) == (40 + 100) * 2
    assert limiter.push(silence) == []
    emitted = limiter.push(speech)
    assert sum(len(item) for item in emitted) == (600 + 100) * 2
    limiter.push(silence)
    assert len(limiter.finish()) == 250 * 2


def test_openai_pcm_alias_returns_native_pcm() -> None:
    pcm = b"\x01\x00\x02\x00"
    payload, media_type = encode_pcm16(pcm, 24000, "pcm")

    assert payload == pcm
    assert media_type == "audio/pcm;rate=24000;channels=1"


def test_compressed_formats_use_ffmpeg(monkeypatch) -> None:
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=b"encoded", stderr=b"")

    monkeypatch.setattr("qwen3_tts_worker.audio.subprocess.run", fake_run)
    payload, media_type = encode_pcm16(b"\x00\x00", 24000, "opus")

    assert payload == b"encoded"
    assert media_type == "audio/ogg"
    assert "opus" in calls[0][0]
    assert calls[0][1]["input"] == b"\x00\x00"
