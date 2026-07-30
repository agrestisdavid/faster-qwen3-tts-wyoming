from __future__ import annotations

import io
import subprocess
import wave

import numpy as np


def float_to_pcm16(audio: np.ndarray) -> bytes:
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        return b""
    samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)
    return np.clip(samples * 32767.0, -32768, 32767).astype("<i2").tobytes()


def pcm16_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    target = io.BytesIO()
    with wave.open(target, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return target.getvalue()


def encode_pcm16(pcm: bytes, sample_rate: int, output_format: str) -> tuple[bytes, str]:
    """Encode native mono PCM16 to an OpenAI-compatible response format."""
    if output_format == "wav":
        return pcm16_to_wav(pcm, sample_rate), "audio/wav"
    if output_format in {"pcm", "pcm_s16le"}:
        return pcm, f"audio/pcm;rate={sample_rate};channels=1"

    formats = {
        "mp3": ("mp3", "audio/mpeg"),
        "opus": ("opus", "audio/ogg"),
        "flac": ("flac", "audio/flac"),
    }
    target = formats.get(output_format)
    if target is None:
        raise ValueError(f"unsupported audio format {output_format!r}")
    muxer, media_type = target
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-i",
        "pipe:0",
        "-f",
        muxer,
        "pipe:1",
    ]
    result = subprocess.run(
        command,
        input=pcm,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"ffmpeg {output_format} encoding failed: {detail}")
    return result.stdout, media_type


def silence_pcm(sample_rate: int, milliseconds: int) -> bytes:
    samples = max(0, round(sample_rate * milliseconds / 1000))
    return bytes(samples * 2)


class TrailingSilenceLimiter:
    """Keep natural pauses while preventing long generated tails."""

    def __init__(
        self,
        *,
        sample_rate: int,
        threshold: int = 160,
        max_initial_ms: int = 40,
        max_internal_ms: int = 600,
        max_trailing_ms: int = 250,
    ) -> None:
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.max_initial_bytes = round(sample_rate * max_initial_ms / 1000) * 2
        self.max_internal_bytes = round(sample_rate * max_internal_ms / 1000) * 2
        self.max_trailing_bytes = round(sample_rate * max_trailing_ms / 1000) * 2
        self._pending = bytearray()
        self._seen_speech = False

    def push(self, pcm: bytes) -> list[bytes]:
        if not pcm:
            return []
        even = len(pcm) - (len(pcm) % 2)
        pcm = pcm[:even]
        samples = np.frombuffer(pcm, dtype="<i2")
        if samples.size and int(np.max(np.abs(samples.astype(np.int32)))) <= self.threshold:
            self._pending.extend(pcm)
            return []

        output: list[bytes] = []
        if self._pending:
            silence_limit = self.max_internal_bytes if self._seen_speech else self.max_initial_bytes
            output.append(bytes(self._pending[-silence_limit:]))
            self._pending.clear()
        output.append(pcm)
        self._seen_speech = True
        return output

    def finish(self) -> bytes:
        tail = bytes(self._pending[-self.max_trailing_bytes :])
        self._pending.clear()
        return tail
