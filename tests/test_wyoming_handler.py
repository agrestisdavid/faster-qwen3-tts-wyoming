from __future__ import annotations

import asyncio

from wyoming_qwen3_tts.__main__ import language_code
from wyoming_qwen3_tts.handler import aligned_pcm16, normalize_text


async def _chunks(parts: list[bytes]):
    for part in parts:
        yield part


def test_normalize_text() -> None:
    assert normalize_text("  Hallo\n\nWelt.  ") == "Hallo Welt."
    assert normalize_text("\x00") == ""


def test_aligned_pcm16_keeps_samples_intact() -> None:
    async def collect() -> list[bytes]:
        return [chunk async for chunk in aligned_pcm16(_chunks([b"\x01", b"\x02\x03", b"\x04"]))]

    assert asyncio.run(collect()) == [b"\x01\x02", b"\x03\x04"]


def test_aligned_pcm16_rejects_trailing_byte() -> None:
    async def collect() -> None:
        async for _chunk in aligned_pcm16(_chunks([b"\x01"])):
            pass

    try:
        asyncio.run(collect())
    except RuntimeError as err:
        assert "incomplete PCM16" in str(err)
    else:
        raise AssertionError("expected an incomplete PCM16 error")


def test_language_code_follows_configured_synthesis_language(monkeypatch) -> None:
    monkeypatch.delenv("TTS_LANGUAGE_CODE", raising=False)

    assert language_code("english") == "en"
    assert language_code("German") == "de"
    assert language_code("spanish") == "es"


def test_language_code_allows_explicit_override(monkeypatch) -> None:
    monkeypatch.setenv("TTS_LANGUAGE_CODE", "en-GB")

    assert language_code("english") == "en-GB"
