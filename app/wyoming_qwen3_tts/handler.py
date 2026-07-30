from __future__ import annotations

import argparse
import logging
from collections.abc import AsyncIterator

import httpx
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
)

LOGGER = logging.getLogger(__name__)


def normalize_text(text: str | None) -> str:
    return " ".join((text or "").replace("\x00", "").split())


async def aligned_pcm16(chunks: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    pending = b""
    async for chunk in chunks:
        if not chunk:
            continue
        data = pending + chunk
        aligned_size = len(data) - (len(data) % 2)
        if aligned_size:
            yield data[:aligned_size]
        pending = data[aligned_size:]

    if pending:
        raise RuntimeError("Qwen3-TTS returned an incomplete PCM16 sample")


class Qwen3TtsEventHandler(AsyncEventHandler):
    def __init__(
        self,
        wyoming_info: Info,
        cli_args: argparse.Namespace,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.wyoming_info_event = wyoming_info.event()
        self.cli_args = cli_args
        self.is_streaming = False
        self._stream_voice = None
        self._stream_text: list[str] = []

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            return True

        try:
            if Synthesize.is_type(event.type):
                if self.is_streaming:
                    return True
                synthesize = Synthesize.from_event(event)
                await self._handle_text(synthesize.text, synthesize.voice)
                return True

            if SynthesizeStart.is_type(event.type):
                start = SynthesizeStart.from_event(event)
                self.is_streaming = True
                self._stream_voice = start.voice
                self._stream_text = []
                return True

            if SynthesizeChunk.is_type(event.type):
                chunk = SynthesizeChunk.from_event(event)
                self._stream_text.append(chunk.text)
                return True

            if SynthesizeStop.is_type(event.type):
                try:
                    await self._handle_text(
                        "".join(self._stream_text),
                        self._stream_voice,
                    )
                    await self.write_event(SynthesizeStopped().event())
                finally:
                    self.is_streaming = False
                    self._stream_voice = None
                    self._stream_text = []
                return True

            return True
        except Exception as err:
            LOGGER.exception("Wyoming Qwen3-TTS synthesis failed")
            await self.write_event(
                Error(text=str(err), code=err.__class__.__name__).event()
            )
            return False

    def _speaker_from_voice(self, voice) -> str:
        if voice is None:
            return self.cli_args.speaker

        speaker: str | None = getattr(voice, "speaker", None)
        if speaker:
            return speaker

        voice_name: str | None = getattr(voice, "name", None)
        if voice_name and voice_name != self.cli_args.voice_name:
            return voice_name

        return self.cli_args.speaker

    async def _handle_text(self, text: str, voice) -> None:
        clean_text = normalize_text(text)
        if not clean_text:
            raise ValueError("TTS text is empty")

        speaker = self._speaker_from_voice(voice)
        payload = {
            "model": "tts-1",
            "input": clean_text,
            "voice": speaker,
            "language": self.cli_args.language,
            "instruct": self.cli_args.instruct,
            "response_format": "pcm",
            "stream": True,
        }
        timeout = httpx.Timeout(
            connect=self.cli_args.connect_timeout,
            read=self.cli_args.response_timeout,
            write=self.cli_args.connect_timeout,
            pool=self.cli_args.connect_timeout,
        )
        audio_started = False
        sample_rate = self.cli_args.sample_rate

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST",
                    self.cli_args.qwen_url,
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    header_rate = response.headers.get("x-audio-sample-rate")
                    if header_rate:
                        sample_rate = int(header_rate)

                    async for audio in aligned_pcm16(response.aiter_bytes()):
                        if not audio_started:
                            await self.write_event(
                                AudioStart(
                                    rate=sample_rate,
                                    width=2,
                                    channels=1,
                                ).event()
                            )
                            audio_started = True
                        await self.write_event(
                            AudioChunk(
                                audio=audio,
                                rate=sample_rate,
                                width=2,
                                channels=1,
                            ).event()
                        )

            if not audio_started:
                raise RuntimeError("Qwen3-TTS returned no audio")

            await self.write_event(AudioStop().event())
            LOGGER.info(
                "Synthesized %d characters with voice=%s at %d Hz",
                len(clean_text),
                speaker,
                sample_rate,
            )
        except Exception:
            if audio_started:
                await self.write_event(AudioStop().event())
            raise
