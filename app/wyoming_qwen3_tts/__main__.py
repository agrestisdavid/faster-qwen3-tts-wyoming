from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
from functools import partial

from wyoming.info import Attribution, Info, TtsProgram, TtsVoice, TtsVoiceSpeaker
from wyoming.server import AsyncServer, AsyncTcpServer

from . import __version__
from .handler import Qwen3TtsEventHandler

LOGGER = logging.getLogger(__name__)

LANGUAGE_CODES = {
    "chinese": "zh",
    "english": "en",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "japanese": "ja",
    "korean": "ko",
    "portuguese": "pt",
    "russian": "ru",
    "spanish": "es",
}


def language_code(language: str) -> str:
    override = os.getenv("TTS_LANGUAGE_CODE", "").strip()
    if override:
        return override
    return LANGUAGE_CODES.get(language.strip().lower(), language.strip().lower())


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--uri",
        default=os.getenv("WYOMING_URI", "tcp://0.0.0.0:10210"),
    )
    parser.add_argument(
        "--qwen-url",
        default=os.getenv(
            "QWEN3_TTS_URL",
            "http://127.0.0.1:7860/v1/audio/speech",
        ),
    )
    # Keep these protocol identifiers stable so Home Assistant retains its
    # existing tts.qwen3_realtime_tts entity and pipeline selection.
    parser.add_argument(
        "--program-name",
        default=os.getenv("WYOMING_TTS_PROGRAM_NAME", "qwen3-realtime-tts"),
    )
    parser.add_argument(
        "--voice-name",
        default=os.getenv("WYOMING_TTS_VOICE_NAME", "qwen3-realtime-de"),
    )
    parser.add_argument(
        "--speaker",
        default=(
            os.getenv("WYOMING_TTS_SPEAKER", "").strip()
            or os.getenv("QWEN3_TTS_DEFAULT_VOICE", "").strip()
            or os.getenv("QWEN3_TTS_SPEAKER", "sohee").strip()
            or "sohee"
        ),
    )
    parser.add_argument(
        "--language",
        default=os.getenv("QWEN3_TTS_LANGUAGE", "english"),
    )
    parser.add_argument(
        "--language-code",
        default=None,
    )
    parser.add_argument(
        "--instruct",
        default=os.getenv("QWEN3_TTS_STYLE_INSTRUCTION", ""),
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=int(os.getenv("QWEN3_TTS_OUTPUT_SAMPLE_RATE", "24000")),
    )
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=float(os.getenv("QWEN3_TTS_CONNECT_TIMEOUT", "15")),
    )
    parser.add_argument(
        "--response-timeout",
        type=float,
        default=float(os.getenv("QWEN3_TTS_RESPONSE_TIMEOUT", "300")),
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if not args.language_code:
        args.language_code = language_code(args.language)

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    wyoming_info = Info(
        tts=[
            TtsProgram(
                name=args.program_name,
                description="Wyoming TTS via the shared Qwen3-TTS worker",
                attribution=Attribution(
                    name="Qwen3-TTS",
                    url="https://huggingface.co/Qwen",
                ),
                installed=True,
                version=__version__,
                supports_synthesize_streaming=True,
                voices=[
                    TtsVoice(
                        name=args.voice_name,
                        description="Qwen3-TTS CustomVoice",
                        attribution=Attribution(
                            name="Qwen",
                            url="https://huggingface.co/Qwen",
                        ),
                        installed=True,
                        version=None,
                        languages=[args.language_code],
                        speakers=[TtsVoiceSpeaker(name=args.speaker)],
                    )
                ],
            )
        ]
    )

    server = AsyncServer.from_uri(args.uri)
    if isinstance(server, AsyncTcpServer):
        LOGGER.info("Wyoming Qwen3-TTS listening on %s", args.uri)
    LOGGER.info(
        "Forwarding directly to %s (voice=%s, language=%s, sample_rate=%d)",
        args.qwen_url,
        args.speaker,
        args.language,
        args.sample_rate,
    )

    server_task = asyncio.create_task(
        server.run(partial(Qwen3TtsEventHandler, wyoming_info, args))
    )

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, server_task.cancel)
        except NotImplementedError:
            pass

    try:
        await server_task
    except asyncio.CancelledError:
        LOGGER.info("Wyoming Qwen3-TTS stopped")


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
