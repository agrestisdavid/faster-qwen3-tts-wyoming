from __future__ import annotations

import asyncio
import logging
import random
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .audio import TrailingSilenceLimiter, float_to_pcm16, silence_pcm
from .config import DEFAULT_VOICES, Settings
from .schemas import SpeechRequest, SynthesisMetrics
from .text import estimate_token_budget, split_tts_text

logger = logging.getLogger(__name__)
_END = object()


@dataclass
class Job:
    request: SpeechRequest
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.perf_counter)
    chunks: asyncio.Queue[Any] = field(default_factory=asyncio.Queue)
    done: asyncio.Future[SynthesisMetrics] | None = None
    cancelled: threading.Event = field(default_factory=threading.Event)


@dataclass
class GenerationProgress:
    started_at: float
    queue_wait_ms: float
    sample_rate: int
    first_audio_at: float | None = None
    total_samples: int = 0

    @classmethod
    def start(cls, job: Job, sample_rate: int) -> GenerationProgress:
        started_at = time.perf_counter()
        return cls(
            started_at=started_at,
            queue_wait_ms=(started_at - job.created_at) * 1000,
            sample_rate=sample_rate,
        )

    def record(
        self,
        pcm: bytes,
        *,
        emitted_at: float | None = None,
        mark_first_audio: bool = True,
    ) -> None:
        if not pcm:
            return
        if mark_first_audio and self.first_audio_at is None:
            self.first_audio_at = (
                time.perf_counter() if emitted_at is None else emitted_at
            )
        self.total_samples += len(pcm) // 2

    def metrics(
        self,
        job: Job,
        *,
        segment_count: int,
        finished_at: float | None = None,
    ) -> SynthesisMetrics:
        request = job.request
        if request.voice is None or request.language is None:
            raise RuntimeError("request voice and language must be resolved")

        if finished_at is None:
            finished_at = time.perf_counter()
        elapsed = finished_at - self.started_at
        audio_seconds = self.total_samples / self.sample_rate
        return SynthesisMetrics(
            request_id=job.request_id,
            queue_wait_ms=self.queue_wait_ms,
            ttfa_ms=(
                (self.first_audio_at - self.started_at) * 1000
                if self.first_audio_at is not None
                else None
            ),
            total_ms=elapsed * 1000,
            audio_seconds=audio_seconds,
            rtf=(elapsed / audio_seconds if audio_seconds else None),
            sample_rate=self.sample_rate,
            samples=self.total_samples,
            segments=segment_count,
            seed=request.seed,
            voice=request.voice,
            language=request.language,
        )


def generation_options(
    request: SpeechRequest,
    *,
    max_new_tokens: int,
) -> dict[str, object]:
    return {
        "non_streaming_mode": request.non_streaming_mode,
        "max_new_tokens": max_new_tokens,
        "temperature": request.temperature,
        "top_k": request.top_k,
        "top_p": request.top_p,
        "do_sample": request.do_sample,
        "repetition_penalty": request.repetition_penalty,
        "chunk_size": request.streaming_chunk_size,
    }


class WorkerRuntime:
    def __init__(self, config: Settings) -> None:
        self.config = config
        self.model: Any = None
        self.voices: list[str] = []
        self.sample_rate = config.sample_rate
        self._pending: asyncio.Queue[Job] = asyncio.Queue(maxsize=config.queue_max_pending)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qwen3-tts")
        self._consumer: asyncio.Task[None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self.active_request_id: str | None = None
        self.last_error: str | None = None
        self.started_at = time.time()

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        await self._loop.run_in_executor(self._executor, self._load_model)
        self._consumer = asyncio.create_task(self._consume(), name="qwen3-tts-consumer")

    async def stop(self) -> None:
        if self._consumer:
            self._consumer.cancel()
            try:
                await self._consumer
            except asyncio.CancelledError:
                pass
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _load_model(self) -> None:
        import torch
        from faster_qwen3_tts import FasterQwen3TTS

        dtype = getattr(torch, self.config.dtype, torch.bfloat16)
        logger.info(
            "Loading Qwen3-TTS model=%s backend=%s device=%s",
            self.config.model,
            self.config.backend,
            self.config.device,
        )
        self.model = FasterQwen3TTS.from_pretrained(
            self.config.model,
            device=self.config.device,
            dtype=dtype,
            attn_implementation=self.config.attn_implementation,
            backend=self.config.backend,
            quant=self.config.quant,
        )
        discovered = getattr(self.model, "get_supported_speakers", lambda: [])()
        self.voices = sorted({str(voice).lower() for voice in discovered})
        if not self.voices:
            self.voices = DEFAULT_VOICES.copy()
        if self.config.default_speaker.lower() not in self.voices:
            raise RuntimeError(
                f"Default speaker {self.config.default_speaker!r} is not supported: {self.voices}"
            )
        native_rate = int(getattr(self.model, "sample_rate", self.config.sample_rate))
        if native_rate != self.config.sample_rate:
            raise RuntimeError(
                f"Model sample rate is {native_rate}, expected native {self.config.sample_rate}"
            )
        self.sample_rate = native_rate
        if self.config.warmup:
            self.model.warmup(prefill_len=100)
        logger.info("Qwen3-TTS loaded with %d voices at %d Hz", len(self.voices), self.sample_rate)

    async def submit(self, request: SpeechRequest) -> Job:
        if self.model is None or self._loop is None:
            raise RuntimeError("model is not ready")
        voice = (request.voice or self.config.default_speaker).lower()
        if voice not in self.voices:
            raise ValueError(f"unsupported voice {voice!r}; choose one of {self.voices}")
        request.voice = voice
        if self.config.force_language:
            request.language = self.config.default_language
        else:
            request.language = (request.language or self.config.default_language).lower()
        if request.instruct is None:
            request.instruct = self.config.style

        job = Job(request=request)
        job.done = self._loop.create_future()
        self._pending.put_nowait(job)
        return job

    async def _consume(self) -> None:
        assert self._loop is not None
        while True:
            job = await self._pending.get()
            self.active_request_id = job.request_id
            try:
                await self._loop.run_in_executor(self._executor, self._run_job, job)
            except Exception as exc:  # noqa: BLE001
                self.last_error = f"{type(exc).__name__}: {exc}"
                logger.exception("Qwen3-TTS request %s failed", job.request_id)
                await job.chunks.put(exc)
                if job.done and not job.done.done():
                    job.done.set_exception(exc)
            finally:
                await job.chunks.put(_END)
                self.active_request_id = None
                self._pending.task_done()

    def _emit(self, job: Job, item: Any) -> None:
        assert self._loop is not None
        future = asyncio.run_coroutine_threadsafe(job.chunks.put(item), self._loop)
        future.result()

    @staticmethod
    def _seed_random_generators(seed: int) -> None:
        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _emit_pcm(
        self,
        job: Job,
        progress: GenerationProgress,
        pcm: bytes,
        *,
        mark_first_audio: bool = True,
    ) -> None:
        if not pcm:
            return
        progress.record(pcm, mark_first_audio=mark_first_audio)
        self._emit(job, pcm)

    def _generate_segment(
        self,
        job: Job,
        segment: str,
        limiter: TrailingSilenceLimiter,
        progress: GenerationProgress,
    ) -> None:
        request = job.request
        budget = estimate_token_budget(
            segment,
            chunk_size=request.streaming_chunk_size,
            configured_cap=request.max_new_tokens,
            safety_margin=request.token_safety_margin,
        )
        stream = self.model.generate_custom_voice_streaming(
            text=segment,
            speaker=request.voice,
            language=request.language,
            instruct=request.instruct,
            **generation_options(request, max_new_tokens=budget.resolved),
        )
        for audio, source_rate, _timing in stream:
            if job.cancelled.is_set():
                raise RuntimeError("request cancelled")
            if int(source_rate) != self.sample_rate:
                raise RuntimeError(
                    f"model returned {source_rate} Hz, expected {self.sample_rate} Hz"
                )
            for pcm in limiter.push(float_to_pcm16(audio)):
                self._emit_pcm(job, progress, pcm)

        self._emit_pcm(job, progress, limiter.finish())

    def _run_job(self, job: Job) -> None:
        request = job.request
        progress = GenerationProgress.start(job, self.sample_rate)
        self._seed_random_generators(request.seed)

        segments = split_tts_text(request.input, max_chars=request.max_segment_chars)
        if not segments:
            raise ValueError("input did not contain synthesizable text")

        limiter = TrailingSilenceLimiter(sample_rate=self.sample_rate)
        for segment_index, segment in enumerate(segments):
            self._generate_segment(job, segment, limiter, progress)

            if segment_index + 1 < len(segments) and request.segment_pause_ms:
                pause = silence_pcm(self.sample_rate, request.segment_pause_ms)
                self._emit_pcm(job, progress, pause, mark_first_audio=False)

        metrics = progress.metrics(job, segment_count=len(segments))
        if job.done and not job.done.done():
            assert self._loop is not None
            self._loop.call_soon_threadsafe(job.done.set_result, metrics)

    async def iter_pcm(self, job: Job):
        while True:
            item = await job.chunks.get()
            if item is _END:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    def health(self) -> dict[str, Any]:
        gpu: dict[str, Any] = {"available": False}
        try:
            import torch

            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                gpu = {
                    "available": True,
                    "device": torch.cuda.get_device_name(),
                    "free_bytes": int(free),
                    "total_bytes": int(total),
                }
        except Exception as exc:  # noqa: BLE001
            gpu = {"available": False, "error": str(exc)}

        return {
            "status": "ok" if self.model is not None else "loading",
            "model": self.config.model,
            "backend": self.config.backend,
            "sample_rate": self.sample_rate,
            "default_voice": self.config.default_speaker,
            "voices": self.voices,
            "queue": {
                "active_request_id": self.active_request_id,
                "waiting": self._pending.qsize(),
                "max_waiting": self.config.queue_max_pending,
            },
            "gpu": gpu,
            "uptime_seconds": round(time.time() - self.started_at, 3),
            "last_error": self.last_error,
        }


END_OF_STREAM = _END
