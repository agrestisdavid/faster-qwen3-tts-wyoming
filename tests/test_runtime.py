from __future__ import annotations

from qwen3_tts_worker.runtime import (
    GenerationProgress,
    Job,
    generation_options,
)
from qwen3_tts_worker.schemas import SpeechRequest


def resolved_request() -> SpeechRequest:
    return SpeechRequest(
        input="A short synthesis test.",
        voice="sohee",
        language="english",
    )


def test_generation_options_map_request_fields() -> None:
    request = resolved_request()

    assert generation_options(request, max_new_tokens=720) == {
        "non_streaming_mode": request.non_streaming_mode,
        "max_new_tokens": 720,
        "temperature": request.temperature,
        "top_k": request.top_k,
        "top_p": request.top_p,
        "do_sample": request.do_sample,
        "repetition_penalty": request.repetition_penalty,
        "chunk_size": request.streaming_chunk_size,
    }


def test_generation_progress_builds_metrics_without_counting_pause_as_ttfa() -> None:
    job = Job(
        request=resolved_request(),
        request_id="request-1",
        created_at=9.0,
    )
    progress = GenerationProgress(
        started_at=10.0,
        queue_wait_ms=1000.0,
        sample_rate=1000,
    )

    progress.record(bytes(100), emitted_at=10.1, mark_first_audio=False)
    assert progress.first_audio_at is None

    progress.record(bytes(400), emitted_at=10.25)
    metrics = progress.metrics(job, segment_count=2, finished_at=11.0)

    assert metrics.request_id == "request-1"
    assert metrics.queue_wait_ms == 1000.0
    assert metrics.ttfa_ms == 250.0
    assert metrics.total_ms == 1000.0
    assert metrics.samples == 250
    assert metrics.audio_seconds == 0.25
    assert metrics.rtf == 4.0
    assert metrics.segments == 2
