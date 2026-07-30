from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

import gradio as gr
from app_metadata import APP_VERSION
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse, Response, StreamingResponse

from .audio import encode_pcm16
from .config import DEFAULT_VOICES, OPENAI_MODEL_ALIASES, settings
from .runtime import WorkerRuntime
from .schemas import SpeechRequest, VoiceList
from .ui import create_ui

runtime = WorkerRuntime(settings)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await runtime.start()
    yield
    await runtime.stop()


api = FastAPI(
    title="Shared Qwen3-TTS Worker",
    version=APP_VERSION,
    lifespan=lifespan,
)


@api.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/ui/")


@api.get("/health")
async def health():
    return runtime.health()


@api.get("/v1/voices", response_model=VoiceList)
async def voices():
    return VoiceList(
        voices=runtime.voices,
        default=settings.default_speaker,
        model=settings.model,
        backend=settings.backend,
        sample_rate=runtime.sample_rate,
    )


@api.post("/v1/audio/speech")
async def speech(request: SpeechRequest):
    if request.model not in (None, "", settings.model, *OPENAI_MODEL_ALIASES):
        raise HTTPException(status_code=422, detail=f"unsupported model {request.model!r}")
    is_pcm = request.response_format in {"pcm", "pcm_s16le"}
    if request.stream and not is_pcm:
        raise HTTPException(
            status_code=422,
            detail="stream=true requires response_format=pcm or pcm_s16le",
        )
    try:
        job = await runtime.submit(request)
    except asyncio.QueueFull:
        raise HTTPException(status_code=429, detail="Qwen3-TTS queue is full") from None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # OpenAI's streaming client controls transport streaming client-side and
    # sends response_format=pcm without a JSON stream flag.
    if request.stream or request.response_format == "pcm":
        headers = {
            "X-Audio-Sample-Rate": str(runtime.sample_rate),
            "X-Audio-Channels": "1",
            "X-Audio-Sample-Format": "s16le",
            "X-Request-ID": job.request_id,
        }
        return StreamingResponse(
            runtime.iter_pcm(job),
            media_type=f"audio/pcm;rate={runtime.sample_rate};channels=1",
            headers=headers,
        )

    pcm = bytearray()
    try:
        async for chunk in runtime.iter_pcm(job):
            pcm.extend(chunk)
        assert job.done is not None
        metrics = await job.done
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        payload, media_type = encode_pcm16(
            bytes(pcm),
            runtime.sample_rate,
            request.response_format,
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return Response(
        content=payload,
        media_type=media_type,
        headers={
            "X-Audio-Sample-Rate": str(runtime.sample_rate),
            "X-Audio-Channels": "1",
            "X-Audio-Sample-Format": "s16le",
            "X-Request-ID": job.request_id,
            "X-Qwen3-TTS-Metrics": json.dumps(metrics.model_dump(), separators=(",", ":")),
        },
    )


ui = create_ui(settings, DEFAULT_VOICES)
app = gr.mount_gradio_app(api, ui, path="/ui")
