FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ARG TORCH_VERSION=2.7.1

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH="/app" \
    HF_HOME="/config/huggingface" \
    TRANSFORMERS_CACHE="/config/huggingface" \
    GRADIO_TEMP_DIR="/config/gradio" \
    TORCH_HOME="/config/torch" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ffmpeg \
        git \
        gosu \
        libsndfile1 \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv \
    && python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install \
        "torch==${TORCH_VERSION}+cu128" \
        "torchaudio==${TORCH_VERSION}+cu128" \
        "faster-qwen3-tts==0.3.2" \
        "qwentts-cpp-python==0.3.1" \
        "gradio==6.17.3" \
        "huggingface-hub==0.36.2" \
        "fastapi==0.139.2" \
        "uvicorn==0.51.0" \
        "httpx==0.28.1" \
        "soundfile==0.14.0" \
        "wyoming>=1.8,<2" \
        --extra-index-url https://download.pytorch.org/whl/cu128

COPY app /app
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

RUN chmod 0755 /usr/local/bin/docker-entrypoint.sh \
    && mkdir -p /config

ARG VERSION=dev
ARG VCS_REF=unknown
ARG BUILD_DATE=unknown

ENV APP_VERSION=${VERSION}

LABEL org.opencontainers.image.title="Faster Qwen3-TTS with Wyoming" \
      org.opencontainers.image.description="Local Qwen3-TTS API, WebUI, and Wyoming TTS server for Unraid" \
      org.opencontainers.image.source="https://github.com/agrestisdavid/faster-qwen3-tts-wyoming" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.created="${BUILD_DATE}"

VOLUME ["/config"]
EXPOSE 7860 10210

HEALTHCHECK --interval=30s --timeout=10s --start-period=20m --retries=3 \
    CMD ["python", "/app/healthcheck.py"]

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["python", "-m", "launcher"]
