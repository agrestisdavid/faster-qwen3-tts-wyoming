# AGENTS.md

Canonical instructions for coding agents working in this repository.

## Project

This repository publishes a native Unraid Community Applications template and
the matching GHCR image:

```text
ghcr.io/agrestisdavid/faster-qwen3-tts-wyoming
```

The container manages exactly two tightly coupled processes:

- Qwen3-TTS FastAPI/Gradio server on port `7860`
- Wyoming TTS adapter on port `10210`

Both processes share one loaded Qwen3-TTS model instance. The Wyoming adapter
calls `http://127.0.0.1:7860/v1/audio/speech` internally.

This repository is intentionally limited to the Qwen3-TTS worker and its
Wyoming TTS adapter. Do not add unrelated services or Docker Compose.

## Important Files

- `Dockerfile`: CUDA image and Python dependencies
- `docker-entrypoint.sh`: PUID/PGID, appdata permissions, and environment setup
- `app/launcher.py`: PID 1 process supervision
- `app/healthcheck.py`: combined Qwen and Wyoming health check
- `app/qwen3_tts_worker/`: FastAPI, WebUI, and model runtime
- `app/wyoming_qwen3_tts/`: Wyoming protocol adapter
- `templates/faster-qwen3-tts-wyoming.xml`: Community Applications template
- `ca_profile.xml`: Community Applications repository profile
- `scripts/validate_templates.py`: local XML validation
- `README.md`: user and operations documentation
- `CONTRIBUTING.md`: contributor validation and release checks

## Compatibility Contracts

Do not change these values without an explicit migration:

- WebUI/API container port: `7860`
- Wyoming container port: `10210`
- OpenAI endpoint: `/v1/audio/speech`
- Wyoming program name: `qwen3-realtime-tts`
- Wyoming voice name: `qwen3-realtime-de`
- audio format: 24 kHz, mono, PCM16
- persistent container path: `/config`

Default model and quality settings are duplicated in
`app/qwen3_tts_worker/config.py`, the Unraid template, and the README. Keep all
three locations synchronized.

## Development Rules

- Never commit secrets, tokens, or private URLs.
- Do not add a Compose file; the native Unraid app is the product.
- Do not integrate unrelated voice services into this image.
- Keep Qwen and the Wyoming adapter in the same container.
- Document new environment variables in code, the template, and the README.
- Validate audio-format changes in both the HTTP API and Wyoming adapter.
- Long text must be segmented and emitted completely.
- The launcher must forward signals and terminate both children after either
  child fails.
- Public documentation, WebUI labels, repository metadata, and template text
  must remain in English.
- Public defaults and examples must remain in English.

## Validation

```bash
python -m pip install -r requirements-test.txt
python -m ruff check app tests scripts
python -m compileall -q app
python scripts/validate_templates.py
python -m pytest
docker build -t faster-qwen3-tts-wyoming:dev .
```

Before a release, also validate on Unraid:

1. Check `/health` and `/v1/voices`.
2. Test WAV synthesis and PCM streaming.
3. Test Wyoming `Describe` and `Synthesize`.
4. Test Home Assistant through the Wyoming integration.
5. Inspect `nvidia-smi` and container memory.
6. Recreate the container and confirm cache persistence.
7. Run Community Applications Validate and Scan.

## Releases

Tags matching `vMAJOR.MINOR.PATCH` publish the image as
`MAJOR.MINOR.PATCH` and `latest`. The Community Applications template points
to `latest`.
