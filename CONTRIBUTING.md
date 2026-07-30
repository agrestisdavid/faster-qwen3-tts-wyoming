# Contributing

Contributions are welcome through GitHub issues and pull requests.

## Local Validation

Install the test dependencies and run the static checks, template validation,
and unit tests:

```bash
python -m pip install -r requirements-test.txt
python -m ruff check app tests scripts
python -m compileall -q app
python scripts/validate_templates.py
docker compose -f compose.1.7b.yaml config --quiet
docker compose -f compose.0.6b.yaml config --quiet
python -m pytest
```

Build the container locally:

```bash
docker build -t faster-qwen3-tts-wyoming:dev .
```

## Hardware Validation

Model startup and real audio output require an NVIDIA GPU. Before a release,
validate the following on an NVIDIA-enabled Docker host:

1. Check `/health` and `/v1/voices`.
2. Test WAV synthesis and PCM streaming.
3. Test Wyoming `Describe` and `Synthesize`.
4. Test Home Assistant through the Wyoming integration.
5. Inspect `nvidia-smi` and container memory.
6. Recreate the container and confirm cache persistence.
7. Validate both Compose model variants.
8. For Unraid releases, run Community Applications Validate and Scan.

GitHub CI covers Python checks, Compose and XML validation, unit tests, and a
reproducible container image build.
