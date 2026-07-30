# Faster Qwen3-TTS + Wyoming

Run a local Faster Qwen3-TTS WebUI, OpenAI-compatible TTS API, and Wyoming TTS
server in one NVIDIA-enabled Docker container. It supports Docker Compose,
Portainer, and Unraid. The HTTP and Wyoming interfaces share a single loaded
model instance.

| Interface | Default address | Intended use |
|---|---|---|
| Gradio WebUI | `http://HOST-IP:7860/ui/` | Test voices and generation settings |
| OpenAI-compatible API | `http://HOST-IP:7860/v1/audio/speech` | OpenAI-compatible TTS clients |
| Wyoming TTS | `HOST-IP:10210` | Home Assistant and Wyoming clients |

## Architecture

```text
Faster-Qwen3-TTS-Wyoming
├── Qwen3-TTS API + WebUI :7860
└── Wyoming TTS            :10210
        └── HTTP -> 127.0.0.1:7860/v1/audio/speech
```

The container starts Qwen3-TTS first, waits for its health endpoint, and then
opens the Wyoming port. If either managed process exits unexpectedly, the
launcher terminates the other process and exits with an error so Docker can
restart the complete app.

## Requirements

- Linux AMD64 Docker host
- Docker Engine with Docker Compose or a Docker Standalone Portainer environment
- CUDA-capable NVIDIA GPU
- NVIDIA driver and NVIDIA Container Toolkit
- approximately 8 GB of VRAM recommended for the default 1.7B model
- internet access during the first model download

The 0.6B model uses less VRAM and is a better starting point for smaller GPUs.
Models are downloaded at first start and persisted under `/config`.

Before deploying, verify that Docker can access the GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.9.0-base-ubuntu22.04 nvidia-smi
```

## Models

Choose one CustomVoice model:

| Compose file | Model | Use case |
|---|---|---|
| `compose.1.7b.yaml` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | Default; higher quality |
| `compose.0.6b.yaml` | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | Lower VRAM use and faster startup |

Both models support the same CustomVoice speakers:

```text
aiden, dylan, eric, ono_anna, ryan, serena, sohee, uncle_fu, vivian
```

The two Compose files use the same ports and must not run simultaneously on the
same host.

## Docker Compose

Clone the repository and start one model variant:

```bash
git clone https://github.com/agrestisdavid/faster-qwen3-tts-wyoming.git
cd faster-qwen3-tts-wyoming

# Higher-quality 1.7B model
docker compose -f compose.1.7b.yaml up -d

# Or the lower-VRAM 0.6B model
docker compose -f compose.0.6b.yaml up -d
```

No external `.env` file is required. Edit the selected Compose file if you want
to change the timezone, user and group IDs, voice, language, or instruction.

The examples intentionally omit `container_name`. Docker Compose and Portainer
derive stable names from the project or stack name, which avoids global name
collisions and allows independent deployments on the same Docker host.

Check the stack:

```bash
docker compose -f compose.1.7b.yaml ps
docker compose -f compose.1.7b.yaml logs -f qwen3-tts
```

Update the pinned image after the repository publishes a newer version:

```bash
git pull
docker compose -f compose.1.7b.yaml pull
docker compose -f compose.1.7b.yaml up -d
```

Switch models without deleting the persistent model cache:

```bash
docker compose -f compose.1.7b.yaml down
docker compose -f compose.0.6b.yaml up -d
```

Both files use the same named `/config` volume when deployed from the same
directory and Compose project. Do not add `--volumes` to `docker compose down`
unless you intentionally want to delete downloaded models and application data.

## Portainer

GPU assignment in Portainer is supported for NVIDIA GPUs on Docker Standalone
environments. Configure NVIDIA support for the Docker host before deploying the
stack.

1. Open **Stacks**, select **Add stack**, and choose **Repository**.
2. Set the repository URL to
   `https://github.com/agrestisdavid/faster-qwen3-tts-wyoming`.
3. Set the Compose path to either `compose.1.7b.yaml` or
   `compose.0.6b.yaml`.
4. Deploy the stack and allow the initial model download to finish.
5. Open `http://HOST-IP:7860/ui/`.

Use only one model file per host unless you assign different ports and volumes.
Keep the same Portainer stack name and do not remove its named volume when
switching models.

See the
[Portainer GPU documentation](https://docs.portainer.io/user/docker/containers/advanced)
and the
[Docker Compose GPU documentation](https://docs.docker.com/compose/how-tos/gpu-support/)
for host-level GPU setup.

## Install on Unraid

Unraid additionally requires version 6.12 or newer and the NVIDIA Driver
plugin.

1. Open **Apps** and search for `Faster Qwen3-TTS Wyoming`.
2. Verify the appdata path, WebUI port `7860`, and Wyoming port `10210`.
3. Select the desired model and keep the NVIDIA runtime settings enabled.
4. Start the container and allow the initial model download to finish.
5. Open `http://UNRAID-IP:7860/ui/`.

The template can also be installed manually:

```text
https://raw.githubusercontent.com/agrestisdavid/faster-qwen3-tts-wyoming/main/templates/faster-qwen3-tts-wyoming.xml
```

Unraid stores the persistent cache under
`/mnt/user/appdata/faster-qwen3-tts-wyoming`.

## Default Settings

```json
{
  "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
  "voice": "sohee",
  "language": "english",
  "temperature": 0.5,
  "top_p": 0.3,
  "top_k": 150,
  "repetition_penalty": 1.3,
  "do_sample": true,
  "seed": 0,
  "max_new_tokens": 4096,
  "non_streaming_mode": false,
  "streaming_chunk_size": 120,
  "token_safety_margin": 8.0,
  "max_segment_chars": 500,
  "segment_pause_ms": 120,
  "instruct": "Speak naturally, clearly, and warmly in English. Use a calm, fluent pace, lively but restrained prosody, and natural sentence pauses. Articulate sibilants and numbers clearly. Avoid monotonous intonation, exaggerated emotion, and dramatic pitch changes."
}
```

The language, voice, instruction, sampling, streaming, and segmentation values
can be changed in the Compose environment or Unraid template. Existing
container settings continue to override these image defaults after an update.

## OpenAI-Compatible API

Use the base URL `http://HOST-IP:7860/v1` with clients that support an
OpenAI-compatible text-to-speech provider.

```text
Base URL: http://HOST-IP:7860/v1
Model:    tts-1
Voice:    sohee
```

Generate WAV audio:

```bash
curl http://HOST-IP:7860/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1","input":"This is a text-to-speech test.","voice":"sohee","language":"english","response_format":"wav"}' \
  --output qwen-test.wav
```

Stream raw 24 kHz mono PCM16:

```bash
curl http://HOST-IP:7860/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1","input":"This audio is streamed as PCM.","voice":"sohee","language":"english","response_format":"pcm","stream":true}' \
  --output qwen-test.pcm
```

Additional endpoints:

```bash
curl http://HOST-IP:7860/health
curl http://HOST-IP:7860/v1/voices
```

## Home Assistant

In Home Assistant, open **Settings -> Devices & services**, add the
**Wyoming Protocol** integration, and connect it to:

```text
Host: HOST-IP
Port: 10210
```

The configured language is advertised to Wyoming clients automatically.

## Troubleshooting

- First startup can take several minutes while the model downloads and loads.
- If the container is unhealthy, inspect its logs and verify that the NVIDIA
  GPU is visible inside the container.
- If the 1.7B model runs out of VRAM, switch `QWEN3_TTS_MODEL` to the 0.6B
  CustomVoice model and restart the container.
- If a port is already in use, change the corresponding host port in the
  Compose file, Portainer stack, or Unraid template.

## Security

The WebUI and API do not include authentication. Publish their ports only on
a trusted LAN or behind an authenticated reverse proxy. The container does
not run in privileged mode.

## License and Upstream Projects

The integration code, Compose files, and Unraid template in this repository
are licensed under the [MIT License](LICENSE). Models and upstream dependencies
retain their own licenses:

- [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
- [Qwen3-TTS 1.7B CustomVoice model card](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)
- [Qwen3-TTS 0.6B CustomVoice model card](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)
- [Faster Qwen3-TTS](https://github.com/andimarafioti/faster-qwen3-tts)
- [qwentts-cpp-python](https://github.com/andimarafioti/qwentts-cpp-python)
- [OHF-Voice Wyoming protocol](https://github.com/OHF-Voice/wyoming)

See [CONTRIBUTING.md](CONTRIBUTING.md) for development and validation
instructions.
