# Faster Qwen3-TTS + Wyoming for Unraid

Run a local Faster Qwen3-TTS WebUI, OpenAI-compatible TTS API, and Wyoming TTS
server in one NVIDIA-enabled Docker container. The HTTP and Wyoming interfaces
share a single loaded model instance.

| Interface | Default address | Intended use |
|---|---|---|
| Gradio WebUI | `http://UNRAID-IP:7860/ui/` | Test voices and generation settings |
| OpenAI-compatible API | `http://UNRAID-IP:7860/v1/audio/speech` | OpenAI-compatible TTS clients |
| Wyoming TTS | `UNRAID-IP:10210` | Home Assistant and Wyoming clients |

## Architecture

```text
Faster-Qwen3-TTS-Wyoming
├── Qwen3-TTS API + WebUI :7860
└── Wyoming TTS            :10210
        └── HTTP -> 127.0.0.1:7860/v1/audio/speech
```

The container starts Qwen3-TTS first, waits for its health endpoint, and then
opens the Wyoming port. If either managed process exits unexpectedly, the
launcher terminates the other process and exits with an error so Unraid can
restart the complete app.

## Requirements

- Unraid 6.12 or newer
- Unraid NVIDIA Driver plugin
- CUDA-capable NVIDIA GPU
- approximately 8 GB of VRAM recommended for the default 1.7B model
- internet access during the first model download

The 0.6B model uses less VRAM and is a better starting point for smaller GPUs.
Models are downloaded at first start and persisted under
`/mnt/user/appdata/faster-qwen3-tts-wyoming/huggingface`.

## Install on Unraid

1. Open **Apps** and search for `Faster Qwen3-TTS Wyoming`.
2. Verify the appdata path, WebUI port `7860`, and Wyoming port `10210`.
3. Select the desired model and keep the NVIDIA runtime settings enabled.
4. Start the container and allow the initial model download to finish.
5. Open `http://UNRAID-IP:7860/ui/`.

The template can also be installed manually:

```text
https://raw.githubusercontent.com/agrestisdavid/faster-qwen3-tts-wyoming/main/templates/faster-qwen3-tts-wyoming.xml
```

Docker CLI reference:

```bash
docker run -d \
  --name Faster-Qwen3-TTS-Wyoming \
  --restart unless-stopped \
  --runtime=nvidia \
  --gpus all \
  -p 7860:7860 \
  -p 10210:10210 \
  -e PUID=99 \
  -e PGID=100 \
  -e TZ=UTC \
  -e QWEN3_TTS_DEFAULT_VOICE=sohee \
  -v /mnt/user/appdata/faster-qwen3-tts-wyoming:/config \
  ghcr.io/agrestisdavid/faster-qwen3-tts-wyoming:latest
```

## Models

Choose a CustomVoice model through `QWEN3_TTS_MODEL`:

| Model | Use case |
|---|---|
| `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | Default; higher quality |
| `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | Lower VRAM use and faster startup |

Changing the model requires a container restart. Both models support the same
CustomVoice speakers:

```text
aiden, dylan, eric, ono_anna, ryan, serena, sohee, uncle_fu, vivian
```

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
can be changed in the Unraid template. Existing container settings continue to
override these image defaults after an update.

## OpenAI-Compatible API

Use the base URL `http://UNRAID-IP:7860/v1` with clients that support an
OpenAI-compatible text-to-speech provider.

```text
Base URL: http://UNRAID-IP:7860/v1
Model:    tts-1
Voice:    sohee
```

Generate WAV audio:

```bash
curl http://UNRAID-IP:7860/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1","input":"This is a text-to-speech test.","voice":"sohee","language":"english","response_format":"wav"}' \
  --output qwen-test.wav
```

Stream raw 24 kHz mono PCM16:

```bash
curl http://UNRAID-IP:7860/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1","input":"This audio is streamed as PCM.","voice":"sohee","language":"english","response_format":"pcm","stream":true}' \
  --output qwen-test.pcm
```

Additional endpoints:

```bash
curl http://UNRAID-IP:7860/health
curl http://UNRAID-IP:7860/v1/voices
```

## Home Assistant

In Home Assistant, open **Settings -> Devices & services**, add the
**Wyoming Protocol** integration, and connect it to:

```text
Host: UNRAID-IP
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
  Unraid template.

## Security

The WebUI and API do not include authentication. Publish their ports only on
a trusted LAN or behind an authenticated reverse proxy. The container does
not run in privileged mode.

## License and Upstream Projects

The integration code and Unraid template in this repository are licensed
under the [MIT License](LICENSE). Models and upstream dependencies retain their
own licenses:

- [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
- [Qwen3-TTS 1.7B CustomVoice model card](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)
- [Qwen3-TTS 0.6B CustomVoice model card](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)
- [Faster Qwen3-TTS](https://github.com/andimarafioti/faster-qwen3-tts)
- [qwentts-cpp-python](https://github.com/andimarafioti/qwentts-cpp-python)
- [OHF-Voice Wyoming protocol](https://github.com/OHF-Voice/wyoming)

See [CONTRIBUTING.md](CONTRIBUTING.md) for development and validation
instructions.
