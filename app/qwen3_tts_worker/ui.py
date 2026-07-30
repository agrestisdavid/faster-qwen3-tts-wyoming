from __future__ import annotations

import json
import tempfile
from typing import Any

import gradio as gr
import httpx

from .config import PRESET_FIELDS, Settings, build_presets


def _preset_values(presets: dict[str, dict[str, Any]], name: str):
    preset = presets[name]
    return tuple(preset[field] for field in PRESET_FIELDS)


def _synthesize(
    api_url: str,
    text: str,
    language: str,
    voice: str,
    *values: Any,
):
    payload = dict(zip(PRESET_FIELDS, values, strict=True))
    payload.update(
        {
            "input": text,
            "language": language,
            "voice": voice,
            "response_format": "wav",
            "stream": False,
        }
    )
    response = httpx.post(f"{api_url}/v1/audio/speech", json=payload, timeout=300)
    response.raise_for_status()
    temp = tempfile.NamedTemporaryFile(prefix="qwen3-tts-", suffix=".wav", delete=False)
    temp.write(response.content)
    temp.close()
    metrics = json.loads(response.headers.get("x-qwen3-tts-metrics", "{}"))
    summary = (
        f"TTFA: {metrics.get('ttfa_ms', 0):.0f} ms · "
        f"Runtime: {metrics.get('total_ms', 0) / 1000:.2f} s · "
        f"Audio: {metrics.get('audio_seconds', 0):.2f} s · "
        f"RTF: {metrics.get('rtf', 0):.3f}"
    )
    return temp.name, temp.name, summary, metrics


def _export_preset(*values: Any) -> str:
    return json.dumps(
        dict(zip(PRESET_FIELDS, values, strict=True)),
        ensure_ascii=False,
        indent=2,
    )


def _import_preset(raw: str, production: dict[str, Any]):
    data = json.loads(raw)
    unknown = sorted(set(data) - set(PRESET_FIELDS))
    if unknown:
        raise gr.Error(f"Unknown fields: {', '.join(unknown)}")
    merged = {**production, **data}
    return tuple(merged[field] for field in PRESET_FIELDS)


def create_ui(config: Settings, voices: list[str]) -> gr.Blocks:
    api_url = "http://127.0.0.1:7860"
    presets = build_presets(config)
    production = presets["Production"]
    with gr.Blocks(title="Qwen3-TTS Lab") as demo:
        gr.Markdown(
            "# Qwen3-TTS Lab\n"
            f"Shared CustomVoice worker using `{config.model}` with native 24 kHz output. "
            "Changes on this page apply only to the current test synthesis."
        )
        with gr.Row():
            text = gr.Textbox(
                label="Text",
                lines=7,
                max_lines=14,
                value="Hello! This is an English quality test using Qwen3-TTS.",
            )
            instruction = gr.Textbox(
                label="Style Instruction (optional)",
                lines=7,
                max_lines=14,
                value=config.style,
            )
        with gr.Row():
            language = gr.Dropdown(
                label="Language",
                choices=["german", "english", "korean", "japanese", "chinese"],
                value=config.default_language,
            )
            voice = gr.Dropdown(
                label="Voice",
                choices=voices,
                value=config.default_speaker,
            )
            preset = gr.Dropdown(
                label="Preset",
                choices=list(presets),
                value="Production",
            )
        with gr.Accordion("Sampling and Segmentation", open=False):
            with gr.Row():
                temperature = gr.Slider(
                    0.05,
                    2.0,
                    config.temperature,
                    step=0.05,
                    label="Temperature",
                )
                top_p = gr.Slider(0.05, 1.0, config.top_p, step=0.05, label="Top-P")
                top_k = gr.Slider(1, 200, config.top_k, step=1, label="Top-K")
                repetition_penalty = gr.Slider(
                    0.8,
                    2.0,
                    config.repetition_penalty,
                    step=0.01,
                    label="Repetition Penalty",
                )
            with gr.Row():
                do_sample = gr.Checkbox(config.do_sample, label="Sampling")
                non_streaming = gr.Checkbox(
                    config.non_streaming_mode,
                    label="Non-Streaming Text Mode",
                )
                seed = gr.Number(value=config.seed, precision=0, label="Seed")
                max_new_tokens = gr.Slider(
                    64,
                    4096,
                    config.max_new_tokens,
                    step=1,
                    label="Maximum Tokens",
                )
            with gr.Row():
                chunk_size = gr.Slider(
                    1,
                    120,
                    config.streaming_chunk_size,
                    step=1,
                    label="Streaming Chunk Size",
                )
                safety_margin = gr.Slider(
                    1.0,
                    8.0,
                    config.token_safety_margin,
                    step=0.1,
                    label="Token Safety Margin",
                )
                segment_chars = gr.Slider(
                    80,
                    500,
                    config.max_segment_chars,
                    step=1,
                    label="Segment Length (Characters)",
                )
                segment_pause = gr.Slider(
                    0,
                    2000,
                    config.segment_pause_ms,
                    step=10,
                    label="Segment Pause (ms)",
                )

        fields = [
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            do_sample,
            seed,
            max_new_tokens,
            non_streaming,
            chunk_size,
            safety_margin,
            segment_chars,
            segment_pause,
            instruction,
        ]
        preset.change(
            lambda name: _preset_values(presets, name),
            inputs=preset,
            outputs=fields,
        )

        with gr.Row():
            generate = gr.Button("Generate Speech", variant="primary")
            export = gr.Button("Export Preset as JSON")
            import_button = gr.Button("Import JSON")
        preset_json = gr.Code(
            label="Preset JSON",
            language="json",
            value=_export_preset(*_preset_values(presets, "Production")),
        )
        export.click(_export_preset, inputs=fields, outputs=preset_json)
        import_button.click(
            lambda raw: _import_preset(raw, production),
            inputs=preset_json,
            outputs=fields,
        )

        audio = gr.Audio(label="24 kHz Output", type="filepath")
        download = gr.File(label="Download WAV")
        summary = gr.Markdown()
        metrics = gr.JSON(label="Metrics")
        generate.click(
            lambda text_value, language_value, voice_value, *field_values: _synthesize(
                api_url,
                text_value,
                language_value,
                voice_value,
                *field_values,
            ),
            inputs=[text, language, voice, *fields],
            outputs=[audio, download, summary, metrics],
            concurrency_limit=8,
        )

    demo.queue(default_concurrency_limit=8, max_size=config.queue_max_pending)
    return demo
