from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPOSE_17B = ROOT / "compose.1.7b.yaml"
COMPOSE_06B = ROOT / "compose.0.6b.yaml"
MODEL_17B = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
MODEL_06B = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"


def test_compose_examples_differ_only_by_model() -> None:
    compose_17b = COMPOSE_17B.read_text(encoding="utf-8")
    compose_06b = COMPOSE_06B.read_text(encoding="utf-8")

    assert MODEL_17B in compose_17b
    assert MODEL_06B not in compose_17b
    assert MODEL_06B in compose_06b
    assert MODEL_17B not in compose_06b
    assert compose_17b.replace(MODEL_17B, "<MODEL>") == compose_06b.replace(
        MODEL_06B,
        "<MODEL>",
    )


def test_compose_examples_are_self_contained() -> None:
    for compose_path in (COMPOSE_17B, COMPOSE_06B):
        compose = compose_path.read_text(encoding="utf-8")

        assert "${" not in compose
        assert "env_file:" not in compose
        assert "container_name:" not in compose
        assert "ghcr.io/agrestisdavid/faster-qwen3-tts-wyoming:1.0.0" in compose
        assert '"7860:7860"' in compose
        assert '"10210:10210"' in compose
        assert "capabilities:" in compose
        assert "- gpu" in compose
        assert "qwen3-tts-config:/config" in compose
