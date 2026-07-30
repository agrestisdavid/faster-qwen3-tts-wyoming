from __future__ import annotations

from app_metadata import APP_VERSION
from wyoming_qwen3_tts import __version__


def test_components_share_the_build_version() -> None:
    assert APP_VERSION == "dev"
    assert __version__ == APP_VERSION
