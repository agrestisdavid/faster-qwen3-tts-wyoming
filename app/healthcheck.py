from __future__ import annotations

import json
import socket
import urllib.request


def check_qwen() -> None:
    with urllib.request.urlopen("http://127.0.0.1:7860/health", timeout=5) as response:
        if response.status != 200:
            raise RuntimeError(f"Qwen health returned HTTP {response.status}")
        payload = json.load(response)
    if payload.get("status") != "ok":
        raise RuntimeError(f"Qwen is not ready: {payload.get('status')!r}")


def check_wyoming() -> None:
    with socket.create_connection(("127.0.0.1", 10210), timeout=5):
        pass


def main() -> int:
    check_qwen()
    check_wyoming()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
