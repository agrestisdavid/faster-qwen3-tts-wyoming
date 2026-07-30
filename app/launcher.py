from __future__ import annotations

import logging
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence

LOGGER = logging.getLogger("launcher")
QWEN_HEALTH_URL = "http://127.0.0.1:7860/health"
WYOMING_HOST = "127.0.0.1"
WYOMING_PORT = 10210


def qwen_command() -> list[str]:
    return [
        sys.executable,
        "-m",
        "uvicorn",
        "qwen3_tts_worker.app:app",
        "--host",
        "0.0.0.0",
        "--port",
        "7860",
    ]


def wyoming_command() -> list[str]:
    return [sys.executable, "-m", "wyoming_qwen3_tts"]


def wait_for_http(
    process: subprocess.Popen[bytes],
    url: str,
    timeout_seconds: float,
    *,
    opener: Callable[..., object] = urllib.request.urlopen,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    should_stop: Callable[[], bool] = lambda: False,
) -> bool:
    deadline = monotonic() + timeout_seconds
    while monotonic() < deadline:
        if should_stop() or process.poll() is not None:
            return False
        try:
            response = opener(url, timeout=5)
            status = getattr(response, "status", 200)
            close = getattr(response, "close", None)
            if close:
                close()
            if status == 200:
                return True
        except (OSError, urllib.error.URLError):
            pass
        sleep(1)
    return False


def wait_for_tcp(
    process: subprocess.Popen[bytes],
    host: str,
    port: int,
    timeout_seconds: float,
    *,
    should_stop: Callable[[], bool] = lambda: False,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if should_stop() or process.poll() is not None:
            return False
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def stop_processes(processes: Sequence[subprocess.Popen[bytes]]) -> None:
    alive = [process for process in processes if process.poll() is None]
    for process in reversed(alive):
        process.terminate()

    deadline = time.monotonic() + 20
    while alive and time.monotonic() < deadline:
        alive = [process for process in alive if process.poll() is None]
        if alive:
            time.sleep(0.2)

    for process in alive:
        process.kill()
    for process in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def startup_failure_code(
    name: str,
    process: subprocess.Popen[bytes],
    timeout_message: str,
    *,
    stopping: bool,
) -> int:
    if stopping:
        return 0
    return_code = process.poll()
    if return_code is None:
        LOGGER.error(timeout_message)
    else:
        LOGGER.error("%s exited with code %s", name, return_code)
    return 1


def monitor_processes(
    services: Sequence[tuple[str, subprocess.Popen[bytes]]],
    *,
    should_stop: Callable[[], bool],
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    while not should_stop():
        for name, process in services:
            return_code = process.poll()
            if return_code is not None:
                LOGGER.error(
                    "%s exited unexpectedly with code %s",
                    name,
                    return_code,
                )
                return return_code or 1
        sleep(0.5)
    return 0


def run() -> int:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    startup_timeout = float(os.getenv("QWEN3_TTS_STARTUP_TIMEOUT_SECONDS", "1800"))
    stopping = False
    processes: list[subprocess.Popen[bytes]] = []

    def request_stop(signum: int, _frame: object) -> None:
        nonlocal stopping
        if not stopping:
            LOGGER.info("Received signal %s, stopping services", signum)
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    try:
        LOGGER.info("Starting Qwen3-TTS API and WebUI")
        qwen = subprocess.Popen(qwen_command())
        processes.append(qwen)
        if not wait_for_http(
            qwen,
            QWEN_HEALTH_URL,
            startup_timeout,
            should_stop=lambda: stopping,
        ):
            return startup_failure_code(
                "Qwen3-TTS",
                qwen,
                f"Qwen3-TTS did not become healthy within {startup_timeout:.0f} seconds",
                stopping=stopping,
            )

        LOGGER.info("Qwen3-TTS is healthy; starting Wyoming adapter")
        wyoming = subprocess.Popen(wyoming_command())
        processes.append(wyoming)
        if not wait_for_tcp(
            wyoming,
            WYOMING_HOST,
            WYOMING_PORT,
            30,
            should_stop=lambda: stopping,
        ):
            return startup_failure_code(
                "Wyoming adapter",
                wyoming,
                f"Wyoming adapter did not open port {WYOMING_PORT}",
                stopping=stopping,
            )

        LOGGER.info("All services are ready")
        return monitor_processes(
            (("Qwen3-TTS", qwen), ("Wyoming", wyoming)),
            should_stop=lambda: stopping,
        )
    finally:
        stop_processes(processes)


if __name__ == "__main__":
    raise SystemExit(run())
