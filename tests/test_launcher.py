from __future__ import annotations

from types import SimpleNamespace

from launcher import (
    monitor_processes,
    qwen_command,
    startup_failure_code,
    wait_for_http,
    wait_for_tcp,
    wyoming_command,
)


class FakeProcess:
    def __init__(self, return_code: int | None = None) -> None:
        self.return_code = return_code

    def poll(self) -> int | None:
        return self.return_code


def test_commands_use_single_container_endpoints() -> None:
    assert qwen_command()[-2:] == ["--port", "7860"]
    assert wyoming_command()[-2:] == ["-m", "wyoming_qwen3_tts"]


def test_wait_for_http_succeeds_when_qwen_is_ready() -> None:
    clock = iter([0.0, 0.1])
    closed: list[bool] = []

    def open_ok(_url: str, timeout: int):
        assert timeout == 5
        return SimpleNamespace(status=200, close=lambda: closed.append(True))

    assert wait_for_http(
        FakeProcess(),
        "http://localhost/health",
        10,
        opener=open_ok,
        monotonic=lambda: next(clock),
        sleep=lambda _seconds: None,
    )
    assert closed == [True]


def test_wait_for_http_stops_if_qwen_exits() -> None:
    clock = iter([0.0, 0.1])
    assert not wait_for_http(
        FakeProcess(return_code=1),
        "http://localhost/health",
        10,
        opener=lambda *_args, **_kwargs: None,
        monotonic=lambda: next(clock),
        sleep=lambda _seconds: None,
    )


def test_wait_for_http_stops_during_container_shutdown() -> None:
    clock = iter([0.0, 0.1])
    assert not wait_for_http(
        FakeProcess(),
        "http://localhost/health",
        10,
        opener=lambda *_args, **_kwargs: None,
        monotonic=lambda: next(clock),
        sleep=lambda _seconds: None,
        should_stop=lambda: True,
    )


def test_wait_for_tcp_stops_during_container_shutdown() -> None:
    assert not wait_for_tcp(
        FakeProcess(),
        "127.0.0.1",
        10210,
        10,
        should_stop=lambda: True,
    )


def test_startup_failure_is_success_during_requested_shutdown() -> None:
    assert (
        startup_failure_code(
            "Qwen3-TTS",
            FakeProcess(),
            "timed out",
            stopping=True,
        )
        == 0
    )


def test_monitor_returns_failed_service_exit_code() -> None:
    assert (
        monitor_processes(
            (("Qwen3-TTS", FakeProcess(return_code=7)),),
            should_stop=lambda: False,
            sleep=lambda _seconds: None,
        )
        == 7
    )


def test_monitor_returns_success_after_requested_shutdown() -> None:
    assert (
        monitor_processes(
            (("Qwen3-TTS", FakeProcess()),),
            should_stop=lambda: True,
            sleep=lambda _seconds: None,
        )
        == 0
    )
