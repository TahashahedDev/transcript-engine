"""
Unit tests for the stuck-job stall watchdog (api/pipeline_runner._stall_watchdog).

Pure threading/timing logic — no pipeline, no models, no GPU. Timeouts here are
in fractions of a second so the suite stays fast; production defaults are in
minutes (see _STALL_TIMEOUT_S).
"""

from __future__ import annotations

import threading
import time

from api.pipeline_runner import _stall_watchdog


def _run_watchdog(
    last_progress: dict[str, float],
    stop: threading.Event,
    timeout_s: float,
    check_interval_s: float,
    stalls: list[float],
) -> threading.Thread:
    t = threading.Thread(
        target=_stall_watchdog,
        args=("job-1", stop, last_progress, stalls.append, timeout_s, check_interval_s),
        daemon=True,
    )
    t.start()
    return t


class TestStallWatchdog:
    def test_fires_when_progress_goes_stale(self) -> None:
        last_progress = {"t": time.monotonic()}
        stop = threading.Event()
        stalls: list[float] = []

        t = _run_watchdog(last_progress, stop, timeout_s=0.15, check_interval_s=0.05, stalls=stalls)
        t.join(timeout=2.0)

        assert len(stalls) == 1, "watchdog should fire exactly once on a stalled job"
        assert stalls[0] >= 0.15
        assert not t.is_alive(), "watchdog must exit after firing, not keep looping"

    def test_does_not_fire_while_progress_keeps_advancing(self) -> None:
        last_progress = {"t": time.monotonic()}
        stop = threading.Event()
        stalls: list[float] = []

        t = _run_watchdog(last_progress, stop, timeout_s=0.3, check_interval_s=0.05, stalls=stalls)

        # Simulate a long-but-healthy job: keep bumping the heartbeat for well
        # past the stall timeout. This is the "no false failure for a
        # legitimately long transcription" guarantee.
        deadline = time.monotonic() + 0.9
        while time.monotonic() < deadline:
            last_progress["t"] = time.monotonic()
            time.sleep(0.02)

        stop.set()
        t.join(timeout=2.0)

        assert stalls == [], "watchdog must not fire while the pipeline is still progressing"

    def test_stop_event_terminates_watchdog_without_firing(self) -> None:
        """The finally block in _run_pipeline_sync sets this on every exit path —
        success, failure, and exception — so the thread must not leak."""
        last_progress = {"t": time.monotonic()}
        stop = threading.Event()
        stalls: list[float] = []

        t = _run_watchdog(last_progress, stop, timeout_s=5.0, check_interval_s=0.05, stalls=stalls)
        stop.set()
        t.join(timeout=2.0)

        assert not t.is_alive(), "watchdog thread must exit when stopped"
        assert stalls == []

    def test_handler_exception_does_not_crash_watchdog_thread(self) -> None:
        last_progress = {"t": time.monotonic()}
        stop = threading.Event()

        def boom(idle_s: float) -> None:
            raise RuntimeError("job_manager unavailable")

        t = threading.Thread(
            target=_stall_watchdog,
            args=("job-1", stop, last_progress, boom, 0.1, 0.05),
            daemon=True,
        )
        t.start()
        t.join(timeout=2.0)

        assert not t.is_alive(), "a failing stall handler must not leave the thread hung"
