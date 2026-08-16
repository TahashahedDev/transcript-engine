"""
Background mode scheduling for the transcription pipeline.

Implements macOS QOS and I/O priority controls so the transcription engine
behaves like a professional background application (Ollama / LM Studio style)
on a developer workstation. On Linux inference servers this is off by default
— see enter_background_mode() for the rationale and the TE_BACKGROUND_MODE
override.

Thread model:
    Each pipeline worker thread calls enter_background_mode() once at startup.
    The QOS class is per-thread; I/O policy is process-wide but only affects
    disk reads — the FastAPI event loop does negligible I/O.

QOS_CLASS_UTILITY (0x11):
    - Transcription threads still run on performance cores.
    - macOS scheduler immediately pre-empts utility threads when an interactive
      app (Chrome, VS Code, Cursor) needs CPU time.
    - When the system is idle, utility threads run at full speed.
    - This is the same class used by Final Cut Pro / DaVinci Resolve renders.

QOS_CLASS_BACKGROUND (0x09) was considered and rejected:
    - On M1, background threads are pinned to efficiency cores exclusively.
    - Efficiency cores run at ~1.0-1.3 GHz vs 3.2 GHz for performance cores.
    - This would slow transcription by ~2-3x (60-min job → 90-130 min).

IOPOL_THROTTLE:
    - Model weight reads and audio file reads use the lowest I/O scheduling band.
    - UI apps (Chrome prefetch, VS Code extension I/O) are never starved by us.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import platform

log = logging.getLogger(__name__)

# macOS QOS classes (sys/qos.h)
_QOS_CLASS_UTILITY = 0x11  # long-running, yields to interactive/user-initiated
_QOS_CLASS_BACKGROUND = 0x09  # maintenance; M1 pins to efficiency cores (not used)

# macOS I/O policy constants (sys/resource.h)
_IOPOL_TYPE_DISK = 0
_IOPOL_SCOPE_PROCESS = 0
_IOPOL_THROTTLE = 3  # lowest I/O priority band


def enter_background_mode() -> None:
    """
    Lower the scheduling priority of the calling thread to utility level.

    Call once from the pipeline worker thread before any CPU-intensive work.
    Safe to call multiple times (idempotent after the first call).

    Effect:
        macOS  — QOS_CLASS_UTILITY + IOPOL_THROTTLE (process-wide I/O)
        Linux  — ionice class idle (best-effort), OFF by default (see below)
        Other  — no-op (graceful degradation)

    Default by platform, and why they differ:
        This whole mechanism exists to keep transcription from starving
        interactive apps on a *workstation*. A dedicated inference server has
        no interactive apps to protect, and de-prioritising its disk I/O only
        slows the multi-GB model reads that dominate cold start. So it is on by
        default on macOS (the dev-laptop case) and off by default on Linux (the
        rented-GPU case).

    Override explicitly with TE_BACKGROUND_MODE=1 (force on) or 0 (force off).
    """
    system = platform.system()
    override = os.environ.get("TE_BACKGROUND_MODE", "").strip().lower()

    if override in ("0", "false", "no"):
        log.debug("Scheduling: background mode disabled via TE_BACKGROUND_MODE")
        return
    forced_on = override in ("1", "true", "yes")

    if system == "Darwin":
        _enter_macos()
    elif system == "Linux":
        if forced_on:
            _enter_linux()
        else:
            log.debug(
                "Scheduling: background mode off (Linux server default). "
                "Set TE_BACKGROUND_MODE=1 to de-prioritise this process."
            )
    # Windows / unknown: no-op; we rely on cpu_threads reduction for impact


def _enter_macos() -> None:
    """macOS: per-thread QOS_CLASS_UTILITY + process-wide IOPOL_THROTTLE."""
    # 1. Thread QOS class: yields to Chrome/VS Code/Cursor when they need CPU
    try:
        libpthread = ctypes.CDLL("libpthread.dylib", use_errno=True)
        ret = libpthread.pthread_set_qos_class_self_np(
            ctypes.c_uint32(_QOS_CLASS_UTILITY),
            ctypes.c_int(0),
        )
        if ret == 0:
            log.info(
                "Scheduling: QOS_CLASS_UTILITY active — "
                "transcription yields CPU to interactive apps"
            )
        else:
            log.debug("pthread_set_qos_class_self_np returned %d", ret)
    except Exception as exc:
        log.debug("QOS class set failed (non-fatal): %s", exc)

    # 2. Process I/O policy: model reads do not compete with UI app I/O
    try:
        libc_path = ctypes.util.find_library("c") or "/usr/lib/libc.dylib"
        libc = ctypes.CDLL(libc_path, use_errno=True)
        ret = libc.setiopolicy_np(
            ctypes.c_int(_IOPOL_TYPE_DISK),
            ctypes.c_int(_IOPOL_SCOPE_PROCESS),
            ctypes.c_int(_IOPOL_THROTTLE),
        )
        if ret == 0:
            log.info("Scheduling: IOPOL_THROTTLE active — I/O does not compete with UI")
        else:
            log.debug("setiopolicy_np returned %d", ret)
    except Exception as exc:
        log.debug("I/O policy set failed (non-fatal): %s", exc)


def _enter_linux() -> None:
    """Linux: ionice class idle via ioprio_set syscall."""
    try:
        # ioprio_set(IOPRIO_WHO_PROCESS=1, pid=0, ioprio)
        # IOPRIO_CLASS_IDLE=3, data=7 → lowest I/O priority
        ioprio = (3 << 13) | 7

        import platform as _p  # noqa: PLC0415

        machine = _p.machine()
        nr = {"x86_64": 251, "aarch64": 30, "arm64": 30}.get(machine)
        if nr:
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            libc.syscall(nr, 1, 0, ioprio)
            log.info("Scheduling: ionice IDLE class active")
    except Exception as exc:
        log.debug("ionice set failed (non-fatal): %s", exc)
