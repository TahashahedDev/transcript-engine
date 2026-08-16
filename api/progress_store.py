"""
Per-job event queues for SSE progress streaming.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

_store: dict[str, asyncio.Queue[dict[str, Any]]] = {}
_lock = threading.Lock()


def create_queue(job_id: str) -> asyncio.Queue[dict[str, Any]]:
    # Cap at 2048 events. At 5s resource-monitor ticks a 90-min job produces
    # ~1080 resource events + stage/ticker events ≈ 1200 total. 2048 gives
    # comfortable headroom without unbounded memory growth on disconnected clients.
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=2048)
    with _lock:
        _store[job_id] = q
    return q


def get_queue(job_id: str) -> asyncio.Queue[dict[str, Any]] | None:
    with _lock:
        return _store.get(job_id)


def remove_queue(job_id: str) -> None:
    with _lock:
        _store.pop(job_id, None)
