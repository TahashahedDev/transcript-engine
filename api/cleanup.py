"""
Background cleanup task: delete output directories older than the configured TTL.
Runs in the FastAPI lifespan as a fire-and-forget asyncio task.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

from api.config import config

_log = logging.getLogger(__name__)

_INTERVAL_SECONDS = 3600  # run every hour


def _cleanup_once() -> tuple[int, int]:
    """
    Remove expired subdirectories from outputs/ and temp/.
    Returns (removed, kept).

    temp/ uses a 2-hour grace period on top of the configured TTL so this task
    never races with an active pipeline's finally block (which deletes the audio
    file after processing completes).
    """
    now = datetime.now(UTC)
    output_cutoff = now - timedelta(hours=config.output_ttl_hours)
    # Extra buffer so we never delete audio files that are still being processed.
    temp_cutoff = now - timedelta(hours=config.output_ttl_hours + 2)

    removed = kept = 0

    for root, cutoff in (
        (Path(config.output_dir), output_cutoff),
        (Path(config.temp_dir), temp_cutoff),
    ):
        if not root.exists():
            continue
        for entry in root.iterdir():
            if not entry.is_dir():
                continue
            mtime = datetime.fromtimestamp(entry.stat().st_mtime, tz=UTC)
            if mtime < cutoff:
                try:
                    shutil.rmtree(entry)
                    removed += 1
                    _log.info(
                        "cleanup: removed %s/%s (age %.1fh)",
                        root.name,
                        entry.name,
                        (now - mtime).total_seconds() / 3600,
                    )
                except Exception as exc:
                    _log.warning(
                        "cleanup: could not remove %s/%s: %s",
                        root.name,
                        entry.name,
                        exc,
                    )
            else:
                kept += 1

    return removed, kept


async def cleanup_loop() -> None:
    """Run cleanup every hour indefinitely. Designed to run inside FastAPI lifespan."""
    while True:
        try:
            loop = asyncio.get_running_loop()
            removed, kept = await loop.run_in_executor(None, _cleanup_once)
            if removed:
                _log.info(
                    "cleanup: removed %d dir(s), %d still within TTL", removed, kept
                )
        except Exception as exc:
            _log.warning("cleanup loop error: %s", exc)
        await asyncio.sleep(_INTERVAL_SECONDS)
