"""
Global model registry singleton.

Models are loaded once and reused across all jobs in the process lifetime.
This avoids the 30-60s reload penalty on every transcription.

The registry is created lazily on first job submission and kept alive
until the process exits.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transcript_engine.model_registry.registry import ModelRegistry

_registry: ModelRegistry | None = None
_lock = threading.Lock()


def get_registry() -> ModelRegistry:
    """Return the process-wide ModelRegistry, creating it if needed."""
    global _registry  # noqa: PLW0603
    if _registry is None:
        with _lock:
            if _registry is None:
                from transcript_engine.config.loader import load_settings
                from transcript_engine.model_registry.registry import ModelRegistry

                settings = load_settings()
                _registry = ModelRegistry(settings)

    return _registry


def reset_registry() -> None:
    """Release all cached models and reset the registry (used in tests / after config change)."""
    global _registry  # noqa: PLW0603
    with _lock:
        if _registry is not None:
            _registry.release_all()
            _registry = None
