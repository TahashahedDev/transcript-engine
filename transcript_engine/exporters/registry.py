from __future__ import annotations

from typing import TYPE_CHECKING

from transcript_engine.logging import get_logger

if TYPE_CHECKING:
    from transcript_engine.exporters.base import Exporter

logger = get_logger(__name__)

_registry: dict[str, type[Exporter]] = {}


def register(cls: type[Exporter]) -> type[Exporter]:
    """Class decorator to register an exporter by its format string."""
    _registry[cls.format] = cls
    logger.debug(f"Registered exporter: {cls.format}")
    return cls


def get_exporter(fmt: str) -> type[Exporter]:
    if fmt not in _registry:
        available = ", ".join(sorted(_registry))
        raise KeyError(f"Exporter '{fmt}' not found. Available: {available}")
    return _registry[fmt]


def available_formats() -> list[str]:
    return sorted(_registry.keys())


def _import_builtins() -> None:
    from transcript_engine.exporters import (  # noqa: F401
        json_exporter,
        markdown,
        srt,
        txt,
        vtt,
    )


_import_builtins()
