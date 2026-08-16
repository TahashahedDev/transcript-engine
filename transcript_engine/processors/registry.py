from __future__ import annotations

import importlib.metadata
from typing import TYPE_CHECKING

from transcript_engine.logging import get_logger

if TYPE_CHECKING:
    from transcript_engine.processors.base import Processor

logger = get_logger(__name__)

_registry: dict[str, type[Processor]] = {}


def register(cls: type[Processor]) -> type[Processor]:
    """
    Class decorator that registers a Processor by its name attribute.

    @register
    class MyProcessor:
        name = "my_processor"
        ...
    """
    _registry[cls.name] = cls
    logger.debug(f"Registered processor: {cls.name}")
    return cls


def get_processor(name: str) -> type[Processor]:
    """Look up a registered processor class by name."""
    if name not in _registry:
        available = ", ".join(sorted(_registry))
        raise KeyError(
            f"Processor '{name}' not found. Available processors: {available}"
        )
    return _registry[name]


def available_processors() -> list[str]:
    return sorted(_registry.keys())


def load_plugins() -> None:
    """
    Discover and load external processors registered via entry_points.
    Call once at startup. Safe to call multiple times.

    External packages register with:
        [project.entry-points."transcript_engine.processors"]
        my_proc = "my_package.module:MyProcessor"
    """
    group = "transcript_engine.processors"
    try:
        eps = importlib.metadata.entry_points(group=group)
    except Exception as exc:
        logger.warning(f"Could not load processor plugins: {exc}")
        return

    for ep in eps:
        try:
            cls = ep.load()
            if cls.name not in _registry:
                register(cls)
                logger.info(
                    f"Loaded external processor plugin: {cls.name} from {ep.value}"
                )
        except Exception as exc:
            logger.error(f"Failed to load processor plugin '{ep.name}': {exc}")


def _import_builtins() -> None:
    """Import built-in processors to trigger their @register decorators."""
    from transcript_engine.processors import (  # noqa: F401
        cleanup,
        context_correction,
        speaker_formatting,
        vocabulary,
    )


_import_builtins()
