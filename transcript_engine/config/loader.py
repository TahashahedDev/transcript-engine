from __future__ import annotations

from pathlib import Path
from typing import Any

from transcript_engine.config.settings import Settings


def load_settings(
    config_file: Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> Settings:
    """
    Build Settings with full layer priority:
      Pydantic defaults → .env → env vars → cli_overrides

    cli_overrides is a flat dict of dotted keys accepted by model_copy,
    e.g. {"pipeline.transcription.language": "en", "log_level": "DEBUG"}.
    """
    settings = Settings()

    if cli_overrides:
        settings = _apply_overrides(settings, cli_overrides)

    return settings


def _apply_overrides(settings: Settings, overrides: dict[str, Any]) -> Settings:
    """Apply a flat dict of overrides using model_copy."""
    update: dict[str, Any] = {}
    for key, value in overrides.items():
        if value is not None:
            update[key] = value
    if update:
        return settings.model_copy(update=update)
    return settings
