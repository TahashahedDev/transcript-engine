from __future__ import annotations

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from transcript_engine.config.paths import project_root, resolve_path


class APIConfig(BaseSettings):
    model_config = SettingsConfigDict(
        # The .env file is resolved against the project root for the same reason
        # the directories below are: the API must behave identically no matter
        # which directory it was launched from.
        env_prefix="TE_API_",
        env_file=project_root() / ".env",
        extra="ignore",
    )

    max_upload_mb: int = 2000
    temp_dir: str = "temp"
    output_dir: str = "outputs"
    profiles_dir: str = "profiles"
    allowed_extensions: set[str] = {
        ".aac",
        ".mp3",
        ".wav",
        ".m4a",
        ".flac",
        ".mp4",
        ".mov",
        ".ogg",
    }
    # Output directories older than this are removed by the background cleanup task.
    # Override with TE_API_OUTPUT_TTL_HOURS=N in .env.
    output_ttl_hours: int = 24

    # Relative values are anchored to the project root instead of the process
    # CWD. Without this, launching the API from another directory made every
    # job fail with "Profile not found: 'generic' in profiles" and scattered
    # outputs into whatever directory happened to be current.
    @field_validator("temp_dir", "output_dir", "profiles_dir", mode="after")
    @classmethod
    def _anchor_to_project_root(cls, value: str) -> str:
        return str(resolve_path(value))

    @property
    def temp_path(self) -> Path:
        return Path(self.temp_dir)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)


config = APIConfig()
