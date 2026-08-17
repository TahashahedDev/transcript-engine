from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

from transcript_engine.config.paths import project_root


class DBSettings(BaseSettings):
    # Anchored to the project root, not the process CWD: alembic and the API are
    # routinely launched from different directories, and a relative ".env" would
    # silently load nothing.
    model_config = SettingsConfigDict(
        env_prefix="TE_", env_file=project_root() / ".env", extra="ignore"
    )

    # postgresql+asyncpg://user:pass@host:5432/dbname
    # Set via TE_DATABASE_URL in .env or environment
    database_url: str = ""


settings = DBSettings()
