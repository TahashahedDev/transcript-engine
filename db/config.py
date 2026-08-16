from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class DBSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TE_", env_file=".env", extra="ignore"
    )

    # postgresql+asyncpg://user:pass@host:5432/dbname
    # Set via TE_DATABASE_URL in .env or environment
    database_url: str = ""


settings = DBSettings()
