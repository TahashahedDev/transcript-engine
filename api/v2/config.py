"""Settings for the v2 cloud API (R2 storage + Supabase auth)."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

from transcript_engine.config.paths import project_root


class V2Settings(BaseSettings):
    # Project root rather than a CWD-relative ".env" — see db.config for why.
    model_config = SettingsConfigDict(
        env_prefix="TE_", env_file=project_root() / ".env", extra="ignore"
    )

    # Cloudflare R2
    r2_account_id: str = ""
    r2_access_key_id: str = ""
    r2_secret_access_key: str = ""
    r2_bucket_name: str = "transcript-engine"

    # Supabase Auth — JWT secret from project Settings → API → JWT Secret
    supabase_jwt_secret: str = ""

    # Pre-signed URL expiry (seconds)
    upload_url_expires: int = 900    # 15 minutes
    download_url_expires: int = 3600  # 1 hour


v2_settings = V2Settings()
