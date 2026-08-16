"""Cloudflare R2 storage client (S3-compatible via boto3)."""

from __future__ import annotations

import asyncio
import re
import uuid
from functools import lru_cache
from typing import Any

import boto3
from botocore.config import Config as BotocoreConfig
from botocore.exceptions import ClientError
from fastapi import HTTPException

from api.v2.config import v2_settings

_UNSAFE_CHARS = re.compile(r"[^\w\-.]")


def _sanitize_filename(filename: str) -> str:
    """Replace unsafe path characters and truncate to 200 chars."""
    clean = _UNSAFE_CHARS.sub("_", filename)
    return clean[:200] or "audio"


@lru_cache(maxsize=1)
def _get_client() -> Any:
    """Return a lazily-created, module-level S3 client for R2.

    lru_cache does not cache exceptions, so a failed init is retried on the
    next call (useful when config is injected after module import in tests).
    """
    return boto3.client(
        "s3",
        endpoint_url=f"https://{v2_settings.r2_account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=v2_settings.r2_access_key_id,
        aws_secret_access_key=v2_settings.r2_secret_access_key,
        config=BotocoreConfig(signature_version="s3v4"),
        region_name="auto",
    )


def _require_client() -> Any:
    """Return the R2 client or raise 503 if R2 is not configured."""
    if not v2_settings.r2_account_id:
        raise HTTPException(status_code=503, detail="Storage not configured")
    return _get_client()


def is_configured() -> bool:
    """Return True when all required R2 credentials are present."""
    return bool(
        v2_settings.r2_account_id
        and v2_settings.r2_access_key_id
        and v2_settings.r2_secret_access_key
    )


# ── Key helpers ────────────────────────────────────────────────────────────


def make_audio_key(user_id: uuid.UUID, job_id: uuid.UUID, filename: str) -> str:
    """Return the R2 key for the raw audio file."""
    return f"audio/{user_id}/{job_id}/{_sanitize_filename(filename)}"


def make_transcript_key(user_id: uuid.UUID, job_id: uuid.UUID, fmt: str) -> str:
    """Return the R2 key for a completed transcript in the given format."""
    return f"transcripts/{user_id}/{job_id}/transcript.{fmt}"


# ── Async operations ───────────────────────────────────────────────────────


async def generate_upload_url(s3_key: str, content_type: str) -> str:
    """Return a pre-signed PUT URL valid for upload_url_expires seconds."""
    client = _require_client()
    url: str = await asyncio.to_thread(
        client.generate_presigned_url,
        "put_object",
        Params={
            "Bucket": v2_settings.r2_bucket_name,
            "Key": s3_key,
            "ContentType": content_type,
        },
        ExpiresIn=v2_settings.upload_url_expires,
    )
    return url


async def generate_download_url(s3_key: str) -> str:
    """Return a pre-signed GET URL valid for download_url_expires seconds."""
    client = _require_client()
    url: str = await asyncio.to_thread(
        client.generate_presigned_url,
        "get_object",
        Params={"Bucket": v2_settings.r2_bucket_name, "Key": s3_key},
        ExpiresIn=v2_settings.download_url_expires,
    )
    return url


async def verify_object_exists(s3_key: str) -> str:
    """HEAD the object and return its ContentType.

    Raises 400 if the object is not found (upload not yet complete).
    Raises 502 on unexpected R2 errors.
    """
    client = _require_client()
    try:
        resp: dict[str, Any] = await asyncio.to_thread(
            client.head_object,
            Bucket=v2_settings.r2_bucket_name,
            Key=s3_key,
        )
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code in ("404", "NoSuchKey"):
            raise HTTPException(
                status_code=400,
                detail="Audio file not found in storage — upload may not have completed",
            ) from exc
        raise HTTPException(status_code=502, detail="Storage error verifying upload") from exc
    return str(resp.get("ContentType", ""))


async def delete_objects_with_prefix(prefix: str) -> None:
    """Delete all R2 objects whose key starts with prefix (no-op if none exist)."""
    client = _require_client()

    def _do_delete() -> None:
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=v2_settings.r2_bucket_name, Prefix=prefix):
            objects = page.get("Contents", [])
            if not objects:
                continue
            client.delete_objects(
                Bucket=v2_settings.r2_bucket_name,
                Delete={"Objects": [{"Key": obj["Key"]} for obj in objects]},
            )

    await asyncio.to_thread(_do_delete)
