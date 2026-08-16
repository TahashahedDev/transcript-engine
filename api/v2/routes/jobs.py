"""Cloud job API — v2 endpoints for upload, status polling, downloads, and deletion."""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.v2.auth import get_current_user_id
from api.v2.config import v2_settings
from api.v2.r2 import (
    delete_objects_with_prefix,
    generate_download_url,
    generate_upload_url,
    is_configured,
    make_audio_key,
    make_transcript_key,
    verify_object_exists,
)
from db.connection import get_session
from db.models import Job

router = APIRouter(prefix="/api/v2/jobs", tags=["jobs-v2"])

_AUDIO_CONTENT_TYPE_RE = re.compile(r"^audio/", re.IGNORECASE)
_ALLOWED_FORMATS = frozenset({"txt", "md", "json", "srt", "vtt", "docx"})
_VALID_MODES = frozenset({"fast", "balanced", "accurate"})
_VALID_PROFILE_RE = re.compile(r"^[a-zA-Z0-9_\-]{1,100}$")


# ── Schemas ────────────────────────────────────────────────────────────────


class UploadUrlRequest(BaseModel):
    content_type: str
    filename: str = Field(..., min_length=1, max_length=255)

    @field_validator("content_type")
    @classmethod
    def must_be_audio(cls, v: str) -> str:
        if not _AUDIO_CONTENT_TYPE_RE.match(v):
            raise ValueError("content_type must be an audio/* MIME type")
        return v.lower()


class UploadUrlResponse(BaseModel):
    upload_url: str
    s3_key: str
    expires_in: int


class CreateJobRequest(BaseModel):
    s3_key: str = Field(..., min_length=1, max_length=500)
    mode: str = "balanced"
    profile: str = "generic"

    @field_validator("mode")
    @classmethod
    def valid_mode(cls, v: str) -> str:
        if v not in _VALID_MODES:
            raise ValueError(f"mode must be one of: {', '.join(sorted(_VALID_MODES))}")
        return v

    @field_validator("profile")
    @classmethod
    def valid_profile(cls, v: str) -> str:
        if not _VALID_PROFILE_RE.match(v):
            raise ValueError(
                "profile must be 1-100 characters (letters, digits, underscores, hyphens)"
            )
        return v


class CreateJobResponse(BaseModel):
    job_id: uuid.UUID
    status: str
    trace_id: uuid.UUID


class JobStatusResponse(BaseModel):
    job_id: uuid.UUID
    status: str
    mode: str
    profile: str
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    word_count: int | None = None
    error: str | None = None
    retry_count: int
    trace_id: uuid.UUID


class DownloadUrlResponse(BaseModel):
    url: str
    expires_in: int


# ── Internal helpers ───────────────────────────────────────────────────────


def _assert_key_ownership(s3_key: str, user_id: uuid.UUID) -> None:
    """Raise 403 if s3_key is not scoped to this user's audio prefix."""
    if not s3_key.startswith(f"audio/{user_id}/"):
        raise HTTPException(status_code=403, detail="s3_key does not belong to this user")


async def _fetch_owned_job(
    job_id: uuid.UUID,
    user_id: uuid.UUID,
    session: AsyncSession,
) -> Job:
    """Return the job if it exists and belongs to this user.

    Returns 404 for missing or soft-deleted jobs; 403 for ownership mismatch.
    Using 404 for ownership mismatch would be more opaque, but 403 is clearer
    for API consumers and is safe because job IDs are UUIDs (not guessable).
    """
    result = await session.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    if job is None or job.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    return job


# ── Endpoints ──────────────────────────────────────────────────────────────


@router.post("/upload-url", response_model=UploadUrlResponse, status_code=200)
async def request_upload_url(
    body: UploadUrlRequest,
    user_id: uuid.UUID = Depends(get_current_user_id),
) -> UploadUrlResponse:
    """Issue a pre-signed PUT URL for uploading audio directly to R2.

    The returned s3_key encodes the caller's user_id so ownership can be
    verified when POST /api/v2/jobs is called with that key.
    """
    job_id = uuid.uuid4()
    s3_key = make_audio_key(user_id, job_id, body.filename)
    upload_url = await generate_upload_url(s3_key, body.content_type)
    return UploadUrlResponse(
        upload_url=upload_url,
        s3_key=s3_key,
        expires_in=v2_settings.upload_url_expires,
    )


@router.post("", response_model=CreateJobResponse, status_code=201)
async def create_job(
    body: CreateJobRequest,
    user_id: uuid.UUID = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_session),
) -> CreateJobResponse:
    """Create a transcription job for an already-uploaded audio file.

    Validates that:
    1. The s3_key path belongs to this user (prefix check).
    2. The object exists in R2 and has an audio/* Content-Type.
    """
    _assert_key_ownership(body.s3_key, user_id)

    content_type = await verify_object_exists(body.s3_key)
    if not _AUDIO_CONTENT_TYPE_RE.match(content_type):
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded object has unexpected Content-Type {content_type!r}. Expected audio/*.",
        )

    job = Job(
        user_id=user_id,
        status="queued",
        s3_audio_key=body.s3_key,
        mode=body.mode,
        profile=body.profile,
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)

    return CreateJobResponse(
        job_id=job.id,
        status=job.status,
        trace_id=job.trace_id,
    )


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job(
    job_id: uuid.UUID,
    user_id: uuid.UUID = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_session),
) -> JobStatusResponse:
    """Return the current status of a job (poll every 3-5 seconds)."""
    job = await _fetch_owned_job(job_id, user_id, session)
    return JobStatusResponse(
        job_id=job.id,
        status=job.status,
        mode=job.mode,
        profile=job.profile,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        word_count=job.word_count,
        error=job.error,
        retry_count=job.retry_count,
        trace_id=job.trace_id,
    )


@router.get("/{job_id}/downloads/{fmt}", response_model=DownloadUrlResponse)
async def get_download_url(
    job_id: uuid.UUID,
    fmt: str,
    user_id: uuid.UUID = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_session),
) -> DownloadUrlResponse:
    """Return a pre-signed GET URL for a completed job's transcript."""
    if fmt not in _ALLOWED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{fmt}'. Supported: {', '.join(sorted(_ALLOWED_FORMATS))}",
        )

    job = await _fetch_owned_job(job_id, user_id, session)
    if job.status != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Job is not completed (current status: {job.status!r})",
        )

    s3_key = make_transcript_key(user_id, job_id, fmt)
    url = await generate_download_url(s3_key)
    return DownloadUrlResponse(url=url, expires_in=v2_settings.download_url_expires)


@router.delete("/{job_id}", status_code=200)
async def delete_job(
    job_id: uuid.UUID,
    user_id: uuid.UUID = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_session),
) -> dict[str, bool]:
    """GDPR delete: remove R2 objects then soft-delete the job row.

    R2 deletion is skipped when storage is not configured (no objects exist).
    The DB soft-delete always runs so the record is always tombstoned.
    """
    job = await _fetch_owned_job(job_id, user_id, session)

    if is_configured():
        await delete_objects_with_prefix(f"audio/{user_id}/{job_id}/")
        await delete_objects_with_prefix(f"transcripts/{user_id}/{job_id}/")

    job.deleted_at = datetime.now(UTC)
    job.s3_audio_key = None
    await session.commit()

    return {"deleted": True}
