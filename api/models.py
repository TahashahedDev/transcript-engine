from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel


class JobStatus(StrEnum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


class Job(BaseModel):
    job_id: str
    status: JobStatus
    profile: str
    mode: str = "balanced"
    audio_filename: str
    created_at: datetime
    output_dir: str
    completed_at: datetime | None = None
    error: str | None = None
    artifacts: list[str] | None = None
    metrics: dict[str, object] | None = None


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    profile: str
    mode: str = "balanced"
    audio_filename: str
    created_at: datetime
    completed_at: datetime | None = None
    error: str | None = None
    artifacts: list[str] | None = None
    metrics: dict[str, object] | None = None


class CreateJobResponse(BaseModel):
    job_id: str
    status: JobStatus


class ArtifactsResponse(BaseModel):
    files: list[str]
    bundle: str | None = None
