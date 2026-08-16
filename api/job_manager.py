from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from api.models import JobStatus


@dataclass
class JobRecord:
    job_id: str
    status: JobStatus
    profile: str
    mode: str
    audio_filename: str
    created_at: datetime
    output_dir: Path
    completed_at: datetime | None = None
    error: str | None = None
    artifacts: list[str] | None = None
    metrics: dict[str, object] | None = None


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()

    def create(
        self, profile: str, mode: str, audio_filename: str, output_dir: Path
    ) -> JobRecord:
        job_id = uuid.uuid4().hex
        job = JobRecord(
            job_id=job_id,
            status=JobStatus.queued,
            profile=profile,
            mode=mode,
            audio_filename=audio_filename,
            created_at=datetime.now(tz=UTC),
            output_dir=output_dir,
        )
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **kwargs: object) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for key, value in kwargs.items():
                setattr(job, key, value)

    def list_all(self) -> list[JobRecord]:
        with self._lock:
            return list(self._jobs.values())

    def delete(self, job_id: str) -> bool:
        with self._lock:
            return self._jobs.pop(job_id, None) is not None


job_manager = JobManager()
