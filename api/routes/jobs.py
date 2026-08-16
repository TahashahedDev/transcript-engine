from __future__ import annotations

import asyncio
import shutil
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Form, HTTPException, UploadFile

from api.config import config
from api.job_manager import job_manager
from api.models import CreateJobResponse, JobResponse, JobStatus
from api.pipeline_runner import run_pipeline_async
from api.progress_store import create_queue, remove_queue

router = APIRouter()


@router.post("/jobs", response_model=CreateJobResponse)
async def create_job(
    file: UploadFile,
    profile: str = Form("generic"),
    timestamps: str = Form("none"),
    highlight_confidence: bool = Form(False),
    mode: str = Form("balanced"),
) -> CreateJobResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in config.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Accepted: {', '.join(sorted(config.allowed_extensions))}",
        )

    max_bytes = config.max_upload_mb * 1024 * 1024

    temp_dir = Path(config.temp_dir)
    temp_dir.mkdir(exist_ok=True)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)

    job = job_manager.create(
        profile=profile,
        mode=mode,
        audio_filename=file.filename or "upload" + suffix,
        output_dir=output_dir / "placeholder",
    )

    job_output_dir = output_dir / job.job_id
    job_manager.update(job.job_id, output_dir=job_output_dir)
    # pipeline_runner.py creates job_output_dir just before writing artifacts,
    # so we don't create it here — that avoids orphaned empty dirs on failure.

    job_temp_dir = temp_dir / job.job_id
    job_temp_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename or "recording").stem
    safe_name = (
        "".join(c if c.isalnum() or c in " _-" else "_" for c in safe_name).strip("_")
        or "recording"
    )
    audio_path = job_temp_dir / f"{safe_name}{suffix}"

    # Stream the upload to disk in 64 KB chunks so the full file is never held
    # in RAM.  The previous code did `await file.read()` which copied the entire
    # file into a Python bytes object — a 500 MB upload would spike 500 MB of heap.
    t_upload_start = time.monotonic()
    total_bytes = 0
    try:
        with audio_path.open("wb") as fout:
            while True:
                chunk = await file.read(65_536)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum size is {config.max_upload_mb} MB.",
                    )
                fout.write(chunk)
    except Exception:
        # Catches both HTTPException (413, etc.) and unexpected errors such as
        # OSError (disk full) so partial files and the job record never leak.
        audio_path.unlink(missing_ok=True)
        shutil.rmtree(job_temp_dir, ignore_errors=True)
        job_manager.delete(job.job_id)
        raise

    upload_seconds = time.monotonic() - t_upload_start

    q = create_queue(job.job_id)
    asyncio.create_task(
        _run_and_cleanup(
            job_id=job.job_id,
            audio_path=audio_path,
            profile_name=profile,
            mode=mode,
            output_dir=job_output_dir,
            timestamps=timestamps,
            highlight_confidence=highlight_confidence,
            q=q,
            upload_seconds=upload_seconds,
        )
    )
    return CreateJobResponse(job_id=job.job_id, status=JobStatus.queued)


async def _run_and_cleanup(
    job_id: str,
    audio_path: Path,
    profile_name: str,
    mode: str,
    output_dir: Path,
    timestamps: str,
    highlight_confidence: bool,
    q: asyncio.Queue[dict[str, Any]],
    upload_seconds: float = 0.0,
) -> None:
    try:
        await run_pipeline_async(
            job_id=job_id,
            audio_path=audio_path,
            profile_name=profile_name,
            mode=mode,
            output_dir=output_dir,
            timestamps=timestamps,
            highlight_confidence=highlight_confidence,
            progress_queue=q,
            upload_seconds=upload_seconds,
        )
    finally:
        await asyncio.sleep(30)
        remove_queue(job_id)


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str) -> JobResponse:
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobResponse(
        job_id=job.job_id,
        status=job.status,
        profile=job.profile,
        mode=job.mode,
        audio_filename=job.audio_filename,
        created_at=job.created_at,
        completed_at=job.completed_at,
        error=job.error,
        artifacts=job.artifacts,
        metrics=job.metrics,
    )


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str) -> dict[str, bool]:
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    output_dir = Path(config.output_dir) / job_id
    if output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)

    temp_job_dir = Path(config.temp_dir) / job_id
    if temp_job_dir.exists():
        shutil.rmtree(temp_job_dir, ignore_errors=True)

    job_manager.delete(job_id)
    return {"deleted": True}
