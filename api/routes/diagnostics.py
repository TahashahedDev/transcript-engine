"""Diagnostics API — serves processing_report.json and resource_usage.json for a job."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from api.config import config
from api.job_manager import job_manager

router = APIRouter()


@router.get("/jobs/{job_id}/diagnostics")
async def get_diagnostics(job_id: str) -> JSONResponse:
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    output_dir = Path(config.output_dir) / job_id

    result: dict[str, object] = {
        "job_id": job_id,
        "status": job.status.value,
        "processing_report": None,
        "resource_usage": None,
    }

    processing_report_path = output_dir / "processing_report.json"
    if processing_report_path.exists():
        with contextlib.suppress(Exception):
            result["processing_report"] = json.loads(
                processing_report_path.read_text(encoding="utf-8")
            )

    resource_usage_path = output_dir / "resource_usage.json"
    if resource_usage_path.exists():
        with contextlib.suppress(Exception):
            result["resource_usage"] = json.loads(
                resource_usage_path.read_text(encoding="utf-8")
            )

    return JSONResponse(result)
