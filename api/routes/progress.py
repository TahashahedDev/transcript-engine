from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from api.job_manager import job_manager
from api.models import JobStatus
from api.progress_store import get_queue

router = APIRouter()


@router.get("/jobs/{job_id}/progress")
async def job_progress(job_id: str) -> EventSourceResponse:
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator() -> AsyncGenerator[dict[str, Any], None]:
        if job.status == JobStatus.completed:
            yield {"data": json.dumps({"type": "completed", "fraction": 1.0})}
            return
        if job.status == JobStatus.failed:
            yield {
                "data": json.dumps(
                    {"type": "error", "message": job.error or "Job failed"}
                )
            }
            return

        q = get_queue(job_id)
        if q is None:
            current = job_manager.get(job_id)
            if current and current.status == JobStatus.completed:
                yield {"data": json.dumps({"type": "completed", "fraction": 1.0})}
            else:
                yield {
                    "data": json.dumps(
                        {"type": "error", "message": "Job queue unavailable"}
                    )
                }
            return

        while True:
            try:
                event = await asyncio.wait_for(q.get(), timeout=30.0)
                yield {"data": json.dumps(event)}
                if event.get("type") in ("completed", "error"):
                    break
            except TimeoutError:
                yield {"comment": "keepalive"}
                current = job_manager.get(job_id)
                if current and current.status in (
                    JobStatus.completed,
                    JobStatus.failed,
                ):
                    if current.status == JobStatus.completed:
                        yield {
                            "data": json.dumps({"type": "completed", "fraction": 1.0})
                        }
                    else:
                        yield {
                            "data": json.dumps(
                                {
                                    "type": "error",
                                    "message": current.error or "Job failed",
                                }
                            )
                        }
                    break

    return EventSourceResponse(event_generator())
