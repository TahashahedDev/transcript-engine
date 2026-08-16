from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from api.config import config
from api.job_manager import job_manager
from api.models import ArtifactsResponse

router = APIRouter()

_INLINE_SUFFIXES = {".md", ".json", ".txt"}


@router.get("/jobs/{job_id}/artifacts", response_model=ArtifactsResponse)
async def list_artifacts(job_id: str) -> ArtifactsResponse:
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    output_dir = Path(config.output_dir) / job_id
    if not output_dir.exists():
        return ArtifactsResponse(files=[], bundle=None)

    files = sorted(
        f.name
        for f in output_dir.iterdir()
        if f.is_file()
        and f.suffix in {".md", ".json", ".txt", ".srt", ".vtt", ".zip"}
        and f.name != "transcript_bundle.zip"
    )
    bundle = (
        "transcript_bundle.zip"
        if (output_dir / "transcript_bundle.zip").exists()
        else None
    )
    return ArtifactsResponse(files=files, bundle=bundle)


@router.get("/jobs/{job_id}/artifacts/{filename}")
async def get_artifact(job_id: str, filename: str) -> FileResponse:
    # Prevent path traversal
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    file_path = Path(config.output_dir) / job_id / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")

    suffix = file_path.suffix.lower()
    if suffix in _INLINE_SUFFIXES:
        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type="text/plain; charset=utf-8",
        )
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
