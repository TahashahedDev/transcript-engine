from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel

from api.config import config
from transcript_engine.config.paths import project_root

router = APIRouter()


# ─── Response models ────────────────────────────────────────────────────────


class ModelInfo(BaseModel):
    name: str
    cached: bool
    size_mb: float | None = None
    path: str | None = None


class HFTokenStatus(BaseModel):
    present: bool
    valid: bool | None = None
    diarization_access: bool | None = None
    segmentation_access: bool | None = None
    error: str | None = None


class GpuStatus(BaseModel):
    available: bool
    name: str | None = None
    cuda_version: str | None = None
    vram_total_gb: float | None = None
    vram_free_gb: float | None = None
    compute_capability: str | None = None
    # None when the installed PyTorch build can run kernels on this GPU;
    # otherwise an actionable explanation of the arch mismatch.
    compatibility_error: str | None = None


class SetupCheckResponse(BaseModel):
    python_version: str
    ffmpeg: bool
    ffmpeg_version: str | None
    hf_token: HFTokenStatus
    profiles: list[str]
    output_dir_writable: bool
    whisper_models: list[ModelInfo]
    alignment_models: list[ModelInfo]
    ready: bool
    asr_backend: str = "whisper"
    parakeet_model: str | None = None
    # Surfaced so the upload UI enforces the same limit the server does,
    # rather than hardcoding its own number that can drift out of sync.
    max_upload_mb: int = 2000
    gpu: GpuStatus = GpuStatus(available=False)


class TokenSaveResponse(BaseModel):
    saved: bool
    valid: bool | None = None
    error: str | None = None


# ─── Helpers ────────────────────────────────────────────────────────────────


def _ffmpeg_version() -> str | None:
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, timeout=5
        )
        first_line = result.stdout.splitlines()[0] if result.stdout else ""
        parts = first_line.split()
        if len(parts) >= 3:
            return parts[2]
    except Exception:
        pass
    return None


def _check_hf_token(token: str) -> HFTokenStatus:
    if not token:
        return HFTokenStatus(present=False)

    try:
        from huggingface_hub import HfApi  # noqa: PLC0415
    except ImportError:
        return HFTokenStatus(
            present=True, valid=None, error="huggingface_hub not installed"
        )

    api = HfApi(token=token)

    try:
        api.whoami()
    except Exception as e:
        msg = str(e).lower()
        if "401" in msg or "invalid" in msg or "unauthorized" in msg:
            return HFTokenStatus(
                present=True, valid=False, error="Token is invalid or expired"
            )
        return HFTokenStatus(
            present=True, valid=None, error=f"Could not reach HuggingFace: {e}"
        )

    diar_ok: bool | None = None
    seg_ok: bool | None = None
    error_msg: str | None = None

    try:
        api.model_info("pyannote/speaker-diarization-3.1", token=token)
        diar_ok = True
    except Exception as e:
        diar_ok = False
        msg = str(e)
        if "403" in msg or "gated" in msg.lower() or "license" in msg.lower():
            error_msg = (
                "Accept the pyannote/speaker-diarization-3.1 license at "
                "huggingface.co/pyannote/speaker-diarization-3.1"
            )
        elif "404" in msg:
            error_msg = "pyannote/speaker-diarization-3.1 not found"
        else:
            error_msg = f"Cannot access speaker-diarization-3.1: {msg}"

    try:
        api.model_info("pyannote/segmentation-3.0", token=token)
        seg_ok = True
    except Exception as e:
        seg_ok = False
        if error_msg is None:
            msg = str(e)
            if "403" in msg or "gated" in msg.lower():
                error_msg = (
                    "Accept the pyannote/segmentation-3.0 license at "
                    "huggingface.co/pyannote/segmentation-3.0"
                )

    return HFTokenStatus(
        present=True,
        valid=True,
        diarization_access=diar_ok,
        segmentation_access=seg_ok,
        error=error_msg,
    )


def _scan_whisper_models(cache_dir: Path) -> list[ModelInfo]:
    known = {
        # The model the pipeline actually uses (mobiuslabsgmbh CTranslate2 int8).
        # Must be listed first so the `ready` flag reflects the real configured model.
        "models--mobiuslabsgmbh--faster-whisper-large-v3-turbo": "large-v3-turbo",
        "models--Systran--faster-whisper-large-v3": "large-v3",
        "models--Systran--faster-whisper-large-v2": "large-v2",
        "models--Systran--faster-whisper-medium": "medium",
        "models--Systran--faster-whisper-small": "small",
        "models--Systran--faster-whisper-base": "base",
        "models--Systran--faster-whisper-tiny": "tiny",
    }
    # The registry stores Whisper models under cache_dir/whisper/
    search_dirs = [cache_dir, cache_dir / "whisper"]

    models = []
    for dir_name, display_name in known.items():
        for base in search_dirs:
            model_dir = base / dir_name
            if model_dir.exists():
                size_bytes = sum(
                    f.stat().st_size for f in model_dir.rglob("*") if f.is_file()
                )
                models.append(
                    ModelInfo(
                        name=display_name,
                        cached=True,
                        size_mb=round(size_bytes / 1_048_576, 1),
                    )
                )
                break  # found in one location, don't double-count

    if not models:
        models.append(ModelInfo(name="large-v3-turbo", cached=False))
    return models


def _scan_alignment_models(cache_dir: Path) -> list[ModelInfo]:
    align_dir = cache_dir / "alignment"
    if not align_dir.exists():
        return []
    return [
        ModelInfo(
            name=f.stem, cached=True, size_mb=round(f.stat().st_size / 1_048_576, 1)
        )
        for f in align_dir.iterdir()
        if f.is_file() and f.suffix == ".pth"
    ]


def _check_output_writable() -> bool:
    out = Path(config.output_dir)
    try:
        out.mkdir(parents=True, exist_ok=True)
        probe = out / ".write_probe"
        probe.touch()
        probe.unlink()
        return True
    except Exception:
        return False


def _read_env_token() -> str:
    # os.environ is populated by _update_env_file() at runtime but is empty
    # after a restart — pydantic-settings reads .env into model fields, not
    # os.environ.  Check both so the setup check works after a cold restart.
    token = os.environ.get("TE_HF_TOKEN", "")
    if not token:
        try:
            from transcript_engine.config.loader import load_settings  # noqa: PLC0415

            token = load_settings().hf_token or ""
        except Exception:
            pass
    return token


def _update_env_file(token: str) -> None:
    # Must be the same file Settings/APIConfig read (project root), not a
    # CWD-relative one. Writing to whatever directory the process happened to
    # start in would save the token somewhere the loader never looks, so it
    # would silently vanish on the next restart.
    env_path = project_root() / ".env"
    lines: list[str] = []
    found = False

    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("TE_HF_TOKEN="):
                lines.append(f"TE_HF_TOKEN={token}")
                found = True
            else:
                lines.append(line)

    if not found:
        lines.append(f"TE_HF_TOKEN={token}")

    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.environ["TE_HF_TOKEN"] = token
    # _load_diarization() reads HF_TOKEN (not TE_HF_TOKEN) as its env-var fallback.
    # Set both so a token saved on a live server reaches an already-created registry.
    os.environ["HF_TOKEN"] = token


# ─── Routes ─────────────────────────────────────────────────────────────────


def _detect_gpu_status() -> GpuStatus:
    """Query CUDA device info without allocating memory.  Returns safe defaults on failure."""
    try:
        from transcript_engine.gpu.hardware import (  # noqa: PLC0415
            check_gpu_compatibility,
            detect_gpu,
        )

        info = detect_gpu()
        if info is None:
            return GpuStatus(available=False)
        return GpuStatus(
            available=True,
            name=info.name,
            cuda_version=info.cuda_version,
            vram_total_gb=info.vram_total_gb,
            vram_free_gb=info.vram_free_gb,
            compute_capability=f"{info.compute_capability[0]}.{info.compute_capability[1]}",
            # Surfaced up front so an unusable GPU/PyTorch pairing is visible on
            # the setup screen rather than only when a job fails mid-run.
            compatibility_error=check_gpu_compatibility(info),
        )
    except Exception:
        return GpuStatus(available=False)


@router.get("/setup/check", response_model=SetupCheckResponse)
async def check_setup() -> SetupCheckResponse:
    from transcript_engine.profiles.loader import list_profiles

    cache_dir = Path.home() / ".cache" / "transcript_engine"
    token = _read_env_token()

    loop = asyncio.get_event_loop()
    try:
        hf_status = await asyncio.wait_for(
            loop.run_in_executor(None, _check_hf_token, token),
            timeout=10.0,
        )
    except TimeoutError:
        hf_status = HFTokenStatus(
            present=bool(token), valid=None, error="HuggingFace check timed out"
        )
    whisper_models = _scan_whisper_models(cache_dir)
    align_models = _scan_alignment_models(cache_dir)
    ffmpeg_ok = shutil.which("ffmpeg") is not None
    output_ok = _check_output_writable()
    whisper_cached = any(m.cached for m in whisper_models)

    asr_backend = os.environ.get("TE_ASR_BACKEND", "whisper").lower()
    parakeet_model: str | None = None
    if asr_backend == "parakeet":
        parakeet_model = os.environ.get(
            "TE_PARAKEET_MODEL", "nvidia/parakeet-tdt-0.6b-v2"
        )

    gpu_status = await loop.run_in_executor(None, _detect_gpu_status)

    # For Parakeet, Whisper models don't need to be cached — the system is ready
    # as long as ffmpeg and the output dir are available.
    if asr_backend == "parakeet":
        ready = ffmpeg_ok and output_ok
    else:
        ready = ffmpeg_ok and whisper_cached and output_ok

    return SetupCheckResponse(
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        ffmpeg=ffmpeg_ok,
        ffmpeg_version=_ffmpeg_version() if ffmpeg_ok else None,
        hf_token=hf_status,
        profiles=list_profiles(Path(config.profiles_dir)),
        output_dir_writable=output_ok,
        whisper_models=whisper_models,
        alignment_models=align_models,
        ready=ready,
        asr_backend=asr_backend,
        parakeet_model=parakeet_model,
        max_upload_mb=config.max_upload_mb,
        gpu=gpu_status,
    )


@router.post("/setup/token", response_model=TokenSaveResponse)
async def save_token(token: str = Body(..., embed=True)) -> TokenSaveResponse:
    """Save TE_HF_TOKEN to .env. The token value is never returned or logged."""
    token = token.strip()
    if not token:
        raise HTTPException(status_code=400, detail="Token must not be empty")
    if not token.startswith("hf_") or len(token) < 10:
        return TokenSaveResponse(
            saved=False, error="Token does not look valid (expected hf_...)"
        )

    try:
        _update_env_file(token)
    except Exception as e:
        return TokenSaveResponse(saved=False, error=f"Could not write .env file: {e}")

    status = _check_hf_token(token)
    if not status.valid:
        return TokenSaveResponse(saved=True, valid=False, error=status.error)
    if not status.diarization_access:
        return TokenSaveResponse(saved=True, valid=True, error=status.error)

    return TokenSaveResponse(saved=True, valid=True)


@router.post("/setup/download-models")
async def download_models() -> dict[str, str]:
    """Trigger a background Whisper model download."""
    import asyncio
    import contextlib

    async def _do_download() -> None:
        from api.model_cache import get_registry  # noqa: PLC0415

        with contextlib.suppress(Exception):
            get_registry()  # warms up Whisper

    asyncio.create_task(_do_download())
    return {
        "status": "downloading",
        "message": "Model download started. Refresh Setup to track progress.",
    }
