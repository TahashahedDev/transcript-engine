from __future__ import annotations

import asyncio
import logging as _logging
import os
import warnings
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# Must run before any CUDA allocation — configures the PyTorch caching allocator.
# Setting here (at import time, before any torch.cuda calls) guarantees the
# allocator is tuned before model warmup or the first pipeline job.
from transcript_engine.gpu.hardware import configure_cuda_allocator  # noqa: E402

configure_cuda_allocator()

# Suppress noisy third-party library warnings that are not actionable
warnings.filterwarnings("ignore", message="std\\(\\): degrees of freedom is <= 0", category=UserWarning)


class _SuppressLightningCheckpointFilter(_logging.Filter):
    def filter(self, record: _logging.LogRecord) -> bool:
        return "Lightning automatically upgraded your loaded checkpoint" not in record.getMessage()


_logging.getLogger("pytorch_lightning.utilities.migration.utils").addFilter(_SuppressLightningCheckpointFilter())

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from api.config import config  # noqa: E402

_log = _logging.getLogger(__name__)


# ── Model warmup (background, triggered by TE_WARM_MODELS=1) ─────────────────


def _warmup_parakeet() -> None:
    """Pre-load Parakeet into VRAM so the first job doesn't pay cold-start."""
    try:
        import os as _os

        from transcript_engine.gpu.hardware import (
            GPU_COMPUTE_LOCK,
            configure_cuda_for_inference,
            detect_gpu,
        )
        from transcript_engine.transcription.parakeet_engine import (
            ParakeetEngine,
            _get_frame_duration,
            _load_nemo_model,
        )

        model_id = _os.environ.get("TE_PARAKEET_MODEL", "nvidia/parakeet-tdt-0.6b-v2")
        with GPU_COMPUTE_LOCK:
            if ParakeetEngine._model is None:
                _log.info("[warmup] Loading Parakeet model: %s", model_id)
                ParakeetEngine._model = _load_nemo_model(model_id)
                ParakeetEngine._model_id = model_id
                ParakeetEngine._frame_duration = _get_frame_duration(ParakeetEngine._model)
                # Must mirror the model-load block in ParakeetEngine.transcribe() so that
                # TF32 and cuDNN benchmark are configured before the first job.
                # VRAM-based chunk sizing does NOT depend on this warmup step — every
                # transcribe() call re-detects free VRAM itself (see parakeet_engine.py).
                gpu_at_warmup = detect_gpu()
                if gpu_at_warmup is not None:
                    configure_cuda_for_inference(gpu_at_warmup)
                _log.info("[warmup] Parakeet loaded and ready.")
    except Exception as exc:
        _log.warning("[warmup] Parakeet warmup failed (non-fatal): %s", exc)


def _warmup_diarization() -> None:
    """Pre-load pyannote diarization pipeline into memory."""
    try:
        from api.model_cache import get_registry
        from transcript_engine.config.loader import load_settings
        from transcript_engine.gpu.hardware import GPU_COMPUTE_LOCK

        settings = load_settings()
        if not settings.hf_token:
            _log.info("[warmup] No HF token — skipping diarization warmup.")
            return
        registry = get_registry()
        _log.info("[warmup] Loading diarization model...")
        # Same lock PyannoteEngine.diarize() uses around this call — without
        # it, this warmup step's .to(cuda) transfer could race a concurrently
        # starting Parakeet warmup/job for VRAM during the startup window.
        with GPU_COMPUTE_LOCK:
            registry.get_diarization_pipeline(settings.pipeline.diarization)
        _log.info("[warmup] Diarization loaded and ready.")
    except Exception as exc:
        _log.warning("[warmup] Diarization warmup failed (non-fatal): %s", exc)


async def _warmup_models() -> None:
    """Background task: pre-load both models so first job skips cold-start."""
    _log.info("[warmup] Starting model pre-load (TE_WARM_MODELS=1)...")
    loop = asyncio.get_running_loop()

    asr_backend = os.environ.get("TE_ASR_BACKEND", "whisper").lower()
    if asr_backend == "parakeet":
        await loop.run_in_executor(None, _warmup_parakeet)

    await loop.run_in_executor(None, _warmup_diarization)
    _log.info("[warmup] Model pre-load complete.")


# ── App lifespan ──────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, Any]:
    from api.cleanup import cleanup_loop

    Path(config.temp_dir).mkdir(exist_ok=True)
    Path(config.output_dir).mkdir(exist_ok=True)

    # Warm-load models in the background if requested.
    # This pays the cold-start cost before the first user job arrives.
    if os.environ.get("TE_WARM_MODELS", "").lower() in ("1", "true", "yes"):
        asyncio.create_task(_warmup_models())

    # Background output-directory cleanup (runs every hour)
    cleanup_task = asyncio.create_task(cleanup_loop())
    try:
        yield
    finally:
        cleanup_task.cancel()
        # Drain the thread pool: cancel queued futures, let active threads finish.
        from api.pipeline_runner import _executor  # noqa: PLC0415

        _executor.shutdown(wait=False, cancel_futures=True)


# ── FastAPI app ───────────────────────────────────────────────────────────────


app = FastAPI(
    title="Transcript Engine API",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────────
# The browser's Origin is whatever address the user opened the UI on. A
# localhost-only allowlist therefore blocks every request as soon as the app is
# reached by IP or hostname — which is the normal case on a rented GPU box, and
# fails as a silent CORS rejection rather than a visible server error.
#
# Precedence:
#   1. TE_API_ALLOW_ALL_ORIGINS=1  — allow any origin (convenient; no cookies).
#   2. TE_API_CORS_ORIGINS         — explicit comma-separated allowlist. Use
#                                    this to lock down a shared/public host.
#   3. Default — any host, but only on the frontend port. Keeps the common
#      single-box deployment working regardless of IP/hostname, while still
#      rejecting arbitrary ports and origins.
_FRONTEND_PORT = os.environ.get("TE_FRONTEND_PORT", "9098")
_allow_all_origins = os.environ.get("TE_API_ALLOW_ALL_ORIGINS", "").lower() in ("1", "true", "yes")
_explicit_origins = [
    o.strip() for o in os.environ.get("TE_API_CORS_ORIGINS", "").split(",") if o.strip()
]

_cors_kwargs: dict[str, Any] = {
    "allow_methods": ["*"],
    "allow_headers": ["*"],
    # Credentials cannot be combined with a wildcard origin, and this API
    # authenticates with bearer tokens rather than cookies.
    "allow_credentials": bool(_explicit_origins),
}
if _allow_all_origins:
    _cors_kwargs["allow_origins"] = ["*"]
    _log.info("CORS: all origins allowed (TE_API_ALLOW_ALL_ORIGINS)")
elif _explicit_origins:
    _cors_kwargs["allow_origins"] = _explicit_origins
    _log.info("CORS: restricted to %s", ", ".join(_explicit_origins))
else:
    _cors_kwargs["allow_origin_regex"] = rf"^https?://[^/:]+:{_FRONTEND_PORT}$"
    _log.info("CORS: any host on port %s (set TE_API_CORS_ORIGINS to restrict)", _FRONTEND_PORT)

app.add_middleware(CORSMiddleware, **_cors_kwargs)

from api.routes import artifacts, diagnostics, jobs, progress, setup  # noqa: E402
from api.v2.routes import jobs as jobs_v2  # noqa: E402

app.include_router(jobs.router, prefix="/api")
app.include_router(artifacts.router, prefix="/api")
app.include_router(progress.router, prefix="/api")
app.include_router(setup.router, prefix="/api")
app.include_router(diagnostics.router, prefix="/api")
app.include_router(jobs_v2.router)


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/")
async def root() -> dict[str, str]:
    return {"name": "Transcript Engine API", "version": "1.0.0", "docs": "/docs"}


@app.get("/health")
async def health() -> dict[str, Any]:
    """
    Readiness probe for start.sh and external health checks.

    Returns GPU info, model load status, and disk space.
    Always returns HTTP 200 — callers check the 'ready' field.
    """
    result: dict[str, Any] = {
        "status": "ok",
        "cuda": False,
        "gpu": None,
        "vram_total_gb": None,
        "vram_free_gb": None,
        "gpu_compatible": None,
        "gpu_compatibility_error": None,
        "parakeet_loaded": False,
        "diarization_loaded": False,
        "disk_free_gb": None,
        "asr_backend": os.environ.get("TE_ASR_BACKEND", "whisper"),
        "ready": False,
    }

    try:
        from transcript_engine.gpu.hardware import check_gpu_compatibility, detect_gpu

        gpu_info = detect_gpu()
        result["cuda"] = gpu_info is not None
        if gpu_info is not None:
            result["gpu"] = gpu_info.name
            result["vram_total_gb"] = gpu_info.vram_total_gb
            # detect_gpu uses torch.cuda.mem_get_info, i.e. free memory on the
            # *device*. An earlier version subtracted memory_allocated(), which
            # counts only this process's tensors and so over-reported free VRAM
            # whenever another process or the CUDA context held memory.
            result["vram_free_gb"] = gpu_info.vram_free_gb
            # Surfaced here because /health is the first thing an operator (and
            # start.sh) checks: a GPU the installed PyTorch cannot run kernels
            # on should be visible now, not at the first failed job.
            compat_error = check_gpu_compatibility(gpu_info)
            result["gpu_compatible"] = compat_error is None
            result["gpu_compatibility_error"] = compat_error
    except Exception as exc:
        _log.warning("Health check: GPU query failed: %s", exc)

    try:
        from transcript_engine.transcription.parakeet_engine import ParakeetEngine
        result["parakeet_loaded"] = ParakeetEngine._model is not None
    except Exception:
        pass

    try:
        from api.model_cache import _registry
        result["diarization_loaded"] = (
            _registry is not None
            and any("diarize:" in k for k in _registry.loaded_models())
        )
    except Exception:
        pass

    try:
        import shutil
        usage = shutil.disk_usage(".")
        result["disk_free_gb"] = round(usage.free / 1e9, 1)
    except Exception:
        pass

    result["ready"] = True
    return result
