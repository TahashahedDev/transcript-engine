from __future__ import annotations

import os
import threading
import time
from typing import Any

from transcript_engine.config.settings import (
    DiarizationConfig,
    Settings,
    TranscriptionConfig,
)
from transcript_engine.diarization.compat import load_pretrained_pipeline
from transcript_engine.logging import get_logger

logger = get_logger(__name__)

# Free VRAM required before placing the diarization pipeline on the GPU.
# pyannote's weights are ~700 MB; the rest is activation headroom so a job
# already holding Parakeet's weights doesn't tip the card into OOM.
_PYANNOTE_VRAM_NEEDED_GB = 2.0


class ModelRegistry:
    """
    Centralized, thread-safe, lazy model loader.

    Models are loaded on first request and cached for the process lifetime.
    Constructed once and injected into every engine that needs it.
    Never use module-level singletons — inject this instead.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._cache: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._configure_hf_cache()
        self._register_torch_safe_globals()

    def get_whisper_model(self, config: TranscriptionConfig) -> Any:
        # CTranslate2 (faster-whisper) does not support MPS — use CPU on Apple Silicon.
        device = self._resolve_ctranslate_device(config.device)
        if config.compute_type == "auto":
            compute_type = self._resolve_compute_type(device)
        else:
            compute_type = config.compute_type
        key = f"whisper:{config.model_id}:{device}:{compute_type}"
        return self._get_or_load(
            key, lambda: self._load_whisper(config, device, compute_type)
        )

    def get_alignment_model(
        self, language: str, config: TranscriptionConfig
    ) -> tuple[Any, Any]:
        # Force CPU for alignment — wav2vec2 on MPS causes Metal queue serialization
        # with concurrent pyannote diarization (both threads compete for the same MPS device).
        # On M1 8 GB the contention turns a 5-min alignment into 2+ hours.
        device = "cpu"
        key = f"align:{language}:{device}"
        result: tuple[Any, Any] = self._get_or_load(
            key, lambda: self._load_alignment(language, device)
        )
        return result

    def get_diarization_pipeline(self, config: DiarizationConfig) -> Any:
        key = f"diarize:{config.model_id}"
        return self._get_or_load(key, lambda: self._load_diarization(config))

    def release(self, key: str) -> None:
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                logger.debug(f"Released model: {key}")

    def release_all(self) -> None:
        with self._lock:
            keys = list(self._cache.keys())
            self._cache.clear()
        logger.debug(f"Released {len(keys)} model(s) from registry")

    def loaded_models(self) -> list[str]:
        with self._lock:
            return list(self._cache.keys())

    def _get_or_load(self, key: str, loader: Any) -> Any:
        # Fast path: cached lookups don't need the lock.
        # CPython's GIL makes dict.__contains__ and dict.__getitem__ atomic,
        # so this read is safe without holding _lock.
        if key in self._cache:
            return self._cache[key]
        with self._lock:
            # Re-check: another thread may have loaded between the two checks.
            if key not in self._cache:
                t0 = time.monotonic()
                logger.debug(f"Loading: {key}")
                self._cache[key] = loader()
                logger.debug(f"Loaded in {time.monotonic() - t0:.1f}s: {key}")
            return self._cache[key]

    def _load_whisper(
        self,
        config: TranscriptionConfig,
        device: str,
        compute_type: str,
    ) -> Any:
        import whisperx  # noqa: PLC0415

        cache_dir = str(self._settings.models_cache_dir / "whisper")
        os.makedirs(cache_dir, exist_ok=True)

        logger.info(
            f"Loading Whisper model '{config.model_id}' "
            f"on {device} ({compute_type})"
        )

        asr_options: dict[str, Any] = {
            "beam_size": config.beam_size,
            "best_of": config.best_of,
            "temperatures": config.temperatures,
        }

        logger.info(
            "Whisper ASR options: beam_size=%d, best_of=%d, temperatures=%s, "
            "cpu_threads=%d, compute_type=%s",
            config.beam_size,
            config.best_of,
            config.temperatures,
            config.cpu_threads,
            compute_type,
        )

        return whisperx.load_model(
            config.model_id,
            device,
            compute_type=compute_type,
            language=config.language,
            asr_options=asr_options,
            download_root=cache_dir,
            threads=config.cpu_threads,
        )

    def _load_alignment(self, language: str, device: str) -> tuple[Any, Any]:
        import whisperx  # noqa: PLC0415

        cache_dir = str(self._settings.models_cache_dir / "alignment")
        os.makedirs(cache_dir, exist_ok=True)

        logger.info(f"Loading alignment model for language: {language}")
        result: tuple[Any, Any] = whisperx.load_align_model(
            language_code=language,
            device=device,
            model_dir=cache_dir,
        )
        return result

    @staticmethod
    def _register_torch_safe_globals() -> None:
        """
        PyTorch 2.6 changed torch.load to default weights_only=True, which
        breaks whisperx VAD and pyannote model loading (they serialize omegaconf
        and typing objects into their checkpoints). Since we only load trusted
        model files downloaded from HuggingFace, we patch torch.load to restore
        the pre-2.6 default of weights_only=False for callers that don't specify.
        """
        try:
            import functools

            import torch

            if getattr(torch.load, "_te_patched", False):
                return

            _orig = torch.load

            @functools.wraps(_orig)
            def _patched(*args: Any, **kwargs: Any) -> Any:
                # lightning_fabric passes weights_only=None explicitly, so setdefault
                # won't help — we must replace None (and missing) with False.
                # Leave explicit weights_only=True alone.
                if kwargs.get("weights_only") is not True:
                    kwargs["weights_only"] = False
                return _orig(*args, **kwargs)

            _patched._te_patched = True  # type: ignore[attr-defined]
            torch.load = _patched  # noqa: B010
        except (ImportError, AttributeError):
            pass

    def _load_diarization(self, config: DiarizationConfig) -> Any:
        import torch  # noqa: PLC0415
        from pyannote.audio import Pipeline  # noqa: PLC0415

        token = self._settings.hf_token
        if not token:
            token = os.environ.get("HF_TOKEN") or os.environ.get(
                "HUGGING_FACE_HUB_TOKEN"
            )

        # pyannote.audio 3.x calls hf_hub_download(use_auth_token=...) internally.
        # huggingface_hub >= 1.0 removed use_auth_token (renamed to token).
        # Fix: set HF_TOKEN env var (auto-read by hf_hub), then patch hf_hub_download
        # to accept and forward the deprecated kwarg so pyannote doesn't crash.
        if token:
            os.environ["HF_TOKEN"] = token

        self._patch_hf_hub_download()

        cache_dir = str(self._settings.models_cache_dir / "pyannote")
        logger.info(f"Loading diarization model: {config.model_id}")
        pipeline = load_pretrained_pipeline(
            Pipeline, config.model_id, token, cache_dir=cache_dir
        )

        # On CUDA machines, run pyannote on GPU when there is real headroom.
        #
        # Why GPU at all: pyannote on CPU is the pipeline's dominant cost for
        # long audio (minutes), versus seconds on GPU. Worth having when it fits.
        #
        # The decision uses *free* VRAM, not total. Total is misleading: by the
        # time diarization loads, Parakeet's ~2.5 GB of weights are typically
        # already resident, and on a shared or multi-tenant box other processes
        # may hold memory too. An 8 GB card with 1 GB actually free must not be
        # treated the same as an idle 8 GB card. Falling back to CPU is slower
        # but always correct; OOMing here would fail the job.
        #
        # Concurrency is handled separately: when this lands on CUDA it is
        # flagged so PyannoteEngine serialises its compute against Parakeet via
        # gpu.hardware.GPU_COMPUTE_LOCK, so the two never allocate at once.
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            try:
                free_bytes, _ = torch.cuda.mem_get_info(0)
                free_gb = free_bytes / 1e9
            except Exception:
                # Older torch without mem_get_info: fall back to the total-VRAM
                # heuristic rather than refusing to use the GPU at all.
                free_gb = vram_gb
            if free_gb >= _PYANNOTE_VRAM_NEEDED_GB:
                try:
                    pipeline.to(torch.device("cuda"))
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower():
                        raise
                    # Weight transfer itself didn't fit — e.g. Parakeet already
                    # holds most of VRAM by the time this loads. No smaller size
                    # to retry with; CPU is the only real fallback, same as the
                    # inference-time OOM fallback in PyannoteEngine.diarize().
                    logger.warning(
                        "Diarization CUDA OOM while loading weights (%.0f GB "
                        "total VRAM) — falling back to CPU: %s", vram_gb, exc,
                    )
                    torch.cuda.empty_cache()
                    pipeline.to(torch.device("cpu"))
                    pipeline._te_on_cuda = False
                else:
                    # Consumed by PyannoteEngine.diarize(): tells it to serialize
                    # against Parakeet transcription via gpu.hardware.GPU_COMPUTE_LOCK
                    # instead of running concurrently. On an 8 GB card, Parakeet's
                    # resident weights (~2.5 GB) plus pyannote's own weights and
                    # activations (~1 GB) fit fine sequentially, but NOT reliably at
                    # the same instant — hence serialize, don't just co-place.
                    pipeline._te_on_cuda = True
                    logger.info(
                        "Diarization pipeline on CUDA (%.1f GB free of %.0f GB) — "
                        "serialised with Parakeet via the shared GPU compute lock",
                        free_gb, vram_gb,
                    )
            else:
                pipeline.to(torch.device("cpu"))
                pipeline._te_on_cuda = False
                logger.info(
                    "Diarization pipeline on CPU (only %.1f GB free of %.0f GB — "
                    "needs ~%.0f GB). Slower, but avoids an OOM mid-job.",
                    free_gb, vram_gb, _PYANNOTE_VRAM_NEEDED_GB,
                )
        else:
            pipeline._te_on_cuda = False

        return pipeline

    @staticmethod
    def _patch_hf_hub_download() -> None:
        """
        Bridge pyannote.audio (uses use_auth_token) with huggingface_hub >= 1.0
        (removed use_auth_token, renamed to token).  Idempotent.

        pyannote.audio 3.x imports hf_hub_download at the module level in three
        submodules.  We must patch all three local references after they have
        been imported, otherwise the `from huggingface_hub import hf_hub_download`
        copies inside pyannote still point to the unpatched function.
        """
        try:
            import huggingface_hub  # noqa: PLC0415

            if getattr(huggingface_hub.hf_hub_download, "_te_patched", False):
                return  # already patched

            _orig = huggingface_hub.hf_hub_download

            def _patched(*args: Any, **kwargs: Any) -> Any:
                if "use_auth_token" in kwargs:
                    uak = kwargs.pop("use_auth_token")
                    if uak and "token" not in kwargs:
                        kwargs["token"] = uak
                return _orig(*args, **kwargs)

            _patched._te_patched = True  # type: ignore[attr-defined]
            huggingface_hub.hf_hub_download = _patched  # noqa: B010

            # Patch each pyannote submodule that imported hf_hub_download directly
            _pyannote_modules = [
                "pyannote.audio.core.pipeline",
                "pyannote.audio.core.model",
                "pyannote.audio.pipelines.speaker_verification",
            ]
            for _mod_name in _pyannote_modules:
                try:
                    import importlib  # noqa: PLC0415

                    _mod = importlib.import_module(_mod_name)
                    if hasattr(_mod, "hf_hub_download"):
                        setattr(_mod, "hf_hub_download", _patched)  # noqa: B010
                except (ImportError, AttributeError):
                    pass

        except (ImportError, AttributeError):
            pass

    def _configure_hf_cache(self) -> None:
        cache_dir = str(self._settings.models_cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        os.environ.setdefault("HF_HOME", cache_dir)
        os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir)
        os.environ.setdefault("HF_HUB_CACHE", cache_dir)

    @staticmethod
    def _resolve_torch_device(device: str) -> str:
        """Resolve device for PyTorch-native models (alignment, diarization). Supports MPS."""
        if device != "auto":
            return device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    @staticmethod
    def _resolve_ctranslate_device(device: str) -> str:
        """
        Resolve device for CTranslate2 / faster-whisper.
        CTranslate2 does NOT support MPS — Apple Silicon falls back to CPU.
        CUDA is supported and preferred when available.
        """
        if device not in ("auto", "mps"):
            return device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    @staticmethod
    def _resolve_compute_type(device: str) -> str:
        """
        Auto-select the best compute type for the given CTranslate2 device.
        CPU uses int8 for 4x faster inference with minimal accuracy loss.
        CUDA uses float16 for maximum throughput.
        """
        return {
            "cuda": "float16",
            "cpu": "int8",
        }.get(device, "float32")
