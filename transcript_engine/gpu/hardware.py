"""
GPU hardware detection and CUDA configuration for inference workloads.

Provides a single query (detect_gpu) and two idempotent configuration
helpers (configure_cuda_for_inference, configure_cuda_allocator) that
must be called early in the process before any CUDA memory is allocated.

Design principles:
  - Never allocates CUDA memory itself (pure introspection)
  - Idempotent — safe to call multiple times
  - Fails gracefully when CUDA is unavailable (returns None / no-ops)
  - No global mutable state (callers cache GpuInfo if needed)
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass

from transcript_engine.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Cross-engine GPU serialization
#
# Parakeet transcription and pyannote diarization can both run on the same
# CUDA device. The orchestrator launches them as concurrent threads (for
# wall-clock speed on multi-GPU or high-VRAM boxes), but on a single GPU with
# limited VRAM (e.g. an 8 GB RTX 3060 Ti) simultaneous allocation from both
# engines can exceed available memory even though each engine's chunk/model
# sizing is individually "safe" in isolation.
#
# GPU_COMPUTE_LOCK serializes actual GPU compute between engines: whichever
# engine acquires it first runs to completion before the other starts its
# GPU work. Model loading and CPU-side work (e.g. reading audio, splitting
# chunks) does not need to hold this lock. This is intentionally a single
# process-wide lock, not a scheduler — see Phase 9 guidance: keep it simple.
# ---------------------------------------------------------------------------
GPU_COMPUTE_LOCK: threading.Lock = threading.Lock()

# ---------------------------------------------------------------------------
# VRAM → safe chunk duration table
#
# Peak VRAM per chunk (empirical on RTX 3060 Ti, Parakeet-TDT-0.6B, FP32):
#   90 s  →  ~1.3 GB activations  (+ 2.5 GB weights = ~3.8 GB peak)
#  180 s  →  ~2.4 GB activations  (+ 2.5 GB weights = ~4.9 GB peak)
#  300 s  →  ~3.0 GB activations  (+ 2.5 GB weights = ~5.5 GB peak)
#  480 s  →  ~4.5 GB activations  (+ 2.5 GB weights = ~7.0 GB peak)
#  720 s  →  ~6.5 GB activations  (+ 2.5 GB weights = ~9.0 GB peak)
#  900 s  →  ~8.0 GB activations  (+ 2.5 GB weights = ~10.5 GB peak)
# 1320 s  → ~11.0 GB activations  (Parakeet's 24-min design limit)
#
# Conservative: we target peak ≤ 70 % of free VRAM to leave headroom for
# the CUDA allocator, diarization tensors, and OS overhead.
# ---------------------------------------------------------------------------
_VRAM_TO_CHUNK_SECONDS: list[tuple[float, float]] = [
    (4.0, 90.0),          # < 4 GB free  →  1.5-min chunks
    (6.0, 180.0),         # 4–6 GB free  →  3-min chunks
    (8.0, 300.0),         # 6–8 GB free  →  5-min chunks  (RTX 3060 Ti target)
    (12.0, 480.0),        # 8–12 GB free →  8-min chunks
    (16.0, 720.0),        # 12–16 GB     →  12-min chunks
    (24.0, 900.0),        # 16–24 GB     →  15-min chunks
    (float("inf"), 1320.0),  # 24+ GB    →  22-min chunks  (model design max)
]

# Default overlap at each chunk boundary.  Parakeet TDT is CTC-based so hard
# cuts are architecturally safe; overlap exists solely to avoid clipping a word
# that straddles the nominal boundary.  0.5 s covers the longest English word
# at normal speech rate.
DEFAULT_OVERLAP_SECONDS: float = 0.5

# CUDA caching allocator tuning.  Must be set before the first CUDA allocation.
#   max_split_size_mb:128   — cap fragmented block size; prevents a single large
#                             allocation from permanently fragmenting the arena
#   garbage_collection_threshold:0.8 — trigger GC when 80 % of reserved memory
#                                      is in use rather than waiting for OOM
_CUDA_ALLOC_CONF = "max_split_size_mb:128,garbage_collection_threshold:0.8"


@dataclass(frozen=True)
class GpuInfo:
    """Immutable snapshot of the detected CUDA device at detection time."""

    index: int
    name: str
    cuda_version: str
    vram_total_gb: float
    vram_free_gb: float
    compute_capability: tuple[int, int]

    @property
    def is_ampere_or_newer(self) -> bool:
        """True for CUDA compute capability ≥ 8.0 (RTX 30xx, A-series, etc.)."""
        return self.compute_capability >= (8, 0)

    @property
    def sm_version(self) -> int:
        """Compute capability as the integer PyTorch uses in its arch list (8.6 → 86)."""
        return self.compute_capability[0] * 10 + self.compute_capability[1]


def _parse_arch(entry: str) -> tuple[str, int] | None:
    """
    Parse one torch.cuda.get_arch_list() entry.

    Entries look like "sm_86" (native cubin) or "compute_86" (PTX, JIT-able
    onto newer devices). Suffixed variants such as "sm_90a" exist; the trailing
    letter denotes an architecture-specific feature set, and the numeric part
    is what matters for compatibility.
    """
    for prefix in ("sm_", "compute_"):
        if entry.startswith(prefix):
            digits = "".join(c for c in entry[len(prefix):] if c.isdigit())
            if digits:
                return prefix.rstrip("_"), int(digits)
    return None


def check_gpu_compatibility(gpu: GpuInfo) -> str | None:
    """
    Verify the installed PyTorch build can actually run kernels on this GPU.

    Returns None when compatible, otherwise a human-readable explanation.

    Why this exists: torch.cuda.is_available() and the device-property queries
    succeed on *any* CUDA device the driver recognises, including one newer
    than the installed PyTorch build. The failure only surfaces later, at the
    first kernel launch, as "CUDA error: no kernel image is available for
    execution on the device" — which reads like a bug in this application
    rather than a mismatched install. Rented-GPU hosts make this likely: a
    brand-new card (e.g. Blackwell / sm_120) paired with an older wheel.

    Compatible when either:
      - the device's exact SM version is in the build's cubin list, or
      - the build ships PTX (compute_XX) at or below the device version, which
        the driver JIT-compiles forward onto newer hardware.
    """
    try:
        import torch

        arch_list = list(torch.cuda.get_arch_list())
    except Exception as exc:
        logger.debug("[GPU] Could not read torch arch list: %s", exc)
        return None  # Can't determine — don't block on a guess.

    if not arch_list:
        return None

    cubin: list[int] = []
    ptx: list[int] = []
    for entry in arch_list:
        parsed = _parse_arch(entry)
        if parsed is None:
            continue
        kind, version = parsed
        (cubin if kind == "sm" else ptx).append(version)

    device_sm = gpu.sm_version
    if device_sm in cubin:
        return None
    if any(p <= device_sm for p in ptx):
        # Forward-compatible via PTX JIT: works, but the first launch pays a
        # one-off compile and performance may trail a native build.
        logger.warning(
            "[GPU] %s (sm_%d) has no native kernels in this PyTorch build; "
            "running via PTX JIT from compute_%d. Expect a slower first launch.",
            gpu.name, device_sm, max(p for p in ptx if p <= device_sm),
        )
        return None

    return (
        f"GPU architecture is not supported by the installed PyTorch build. "
        f"{gpu.name} reports compute capability {gpu.compute_capability[0]}."
        f"{gpu.compute_capability[1]} (sm_{device_sm}), but this PyTorch "
        f"(CUDA {gpu.cuda_version}) was built for: {', '.join(arch_list)}. "
        f"Install a PyTorch build that targets sm_{device_sm} — see "
        f"https://pytorch.org/get-started/locally/ — or use a supported GPU. "
        f"Without this, inference fails at the first kernel launch with "
        f"'no kernel image is available for execution on the device'."
    )


def detect_gpu() -> GpuInfo | None:
    """
    Query the primary CUDA device.

    Returns None when CUDA is unavailable or device query fails.
    Does not allocate any CUDA memory.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None

        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        free_bytes, total_bytes = torch.cuda.mem_get_info(idx)

        info = GpuInfo(
            index=idx,
            name=props.name,
            cuda_version=str(torch.version.cuda or "unknown"),
            vram_total_gb=round(total_bytes / 1e9, 2),
            vram_free_gb=round(free_bytes / 1e9, 2),
            compute_capability=(props.major, props.minor),
        )
        logger.info(
            "[GPU] %s | CUDA %s | %.1f GB total / %.1f GB free | CC %d.%d",
            info.name,
            info.cuda_version,
            info.vram_total_gb,
            info.vram_free_gb,
            info.compute_capability[0],
            info.compute_capability[1],
        )
        return info

    except Exception as exc:
        logger.warning("[GPU] Detection failed: %s", exc)
        return None


def optimal_chunk_seconds(vram_free_gb: float) -> float:
    """
    Map available VRAM to a safe Parakeet chunk duration.

    Conservative by design: targets peak VRAM ≤ 70 % of free VRAM to leave
    headroom for allocator fragmentation, concurrent diarization tensors, and OS.
    """
    for threshold, chunk_s in _VRAM_TO_CHUNK_SECONDS:
        if vram_free_gb <= threshold:
            return chunk_s
    return 1320.0  # unreachable given the inf sentinel, but makes mypy happy


def configure_cuda_for_inference(gpu: GpuInfo) -> None:
    """
    Apply one-time CUDA settings that improve throughput without accuracy loss.

    TF32 (Ampere+): uses tensor cores for matrix ops — numerically equivalent
    to FP32 for typical ASR models, no code changes needed in NeMo.

    cuDNN benchmark: profiles kernel selection once per input shape, then
    caches the fastest algorithm.  Effective because fixed chunk_seconds gives
    consistent mel spectrogram lengths across chunks.

    Safe to call multiple times — all ops are idempotent.
    """
    try:
        import torch

        if gpu.is_ampere_or_newer:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("[GPU] TF32 enabled for Ampere+ GPU: %s", gpu.name)

        torch.backends.cudnn.benchmark = True
        logger.info("[GPU] cuDNN benchmark mode enabled")

    except Exception as exc:
        logger.warning("[GPU] CUDA configuration failed (non-fatal): %s", exc)


def configure_cuda_allocator() -> None:
    """
    Configure the PyTorch CUDA caching allocator.

    MUST be called before the first CUDA memory allocation.  The right place
    is process startup (before any model loading), not model load time.

    No-op if the env var is already set by the operator (operator wins).
    """
    if not os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = _CUDA_ALLOC_CONF
        logger.info("[GPU] CUDA allocator config: %s", _CUDA_ALLOC_CONF)


def log_vram_stats(label: str) -> None:
    """Log the current CUDA memory state.  No-op when CUDA is unavailable."""
    try:
        import torch

        if not torch.cuda.is_available():
            return
        idx = torch.cuda.current_device()
        alloc_gb = torch.cuda.memory_allocated(idx) / 1e9
        reserved_gb = torch.cuda.memory_reserved(idx) / 1e9
        peak_gb = torch.cuda.max_memory_allocated(idx) / 1e9
        logger.info(
            "[VRAM][%s] allocated=%.2f GB  reserved=%.2f GB  peak=%.2f GB",
            label,
            alloc_gb,
            reserved_gb,
            peak_gb,
        )
    except Exception:
        pass
