#!/usr/bin/env python3
"""
Preflight check for Vast.ai GPU production run.

Run BEFORE starting the API server and submitting any job:
    python scripts/preflight_check.py

Exits 0 and prints READY FOR GPU TRANSCRIPTION if all checks pass.
Exits 1 and prints exactly how to fix each failure.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

# ── Helpers ───────────────────────────────────────────────────────────────────

PASS  = "\033[32m✓\033[0m"
FAIL  = "\033[31m✗\033[0m"
WARN  = "\033[33m!\033[0m"
RESET = "\033[0m"
BOLD  = "\033[1m"

_failures: list[tuple[str, str]] = []   # (label, fix_command)
_warnings: list[tuple[str, str]] = []   # (label, note)


def ok(label: str, detail: str = "") -> None:
    suffix = f"  ({detail})" if detail else ""
    print(f"  {PASS}  {label}{suffix}")


def fail(label: str, fix: str) -> None:
    print(f"  {FAIL}  {label}")
    print(f"       Fix: {fix}")
    _failures.append((label, fix))


def warn(label: str, note: str = "") -> None:
    suffix = f"  ({note})" if note else ""
    print(f"  {WARN}  {label}{suffix}")
    _warnings.append((label, note))


def section(title: str) -> None:
    print(f"\n{BOLD}{title}{RESET}")


def _run(cmd: list[str], timeout: int = 10) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout + r.stderr).strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return 1, ""


def _load_env() -> None:
    """Load .env if it exists so checks see the same vars the API sees."""
    env_path = Path(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


# ── Check functions ────────────────────────────────────────────────────────────


def check_python() -> None:
    section("Python")
    v = sys.version_info
    ver_str = f"{v.major}.{v.minor}.{v.micro}"
    if v >= (3, 12):
        ok("Python ≥ 3.12", ver_str)
    else:
        fail(f"Python ≥ 3.12 required (found {ver_str})",
             "apt-get install python3.12 && python3.12 -m venv .venv")


def check_nvidia() -> None:
    section("NVIDIA Driver")
    rc, out = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                     "--format=csv,noheader"])
    if rc == 0 and out:
        for line in out.splitlines()[:1]:
            ok("nvidia-smi", line.strip())
    else:
        fail("nvidia-smi not found or failed",
             "Rent a GPU instance with NVIDIA drivers pre-installed")


def check_cuda() -> None:
    section("PyTorch / CUDA")
    try:
        import torch
        ok("PyTorch installed", torch.__version__)

        if not torch.cuda.is_available():
            fail("CUDA not available (torch.cuda.is_available() = False)",
                 "pip install torch --index-url https://download.pytorch.org/whl/cu124")
            return

        dev_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        ok("CUDA available", f"{dev_name} — {vram_gb:.1f} GB VRAM")

        if vram_gb < 16:
            warn("VRAM < 16 GB",
                 "Parakeet + pyannote need ~5-6 GB peak; less headroom for large batches")
        else:
            ok("VRAM ≥ 16 GB", f"{vram_gb:.1f} GB")

        # Quick smoke test
        try:
            x = torch.ones(1, device="cuda")
            _ = x + x
            torch.cuda.synchronize()
            ok("CUDA kernel execution")
        except Exception as exc:
            fail(f"CUDA kernel test failed: {exc}",
                 "Reinstall CUDA-compatible PyTorch matching your driver version")

    except ImportError:
        fail("PyTorch not installed",
             "pip install torch --index-url https://download.pytorch.org/whl/cu124")


def check_nemo() -> None:
    section("NeMo / Parakeet")
    try:
        import nemo
        ok("nemo_toolkit installed", getattr(nemo, "__version__", "unknown"))
    except ImportError:
        fail("nemo_toolkit not installed",
             'pip install ".[nvidia]"  or  pip install nemo_toolkit[asr]')
        return

    try:
        from nemo.collections.asr.models import EncDecRNNTBPEModel  # noqa: F401
        ok("EncDecRNNTBPEModel importable")
    except ImportError as exc:
        fail(f"EncDecRNNTBPEModel import failed: {exc}",
             'pip install ".[nvidia]"')
        return

    # Check if model is cached locally (avoids 5-10 min download on first job)
    model_id = os.environ.get("TE_PARAKEET_MODEL", "nvidia/parakeet-tdt-0.6b-v2")
    org, name = model_id.split("/", 1)
    cache_slug = f"models--{org}--{name}"

    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    te_cache = os.environ.get("TE_MODELS_CACHE_DIR", str(Path.home() / ".cache" / "transcript_engine"))
    nemo_cache = Path.home() / ".cache" / "torch" / "NeMo"

    found_in: str | None = None
    for search_root in [te_cache, hf_home, str(Path.home() / ".cache" / "huggingface")]:
        candidate = Path(search_root) / "hub" / cache_slug
        if candidate.exists() and any(candidate.iterdir()):
            found_in = str(candidate)
            break
    # NeMo native cache
    if found_in is None and nemo_cache.exists():
        for p in nemo_cache.rglob("parakeet*"):
            if p.is_dir():
                found_in = str(p)
                break

    if found_in:
        ok("Parakeet model cached", found_in)
    else:
        warn("Parakeet model not found in cache",
             "First job will download ~2.5 GB — run: python scripts/download_models.py")


def check_pyannote() -> None:
    section("pyannote / Diarization")
    try:
        from pyannote.audio import Pipeline  # noqa: F401
        ok("pyannote.audio installed")
    except ImportError:
        fail("pyannote.audio not installed",
             'pip install -e "."  (included in base dependencies)')
        return

    # Check model cache
    model_id = "pyannote/speaker-diarization-3.1"
    org, name = model_id.split("/", 1)
    cache_slug = f"models--{org}--{name}"

    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    te_cache = os.environ.get("TE_MODELS_CACHE_DIR",
                               str(Path.home() / ".cache" / "transcript_engine"))

    found_in: str | None = None
    for search_root in [te_cache, hf_home, str(Path.home() / ".cache" / "huggingface")]:
        candidate = Path(search_root) / "hub" / cache_slug
        if candidate.exists() and any(candidate.iterdir()):
            found_in = str(candidate)
            break

    if found_in:
        ok("pyannote/speaker-diarization-3.1 cached", found_in)
    else:
        warn("pyannote model not found in cache",
             "First job will download — run: python scripts/download_models.py")


def check_hf_token() -> None:
    section("HuggingFace Token")
    token = os.environ.get("TE_HF_TOKEN") or os.environ.get("HF_TOKEN", "")
    if not token or token in ("hf_your_token_here", ""):
        fail("TE_HF_TOKEN not set",
             "Edit .env and set TE_HF_TOKEN=hf_... (get one at huggingface.co/settings/tokens)")
        return
    ok("TE_HF_TOKEN is set", f"{token[:8]}...")

    # Validate token
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        user = api.whoami()
        ok("HF token valid", f"logged in as {user.get('name', '?')}")
    except Exception as exc:
        msg = str(exc)
        if "401" in msg or "invalid" in msg.lower():
            fail("HF token is invalid or expired",
                 "Generate a new token at huggingface.co/settings/tokens and update .env")
        else:
            warn(f"Could not validate HF token: {msg}",
                 "Check internet connection; token may still be valid")


def check_ffmpeg() -> None:
    section("ffmpeg")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        fail("ffmpeg not found on PATH",
             "apt-get install -y ffmpeg")
        return
    rc, out = _run(["ffmpeg", "-version"])
    ver = out.splitlines()[0] if out else "unknown"
    ok("ffmpeg installed", ver[:60])


def check_env_vars() -> None:
    section("Environment Variables")
    asr_backend = os.environ.get("TE_ASR_BACKEND", "")
    if asr_backend.lower() == "parakeet":
        ok("TE_ASR_BACKEND=parakeet")
    elif asr_backend == "":
        fail("TE_ASR_BACKEND not set",
             "Add  TE_ASR_BACKEND=parakeet  to .env")
    else:
        warn(f"TE_ASR_BACKEND={asr_backend!r}",
             "Expected 'parakeet' for GPU deployment")

    allow_origins = os.environ.get("TE_API_ALLOW_ALL_ORIGINS", "")
    if allow_origins in ("1", "true", "yes"):
        ok("TE_API_ALLOW_ALL_ORIGINS=1 (remote access enabled)")
    else:
        warn("TE_API_ALLOW_ALL_ORIGINS not set",
             "Add  TE_API_ALLOW_ALL_ORIGINS=1  to .env to allow browser access")

    api_host = os.environ.get("TE_API_HOST", "")
    if api_host in ("", "0.0.0.0"):
        ok("TE_API_HOST=0.0.0.0 (default)")
    else:
        warn(f"TE_API_HOST={api_host!r}", "Expected 0.0.0.0 for remote access on Vast.ai")

    model_override = os.environ.get("TE_PARAKEET_MODEL", "")
    if model_override:
        ok("TE_PARAKEET_MODEL", model_override)
    else:
        ok("TE_PARAKEET_MODEL (default)", "nvidia/parakeet-tdt-0.6b-v2")


def check_dirs() -> None:
    section("Directories & Disk")
    api_temp = os.environ.get("TE_API_TEMP_DIR", "temp")
    api_out  = os.environ.get("TE_API_OUTPUT_DIR", "outputs")

    for label, path_str in [("temp dir", api_temp), ("output dir", api_out)]:
        p = Path(path_str)
        try:
            p.mkdir(parents=True, exist_ok=True)
            test_file = p / ".preflight_write_test"
            test_file.write_text("ok")
            test_file.unlink()
            ok(f"{label} writable", str(p.resolve()))
        except Exception as exc:
            fail(f"{label} not writable: {exc}",
                 f"chmod 777 {path_str}  or fix permissions")

    # Disk space: need at least 10 GB for a 2 GB upload + processing + models
    try:
        import shutil as _shutil
        usage = _shutil.disk_usage(".")
        free_gb = usage.free / 1e9
        if free_gb >= 10:
            ok("Disk space", f"{free_gb:.1f} GB free")
        elif free_gb >= 5:
            warn(f"Low disk space ({free_gb:.1f} GB free)",
                 "Recommended ≥ 10 GB; a 90-min MP4 + WAV + outputs can use 3-4 GB")
        else:
            fail(f"Insufficient disk space ({free_gb:.1f} GB free)",
                 "Free at least 10 GB before processing large recordings")
    except Exception as exc:
        warn(f"Could not check disk space: {exc}")


def check_transcript_engine() -> None:
    section("Transcript Engine Package")
    try:
        import transcript_engine  # noqa: F401
        ok("transcript_engine importable")
    except ImportError as exc:
        fail(f"transcript_engine not installed: {exc}",
             'pip install -e ".[nvidia]"')
        return

    try:
        from transcript_engine.transcription.parakeet_engine import ParakeetEngine  # noqa: F401
        ok("ParakeetEngine importable")
    except ImportError as exc:
        fail(f"ParakeetEngine import failed: {exc}",
             'pip install -e ".[nvidia]"')

    try:
        from transcript_engine.config.settings import Settings
        s = Settings()
        ok("Settings loadable", f"cache_dir={s.models_cache_dir}")
    except Exception as exc:
        fail(f"Settings failed to load: {exc}",
             "Check .env for syntax errors")


def check_api_deps() -> None:
    section("API Dependencies")
    for pkg, install in [
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn[standard]"),
        ("multipart", "python-multipart"),
        ("sse_starlette", "sse-starlette"),
    ]:
        try:
            __import__(pkg)
            ok(pkg)
        except ImportError:
            fail(f"{pkg} not installed",
                 f'pip install "{install}"  or  pip install -e "."')


def check_internet(required: bool = False) -> None:
    section("Internet Connectivity")
    rc, _ = _run(["curl", "-s", "--max-time", "5", "-o", "/dev/null",
                   "-w", "%{http_code}", "https://huggingface.co"], timeout=10)
    if rc == 0:
        ok("huggingface.co reachable")
    elif required:
        fail("Cannot reach huggingface.co",
             "Check network/firewall — required for first-time model download")
    else:
        warn("Cannot reach huggingface.co (offline or firewall)",
             "OK if models are already cached; run download_models.py while online")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    print(f"\n{BOLD}Transcript Engine — Production Preflight Check{RESET}")
    print("=" * 52)

    _load_env()

    check_python()
    check_nvidia()
    check_cuda()
    check_nemo()
    check_pyannote()
    check_hf_token()
    check_ffmpeg()
    check_env_vars()
    check_dirs()
    check_transcript_engine()
    check_api_deps()
    check_internet()

    print("\n" + "=" * 52)
    if _failures:
        print(f"\n{BOLD}\033[31mFAILED — {len(_failures)} blocker(s){RESET}")
        for label, fix in _failures:
            print(f"\n  Problem: {label}")
            print(f"  Fix:     {fix}")
        if _warnings:
            print(f"\n{WARN}  {len(_warnings)} warning(s) — see above")
        print()
        return 1

    if _warnings:
        print(f"\n{BOLD}\033[33mPASSED with {len(_warnings)} warning(s){RESET}")
        for label, note in _warnings:
            print(f"  {WARN}  {label}")
            if note:
                print(f"       {note}")
        print()

    print(f"\n{BOLD}\033[32mREADY FOR GPU TRANSCRIPTION{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
