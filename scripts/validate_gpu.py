#!/usr/bin/env python3
"""
Validate that a Vast.ai GPU instance is ready to run Transcript Engine.

Run after setup_gpu.sh completes:
    python scripts/validate_gpu.py

Exits 0 if all checks pass, 1 if any critical check fails.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
WARN = "\033[33m!\033[0m"

errors: list[str] = []
warnings: list[str] = []


def check(label: str, ok: bool, detail: str = "", critical: bool = True) -> None:
    symbol = PASS if ok else (FAIL if critical else WARN)
    line = f"  {symbol}  {label}"
    if detail:
        line += f" — {detail}"
    print(line)
    if not ok:
        if critical:
            errors.append(label)
        else:
            warnings.append(label)


# ── Python ───────────────────────────────────────────────────────────────────
print("\nPython")
v = sys.version_info
check("Python ≥ 3.12", v >= (3, 12), f"{v.major}.{v.minor}.{v.micro}")

# ── CUDA ─────────────────────────────────────────────────────────────────────
print("\nCUDA")
try:
    import torch
    cuda_ok = torch.cuda.is_available()
    if cuda_ok:
        dev = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        check("CUDA available", True, f"{dev} — {vram:.1f} GB VRAM")
        check("VRAM ≥ 16 GB", vram >= 16, f"{vram:.1f} GB", critical=False)
    else:
        check("CUDA available", False, "torch.cuda.is_available() returned False")
    check("PyTorch", True, torch.__version__)
except ImportError:
    check("PyTorch installed", False, "pip install torch --index-url https://download.pytorch.org/whl/cu124")

# ── ffmpeg ───────────────────────────────────────────────────────────────────
print("\nffmpeg")
ffmpeg_path = shutil.which("ffmpeg")
check("ffmpeg on PATH", ffmpeg_path is not None, ffmpeg_path or "")
if ffmpeg_path:
    result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
    ver = result.stdout.splitlines()[0] if result.stdout else "unknown"
    check("ffmpeg runs", result.returncode == 0, ver)

# ── NeMo / Parakeet ──────────────────────────────────────────────────────────
print("\nNeMo / Parakeet")
try:
    import nemo
    check("nemo_toolkit installed", True, nemo.__version__)
    try:
        from nemo.collections.asr.models import EncDecRNNTBPEModel  # noqa: F401
        check("EncDecRNNTBPEModel importable", True)
    except ImportError as e:
        check("EncDecRNNTBPEModel importable", False, str(e))
except ImportError:
    check("nemo_toolkit installed", False, "pip install nemo_toolkit[asr]")

# ── Pyannote / diarization ───────────────────────────────────────────────────
print("\nPyannote / diarization")
try:
    from pyannote.audio import Pipeline  # noqa: F401
    check("pyannote.audio importable", True)
except ImportError:
    check("pyannote.audio importable", False, "included in base install")

# ── API deps ─────────────────────────────────────────────────────────────────
print("\nAPI dependencies")
for pkg in ("fastapi", "uvicorn", "multipart", "sse_starlette"):
    try:
        __import__(pkg)
        check(pkg, True)
    except ImportError:
        check(pkg, False, f"pip install {pkg}")

# ── HuggingFace token ────────────────────────────────────────────────────────
print("\nHuggingFace")
hf_token = os.environ.get("TE_HF_TOKEN", os.environ.get("HF_TOKEN", ""))
check("HF token set", bool(hf_token) and hf_token != "hf_your_token_here",
      "set TE_HF_TOKEN in .env", critical=False)
if hf_token and hf_token != "hf_your_token_here":
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        api.whoami()
        check("HF token valid", True)
    except Exception as e:
        check("HF token valid", False, str(e), critical=False)

# ── Transcript Engine package ────────────────────────────────────────────────
print("\nTranscript Engine")
try:
    import transcript_engine  # noqa: F401
    check("transcript_engine importable", True)
    from transcript_engine.transcription.parakeet_engine import ParakeetEngine  # noqa: F401
    check("ParakeetEngine importable", True)
except ImportError as e:
    check("transcript_engine importable", False, str(e))

# ── Summary ───────────────────────────────────────────────────────────────────
print()
if errors:
    print(f"\033[31mFAILED — {len(errors)} critical issue(s):\033[0m")
    for err in errors:
        print(f"  • {err}")
    sys.exit(1)
elif warnings:
    print(f"\033[33mPASSED with {len(warnings)} warning(s)\033[0m")
    for w in warnings:
        print(f"  • {w}")
    sys.exit(0)
else:
    print("\033[32mAll checks passed — ready to transcribe.\033[0m")
    sys.exit(0)
