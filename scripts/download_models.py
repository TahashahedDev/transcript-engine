#!/usr/bin/env python3
"""
Pre-download all models required by Transcript Engine to local cache.

Run once after setup_gpu.sh to avoid cold-start delays on first transcription:
    python scripts/download_models.py

Models downloaded:
  - nvidia/parakeet-tdt-0.6b-v2  (~2.5 GB, NeMo ASR)
  - pyannote/speaker-diarization-3.1  (requires HF token + accepted terms)
  - pyannote/segmentation-3.0  (pulled as dependency of speaker-diarization-3.1)
"""

from __future__ import annotations

import importlib
import os
import sys
import time
from pathlib import Path
from typing import Any, cast

# ── Load .env before checking tokens ─────────────────────────────────────────

def _load_env() -> None:
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

_load_env()

HF_TOKEN = os.environ.get("TE_HF_TOKEN", os.environ.get("HF_TOKEN", ""))
PARAKEET_MODEL = os.environ.get("TE_PARAKEET_MODEL", "nvidia/parakeet-tdt-0.6b-v2")


def _require_token() -> str:
    if not HF_TOKEN or HF_TOKEN == "hf_your_token_here":
        print("\nERROR: TE_HF_TOKEN not set — required for pyannote models.")
        print("  1. Get a token at https://huggingface.co/settings/tokens")
        print("  2. Accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1")
        print("  3. Accept terms at https://huggingface.co/pyannote/segmentation-3.0")
        print("  4. Set TE_HF_TOKEN=hf_... in .env, then re-run this script.")
        sys.exit(1)
    return HF_TOKEN


# ── Compatibility patches (must run BEFORE any pyannote/torch imports) ────────

def _patch_torch_load() -> None:
    """
    PyTorch 2.6 changed torch.load default to weights_only=True, breaking
    pyannote and NeMo checkpoints that use omegaconf/typing objects. Restore
    pre-2.6 behaviour for callers that don't specify weights_only explicitly.
    """
    # Only the import is tolerated failing (torch genuinely absent). The patch
    # itself is not wrapped: a previous version had a typo here that raised
    # AttributeError, which a broad `except (ImportError, AttributeError): pass`
    # swallowed — so the patch silently never applied and the failure surfaced
    # later as an unpickling error during model download.
    try:
        import torch
    except ImportError:
        return

    if getattr(torch.load, "_te_patched", False):
        return
    _orig = torch.load

    def _patched(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("weights_only") is not True:
            kwargs["weights_only"] = False
        return _orig(*args, **kwargs)

    cast(Any, _patched)._te_patched = True
    torch.load = _patched


def _patch_hf_hub() -> None:
    """
    Bridge pyannote.audio 3.x (uses deprecated use_auth_token kwarg) with
    huggingface_hub >= 1.0 (renamed it to token, raises TypeError otherwise).
    Must be applied before any Pipeline.from_pretrained() call.
    """
    try:
        import huggingface_hub
        if getattr(huggingface_hub.hf_hub_download, "_te_patched", False):
            return
        _orig = huggingface_hub.hf_hub_download

        def _patched(*args: Any, **kwargs: Any) -> Any:
            if "use_auth_token" in kwargs:
                uak = kwargs.pop("use_auth_token")
                if uak and "token" not in kwargs:
                    kwargs["token"] = uak
            return _orig(*args, **kwargs)

        cast(Any, _patched)._te_patched = True
        huggingface_hub.hf_hub_download = _patched

        for _mod_name in [
            "pyannote.audio.core.pipeline",
            "pyannote.audio.core.model",
            "pyannote.audio.pipelines.speaker_verification",
        ]:
            try:
                _mod = importlib.import_module(_mod_name)
                if hasattr(_mod, "hf_hub_download"):
                    cast(Any, _mod).hf_hub_download = _patched
            except (ImportError, AttributeError):
                pass
    except (ImportError, AttributeError):
        pass


# ── Downloaders ───────────────────────────────────────────────────────────────

def download_parakeet() -> None:
    print(f"\n[1/2] Downloading Parakeet: {PARAKEET_MODEL}")
    print("      (~2.5 GB — takes 3-10 min on first download)")
    t0 = time.monotonic()
    try:
        from nemo.collections.asr.models import EncDecRNNTBPEModel
    except ImportError:
        print("\nERROR: nemo_toolkit[asr] not installed.")
        print("  Run: pip install -e \".[nvidia]\"")
        sys.exit(1)

    _patch_torch_load()

    try:
        model = EncDecRNNTBPEModel.from_pretrained(PARAKEET_MODEL)
    except Exception as exc:
        print(f"\nERROR: Failed to download Parakeet: {exc}")
        print("  Check your internet connection and try again.")
        sys.exit(1)

    elapsed = time.monotonic() - t0
    print(f"  Downloaded in {elapsed:.1f}s")

    import torch
    if torch.cuda.is_available():
        model = model.cuda()
        gpu = torch.cuda.get_device_name(0)
        # Quick smoke test: forward pass on a tiny tensor
        try:
            import tempfile
            import wave
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_wav = f.name
            # Write minimal 1-second 16kHz mono WAV
            import array as _array
            with wave.open(tmp_wav, "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(_array.array("h", [0] * 16000).tobytes())
            model.transcribe(audio=[tmp_wav], batch_size=1, return_hypotheses=True, verbose=False)
            Path(tmp_wav).unlink(missing_ok=True)
            print(f"  Verified on CUDA: {gpu} (inference test passed)")
        except Exception as exc:
            print(f"  WARNING: Inference test failed: {exc}")
            print("  Model downloaded but GPU test could not complete.")
    else:
        print("  WARNING: No CUDA available — model loaded on CPU (will be slow)")

    del model
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("  Parakeet ready.")


def download_pyannote() -> None:
    token = _require_token()
    print("\n[2/2] Downloading pyannote/speaker-diarization-3.1")
    print("      (~500 MB — takes 1-3 min on first download)")
    t0 = time.monotonic()

    try:
        from pyannote.audio import Pipeline
    except ImportError:
        print("\nERROR: pyannote.audio not installed (should be in base dependencies).")
        print("  Run: pip install -e \".\"")
        sys.exit(1)

    from transcript_engine.diarization.compat import load_pretrained_pipeline

    _patch_torch_load()
    _patch_hf_hub()

    try:
        pipeline = load_pretrained_pipeline(Pipeline, "pyannote/speaker-diarization-3.1", token)
    except Exception as exc:
        msg = str(exc)
        print(f"\nERROR: Failed to download pyannote: {msg}")
        if "401" in msg or "invalid" in msg.lower():
            print("  Your HF token is invalid. Generate a new one at huggingface.co/settings/tokens")
        elif "403" in msg or "gated" in msg.lower() or "license" in msg.lower():
            print("  You must accept the model license:")
            print("    https://huggingface.co/pyannote/speaker-diarization-3.1")
            print("    https://huggingface.co/pyannote/segmentation-3.0")
        else:
            print("  Check your internet connection and HF token, then retry.")
        sys.exit(1)

    elapsed = time.monotonic() - t0
    print(f"  Downloaded in {elapsed:.1f}s")
    del pipeline
    print("  pyannote ready.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Transcript Engine — Model Download")
    print("=" * 44)

    download_parakeet()
    download_pyannote()

    print("\n" + "=" * 44)
    print("All models downloaded successfully.")
    print()
    print("Next: run python scripts/preflight_check.py to verify the full setup.")
