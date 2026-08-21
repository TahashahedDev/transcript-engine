"""
Unit tests for transcript_engine.identity.embedding_extractor._select_device.

This machine has no CUDA device, so the CUDA-available branches are
exercised by mocking torch.cuda rather than by running on real hardware —
that's deliberate: the LOGIC (does it read free VRAM correctly, does it
respect the threshold) is fully testable without a GPU. What it can't test,
and doesn't claim to, is whether 1.0 GB is actually enough headroom on a
real RTX 3080 alongside resident Parakeet/pyannote weights — that requires
the real GPU box.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from transcript_engine.identity.embedding_extractor import (
    _EMBEDDING_VRAM_NEEDED_GB,
    _select_device,
)


def test_no_cuda_available_selects_cpu():
    with patch("torch.cuda.is_available", return_value=False):
        device, on_cuda = _select_device()
    assert device.type == "cpu"
    assert on_cuda is False


def test_cuda_available_with_enough_free_vram_selects_cuda():
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.mem_get_info", return_value=(int(4e9), int(20e9))),
    ):
        device, on_cuda = _select_device()
    assert device.type == "cuda"
    assert on_cuda is True


def test_cuda_available_with_insufficient_free_vram_falls_back_to_cpu():
    # Free VRAM below the threshold — e.g. Parakeet + pyannote already
    # resident and holding most of the card.
    scant_free_bytes = int((_EMBEDDING_VRAM_NEEDED_GB - 0.1) * 1e9)
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.mem_get_info", return_value=(scant_free_bytes, int(20e9))),
    ):
        device, on_cuda = _select_device()
    assert device.type == "cpu"
    assert on_cuda is False


def test_mem_get_info_unavailable_falls_back_to_total_vram_heuristic():
    # Older torch without mem_get_info — same fallback registry.py already
    # uses for pyannote's own device selection.
    props = MagicMock()
    props.total_memory = int(20e9)
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.mem_get_info", side_effect=RuntimeError("not supported")),
        patch("torch.cuda.get_device_properties", return_value=props),
    ):
        device, on_cuda = _select_device()
    assert device.type == "cuda"
    assert on_cuda is True


def test_explicit_device_argument_bypasses_auto_selection():
    # SpeakerEmbeddingExtractor(device=...) must respect an explicit choice
    # rather than silently overriding it with the free-VRAM heuristic.
    from transcript_engine.identity.embedding_extractor import SpeakerEmbeddingExtractor

    with patch(
        "pyannote.audio.pipelines.speaker_verification.PretrainedSpeakerEmbedding"
    ) as mock_embedding_cls:
        mock_embedding_cls.return_value = MagicMock()
        extractor = SpeakerEmbeddingExtractor(hf_token=None, device=torch.device("cpu"))

    assert extractor._device.type == "cpu"  # noqa: SLF001
    assert extractor._on_cuda is False  # noqa: SLF001
