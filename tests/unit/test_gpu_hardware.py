"""
Unit tests for GPU hardware detection helpers.

No CUDA required: torch is faked in sys.modules so the arch-compatibility
logic can be exercised on any machine, including CI without a GPU.
"""

from __future__ import annotations

import sys
import types

import pytest

from transcript_engine.gpu.hardware import (
    GpuInfo,
    _parse_arch,
    check_gpu_compatibility,
    optimal_chunk_seconds,
)


def _gpu(cc: tuple[int, int], name: str = "Test GPU") -> GpuInfo:
    return GpuInfo(
        index=0,
        name=name,
        cuda_version="12.1",
        vram_total_gb=24.0,
        vram_free_gb=20.0,
        compute_capability=cc,
    )


def _fake_torch(monkeypatch: pytest.MonkeyPatch, arch_list: list[str]) -> None:
    mod = types.ModuleType("torch")
    mod.cuda = types.SimpleNamespace(get_arch_list=lambda: arch_list)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", mod)


class TestParseArch:
    def test_parses_cubin_entry(self) -> None:
        assert _parse_arch("sm_86") == ("sm", 86)

    def test_parses_ptx_entry(self) -> None:
        assert _parse_arch("compute_90") == ("compute", 90)

    def test_parses_suffixed_arch(self) -> None:
        """Hopper ships arch-specific variants like sm_90a."""
        assert _parse_arch("sm_90a") == ("sm", 90)

    def test_ignores_unknown_entry(self) -> None:
        assert _parse_arch("nonsense") is None


class TestGpuCompatibility:
    def test_exact_arch_match_is_compatible(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fake_torch(monkeypatch, ["sm_80", "sm_86", "sm_90"])
        assert check_gpu_compatibility(_gpu((8, 6), "RTX 3060 Ti")) is None

    def test_newer_gpu_than_build_is_rejected_with_actionable_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        The RTX 5090 case: Blackwell (sm_120) on a wheel built up to sm_90.
        CUDA reports the device fine, so without this check the job would die
        later with 'no kernel image is available for execution on the device'.
        """
        _fake_torch(monkeypatch, ["sm_70", "sm_80", "sm_86", "sm_90"])

        msg = check_gpu_compatibility(_gpu((12, 0), "RTX 5090"))

        assert msg is not None
        assert "not supported by the installed PyTorch build" in msg
        assert "sm_120" in msg
        assert "RTX 5090" in msg
        assert "pytorch.org" in msg  # tells the operator how to fix it

    def test_ptx_forward_compat_is_allowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A build shipping PTX can JIT forward onto a newer device."""
        _fake_torch(monkeypatch, ["sm_80", "sm_90", "compute_90"])
        assert check_gpu_compatibility(_gpu((12, 0), "RTX 5090")) is None

    def test_ptx_newer_than_device_does_not_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """PTX only JITs forward, never backward onto older hardware."""
        _fake_torch(monkeypatch, ["sm_90", "compute_90"])
        assert check_gpu_compatibility(_gpu((7, 5), "RTX 2080")) is not None

    def test_empty_arch_list_does_not_block(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fake_torch(monkeypatch, [])
        assert check_gpu_compatibility(_gpu((8, 6))) is None

    def test_unreadable_torch_does_not_block(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Never fail a job because compatibility could not be determined."""
        mod = types.ModuleType("torch")

        def _boom() -> list[str]:
            raise RuntimeError("no cuda")

        mod.cuda = types.SimpleNamespace(get_arch_list=_boom)  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "torch", mod)

        assert check_gpu_compatibility(_gpu((8, 6))) is None


class TestSmVersion:
    def test_sm_version_packs_capability(self) -> None:
        assert _gpu((8, 6)).sm_version == 86
        assert _gpu((12, 0)).sm_version == 120
        assert _gpu((9, 0)).sm_version == 90


class TestOptimalChunkSeconds:
    def test_small_vram_gets_short_chunks(self) -> None:
        assert optimal_chunk_seconds(3.0) == 90.0

    def test_target_8gb_card(self) -> None:
        assert optimal_chunk_seconds(7.0) == 300.0

    def test_large_vram_capped_at_model_limit(self) -> None:
        """Parakeet's full-attention design limit — never exceed it."""
        assert optimal_chunk_seconds(80.0) == 1320.0

    def test_thresholds_are_monotonic(self) -> None:
        vrams = [1.0, 5.0, 7.0, 10.0, 14.0, 20.0, 48.0]
        chunks = [optimal_chunk_seconds(v) for v in vrams]
        assert chunks == sorted(chunks), "more VRAM must never mean smaller chunks"
