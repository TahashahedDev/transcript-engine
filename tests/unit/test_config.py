"""Unit tests for configuration loading and layering."""

from __future__ import annotations

import pytest

from transcript_engine.config.settings import Settings, TranscriptionConfig


class TestSettingsDefaults:
    def test_default_model(self) -> None:
        s = Settings()
        assert "large-v3" in s.pipeline.transcription.model_id

    def test_default_device_auto(self) -> None:
        s = Settings()
        assert s.pipeline.transcription.device == "auto"

    def test_default_diarization_enabled(self) -> None:
        s = Settings()
        assert s.pipeline.diarization.enabled is True

    def test_default_formats(self) -> None:
        s = Settings()
        assert "markdown" in s.pipeline.export.formats
        assert "json" in s.pipeline.export.formats


class TestSettingsEnvOverride:
    def test_env_overrides_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TE_PIPELINE__TRANSCRIPTION__MODEL_ID", "tiny")
        s = Settings()
        assert s.pipeline.transcription.model_id == "tiny"

    def test_env_log_level(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TE_LOG_LEVEL", "DEBUG")
        s = Settings()
        assert s.log_level == "DEBUG"

    def test_hf_token_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TE_HF_TOKEN", "hf_test_token")
        s = Settings()
        assert s.hf_token == "hf_test_token"


class TestTranscriptionConfig:
    def test_defaults(self) -> None:
        c = TranscriptionConfig()
        assert c.word_timestamps is True
        assert c.vad_filter is True
        assert c.beam_size == 2

    def test_language_default_is_en(self) -> None:
        c = TranscriptionConfig()
        assert c.language == "en"
