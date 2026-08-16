"""Unit tests for all exporters using synthetic Transcript objects."""

from __future__ import annotations

import json

import pytest

from tests.conftest import make_segment, make_transcript
from transcript_engine.config.settings import ExportConfig
from transcript_engine.exporters.json_exporter import JSONExporter
from transcript_engine.exporters.markdown import MarkdownExporter
from transcript_engine.exporters.srt import SRTExporter
from transcript_engine.exporters.txt import TxtExporter
from transcript_engine.exporters.vtt import VTTExporter


@pytest.fixture
def config() -> ExportConfig:
    return ExportConfig()


@pytest.fixture
def two_speaker_transcript():
    segs = [
        make_segment("SPEAKER_00", [("Hello", 0.0, 0.5), ("there", 0.5, 1.0)]),
        make_segment("SPEAKER_01", [("Hi", 2.0, 2.4), ("back", 2.4, 2.8)]),
    ]
    return make_transcript(
        segments=segs,
        speakers={"SPEAKER_00": "John", "SPEAKER_01": "Neel"},
    )


class TestMarkdownExporter:
    def test_output_starts_with_title(self, two_speaker_transcript, config) -> None:
        result = MarkdownExporter().export(two_speaker_transcript, config).decode()
        assert result.startswith("# ")

    def test_speaker_headers(self, two_speaker_transcript, config) -> None:
        result = MarkdownExporter().export(two_speaker_transcript, config).decode()
        assert "## John" in result
        assert "## Neel" in result

    def test_text_present(self, two_speaker_transcript, config) -> None:
        result = MarkdownExporter().export(two_speaker_transcript, config).decode()
        assert "Hello there" in result
        assert "Hi back" in result

    def test_blank_line_between_speakers(self, two_speaker_transcript, config) -> None:
        result = MarkdownExporter().export(two_speaker_transcript, config).decode()
        assert "\n\n" in result

    def test_utf8_encoding(self, two_speaker_transcript, config) -> None:
        raw = MarkdownExporter().export(two_speaker_transcript, config)
        assert isinstance(raw, bytes)
        raw.decode("utf-8")


class TestJSONExporter:
    def test_valid_json(self, two_speaker_transcript, config) -> None:
        raw = JSONExporter().export(two_speaker_transcript, config)
        parsed = json.loads(raw)
        assert "segments" in parsed
        assert "speakers" in parsed

    def test_speakers_preserved(self, two_speaker_transcript, config) -> None:
        raw = JSONExporter().export(two_speaker_transcript, config)
        parsed = json.loads(raw)
        assert parsed["speakers"]["SPEAKER_00"] == "John"

    def test_round_trip(self, two_speaker_transcript, config) -> None:
        from transcript_engine.models.transcript import Transcript

        raw = JSONExporter().export(two_speaker_transcript, config)
        restored = Transcript.model_validate_json(raw)
        assert restored.word_count == two_speaker_transcript.word_count
        assert restored.speakers == two_speaker_transcript.speakers


class TestTxtExporter:
    def test_speaker_colon_format(self, two_speaker_transcript, config) -> None:
        result = TxtExporter().export(two_speaker_transcript, config).decode()
        assert "John: Hello there" in result
        assert "Neel: Hi back" in result


class TestSRTExporter:
    def test_starts_with_index(self, two_speaker_transcript, config) -> None:
        result = SRTExporter().export(two_speaker_transcript, config).decode()
        assert result.startswith("1\n")

    def test_timecode_format(self, two_speaker_transcript, config) -> None:
        result = SRTExporter().export(two_speaker_transcript, config).decode()
        assert "00:00:00,000 --> 00:00:01,000" in result

    def test_speaker_in_brackets(self, two_speaker_transcript, config) -> None:
        result = SRTExporter().export(two_speaker_transcript, config).decode()
        assert "[John]" in result


class TestVTTExporter:
    def test_starts_with_webvtt(self, two_speaker_transcript, config) -> None:
        result = VTTExporter().export(two_speaker_transcript, config).decode()
        assert result.startswith("WEBVTT")

    def test_voice_tag(self, two_speaker_transcript, config) -> None:
        result = VTTExporter().export(two_speaker_transcript, config).decode()
        assert "<v John>" in result

    def test_timecode_dot_format(self, two_speaker_transcript, config) -> None:
        result = VTTExporter().export(two_speaker_transcript, config).decode()
        assert "00:00:00.000 --> " in result
