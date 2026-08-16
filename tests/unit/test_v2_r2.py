"""Unit tests for R2 key helpers and sanitization logic."""

from __future__ import annotations

import uuid

from api.v2.r2 import _sanitize_filename, make_audio_key, make_transcript_key

_USER = uuid.UUID("00000000-0000-0000-0000-000000000001")
_JOB = uuid.UUID("00000000-0000-0000-0000-000000000002")


def test_make_audio_key_structure() -> None:
    key = make_audio_key(_USER, _JOB, "meeting.aac")
    assert key == f"audio/{_USER}/{_JOB}/meeting.aac"


def test_make_audio_key_sanitizes_filename() -> None:
    key = make_audio_key(_USER, _JOB, "my meeting (2024).aac")
    assert " " not in key
    assert "(" not in key
    assert key.startswith(f"audio/{_USER}/{_JOB}/")


def test_make_transcript_key_structure() -> None:
    for fmt in ("txt", "md", "json", "srt", "vtt", "docx"):
        key = make_transcript_key(_USER, _JOB, fmt)
        assert key == f"transcripts/{_USER}/{_JOB}/transcript.{fmt}"


def test_sanitize_filename_removes_special_chars() -> None:
    assert _sanitize_filename("hello world!.mp3") == "hello_world_.mp3"


def test_sanitize_filename_preserves_dots_hyphens_underscores() -> None:
    name = "my-file_name.wav"
    assert _sanitize_filename(name) == name


def test_sanitize_filename_truncates_to_200() -> None:
    long_name = "a" * 300 + ".mp3"
    result = _sanitize_filename(long_name)
    assert len(result) == 200


def test_sanitize_filename_empty_falls_back() -> None:
    assert _sanitize_filename("!!!") == "___"


def test_audio_key_prefix_encodes_user() -> None:
    """Ownership validation depends on key starting with audio/{user_id}/."""
    key = make_audio_key(_USER, _JOB, "audio.mp3")
    assert key.startswith(f"audio/{_USER}/")
