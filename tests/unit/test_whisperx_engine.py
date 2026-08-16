"""
Unit tests for WhisperXEngine._extract_words.
No ML models loaded — tests the pure extraction logic only.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from transcript_engine.transcription.whisperx_engine import WhisperXEngine


def _engine() -> WhisperXEngine:
    registry = MagicMock()
    return WhisperXEngine(registry)


class TestExtractWords:
    def test_aligned_output_with_word_data(self) -> None:
        """Normal post-alignment path: each segment has a 'words' list."""
        engine = _engine()
        aligned = {
            "segments": [
                {
                    "text": "Hello world",
                    "start": 0.0,
                    "end": 2.0,
                    "words": [
                        {"word": "Hello", "start": 0.0, "end": 0.8, "score": 0.95},
                        {"word": "world", "start": 0.9, "end": 2.0, "score": 0.88},
                    ],
                }
            ]
        }
        words = engine._extract_words(aligned)
        assert len(words) == 2
        assert words[0].text == "Hello"
        assert words[0].confidence == 0.95
        assert words[1].text == "world"

    def test_skip_alignment_path_no_word_data(self) -> None:
        """
        Root-cause regression test: raw CT2 segments (skip_alignment=True) have no
        'words' key. _extract_words must fall back to distributing segment text
        evenly instead of returning 0 words.
        """
        engine = _engine()
        # This is the exact structure whisperx.transcribe() returns without alignment
        raw_ct2_result = {
            "segments": [
                {"text": "Hello world how are you", "start": 0.0, "end": 5.0},
                {"text": "I am fine thank you", "start": 5.5, "end": 9.0},
            ]
        }
        words = engine._extract_words(raw_ct2_result)

        # Must not return 0 words — that was the bug
        assert len(words) > 0, "skip_alignment path returned 0 words (regression)"

        # Should have one word per token across both segments
        assert len(words) == 5 + 5  # 5 tokens + 5 tokens

        # Words should be ordered by time
        for i in range(1, len(words)):
            assert words[i].start >= words[i - 1].start

        # Confidence is None because CT2 native timestamps have no per-word score
        assert all(w.confidence is None for w in words)

        # Text must cover all tokens
        texts = [w.text for w in words]
        assert "Hello" in texts
        assert "you" in texts
        assert "fine" in texts

    def test_empty_segments(self) -> None:
        engine = _engine()
        assert engine._extract_words({"segments": []}) == []

    def test_empty_segment_text_skipped(self) -> None:
        engine = _engine()
        result = {"segments": [{"text": "  ", "start": 0.0, "end": 1.0}]}
        assert engine._extract_words(result) == []

    def test_missing_start_end_falls_back(self) -> None:
        """Segments with missing start/end should not raise."""
        engine = _engine()
        result = {
            "segments": [
                {
                    "text": "hello",
                    "words": [{"word": "hello", "score": 0.9}],
                }
            ]
        }
        words = engine._extract_words(result)
        assert len(words) == 1
        assert words[0].text == "hello"

    def test_mixed_segments_word_and_no_word(self) -> None:
        """Some segments with word data, some without — both must produce words."""
        engine = _engine()
        aligned = {
            "segments": [
                {
                    "text": "aligned segment",
                    "start": 0.0,
                    "end": 2.0,
                    "words": [
                        {"word": "aligned", "start": 0.0, "end": 0.9, "score": 0.9},
                        {"word": "segment", "start": 1.0, "end": 2.0, "score": 0.85},
                    ],
                },
                # Raw CT2 segment with no word data
                {"text": "raw segment", "start": 2.5, "end": 4.5},
            ]
        }
        words = engine._extract_words(aligned)
        assert len(words) == 4  # 2 from aligned + 2 from raw
        assert words[0].text == "aligned"
        assert words[2].text == "raw"
        assert words[3].text == "segment"
