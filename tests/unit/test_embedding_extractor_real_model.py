"""
Integration test for transcript_engine.identity.embedding_extractor against
the REAL pyannote/wespeaker-voxceleb-resnet34-LM model — not a mock.

This is the one test in the identity module that is intentionally NOT purely
synthetic-algorithmic: it loads the actual model from the local HF cache
(already present on this machine because pyannote's own diarization pipeline
downloaded it) and runs real inference. What is synthetic is the *audio*:
there is no real multi-speaker speech recording anywhere in this repository
(verified — see IDENTITY_ARCHITECTURE.md), so the two "voices" here are
distinct synthetic waveforms (different frequency sine tones), not real
speech. This proves the model-loading and inference code path is correct
end-to-end and that embeddings behave sanely (same signal → higher
similarity than different signal); it does NOT and cannot demonstrate real
voice-identification accuracy, since no voice was ever involved.

Skips (does not fail) if the model or HF token is unavailable, so this
suite still runs somewhere without pyannote credentials configured.
"""

from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np
import pytest


def _write_sine_wav(path: Path, freq_hz: float, duration_s: float, sample_rate: int = 16_000) -> None:
    n_samples = int(duration_s * sample_rate)
    t = np.arange(n_samples) / sample_rate
    # Two harmonics, not a pure tone, so the "voice" has more spectral
    # structure than a single sine — still not speech, but less trivially
    # separable than one frequency alone.
    signal = 0.6 * np.sin(2 * math.pi * freq_hz * t) + 0.3 * np.sin(2 * math.pi * freq_hz * 2.01 * t)
    pcm = (signal * 32767 * 0.8).astype(np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.tobytes())


@pytest.fixture
def extractor():
    try:
        from transcript_engine.config.loader import load_settings
        from transcript_engine.identity.embedding_extractor import SpeakerEmbeddingExtractor
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"identity embedding stack unavailable: {exc}")

    settings = load_settings()
    if not settings.hf_token:
        pytest.skip("no HF token configured locally — cannot load the gated embedding model")

    try:
        return SpeakerEmbeddingExtractor(hf_token=settings.hf_token)
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"embedding model unavailable in this environment: {exc}")


def test_real_model_produces_256_dim_embedding(tmp_path: Path, extractor):
    from transcript_engine.identity.embedding_extractor import EmbeddingWindow

    wav_path = tmp_path / "tone_a.wav"
    _write_sine_wav(wav_path, freq_hz=220.0, duration_s=3.0)

    windows = [EmbeddingWindow(speaker_id="SPEAKER_00", start=0.0, end=3.0)]
    result = extractor.extract(wav_path, windows)

    assert 0 in result
    assert result[0].shape == (256,)


def test_same_signal_is_more_similar_than_different_signal(tmp_path: Path, extractor):
    from transcript_engine.identity.embedding_extractor import EmbeddingWindow
    from transcript_engine.identity.matcher import cosine_similarity

    wav_a1 = tmp_path / "tone_a1.wav"
    wav_a2 = tmp_path / "tone_a2.wav"
    wav_b = tmp_path / "tone_b.wav"
    _write_sine_wav(wav_a1, freq_hz=220.0, duration_s=3.0)
    _write_sine_wav(wav_a2, freq_hz=221.0, duration_s=3.0)  # near-identical "voice"
    _write_sine_wav(wav_b, freq_hz=880.0, duration_s=3.0)  # a very different "voice"

    windows = [EmbeddingWindow(speaker_id="SPEAKER_00", start=0.0, end=3.0)]
    emb_a1 = extractor.extract(wav_a1, windows)[0]
    emb_a2 = extractor.extract(wav_a2, windows)[0]
    emb_b = extractor.extract(wav_b, windows)[0]

    same_signal_similarity = cosine_similarity(emb_a1, emb_a2)
    different_signal_similarity = cosine_similarity(emb_a1, emb_b)

    assert same_signal_similarity > different_signal_similarity


def test_extract_skips_out_of_range_window_without_failing_the_batch(tmp_path: Path, extractor):
    from transcript_engine.identity.embedding_extractor import EmbeddingWindow

    wav_path = tmp_path / "tone.wav"
    _write_sine_wav(wav_path, freq_hz=220.0, duration_s=2.0)

    windows = [
        EmbeddingWindow(speaker_id="SPEAKER_00", start=0.0, end=2.0),
        EmbeddingWindow(speaker_id="SPEAKER_01", start=100.0, end=103.0),  # past end of file
    ]
    result = extractor.extract(wav_path, windows)

    assert 0 in result
    assert 1 not in result
