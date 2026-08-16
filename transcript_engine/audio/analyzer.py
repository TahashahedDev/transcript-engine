"""
Fast pre-pipeline audio analysis.

Runs in <5 seconds on any audio length. Produces an AudioProfile with
measurable signals and a PipelineDecision with reasons and estimated savings.

Signals computed:
  - duration_s: from torchaudio header (no decode)
  - speech_ratio: fraction of audio with speech (energy-based, first 60s sample)
  - silence_ratio: 1 - speech_ratio
  - estimated_speakers: spectral MFCC variance estimate (fast, no ML model)
  - snr_estimate_db: rough dB estimate from speech vs silence RMS
  - sample_confidence: how reliable these estimates are (0-1)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from transcript_engine.logging import get_logger

logger = get_logger(__name__)

# Speaker count thresholds on std-dev of MFCC-0 coefficient.
_SINGLE_SPEAKER_VAR_THRESHOLD = 0.30
_TWO_SPEAKER_VAR_THRESHOLD = 0.60

# Energy threshold for speech vs silence detection (fraction of mean frame RMS).
_SPEECH_ENERGY_RATIO = 0.15

# Minimum confidence required to skip diarization.
# Set conservatively — a false positive (skipping diarization on a multi-speaker
# recording) produces no speaker labels, which is a visible user-facing bug.
_DIAR_SKIP_CONFIDENCE = 0.85

# Typical pyannote runtime on 60-min audio (seconds) — used for savings estimate.
_DIAR_TYPICAL_SAVINGS_S = 540

# Speaker estimation samples this many seconds per window, at up to this many
# points spread across the recording. Analysis stays bounded (a handful of short
# decodes) regardless of whether the file is 3 minutes or 3 hours.
_WINDOW_SECONDS = 60
_MAX_WINDOWS = 4


@dataclass
class AudioProfile:
    duration_s: float
    speech_ratio: float  # 0.0-1.0
    silence_ratio: float  # 0.0-1.0
    estimated_speakers: int  # 1, 2, or 3
    snr_estimate_db: float  # rough signal-to-noise in dB
    sample_confidence: float  # 0.0-1.0 (how reliable the estimates are)
    analysis_seconds: float  # how long the analysis took


@dataclass
class PipelineDecision:
    use_diarization: bool
    use_wav2vec2_alignment: bool  # False → rely on CT2 native word timestamps
    num_chunks: int  # 1 = no chunking (Phase 2 will increase this)
    reasons: list[str] = field(default_factory=list)
    estimated_savings_s: float = 0.0
    audio_profile: AudioProfile | None = None


def _default_decision(profile: AudioProfile | None = None) -> PipelineDecision:
    """Safe fallback — full quality pipeline, no skipping."""
    return PipelineDecision(
        use_diarization=True,
        use_wav2vec2_alignment=True,
        num_chunks=1,
        reasons=["Analysis failed — using full quality pipeline."],
        estimated_savings_s=0.0,
        audio_profile=profile,
    )


class AudioAnalyzer:
    """
    Fast audio pre-analysis using torchaudio (no model inference).

    All signals are computed from the first/middle 60 seconds of audio so
    analysis time is bounded regardless of recording length.
    """

    def analyze(self, wav_path: Path) -> tuple[AudioProfile, PipelineDecision]:
        t0 = time.monotonic()
        try:
            return self._analyze(wav_path, t0)
        except Exception as exc:
            logger.warning(
                "AudioAnalyzer failed for %s: %s — using full pipeline",
                wav_path.name,
                exc,
            )
            fallback_profile = AudioProfile(
                duration_s=0.0,
                speech_ratio=0.5,
                silence_ratio=0.5,
                estimated_speakers=2,
                snr_estimate_db=0.0,
                sample_confidence=0.0,
                analysis_seconds=time.monotonic() - t0,
            )
            return fallback_profile, _default_decision(fallback_profile)

    def _analyze(
        self, wav_path: Path, t0: float
    ) -> tuple[AudioProfile, PipelineDecision]:
        import torchaudio  # noqa: PLC0415

        # ── Duration (header-only, no decode) ─────────────────────────────────
        info = torchaudio.info(str(wav_path))
        duration_s = float(info.num_frames) / float(info.sample_rate)
        sample_rate = info.sample_rate

        # ── Speech/silence ratio from first 60s ───────────────────────────────
        max_frames = min(info.num_frames, int(60 * sample_rate))
        waveform, _ = torchaudio.load(str(wav_path), num_frames=max_frames)
        # Mix to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)  # (T,)

        frame_size = int(0.1 * sample_rate)  # 100ms frames
        speech_ratio, silence_ratio, snr_db = self._energy_analysis(
            waveform, frame_size
        )

        # ── Speaker count estimate from several windows ───────────────────────
        # Sampled at multiple points rather than one mid-audio window. A single
        # window is easily unrepresentative: one person presenting for the two
        # minutes that happen to be sampled looks exactly like a single-speaker
        # recording, and that verdict was then applied to the entire meeting.
        window_offsets = self._sample_offsets(info.num_frames, sample_rate)
        window_vars: list[float] = []
        window_counts: list[int] = []
        for offset in window_offsets:
            if offset == 0 and max_frames >= min(info.num_frames, int(_WINDOW_SECONDS * sample_rate)):
                window = waveform.unsqueeze(0)  # reuse the head we already decoded
            else:
                window, _ = torchaudio.load(
                    str(wav_path),
                    frame_offset=offset,
                    num_frames=int(_WINDOW_SECONDS * sample_rate),
                )
                if window.shape[0] > 1:
                    window = window.mean(dim=0, keepdim=True)
            var, count = self._speaker_estimate(window.squeeze(0), sample_rate)
            window_vars.append(var)
            window_counts.append(count)

        # Take the highest per-window estimate: hearing two voices anywhere is
        # positive evidence of multiple speakers, whereas hearing one is only
        # ever weak evidence about the rest of the recording.
        mfcc_var = max(window_vars) if window_vars else 0.0
        estimated_speakers = max(window_counts) if window_counts else 2

        # ── sample_confidence ─────────────────────────────────────────────────
        # Reflects how much of the recording was actually examined, and whether
        # the windows agreed. The previous version returned 0.9 for anything
        # longer than two minutes regardless of what was sampled, which meant
        # the 0.85 skip gate was effectively always open.
        analyzed_s = len(window_offsets) * _WINDOW_SECONDS
        coverage = min(1.0, analyzed_s / duration_s) if duration_s > 0 else 0.0
        unanimous = len(set(window_counts)) <= 1

        if duration_s < 120 or len(window_offsets) < 2:
            # Too short to sample independently — the estimate is a coin flip.
            sample_confidence = 0.5
        elif unanimous:
            # Cap below 1.0: agreement across samples is suggestive, not proof.
            sample_confidence = round(min(0.95, 0.55 + 0.45 * coverage), 3)
        else:
            # Windows disagreed — that alone means the estimate is unreliable.
            sample_confidence = 0.4

        elapsed = time.monotonic() - t0
        profile = AudioProfile(
            duration_s=duration_s,
            speech_ratio=round(speech_ratio, 3),
            silence_ratio=round(silence_ratio, 3),
            estimated_speakers=estimated_speakers,
            snr_estimate_db=round(snr_db, 1),
            sample_confidence=sample_confidence,
            analysis_seconds=round(elapsed, 2),
        )

        logger.info(
            "AudioAnalyzer: duration=%.0fs, speech=%.0f%%, speakers=%d, "
            "snr=%.1fdB, confidence=%.2f, elapsed=%.2fs",
            duration_s,
            speech_ratio * 100,
            estimated_speakers,
            snr_db,
            sample_confidence,
            elapsed,
        )

        decision = self._make_decision(profile, mfcc_var)
        return profile, decision

    # ── Signal extraction ─────────────────────────────────────────────────────

    def _energy_analysis(
        self, waveform: Any, frame_size: int
    ) -> tuple[float, float, float]:
        """Compute speech/silence ratio and SNR from amplitude frames."""
        try:
            n_frames = len(waveform) // frame_size
            if n_frames == 0:
                return 0.5, 0.5, 0.0

            rms_frames = []
            for i in range(n_frames):
                chunk = waveform[i * frame_size : (i + 1) * frame_size]
                rms = float((chunk**2).mean().sqrt())
                rms_frames.append(rms)

            mean_rms = sum(rms_frames) / len(rms_frames)
            threshold = mean_rms * _SPEECH_ENERGY_RATIO

            speech_frames = [r for r in rms_frames if r > threshold]
            silence_frames = [r for r in rms_frames if r <= threshold]

            speech_ratio = len(speech_frames) / len(rms_frames) if rms_frames else 0.5
            silence_ratio = 1.0 - speech_ratio

            # SNR estimate
            if silence_frames and speech_frames:
                avg_speech = sum(speech_frames) / len(speech_frames)
                avg_silence = sum(silence_frames) / len(silence_frames)
                snr_db = (
                    20 * math.log10(avg_speech / avg_silence)
                    if avg_silence > 0
                    else 40.0
                )
            else:
                snr_db = 20.0

            return speech_ratio, silence_ratio, snr_db
        except Exception:
            return 0.5, 0.5, 0.0

    @staticmethod
    def _sample_offsets(total_frames: int, sample_rate: int) -> list[int]:
        """
        Frame offsets of the windows to inspect, spread across the recording.

        Returns a single offset (0) when the audio is too short to hold more
        than one window, so short files behave exactly as before.
        """
        window_frames = int(_WINDOW_SECONDS * sample_rate)
        if total_frames <= window_frames:
            return [0]

        n_windows = min(_MAX_WINDOWS, max(2, total_frames // window_frames))
        last_start = total_frames - window_frames
        if n_windows == 1:
            return [0]
        step = last_start / (n_windows - 1)
        # dict.fromkeys de-duplicates while preserving order, which matters when
        # rounding collapses two offsets on a short file.
        return list(dict.fromkeys(int(i * step) for i in range(n_windows)))

    def _speaker_estimate(self, waveform: Any, sample_rate: int) -> tuple[float, int]:
        """Estimate speaker count from MFCC-0 std-dev. Returns (variance, count)."""
        try:
            import torchaudio.transforms as transforms  # noqa: PLC0415

            mfcc_transform = transforms.MFCC(
                sample_rate=sample_rate,
                n_mfcc=13,
                melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 40},
            )
            mfcc = mfcc_transform(waveform.unsqueeze(0))  # (1, n_mfcc, T)
            # First coefficient captures fundamental frequency / vocal tract shape
            mfcc0 = mfcc[0, 0, :]  # (T,)
            std_val = float(mfcc0.std())

            if std_val < _SINGLE_SPEAKER_VAR_THRESHOLD:
                return std_val, 1
            if std_val < _TWO_SPEAKER_VAR_THRESHOLD:
                return std_val, 2
            return std_val, 3
        except Exception:
            return 0.5, 2  # safe fallback

    # ── Decision engine ───────────────────────────────────────────────────────

    def _make_decision(
        self, profile: AudioProfile, mfcc_var: float
    ) -> PipelineDecision:
        reasons: list[str] = []
        savings_s = 0.0

        # ── Diarization decision ──────────────────────────────────────────────
        use_diarization = True
        if (
            profile.estimated_speakers == 1
            and profile.sample_confidence >= _DIAR_SKIP_CONFIDENCE
        ):
            use_diarization = False
            savings_s += _DIAR_TYPICAL_SAVINGS_S
            reasons.append(
                f"Detected single speaker (MFCC-0 variance {mfcc_var:.2f} < "
                f"{_SINGLE_SPEAKER_VAR_THRESHOLD}). Skipping diarization. "
                f"Estimated saving: 9 min."
            )
        else:
            reasons.append(
                f"Diarization enabled: estimated {profile.estimated_speakers} speaker(s) "
                f"(MFCC variance {mfcc_var:.2f}, confidence {profile.sample_confidence:.1f})."
            )

        # ── Alignment decision ────────────────────────────────────────────────
        # wav2vec2 alignment costs ~8% of audio duration on CPU (e.g. ~5 min for 60-min audio).
        # CT2 native word timestamps have ±100ms accuracy vs ±10ms for wav2vec2 — acceptable
        # for business meeting speaker attribution.  Skip alignment for audio >10 min to stay
        # within the 20-min processing target for 60-min recordings.
        use_wav2vec2 = True
        long_audio = profile.duration_s >= 600  # >10 min
        skip_alignment_cond = long_audio

        if skip_alignment_cond:
            use_wav2vec2 = False
            align_saving = round(profile.duration_s * 0.08)
            savings_s += align_saving
            reasons.append(
                f"Audio is {profile.duration_s / 60:.0f} min. "
                f"Using CT2 native word timestamps (±100ms accuracy). "
                f"Estimated saving: {align_saving}s."
            )
        else:
            reasons.append(
                f"wav2vec2 alignment enabled: audio is {profile.duration_s / 60:.1f} min "
                f"(under 10 min threshold)."
            )

        return PipelineDecision(
            use_diarization=use_diarization,
            use_wav2vec2_alignment=use_wav2vec2,
            num_chunks=1,
            reasons=reasons,
            estimated_savings_s=round(savings_s),
            audio_profile=profile,
        )
