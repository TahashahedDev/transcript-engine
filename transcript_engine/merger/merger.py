from __future__ import annotations

from uuid import uuid4

from transcript_engine.logging import get_logger
from transcript_engine.models.audio import PreparedAudio
from transcript_engine.models.pipeline import (
    DiarizationResult,
    RawTranscription,
    RawWord,
    SpeakerSegment,
)
from transcript_engine.models.transcript import Segment, Transcript, Word

logger = get_logger(__name__)

_UNKNOWN_SPEAKER = "SPEAKER_UNKNOWN"


class TranscriptMerger:
    """
    Combines word-level transcription timestamps with diarization speaker segments
    into a unified Transcript.

    Pure logic — no models, no I/O. Fully unit-testable with synthetic data.
    """

    def merge(
        self,
        transcription: RawTranscription,
        diarization: DiarizationResult,
        audio: PreparedAudio,
    ) -> Transcript:
        logger.info(
            f"Merging {len(transcription.words)} words "
            f"with {len(diarization.segments)} diarization segments"
        )

        # Sort both inputs by start time. _assign_speakers uses a pointer that
        # only ever advances, so a single out-of-order word would be matched
        # against the wrong segment window and silently misattributed. Chunked
        # ASR emits words in order today, but that is an assumption about an
        # upstream component rather than something this function can verify,
        # and sorting ~18K words for a 90-minute meeting is negligible.
        words_with_speakers = self._assign_speakers(
            sorted(transcription.words, key=lambda w: w.start),
            sorted(diarization.segments, key=lambda s: s.start),
        )
        segments = self._group_into_segments(words_with_speakers)
        speakers = self._build_speakers_dict(segments)

        transcript = Transcript(
            audio_path=audio.original_path,
            duration=audio.duration,
            language=transcription.language,
            segments=tuple(segments),
            speakers=speakers,
        )

        logger.info(
            f"Merge complete: {len(segments)} segments, "
            f"{len(speakers)} speaker(s), "
            f"{transcript.word_count} words"
        )
        return transcript

    def _assign_speakers(
        self,
        words: list[RawWord],
        diar_segments: list[SpeakerSegment],
    ) -> list[Word]:
        """
        Assign a speaker_id to each RawWord using an O(n+m) sliding-pointer scan.

        Both words and diar_segments are sorted by start time. The pointer `lo`
        advances past segments that have fully ended before the current word,
        so each segment is visited at most once per word that overlaps it.

        For a 90-minute meeting (18K words, 3K segments) this reduces iterations
        from ~54M (O(n×m) full scan) to ~21K (O(n+m)).
        """
        if not diar_segments:
            return [
                Word(
                    text=w.text, start=w.start, end=w.end,
                    confidence=w.confidence, speaker_id=_UNKNOWN_SPEAKER,
                )
                for w in words
            ]

        assigned: list[Word] = []
        last_known_speaker = _UNKNOWN_SPEAKER
        n_segs = len(diar_segments)
        lo = 0  # first segment that has not yet fully ended

        for raw_word in words:
            word_start = raw_word.start
            word_end = raw_word.end if raw_word.end > raw_word.start else raw_word.start + 0.001

            # Advance lo past segments that ended before this word started.
            while lo < n_segs and diar_segments[lo].end <= word_start:
                lo += 1

            best_speaker = _UNKNOWN_SPEAKER
            best_overlap = 0.0

            # Scan forward from lo; stop once segments start after the word ends.
            for seg in diar_segments[lo:]:
                if seg.start >= word_end:
                    break
                overlap = min(word_end, seg.end) - max(word_start, seg.start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = seg.speaker_id

            if best_speaker == _UNKNOWN_SPEAKER and last_known_speaker != _UNKNOWN_SPEAKER:
                best_speaker = last_known_speaker
            elif best_speaker != _UNKNOWN_SPEAKER:
                last_known_speaker = best_speaker

            assigned.append(
                Word(
                    text=raw_word.text,
                    start=raw_word.start,
                    end=raw_word.end,
                    confidence=raw_word.confidence,
                    speaker_id=best_speaker,
                )
            )

        return assigned

    def _group_into_segments(self, words: list[Word]) -> list[Segment]:
        """
        Group consecutive same-speaker words into Segment objects.
        A segment break occurs on a speaker change.
        """
        if not words:
            return []

        segments: list[Segment] = []
        current_speaker = words[0].speaker_id or _UNKNOWN_SPEAKER
        current_words: list[Word] = []

        for word in words:
            word_speaker = word.speaker_id or _UNKNOWN_SPEAKER
            if word_speaker != current_speaker and current_words:
                segments.append(self._build_segment(current_speaker, current_words))
                current_words = []
                current_speaker = word_speaker
            current_words.append(word)

        if current_words:
            segments.append(self._build_segment(current_speaker, current_words))

        return segments

    def _build_segment(self, speaker_id: str, words: list[Word]) -> Segment:
        text = " ".join(w.text for w in words).strip()
        return Segment(
            id=str(uuid4()),
            speaker_id=speaker_id,
            start=words[0].start,
            end=words[-1].end,
            words=tuple(words),
            text=text,
        )

    def _build_speakers_dict(self, segments: list[Segment]) -> dict[str, str]:
        """
        Assign human-readable display names in order of first appearance:
        SPEAKER_00 → Speaker 1, SPEAKER_01 → Speaker 2, etc.

        The counter deliberately skips the unknown-speaker bucket. Enumerating
        every key instead meant that when any word was unattributed — which
        happens whenever transcription starts before pyannote's first speech
        segment, e.g. a cough or "um" at t=0 — "Unknown" consumed index 1 and
        the transcript displayed "Speaker 2, Speaker 3" with no Speaker 1.
        """
        seen: dict[str, None] = {}
        for seg in segments:
            seen[seg.speaker_id] = None

        speakers: dict[str, str] = {}
        next_index = 1
        for speaker_id in seen:
            if speaker_id == _UNKNOWN_SPEAKER:
                speakers[speaker_id] = "Unknown"
            else:
                speakers[speaker_id] = f"Speaker {next_index}"
                next_index += 1

        return speakers
