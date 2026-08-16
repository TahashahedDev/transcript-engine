"""Question extractor — finds open questions and unanswered items."""

from __future__ import annotations

import re
from typing import ClassVar

from transcript_engine.intelligence.base import IntelligenceContext
from transcript_engine.intelligence.extractors.action_items import _iter_sentences
from transcript_engine.intelligence.models import Question
from transcript_engine.models.transcript import Segment, Transcript

# Very short filler questions to ignore (tag questions and backchannels)
_FILLER_QUESTIONS = frozenset(
    {
        "right",
        "right?",
        "okay",
        "okay?",
        "ok",
        "ok?",
        "yeah",
        "yeah?",
        "yes",
        "yes?",
        "no",
        "no?",
        "really",
        "really?",
        "sure",
        "sure?",
        "got it",
        "got it?",
        "correct",
        "correct?",
        "good",
        "good?",
    }
)

# Minimum character length for a question to be considered meaningful
_MIN_QUESTION_LEN = 20

# Tag question suffixes — sentences ending with these are likely rhetorical
# confirmation-seeking, not information-seeking questions
_TAG_ENDINGS = re.compile(
    r",?\s*(?:right|correct|okay|ok|yeah|isn\'t it|aren\'t they|don\'t you|isn\'t that right|you know)\?$",
    re.IGNORECASE,
)

# WH-words and question-opening modals that mark genuine questions
_QUESTION_OPENERS = frozenset(
    {
        "what",
        "who",
        "where",
        "when",
        "why",
        "how",
        "which",
        "is",
        "are",
        "was",
        "were",
        "can",
        "could",
        "will",
        "would",
        "should",
        "shall",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
    }
)

# Phrases that signal a question was not answered
_UNANSWERED_STARTS = (
    "i don't know",
    "i dont know",
    "not sure",
    "good question",
    "no idea",
    "i'm not sure",
    "im not sure",
)


class QuestionExtractor:
    name: ClassVar[str] = "questions"

    def extract(
        self, transcript: Transcript, ctx: IntelligenceContext
    ) -> list[Question]:
        # Collect all segments with their indices for answer-detection
        segments = list(transcript.segments)
        results: list[Question] = []
        seen: set[str] = set()

        for seg_idx, segment in enumerate(segments):
            for sentence, ts in _iter_sentences(segment):
                if not sentence.rstrip().endswith("?"):
                    continue
                # Filter out filler/tag questions ("Right?", "Yeah?", "...right?", etc.)
                bare = sentence.rstrip("? ").lower().strip()
                if bare in _FILLER_QUESTIONS or len(sentence) < _MIN_QUESTION_LEN:
                    continue
                # Detect tag questions: "..., right?" where the core sentence is not a question
                tag_stripped = _TAG_ENDINGS.sub("", sentence.rstrip()).strip()
                if tag_stripped and not tag_stripped.rstrip().endswith("?"):
                    first_word = tag_stripped.split()[0].lower().rstrip("?,;:")
                    if first_word not in _QUESTION_OPENERS:
                        continue  # tag question — skip it
                key = sentence.lower().strip()
                if key in seen:
                    continue
                seen.add(key)

                is_answered = _detect_answer(segments, seg_idx, ts)

                results.append(
                    Question(
                        text=sentence[:200],
                        speaker_id=segment.speaker_id,
                        timestamp=ts,
                        confidence=0.95,
                        reason="Sentence ends with question mark",
                        question=sentence.strip(),
                        is_answered=is_answered,
                    )
                )

        return results


def _detect_answer(
    segments: list[Segment], question_seg_idx: int, question_ts: float
) -> bool:
    """
    Heuristic: scan the next segment by a different speaker within 30 seconds.
    If their first sentence starts with an "I don't know" phrase → unanswered.
    Otherwise → answered.
    """
    question_speaker = segments[question_seg_idx].speaker_id
    for seg in segments[question_seg_idx + 1 :]:
        if seg.start - question_ts > 30.0:
            break
        if seg.speaker_id == question_speaker:
            continue
        # Different speaker responded; answered unless they express uncertainty
        first_text = seg.text.lower().strip()
        return not any(first_text.startswith(phrase) for phrase in _UNANSWERED_STARTS)
    return False  # No response found
