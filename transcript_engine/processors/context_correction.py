"""
Context Correction Processor

Corrects terms that cannot be resolved by alias matching alone.
Uses a sliding window over consecutive words to detect context clues.

Rules are data-driven: each rule defines a target pattern and context signals.
If context confidence is below threshold, the original text is preserved.
Never hallucinate — when in doubt, do nothing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import ClassVar

from transcript_engine.logging import get_logger
from transcript_engine.models.transcript import Segment, Transcript, Word
from transcript_engine.processors.base import ProcessorContext
from transcript_engine.processors.registry import register

logger = get_logger(__name__)

_WINDOW = 8


@dataclass
class ContextRule:
    """
    A rule that replaces `target_pattern` with `canonical` when at least
    one of the `context_signals` is found in the surrounding word window.
    """

    canonical: str
    target_patterns: list[str]
    context_signals: list[str]
    compiled_targets: list[re.Pattern[str]] = field(default_factory=list, repr=False)
    compiled_signals: list[re.Pattern[str]] = field(default_factory=list, repr=False)


_RULES: list[ContextRule] = [
    ContextRule(
        canonical="APPL_ID",
        target_patterns=[r"\bApple\s*I[Dd]?\b", r"\bAPL\s*ID\b", r"\bapp\s*I\b"],
        context_signals=[
            r"\bAL\b",
            r"\bLN\b",
            r"\bCBR\b",
            r"\bMCC\b",
            r"\bWCA\b",
            r"\bWL\b",
        ],
    ),
    ContextRule(
        canonical="LN",
        target_patterns=[r"\bLane\b", r"\bEllen\b"],
        context_signals=[
            r"\bloan\b",
            r"\bservice\b",
            r"\bservicing\b",
            r"\bsystem\b",
            r"\bAL\b",
        ],
    ),
    ContextRule(
        canonical="CBR",
        target_patterns=[r"\bCyber\b", r"\bSidebar\b", r"\bCiber\b"],
        context_signals=[r"\bloan\s+servicing\b", r"\bLN\b", r"\bCBR\b"],
    ),
    ContextRule(
        canonical="BCC",
        target_patterns=[r"\bPCC\b", r"\bPCS\b"],
        context_signals=[r"\bbusiness\b", r"\bcredit\b", r"\bcenter\b"],
    ),
]


def _compile_rules() -> None:
    for rule in _RULES:
        rule.compiled_targets = [
            re.compile(p, re.IGNORECASE) for p in rule.target_patterns
        ]
        rule.compiled_signals = [
            re.compile(p, re.IGNORECASE) for p in rule.context_signals
        ]


_compile_rules()


@register
class ContextCorrectionProcessor:
    """
    Applies context-aware corrections using a sliding word window.

    A correction is only applied when at least one context signal
    is present in the surrounding window of words.
    """

    name: ClassVar[str] = "context_correction"

    def __init__(self) -> None:
        pass

    def process(self, transcript: Transcript, ctx: ProcessorContext) -> Transcript:
        all_words = [w for seg in transcript.segments for w in seg.words]
        corrected_words = self._apply_context_rules(all_words)

        word_iter = iter(corrected_words)
        new_segments: list[Segment] = []

        for segment in transcript.segments:
            seg_words = tuple(next(word_iter) for _ in segment.words)
            new_text = " ".join(w.text for w in seg_words).strip()
            if new_text != segment.text:
                new_segments.append(
                    segment.model_copy(update={"words": seg_words, "text": new_text})
                )
            else:
                new_segments.append(segment)

        return transcript.with_segments(tuple(new_segments))

    def _apply_context_rules(self, words: list[Word]) -> list[Word]:
        result = list(words)

        for i, word in enumerate(result):
            window_start = max(0, i - _WINDOW)
            window_end = min(len(result), i + _WINDOW + 1)
            context_text = " ".join(w.text for w in result[window_start:window_end])

            for rule in _RULES:
                matched_target = self._matches_any(word.text, rule.compiled_targets)
                if not matched_target:
                    continue

                signal_found = any(
                    sig.search(context_text) for sig in rule.compiled_signals
                )
                if signal_found and word.text != rule.canonical:
                    logger.debug(
                        f"Context correction: '{word.text}' → '{rule.canonical}' "
                        f"(context window: '...{context_text[:60]}...')"
                    )
                    result[i] = word.model_copy(update={"text": rule.canonical})
                    break

        return result

    def _matches_any(self, text: str, patterns: list[re.Pattern[str]]) -> bool:
        return any(p.search(text) for p in patterns)
