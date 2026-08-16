"""
AI Grammar Processor (Optional)

Performs grammar-only cleanup using any OpenAI-compatible LLM.
Requires TE_AI_BASE_URL and TE_AI_MODEL environment variables.

Rules enforced in the prompt:
  - Fix punctuation, capitalization, sentence boundaries ONLY
  - Never change words, technical terms, acronyms, numbers, or proper nouns
  - Never summarize or paraphrase
  - Return ONLY the corrected text

If not configured, this processor is a no-op and logs a warning.
Add to a profile's processors list as "ai_grammar" to enable it.
"""

from __future__ import annotations

import os
from typing import ClassVar

from transcript_engine.logging import get_logger
from transcript_engine.models.transcript import Segment, Transcript, Word
from transcript_engine.processors.base import ProcessorContext
from transcript_engine.processors.registry import register

logger = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a transcript grammar assistant. Your ONLY job is to fix punctuation,
capitalization, and sentence boundaries.

You MUST NOT:
- Change any words
- Summarize or paraphrase
- Change technical terms, acronyms, numbers, or proper nouns
- Add or remove information

You MAY ONLY fix:
- Missing or wrong punctuation (periods, commas, question marks)
- Incorrect capitalization
- Run-on sentences that need a period inserted
- Obvious sentence boundary errors

Return ONLY the corrected text. No explanations. No preamble."""


@register
class AIGrammarProcessor:
    """
    Optional LLM-assisted grammar correction.
    Silently skips if TE_AI_BASE_URL or TE_AI_MODEL are not set.
    """

    name: ClassVar[str] = "ai_grammar"

    def __init__(self) -> None:
        self._base_url = os.environ.get("TE_AI_BASE_URL")
        self._model = os.environ.get("TE_AI_MODEL")
        self._configured = bool(self._base_url and self._model)

        if not self._configured:
            logger.info(
                "AI grammar processor not configured "
                "(set TE_AI_BASE_URL and TE_AI_MODEL to enable). Skipping."
            )

    def process(self, transcript: Transcript, ctx: ProcessorContext) -> Transcript:
        if not self._configured:
            return transcript

        new_segments = tuple(self._process_segment(seg) for seg in transcript.segments)
        changed = sum(
            1
            for old, new in zip(transcript.segments, new_segments, strict=True)
            if old is not new
        )
        logger.info(f"AI grammar: improved {changed}/{len(new_segments)} segments")
        return transcript.with_segments(new_segments)

    def _process_segment(self, segment: Segment) -> Segment:
        try:
            corrected_text = self._call_ai(segment.text)
        except Exception as exc:
            logger.warning(f"AI grammar failed for segment {segment.id[:8]}: {exc}")
            return segment

        if corrected_text == segment.text or not corrected_text.strip():
            return segment

        # Rebuild words: update texts to match corrected output where possible
        new_words = self._align_words(segment, corrected_text)
        return segment.model_copy(update={"text": corrected_text, "words": new_words})

    def _call_ai(self, text: str) -> str:
        """Send text to the configured OpenAI-compatible endpoint."""
        try:
            from openai import OpenAI  # noqa: PLC0415
        except ImportError:
            logger.warning("openai package not installed. Run: pip install openai")
            return text

        client = OpenAI(
            base_url=self._base_url,
            api_key=os.environ.get("TE_AI_API_KEY", "sk-no-key"),
        )
        response = client.chat.completions.create(
            model=self._model or "gpt-4o-mini",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0.0,
            max_tokens=len(text.split()) * 3,
        )
        return response.choices[0].message.content or text

    def _align_words(self, segment: Segment, new_text: str) -> tuple[Word, ...]:
        """Best-effort word alignment after AI correction."""
        new_tokens = new_text.split()
        old_words = list(segment.words)

        if len(new_tokens) == len(old_words):
            return tuple(
                w.model_copy(update={"text": t}) if w.text != t else w
                for w, t in zip(old_words, new_tokens, strict=True)
            )

        # Token count changed — rebuild with interpolated timing
        if not old_words:
            return ()
        start, end = old_words[0].start, old_words[-1].end
        step = (end - start) / max(len(new_tokens), 1)
        return tuple(
            Word(
                text=tok,
                start=round(start + i * step, 3),
                end=round(start + (i + 1) * step, 3),
                speaker_id=old_words[0].speaker_id,
            )
            for i, tok in enumerate(new_tokens)
        )
