"""Topic extractor — identifies major discussion sections by finding natural boundaries."""

from __future__ import annotations

import re
from collections import Counter
from typing import ClassVar

from transcript_engine.intelligence.base import IntelligenceContext
from transcript_engine.intelligence.models import Topic
from transcript_engine.models.transcript import Segment, Transcript

_LONG_PAUSE = 8.0
_MAX_CHUNK_DURATION = 300.0  # 5 minutes

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "should",
        "could",
        "may",
        "might",
        "shall",
        "can",
        "need",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "and",
        "or",
        "but",
        "not",
        "with",
        "by",
        "from",
        "about",
        "than",
        "so",
        "if",
        "as",
        "we",
        "i",
        "you",
        "he",
        "she",
        "they",
        "what",
        "when",
        "where",
        "who",
        "how",
        "which",
        "well",
        "just",
        "yeah",
        "yes",
        "no",
        "okay",
        "right",
        "um",
        "uh",
        "mm",
        "get",
        "got",
        "go",
        "going",
        "all",
        "some",
        "any",
        "there",
        "here",
        "their",
        "our",
        "my",
        "your",
        "his",
        "her",
        "also",
        "then",
        "now",
        "think",
        "know",
        "like",
        "really",
        "very",
        "pretty",
        "actually",
        "basically",
        "mean",
        "kind",
        "sort",
        "thing",
        "things",
        "way",
        "lot",
        "bit",
        "make",
        "see",
        "look",
        "come",
        "one",
        "two",
        "three",
    }
)

_WORD_RE = re.compile(r"[a-zA-Z]{3,}")


class TopicExtractor:
    name: ClassVar[str] = "topics"

    def extract(self, transcript: Transcript, ctx: IntelligenceContext) -> list[Topic]:
        if not transcript.segments:
            return []

        chunks = _split_into_chunks(transcript)
        topics = []
        for chunk_segs in chunks:
            start = chunk_segs[0].start
            end = chunk_segs[-1].end
            words = _collect_words(chunk_segs)
            keywords = _top_keywords(words)
            label = _make_label(keywords)

            if ctx.ai_backend and ctx.ai_backend.configured:
                chunk_text = " ".join(s.text for s in chunk_segs)
                ai_label = _ai_label(ctx, chunk_text)
                if ai_label:
                    label = ai_label

            topics.append(
                Topic(start=start, end=end, label=label, keywords=tuple(keywords))
            )

        return topics


def _split_into_chunks(transcript: Transcript) -> list[list[Segment]]:
    """Split segments into topic chunks at long pauses or every 5 minutes."""
    segs = list(transcript.segments)
    if not segs:
        # extract() already guards this, but the helper is module-level and
        # must not raise IndexError if called from anywhere else.
        return []
    chunks: list[list[Segment]] = []
    current: list[Segment] = [segs[0]]
    chunk_start = segs[0].start

    for prev, seg in zip(segs, segs[1:], strict=False):
        gap = seg.start - prev.end
        elapsed = seg.end - chunk_start
        if gap >= _LONG_PAUSE or elapsed >= _MAX_CHUNK_DURATION:
            chunks.append(current)
            current = [seg]
            chunk_start = seg.start
        else:
            current.append(seg)

    if current:
        chunks.append(current)

    return chunks


def _collect_words(segs: list[Segment]) -> list[str]:
    words = []
    for seg in segs:
        for m in _WORD_RE.finditer(seg.text):
            w = m.group(0).lower()
            if w not in _STOPWORDS:
                words.append(w)
    return words


def _top_keywords(words: list[str], n: int = 3) -> list[str]:
    if not words:
        return []
    counts = Counter(words)
    return [w for w, _ in counts.most_common(n)]


def _make_label(keywords: list[str]) -> str:
    if not keywords:
        return "General discussion"
    if len(keywords) == 1:
        return f"Discussion about {keywords[0]}"
    return f"Discussion about {keywords[0]}, {keywords[1]}"


def _ai_label(ctx: IntelligenceContext, chunk_text: str) -> str:
    if not ctx.ai_backend:
        return ""
    system = (
        "Summarize the topic of this transcript excerpt in 5 words or fewer. "
        "Return only the topic, no explanation."
    )
    result = ctx.ai_backend.extract_json(system, chunk_text)
    return str(result.get("topic", "")).strip()
