"""Summary generator — produces bullet-point meeting summaries."""

from __future__ import annotations

import re
from collections import Counter
from typing import ClassVar

from transcript_engine.intelligence.base import IntelligenceContext
from transcript_engine.intelligence.models import ActionItem, Decision, EntityCollection
from transcript_engine.models.transcript import Transcript

_STOPWORDS = frozenset(
    {
        # Articles, conjunctions, prepositions
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "into",
        "onto",
        "over",
        "under",
        "about",
        "if",
        "as",
        "so",
        "then",
        "when",
        "what",
        "which",
        "where",
        "while",
        "because",
        "although",
        "though",
        "since",
        "after",
        "before",
        "some",
        # Auxiliary verbs
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
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        # Pronouns
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "we",
        "i",
        "you",
        "he",
        "she",
        "they",
        "them",
        "their",
        "our",
        "my",
        "your",
        "his",
        "her",
        # Common verbs (too generic to be topics)
        "not",
        "no",
        "go",
        "going",
        "just",
        "all",
        "up",
        "out",
        "know",
        "think",
        "want",
        "need",
        "make",
        "take",
        "come",
        "look",
        "talk",
        "work",
        "feel",
        "mean",
        "said",
        "says",
        "say",
        # Common adverbs/adjectives
        "well",
        "also",
        "more",
        "much",
        "here",
        "there",
        "back",
        "down",
        "good",
        "kind",
        "okay",
        "even",
        "way",
        "time",
        "next",
        "first",
        "last",
        "many",
        "most",
        "such",
        "both",
        "each",
        "very",
        "only",
        "than",  # Filler words
        "yeah",
        "yes",
        "ok",
        "um",
        "uh",
        "mm",
        "right",
        "like",
    }
)

_AI_SUMMARY_PROMPT = """\
You are a meeting summarizer. Given a meeting transcript, produce a structured bullet-point summary.

Return bullet points in this order:
1. First 1-2 bullets: what the meeting was about (main purpose, who was involved, key topics).
2. Next up to 3 bullets: key decisions made — prefix each with "**Decision:**".
3. Next up to 3 bullets: action items and next steps — prefix each with "**Action:**" and include who is responsible.
4. Finally 1-2 bullets: important context, open questions, or risks — only if present in the transcript.

Rules:
- Only include facts stated in the transcript. Do not infer or hallucinate.
- Each bullet must be a complete, specific sentence. Avoid vague statements.
- Return ONLY bullet points, one per line, starting with "- ".
- Omit sections that have nothing to report.
- Total: 5-10 bullets. Be specific and concise.

TRANSCRIPT:
{transcript_text}"""


class SummaryGenerator:
    name: ClassVar[str] = "summary"

    def extract(
        self,
        transcript: Transcript,
        ctx: IntelligenceContext,
        decisions: list[Decision] | None = None,
        action_items: list[ActionItem] | None = None,
        entities: EntityCollection | None = None,
    ) -> list[str]:
        if ctx.ai_backend and ctx.ai_backend.configured:
            bullets = self._ai_summary(transcript, ctx)
            if bullets:
                return bullets[:10]

        return self._deterministic_summary(
            transcript, decisions or [], action_items or []
        )

    def _ai_summary(
        self, transcript: Transcript, ctx: IntelligenceContext
    ) -> list[str]:
        transcript_text = "\n".join(
            f"{transcript.display_name(seg.speaker_id)}: {seg.text}"
            for seg in transcript.segments
        )
        assert ctx.ai_backend is not None
        response = ctx.ai_backend.complete(
            system_prompt=_AI_SUMMARY_PROMPT.split("TRANSCRIPT:")[0].strip(),
            user_text=f"TRANSCRIPT:\n{transcript_text}",
            max_tokens=512,
        )
        bullets = []
        for line in response.splitlines():
            line = line.strip()
            if line.startswith("- "):
                bullets.append(line[2:].strip())
            elif line:
                bullets.append(line)
        return bullets

    def _deterministic_summary(
        self,
        transcript: Transcript,
        decisions: list[Decision],
        action_items: list[ActionItem],
    ) -> list[str]:
        bullets: list[str] = []

        # Context: meeting participants and duration
        mins = int(transcript.duration) // 60
        secs = int(transcript.duration) % 60
        duration_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
        if len(transcript.speakers) == 2:
            names = [transcript.display_name(s) for s in sorted(transcript.speakers)]
            bullets.append(f"{names[0]} and {names[1]} met for {duration_str}.")
        elif len(transcript.speakers) > 2:
            n = len(transcript.speakers)
            bullets.append(f"Meeting with {n} participants, {duration_str} long.")
        else:
            bullets.append(f"Single-speaker recording, {duration_str} long.")

        # Top topics (lead with context)
        all_words: list[str] = []
        for seg in transcript.segments:
            for word in re.findall(r"\b[a-zA-Z]{4,}\b", seg.text):
                if word.lower() not in _STOPWORDS:
                    all_words.append(word.lower())
        top = [w for w, _ in Counter(all_words).most_common(10) if w not in _STOPWORDS][
            :5
        ]
        if top:
            bullets.append(f"Topics discussed: {', '.join(top)}.")

        # Decisions
        for dec in decisions[:3]:
            text = dec.decision.rstrip(".")
            bullets.append(f"**Decision:** {text}.")

        # Action items
        for item in action_items[:5]:
            owner = item.owner or "Team"
            task = item.task.rstrip(".")
            due = f" by {item.due_date}" if item.due_date else ""
            bullets.append(f"**Action — {owner}:** {task}{due}.")

        return bullets[:10]
