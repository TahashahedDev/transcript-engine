"""MeetingIntelligenceEngine — orchestrates all extractors."""

from __future__ import annotations

from transcript_engine.intelligence.ai_backend import AIBackend
from transcript_engine.intelligence.base import IntelligenceContext
from transcript_engine.intelligence.extractors.action_items import ActionItemExtractor
from transcript_engine.intelligence.extractors.decisions import DecisionExtractor
from transcript_engine.intelligence.extractors.entities import EntityExtractor
from transcript_engine.intelligence.extractors.questions import QuestionExtractor
from transcript_engine.intelligence.extractors.summary import SummaryGenerator
from transcript_engine.intelligence.extractors.timeline import TimelineExtractor
from transcript_engine.intelligence.extractors.topics import TopicExtractor
from transcript_engine.intelligence.models import IntelligenceResult
from transcript_engine.logging import get_logger
from transcript_engine.models.transcript import Transcript

logger = get_logger(__name__)


class MeetingIntelligenceEngine:
    """
    Analyzes a completed Transcript and produces structured knowledge artifacts.
    Never modifies the Transcript. Only produces IntelligenceResult.
    """

    def __init__(self, ai_backend: AIBackend | None = None) -> None:
        self._ai_backend = ai_backend

    def analyze(
        self, transcript: Transcript, profile_name: str = "generic"
    ) -> IntelligenceResult:
        ctx = IntelligenceContext(
            profile_name=profile_name,
            ai_backend=self._ai_backend,
        )

        logger.info(
            f"Intelligence: analyzing {transcript.word_count} words, {len(transcript.segments)} segments"
        )

        action_items = ActionItemExtractor().extract(transcript, ctx)
        decisions = DecisionExtractor().extract(transcript, ctx)
        questions = QuestionExtractor().extract(transcript, ctx)
        entities = EntityExtractor().extract(transcript, ctx)

        # Topics must be extracted before timeline so they can appear as events
        topics = TopicExtractor().extract(transcript, ctx)

        # Timeline runs after decisions, action items, and topics so it can reference all of them
        timeline = TimelineExtractor().extract(
            transcript,
            ctx,
            decisions=decisions,
            action_items=action_items,
            topics=topics,
        )

        summary = SummaryGenerator().extract(
            transcript,
            ctx,
            decisions=decisions,
            action_items=action_items,
            entities=entities,
        )

        logger.info(
            f"Intelligence: {len(action_items)} actions, {len(decisions)} decisions, "
            f"{len(questions)} questions, {entities.total_count} entities, "
            f"{len(topics)} topics"
        )

        return IntelligenceResult(
            summary_bullets=summary,
            action_items=action_items,
            decisions=decisions,
            questions=questions,
            timeline=timeline,
            entities=entities,
            topics=topics,
            ai_enhanced=bool(self._ai_backend and self._ai_backend.configured),
        )
