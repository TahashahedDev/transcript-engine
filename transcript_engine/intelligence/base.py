"""Base protocol and context for intelligence extractors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from transcript_engine.models.transcript import Transcript

if TYPE_CHECKING:
    from transcript_engine.intelligence.ai_backend import AIBackend


class Extractor(Protocol):
    @property
    def name(self) -> str: ...

    def extract(self, transcript: Transcript, ctx: IntelligenceContext) -> Any: ...


@dataclass
class IntelligenceContext:
    profile_name: str = "generic"
    ai_backend: AIBackend | None = None
