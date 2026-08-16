from __future__ import annotations

from pydantic import BaseModel, Field


class VocabularyEntry(BaseModel, frozen=True):
    canonical: str
    aliases: list[str] = []
    confidence: float = 0.9
    context: list[str] = []
    fuzzy: bool = False
    case_sensitive: bool = False


class Profile(BaseModel, frozen=True):
    name: str
    description: str = ""
    industry: str = "generic"
    vocabulary: tuple[VocabularyEntry, ...] = ()
    processors: list[str] = Field(
        default_factory=lambda: [
            "vocabulary_correction",
            "context_correction",
            "transcript_cleanup",
            "speaker_formatting",
        ]
    )
