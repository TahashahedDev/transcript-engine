"""Entity extractor — finds structured entities in the transcript text."""

from __future__ import annotations

import re
from typing import ClassVar

from transcript_engine.intelligence.base import IntelligenceContext
from transcript_engine.intelligence.models import EntityCollection
from transcript_engine.models.transcript import Transcript

_DOLLAR = re.compile(
    r"\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|thousand|k))?"
    r"|\b\d+(?:\.\d+)?\s*(?:million|billion|thousand)\s*dollars?\b",
    re.IGNORECASE,
)
_PCT = re.compile(r"\b\d+(?:\.\d+)?%|\b\d+(?:\.\d+)?\s*percent\b", re.IGNORECASE)
_DATE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September"
    r"|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?\b"
    r"|\b\d{1,2}/\d{1,2}/\d{2,4}\b"
    r"|\bQ[1-4]\s+\d{4}\b",
    re.IGNORECASE,
)
_LOAN_ID = re.compile(r"\b(?:LN|L)\s*[-\s]?\d{4,}\b", re.IGNORECASE)
_TICKET = re.compile(r"\b[A-Z]{2,5}-\d{3,}\b")
_URL = re.compile(r"https?://\S+")
_EMAIL = re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b")
_ACRONYM = re.compile(r"\b[A-Z][A-Z_]{1,}[A-Z]\b|\b[A-Z]{2,}\b")
_ORG = re.compile(
    r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+"
    r"(?:Corp|Inc|LLC|Ltd|Bank|Group|Financial|Services|Solutions|Technologies|Systems)\b"
)
_TITLE_PERSON = re.compile(
    r"\b(?:Mr|Ms|Mrs|Dr|Prof)\.\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b"
)

# Words that look like acronyms but are common English words (exclude from acronym list)
_COMMON_WORDS = frozenset(
    {
        "I",
        "A",
        "IS",
        "IT",
        "IN",
        "AT",
        "TO",
        "DO",
        "BE",
        "BY",
        "AN",
        "OR",
        "IF",
        "OF",
        "AS",
        "ON",
        "NO",
        "OK",
        "GO",
        "UP",
        "US",
        "WE",
        "MY",
        "ME",
    }
)


class EntityExtractor:
    name: ClassVar[str] = "entities"

    def extract(
        self, transcript: Transcript, ctx: IntelligenceContext
    ) -> EntityCollection:
        full_text = " ".join(seg.text for seg in transcript.segments)

        # Known people from speaker map
        known_people = list(transcript.speakers.values())
        # Title-based people
        title_people = [m.group(0) for m in _TITLE_PERSON.finditer(full_text)]
        people = _dedup(known_people + title_people)

        orgs = _dedup(m.group(0) for m in _ORG.finditer(full_text))
        acronyms = _dedup(
            m.group(0)
            for m in _ACRONYM.finditer(full_text)
            if m.group(0) not in _COMMON_WORDS
        )
        dollars = _dedup(m.group(0) for m in _DOLLAR.finditer(full_text))
        pcts = _dedup(m.group(0) for m in _PCT.finditer(full_text))
        dates = _dedup(m.group(0) for m in _DATE.finditer(full_text))
        loans = _dedup(m.group(0) for m in _LOAN_ID.finditer(full_text))
        tickets = _dedup(m.group(0) for m in _TICKET.finditer(full_text))
        urls = _dedup(m.group(0) for m in _URL.finditer(full_text))
        emails = _dedup(m.group(0) for m in _EMAIL.finditer(full_text))

        return EntityCollection(
            people=tuple(people),
            organizations=tuple(orgs),
            acronyms=tuple(acronyms),
            dollar_amounts=tuple(dollars),
            percentages=tuple(pcts),
            dates=tuple(dates),
            loan_ids=tuple(loans),
            ticket_numbers=tuple(tickets),
            urls=tuple(urls),
            emails=tuple(emails),
        )


def _dedup(items: object) -> list[str]:
    from collections.abc import Iterable  # noqa: PLC0415

    if not isinstance(items, Iterable):
        return []
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        s = str(item)
        if s not in seen:
            seen.add(s)
            result.append(s)
    return sorted(result)
