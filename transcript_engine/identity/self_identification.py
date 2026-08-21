"""
Detects self-introduction statements in transcript text — "I'm Neel", "this is
Mahar", "John speaking" — as opposed to mere name mentions like "Neel will
join tomorrow".

This is the automatic identity-anchor discovery mechanism from the mission
brief: it lets Echo assign a real name to a diarized speaker cluster from
existing meeting audio, without ever asking anyone for a separate enrollment
recording.

A pattern only counts as self-identification if it is a first-person
self-reference construction. "This is Neel's report" is excluded because the
possessive breaks the pattern; "Neel will join tomorrow" is excluded because
none of these patterns match a bare third-person subject. This is a
deliberately conservative rule set, not a general coreference resolver: it
will miss self-introductions phrased unusually, but it will not confidently
mislabel a third-person mention as a self-introduction, which is the failure
mode the mission brief calls out as unacceptable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Confidence = Literal["high", "medium"]

# HIGH: the verb itself is unambiguously first-person ("I'm", "I am", "my name
# is") — there is no reading of these constructions where the named person is
# someone other than the speaker.
_HIGH_PATTERNS = [
    re.compile(r"\bI'?m\s+([A-Z][a-zA-Z]+)\b"),
    re.compile(r"\bI\s+am\s+([A-Z][a-zA-Z]+)\b"),
    re.compile(r"\bmy\s+name\s+is\s+([A-Z][a-zA-Z]+)\b", re.IGNORECASE),
    re.compile(r"\bmy\s+name'?s\s+([A-Z][a-zA-Z]+)\b", re.IGNORECASE),
]

# MEDIUM: these constructions are usually self-introductions in meeting
# transcripts ("Neel here", "Mahar speaking", "this is John") but "this is X"
# in particular can also introduce an object ("this is the report"), so these
# require the captured word to look like a name (capitalized, not a stopword)
# and are scored lower than the HIGH tier.
_MEDIUM_PATTERNS = [
    re.compile(r"^\s*([A-Z][a-zA-Z]+)\s+here\b"),
    re.compile(r"^\s*([A-Z][a-zA-Z]+)\s+speaking\b"),
    # [Tt]his rather than re.IGNORECASE: IGNORECASE would also relax the
    # captured group's [A-Z] requirement, defeating the "looks like a name"
    # capitalization check this pattern relies on to reject "this is fine".
    re.compile(r"\b[Tt]his\s+is\s+([A-Z][a-zA-Z]+)\b(?!\s*(?:'s|the|a|an)\b)"),
]

# Capitalized words after "this is" / at segment start that are not names.
# Not exhaustive by construction — this is a stopword filter, not a named-
# entity model, so it will not catch every false positive on real audio.
_NOT_A_NAME = frozenset(
    {
        "The", "This", "That", "There", "It", "So", "Just", "Actually",
        "Basically", "Also", "Well", "Okay", "Ok", "Right", "Sure", "Yes",
        "No", "Great", "Good", "Fine", "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday", "January", "February",
        "March", "April", "May", "June", "July", "August", "September",
        "October", "November", "December",
    }
)


@dataclass(frozen=True)
class SelfIdentification:
    """One candidate self-introduction found in a segment's text."""

    name: str
    confidence: Confidence
    matched_text: str
    char_offset: int


def extract_self_identifications(text: str) -> list[SelfIdentification]:
    """
    Find self-introduction statements in a single speaker's utterance.

    Callers are expected to pass the text of one Segment at a time (all of
    which already belongs to one speaker_id per the Segment model) — this
    function does not do speaker attribution itself, it only decides whether
    the text contains a first-person self-introduction at all.
    """
    found: list[SelfIdentification] = []
    seen_spans: set[tuple[int, int]] = set()

    for pattern in _HIGH_PATTERNS:
        for m in pattern.finditer(text):
            span = m.span()
            if span in seen_spans:
                continue
            seen_spans.add(span)
            found.append(
                SelfIdentification(
                    name=m.group(1),
                    confidence="high",
                    matched_text=m.group(0),
                    char_offset=m.start(),
                )
            )

    for pattern in _MEDIUM_PATTERNS:
        for m in pattern.finditer(text):
            span = m.span()
            if span in seen_spans:
                continue
            name = m.group(1)
            if name in _NOT_A_NAME:
                continue
            seen_spans.add(span)
            found.append(
                SelfIdentification(
                    name=name,
                    confidence="medium",
                    matched_text=m.group(0),
                    char_offset=m.start(),
                )
            )

    found.sort(key=lambda s: s.char_offset)
    return found
