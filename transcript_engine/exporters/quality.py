"""
Quality Report Exporter — meeting.quality.md

Generates a human-readable review guide flagging areas that need
manual inspection: low-confidence segments, potential proper nouns,
unknown acronyms, and repeated uncertain words.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import UTC, datetime

from transcript_engine.models.correction import CorrectionRecord
from transcript_engine.models.transcript import Transcript, Word

_ACRONYM = re.compile(r"^[A-Z]{2,}$")
_PROPER_NOUN = re.compile(r"^[A-Z][a-z]{2,}$")
_TRAILING_PUNCT = re.compile(r"[.,;:!?]+$")

# Common English words that start with a capital letter only because they appear
# at the beginning of a sentence — not actual proper nouns.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "The",
        "This",
        "That",
        "These",
        "Those",
        "There",
        "Their",
        "They",
        "Them",
        "Then",
        "When",
        "Where",
        "What",
        "Which",
        "Who",
        "How",
        "Why",
        "It",
        "Its",
        "In",
        "If",
        "Is",
        "Into",
        "And",
        "But",
        "For",
        "Or",
        "Not",
        "No",
        "Yet",
        "So",
        "Has",
        "Have",
        "Had",
        "Her",
        "Him",
        "His",
        "We",
        "You",
        "He",
        "She",
        "Our",
        "Your",
        "Was",
        "Were",
        "Will",
        "With",
        "Would",
        "Can",
        "Could",
        "As",
        "At",
        "Be",
        "By",
        "Do",
        "Did",
        "Does",
        "Also",
        "Just",
        "Very",
        "All",
        "Any",
        "Some",
        "From",
        "Upon",
        "Over",
        "Under",
        "About",
        "Before",
        "After",
        "During",
        "While",
        "Since",
        "Until",
        "Both",
        "Either",
        "One",
        "Two",
        "Three",
        "Four",
        "Five",
        "Six",
        "Please",
        "Thank",
        "Thanks",
    }
)


def export_quality(
    transcript: Transcript,
    corrections: list[CorrectionRecord],
    profile_name: str,
    low_conf_threshold: float = 0.65,
) -> str:
    """Return a quality report Markdown string."""
    words = [w for seg in transcript.segments for w in seg.words]
    confidences = [w.confidence for w in words if w.confidence is not None]
    avg_conf = sum(confidences) / len(confidences) if confidences else None
    low_conf_words = [
        w
        for w in words
        if w.confidence is not None and w.confidence < low_conf_threshold
    ]
    low_conf_pct = len(low_conf_words) / max(len(words), 1) * 100

    low_conf_segs = _low_confidence_segments(transcript, low_conf_threshold)
    potential_nouns = _find_potential_proper_nouns(words, transcript)
    unknown_acronyms = _find_unknown_acronyms(words, corrections)
    repeated_uncertain = _find_repeated_uncertain(low_conf_words)

    lines: list[str] = [
        "# Quality Report",
        "",
        f"**Audio:** {transcript.audio_path.name}  ",
        f"**Profile:** {profile_name}  ",
        f"**Generated:** {datetime.now(UTC).strftime('%Y-%m-%d')}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Words | {transcript.word_count} |",
        f"| Speakers | {len(transcript.speakers)} |",
        f"| Segments | {len(transcript.segments)} |",
        f"| Corrections applied | {len(corrections)} |",
        f"| Avg confidence | {f'{avg_conf:.0%}' if avg_conf is not None else 'N/A'} |",
        f"| Low-confidence words | {len(low_conf_words)} ({low_conf_pct:.1f}%) |",
        "",
    ]

    # Low-confidence segments
    lines += ["## Low-Confidence Segments", ""]
    if low_conf_segs:
        lines += [
            "The following time ranges contain multiple uncertain words "
            "and may need manual review:",
            "",
        ]
        for start, _end, text_preview in low_conf_segs[:10]:
            ts = f"[{int(start)//60:02d}:{int(start)%60:02d}]"
            lines.append(f"- {ts} `{text_preview}`")
        lines.append("")
    else:
        lines += ["No significant low-confidence segments detected.", ""]

    # Corrections applied
    lines += ["## Corrections Applied", ""]
    if corrections:
        lines += [
            "| Original | Replacement | Confidence |",
            "|----------|-------------|------------|",
        ]
        for rec in corrections:
            lines.append(
                f"| {rec.original} | {rec.replacement} | {rec.confidence:.0%} |"
            )
        lines.append("")
    else:
        lines += ["No vocabulary corrections were applied.", ""]

    # Potential proper nouns
    lines += ["## Potential Proper Nouns", ""]
    if potential_nouns:
        lines += [
            "These capitalized words appear frequently and may be proper nouns "
            "that could benefit from vocabulary entries:",
            "",
        ]
        for word, count in potential_nouns[:15]:
            lines.append(f"- **{word}** (×{count})")
        lines.append("")
    else:
        lines += ["No unusual proper nouns detected.", ""]

    # Unknown acronyms
    lines += ["## Potential Unknown Acronyms", ""]
    if unknown_acronyms:
        lines += [
            "These all-caps sequences were not matched by any vocabulary rule "
            "and may need to be added to the profile:",
            "",
        ]
        for word, count in unknown_acronyms[:15]:
            lines.append(f"- **{word}** (×{count})")
        lines.append("")
    else:
        lines += ["No unknown acronyms detected.", ""]

    # Repeated uncertain words
    lines += ["## Repeated Uncertain Words", ""]
    if repeated_uncertain:
        lines += [
            "These words appeared multiple times with low confidence "
            "and may be systematic mishearings:",
            "",
        ]
        for word, count in repeated_uncertain[:10]:
            lines.append(f"- `{word}` (×{count} times, low confidence)")
        lines.append("")
    else:
        lines += ["No repeated uncertain words detected.", ""]

    return "\n".join(lines)


def _low_confidence_segments(
    transcript: Transcript,
    threshold: float,
    min_low_words: int = 2,
) -> list[tuple[float, float, str]]:
    """Find segments that contain at least min_low_words low-confidence words."""
    results = []
    for seg in transcript.segments:
        low = [
            w
            for w in seg.words
            if w.confidence is not None and w.confidence < threshold
        ]
        if len(low) >= min_low_words:
            preview = seg.text[:80] + ("..." if len(seg.text) > 80 else "")
            results.append((seg.start, seg.end, preview))
    return results


def _find_potential_proper_nouns(
    words: list[Word], transcript: Transcript
) -> list[tuple[str, int]]:
    """Find mid-sentence capitalized words that aren't in the speakers dict."""
    known_names: set[str] = set()
    for name in transcript.speakers.values():
        known_names.update(name.lower().split())

    counter: Counter[str] = Counter()
    for word in words:
        bare = _TRAILING_PUNCT.sub("", word.text)
        if (
            _PROPER_NOUN.match(bare)
            and bare not in _STOPWORDS
            and bare.lower() not in known_names
        ):
            counter[bare] += 1

    return [(w, c) for w, c in counter.most_common(20) if c >= 2]


def _find_unknown_acronyms(
    words: list[Word],
    corrections: list[CorrectionRecord],
) -> list[tuple[str, int]]:
    """Find ALL-CAPS words not covered by any correction."""
    corrected_originals = {rec.original.upper() for rec in corrections}
    corrected_replacements = {rec.replacement.upper() for rec in corrections}
    known = corrected_originals | corrected_replacements

    counter: Counter[str] = Counter()
    for word in words:
        bare = _TRAILING_PUNCT.sub("", word.text)
        if _ACRONYM.match(bare) and bare not in known:
            counter[bare] += 1

    return [(w, c) for w, c in counter.most_common(20) if c >= 1]


def _find_repeated_uncertain(
    low_conf_words: list[Word],
    min_count: int = 2,
) -> list[tuple[str, int]]:
    """Find words that appear multiple times with low confidence."""
    counter: Counter[str] = Counter()
    for word in low_conf_words:
        bare = _TRAILING_PUNCT.sub("", word.text).lower()
        if len(bare) > 2:
            counter[bare] += 1
    return [(w, c) for w, c in counter.most_common(15) if c >= min_count]
