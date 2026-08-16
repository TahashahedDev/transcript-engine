"""
Review Report Generator — meeting.review.md

Produces a human review guide that surfaces only the parts of a transcript
that need manual attention. Reviewers can verify a 60-minute meeting in
5–10 minutes by focusing on flagged items rather than rereading everything.
"""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import UTC, datetime

from transcript_engine.intelligence.extractors.entities import (
    _DATE,
    _DOLLAR,
    _EMAIL,
    _LOAN_ID,
    _PCT,
    _TICKET,
    _URL,
)
from transcript_engine.models.correction import CorrectionRecord
from transcript_engine.models.transcript import Transcript, Word

_BARE_NUMBER = re.compile(r"\b\d{4,}\b")
_TITLE_WORD = re.compile(r"^[A-Z][a-z]{2,}$")


def generate_review(
    transcript: Transcript,
    corrections: list[CorrectionRecord],
    low_conf_threshold: float = 0.65,
    short_segment_threshold: float = 1.5,
    rapid_switch_window: float = 30.0,
    rapid_switch_count: int = 4,
) -> str:
    seg_index = {seg.id: seg for seg in transcript.segments}
    date_str = datetime.now(UTC).strftime("%Y-%m-%d")

    lines: list[str] = [
        f"# Review Report — {transcript.audio_path.name}",
        "",
        f"> Generated {date_str}. Confidence threshold: {low_conf_threshold:.0%}.",
        "",
    ]

    # ── Section 1: Low-Confidence Words ──────────────────────────────────────
    lines += ["## 1. Low-Confidence Words", ""]
    low_words = _collect_low_confidence_words(transcript, low_conf_threshold)
    if low_words:
        lines.append(
            f"{len(low_words)} word(s) below {low_conf_threshold:.0%} confidence."
        )
        lines.append("")
        lines += [
            "| Time | Speaker | Word | Confidence | Context |",
            "|------|---------|------|------------|---------|",
        ]
        shown = low_words[:50]
        for ts, speaker, word_text, conf, context in shown:
            conf_str = f"{conf:.0%}"
            context_safe = context.replace("|", "\\|")[:100]
            lines.append(
                f"| {_fmt(ts)} | {speaker} | {word_text} | {conf_str} | {context_safe} |"
            )
        if len(low_words) > 50:
            lines.append(f"\n…and {len(low_words) - 50} more.")
    else:
        lines.append(f"No words below {low_conf_threshold:.0%} confidence.")
    lines.append("")

    # ── Section 2: Low-Confidence Segments ───────────────────────────────────
    lines += ["## 2. Low-Confidence Segments", ""]
    low_segs = _collect_low_confidence_segments(transcript, low_conf_threshold)
    if low_segs:
        lines += [
            "| Time | Speaker | Avg Confidence | Preview |",
            "|------|---------|----------------|---------|",
        ]
        for seg_ts, seg_speaker, seg_avg_conf, seg_preview in low_segs:
            lines.append(
                f"| {_fmt(seg_ts)} | {seg_speaker} | {seg_avg_conf:.0%} | {seg_preview} |"
            )
    else:
        lines.append("No low-confidence segments detected.")
    lines.append("")

    # ── Section 3: Corrections Applied ───────────────────────────────────────
    lines += ["## 3. Corrections Applied", ""]
    if corrections:
        lines += [
            "| Original | Replacement | Reason | Profile | Confidence | Time |",
            "|----------|-------------|--------|---------|------------|------|",
        ]
        for rec in corrections:
            rec_seg = seg_index.get(rec.segment_id)
            rec_ts = _fmt(rec_seg.start) if rec_seg else "—"
            lines.append(
                f"| {rec.original} | {rec.replacement} | {rec.reason} "
                f"| {rec.profile} | {rec.confidence:.0%} | {rec_ts} |"
            )
    else:
        lines.append("No corrections were applied.")
    lines.append("")

    # ── Section 4: Potential Proper Nouns ────────────────────────────────────
    lines += ["## 4. Potential Proper Nouns", ""]
    proper_nouns = _find_proper_noun_candidates(transcript, low_conf_threshold)
    if proper_nouns:
        lines += [
            "The following capitalized terms appear only once or with low confidence.",
            "Review to verify they are correctly transcribed.",
            "",
            "| Word | Occurrences | Avg Confidence | First Seen |",
            "|------|-------------|----------------|------------|",
        ]
        for pn_word, pn_count, pn_avg_conf, pn_first_ts in proper_nouns[:30]:
            conf_str = f"{pn_avg_conf:.0%}" if pn_avg_conf is not None else "—"
            lines.append(
                f"| {pn_word} | {pn_count} | {conf_str} | {_fmt(pn_first_ts)} |"
            )
    else:
        lines.append("No unusual proper nouns detected.")
    lines.append("")

    # ── Section 5: Numbers & Codes ────────────────────────────────────────────
    lines += ["## 5. Numbers & Codes", ""]
    _append_entity_section(lines, "Dollar Amounts", _DOLLAR, transcript)
    _append_entity_section(lines, "Percentages", _PCT, transcript)
    _append_entity_section(lines, "Dates", _DATE, transcript)
    _append_entity_section(lines, "Loan IDs", _LOAN_ID, transcript)
    _append_entity_section(lines, "Ticket Numbers", _TICKET, transcript)
    _append_entity_section(lines, "URLs", _URL, transcript)
    _append_entity_section(lines, "Emails", _EMAIL, transcript)
    _append_bare_numbers_section(lines, transcript)

    # ── Section 6: Possible Speaker Issues ────────────────────────────────────
    lines += ["## 6. Possible Speaker Issues", ""]
    issues = _detect_speaker_issues(
        transcript,
        short_threshold=short_segment_threshold,
        window=rapid_switch_window,
        window_count=rapid_switch_count,
    )
    if issues:
        lines += [
            "| Type | Time Range | Description |",
            "|------|------------|-------------|",
        ]
        for issue_type, time_range, desc in issues:
            lines.append(f"| {issue_type} | {time_range} | {desc} |")
    else:
        lines.append("No speaker issues detected.")
    lines.append("")

    return "\n".join(lines)


# ── helpers ────────────────────────────────────────────────────────────────────


def _fmt(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"[{m:02d}:{s:02d}]"


def _find_sentence_context(word: Word, words_in_seg: tuple[Word, ...]) -> str:
    """Return the sentence containing `word` from the segment's word list."""
    sentence_words: list[str] = []
    in_sentence = False
    for w in words_in_seg:
        if not in_sentence:
            sentence_words = [w.text]
            if w is word:
                in_sentence = True
        else:
            sentence_words.append(w.text)
        # sentence ends at word with .!?
        if re.search(r"[.!?]$", w.text):
            if in_sentence:
                return " ".join(sentence_words)
            sentence_words = []
    return " ".join(sentence_words)


def _collect_low_confidence_words(
    transcript: Transcript,
    threshold: float,
) -> list[tuple[float, str, str, float, str]]:
    """Return (timestamp, speaker_display, word_text, confidence, context)."""
    results = []
    for seg in transcript.segments:
        display = transcript.display_name(seg.speaker_id)
        for word in seg.words:
            if word.confidence is not None and word.confidence < threshold:
                context = _find_sentence_context(word, seg.words) or seg.text[:100]
                results.append(
                    (word.start, display, word.text, word.confidence, context)
                )
    return results


def _collect_low_confidence_segments(
    transcript: Transcript,
    threshold: float,
) -> list[tuple[float, str, float, str]]:
    """Return (timestamp, speaker_display, avg_confidence, preview)."""
    results = []
    for seg in transcript.segments:
        confs = [w.confidence for w in seg.words if w.confidence is not None]
        if not confs:
            continue
        avg = sum(confs) / len(confs)
        if avg < threshold:
            display = transcript.display_name(seg.speaker_id)
            preview = (seg.text[:80] + "…") if len(seg.text) > 80 else seg.text
            preview = preview.replace("|", "\\|")
            results.append((seg.start, display, avg, preview))
    return results


def _find_proper_noun_candidates(
    transcript: Transcript,
    threshold: float,
) -> list[tuple[str, int, float | None, float]]:
    """Return (word, count, avg_confidence, first_timestamp) for potential proper nouns."""
    # Known speaker names
    known_names: set[str] = set()
    for name in transcript.speakers.values():
        known_names.update(n.lower() for n in name.split())

    # Track every occurrence of every word (both capitalized and lowercase)
    word_data: dict[str, list[tuple[float | None, float]]] = defaultdict(list)
    lowercase_seen: set[str] = set()

    for seg in transcript.segments:
        for word in seg.words:
            bare = re.sub(r"[.,!?;:]+$", "", word.text)
            if not bare:
                continue
            if bare[0].islower():
                lowercase_seen.add(bare.lower())
            if _TITLE_WORD.match(bare):
                word_data[bare].append((word.confidence, seg.start))

    results = []
    for wtext, occurrences in word_data.items():
        if wtext.lower() in known_names:
            continue
        if wtext.lower() in lowercase_seen:
            continue  # appears in lowercase → probably not a proper noun

        count = len(occurrences)
        confs = [c for c, _ in occurrences if c is not None]
        pn_avg_conf: float | None = sum(confs) / len(confs) if confs else None
        first_ts = occurrences[0][1]

        if count == 1 or (pn_avg_conf is not None and pn_avg_conf < threshold):
            results.append((wtext, count, pn_avg_conf, first_ts))

    return sorted(results, key=lambda x: x[0])


def _append_entity_section(
    lines: list[str],
    title: str,
    pattern: re.Pattern,  # type: ignore[type-arg]
    transcript: Transcript,
) -> None:
    matches = _find_entities_with_context(transcript, pattern)
    lines.append(f"### {title}")
    lines.append("")
    if matches:
        lines += [
            "| Value | Time | Speaker |",
            "|-------|------|---------|",
        ]
        for value, ts, speaker in matches:
            lines.append(f"| {value} | {_fmt(ts)} | {speaker} |")
    else:
        lines.append(f"*No {title.lower()} found.*")
    lines.append("")


def _append_bare_numbers_section(lines: list[str], transcript: Transcript) -> None:
    # Numbers already caught by other patterns (loan IDs include digits, etc.)
    caught: set[str] = set()
    full_text = " ".join(seg.text for seg in transcript.segments)
    for pat in (_DOLLAR, _PCT, _LOAN_ID, _TICKET, _DATE):
        for m in pat.finditer(full_text):
            for sub in re.findall(r"\d{4,}", m.group(0)):
                caught.add(sub)

    matches = []
    for seg in transcript.segments:
        display = transcript.display_name(seg.speaker_id)
        for m in _BARE_NUMBER.finditer(seg.text):
            if m.group(0) not in caught:
                matches.append((m.group(0), seg.start, display))

    lines.append("### Bare Numbers (4+ digits)")
    lines.append("")
    if matches:
        lines += [
            "| Value | Time | Speaker |",
            "|-------|------|---------|",
        ]
        seen: set[str] = set()
        for value, ts, speaker in matches:
            if value not in seen:
                seen.add(value)
                lines.append(f"| {value} | {_fmt(ts)} | {speaker} |")
    else:
        lines.append("*No standalone numbers found.*")
    lines.append("")


def _find_entities_with_context(
    transcript: Transcript,
    pattern: re.Pattern,  # type: ignore[type-arg]
) -> list[tuple[str, float, str]]:
    results = []
    seen: set[str] = set()
    for seg in transcript.segments:
        display = transcript.display_name(seg.speaker_id)
        for m in pattern.finditer(seg.text):
            val = m.group(0)
            if val not in seen:
                seen.add(val)
                results.append((val, seg.start, display))
    return results


def _detect_speaker_issues(
    transcript: Transcript,
    short_threshold: float,
    window: float,
    window_count: int,
) -> list[tuple[str, str, str]]:
    issues: list[tuple[str, str, str]] = []
    segs = list(transcript.segments)

    # 1. Very short segments
    for seg in segs:
        dur = seg.end - seg.start
        if dur < short_threshold and seg.words:
            display = transcript.display_name(seg.speaker_id)
            issues.append(
                (
                    "Short segment",
                    f"{_fmt(seg.start)}–{_fmt(seg.end)}",
                    f'{display}: "{seg.text[:60]}" ({dur:.1f}s)',
                )
            )

    # 2. Rapid speaker switching within rolling window
    changes = []
    for prev, cur in zip(segs, segs[1:], strict=False):
        if prev.speaker_id != cur.speaker_id:
            changes.append(cur.start)

    i = 0
    while i < len(changes):
        window_start = changes[i]
        count_in_window = sum(
            1 for t in changes if window_start <= t < window_start + window
        )
        if count_in_window >= window_count:
            window_end = window_start + window
            issues.append(
                (
                    "Rapid switching",
                    f"{_fmt(window_start)}–{_fmt(window_end)}",
                    f"{count_in_window} speaker changes in {window:.0f}s window",
                )
            )
            # Skip ahead past this window to avoid duplicates
            while i < len(changes) and changes[i] < window_end:
                i += 1
        else:
            i += 1

    # 3. Alternating pair (ABABAB... ≥ 5 consecutive alternations = 6 segments)
    if len(segs) >= 6:
        for start_idx in range(len(segs) - 5):
            chunk = segs[start_idx : start_idx + 6]
            ids = [s.speaker_id for s in chunk]
            if len(set(ids)) == 2 and all(ids[j] != ids[j + 1] for j in range(5)):
                ts_start = chunk[0].start
                ts_end = chunk[-1].end
                a, b = ids[0], ids[1]
                issues.append(
                    (
                        "Alternating pair",
                        f"{_fmt(ts_start)}–{_fmt(ts_end)}",
                        f"{transcript.display_name(a)} and {transcript.display_name(b)} alternate ≥5 times — possible diarization error",
                    )
                )
                break  # report once per transcript

    return issues
