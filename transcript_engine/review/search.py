"""Standalone search logic for meeting index files.

Kept separate from the CLI so it can be unit-tested without Typer.
"""

from __future__ import annotations

import re
from typing import Any


def search_index(
    index_data: dict[str, Any],
    query: str,
    context_words: int = 8,
) -> list[dict[str, Any]]:
    """
    Search the meeting index for a query string.

    Returns a list of match dicts, each with:
        type, timestamp, speaker_name, text, context

    Special queries (case-insensitive):
        "action items"  → returns all action items
        "decisions"     → returns all decisions
        "questions"     → returns all questions
    """
    q_lower = query.strip().lower()

    # Special keyword queries
    if q_lower in ("action items", "action item"):
        return _list_structured(index_data.get("action_items", []), "action_item")
    if q_lower == "decisions":
        return _list_structured(index_data.get("decisions", []), "decision")
    if q_lower == "questions":
        return _list_structured(index_data.get("questions", []), "question")

    results: list[dict[str, Any]] = []
    pattern = re.compile(re.escape(query), re.IGNORECASE)

    # Search segments
    for seg in index_data.get("segments", []):
        text = seg.get("text", "")
        if not pattern.search(text):
            continue
        context = _build_context(text, pattern, context_words)
        results.append(
            {
                "type": "segment",
                "timestamp": seg.get("start", 0.0),
                "speaker_name": seg.get("speaker_name", "—"),
                "text": text,
                "context": context,
            }
        )

    # Search structured items
    for item in index_data.get("action_items", []):
        if pattern.search(item.get("task", "")):
            results.append(
                {
                    "type": "action_item",
                    "timestamp": item.get("timestamp", 0.0),
                    "speaker_name": item.get("speaker_id", "—"),
                    "text": item.get("task", ""),
                    "context": f"Action item — owner: {item.get('owner') or '—'}",
                }
            )

    for item in index_data.get("decisions", []):
        if pattern.search(item.get("decision", "")):
            results.append(
                {
                    "type": "decision",
                    "timestamp": item.get("timestamp", 0.0),
                    "speaker_name": item.get("speaker_id", "—"),
                    "text": item.get("decision", ""),
                    "context": "Decision",
                }
            )

    for item in index_data.get("questions", []):
        if pattern.search(item.get("question", "")):
            results.append(
                {
                    "type": "question",
                    "timestamp": item.get("timestamp", 0.0),
                    "speaker_name": item.get("speaker_id", "—"),
                    "text": item.get("question", ""),
                    "context": "Open question",
                }
            )

    # Sort by timestamp
    results.sort(key=lambda r: r["timestamp"])
    return results


def _build_context(text: str, pattern: re.Pattern[str], window: int) -> str:
    """Return words surrounding the first match, with the match in **bold**."""
    words = text.split()
    m = pattern.search(text)
    if not m:
        return text[:120]

    # Find which word index contains the match start
    pos = 0
    match_idx = 0
    for i, w in enumerate(words):
        if pos + len(w) > m.start():
            match_idx = i
            break
        pos += len(w) + 1

    lo = max(0, match_idx - window)
    hi = min(len(words), match_idx + window + 1)
    snippet = words[lo:hi]

    # Bold the matched word
    rel = match_idx - lo
    if 0 <= rel < len(snippet):
        original = snippet[rel]
        snippet[rel] = f"**{original}**"

    prefix = "…" if lo > 0 else ""
    suffix = "…" if hi < len(words) else ""
    return prefix + " ".join(snippet) + suffix


def _list_structured(
    items: list[dict[str, Any]], item_type: str
) -> list[dict[str, Any]]:
    """Convert structured items to result dicts for display."""
    results = []
    for item in items:
        if item_type == "action_item":
            text = item.get("task", "")
        elif item_type == "decision":
            text = item.get("decision", "")
        else:
            text = item.get("question", "")
        results.append(
            {
                "type": item_type,
                "timestamp": item.get("timestamp", 0.0),
                "speaker_name": item.get("speaker_id", "—"),
                "text": text,
                "context": "",
            }
        )
    return results
