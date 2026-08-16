from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .model import Profile, VocabularyEntry

_DEFAULT_PROCESSORS = [
    "vocabulary_correction",
    "context_correction",
    "transcript_cleanup",
    "speaker_formatting",
]


def load_profile(name: str, profiles_dir: Path = Path("profiles")) -> Profile:
    """Load a named profile from profiles_dir/name/."""
    profile_dir = profiles_dir / name
    if not profile_dir.exists():
        raise FileNotFoundError(f"Profile not found: '{name}' in {profiles_dir}")

    meta: dict[str, Any] = {}
    meta_file = profile_dir / "profile.json"
    if meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))

    vocab_file = profile_dir / "vocabulary.yaml"
    entries = _load_vocab_entries(vocab_file)

    return Profile(
        name=meta.get("name", name),
        description=meta.get("description", ""),
        industry=meta.get("industry", name),
        vocabulary=tuple(entries),
        processors=meta.get("processors", _DEFAULT_PROCESSORS),
    )


def list_profiles(profiles_dir: Path = Path("profiles")) -> list[str]:
    if not profiles_dir.exists():
        return []
    return sorted(
        d.name
        for d in profiles_dir.iterdir()
        if d.is_dir() and (d / "profile.json").exists()
    )


def _load_vocab_entries(vocab_file: Path) -> list[VocabularyEntry]:
    """Load vocabulary entries from a raw YAML file without a profile.json."""
    if not vocab_file.exists():
        return []
    raw: dict[str, Any] = yaml.safe_load(vocab_file.read_text(encoding="utf-8")) or {}
    entries = []
    for canonical, entry in raw.items():
        if isinstance(entry, dict):
            entries.append(
                VocabularyEntry(
                    canonical=str(canonical),
                    aliases=[str(a) for a in entry.get("aliases", [])],
                    confidence=float(entry.get("confidence", 0.9)),
                    context=[str(c) for c in entry.get("context", [])],
                    fuzzy=bool(entry.get("fuzzy", False)),
                    case_sensitive=bool(entry.get("case_sensitive", False)),
                )
            )
    return entries
