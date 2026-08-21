"""
Local JSON persistence for SpeakerProfiles across jobs.

Not the Postgres db/ module: that module exists in the repo but is unwired
(no runtime import from the v1 pipeline that actually runs jobs — see
IDENTITY_ARCHITECTURE.md) and adding speaker tables to it is a larger,
separate integration decision than this mission's scope. A profile store
needs to survive process restarts and nothing more at this scale (section 12
of the mission brief: tens to low hundreds of people, a handful of
embeddings each), so a directory of JSON files is the simplest thing that
can possibly persist across jobs — one file per profile, human-inspectable,
trivially deletable for a privacy/consent request without needing a
database migration.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from transcript_engine.identity.profile import SpeakerProfile

DEFAULT_STORE_DIR = Path("data/speaker_profiles")


class SpeakerProfileStore:
    def __init__(self, directory: Path = DEFAULT_STORE_DIR) -> None:
        self.directory = directory

    def _path_for(self, profile_id: str) -> Path:
        return self.directory / f"{profile_id}.json"

    def save(self, profile: SpeakerProfile) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        data = {
            "profile_id": profile.profile_id,
            "display_name": profile.display_name,
            "embeddings": [e.tolist() for e in profile.embeddings],
            "source_job_ids": profile.source_job_ids,
            "identity_evidence": profile.identity_evidence,
            "confirmation_status": profile.confirmation_status,
            "created_at": profile.created_at.isoformat(),
            "updated_at": profile.updated_at.isoformat(),
        }
        # Write-then-rename: a crash mid-write must never leave a truncated,
        # unparseable profile file behind — the next load() would then fail
        # for every profile, not just the one being written.
        tmp_path = self._path_for(profile.profile_id).with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp_path.replace(self._path_for(profile.profile_id))

    def load(self, profile_id: str) -> SpeakerProfile | None:
        path = self._path_for(profile_id)
        if not path.exists():
            return None
        return _profile_from_json(json.loads(path.read_text(encoding="utf-8")))

    def load_all(self) -> list[SpeakerProfile]:
        if not self.directory.exists():
            return []
        profiles = []
        for path in sorted(self.directory.glob("*.json")):
            profiles.append(_profile_from_json(json.loads(path.read_text(encoding="utf-8"))))
        return profiles

    def delete(self, profile_id: str) -> bool:
        path = self._path_for(profile_id)
        if not path.exists():
            return False
        path.unlink()
        return True


def _profile_from_json(data: dict[str, Any]) -> SpeakerProfile:
    profile = SpeakerProfile(
        display_name=data["display_name"],
        profile_id=data["profile_id"],
        source_job_ids=list(data["source_job_ids"]),
        identity_evidence=list(data["identity_evidence"]),
        confirmation_status=data["confirmation_status"],
        created_at=datetime.fromisoformat(data["created_at"]),
        updated_at=datetime.fromisoformat(data["updated_at"]),
    )
    profile.embeddings = [np.array(e, dtype=np.float64) for e in data["embeddings"]]
    return profile
