"""
Project-root resolution for configured paths.

Several defaults (``profiles``, ``outputs``, ``temp``) are relative paths. Read
literally they depend on the process's working directory, so starting the API
from anywhere other than the repo root — ``nohup uvicorn api.main:app &`` from
a home directory, a systemd unit without ``WorkingDirectory``, a container
entrypoint — silently changed their meaning. Profiles then resolved to a
directory that does not exist and *every* job failed with "Profile not found:
'generic' in profiles", which says nothing about the real cause.

Relative paths are therefore resolved against the project root rather than the
CWD. Absolute paths are always honoured as-is, so operators can still point
these anywhere (a mounted volume on a rented GPU host, for example).
"""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """
    Absolute path to the project root.

    ``TE_PROJECT_ROOT`` wins when set — needed when the package is installed
    into site-packages and the data directories live elsewhere. Otherwise this
    file is at ``<root>/transcript_engine/config/paths.py``, so the root is
    three levels up.
    """
    env_root = os.environ.get("TE_PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parent.parent.parent


def resolve_path(value: str | Path) -> Path:
    """Resolve *value* against the project root unless it is already absolute."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return project_root() / path
