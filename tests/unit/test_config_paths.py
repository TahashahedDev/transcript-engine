"""
Regression tests for CWD-independent path resolution.

Relative defaults ("profiles", "outputs", "temp") used to be interpreted
against the process working directory. Launching the API from anywhere other
than the repo root — a plain `nohup uvicorn api.main:app &` from a home
directory, a systemd unit with no WorkingDirectory, a container entrypoint —
made every job fail with "Profile not found: 'generic' in profiles", an error
that pointed nowhere near the real cause.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from transcript_engine.config.paths import project_root, resolve_path


class TestProjectRoot:
    def test_project_root_contains_the_package(self) -> None:
        assert (project_root() / "transcript_engine").is_dir()

    def test_project_root_contains_profiles(self) -> None:
        """The directory jobs actually failed to find."""
        assert (project_root() / "profiles").is_dir()

    def test_env_override_wins(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Needed when the package is installed and data lives elsewhere."""
        monkeypatch.setenv("TE_PROJECT_ROOT", str(tmp_path))
        assert project_root() == tmp_path.resolve()


class TestResolvePath:
    def test_relative_path_anchors_to_project_root(self) -> None:
        assert resolve_path("profiles") == project_root() / "profiles"

    def test_absolute_path_is_left_alone(self) -> None:
        """Operators must still be able to point these at a mounted volume."""
        absolute = Path("/mnt/data/outputs")
        assert resolve_path(absolute) == absolute

    def test_resolution_is_independent_of_cwd(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.chdir(tmp_path)
        assert resolve_path("profiles") == project_root() / "profiles"
        assert os.getcwd() != str(project_root())


class TestSettingsAnchoring:
    def test_pipeline_profiles_dir_is_absolute(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        from transcript_engine.config.loader import load_settings

        monkeypatch.chdir(tmp_path)
        settings = load_settings()

        profiles_dir = Path(settings.pipeline.processing.profiles_dir)
        assert profiles_dir.is_absolute()
        assert profiles_dir.is_dir(), "profiles must resolve to a real directory from any CWD"

    def test_api_config_dirs_are_absolute(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        from api.config import APIConfig

        monkeypatch.chdir(tmp_path)
        cfg = APIConfig()

        for value in (cfg.temp_dir, cfg.output_dir, cfg.profiles_dir):
            assert Path(value).is_absolute(), f"{value} must be absolute"

    def test_absolute_override_is_preserved(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        from api.config import APIConfig

        target = tmp_path / "custom-outputs"
        monkeypatch.setenv("TE_API_OUTPUT_DIR", str(target))
        assert Path(APIConfig().output_dir) == target
