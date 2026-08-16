"""Unit tests for /api/v2/jobs endpoints.

All external dependencies (auth, DB session, R2) are replaced with fakes so
these tests run without a live database or R2 bucket.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.v2.auth import get_current_user_id
from api.v2.routes.jobs import router
from db.connection import get_session

TEST_USER_ID = uuid.UUID("aaaaaaaa-0000-0000-0000-000000000001")
OTHER_USER_ID = uuid.UUID("bbbbbbbb-0000-0000-0000-000000000002")
TEST_JOB_ID = uuid.UUID("cccccccc-0000-0000-0000-000000000003")
TEST_TRACE_ID = uuid.UUID("dddddddd-0000-0000-0000-000000000004")


# ── Fixtures ───────────────────────────────────────────────────────────────


def _make_test_app(mock_session: AsyncMock) -> FastAPI:
    """Build a minimal FastAPI app mounting only the v2 jobs router."""
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_current_user_id] = lambda: TEST_USER_ID

    async def _fake_session() -> AsyncGenerator[AsyncMock, None]:
        yield mock_session

    app.dependency_overrides[get_session] = _fake_session
    return app


def _make_mock_session() -> AsyncMock:
    mock = AsyncMock()
    mock.add = MagicMock()
    mock.commit = AsyncMock()
    mock.refresh = AsyncMock()
    return mock


def _make_job(
    job_id: uuid.UUID = TEST_JOB_ID,
    user_id: uuid.UUID = TEST_USER_ID,
    status: str = "queued",
    deleted_at: datetime | None = None,
) -> MagicMock:
    """Return a MagicMock that mimics a populated Job ORM row."""
    job = MagicMock()
    job.id = job_id
    job.user_id = user_id
    job.status = status
    job.mode = "balanced"
    job.profile = "generic"
    job.s3_audio_key = f"audio/{user_id}/{job_id}/audio.aac"
    job.trace_id = TEST_TRACE_ID
    job.word_count = None
    job.error = None
    job.retry_count = 0
    job.created_at = datetime(2024, 1, 1, tzinfo=UTC)
    job.started_at = None
    job.completed_at = None
    job.deleted_at = deleted_at
    return job


# ── POST /api/v2/jobs/upload-url ──────────────────────────────────────────


def test_upload_url_rejects_non_audio_content_type() -> None:
    mock_session = _make_mock_session()
    client = TestClient(_make_test_app(mock_session))
    resp = client.post(
        "/api/v2/jobs/upload-url",
        json={"content_type": "video/mp4", "filename": "meeting.mp4"},
        headers={"Authorization": "Bearer fake"},
    )
    assert resp.status_code == 422


def test_upload_url_rejects_empty_filename() -> None:
    mock_session = _make_mock_session()
    client = TestClient(_make_test_app(mock_session))
    resp = client.post(
        "/api/v2/jobs/upload-url",
        json={"content_type": "audio/aac", "filename": ""},
        headers={"Authorization": "Bearer fake"},
    )
    assert resp.status_code == 422


def test_upload_url_success() -> None:
    mock_session = _make_mock_session()
    client = TestClient(_make_test_app(mock_session))
    with patch(
        "api.v2.routes.jobs.generate_upload_url",
        new=AsyncMock(return_value="https://r2.example.com/presigned-put"),
    ):
        resp = client.post(
            "/api/v2/jobs/upload-url",
            json={"content_type": "audio/aac", "filename": "meeting.aac"},
            headers={"Authorization": "Bearer fake"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["upload_url"] == "https://r2.example.com/presigned-put"
    assert data["s3_key"].startswith(f"audio/{TEST_USER_ID}/")
    assert data["s3_key"].endswith("/meeting.aac")
    assert data["expires_in"] > 0


# ── POST /api/v2/jobs ─────────────────────────────────────────────────────


def test_create_job_rejects_key_owned_by_other_user() -> None:
    mock_session = _make_mock_session()
    client = TestClient(_make_test_app(mock_session))
    foreign_key = f"audio/{OTHER_USER_ID}/{TEST_JOB_ID}/audio.aac"
    resp = client.post(
        "/api/v2/jobs",
        json={"s3_key": foreign_key},
        headers={"Authorization": "Bearer fake"},
    )
    assert resp.status_code == 403


def test_create_job_rejects_invalid_mode() -> None:
    mock_session = _make_mock_session()
    client = TestClient(_make_test_app(mock_session))
    resp = client.post(
        "/api/v2/jobs",
        json={"s3_key": f"audio/{TEST_USER_ID}/{TEST_JOB_ID}/audio.aac", "mode": "turbo"},
        headers={"Authorization": "Bearer fake"},
    )
    assert resp.status_code == 422


def test_create_job_rejects_non_audio_content_type_from_r2() -> None:
    mock_session = _make_mock_session()
    client = TestClient(_make_test_app(mock_session))
    valid_key = f"audio/{TEST_USER_ID}/{TEST_JOB_ID}/audio.aac"
    with patch(
        "api.v2.routes.jobs.verify_object_exists",
        new=AsyncMock(return_value="application/octet-stream"),
    ):
        resp = client.post(
            "/api/v2/jobs",
            json={"s3_key": valid_key},
            headers={"Authorization": "Bearer fake"},
        )
    assert resp.status_code == 400
    assert "Content-Type" in resp.json()["detail"]


def test_create_job_success() -> None:
    mock_session = _make_mock_session()
    valid_key = f"audio/{TEST_USER_ID}/{TEST_JOB_ID}/audio.aac"

    # SQLAlchemy evaluates column defaults at flush time, not at object construction.
    # Since the session is a mock, refresh must manually populate id and trace_id.
    async def _fake_refresh(obj: object) -> None:
        object.__setattr__(obj, "id", TEST_JOB_ID)
        object.__setattr__(obj, "trace_id", TEST_TRACE_ID)

    mock_session.refresh = AsyncMock(side_effect=_fake_refresh)

    client = TestClient(_make_test_app(mock_session))
    with patch(
        "api.v2.routes.jobs.verify_object_exists",
        new=AsyncMock(return_value="audio/aac"),
    ):
        resp = client.post(
            "/api/v2/jobs",
            json={"s3_key": valid_key},
            headers={"Authorization": "Bearer fake"},
        )

    assert resp.status_code == 201
    data = resp.json()
    assert data["status"] == "queued"
    assert data["job_id"] == str(TEST_JOB_ID)
    assert data["trace_id"] == str(TEST_TRACE_ID)
    mock_session.add.assert_called_once()
    mock_session.commit.assert_awaited_once()


# ── GET /api/v2/jobs/{job_id} ─────────────────────────────────────────────


def test_get_job_not_found() -> None:
    mock_session = _make_mock_session()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute = AsyncMock(return_value=mock_result)

    client = TestClient(_make_test_app(mock_session))
    resp = client.get(
        f"/api/v2/jobs/{TEST_JOB_ID}",
        headers={"Authorization": "Bearer fake"},
    )
    assert resp.status_code == 404


def test_get_job_wrong_user_returns_403() -> None:
    mock_session = _make_mock_session()
    job_owned_by_other = _make_job(user_id=OTHER_USER_ID)
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = job_owned_by_other
    mock_session.execute = AsyncMock(return_value=mock_result)

    client = TestClient(_make_test_app(mock_session))
    resp = client.get(
        f"/api/v2/jobs/{TEST_JOB_ID}",
        headers={"Authorization": "Bearer fake"},
    )
    assert resp.status_code == 403


def test_get_job_deleted_returns_404() -> None:
    mock_session = _make_mock_session()
    deleted_job = _make_job(deleted_at=datetime(2024, 6, 1, tzinfo=UTC))
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = deleted_job
    mock_session.execute = AsyncMock(return_value=mock_result)

    client = TestClient(_make_test_app(mock_session))
    resp = client.get(
        f"/api/v2/jobs/{TEST_JOB_ID}",
        headers={"Authorization": "Bearer fake"},
    )
    assert resp.status_code == 404


def test_get_job_success() -> None:
    mock_session = _make_mock_session()
    job = _make_job(status="processing")
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = job
    mock_session.execute = AsyncMock(return_value=mock_result)

    client = TestClient(_make_test_app(mock_session))
    resp = client.get(
        f"/api/v2/jobs/{TEST_JOB_ID}",
        headers={"Authorization": "Bearer fake"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "processing"
    assert data["job_id"] == str(TEST_JOB_ID)
    assert data["trace_id"] == str(TEST_TRACE_ID)


# ── GET /api/v2/jobs/{job_id}/downloads/{fmt} ─────────────────────────────


def test_download_url_invalid_format() -> None:
    mock_session = _make_mock_session()
    client = TestClient(_make_test_app(mock_session))
    resp = client.get(
        f"/api/v2/jobs/{TEST_JOB_ID}/downloads/pdf",
        headers={"Authorization": "Bearer fake"},
    )
    assert resp.status_code == 400
    assert "Unsupported format" in resp.json()["detail"]


def test_download_url_job_not_completed() -> None:
    mock_session = _make_mock_session()
    job = _make_job(status="processing")
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = job
    mock_session.execute = AsyncMock(return_value=mock_result)

    client = TestClient(_make_test_app(mock_session))
    resp = client.get(
        f"/api/v2/jobs/{TEST_JOB_ID}/downloads/txt",
        headers={"Authorization": "Bearer fake"},
    )
    assert resp.status_code == 409
    assert "not completed" in resp.json()["detail"]


def test_download_url_success() -> None:
    mock_session = _make_mock_session()
    job = _make_job(status="completed")
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = job
    mock_session.execute = AsyncMock(return_value=mock_result)

    client = TestClient(_make_test_app(mock_session))
    with patch(
        "api.v2.routes.jobs.generate_download_url",
        new=AsyncMock(return_value="https://r2.example.com/presigned-get"),
    ):
        resp = client.get(
            f"/api/v2/jobs/{TEST_JOB_ID}/downloads/txt",
            headers={"Authorization": "Bearer fake"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["url"] == "https://r2.example.com/presigned-get"
    assert data["expires_in"] > 0


# ── DELETE /api/v2/jobs/{job_id} ──────────────────────────────────────────


def test_delete_job_not_found() -> None:
    mock_session = _make_mock_session()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute = AsyncMock(return_value=mock_result)

    client = TestClient(_make_test_app(mock_session))
    resp = client.delete(
        f"/api/v2/jobs/{TEST_JOB_ID}",
        headers={"Authorization": "Bearer fake"},
    )
    assert resp.status_code == 404


def test_delete_job_success_skips_r2_when_not_configured() -> None:
    mock_session = _make_mock_session()
    job = _make_job()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = job
    mock_session.execute = AsyncMock(return_value=mock_result)

    client = TestClient(_make_test_app(mock_session))
    with patch("api.v2.routes.jobs.is_configured", return_value=False):
        resp = client.delete(
            f"/api/v2/jobs/{TEST_JOB_ID}",
            headers={"Authorization": "Bearer fake"},
        )
    assert resp.status_code == 200
    assert resp.json() == {"deleted": True}
    mock_session.commit.assert_awaited_once()
    assert job.deleted_at is not None
    assert job.s3_audio_key is None


def test_delete_job_deletes_r2_when_configured() -> None:
    mock_session = _make_mock_session()
    job = _make_job()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = job
    mock_session.execute = AsyncMock(return_value=mock_result)

    client = TestClient(_make_test_app(mock_session))
    with (
        patch("api.v2.routes.jobs.is_configured", return_value=True),
        patch(
            "api.v2.routes.jobs.delete_objects_with_prefix",
            new=AsyncMock(),
        ) as mock_delete,
    ):
        resp = client.delete(
            f"/api/v2/jobs/{TEST_JOB_ID}",
            headers={"Authorization": "Bearer fake"},
        )
    assert resp.status_code == 200
    assert mock_delete.await_count == 2
