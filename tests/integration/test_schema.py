"""Integration tests for the PostgreSQL schema.

Requires TE_DATABASE_URL to be set and the migration to have been applied.
All tests are skipped automatically when the variable is absent.

Run against a live Supabase database:
    TE_DATABASE_URL=postgresql+asyncpg://... pytest tests/integration/test_schema.py -v
"""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.connection import make_engine, make_session_factory
from db.models import Job, JobEvent, Webhook, WorkerLog

pytestmark = pytest.mark.integration

DATABASE_URL = os.environ.get("TE_DATABASE_URL", "")
skip_no_db = pytest.mark.skipif(
    not DATABASE_URL,
    reason="TE_DATABASE_URL not set — skipping database integration tests",
)


@pytest_asyncio.fixture(scope="module")
async def db_engine():  # type: ignore[misc]
    """Single engine for the entire test module. Disposed after all tests run."""
    engine = make_engine(DATABASE_URL)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def session(db_engine) -> AsyncSession:  # type: ignore[misc]
    """Transactional session rolled back after each test — no data persists."""
    factory = make_session_factory(db_engine)
    async with factory() as s:
        yield s
        await s.rollback()


@skip_no_db
async def test_create_job(session: AsyncSession) -> None:
    """A job row can be inserted and retrieved by id."""
    user_id = uuid.uuid4()
    job = Job(
        user_id=user_id,
        status="queued",
        s3_audio_key=f"audio/{user_id}/test.aac",
        mode="balanced",
        profile="generic",
    )
    session.add(job)
    await session.flush()

    fetched = await session.get(Job, job.id)
    assert fetched is not None
    assert fetched.status == "queued"
    assert fetched.user_id == user_id
    assert fetched.retry_count == 0
    assert fetched.deleted_at is None
    assert fetched.trace_id is not None
    assert fetched.created_at is not None


@skip_no_db
async def test_update_job_status(session: AsyncSession) -> None:
    """Job status transitions from queued → processing → completed."""
    job = Job(user_id=uuid.uuid4(), status="queued")
    session.add(job)
    await session.flush()

    job.status = "processing"
    job.started_at = datetime.now(UTC)
    await session.flush()

    job.status = "completed"
    job.completed_at = datetime.now(UTC)
    job.word_count = 6845
    await session.flush()

    result = await session.execute(select(Job).where(Job.id == job.id))
    updated = result.scalar_one()
    assert updated.status == "completed"
    assert updated.word_count == 6845
    assert updated.started_at is not None
    assert updated.completed_at is not None


@skip_no_db
async def test_insert_job_event(session: AsyncSession) -> None:
    """A job_event row is inserted and linked to the parent job."""
    job = Job(user_id=uuid.uuid4(), status="processing")
    session.add(job)
    await session.flush()

    event = JobEvent(
        job_id=job.id,
        stage="Transcribing",
        progress_pct=0.45,
        elapsed_ms=12500,
        trace_id=job.trace_id,
    )
    session.add(event)
    await session.flush()

    result = await session.execute(
        select(JobEvent).where(JobEvent.job_id == job.id)
    )
    events = result.scalars().all()
    assert len(events) == 1
    assert events[0].stage == "Transcribing"
    assert events[0].progress_pct == pytest.approx(0.45)
    assert events[0].trace_id == job.trace_id


@skip_no_db
async def test_soft_delete_job(session: AsyncSession) -> None:
    """Soft-deleting a job sets deleted_at and nullifies s3_audio_key."""
    job = Job(
        user_id=uuid.uuid4(),
        status="completed",
        s3_audio_key="audio/user/recording.aac",
    )
    session.add(job)
    await session.flush()

    job.deleted_at = datetime.now(UTC)
    job.s3_audio_key = None
    await session.flush()

    result = await session.execute(select(Job).where(Job.id == job.id))
    deleted = result.scalar_one()
    assert deleted.deleted_at is not None
    assert deleted.s3_audio_key is None
    assert deleted.status == "completed"


@skip_no_db
async def test_worker_log_and_webhook(session: AsyncSession) -> None:
    """worker_log and webhooks rows can be created and read back."""
    user_id = uuid.uuid4()
    job = Job(user_id=user_id, status="completed")
    session.add(job)
    await session.flush()

    log = WorkerLog(
        job_id=job.id,
        modal_call_id="modal-call-abc123",
        gpu_type="L4",
        vram_peak_gb=11.4,
        cost_usd=0.0067,
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )
    session.add(log)

    hook = Webhook(
        user_id=user_id,
        url="https://example.com/webhook",
        secret_hash="sha256:abc123def456",
        active=True,
    )
    session.add(hook)
    await session.flush()

    result_log = await session.execute(
        select(WorkerLog).where(WorkerLog.job_id == job.id)
    )
    logs = result_log.scalars().all()
    assert len(logs) == 1
    assert logs[0].modal_call_id == "modal-call-abc123"
    assert logs[0].gpu_type == "L4"
    assert logs[0].cost_usd == pytest.approx(0.0067)

    result_hook = await session.execute(
        select(Webhook).where(Webhook.user_id == user_id)
    )
    hooks = result_hook.scalars().all()
    assert len(hooks) == 1
    assert hooks[0].active is True
