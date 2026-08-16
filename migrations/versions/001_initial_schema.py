"""Initial schema: jobs, job_events, webhooks, worker_log

Revision ID: 001
Revises:
Create Date: 2026-07-29
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "jobs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "status",
            sa.String(20),
            nullable=False,
            server_default="queued",
        ),
        sa.Column("s3_audio_key", sa.Text, nullable=True),
        sa.Column("mode", sa.String(50), nullable=False, server_default="balanced"),
        sa.Column("profile", sa.String(100), nullable=False, server_default="generic"),
        sa.Column(
            "trace_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("word_count", sa.Integer, nullable=True),
        sa.Column("error", sa.Text, nullable=True),
        sa.Column("retry_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_table(
        "job_events",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("jobs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("stage", sa.String(100), nullable=False),
        sa.Column("progress_pct", sa.Float, nullable=True),
        sa.Column("word_count", sa.Integer, nullable=True),
        sa.Column("elapsed_ms", sa.Integer, nullable=True),
        sa.Column("trace_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )
    op.create_index("idx_job_events_job_id", "job_events", ["job_id"])

    op.create_table(
        "webhooks",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("url", sa.Text, nullable=False),
        sa.Column("secret_hash", sa.Text, nullable=False),
        sa.Column("active", sa.Boolean, nullable=False, server_default="true"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )
    op.create_index("idx_webhooks_user_id", "webhooks", ["user_id"])

    op.create_table(
        "worker_log",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("modal_call_id", sa.Text, nullable=True),
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("jobs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("vram_peak_gb", sa.Float, nullable=True),
        sa.Column("gpu_type", sa.Text, nullable=True),
        sa.Column("cost_usd", sa.Float, nullable=True),
    )
    op.create_index("idx_worker_log_job_id", "worker_log", ["job_id"])


def downgrade() -> None:
    op.drop_table("worker_log")
    op.drop_table("webhooks")
    op.drop_table("job_events")
    op.drop_table("jobs")
