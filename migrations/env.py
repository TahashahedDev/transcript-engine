from __future__ import annotations

import asyncio
import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

# Make project root importable when running alembic from any directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.models import Base  # noqa: E402

alembic_config = context.config

if alembic_config.config_file_name is not None:
    fileConfig(alembic_config.config_file_name)

target_metadata = Base.metadata


def _get_url() -> str:
    url = os.environ.get("TE_DATABASE_URL", "")
    if not url:
        raise RuntimeError(
            "TE_DATABASE_URL environment variable is not set.\n"
            "Set it in .env or export it before running alembic.\n"
            "Format: postgresql+asyncpg://user:pass@host:5432/dbname"
        )
    return url


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_offline() -> None:
    """Generate SQL to stdout without a live database connection.

    Useful for reviewing the exact DDL before applying: alembic upgrade head --sql
    """
    url = _get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    engine = create_async_engine(_get_url(), echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(do_run_migrations)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
