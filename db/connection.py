"""Async database engine and session factory."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from db.config import settings

# Supabase Supavisor in Transaction mode does not support prepared statements.
# asyncpg caches them by default; statement_cache_size=0 disables the cache.
# Without this, every query fails with "prepared statement already exists".
_ASYNCPG_CONNECT_ARGS = {"statement_cache_size": 0}

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None

if settings.database_url:
    _engine = create_async_engine(
        settings.database_url,
        echo=False,
        pool_pre_ping=True,
        connect_args=_ASYNCPG_CONNECT_ARGS,
    )
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency. Yields a transactional async session."""
    if _session_factory is None:
        raise RuntimeError("Database not configured — set TE_DATABASE_URL")
    async with _session_factory() as session:
        yield session


def make_engine(url: str) -> AsyncEngine:
    """Create an async engine for the given URL.

    Callers are responsible for calling ``await engine.dispose()`` when done.
    """
    return create_async_engine(
        url,
        echo=False,
        pool_pre_ping=True,
        connect_args=_ASYNCPG_CONNECT_ARGS,
    )


def make_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Create a session factory bound to an existing engine (used by tests)."""
    return async_sessionmaker(engine, expire_on_commit=False)
