"""
database/database.py
--------------------
Async SQLAlchemy 2.0 engine and session factory for ADVOCATE.

Architecture decisions:
- We use ``aiosqlite`` as the async SQLite driver so the event loop is never
  blocked by filesystem I/O (important when running analysis pipelines that
  can take several seconds).
- A single ``AsyncEngine`` is created at import time and reused for the
  lifetime of the process.
- ``get_db()`` is a FastAPI dependency that yields a scoped ``AsyncSession``
  and guarantees rollback on exception + commit on clean exit.
- ``init_db()`` is called once at application startup to create any tables
  that do not yet exist (idempotent via ``checkfirst=True`` semantics inside
  ``create_all``).
"""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Build the database URL from the environment.
# Default: SQLite file stored alongside the backend package.
# ---------------------------------------------------------------------------
_DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "sqlite+aiosqlite:///./advocate.db",
)


class Base(DeclarativeBase):
    """
    Shared declarative base that all ORM models inherit from.

    Using a single ``Base`` per application ensures ``metadata.create_all``
    discovers every table defined anywhere in the codebase when called once.
    """


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
# ``echo=False`` in production – set ``SQL_ECHO=true`` env var for debug SQL.
_echo_sql: bool = os.getenv("SQL_ECHO", "false").lower() == "true"

engine: AsyncEngine = create_async_engine(
    _DATABASE_URL,
    echo=_echo_sql,
    # SQLite-specific: allow the same connection to be used across threads.
    connect_args={"check_same_thread": False} if "sqlite" in _DATABASE_URL else {},
    # Pool settings – SQLite uses StaticPool which is correct for :memory:
    # or file-based databases in async contexts.
    pool_pre_ping=True,
)

# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------
# ``expire_on_commit=False`` prevents SQLAlchemy from expiring ORM attributes
# after commit, which would require additional round-trips in async code.
AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


async def init_db() -> None:
    """
    Create all database tables defined via the shared :class:`Base` metadata.

    Safe to call multiple times – SQLAlchemy's ``create_all`` is idempotent.
    Called once from :func:`app.main.lifespan` at application startup.
    """
    log.info("Initialising database (creating tables if absent)…")
    async with engine.begin() as conn:
        # Import the model so its table is registered on Base.metadata
        # before create_all is called.
        from app.models.session import AnalysisSession  # noqa: F401

        await conn.run_sync(Base.metadata.create_all)
    log.info("Database initialisation complete.")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields an :class:`AsyncSession`.

    Usage::

        @router.get("/example")
        async def example(db: AsyncSession = Depends(get_db)):
            ...

    The session is committed on clean exit and rolled back on any exception,
    then always closed regardless of outcome.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_db_context() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context-manager variant of :func:`get_db` for use outside of
    FastAPI's dependency injection system (e.g. background tasks, scripts).

    Usage::

        async with get_db_context() as db:
            result = await db.execute(select(AnalysisSession))
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
