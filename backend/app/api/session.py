"""
api/session.py
--------------
Session retrieval endpoints:
  GET /session/{session_id} – retrieve a specific analysis session.
  GET /sessions             – list all sessions (with optional pagination).

Database interaction:
  Both endpoints use SQLAlchemy 2.0 async ``select()`` statements rather than
  the legacy ``Query`` API.  Results are fetched as ORM instances and then
  serialised by Pydantic's ``from_attributes=True`` configuration, which
  means we never need explicit ``.dict()`` calls.
"""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.database import get_db
from app.models.session import AnalysisSession
from app.schemas.response import AnalysisResponse, SessionListResponse, SessionSummary
from app.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter(tags=["Sessions"])


@router.get(
    "/session/{session_id}",
    response_model=AnalysisResponse,
    summary="Retrieve a specific analysis session by ID",
    responses={
        404: {"description": "Session not found."},
    },
)
async def get_session(
    session_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> AnalysisResponse:
    """
    Fetch the full analysis result for a previously completed session.

    Parameters
    ----------
    session_id:
        The UUID v4 identifier returned by ``POST /analyze``.

    Returns
    -------
    AnalysisResponse
        The complete analysis record including assumptions, counter-arguments,
        verdict, and metadata.

    Raises
    ------
    HTTPException(404)
        If no session with the given ID exists in the database.
    """
    log.debug("GET /session/%s", session_id)

    # SQLAlchemy 2.0 style: use select() with the ORM model class.
    stmt = select(AnalysisSession).where(
        AnalysisSession.session_id == str(session_id)
    )
    result = await db.execute(stmt)
    session = result.scalar_one_or_none()

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )

    log.debug("GET /session/%s – found (score=%.1f)", session_id, session.score)

    # Pydantic's from_attributes=True deserialises the ORM object directly.
    return AnalysisResponse.model_validate(session)


@router.get(
    "/sessions",
    response_model=SessionListResponse,
    summary="List all analysis sessions",
)
async def list_sessions(
    db: AsyncSession = Depends(get_db),
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of sessions to return (1–100).",
    ),
    offset: int = Query(
        default=0,
        ge=0,
        description="Number of sessions to skip (for pagination).",
    ),
    mode: Optional[str] = Query(
        default=None,
        description="Filter by analysis mode: standard | quick | deep.",
    ),
) -> SessionListResponse:
    """
    Return a paginated list of all completed analysis sessions.

    Sessions are ordered by ``created_at`` descending (newest first).

    Parameters
    ----------
    limit:
        Number of sessions to return per page (max 100).
    offset:
        Number of sessions to skip for pagination.
    mode:
        Optional filter to return only sessions of a specific mode.

    Returns
    -------
    SessionListResponse
        Contains ``total`` (count of all matching sessions) and
        ``sessions`` (the paginated slice of lightweight summaries).
    """
    log.debug("GET /sessions – limit=%d offset=%d mode=%s", limit, offset, mode)

    # -----------------------------------------------------------------------
    # Build base query with optional mode filter.
    # -----------------------------------------------------------------------
    base_stmt = select(AnalysisSession)
    count_stmt = select(func.count(AnalysisSession.session_id))

    if mode:
        base_stmt = base_stmt.where(AnalysisSession.mode == mode)
        count_stmt = count_stmt.where(AnalysisSession.mode == mode)

    # -----------------------------------------------------------------------
    # Count total matching records for the pagination envelope.
    # -----------------------------------------------------------------------
    total_result = await db.execute(count_stmt)
    total: int = total_result.scalar_one()

    # -----------------------------------------------------------------------
    # Fetch the paginated slice, ordered newest-first.
    # -----------------------------------------------------------------------
    page_stmt = (
        base_stmt.order_by(AnalysisSession.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    page_result = await db.execute(page_stmt)
    sessions = page_result.scalars().all()

    log.debug("GET /sessions – returning %d of %d total", len(sessions), total)

    return SessionListResponse(
        total=total,
        sessions=[SessionSummary.model_validate(s) for s in sessions],
    )
