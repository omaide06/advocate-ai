"""
api/health.py
-------------
GET /health – liveness and readiness probe endpoint.

Returns the system status including:
- Application version.
- Database connectivity status (async SELECT 1 probe).
- Active LLM provider name.
- UTC timestamp.

Health checks are intentionally lightweight – they must complete in
milliseconds even under heavy load so load balancers and orchestrators
(Kubernetes, ECS, etc.) can probe frequently without impacting performance.
"""

from datetime import datetime

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.database import get_db
from app.schemas.response import HealthResponse
from app.services.llm_service import llm_service
from app.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter(tags=["Health"])

# Application version – bump this when deploying a new release.
_APP_VERSION = "1.0.0"


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service liveness and readiness probe",
    response_description="System status including DB connectivity and LLM provider.",
)
async def health_check(db: AsyncSession = Depends(get_db)) -> HealthResponse:
    """
    Check that the ADVOCATE service is alive and all critical dependencies
    are reachable.

    Database check:
        Executes ``SELECT 1`` via the async session to verify the database
        connection is alive.  On failure, the status field reports the error
        text rather than raising an exception so the endpoint always returns
        a parseable JSON body.

    LLM provider check:
        Reads the active provider name from the module-level
        :data:`~app.services.llm_service.llm_service` singleton.
        No network call is made; we only report which provider is configured.

    Returns
    -------
    HealthResponse
        ``status="ok"`` when all checks pass; the ``database`` field
        describes any connectivity issues.
    """
    log.debug("GET /health")

    # -----------------------------------------------------------------------
    # Database probe – simple SELECT 1.
    # -----------------------------------------------------------------------
    db_status: str
    try:
        await db.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as exc:  # pragma: no cover
        log.error("Health check: database probe failed – %s", exc)
        db_status = f"error: {exc}"

    return HealthResponse(
        status="ok",
        version=_APP_VERSION,
        database=db_status,
        llm_provider=llm_service.get_provider_name(),
        timestamp=datetime.utcnow(),
    )
