"""
api/analyze.py
--------------
POST /analyze endpoint – the primary entry point for the ADVOCATE pipeline.

This module wires together:
- Request validation (delegated to Pydantic via FastAPI).
- The multi-agent orchestrator that runs the full analysis pipeline.
- Per-request LLM provider selection (provider / model / api_key fields).
- Response serialisation back to the caller.

Error handling strategy:
- Pydantic validation errors → FastAPI automatically returns 422.
- Business-logic / LLM errors → caught here and returned as 500 with a
  structured ErrorResponse body so clients always receive consistent JSON.
- Database errors → also caught at this layer; the session rollback is handled
  by the get_db() dependency.
"""

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.orchestrator import AnalysisOrchestrator
from app.database.database import get_db
from app.schemas.request import AnalyzeRequest
from app.schemas.response import AnalysisResponse, ErrorResponse
from app.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/analyze", tags=["Analysis"])

# Module-level orchestrator instance (stateless – safe for concurrent use).
_orchestrator = AnalysisOrchestrator()


@router.post(
    "",
    response_model=AnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyse an idea with the ADVOCATE multi-agent pipeline",
    response_description="Full structured analysis including score, assumptions, "
    "counter-arguments, and final verdict.",
    responses={
        422: {"description": "Validation error – idea too short or mode invalid."},
        500: {"model": ErrorResponse, "description": "Internal pipeline error."},
    },
)
async def analyze_idea(
    request: Request,
    payload: AnalyzeRequest,
    db: AsyncSession = Depends(get_db),
) -> AnalysisResponse:
    """
    Run the full ADVOCATE multi-agent analysis on the submitted idea.

    The pipeline executes the following agents in order:
    1. **Quality Assessor** – scores the idea 1–5, derives attack intensity.
    2. **Assumption Scanner** – detects hidden assumptions.
    3. **Steelman Generator** – generates strongest counter-arguments.
    4. **Formatter** – synthesises a final verdict and TL;DR.

    The session is persisted to the database before the response is returned,
    making it retrievable via ``GET /session/{session_id}``.

    Parameters
    ----------
    payload.idea:
        The idea or claim to be challenged (10–5000 characters).
    payload.mode:
        ``standard`` (default), ``quick`` (assumptions only), or
        ``deep`` (extra assumption passes + double steelman).
    payload.context:
        Optional background information for the agents to consider.
    payload.provider:
        LLM provider override for this request (anthropic | openai | gemini |
        nvidia | mock).  Falls back to server env-key resolution if omitted.
    payload.model:
        Specific model ID within the chosen provider.  Uses provider default
        if omitted.
    payload.api_key:
        User-supplied API key for paid providers.  Never stored or logged.
    """
    log.info(
        "POST /analyze – mode=%s provider=%s model=%s idea_len=%d",
        payload.mode.value,
        payload.provider.value if payload.provider else "server-default",
        payload.model or "provider-default",
        len(payload.idea),
    )

    try:
        result_dict = await _orchestrator.run(
            idea=payload.idea,
            mode=payload.mode,
            context=payload.context,
            db=db,
            provider=payload.provider.value if payload.provider else None,
            model=payload.model,
            api_key=payload.api_key,
        )
    except ValueError as exc:
        # Invalid provider / missing key – return 400
        log.warning("Bad request: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        log.error("Pipeline runtime error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        log.exception("Unexpected error in analysis pipeline: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during analysis. Please try again.",
        ) from exc

    # FastAPI will validate result_dict against AnalysisResponse automatically.
    return AnalysisResponse(**result_dict)
