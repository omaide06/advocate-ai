"""
schemas/response.py
-------------------
Pydantic v2 response schemas for the ADVOCATE API.

Every endpoint returns one of these models so clients always receive a
consistent, well-typed JSON envelope.  Pydantic's ``model_config`` sets
``from_attributes=True`` which lets FastAPI serialise SQLAlchemy ORM objects
directly without an explicit conversion step.
"""

from datetime import datetime
from typing import Any, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class AssumptionItem(BaseModel):
    """A single hidden assumption detected by the Assumption Scanner agent."""

    assumption: str = Field(..., description="The assumption text.")
    severity: str = Field(
        ...,
        description="Risk level of the assumption: low | medium | high | critical.",
    )
    explanation: str = Field(
        ..., description="Why this assumption is risky or potentially flawed."
    )


class CounterArgumentItem(BaseModel):
    """A single counter-argument produced by the Steelman Generator agent."""

    argument: str = Field(..., description="The counter-argument text.")
    strength: str = Field(
        ...,
        description="How compelling this argument is: weak | moderate | strong | devastating.",
    )
    evidence_type: str = Field(
        ...,
        description="The type of evidence backing this argument (e.g. empirical, logical, historical).",
    )


class AnalysisResponse(BaseModel):
    """
    Full structured response returned by ``POST /analyze``.

    This is the primary output of the ADVOCATE pipeline.  It bundles
    every agent's output into a single coherent JSON document.

    Attributes
    ----------
    session_id:    Unique ID for this analysis session (UUID v4).
    idea:          The original idea text submitted by the user.
    mode:          The analysis mode used (standard / quick / deep).
    score:         Numeric quality score 1–5 assigned by the Quality Assessor.
    quality_label: Human-readable label for the score (e.g. "Moderate").
    attack_intensity:
        Derived from score – "aggressive" | "balanced" | "surgical".
    assumptions:   List of hidden assumptions the system detected.
    counter_arguments:
        Strongest arguments against the idea.
    verdict:       Single-paragraph final judgement from the Formatter agent.
    summary:       Short ≤ 3-sentence TL;DR for quick consumption.
    processing_time_seconds:
        Wall-clock time the pipeline took to complete.
    created_at:    UTC timestamp of when the session was stored.
    llm_provider:  Which LLM backend was used (anthropic | openai | mock).
    """

    model_config = ConfigDict(from_attributes=True)

    session_id: UUID
    idea: str
    mode: str
    score: float
    quality_label: str
    attack_intensity: str
    assumptions: List[AssumptionItem]
    counter_arguments: List[CounterArgumentItem]
    verdict: str
    summary: str
    processing_time_seconds: float
    created_at: datetime
    llm_provider: str


class SessionSummary(BaseModel):
    """
    Lightweight session representation used in the ``GET /sessions`` list.

    Only the top-level fields are returned to avoid bloating list responses
    with full assumption and counter-argument payloads.
    """

    model_config = ConfigDict(from_attributes=True)

    session_id: UUID
    idea: str
    mode: str
    score: float
    quality_label: str
    attack_intensity: str
    verdict: str
    processing_time_seconds: float
    created_at: datetime


class SessionListResponse(BaseModel):
    """Paginated wrapper around a list of :class:`SessionSummary` objects."""

    total: int = Field(..., description="Total number of sessions in the database.")
    sessions: List[SessionSummary]


class HealthResponse(BaseModel):
    """Response model for ``GET /health``."""

    status: str = Field(..., description="Always 'ok' when the service is up.")
    version: str
    database: str = Field(
        ..., description="'connected' or an error description."
    )
    llm_provider: str = Field(
        ..., description="Active LLM provider (anthropic | openai | mock)."
    )
    timestamp: datetime


class ErrorResponse(BaseModel):
    """
    Standardised error envelope returned on 4xx / 5xx responses.

    Using a consistent error shape makes client-side error handling trivial.
    """

    error: str = Field(..., description="Short machine-readable error code.")
    detail: str = Field(..., description="Human-readable explanation.")
    status_code: int
    request_id: Optional[str] = None
