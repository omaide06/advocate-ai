"""
schemas/request.py
------------------
Pydantic v2 request schemas for the ADVOCATE API.

These models are the single source of truth for what the API accepts.
FastAPI uses them for automatic validation, serialisation, and OpenAPI
documentation generation.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class AnalysisMode(str, Enum):
    """
    The operating mode for an analysis session.

    - ``standard`` – full multi-agent pipeline (default).
    - ``quick``    – skip steelmanning; return score + assumptions only.
    - ``deep``     – run extra assumption-mining passes before generating
                     counter-arguments.
    """

    standard = "standard"
    quick = "quick"
    deep = "deep"


class AnalyzeRequest(BaseModel):
    """
    Payload accepted by ``POST /analyze``.

    Attributes
    ----------
    idea:
        The user's idea or claim to be analysed.  Must be at least 10
        characters so the agents have meaningful content to work with.
    mode:
        Controls pipeline depth.  Defaults to ``AnalysisMode.standard``.
    context:
        Optional background information the user wants the agents to
        consider when evaluating the idea (e.g. domain, constraints).
    """

    idea: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        description="The idea or claim to be challenged by ADVOCATE.",
        examples=["We should ban all social media for people under 18."],
    )
    mode: AnalysisMode = Field(
        default=AnalysisMode.standard,
        description="Analysis depth: standard | quick | deep.",
    )
    context: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Optional background context to aid analysis.",
    )

    @field_validator("idea")
    @classmethod
    def strip_and_validate_idea(cls, v: str) -> str:
        """
        Strip leading/trailing whitespace and reject ideas that are
        effectively empty after stripping.
        """
        stripped = v.strip()
        if len(stripped) < 10:
            raise ValueError(
                "The idea must contain at least 10 non-whitespace characters."
            )
        return stripped
