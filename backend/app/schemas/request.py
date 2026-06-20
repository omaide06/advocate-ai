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


class ModelProvider(str, Enum):
    """
    The LLM provider to use for this analysis request.

    - ``anthropic`` – Anthropic Claude (requires user-supplied API key).
    - ``openai``    – OpenAI ChatGPT (requires user-supplied API key).
    - ``gemini``    – Google Gemini (requires user-supplied API key).
    - ``nvidia``    – NVIDIA-hosted free open-source models (no key required;
                      optionally supply a free NVIDIA key for higher throughput).
    - ``mock``      – Deterministic offline responses; useful for testing.

    If omitted, the server resolves the provider from its environment variables
    (ANTHROPIC_API_KEY → OPENAI_API_KEY → GOOGLE_API_KEY → NVIDIA free tier).
    """

    anthropic = "anthropic"
    openai = "openai"
    gemini = "gemini"
    nvidia = "nvidia"
    mock = "mock"


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
    provider:
        Which LLM provider to use for this request.  Defaults to server-
        configured priority (env keys → NVIDIA free tier).
    model:
        Specific model ID within the chosen provider (e.g.
        ``claude-3-5-sonnet-20241022``).  Uses the provider's default
        if omitted.  See ``GET /models`` for the full catalog.
    api_key:
        User-supplied API key for paid providers (Anthropic, OpenAI,
        Gemini).  **Never stored** – used only for the duration of this
        request.  Not required for ``nvidia`` or ``mock`` providers.
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
    provider: Optional[ModelProvider] = Field(
        default=None,
        description=(
            "LLM provider to use: anthropic | openai | gemini | nvidia | mock. "
            "Defaults to the server's env-key priority (NVIDIA free tier if no keys set)."
        ),
    )
    model: Optional[str] = Field(
        default=None,
        max_length=100,
        description=(
            "Specific model ID within the provider "
            "(e.g. 'claude-3-5-sonnet-20241022', 'gpt-4o', 'gemini-2.5-pro', "
            "'meta/llama-3.1-70b-instruct'). Uses provider default if omitted."
        ),
    )
    api_key: Optional[str] = Field(
        default=None,
        description=(
            "User-supplied API key for paid providers (Anthropic, OpenAI, Gemini). "
            "Never stored or logged. Not required for 'nvidia' or 'mock' providers."
        ),
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
