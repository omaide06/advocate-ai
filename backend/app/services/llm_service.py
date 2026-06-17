"""
services/llm_service.py
-----------------------
Unified LLM abstraction layer for the ADVOCATE backend.

Priority resolution (first available wins):
  1. Anthropic   – if ``ANTHROPIC_API_KEY`` is set in the environment.
  2. OpenAI      – if ``OPENAI_API_KEY`` is set in the environment.
  3. Mock        – deterministic, key-free fallback that returns realistic
                   structured data so the entire system runs without any
                   external API credentials.

All providers expose the same async interface:
  ``async def complete(system: str, user: str) -> str``

This means agents never need to know which backend is in use; they simply
call ``llm_service.complete(...)`` and receive a string back.
"""

import json
import os
import random
from abc import ABC, abstractmethod
from typing import Optional

from app.utils.logger import get_logger

log = get_logger(__name__)


# ===========================================================================
# Abstract base – every provider must implement this contract.
# ===========================================================================


class BaseLLMProvider(ABC):
    """
    Abstract LLM provider interface.

    Sub-classes wrap a specific API client (Anthropic, OpenAI, etc.) and
    expose a single async ``complete`` method so agents remain provider-
    agnostic.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a short identifier for this provider (used in logs/DB)."""

    @abstractmethod
    async def complete(self, system: str, user: str) -> str:
        """
        Send a chat completion request and return the assistant's reply.

        Parameters
        ----------
        system:
            The system-level instruction that sets the LLM's persona and
            output format.
        user:
            The user-turn prompt containing the idea and task description.

        Returns
        -------
        str
            Raw text response from the model.  Agents are responsible for
            parsing this into structured data (typically JSON).
        """


# ===========================================================================
# Provider implementations
# ===========================================================================


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic Claude provider.

    Uses the ``anthropic`` Python SDK (>=0.25).  The model defaults to
    ``claude-3-5-haiku-20241022`` which offers a good balance of speed and
    quality for structured-output tasks.  Override via the ``ANTHROPIC_MODEL``
    environment variable.
    """

    def __init__(self, api_key: str) -> None:
        try:
            import anthropic  # lazy import – only needed when this provider is active
        except ImportError as exc:
            raise RuntimeError(
                "The 'anthropic' package is required to use the Anthropic provider. "
                "Install it with: pip install anthropic"
            ) from exc

        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model: str = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")
        log.info("AnthropicProvider initialised (model=%s)", self._model)

    @property
    def name(self) -> str:
        return "anthropic"

    async def complete(self, system: str, user: str) -> str:
        """
        Call the Anthropic Messages API with the given system / user pair.

        We request up to 4 096 output tokens which is sufficient for the
        longest ADVOCATE agent responses.
        """
        message = await self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return message.content[0].text


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI Chat Completion provider.

    Uses the ``openai`` Python SDK (>=1.0).  Defaults to ``gpt-4o-mini``
    which is fast and cost-effective for structured tasks.  Override via the
    ``OPENAI_MODEL`` environment variable.
    """

    def __init__(self, api_key: str) -> None:
        try:
            from openai import AsyncOpenAI  # lazy import
        except ImportError as exc:
            raise RuntimeError(
                "The 'openai' package is required to use the OpenAI provider. "
                "Install it with: pip install openai"
            ) from exc

        self._client = AsyncOpenAI(api_key=api_key)
        self._model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        log.info("OpenAIProvider initialised (model=%s)", self._model)

    @property
    def name(self) -> str:
        return "openai"

    async def complete(self, system: str, user: str) -> str:
        """
        Call the OpenAI Chat Completions endpoint.

        JSON mode is NOT explicitly requested here because our system prompts
        already instruct the model to output valid JSON and we need the
        flexibility to parse streaming or plain-text responses from agents.
        """
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=4096,
            temperature=0.7,
        )
        return response.choices[0].message.content or ""


class MockProvider(BaseLLMProvider):
    """
    Deterministic mock LLM provider.

    Returns realistic, structurally correct JSON responses without making any
    network requests.  This allows the entire ADVOCATE pipeline – including
    the orchestrator, all agents, and the database – to be exercised with zero
    API credentials.

    The responses are templated with the idea text so they feel contextual
    rather than completely static, even though no actual language model is
    involved.

    Agent-specific dispatch:
    The mock provider inspects the ``system`` prompt for known agent keywords
    (``quality_assessor``, ``assumption_scanner``, ``steelman``, ``formatter``)
    and returns a matching mock structure.  Any unknown prompt falls back to a
    generic quality-assessor response.
    """

    @property
    def name(self) -> str:
        return "mock"

    async def complete(self, system: str, user: str) -> str:
        """
        Return a structurally valid JSON string that matches the schema
        expected by whichever agent is calling.

        The method inspects ``system`` (lowercased) to determine context.
        """
        system_lower = system.lower()

        # Extract a short snippet of the idea for contextual flavour.
        idea_snippet = user[:80].strip().rstrip(".,;") if user else "this idea"

        if "quality_assessor" in system_lower or "score" in system_lower:
            return self._quality_response(idea_snippet)
        elif "assumption" in system_lower:
            return self._assumption_response(idea_snippet)
        elif "steelman" in system_lower or "counter" in system_lower:
            return self._steelman_response(idea_snippet)
        elif "formatter" in system_lower or "verdict" in system_lower:
            return self._formatter_response(idea_snippet)
        else:
            # Default fallback
            return self._quality_response(idea_snippet)

    # ------------------------------------------------------------------
    # Mock response templates
    # ------------------------------------------------------------------

    def _quality_response(self, snippet: str) -> str:
        """Return a mock quality-assessor JSON payload."""
        score = round(random.uniform(1.5, 4.5), 1)
        labels = {
            range(1, 3): ("Poor", "aggressive"),
            range(2, 4): ("Moderate", "balanced"),
            range(4, 6): ("Strong", "surgical"),
        }
        quality_label, attack_intensity = "Moderate", "balanced"
        for r, (label, intensity) in labels.items():
            if int(score) in r:
                quality_label, attack_intensity = label, intensity

        return json.dumps(
            {
                "score": score,
                "quality_label": quality_label,
                "attack_intensity": attack_intensity,
                "reasoning": (
                    f"The idea '{snippet}' demonstrates some merit but suffers from "
                    "a lack of empirical grounding. The proposal makes broad generalisations "
                    "without accounting for systemic complexity or second-order effects."
                ),
            }
        )

    def _assumption_response(self, snippet: str) -> str:
        """Return a mock assumption-scanner JSON payload."""
        return json.dumps(
            {
                "assumptions": [
                    {
                        "assumption": "The proposed change will be adopted uniformly across all affected groups.",
                        "severity": "high",
                        "explanation": (
                            "Adoption is rarely uniform. Different demographics, "
                            "institutions, and incentive structures will respond asymmetrically, "
                            "leading to uneven outcomes that the idea does not account for."
                        ),
                    },
                    {
                        "assumption": f"The status quo is sufficiently understood to justify replacing it with '{snippet}'.",
                        "severity": "medium",
                        "explanation": (
                            "Complex systems are often poorly understood even by experts. "
                            "Proposing a replacement without a rigorous baseline analysis "
                            "risks creating new problems while solving old ones."
                        ),
                    },
                    {
                        "assumption": "Short-term costs will be offset by long-term gains.",
                        "severity": "critical",
                        "explanation": (
                            "This assumption is empirically fragile. Discount rates, "
                            "political cycles, and shifting priorities mean that long-term "
                            "benefits are frequently never realised or captured by different "
                            "stakeholders than those who bore the costs."
                        ),
                    },
                    {
                        "assumption": "There are no significant unintended consequences.",
                        "severity": "high",
                        "explanation": (
                            "Every systemic intervention produces side-effects. The idea "
                            "does not propose any monitoring or adaptive management framework "
                            "to detect and respond to unintended consequences."
                        ),
                    },
                ]
            }
        )

    def _steelman_response(self, snippet: str) -> str:
        """Return a mock steelman-generator JSON payload."""
        return json.dumps(
            {
                "counter_arguments": [
                    {
                        "argument": (
                            "Historical precedent consistently shows that proposals similar to "
                            f"'{snippet}' increase systemic fragility over time. When comparable "
                            "interventions were tried in analogous domains, the failure rate "
                            "exceeded 60% within a decade."
                        ),
                        "strength": "devastating",
                        "evidence_type": "historical",
                    },
                    {
                        "argument": (
                            "The idea fundamentally misidentifies the root cause of the problem "
                            "it purports to solve. Addressing symptoms rather than causes "
                            "produces temporary relief at best and systemic lock-in at worst."
                        ),
                        "strength": "strong",
                        "evidence_type": "logical",
                    },
                    {
                        "argument": (
                            "Economic analysis suggests the total cost of implementation — "
                            "including opportunity costs, transition friction, and enforcement "
                            "overhead — significantly exceeds any quantifiable benefit in "
                            "realistic time horizons."
                        ),
                        "strength": "strong",
                        "evidence_type": "empirical",
                    },
                    {
                        "argument": (
                            "The idea assumes a level of coordination and compliance that has "
                            "never been achieved in comparable real-world scenarios, making the "
                            "theoretical model non-transferable to practice."
                        ),
                        "strength": "moderate",
                        "evidence_type": "empirical",
                    },
                ]
            }
        )

    def _formatter_response(self, snippet: str) -> str:
        """Return a mock formatter JSON payload."""
        return json.dumps(
            {
                "verdict": (
                    f"After rigorous multi-agent analysis, ADVOCATE finds the idea '{snippet}' "
                    "to be intellectually stimulating but fundamentally underprepared for real-world "
                    "implementation. The core proposition rests on at least four critical assumptions "
                    "that lack empirical validation, and the strongest counter-arguments — rooted in "
                    "historical precedent and economic analysis — expose significant structural "
                    "weaknesses. While the underlying motivation may be sound, the idea in its "
                    "current form would benefit enormously from a deeper engagement with the "
                    "existing evidence base, a phased implementation model, and explicit mechanisms "
                    "for detecting and correcting unintended consequences. ADVOCATE recommends "
                    "returning to first principles before advocating for adoption."
                ),
                "summary": (
                    f"The idea '{snippet}' scores moderately but harbours critical hidden "
                    "assumptions and faces devastating historical counter-evidence. It requires "
                    "substantial reworking before it can withstand serious scrutiny."
                ),
            }
        )


# ===========================================================================
# Factory – resolves and instantiates the correct provider at startup.
# ===========================================================================


class LLMService:
    """
    Singleton-style LLM service that wraps a :class:`BaseLLMProvider`.

    Call :meth:`get_provider_name` to find out which backend is active.
    Agents call :meth:`complete` directly; they never interact with the
    underlying provider.

    Resolution order
    ----------------
    1. ``ANTHROPIC_API_KEY`` → :class:`AnthropicProvider`
    2. ``OPENAI_API_KEY``    → :class:`OpenAIProvider`
    3. (no keys)             → :class:`MockProvider`
    """

    def __init__(self) -> None:
        self._provider: BaseLLMProvider = self._resolve_provider()
        log.info("LLMService active provider: %s", self._provider.name)

    # ------------------------------------------------------------------
    # Provider resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_provider() -> BaseLLMProvider:
        """
        Inspect the environment for API keys and return the highest-priority
        available provider.
        """
        anthropic_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
        openai_key: Optional[str] = os.getenv("OPENAI_API_KEY")

        if anthropic_key and anthropic_key.strip():
            log.info("Anthropic API key detected – using AnthropicProvider.")
            return AnthropicProvider(api_key=anthropic_key.strip())

        if openai_key and openai_key.strip():
            log.info("OpenAI API key detected – using OpenAIProvider.")
            return OpenAIProvider(api_key=openai_key.strip())

        log.warning(
            "No LLM API keys found. Falling back to MockProvider. "
            "Set ANTHROPIC_API_KEY or OPENAI_API_KEY for live responses."
        )
        return MockProvider()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def complete(self, system: str, user: str) -> str:
        """
        Proxy a completion request to the active provider.

        Parameters
        ----------
        system:
            The system-level instruction string.
        user:
            The user-turn prompt.

        Returns
        -------
        str
            Raw model response text.

        Raises
        ------
        RuntimeError
            If the underlying provider raises an unexpected error.
        """
        try:
            return await self._provider.complete(system=system, user=user)
        except Exception as exc:
            log.error(
                "LLM provider '%s' raised an error: %s", self._provider.name, exc
            )
            raise RuntimeError(f"LLM completion failed: {exc}") from exc

    def get_provider_name(self) -> str:
        """Return the name of the currently active LLM provider."""
        return self._provider.name


# ---------------------------------------------------------------------------
# Module-level singleton – import and use ``llm_service`` directly.
# ---------------------------------------------------------------------------
llm_service = LLMService()
