"""
services/llm_service.py
-----------------------
Unified LLM abstraction layer for the ADVOCATE backend.

Supported providers
-------------------
  - Anthropic  (Claude)      – ``anthropic`` SDK, requires ANTHROPIC_API_KEY
  - OpenAI     (ChatGPT)     – ``openai`` SDK, requires OPENAI_API_KEY
  - Gemini     (Google)      – ``google-generativeai`` SDK, requires GOOGLE_API_KEY
  - NVIDIA     (free models) – ``openai`` SDK pointed at NVIDIA's endpoint;
                               free tier works with no key, higher throughput
                               with a free NVIDIA_API_KEY from build.nvidia.com
  - Mock                     – deterministic, key-free fallback for unit tests

Default resolution order (first available wins, env keys only):
  1. Anthropic   – if ANTHROPIC_API_KEY is set
  2. OpenAI      – if OPENAI_API_KEY is set
  3. Gemini      – if GOOGLE_API_KEY is set
  4. NVIDIA      – always available (free tier, no key required) ← new default
  5. Mock        – explicit override only (provider=mock in request)

Per-request override
---------------------
  Agents remain provider-agnostic.  The orchestrator may call
  ``create_provider(provider_id, api_key, model)`` to build a request-scoped
  provider that overrides the singleton for that single analysis run.

All providers expose the same async interface:
  ``async def complete(system: str, user: str) -> str``
"""

import json
import os
import random
from abc import ABC, abstractmethod
from typing import Optional

from app.utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Provider IDs (string literals used in the request schema and catalog)
# ---------------------------------------------------------------------------
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_OPENAI = "openai"
PROVIDER_GEMINI = "gemini"
PROVIDER_NVIDIA = "nvidia"
PROVIDER_MOCK = "mock"

# ---------------------------------------------------------------------------
# Supported models catalog – also served by GET /models
# ---------------------------------------------------------------------------
MODELS_CATALOG: dict[str, list[dict]] = {
    PROVIDER_ANTHROPIC: [
        {"id": "claude-3-5-haiku-20241022",  "name": "Claude 3.5 Haiku",   "description": "Fast & efficient",       "default": True},
        {"id": "claude-3-5-sonnet-20241022", "name": "Claude 3.5 Sonnet",  "description": "Balanced performance",   "default": False},
        {"id": "claude-opus-4-5",            "name": "Claude Opus 4.5",    "description": "Most powerful Claude",   "default": False},
    ],
    PROVIDER_OPENAI: [
        {"id": "gpt-4o-mini", "name": "GPT-4o Mini", "description": "Fast & cost-effective",  "default": True},
        {"id": "gpt-4o",      "name": "GPT-4o",      "description": "Balanced performance",   "default": False},
        {"id": "o3-mini",     "name": "o3-mini",     "description": "Advanced reasoning",     "default": False},
    ],
    PROVIDER_GEMINI: [
        {"id": "gemini-2.0-flash",   "name": "Gemini 2.0 Flash",   "description": "Fast & efficient",     "default": True},
        {"id": "gemini-2.5-pro",     "name": "Gemini 2.5 Pro",     "description": "Most powerful Gemini", "default": False},
    ],
    PROVIDER_NVIDIA: [
        {"id": "meta/llama-3.1-70b-instruct",            "name": "Llama 3.1 70B",    "description": "Free – Meta's flagship open model",    "default": True},
        {"id": "mistralai/mixtral-8x7b-instruct-v0.1",   "name": "Mixtral 8x7B",     "description": "Free – Mistral mixture-of-experts",    "default": False},
        {"id": "microsoft/phi-3-medium-128k-instruct",   "name": "Phi-3 Medium",     "description": "Free – Microsoft small language model", "default": False},
        {"id": "google/gemma-7b",                        "name": "Gemma 7B",         "description": "Free – Google open model",             "default": False},
    ],
    PROVIDER_MOCK: [
        {"id": "mock",  "name": "Mock (test)", "description": "Deterministic offline responses", "default": True},
    ],
}

_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


# ===========================================================================
# Abstract base
# ===========================================================================


class BaseLLMProvider(ABC):
    """
    Abstract LLM provider interface.

    Sub-classes wrap a specific API client and expose a single async
    ``complete`` method so agents remain provider-agnostic.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this provider (used in logs / DB)."""

    @abstractmethod
    async def complete(self, system: str, user: str) -> str:
        """
        Send a chat completion request and return the assistant's reply.

        Parameters
        ----------
        system:
            System-level instruction that sets the LLM's persona and output
            format.
        user:
            User-turn prompt containing the idea and task description.

        Returns
        -------
        str
            Raw text response from the model.  Agents parse this into JSON.
        """


# ===========================================================================
# Provider implementations
# ===========================================================================


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic Claude provider.

    Uses the ``anthropic`` Python SDK (>=0.25).  Defaults to
    ``claude-3-5-haiku-20241022``; override via ``ANTHROPIC_MODEL`` env var
    or pass ``model`` explicitly to the constructor.
    """

    def __init__(self, api_key: str, model: Optional[str] = None) -> None:
        try:
            import anthropic  # lazy import
        except ImportError as exc:
            raise RuntimeError(
                "The 'anthropic' package is required. Install: pip install anthropic"
            ) from exc

        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model: str = model or os.getenv(
            "ANTHROPIC_MODEL", "claude-3-5-haiku-20241022"
        )
        log.info("AnthropicProvider initialised (model=%s)", self._model)

    @property
    def name(self) -> str:
        return f"anthropic/{self._model}"

    async def complete(self, system: str, user: str) -> str:
        message = await self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return message.content[0].text


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI Chat Completion provider (ChatGPT).

    Uses the ``openai`` Python SDK (>=1.0).  Defaults to ``gpt-4o-mini``;
    override via ``OPENAI_MODEL`` env var or pass ``model`` explicitly.
    """

    def __init__(self, api_key: str, model: Optional[str] = None) -> None:
        try:
            from openai import AsyncOpenAI  # lazy import
        except ImportError as exc:
            raise RuntimeError(
                "The 'openai' package is required. Install: pip install openai"
            ) from exc

        self._client = AsyncOpenAI(api_key=api_key)
        self._model: str = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        log.info("OpenAIProvider initialised (model=%s)", self._model)

    @property
    def name(self) -> str:
        return f"openai/{self._model}"

    async def complete(self, system: str, user: str) -> str:
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


class GeminiProvider(BaseLLMProvider):
    """
    Google Gemini provider.

    Uses the ``google-generativeai`` Python SDK (>=0.8).  Defaults to
    ``gemini-2.0-flash``; override via ``GOOGLE_MODEL`` env var or pass
    ``model`` explicitly.
    """

    def __init__(self, api_key: str, model: Optional[str] = None) -> None:
        try:
            import google.generativeai as genai  # lazy import
        except ImportError as exc:
            raise RuntimeError(
                "The 'google-generativeai' package is required. "
                "Install: pip install google-generativeai"
            ) from exc

        genai.configure(api_key=api_key)
        self._model_id: str = model or os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
        self._client = genai.GenerativeModel(self._model_id)
        log.info("GeminiProvider initialised (model=%s)", self._model_id)

    @property
    def name(self) -> str:
        return f"gemini/{self._model_id}"

    async def complete(self, system: str, user: str) -> str:
        """
        Gemini uses a combined system+user prompt because the SDK's
        ``system_instruction`` param is simpler for our use-case than
        multi-turn history.
        """
        import asyncio

        combined_prompt = f"{system}\n\n---\n\n{user}"
        # google-generativeai's generate_content is sync; run in executor
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.generate_content(combined_prompt),
        )
        return response.text


class NvidiaProvider(BaseLLMProvider):
    """
    NVIDIA-hosted free open-source models.

    Powered by NVIDIA's OpenAI-compatible API endpoint at
    ``https://integrate.api.nvidia.com/v1``.  No key is required for the
    free tier (anonymous access), making this the **default fallback** when
    no other API keys are configured.

    For higher throughput, users can supply a free NVIDIA API key from
    https://build.nvidia.com

    Key resolution order:
      1. ``api_key`` argument (user-supplied per-request)
      2. ``NVIDIA_API_KEY`` environment variable
      3. Empty string → NVIDIA anonymous free tier
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        try:
            from openai import AsyncOpenAI  # lazy import – reuses openai SDK
        except ImportError as exc:
            raise RuntimeError(
                "The 'openai' package is required for the NVIDIA provider. "
                "Install: pip install openai"
            ) from exc

        resolved_key: str = (
            api_key
            or os.getenv("NVIDIA_API_KEY", "")
        )
        self._client = AsyncOpenAI(
            api_key=resolved_key or "no-key-required",  # SDK requires a non-empty string
            base_url=_NVIDIA_BASE_URL,
        )
        self._model: str = model or os.getenv(
            "NVIDIA_MODEL", "meta/llama-3.1-70b-instruct"
        )
        log.info(
            "NvidiaProvider initialised (model=%s, key_provided=%s)",
            self._model,
            bool(resolved_key),
        )

    @property
    def name(self) -> str:
        return f"nvidia/{self._model}"

    async def complete(self, system: str, user: str) -> str:
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

    Returns realistic, structurally correct JSON responses without any
    network requests.  Retained for explicit test use via ``provider=mock``.
    No longer used as a production fallback (NVIDIA free tier fills that role).
    """

    @property
    def name(self) -> str:
        return "mock"

    async def complete(self, system: str, user: str) -> str:
        system_lower = system.lower()
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
            return self._quality_response(idea_snippet)

    # ------------------------------------------------------------------
    # Mock response templates
    # ------------------------------------------------------------------

    def _quality_response(self, snippet: str) -> str:
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
# Per-request provider factory
# ===========================================================================


def create_provider(
    provider_id: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> BaseLLMProvider:
    """
    Instantiate and return a provider for a single analysis request.

    Parameters
    ----------
    provider_id:
        One of ``anthropic``, ``openai``, ``gemini``, ``nvidia``, ``mock``.
    api_key:
        User-supplied API key for paid providers.  For ``nvidia``, falls back
        to the ``NVIDIA_API_KEY`` env var, then to anonymous free tier.
    model:
        Specific model ID within the provider.  Uses the provider's default
        if omitted.

    Returns
    -------
    BaseLLMProvider
        A freshly constructed provider instance.

    Raises
    ------
    ValueError
        If ``provider_id`` is not recognised.
    """
    if provider_id == PROVIDER_ANTHROPIC:
        if not api_key:
            raise ValueError(
                "An Anthropic API key is required. Provide 'api_key' in the request body."
            )
        return AnthropicProvider(api_key=api_key, model=model)

    if provider_id == PROVIDER_OPENAI:
        if not api_key:
            raise ValueError(
                "An OpenAI API key is required. Provide 'api_key' in the request body."
            )
        return OpenAIProvider(api_key=api_key, model=model)

    if provider_id == PROVIDER_GEMINI:
        if not api_key:
            raise ValueError(
                "A Google API key is required for Gemini. Provide 'api_key' in the request body."
            )
        return GeminiProvider(api_key=api_key, model=model)

    if provider_id == PROVIDER_NVIDIA:
        return NvidiaProvider(api_key=api_key, model=model)

    if provider_id == PROVIDER_MOCK:
        return MockProvider()

    raise ValueError(
        f"Unknown provider '{provider_id}'. "
        f"Valid options: {', '.join([PROVIDER_ANTHROPIC, PROVIDER_OPENAI, PROVIDER_GEMINI, PROVIDER_NVIDIA, PROVIDER_MOCK])}"
    )


# ===========================================================================
# LLMService – wraps a provider; used as the module-level singleton AND as
# a per-request wrapper when the orchestrator creates a scoped provider.
# ===========================================================================


class LLMService:
    """
    LLM service that wraps a :class:`BaseLLMProvider`.

    Default singleton resolution order (env keys):
    -----------------------------------------------
    1. ``ANTHROPIC_API_KEY`` → :class:`AnthropicProvider`
    2. ``OPENAI_API_KEY``    → :class:`OpenAIProvider`
    3. ``GOOGLE_API_KEY``    → :class:`GeminiProvider`
    4. (none)                → :class:`NvidiaProvider` (free tier, always available)

    For per-request overrides, the orchestrator calls ``LLMService(provider)``
    with a freshly built provider from :func:`create_provider`.
    """

    def __init__(self, provider: Optional[BaseLLMProvider] = None) -> None:
        self._provider: BaseLLMProvider = provider or self._resolve_provider()
        log.info("LLMService active provider: %s", self._provider.name)

    # ------------------------------------------------------------------
    # Environment-based provider resolution (startup singleton)
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_provider() -> BaseLLMProvider:
        """
        Inspect environment variables and return the highest-priority
        available provider.  NVIDIA free tier is always the last resort.
        """
        anthropic_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
        openai_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        google_key: Optional[str] = os.getenv("GOOGLE_API_KEY")

        if anthropic_key and anthropic_key.strip():
            log.info("Anthropic API key detected – using AnthropicProvider.")
            return AnthropicProvider(api_key=anthropic_key.strip())

        if openai_key and openai_key.strip():
            log.info("OpenAI API key detected – using OpenAIProvider.")
            return OpenAIProvider(api_key=openai_key.strip())

        if google_key and google_key.strip():
            log.info("Google API key detected – using GeminiProvider.")
            return GeminiProvider(api_key=google_key.strip())

        log.info(
            "No paid LLM API keys found. Using NvidiaProvider (free tier) as default. "
            "Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GOOGLE_API_KEY for a specific provider, "
            "or NVIDIA_API_KEY for higher NVIDIA throughput."
        )
        return NvidiaProvider()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def complete(self, system: str, user: str) -> str:
        """
        Proxy a completion request to the active provider.

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
# Module-level singleton – resolved from environment variables at startup.
# Import and use ``llm_service`` directly in agents/orchestrator unless a
# per-request override is needed.
# ---------------------------------------------------------------------------
llm_service = LLMService()
