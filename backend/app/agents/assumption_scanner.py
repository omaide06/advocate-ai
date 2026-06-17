"""
agents/assumption_scanner.py
-----------------------------
Agent 2 – Assumption Scanner.

RESPONSIBILITY:
  Detect hidden, unstated, or dangerous assumptions embedded in the idea.
  For each assumption found, provide:
  - The assumption text.
  - A severity rating: low | medium | high | critical.
  - An explanation of why the assumption is risky or potentially wrong.

AGENT FLOW:
  1. Receive the idea, context, attack_intensity, and quality_reasoning from
     the orchestrator.
  2. Adjust the system prompt based on attack_intensity:
     - aggressive → find as many assumptions as possible (5–7).
     - balanced   → focus on the most impactful assumptions (3–5).
     - surgical   → identify only the hidden, non-obvious ones (2–4).
  3. Call the shared LLM service.
  4. Parse and validate the JSON response.
  5. Return a list of ``AssumptionItem``-compatible dicts.

WHY ASSUMPTIONS MATTER:
  Most ideas fail not because their logic is wrong, but because they rely on
  assumptions that are never explicitly stated and therefore never tested.
  Surfacing these assumptions is often the highest-leverage critique.
"""

import json
from dataclasses import dataclass, field
from typing import List

from app.services.llm_service import LLMService
from app.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class AssumptionResult:
    """
    A single detected hidden assumption.

    Attributes
    ----------
    assumption:   The assumption text.
    severity:     Risk level: "low" | "medium" | "high" | "critical".
    explanation:  Why this assumption is problematic.
    """

    assumption: str
    severity: str
    explanation: str


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------


class AssumptionScannerAgent:
    """
    Scans the submitted idea for hidden, unstated, or fragile assumptions.

    The depth of scanning adapts to the ``attack_intensity`` provided by the
    Quality Assessor so that weak ideas are examined exhaustively while strong
    ideas are probed for subtle, non-obvious assumptions only.

    Parameters
    ----------
    llm:
        Shared :class:`~app.services.llm_service.LLMService` instance.
    """

    # Base system prompt – attack depth instruction is injected dynamically.
    _SYSTEM_TEMPLATE = """\
You are a critical thinking expert specialised in uncovering hidden assumptions.
Your task is to identify the unstated premises that the following idea depends on.

Attack intensity: {intensity}
{intensity_instruction}

For each assumption found, assess:
- How risky it is if wrong: low | medium | high | critical
- A clear explanation of why it could be dangerously wrong

You MUST respond with valid JSON only. No prose before or after the JSON block.

Required JSON schema:
{{
  "assumptions": [
    {{
      "assumption": "<concise statement of the assumption>",
      "severity": "<low | medium | high | critical>",
      "explanation": "<why this assumption is risky or flawed, 2–3 sentences>"
    }}
  ]
}}
"""

    _INTENSITY_INSTRUCTIONS: dict[str, str] = {
        "aggressive": (
            "AGGRESSIVE MODE: Find every assumption you can — obvious and subtle. "
            "Aim for 5 to 7 assumptions. Be relentless."
        ),
        "balanced": (
            "BALANCED MODE: Focus on the 3 to 5 most impactful assumptions. "
            "Cover both obvious and non-obvious ones."
        ),
        "surgical": (
            "SURGICAL MODE: Ignore trivial assumptions. Identify only the 2 to 4 "
            "deep, hidden, non-obvious assumptions that a naive reader would miss entirely."
        ),
    }

    def __init__(self, llm: LLMService) -> None:
        self._llm = llm

    async def scan(
        self,
        idea: str,
        attack_intensity: str,
        context: str | None = None,
        quality_reasoning: str | None = None,
    ) -> List[AssumptionResult]:
        """
        Detect hidden assumptions in the idea.

        Parameters
        ----------
        idea:
            The idea text.
        attack_intensity:
            "aggressive" | "balanced" | "surgical" – controls scan depth.
        context:
            Optional user-provided background information.
        quality_reasoning:
            Reasoning text from the Quality Assessor (provided to the LLM as
            additional signal about the idea's weaknesses).

        Returns
        -------
        List[AssumptionResult]
            A list of detected assumptions with severity and explanation.
        """
        log.debug(
            "AssumptionScanner: scanning (intensity=%s, idea_len=%d)",
            attack_intensity,
            len(idea),
        )

        intensity_instruction = self._INTENSITY_INSTRUCTIONS.get(
            attack_intensity, self._INTENSITY_INSTRUCTIONS["balanced"]
        )

        system_prompt = self._SYSTEM_TEMPLATE.format(
            intensity=attack_intensity.upper(),
            intensity_instruction=intensity_instruction,
        )

        user_prompt = f"Idea to analyse:\n{idea}"
        if context:
            user_prompt += f"\n\nContext:\n{context}"
        if quality_reasoning:
            user_prompt += f"\n\nPrior quality assessment reasoning:\n{quality_reasoning}"

        raw_response = await self._llm.complete(system=system_prompt, user=user_prompt)

        assumptions = self._parse_response(raw_response)

        log.info(
            "AssumptionScanner: found %d assumptions (intensity=%s)",
            len(assumptions),
            attack_intensity,
        )
        return assumptions

    def _parse_response(self, raw: str) -> List[AssumptionResult]:
        """
        Parse the LLM's JSON response into a list of :class:`AssumptionResult`.

        On parse failure, returns a single placeholder assumption so the
        pipeline continues rather than aborting.
        """
        try:
            data = json.loads(raw.strip())
            items = data.get("assumptions", [])
            results: List[AssumptionResult] = []
            for item in items:
                results.append(
                    AssumptionResult(
                        assumption=str(item.get("assumption", "Unknown assumption")),
                        severity=str(item.get("severity", "medium")),
                        explanation=str(item.get("explanation", "")),
                    )
                )
            return results
        except (json.JSONDecodeError, TypeError, AttributeError) as exc:
            log.error("AssumptionScanner: parse error – %s", exc)
            return [
                AssumptionResult(
                    assumption="Analysis encountered a parsing error.",
                    severity="low",
                    explanation=(
                        "The assumption scanner could not parse the model's response. "
                        "This is a transient error; retry the request for a full analysis."
                    ),
                )
            ]
