"""
agents/steelman_generator.py
-----------------------------
Agent 3 – Steelman Generator.

RESPONSIBILITY:
  Generate the strongest possible counter-arguments against the idea.

  "Steelmanning" is the opposite of strawmanning: we construct the most
  powerful, well-evidenced version of the opposing view.  ADVOCATE inverts
  this for the user's idea – it steelmans the *opposition* to find the
  hardest-hitting critiques.

  For each counter-argument:
  - Provide the argument text.
  - Rate its strength: weak | moderate | strong | devastating.
  - Identify the evidence type: empirical | logical | historical | economic |
    ethical | scientific.

AGENT FLOW:
  1. Receive idea, attack_intensity, and assumption list from orchestrator.
  2. Adjust system prompt aggressiveness based on attack_intensity.
  3. Incorporate known assumptions so the LLM can build arguments that
     directly exploit them.
  4. Call LLM service, parse JSON, return list of counter-argument dicts.
"""

import json
from dataclasses import dataclass
from typing import List

from app.agents.assumption_scanner import AssumptionResult
from app.services.llm_service import LLMService
from app.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class CounterArgumentResult:
    """
    A single counter-argument generated against the idea.

    Attributes
    ----------
    argument:      The counter-argument text.
    strength:      "weak" | "moderate" | "strong" | "devastating".
    evidence_type: The category of evidence backing the argument.
    """

    argument: str
    strength: str
    evidence_type: str


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------


class SteelmanGeneratorAgent:
    """
    Generates the strongest possible counter-arguments against the idea.

    Adapts its aggression level to the ``attack_intensity`` determined by
    the Quality Assessor:
    - aggressive → 4–6 arguments, prioritise devastating ones.
    - balanced   → 3–4 arguments, mix of strong and moderate.
    - surgical   → 2–3 arguments, only the most precise critiques.

    Parameters
    ----------
    llm:
        Shared :class:`~app.services.llm_service.LLMService` instance.
    """

    _SYSTEM_TEMPLATE = """\
You are a world-class devil's advocate and critical analyst. Your task is to \
generate the most powerful counter-arguments against the submitted idea.

Attack intensity: {intensity}
{intensity_instruction}

Use the known hidden assumptions below (if any) as ammunition – build \
arguments that exploit these weaknesses directly.

Known hidden assumptions:
{assumptions_block}

You MUST respond with valid JSON only. No prose before or after the JSON block.

Required JSON schema:
{{
  "counter_arguments": [
    {{
      "argument": "<the counter-argument, 2–4 sentences, specific and hard-hitting>",
      "strength": "<weak | moderate | strong | devastating>",
      "evidence_type": "<empirical | logical | historical | economic | ethical | scientific>"
    }}
  ]
}}
"""

    _INTENSITY_INSTRUCTIONS: dict[str, str] = {
        "aggressive": (
            "AGGRESSIVE MODE: Generate 4 to 6 counter-arguments. "
            "Prioritise devastating and strong arguments. "
            "Be merciless. Assume the idea is deeply flawed."
        ),
        "balanced": (
            "BALANCED MODE: Generate 3 to 4 counter-arguments. "
            "Balance strong critiques with acknowledgment of partial merit where genuine."
        ),
        "surgical": (
            "SURGICAL MODE: Generate 2 to 3 counter-arguments. "
            "Focus only on the precise, non-obvious weaknesses that a strong version "
            "of this idea would still have to grapple with. Avoid generic critiques."
        ),
    }

    def __init__(self, llm: LLMService) -> None:
        self._llm = llm

    async def generate(
        self,
        idea: str,
        attack_intensity: str,
        assumptions: List[AssumptionResult],
        context: str | None = None,
    ) -> List[CounterArgumentResult]:
        """
        Generate counter-arguments against the idea.

        Parameters
        ----------
        idea:
            The idea text.
        attack_intensity:
            Controls the number and severity of generated arguments.
        assumptions:
            Assumptions from :class:`AssumptionScannerAgent` – used as
            ammunition to make counter-arguments sharper.
        context:
            Optional user-provided background information.

        Returns
        -------
        List[CounterArgumentResult]
            A list of counter-arguments with strength and evidence-type labels.
        """
        log.debug(
            "SteelmanGenerator: generating (intensity=%s, assumptions=%d)",
            attack_intensity,
            len(assumptions),
        )

        intensity_instruction = self._INTENSITY_INSTRUCTIONS.get(
            attack_intensity, self._INTENSITY_INSTRUCTIONS["balanced"]
        )

        # Build a readable assumptions block to include in the prompt.
        if assumptions:
            assumptions_block = "\n".join(
                f"- [{a.severity.upper()}] {a.assumption}" for a in assumptions
            )
        else:
            assumptions_block = "None identified."

        system_prompt = self._SYSTEM_TEMPLATE.format(
            intensity=attack_intensity.upper(),
            intensity_instruction=intensity_instruction,
            assumptions_block=assumptions_block,
        )

        user_prompt = f"Idea to challenge:\n{idea}"
        if context:
            user_prompt += f"\n\nContext:\n{context}"

        raw_response = await self._llm.complete(system=system_prompt, user=user_prompt)
        results = self._parse_response(raw_response)

        log.info(
            "SteelmanGenerator: produced %d counter-arguments (intensity=%s)",
            len(results),
            attack_intensity,
        )
        return results

    def _parse_response(self, raw: str) -> List[CounterArgumentResult]:
        """
        Parse the LLM's JSON response into a list of
        :class:`CounterArgumentResult` objects.

        Returns a single fallback item on parse failure so the pipeline
        never hard-crashes due to a malformed LLM response.
        """
        try:
            data = json.loads(raw.strip())
            items = data.get("counter_arguments", [])
            results: List[CounterArgumentResult] = []
            for item in items:
                results.append(
                    CounterArgumentResult(
                        argument=str(item.get("argument", "Unspecified argument")),
                        strength=str(item.get("strength", "moderate")),
                        evidence_type=str(item.get("evidence_type", "logical")),
                    )
                )
            return results
        except (json.JSONDecodeError, TypeError, AttributeError) as exc:
            log.error("SteelmanGenerator: parse error – %s", exc)
            return [
                CounterArgumentResult(
                    argument=(
                        "Counter-argument generation encountered a parsing error. "
                        "Please retry for a complete analysis."
                    ),
                    strength="moderate",
                    evidence_type="logical",
                )
            ]
