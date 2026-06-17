"""
agents/formatter.py
--------------------
Agent 4 – Formatter.

RESPONSIBILITY:
  Synthesise all prior agent outputs into a coherent human-readable verdict
  and a short summary.

  Inputs from prior agents:
  - Idea text
  - Quality score + label + attack_intensity
  - List of hidden assumptions
  - List of counter-arguments

  Outputs:
  - ``verdict``:  A detailed paragraph (150–250 words) that constitutes the
                  final ruling from ADVOCATE.
  - ``summary``:  A short ≤ 3-sentence TL;DR for users who want the gist.

AGENT FLOW:
  1. Receive all prior agent outputs via the orchestrator.
  2. Build a consolidation prompt that surfaces the most important findings.
  3. Call the LLM service to generate the synthesised verdict.
  4. Parse and return the verdict + summary pair.

WHY A DEDICATED FORMATTER?
  Without a dedicated synthesis step, the verdict would either be assembled
  by the orchestrator (mechanical, lifeless) or by the last analytical agent
  (which might skew toward its own output).  A separate agent ensures the
  verdict integrates all signals fairly and maintains a consistent authorial
  voice across analysis runs.
"""

import json
from dataclasses import dataclass
from typing import List

from app.agents.assumption_scanner import AssumptionResult
from app.agents.steelman_generator import CounterArgumentResult
from app.services.llm_service import LLMService
from app.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class FormatterResult:
    """
    Final synthesised output from the Formatter agent.

    Attributes
    ----------
    verdict: Detailed final ruling paragraph.
    summary: Short TL;DR (≤ 3 sentences).
    """

    verdict: str
    summary: str


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------


class FormatterAgent:
    """
    Synthesises all prior agent outputs into a final verdict and TL;DR.

    Parameters
    ----------
    llm:
        Shared :class:`~app.services.llm_service.LLMService` instance.
    """

    _SYSTEM_PROMPT = """\
You are ADVOCATE – an AI designed to challenge and stress-test ideas. You have \
just completed a full multi-agent analysis of a submitted idea. Your task is \
now to deliver a final, authoritative verdict.

Write with clarity, precision, and intellectual authority. Acknowledge what \
is genuinely good in the idea (if anything) before delivering the core critique. \
Do NOT be unnecessarily harsh, but do NOT pull punches either.

You MUST respond with valid JSON only. No prose before or after the JSON block.

Required JSON schema:
{{
  "verdict": "<final ruling paragraph, 150–250 words, integrating score, assumptions, and counter-arguments>",
  "summary": "<TL;DR of 1–3 sentences capturing the essential judgement>"
}}
"""

    def __init__(self, llm: LLMService) -> None:
        self._llm = llm

    async def format(
        self,
        idea: str,
        score: float,
        quality_label: str,
        attack_intensity: str,
        assumptions: List[AssumptionResult],
        counter_arguments: List[CounterArgumentResult],
        context: str | None = None,
    ) -> FormatterResult:
        """
        Generate the final verdict and summary.

        Parameters
        ----------
        idea:              The original idea text.
        score:             Quality score from the Quality Assessor.
        quality_label:     Human-readable score label.
        attack_intensity:  The intensity level used in this run.
        assumptions:       Assumptions from the Assumption Scanner.
        counter_arguments: Counter-arguments from the Steelman Generator.
        context:           Optional user-provided background information.

        Returns
        -------
        FormatterResult
            Dataclass holding the verdict and summary strings.
        """
        log.debug("Formatter: synthesising verdict (score=%.1f)", score)

        # Build a structured summary of all prior agent outputs for the prompt.
        assumptions_text = "\n".join(
            f"  [{a.severity.upper()}] {a.assumption}" for a in assumptions
        ) or "  None detected."

        counter_args_text = "\n".join(
            f"  [{c.strength.upper()} / {c.evidence_type}] {c.argument}"
            for c in counter_arguments
        ) or "  None generated."

        user_prompt = f"""Idea submitted for analysis:
\"{idea}\"

{f'Context: {context}' if context else ''}

Quality Assessment:
  Score: {score}/5.0
  Label: {quality_label}
  Attack intensity used: {attack_intensity}

Hidden Assumptions Detected:
{assumptions_text}

Strongest Counter-Arguments:
{counter_args_text}

Based on all of the above, deliver the final ADVOCATE verdict.
"""

        raw_response = await self._llm.complete(
            system=self._SYSTEM_PROMPT, user=user_prompt
        )
        result = self._parse_response(raw_response, idea, score)

        log.info(
            "Formatter: verdict generated (length=%d chars)", len(result.verdict)
        )
        return result

    def _parse_response(
        self, raw: str, idea: str, score: float
    ) -> FormatterResult:
        """
        Parse the LLM's JSON response into a :class:`FormatterResult`.

        Falls back to a basic mechanical verdict if JSON parsing fails,
        ensuring the API always returns a complete response.
        """
        try:
            data = json.loads(raw.strip())
            return FormatterResult(
                verdict=str(data.get("verdict", "")),
                summary=str(data.get("summary", "")),
            )
        except (json.JSONDecodeError, TypeError, AttributeError) as exc:
            log.error("Formatter: parse error – %s. Using fallback verdict.", exc)
            snippet = idea[:60].rstrip()
            return FormatterResult(
                verdict=(
                    f"ADVOCATE has analysed the idea '{snippet}…' and assigned it a quality "
                    f"score of {score}/5.0. Multiple hidden assumptions and counter-arguments "
                    "were identified during analysis. A full synthesis could not be generated "
                    "due to a formatting error; the structured data in this response remains "
                    "complete and accurate."
                ),
                summary=(
                    f"Idea scored {score}/5.0. Key assumptions and counter-arguments are "
                    "detailed in the structured response fields. Verdict synthesis encountered "
                    "a transient error."
                ),
            )
