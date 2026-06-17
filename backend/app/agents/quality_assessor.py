"""
agents/quality_assessor.py
--------------------------
Agent 1 – Quality Assessor.

RESPONSIBILITY:
  Evaluate the submitted idea and produce:
  - A numeric quality score from 1.0 (very poor) to 5.0 (excellent).
  - A human-readable quality label (e.g. "Poor", "Moderate", "Strong").
  - An attack intensity level derived from the score:
      * Score 1–2 → aggressive  (idea is weak; attack hard)
      * Score 3   → balanced    (idea has some merit; balanced critique)
      * Score 4–5 → surgical    (idea is strong; find precise weak spots)

AGENT FLOW:
  1. Build a system prompt that instructs the LLM to act as a rigorous
     idea evaluator who returns structured JSON.
  2. Build a user prompt that includes the raw idea text and optional context.
  3. Call the shared LLM service.
  4. Parse the JSON response and map score → intensity.
  5. Return a typed ``QualityResult`` dataclass to the orchestrator.

This agent intentionally does NOT perform assumption detection or argument
generation – those are handled by downstream agents.  Single-responsibility
keeps each agent's prompt focused and its output predictable.
"""

import json
from dataclasses import dataclass

from app.services.llm_service import LLMService
from app.utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class QualityResult:
    """
    Output of the Quality Assessor agent.

    Attributes
    ----------
    score:            Float 1.0–5.0 representing idea quality.
    quality_label:    Human-readable bucket label.
    attack_intensity: Derived intensity for downstream agents.
    reasoning:        Internal reasoning text (not exposed in final API response).
    """

    score: float
    quality_label: str
    attack_intensity: str
    reasoning: str


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------


class QualityAssessorAgent:
    """
    Evaluates the quality of a submitted idea and determines attack intensity.

    This is the first agent in the ADVOCATE pipeline.  Its output
    (specifically ``attack_intensity``) governs how aggressively the
    downstream Assumption Scanner and Steelman Generator will challenge
    the idea.

    Parameters
    ----------
    llm:
        A shared :class:`~app.services.llm_service.LLMService` instance
        injected by the orchestrator.
    """

    # System prompt that sets the LLM's persona and output format.
    _SYSTEM_PROMPT = """\
You are a rigorous idea quality evaluator. Your job is to score an idea on a \
scale of 1 to 5 based on its intellectual coherence, practical viability, \
originality, and evidence of critical thinking.

Scoring rubric:
  1 – Very poor: vague, incoherent, or clearly harmful.
  2 – Poor: some merit but fundamentally flawed.
  3 – Moderate: reasonable but lacks depth or novelty.
  4 – Strong: well-reasoned with minor weaknesses.
  5 – Excellent: rigorous, original, and evidence-grounded.

You MUST respond with valid JSON only. No prose before or after the JSON block.

Required JSON schema:
{
  "score": <float between 1.0 and 5.0>,
  "quality_label": <"Very Poor" | "Poor" | "Moderate" | "Strong" | "Excellent">,
  "attack_intensity": <"aggressive" | "balanced" | "surgical">,
  "reasoning": <string – your internal reasoning, 2–4 sentences>
}

Attack intensity mapping:
  score 1–2 → "aggressive"
  score 3   → "balanced"
  score 4–5 → "surgical"
"""

    def __init__(self, llm: LLMService) -> None:
        self._llm = llm

    async def assess(self, idea: str, context: str | None = None) -> QualityResult:
        """
        Score the idea and derive the attack intensity.

        Parameters
        ----------
        idea:
            The idea text to evaluate.
        context:
            Optional background information the user provided.

        Returns
        -------
        QualityResult
            Typed result containing score, label, intensity, and reasoning.

        Raises
        ------
        ValueError
            If the LLM response cannot be parsed as valid JSON or the score
            is outside the expected 1–5 range.
        """
        log.debug("QualityAssessor: assessing idea (length=%d chars)", len(idea))

        user_prompt = f"Idea to evaluate:\n{idea}"
        if context:
            user_prompt += f"\n\nAdditional context provided by the user:\n{context}"

        raw_response = await self._llm.complete(
            system=self._SYSTEM_PROMPT,
            user=user_prompt,
        )

        try:
            data = json.loads(raw_response.strip())
        except json.JSONDecodeError as exc:
            log.error("QualityAssessor: failed to parse LLM response as JSON: %s", exc)
            # Graceful degradation: return a moderate score so the pipeline
            # can continue rather than hard-failing on a parse error.
            return QualityResult(
                score=3.0,
                quality_label="Moderate",
                attack_intensity="balanced",
                reasoning="Score estimation failed; defaulting to moderate.",
            )

        # Clamp score to valid range.
        raw_score = float(data.get("score", 3.0))
        score = max(1.0, min(5.0, raw_score))

        # Derive attack intensity from score (in case LLM got it wrong).
        attack_intensity = self._derive_intensity(score)

        result = QualityResult(
            score=round(score, 2),
            quality_label=data.get("quality_label", _label_for_score(score)),
            attack_intensity=attack_intensity,
            reasoning=data.get("reasoning", ""),
        )

        log.info(
            "QualityAssessor: score=%.1f  label=%s  intensity=%s",
            result.score,
            result.quality_label,
            result.attack_intensity,
        )
        return result

    @staticmethod
    def _derive_intensity(score: float) -> str:
        """
        Map a numeric score to an attack intensity string.

        Attack intensity logic
        ----------------------
        - Scores 1–2 (weak ideas) receive **aggressive** attacks because weak
          ideas have many fundamental flaws worth exposing thoroughly.
        - Score 3 (middling ideas) receives a **balanced** critique that
          acknowledges merits alongside weaknesses.
        - Scores 4–5 (strong ideas) receive **surgical** attacks that zero
          in on the precise, non-obvious weaknesses of an otherwise solid
          proposal.
        """
        if score <= 2.0:
            return "aggressive"
        elif score <= 3.0:
            return "balanced"
        else:
            return "surgical"


def _label_for_score(score: float) -> str:
    """Return the quality label bucket for a given score."""
    if score <= 1.5:
        return "Very Poor"
    elif score <= 2.5:
        return "Poor"
    elif score <= 3.5:
        return "Moderate"
    elif score <= 4.5:
        return "Strong"
    else:
        return "Excellent"
