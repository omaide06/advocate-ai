"""
agents/orchestrator.py
-----------------------
Central Orchestrator – coordinates all ADVOCATE agents.

ORCHESTRATOR FLOW:
  The orchestrator is the brain of the ADVOCATE pipeline.  It:
  1. Receives a validated analysis request from the API layer.
  2. Runs the Quality Assessor to score the idea and determine attack intensity.
  3. Runs the Assumption Scanner (always).
  4. Runs the Steelman Generator (always in standard/deep mode; skipped in quick).
  5. Runs the Formatter to produce the final verdict + summary.
  6. Assembles the complete result dict.
  7. Persists the session to the database.
  8. Returns the fully assembled response dict.

AGENT SEQUENCING:
  Agents 2 and 3 (Assumption Scanner + Steelman Generator) are run
  concurrently using ``asyncio.gather`` in standard/deep mode because neither
  depends on the other's output – they both only need the Quality Assessor's
  output.  The Formatter runs last because it needs all prior outputs.

  In QUICK mode, only the Quality Assessor and Assumption Scanner run
  (no steelmanning, no formatter) and a simplified response is returned.

  In DEEP mode, the Assumption Scanner runs an extra pass with a higher
  assumption count before the Steelman Generator is invoked.

DATABASE INTERACTION:
  After all agents complete, the orchestrator constructs an
  ``AnalysisSession`` ORM object and adds it to the database via the
  injected ``AsyncSession``.  This keeps persistence logic outside the
  agents themselves, which remain pure data-transformation units.
"""

import asyncio
import time
import uuid
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.assumption_scanner import AssumptionResult, AssumptionScannerAgent
from app.agents.formatter import FormatterAgent, FormatterResult
from app.agents.quality_assessor import QualityAssessorAgent, QualityResult
from app.agents.steelman_generator import CounterArgumentResult, SteelmanGeneratorAgent
from app.models.session import AnalysisSession
from app.schemas.request import AnalysisMode
from app.services.llm_service import LLMService, create_provider, llm_service
from app.utils.logger import get_logger

log = get_logger(__name__)


class AnalysisOrchestrator:
    """
    Coordinates all ADVOCATE agents for a single analysis run.

    Each call to :meth:`run` is stateless – the orchestrator can safely handle
    concurrent requests because it creates fresh agent instances per call
    and passes the database session in from outside.

    Parameters
    ----------
    llm:
        The shared LLM service singleton.  Injected at construction time
        so the orchestrator can be tested with a mock LLM without touching
        module globals.
    """

    def __init__(self, llm: LLMService = llm_service) -> None:
        # Instantiate all four agents, all sharing the same LLM service.
        self._quality_assessor = QualityAssessorAgent(llm=llm)
        self._assumption_scanner = AssumptionScannerAgent(llm=llm)
        self._steelman_generator = SteelmanGeneratorAgent(llm=llm)
        self._formatter = FormatterAgent(llm=llm)
        self._llm = llm

    async def run(
        self,
        idea: str,
        mode: AnalysisMode,
        context: Optional[str],
        db: AsyncSession,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> dict:
        """
        Execute the full ADVOCATE multi-agent analysis pipeline.

        Parameters
        ----------
        idea:
            The idea text to analyse.
        mode:
            ``AnalysisMode`` controlling pipeline depth.
        context:
            Optional user-supplied background information.
        db:
            An open ``AsyncSession`` from FastAPI's dependency injection.
            The orchestrator adds the new ``AnalysisSession`` record and the
            caller's dependency commits the transaction on clean exit.
        provider:
            Optional provider ID override for this request
            (``anthropic`` | ``openai`` | ``gemini`` | ``nvidia`` | ``mock``).
            If ``None``, the singleton ``llm_service`` is used (env-key priority).
        model:
            Specific model ID within the chosen provider.  Provider default
            is used if ``None``.
        api_key:
            User-supplied API key.  Required for paid providers; ignored for
            ``nvidia`` (falls back to env var / anonymous) and ``mock``.

        Returns
        -------
        dict
            Fully assembled analysis result dictionary ready to be
            serialised into an :class:`~app.schemas.response.AnalysisResponse`.

        Raises
        ------
        ValueError
            If ``provider`` is unrecognised or a required ``api_key`` is missing.
        RuntimeError
            If a critical agent fails in a way that cannot be recovered.
        """
        session_id = str(uuid.uuid4())
        start_time = time.monotonic()

        # ---------------------------------------------------------------
        # Resolve LLM service: per-request override OR module singleton.
        # ---------------------------------------------------------------
        if provider is not None:
            request_provider = create_provider(
                provider_id=provider,
                api_key=api_key,
                model=model,
            )
            active_llm = LLMService(provider=request_provider)
        else:
            active_llm = self._llm

        # Rebuild agents with the resolved LLM for this request.
        from app.agents.assumption_scanner import AssumptionScannerAgent
        from app.agents.formatter import FormatterAgent
        from app.agents.quality_assessor import QualityAssessorAgent
        from app.agents.steelman_generator import SteelmanGeneratorAgent

        quality_assessor = QualityAssessorAgent(llm=active_llm)
        assumption_scanner = AssumptionScannerAgent(llm=active_llm)
        steelman_generator = SteelmanGeneratorAgent(llm=active_llm)
        formatter = FormatterAgent(llm=active_llm)

        log.info(
            "Orchestrator: starting analysis session=%s mode=%s provider=%s idea_len=%d",
            session_id,
            mode.value,
            active_llm.get_provider_name(),
            len(idea),
        )

        # ---------------------------------------------------------------
        # STEP 1 – Quality Assessment
        # Always runs first.  Its score determines attack_intensity which
        # every subsequent agent needs.
        # ---------------------------------------------------------------
        quality: QualityResult = await quality_assessor.assess(
            idea=idea, context=context
        )
        log.debug(
            "Orchestrator: quality step complete – score=%.1f intensity=%s",
            quality.score,
            quality.attack_intensity,
        )

        # ---------------------------------------------------------------
        # STEP 2 – Assumption Scanning (always runs)
        # In DEEP mode we pass a hint to the scanner to be extra thorough.
        # ---------------------------------------------------------------
        # In deep mode, override intensity to always be "aggressive" for the
        # assumption scan regardless of score, giving us maximum coverage.
        scan_intensity = (
            "aggressive" if mode == AnalysisMode.deep else quality.attack_intensity
        )

        if mode == AnalysisMode.quick:
            # QUICK mode: run assumption scanner only, skip steelmanning.
            log.debug("Orchestrator: quick mode – skipping steelman generator.")
            assumptions: list[AssumptionResult] = await assumption_scanner.scan(
                idea=idea,
                attack_intensity=scan_intensity,
                context=context,
                quality_reasoning=quality.reasoning,
            )
            counter_arguments: list[CounterArgumentResult] = []
            formatter_result = FormatterResult(
                verdict=(
                    f"Quick analysis complete. The idea scored {quality.score}/5.0 "
                    f"({quality.quality_label}). {len(assumptions)} assumption(s) were "
                    "detected. Run in standard or deep mode for a full critique."
                ),
                summary=(
                    f"Scored {quality.score}/5.0 ({quality.quality_label}). "
                    f"{len(assumptions)} hidden assumption(s) found. Full critique requires standard/deep mode."
                ),
            )
        else:
            # STANDARD / DEEP mode: run assumption scanner AND steelman
            # generator concurrently (they are independent of each other).
            log.debug(
                "Orchestrator: running assumption scanner + steelman concurrently."
            )
            assumptions, counter_arguments = await asyncio.gather(
                assumption_scanner.scan(
                    idea=idea,
                    attack_intensity=scan_intensity,
                    context=context,
                    quality_reasoning=quality.reasoning,
                ),
                steelman_generator.generate(
                    idea=idea,
                    attack_intensity=quality.attack_intensity,
                    assumptions=[],  # First pass without assumptions for speed;
                    # the formatter will still see both lists.
                    context=context,
                ),
            )

            # In DEEP mode, run a second steelman pass that exploits the
            # assumptions discovered in the first concurrent pass.
            if mode == AnalysisMode.deep:
                log.debug("Orchestrator: deep mode – running second steelman pass.")
                deep_counter_args = await steelman_generator.generate(
                    idea=idea,
                    attack_intensity="aggressive",  # deep mode is always aggressive
                    assumptions=assumptions,
                    context=context,
                )
                # Merge the two passes, deduplicate by argument text.
                seen: set[str] = {c.argument for c in counter_arguments}
                for ca in deep_counter_args:
                    if ca.argument not in seen:
                        counter_arguments.append(ca)
                        seen.add(ca.argument)

            # STEP 3 – Final Formatter (synthesis)
            formatter_result = await formatter.format(
                idea=idea,
                score=quality.score,
                quality_label=quality.quality_label,
                attack_intensity=quality.attack_intensity,
                assumptions=assumptions,
                counter_arguments=counter_arguments,
                context=context,
            )

        # ---------------------------------------------------------------
        # STEP 4 – Assemble result payload
        # ---------------------------------------------------------------
        elapsed = round(time.monotonic() - start_time, 3)

        # Serialise agent dataclasses to plain dicts for JSON storage.
        assumptions_dicts = [
            {
                "assumption": a.assumption,
                "severity": a.severity,
                "explanation": a.explanation,
            }
            for a in assumptions
        ]

        counter_arguments_dicts = [
            {
                "argument": ca.argument,
                "strength": ca.strength,
                "evidence_type": ca.evidence_type,
            }
            for ca in counter_arguments
        ]

        provider_name = active_llm.get_provider_name()

        # ---------------------------------------------------------------
        # DATABASE INTERACTION:
        # Build the ORM object and add to the session.  The calling
        # dependency (get_db) will commit the transaction after this
        # coroutine returns.
        # ---------------------------------------------------------------
        db_session = AnalysisSession(
            session_id=session_id,
            idea=idea,
            mode=mode.value,
            score=quality.score,
            quality_label=quality.quality_label,
            attack_intensity=quality.attack_intensity,
            assumptions=assumptions_dicts,
            counter_arguments=counter_arguments_dicts,
            verdict=formatter_result.verdict,
            summary=formatter_result.summary,
            llm_provider=provider_name,
            processing_time_seconds=elapsed,
        )
        db.add(db_session)
        # Note: commit happens in the get_db() FastAPI dependency after this
        # function returns, not here.  This keeps transaction boundaries clean.

        log.info(
            "Orchestrator: session=%s complete in %.3fs (provider=%s)",
            session_id,
            elapsed,
            provider_name,
        )

        # Note: api_key is intentionally NOT included in the returned dict
        # or stored in the database.

        # Return a flat dict that maps 1:1 with AnalysisResponse schema.
        return {
            "session_id": session_id,
            "idea": idea,
            "mode": mode.value,
            "score": quality.score,
            "quality_label": quality.quality_label,
            "attack_intensity": quality.attack_intensity,
            "assumptions": assumptions_dicts,
            "counter_arguments": counter_arguments_dicts,
            "verdict": formatter_result.verdict,
            "summary": formatter_result.summary,
            "processing_time_seconds": elapsed,
            "created_at": db_session.created_at,
            "llm_provider": provider_name,
        }
