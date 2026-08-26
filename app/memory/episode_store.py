"""Episode store: persists query execution traces to PostgreSQL for experience memory."""

import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.base import EpisodeRecord

logger = get_logger(__name__)


class EpisodeStore:
    """Writes EpisodeRecord rows to PostgreSQL after each agent pipeline run.

    Gracefully no-ops when ``db`` is ``None`` so offline and test runs are
    unaffected.
    """

    async def store(
        self,
        db: AsyncSession | None,
        *,
        question: str,
        query_type: str | None = None,
        retrieval_strategy: str | None = None,
        plan: str | None = None,
        final_answer: str | None = None,
        final_decision: str = "accept",
        critic_score: float | None = None,
        confidence: float | None = None,
        evidence_coverage: float | None = None,
        repair_attempts: int = 0,
        was_killed: bool = False,
        latency_ms: float | None = None,
        cost: float | None = None,
        issues: list[str] | None = None,
        workspace_id: str | None = None,
    ) -> "EpisodeRecord | None":
        """Persist a completed query episode to the database.

        Args:
            db: Active async SQLAlchemy session. Pass ``None`` to silently skip.
            question: Original user question.
            query_type: Planner query classification.
            retrieval_strategy: Strategy selected by the Planner.
            plan: Plain-text planning summary.
            final_answer: Final accepted/refused answer text.
            final_decision: 'accept' or 'kill'.
            critic_score: Critic agent score (0.0-1.0).
            confidence: System-level confidence estimate (0.0-1.0).
            evidence_coverage: Fraction of claims that were SUPPORTED.
            repair_attempts: How many repair iterations were triggered.
            was_killed: True if the pipeline was hard-killed.
            latency_ms: Total pipeline wall-clock latency in milliseconds.
            cost: Cumulative token/API cost estimate.
            issues: List of critic-identified issue strings.
            workspace_id: Optional workspace UUID string.

        Returns:
            Persisted EpisodeRecord on success, or None if skipped.
        """
        if db is None:
            return None

        try:
            ws_uuid: uuid.UUID | None = None
            if workspace_id:
                ws_uuid = uuid.UUID(workspace_id)

            episode = EpisodeRecord(
                workspace_id=ws_uuid,
                question=question,
                query_type=query_type,
                retrieval_strategy=retrieval_strategy,
                plan=plan,
                final_answer=final_answer,
                final_decision=final_decision.lower(),
                critic_score=critic_score,
                confidence=confidence,
                evidence_coverage=evidence_coverage,
                repair_attempts=repair_attempts,
                was_killed=was_killed,
                latency_ms=latency_ms,
                cost=cost,
                issues_json={"issues": issues or []},
            )
            db.add(episode)
            await db.commit()
            await db.refresh(episode)
            logger.info(
                "[EpisodeStore] Persisted episode %s — decision=%s, confidence=%.2f",
                episode.id,
                final_decision,
                confidence or 0.0,
            )
            return episode

        except Exception as exc:
            logger.warning("[EpisodeStore] Failed to persist episode: %s", str(exc))
            try:
                await db.rollback()
            except Exception:
                pass
            return None
