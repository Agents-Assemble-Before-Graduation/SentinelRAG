"""Lesson store: reads and writes generalised lessons extracted from episode memory."""

import hashlib
import uuid
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.base import LessonRecord

logger = get_logger(__name__)

# Quality-control thresholds
MIN_CONFIDENCE_TO_STORE: float = 0.5
MAX_LESSONS_TO_RETRIEVE: int = 5


def _hash_lesson(lesson: str) -> str:
    """Create a stable SHA-256 hash for deduplication of lesson text."""
    return hashlib.sha256(lesson.strip().lower().encode()).hexdigest()


class LessonStore:
    """Manages structured lessons derived from notable query episodes.

    Provides:
    - Deduplicated storage (same lesson hash is never inserted twice).
    - Relevance-based retrieval using keyword and category matching.
    - Usage tracking so frequently applied lessons can be weighted higher.
    """

    async def store_lesson(
        self,
        db: AsyncSession | None,
        *,
        lesson: str,
        category: str,
        confidence: float,
        source_episode_id: "uuid.UUID | str | None" = None,
    ) -> "LessonRecord | None":
        """Persist a lesson, skipping if confidence is below threshold or duplicate.

        Args:
            db: Active async SQLAlchemy session. Pass None to silently skip.
            lesson: The generalised lesson text.
            category: Lesson category tag (e.g. 'retrieval_strategy', 'evidence_gap').
            confidence: Extraction confidence score (0.0-1.0).
            source_episode_id: UUID of the originating episode (optional).

        Returns:
            The created LessonRecord, or None if skipped/failed.
        """
        if db is None:
            return None

        # Quality gate
        if confidence < MIN_CONFIDENCE_TO_STORE:
            logger.debug(
                "[LessonStore] Skipping low-confidence lesson (%.2f < %.2f): %.60s",
                confidence, MIN_CONFIDENCE_TO_STORE, lesson,
            )
            return None

        lesson_hash = _hash_lesson(lesson)

        try:
            # --- Deduplication check ---
            existing = await db.scalar(
                select(LessonRecord).where(LessonRecord.lesson_hash == lesson_hash)
            )
            if existing is not None:
                logger.debug("[LessonStore] Duplicate lesson skipped: %s", lesson_hash[:16])
                return existing

            # Parse source_episode_id to UUID if string
            ep_uuid: uuid.UUID | None = None
            if source_episode_id is not None:
                ep_uuid = (
                    source_episode_id
                    if isinstance(source_episode_id, uuid.UUID)
                    else uuid.UUID(str(source_episode_id))
                )

            record = LessonRecord(
                lesson=lesson.strip(),
                category=category,
                confidence=confidence,
                lesson_hash=lesson_hash,
                source_episode_id=ep_uuid,
                usage_count=0,
            )
            db.add(record)
            await db.commit()
            await db.refresh(record)
            logger.info(
                "[LessonStore] Stored lesson [%s] (confidence=%.2f): %.80s",
                category, confidence, lesson,
            )
            return record

        except Exception as exc:
            logger.warning("[LessonStore] Failed to store lesson: %s", str(exc))
            try:
                await db.rollback()
            except Exception:
                pass
            return None

    async def retrieve_relevant(
        self,
        db: AsyncSession | None,
        query: str,
        limit: int = MAX_LESSONS_TO_RETRIEVE,
    ) -> list[dict[str, Any]]:
        """Retrieve lessons relevant to the given query using keyword matching.

        Matches lesson text against tokens in the query and orders by confidence.
        Only returns lessons above the minimum confidence threshold.

        Args:
            db: Active async SQLAlchemy session. Pass None to return empty list.
            query: The user question to match lessons against.
            limit: Maximum number of lessons to return.

        Returns:
            List of lesson dicts with keys: lesson, category, confidence, usage_count.
        """
        if db is None:
            return []

        try:
            stmt = (
                select(LessonRecord)
                .where(LessonRecord.confidence >= MIN_CONFIDENCE_TO_STORE)
                .order_by(LessonRecord.confidence.desc())
                .limit(limit * 5)  # Over-fetch then filter by relevance
            )
            result = await db.execute(stmt)
            all_lessons = result.scalars().all()

            if not all_lessons:
                return []

            # Keyword relevance scoring
            query_tokens = set(query.lower().split())
            # Strip very common stop-words for better signal
            stop_words = {"what", "is", "are", "the", "a", "an", "of", "in", "for", "and", "or"}
            query_tokens -= stop_words

            scored: list[tuple[float, LessonRecord]] = []
            for record in all_lessons:
                lesson_tokens = set(record.lesson.lower().split())
                overlap = len(query_tokens & lesson_tokens)
                # Also boost if category matches query token
                category_boost = 1.0 if record.category.replace("_", " ") in query.lower() else 0.0
                relevance = overlap + category_boost
                if relevance > 0 or len(all_lessons) <= limit:
                    # If fewer lessons than limit exist, include all regardless of overlap
                    scored.append((relevance * record.confidence, record))

            scored.sort(key=lambda x: x[0], reverse=True)
            top = [rec for _, rec in scored[:limit]]

            # Increment usage_count for retrieved lessons
            if top:
                ids = [r.id for r in top]
                await db.execute(
                    update(LessonRecord)
                    .where(LessonRecord.id.in_(ids))
                    .values(usage_count=LessonRecord.usage_count + 1)
                )
                await db.commit()

            return [
                {
                    "lesson": rec.lesson,
                    "category": rec.category,
                    "confidence": rec.confidence,
                    "usage_count": rec.usage_count,
                }
                for rec in top
            ]

        except Exception as exc:
            logger.warning("[LessonStore] Failed to retrieve lessons: %s", str(exc))
            return []
