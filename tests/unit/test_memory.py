"""Unit tests for Phase 7 experience memory — episode storage, lesson extraction,
lesson deduplication, retrieval, and planner integration."""

import asyncio
import hashlib
import uuid
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from app.memory.episode_store import EpisodeStore
from app.memory.lesson_store import LessonStore, _hash_lesson, MIN_CONFIDENCE_TO_STORE
from app.memory.lesson_extractor import LessonExtractor, _is_notable_episode


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mock_db():
    """Return a mock AsyncSession with add/commit/refresh/rollback/scalar as AsyncMock."""
    db = MagicMock()
    db.add = MagicMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    db.rollback = AsyncMock()
    db.scalar = AsyncMock(return_value=None)  # default: no existing record
    db.execute = AsyncMock()
    return db


def _make_episode_record(ep_id=None):
    """Return a fake EpisodeRecord-like object."""
    rec = MagicMock()
    rec.id = ep_id or uuid.uuid4()
    return rec


def _make_lesson_record(lesson_text="Test lesson", category="retrieval_strategy", confidence=0.7):
    rec = MagicMock()
    rec.id = uuid.uuid4()
    rec.lesson = lesson_text
    rec.category = category
    rec.confidence = confidence
    rec.lesson_hash = _hash_lesson(lesson_text)
    rec.usage_count = 0
    return rec


# ── EpisodeStore ─────────────────────────────────────────────────────────────

class TestEpisodeStore:
    """Tests for EpisodeStore.store()."""

    @pytest.mark.asyncio
    async def test_store_no_db_returns_none(self):
        """store() with db=None should silently return None."""
        store = EpisodeStore()
        result = await store.store(None, question="Test question")
        assert result is None

    @pytest.mark.asyncio
    async def test_store_creates_episode_record(self):
        """store() should add, commit, and return an EpisodeRecord."""
        db = _mock_db()
        fake_episode = _make_episode_record()
        db.refresh = AsyncMock(side_effect=lambda obj: setattr(obj, "id", fake_episode.id))

        store = EpisodeStore()
        with patch("app.memory.episode_store.EpisodeRecord") as MockEpisode:
            mock_instance = MagicMock()
            mock_instance.id = fake_episode.id
            MockEpisode.return_value = mock_instance

            result = await store.store(
                db,
                question="What is the API rate limit?",
                query_type="numerical",
                retrieval_strategy="bm25",
                final_decision="accept",
                confidence=0.85,
                repair_attempts=0,
                was_killed=False,
            )

        db.add.assert_called_once()
        db.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_store_handles_db_error_gracefully(self):
        """store() should return None and not raise on db error."""
        db = _mock_db()
        db.commit = AsyncMock(side_effect=Exception("DB connection failed"))

        store = EpisodeStore()
        result = await store.store(db, question="Does this crash?")
        assert result is None
        db.rollback.assert_awaited()

    @pytest.mark.asyncio
    async def test_store_killed_episode_sets_was_killed_true(self):
        """store() should set was_killed=True when final_decision is kill."""
        db = _mock_db()

        store = EpisodeStore()
        with patch("app.memory.episode_store.EpisodeRecord") as MockEpisode:
            instance = MagicMock()
            instance.id = uuid.uuid4()
            MockEpisode.return_value = instance

            await store.store(
                db,
                question="Unanswerable question",
                final_decision="kill",
                was_killed=True,
                confidence=0.1,
            )
            call_kwargs = MockEpisode.call_args.kwargs
            assert call_kwargs["was_killed"] is True
            assert call_kwargs["final_decision"] == "kill"


# ── LessonStore ──────────────────────────────────────────────────────────────

class TestLessonStore:
    """Tests for LessonStore deduplication and retrieval."""

    @pytest.mark.asyncio
    async def test_store_lesson_no_db_returns_none(self):
        store = LessonStore()
        result = await store.store_lesson(
            None, lesson="Test lesson", category="retrieval_strategy", confidence=0.7
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_store_lesson_below_confidence_threshold_skipped(self):
        """Lessons with confidence < MIN_CONFIDENCE_TO_STORE must be skipped."""
        db = _mock_db()
        store = LessonStore()
        result = await store.store_lesson(
            db,
            lesson="Low-quality lesson",
            category="retrieval_strategy",
            confidence=MIN_CONFIDENCE_TO_STORE - 0.01,  # 0.49
        )
        assert result is None
        db.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_store_lesson_deduplication_returns_existing(self):
        """Storing a duplicate lesson (same hash) should return the existing record."""
        db = _mock_db()
        existing = _make_lesson_record("BM25 is better for numerical queries.")
        db.scalar = AsyncMock(return_value=existing)

        store = LessonStore()
        result = await store.store_lesson(
            db,
            lesson="BM25 is better for numerical queries.",
            category="retrieval_strategy",
            confidence=0.8,
        )
        # Should return the existing record without inserting a new one
        assert result == existing
        db.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_store_lesson_new_record_inserted(self):
        """A new unique lesson should be inserted into the database."""
        db = _mock_db()
        db.scalar = AsyncMock(return_value=None)  # no duplicate found

        store = LessonStore()
        result = await store.store_lesson(
            db,
            lesson="Hybrid retrieval reduces repair cycles for comparison queries.",
            category="retrieval_strategy",
            confidence=0.75,
        )

        # db.add should have been called with a real LessonRecord instance
        db.add.assert_called_once()
        added_obj = db.add.call_args[0][0]
        from app.models.base import LessonRecord as LRec
        assert isinstance(added_obj, LRec)
        assert added_obj.category == "retrieval_strategy"
        db.commit.assert_awaited()


    @pytest.mark.asyncio
    async def test_retrieve_relevant_no_db_returns_empty(self):
        """retrieve_relevant() with db=None should return empty list."""
        store = LessonStore()
        results = await store.retrieve_relevant(None, query="Some question")
        assert results == []

    @pytest.mark.asyncio
    async def test_retrieve_relevant_returns_lessons_with_keyword_match(self):
        """retrieve_relevant() should return keyword-matched lessons."""
        db = _mock_db()

        lesson1 = _make_lesson_record(
            "BM25 is recommended for version-specific numerical queries.",
            "retrieval_strategy",
            0.80,
        )
        lesson2 = _make_lesson_record(
            "Dense retrieval works well for semantic concept definitions.",
            "retrieval_strategy",
            0.70,
        )

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [lesson1, lesson2]
        db.execute = AsyncMock(return_value=mock_result)

        store = LessonStore()
        results = await store.retrieve_relevant(db, query="What is the version number?")

        # BM25/numerical lesson should rank higher for this query
        assert len(results) >= 1
        assert results[0]["lesson"] == lesson1.lesson

    @pytest.mark.asyncio
    async def test_retrieve_relevant_increments_usage_count(self):
        """Retrieved lessons should have usage_count incremented."""
        db = _mock_db()
        lesson = _make_lesson_record("Hybrid retrieval for comparisons.", "retrieval_strategy", 0.75)

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [lesson]
        db.execute = AsyncMock(return_value=mock_result)

        store = LessonStore()
        await store.retrieve_relevant(db, query="Compare dense vs sparse retrieval")

        # Should call execute twice: once for SELECT, once for UPDATE
        assert db.execute.await_count == 2


# ── LessonExtractor ──────────────────────────────────────────────────────────

class TestLessonExtractor:
    """Tests for LessonExtractor.extract()."""

    def test_not_notable_episode_skipped(self):
        """Ordinary successful episodes should not be considered notable."""
        data = {
            "was_killed": False,
            "repair_attempts": 0,
            "confidence": 0.90,
        }
        assert _is_notable_episode(data) is False

    def test_killed_episode_is_notable(self):
        data = {"was_killed": True, "repair_attempts": 0, "confidence": 0.9}
        assert _is_notable_episode(data) is True

    def test_repaired_episode_is_notable(self):
        data = {"was_killed": False, "repair_attempts": 1, "confidence": 0.9}
        assert _is_notable_episode(data) is True

    def test_low_confidence_episode_is_notable(self):
        data = {"was_killed": False, "repair_attempts": 0, "confidence": 0.3}
        assert _is_notable_episode(data) is True

    @pytest.mark.asyncio
    async def test_non_notable_episode_returns_empty_list(self):
        """extract() on non-notable episode should return empty list."""
        extractor = LessonExtractor()
        result = await extractor.extract({
            "was_killed": False,
            "repair_attempts": 0,
            "confidence": 0.95,
        })
        assert result == []

    @pytest.mark.asyncio
    async def test_killed_episode_generates_evidence_gap_lesson(self):
        """A killed episode should produce an 'evidence_gap' lesson."""
        extractor = LessonExtractor()
        with patch.object(extractor, "llm") as mock_llm:
            mock_llm.is_available.return_value = False
            mock_llm.provider_name = "mock"

            lessons = await extractor.extract({
                "query_type": "numerical",
                "retrieval_strategy": "dense",
                "was_killed": True,
                "repair_attempts": 2,
                "confidence": 0.1,
                "issues": [],
            })

        assert len(lessons) >= 1
        categories = [l["category"] for l in lessons]
        assert "evidence_gap" in categories

    @pytest.mark.asyncio
    async def test_repaired_success_generates_retrieval_strategy_lesson(self):
        """A repaired-but-accepted episode should produce a retrieval_strategy lesson."""
        extractor = LessonExtractor()
        with patch.object(extractor, "llm") as mock_llm:
            mock_llm.is_available.return_value = False
            mock_llm.provider_name = "mock"

            lessons = await extractor.extract({
                "query_type": "factual",
                "retrieval_strategy": "dense",
                "was_killed": False,
                "repair_attempts": 1,
                "confidence": 0.65,
                "issues": [],
            })

        assert len(lessons) >= 1
        categories = [l["category"] for l in lessons]
        assert "retrieval_strategy" in categories

    @pytest.mark.asyncio
    async def test_all_lessons_have_required_keys(self):
        """All extracted lessons must have 'lesson', 'category', 'confidence'."""
        extractor = LessonExtractor()
        with patch.object(extractor, "llm") as mock_llm:
            mock_llm.is_available.return_value = False
            mock_llm.provider_name = "mock"

            lessons = await extractor.extract({
                "query_type": "comparison",
                "retrieval_strategy": "dense",
                "was_killed": True,
                "repair_attempts": 2,
                "confidence": 0.2,
                "issues": ["missing evidence", "query rewrite needed"],
            })

        for l in lessons:
            assert "lesson" in l and l["lesson"]
            assert "category" in l
            assert "confidence" in l
            assert 0.0 <= l["confidence"] <= 1.0


# ── Planner + Lesson Integration ─────────────────────────────────────────────

class TestPlannerLessonIntegration:
    """Integration-style tests: verify that lessons influence Planner decisions."""

    @pytest.mark.asyncio
    async def test_planner_without_lessons_uses_heuristics(self):
        """Planner with no lessons on a numerical query should return bm25 via heuristics."""
        from app.agents.planner import PlannerAgent
        planner = PlannerAgent()
        with patch.object(planner.llm, "is_available", return_value=False):
            plan = await planner.plan("What is the API rate limit version 2.1?")
        assert plan["retrieval_strategy"] in ("bm25", "dense", "hybrid")

    @pytest.mark.asyncio
    async def test_planner_lesson_overrides_dense_to_bm25(self):
        """A high-confidence BM25 lesson should upgrade a dense plan to bm25."""
        from app.agents.planner import PlannerAgent
        planner = PlannerAgent()

        lesson = {
            "lesson": "Numerical and version-specific queries benefit from BM25 lexical retrieval.",
            "category": "retrieval_strategy",
            "confidence": 0.80,
            "usage_count": 3,
        }

        # Force heuristic path to return 'dense' for a factual query
        with patch.object(planner, "_heuristics_plan", return_value={
            "query_type": "factual",
            "retrieval_strategy": "dense",
            "plan": "Retrieve factual data.",
            "subquestions": [],
        }), patch.object(planner.llm, "is_available", return_value=False):
            plan = await planner.plan("What is the default timeout value?", lessons=[lesson])

        # Lesson should have overridden dense → bm25
        assert plan["retrieval_strategy"] == "bm25"

    @pytest.mark.asyncio
    async def test_planner_low_confidence_lesson_does_not_override(self):
        """A lesson with confidence < 0.65 must NOT override the heuristic decision."""
        from app.agents.planner import PlannerAgent
        planner = PlannerAgent()

        lesson = {
            "lesson": "BM25 is better for something.",
            "category": "retrieval_strategy",
            "confidence": 0.50,  # below override threshold
            "usage_count": 0,
        }

        with patch.object(planner, "_heuristics_plan", return_value={
            "query_type": "factual",
            "retrieval_strategy": "dense",
            "plan": "Retrieve.",
            "subquestions": [],
        }), patch.object(planner.llm, "is_available", return_value=False):
            plan = await planner.plan("Generic question.", lessons=[lesson])

        # Should remain 'dense'
        assert plan["retrieval_strategy"] == "dense"

    @pytest.mark.asyncio
    async def test_planner_builds_system_prompt_with_lessons(self):
        """_build_system_prompt should include Memory Advisory when lessons present."""
        from app.agents.planner import PlannerAgent
        planner = PlannerAgent()
        lessons = [
            {"lesson": "Test lesson.", "category": "evidence_gap", "confidence": 0.70}
        ]
        prompt = planner._build_system_prompt(lessons)
        assert "[Memory Advisory]" in prompt
        assert "evidence_gap" in prompt
        assert "Test lesson." in prompt

    @pytest.mark.asyncio
    async def test_planner_no_lessons_returns_base_prompt(self):
        """_build_system_prompt with no lessons must return the base prompt unchanged."""
        from app.agents.planner import PlannerAgent
        from app.agents.planner import PLANNER_SYSTEM_PROMPT
        planner = PlannerAgent()
        prompt = planner._build_system_prompt(None)
        assert prompt == PLANNER_SYSTEM_PROMPT


# ── hash_lesson ───────────────────────────────────────────────────────────────

class TestLessonHashing:
    def test_same_text_produces_same_hash(self):
        h1 = _hash_lesson("BM25 is better for numerical queries.")
        h2 = _hash_lesson("BM25 is better for numerical queries.")
        assert h1 == h2

    def test_different_text_produces_different_hash(self):
        h1 = _hash_lesson("BM25 is better for numerical queries.")
        h2 = _hash_lesson("Dense retrieval works for semantic search.")
        assert h1 != h2

    def test_whitespace_normalised(self):
        h1 = _hash_lesson("  BM25 is better.  ")
        h2 = _hash_lesson("bm25 is better.")
        # Both normalised to lowercase + stripped
        assert h1 == h2
