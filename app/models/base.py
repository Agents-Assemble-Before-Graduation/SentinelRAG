"""Foundational and Ingestion database models for SentinelRAG."""

import uuid
from typing import Any

from sqlalchemy import Boolean, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class Workspace(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Workspace isolating document corpora, agent runs, and configuration."""

    __tablename__ = "workspaces"

    name: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Relationships
    documents: Mapped[list["DocumentRecord"]] = relationship(
        "DocumentRecord", back_populates="workspace", cascade="all, delete-orphan"
    )
    agent_runs: Mapped[list["AgentRunRecord"]] = relationship(
        "AgentRunRecord", back_populates="workspace", cascade="all, delete-orphan"
    )


class DocumentRecord(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Metadata record for ingested documents and knowledge assets."""

    __tablename__ = "documents"

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    source_type: Mapped[str] = mapped_column(String(64), nullable=False)  # pdf, markdown, text, docx
    content_hash: Mapped[str] = mapped_column(String(64), index=True, nullable=False)  # SHA-256
    file_size_bytes: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    status: Mapped[str] = mapped_column(
        String(64), default="pending", nullable=False
    )  # pending, indexed, duplicate, failed
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    # Relationships
    workspace: Mapped["Workspace"] = relationship("Workspace", back_populates="documents")
    chunks: Mapped[list["DocumentChunkRecord"]] = relationship(
        "DocumentChunkRecord", back_populates="document", cascade="all, delete-orphan"
    )


class DocumentChunkRecord(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Granular text chunk extracted from a document with structural provenance."""

    __tablename__ = "document_chunks"

    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    workspace_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    section_heading: Mapped[str | None] = mapped_column(String(255), nullable=True)
    chunk_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    # Relationships
    document: Mapped["DocumentRecord"] = relationship("DocumentRecord", back_populates="chunks")


class AgentRunRecord(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Audit and trace record of multi-agent pipeline executions."""

    __tablename__ = "agent_runs"

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True
    )
    session_id: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    user_query: Mapped[str] = mapped_column(Text, nullable=False)
    final_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        String(64), default="pending", nullable=False
    )  # pending, completed, killed, failed
    termination_reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    total_duration_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    metrics_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    # Relationships
    workspace: Mapped["Workspace"] = relationship("Workspace", back_populates="agent_runs")


class EpisodeRecord(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Complete query execution episode stored for experience memory.

    Captures the full trace of one agent run — from question to final decision —
    enabling lesson extraction and orchestration self-improvement over time.
    """

    __tablename__ = "episodes"

    workspace_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="SET NULL"), nullable=True, index=True
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    query_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    retrieval_strategy: Mapped[str | None] = mapped_column(String(32), nullable=True)
    plan: Mapped[str | None] = mapped_column(Text, nullable=True)
    final_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    final_decision: Mapped[str] = mapped_column(String(32), default="accept", nullable=False)  # accept / kill
    critic_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    evidence_coverage: Mapped[float | None] = mapped_column(Float, nullable=True)
    repair_attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    was_killed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    cost: Mapped[float | None] = mapped_column(Float, nullable=True)
    issues_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    # Lessons derived from this episode
    lessons: Mapped[list["LessonRecord"]] = relationship(
        "LessonRecord", back_populates="source_episode", cascade="all, delete-orphan"
    )


class LessonRecord(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """A generalised, deduplicated lesson extracted from notable query episodes.

    Lessons encode retrieval strategy patterns learnt from failures and successes.
    They are injected into the Planner as advisory context before future runs.
    """

    __tablename__ = "lessons"

    source_episode_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("episodes.id", ondelete="SET NULL"), nullable=True, index=True
    )
    lesson: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    # e.g. "retrieval_strategy", "query_rewriting", "evidence_gap", "contradiction"
    confidence: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    lesson_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    usage_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Relationship back to source episode
    source_episode: Mapped["EpisodeRecord | None"] = relationship(
        "EpisodeRecord", back_populates="lessons"
    )


class QueryLog(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Telemetry log for incoming queries and decision auditing."""

    __tablename__ = "query_logs"

    workspace_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="SET NULL"), nullable=True, index=True
    )
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    answer_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_killed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    confidence_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
