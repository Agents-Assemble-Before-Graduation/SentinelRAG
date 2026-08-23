"""Health and Readiness probe endpoints for SentinelRAG."""

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Response, status
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.database.session import check_database_health
from app.services.vector_store import get_vector_store

router = APIRouter(tags=["Health & Monitoring"])
settings = get_settings()


class HealthResponse(BaseModel):
    """Liveness probe response model."""

    status: str = Field(default="healthy", description="Application process health status")
    app_name: str = Field(description="Name of the application")
    version: str = Field(description="Current application version")
    environment: str = Field(description="Current runtime environment")
    timestamp: str = Field(description="Current timestamp in ISO 8601 UTC")


class ComponentStatus(BaseModel):
    """Individual infrastructure component health status."""

    status: str = Field(description="'healthy', 'degraded', or 'unhealthy'")
    connected: bool = Field(description="True if connection succeeded")
    latency_ms: float | None = Field(default=None, description="Check latency in milliseconds")
    details: dict[str, Any] | None = Field(default=None, description="Non-sensitive component details")
    error: str | None = Field(default=None, description="Sanitized error message if failed")


class ReadinessResponse(BaseModel):
    """Readiness probe response model verifying dependent services."""

    status: str = Field(description="Overall readiness: 'ready', 'degraded', or 'unready'")
    environment: str = Field(description="Current runtime environment")
    timestamp: str = Field(description="Current timestamp in ISO 8601 UTC")
    components: dict[str, ComponentStatus] = Field(description="Status of each critical dependency")


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Process Liveness Probe",
    description="Verifies that the backend application process is running and able to handle HTTP traffic.",
)
async def get_health() -> HealthResponse:
    """Liveness check verifying the backend process is alive."""
    return HealthResponse(
        status="healthy",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
        timestamp=datetime.now(UTC).isoformat(),
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Infrastructure Readiness Probe",
    description="Verifies that downstream dependencies (PostgreSQL, Qdrant) are reachable and healthy.",
)
async def get_readiness(response: Response) -> ReadinessResponse:
    """Readiness check verifying all backing infrastructure components."""
    # Check PostgreSQL database
    db_result = await check_database_health()
    db_component = ComponentStatus(
        status=db_result.get("status", "unhealthy"),
        connected=db_result.get("connected", False),
        latency_ms=db_result.get("latency_ms"),
        error=db_result.get("error"),
    )

    # Check Qdrant vector store
    vector_store = get_vector_store()
    vs_result = await vector_store.health_check()
    vs_details = {}
    if "collections_count" in vs_result:
        vs_details["collections_count"] = vs_result["collections_count"]

    vs_component = ComponentStatus(
        status=vs_result.get("status", "unhealthy"),
        connected=vs_result.get("connected", False),
        latency_ms=vs_result.get("latency_ms"),
        details=vs_details if vs_details else None,
        error=vs_result.get("error"),
    )

    components = {
        "database": db_component,
        "vector_store": vs_component,
    }

    # Aggregate status determination
    all_connected = db_component.connected and vs_component.connected
    any_connected = db_component.connected or vs_component.connected

    if all_connected:
        overall_status = "ready"
        response.status_code = status.HTTP_200_OK
    elif any_connected:
        overall_status = "degraded"
        response.status_code = status.HTTP_200_OK
    else:
        overall_status = "unready"
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessResponse(
        status=overall_status,
        environment=settings.ENVIRONMENT,
        timestamp=datetime.now(UTC).isoformat(),
        components=components,
    )
