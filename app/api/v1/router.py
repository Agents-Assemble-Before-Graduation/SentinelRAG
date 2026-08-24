"""API v1 master router aggregating all domain sub-routers."""

from fastapi import APIRouter

from app.api.v1.documents import router as documents_router
from app.api.v1.health import router as health_router
from app.api.v1.query import router as query_router

api_v1_router = APIRouter()

# Register system health endpoints
api_v1_router.include_router(health_router, prefix="", tags=["System"])

# Register document ingestion and management endpoints
api_v1_router.include_router(documents_router, prefix="", tags=["Documents"])

# Register RAG query endpoint
api_v1_router.include_router(query_router, prefix="", tags=["RAG Query"])
