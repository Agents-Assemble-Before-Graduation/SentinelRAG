"""SentinelRAG FastAPI Application Entrypoint."""

import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.api.v1.health import router as health_router
from app.api.v1.router import api_v1_router
from app.core.config import get_settings
from app.core.exceptions import SentinelRAGException
from app.core.logging import get_logger, request_id_ctx_var, setup_logging
from app.services.vector_store import get_vector_store

settings = get_settings()
setup_logging(log_level=settings.LOG_LEVEL, json_logs=settings.is_production)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context manager handling startup and shutdown hooks."""
    logger.info(
        "Starting %s v%s in %s mode (Debug=%s)",
        settings.APP_NAME,
        settings.APP_VERSION,
        settings.ENVIRONMENT,
        settings.DEBUG,
    )
    yield
    logger.info("Shutting down %s...", settings.APP_NAME)
    # Cleanup vector store client connections
    try:
        vector_store = get_vector_store()
        await vector_store.close()
        logger.info("Vector store connections closed.")
    except Exception as e:
        logger.warning("Error during vector store shutdown cleanup: %s", str(e))


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Self-improving Multi-Agent RAG with evidence verification, critique, repair, "
        "and experience memory."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Middleware for extracting or generating Request ID and setting logging context."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Check if caller passed an X-Request-ID header, otherwise generate a new UUID
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        token = request_id_ctx_var.set(request_id)
        start_time = time.perf_counter()

        try:
            response = await call_next(request)
            process_time = (time.perf_counter() - start_time) * 1000
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
            return response
        finally:
            request_id_ctx_var.reset(token)


# Register Middlewares
app.add_middleware(RequestIdMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception Handlers
@app.exception_handler(SentinelRAGException)
async def handle_sentinelrag_exception(
    _request: Request, exc: SentinelRAGException
) -> JSONResponse:
    """Handle custom application domain exceptions without leaking secrets."""
    request_id = request_id_ctx_var.get()
    logger.warning("Domain exception occurred: %s (request_id=%s)", exc.message, request_id)
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": exc.__class__.__name__,
            "message": exc.message,
            "details": exc.details,
            "request_id": request_id,
        },
    )


@app.exception_handler(Exception)
async def handle_unhandled_exception(_request: Request, exc: Exception) -> JSONResponse:
    """Handle uncaught exceptions, shielding internal implementation details in production."""
    request_id = request_id_ctx_var.get()
    logger.error("Unhandled internal server error: %s", str(exc), exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred. Please check server logs.",
            "request_id": request_id,
        },
    )


# Top-level Health Endpoints
app.include_router(health_router, prefix="", tags=["Health & Monitoring"])

# Versioned API Router
app.include_router(api_v1_router, prefix=settings.API_V1_PREFIX)


@app.get("/", tags=["General"])
async def root_info() -> dict:
    """Root endpoint returning basic service metadata and links."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "status": "online",
        "docs": "/docs",
        "health": "/health",
        "ready": "/ready",
    }
