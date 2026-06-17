"""
main.py
-------
ADVOCATE FastAPI application entry point.

Responsibilities:
- Create and configure the FastAPI application instance.
- Register all API routers.
- Define the lifespan context manager that handles startup / shutdown logic:
    * Loads environment variables from .env on startup.
    * Initialises the database (creates tables if absent).
    * Logs that the service is ready.
- Configure global exception handlers for consistent error JSON.
- Add CORS middleware so the API can be consumed by browser-based frontends.

Running the server:
    uvicorn app.main:app --reload
"""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.analyze import router as analyze_router
from app.api.health import router as health_router
from app.api.session import router as session_router
from app.database.database import init_db
from app.schemas.response import ErrorResponse
from app.utils.logger import configure_logging, get_logger

# ---------------------------------------------------------------------------
# Logging must be configured before anything else so the startup log lines
# are captured with the correct format.
# ---------------------------------------------------------------------------
configure_logging(level=os.getenv("LOG_LEVEL", "INFO"))
log = get_logger(__name__)

# Load .env file from the project root (one directory above app/).
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))


# ===========================================================================
# Lifespan context manager (replaces deprecated @app.on_event decorators)
# ===========================================================================


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manage application startup and shutdown within a single async context.

    Startup:
        1. Calls ``init_db()`` to create database tables if they do not exist.
        2. Logs a ready message.

    Shutdown:
        1. Currently no teardown needed (SQLAlchemy engine cleanup is handled
           by the GC / process exit).  Add any cleanup here if needed.
    """
    # --- STARTUP ---
    log.info("ADVOCATE backend starting up…")
    await init_db()
    log.info("ADVOCATE backend ready.  Navigate to /docs for the Swagger UI.")

    yield  # Application runs here.

    # --- SHUTDOWN ---
    log.info("ADVOCATE backend shutting down.")


# ===========================================================================
# FastAPI application
# ===========================================================================

app = FastAPI(
    title="ADVOCATE – The AI That's Designed to Disagree With You",
    description=(
        "A multi-agent AI system that challenges user ideas by scoring them, "
        "exposing hidden assumptions, generating strong counter-arguments, and "
        "delivering a final verdict.  Works in mock mode with no API keys required."
    ),
    version="1.0.0",
    contact={
        "name": "ADVOCATE Engineering",
        "url": "https://github.com/your-org/advocate",
    },
    license_info={
        "name": "MIT",
    },
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ===========================================================================
# CORS Middleware
# Allow any origin in development.  Restrict in production by setting the
# CORS_ORIGINS environment variable (comma-separated list).
# ===========================================================================
_cors_origins_raw = os.getenv("CORS_ORIGINS", "*")
_cors_origins = (
    ["*"]
    if _cors_origins_raw.strip() == "*"
    else [o.strip() for o in _cors_origins_raw.split(",")]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================================================================
# Global exception handlers
# ===========================================================================


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all handler that converts any unhandled exception into a structured
    :class:`~app.schemas.response.ErrorResponse` JSON body.

    This prevents internal Python tracebacks from leaking to API consumers.
    """
    log.exception("Unhandled exception on %s %s: %s", request.method, request.url, exc)
    error_body = ErrorResponse(
        error="internal_server_error",
        detail="An unexpected error occurred. Please try again.",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_body.model_dump(),
    )


# ===========================================================================
# Router registration
# ===========================================================================

# Register all three sub-routers.  The /analyze router already has its own
# prefix set; health and session routers use the root prefix.
app.include_router(health_router)
app.include_router(analyze_router)
app.include_router(session_router)


# ===========================================================================
# Root redirect → docs
# ===========================================================================


@app.get("/", include_in_schema=False)
async def root() -> JSONResponse:
    """Redirect API root to the Swagger docs page for convenience."""
    return JSONResponse(
        content={
            "message": "Welcome to ADVOCATE. Visit /docs for the interactive API.",
            "docs": "/docs",
            "health": "/health",
        }
    )
