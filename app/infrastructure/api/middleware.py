"""FastAPI middleware: timing, CORS, error handling."""
from __future__ import annotations

import time

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.domain.exceptions import FaceNotFoundError, InvalidImageError, ModelLoadError

logger = structlog.get_logger(__name__)


def setup_middleware(app: FastAPI, cors_origins: list[str]) -> None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def timing_middleware(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
        logger.info("request_handled", method=request.method,
                     path=request.url.path, elapsed_ms=round(elapsed_ms, 1),
                     status=response.status_code)
        return response

    @app.exception_handler(FaceNotFoundError)
    async def face_not_found_handler(request: Request, exc: FaceNotFoundError):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(InvalidImageError)
    async def invalid_image_handler(request: Request, exc: InvalidImageError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(ModelLoadError)
    async def model_load_handler(request: Request, exc: ModelLoadError):
        return JSONResponse(status_code=500, content={"detail": str(exc)})
