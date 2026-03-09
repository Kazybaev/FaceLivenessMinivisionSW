"""FastAPI application factory."""
from __future__ import annotations

import uvicorn
from fastapi import FastAPI

from app.infrastructure.config import get_settings
from app.infrastructure.logging_setup import setup_logging
from app.infrastructure.api.routes import router
from app.infrastructure.api.middleware import setup_middleware


def create_app() -> FastAPI:
    setup_logging()

    settings = get_settings()
    app = FastAPI(
        title="Face Liveness Detection API",
        version="1.0.0",
        description="Heuristic + Deep Learning face anti-spoofing",
    )
    setup_middleware(app, settings.api.cors_origins)
    app.include_router(router)
    return app


app = create_app()


def run() -> None:
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.api.host,
        port=settings.api.port,
        workers=settings.api.workers,
        reload=False,
    )


if __name__ == "__main__":
    run()
