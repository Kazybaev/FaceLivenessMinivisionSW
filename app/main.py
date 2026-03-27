"""FastAPI application factory."""
from __future__ import annotations

import argparse
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from app.api.routes import router as access_router
from app.config import Settings as AccessSettings
from app.config import get_settings as get_access_settings
from app.core.pipeline import AccessControlPipeline
from app.infrastructure.config import get_settings as get_legacy_settings
from app.infrastructure.logging_setup import setup_logging
from app.infrastructure.api.routes import router
from app.infrastructure.api.middleware import setup_middleware


def create_app(access_settings: AccessSettings | None = None) -> FastAPI:
    setup_logging()

    legacy_settings = get_legacy_settings()
    runtime_settings = access_settings or get_access_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if runtime_settings.runtime.autostart:
            app.state.access_pipeline.start()
        try:
            yield
        finally:
            app.state.access_pipeline.stop()

    app = FastAPI(
        title="Face Liveness Detection API",
        version="1.0.0",
        description="Heuristic + Deep Learning face anti-spoofing",
        lifespan=lifespan,
    )
    setup_middleware(app, legacy_settings.api.cors_origins)
    app.include_router(router)
    app.include_router(access_router)
    app.state.access_pipeline = AccessControlPipeline(runtime_settings)

    return app


app = create_app()


def run(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Access-control MVP runtime")
    parser.add_argument("--preview", action="store_true", help="Open a visible OpenCV camera preview")
    parser.add_argument("--camera", type=int, default=None, help="Camera index override")
    parser.add_argument("--video", type=str, default=None, help="Video file path instead of a live camera")
    parser.add_argument("--width", type=int, default=None, help="Camera width override")
    parser.add_argument("--height", type=int, default=None, help="Camera height override")
    args = parser.parse_args(argv)

    if args.preview:
        from app.cli.access_preview import main as preview_main

        preview_args: list[str] = []
        if args.camera is not None:
            preview_args.extend(["--camera", str(args.camera)])
        if args.video is not None:
            preview_args.extend(["--video", args.video])
        if args.width is not None:
            preview_args.extend(["--width", str(args.width)])
        if args.height is not None:
            preview_args.extend(["--height", str(args.height)])
        preview_main(preview_args)
        return

    settings = get_access_settings().model_copy(deep=True)
    if args.camera is not None:
        settings.camera.index = args.camera
    if args.video is not None:
        settings.camera.source = args.video
    if args.width is not None:
        settings.camera.width = args.width
    if args.height is not None:
        settings.camera.height = args.height
    runtime_app = create_app(settings)
    uvicorn.run(
        runtime_app,
        host=settings.api.host,
        port=settings.api.port,
        workers=1,
        reload=False,
    )


if __name__ == "__main__":
    run()
