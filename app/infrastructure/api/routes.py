"""FastAPI endpoints for liveness detection."""
from __future__ import annotations

import asyncio
import uuid

import cv2
import numpy as np
import structlog
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.infrastructure.api.schemas import (
    LivenessResponse, SessionResponse, HealthResponse, ReadyResponse,
)
from app.infrastructure.api.dependencies import get_container
from app.infrastructure.container import Container
from app.domain.enums import LivenessVerdict

logger = structlog.get_logger(__name__)

router = APIRouter()

# In-memory session storage (for stateful heuristic analysis)
_sessions: dict[str, object] = {}


def _decode_image(data: bytes) -> np.ndarray:
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    return img


def _build_session_state(container: Container) -> dict[str, object]:
    from app.adapters.analyzers.heuristic_analyzer import HeuristicAnalyzer

    return {
        "container": container,
        "analyzer": HeuristicAnalyzer(container._settings.analyzer.heuristic),
        "use_case": None,
    }


def _get_session_use_case(session_id: str):
    from app.use_cases.analyze_video_frame import AnalyzeVideoFrameUseCase

    session = _sessions[session_id]
    if not isinstance(session, dict):
        return session

    use_case = session.get("use_case")
    if use_case is not None:
        return use_case

    container = session["container"]
    analyzer = session["analyzer"]
    try:
        use_case = AnalyzeVideoFrameUseCase(
            detector=container.mediapipe_detector,
            analyzer=analyzer,
        )
    except Exception as exc:
        logger.exception("session_detector_initialization_failed", session_id=session_id)
        raise HTTPException(
            status_code=503,
            detail="Session detector is unavailable on this host",
        ) from exc

    session["use_case"] = use_case
    return use_case


@router.post("/api/v1/liveness/analyze", response_model=LivenessResponse)
async def analyze_image(
    file: UploadFile = File(...),
    container: Container = Depends(get_container),
):
    data = await file.read()
    image = _decode_image(data)
    use_case = container.analyze_single_image
    result = await asyncio.to_thread(use_case.execute, image)
    return LivenessResponse(
        verdict=result.verdict,
        confidence=result.confidence,
        method=result.method,
        details=result.details,
    )


@router.post("/api/v1/liveness/session", response_model=SessionResponse)
async def create_session(container: Container = Depends(get_container)):
    session_id = str(uuid.uuid4())
    _sessions[session_id] = _build_session_state(container)
    logger.info("session_created", session_id=session_id)
    return SessionResponse(session_id=session_id)


@router.post("/api/v1/liveness/session/{session_id}/frame", response_model=LivenessResponse)
async def process_session_frame(
    session_id: str,
    file: UploadFile = File(...),
):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    data = await file.read()
    image = _decode_image(data)
    use_case = _get_session_use_case(session_id)
    result = await asyncio.to_thread(use_case.execute, image)
    return LivenessResponse(
        verdict=result.verdict,
        confidence=result.confidence,
        method=result.method,
        details=result.details,
    )


@router.delete("/api/v1/liveness/session/{session_id}")
async def delete_session(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    del _sessions[session_id]
    logger.info("session_deleted", session_id=session_id)
    return {"status": "deleted"}


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok")


@router.get("/health/ready", response_model=ReadyResponse)
async def readiness(container: Container = Depends(get_container)):
    loaded = container.models_loaded
    return ReadyResponse(status="ready" if loaded else "not_ready", models_loaded=loaded)
