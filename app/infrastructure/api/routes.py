"""FastAPI endpoints for liveness detection and local turnstile control plane."""
from __future__ import annotations

import asyncio
import uuid

import cv2
import numpy as np
import structlog
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.infrastructure.api.dependencies import get_container
from app.infrastructure.api.schemas import (
    DecisionSnapshot,
    HealthResponse,
    LatestDecisionResponse,
    LivenessResponse,
    ReadyResponse,
    SessionResponse,
)
from app.infrastructure.container import Container

logger = structlog.get_logger(__name__)

router = APIRouter()

# In-memory session storage for legacy stateful heuristic analysis.
_sessions: dict[str, object] = {}


def _decode_image(data: bytes) -> np.ndarray:
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    return img


def _decision_snapshot(decision) -> DecisionSnapshot:
    verdict = decision.controller_verdict if decision.controller_verdict is not None else None
    return DecisionSnapshot(
        session_id=decision.session_id,
        state=decision.state,
        verdict=verdict,
        confidence=decision.confidence,
        reason=decision.reason,
        reason_codes=decision.reason_codes,
        latency_ms=decision.latency_ms,
        camera_id=decision.camera_id,
        device_id=decision.device_id,
        timestamp_utc=decision.timestamp_utc,
        live_score=decision.live_score,
        heuristic_score=decision.heuristic_score,
        deep_learning_score=decision.deep_learning_score,
        details=decision.details,
    )


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


@router.post("/api/v1/turnstile/frame", response_model=DecisionSnapshot)
async def process_turnstile_frame(
    file: UploadFile = File(...),
    container: Container = Depends(get_container),
):
    data = await file.read()
    image = _decode_image(data)
    decision = await asyncio.to_thread(container.turnstile_engine.process_frame, image)
    return _decision_snapshot(decision)


@router.post("/api/v1/turnstile/reset")
async def reset_turnstile(container: Container = Depends(get_container)):
    container.turnstile_engine.reset()
    return {"status": "reset"}


@router.get("/decision/latest", response_model=LatestDecisionResponse)
async def latest_decision(container: Container = Depends(get_container)):
    decision = container.turnstile_engine.get_latest_decision()
    return LatestDecisionResponse(
        status="ok",
        latest_decision=_decision_snapshot(decision) if decision is not None else None,
    )


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadyResponse)
@router.get("/health/ready", response_model=ReadyResponse)
async def readiness(container: Container = Depends(get_container)):
    readiness_info = container.readiness
    return ReadyResponse(
        status=readiness_info["status"],
        models_loaded=bool(readiness_info["models_loaded"]),
        asset_error=readiness_info["asset_error"],
    )
