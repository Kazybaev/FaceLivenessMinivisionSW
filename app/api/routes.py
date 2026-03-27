from __future__ import annotations

from fastapi import APIRouter, Request

from app.core.schemas import (
    CurrentDecisionResponse,
    EventsResponse,
    HealthResponse,
    SessionResetResponse,
    StatusResponse,
)

router = APIRouter()


def _runtime(request: Request):
    return request.app.state.access_pipeline


@router.get("/status", response_model=StatusResponse)
async def status(request: Request):
    return StatusResponse(**_runtime(request).get_status())


@router.post("/session/reset", response_model=SessionResetResponse)
async def reset_session(request: Request):
    session_id = _runtime(request).reset_session()
    return SessionResetResponse(status="reset", session_id=session_id)


@router.get("/events", response_model=EventsResponse)
async def events(request: Request):
    return EventsResponse(events=_runtime(request).get_events())


@router.get("/current-decision", response_model=CurrentDecisionResponse)
async def current_decision(request: Request):
    return CurrentDecisionResponse(decision=_runtime(request).get_current_decision())


@router.get("/health/runtime", response_model=HealthResponse)
async def runtime_health(request: Request):
    status = _runtime(request).get_status()
    return HealthResponse(status="ok" if status["runtime_running"] else "starting")
