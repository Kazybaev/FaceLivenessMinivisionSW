from __future__ import annotations

from collections import deque
import threading
import uuid

import structlog

from app.core.models import EventSeverity, SecurityEvent
from app.utils.timers import utc_iso

logger = structlog.get_logger(__name__)


class EventLogger:
    """Thread-safe in-memory security event store."""

    def __init__(self, max_events: int = 200):
        self._events: deque[SecurityEvent] = deque(maxlen=max_events)
        self._lock = threading.Lock()

    def log(
        self,
        event_type: str,
        message: str,
        severity: EventSeverity = EventSeverity.INFO,
        session_id: str | None = None,
        payload: dict | None = None,
    ) -> SecurityEvent:
        event = SecurityEvent(
            event_id=str(uuid.uuid4()),
            timestamp=utc_iso(),
            event_type=event_type,
            severity=severity,
            message=message,
            session_id=session_id,
            payload=payload or {},
        )
        with self._lock:
            self._events.appendleft(event)
        logger.info(
            "security_event",
            event_type=event_type,
            severity=severity.value,
            session_id=session_id,
            payload=payload or {},
        )
        return event

    def list_events(self, limit: int = 100) -> list[SecurityEvent]:
        with self._lock:
            return list(self._events)[:limit]
