from __future__ import annotations

from datetime import datetime, timezone
import time


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def monotonic_seconds() -> float:
    return time.monotonic()


def remaining_seconds(deadline: float | None, now: float) -> float:
    if deadline is None:
        return 0.0
    return max(0.0, deadline - now)
