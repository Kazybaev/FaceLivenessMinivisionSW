"""Shared labels used by the runtime."""
from __future__ import annotations

from enum import Enum


class AntiSpoofLabel(str, Enum):
    REAL = "real"
    FAKE = "fake"
    NO_FACE = "no_face"
