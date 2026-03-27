from enum import Enum


class LivenessVerdict(str, Enum):
    REAL = "REAL"
    FAKE = "FAKE"
    UNCERTAIN = "UNCERTAIN"
    NO_FACE = "NO_FACE"


class DetectionMethod(str, Enum):
    HEURISTIC = "heuristic"
    DEEP_LEARNING = "deep_learning"
    COMBINED = "combined"


class TurnstileState(str, Enum):
    NO_FACE = "NO_FACE"
    POSITIONING = "POSITIONING"
    ANALYZING = "ANALYZING"
    ACCESS_GRANTED = "ACCESS_GRANTED"
    ACCESS_DENIED = "ACCESS_DENIED"
    COOLDOWN = "COOLDOWN"


class ControllerVerdict(str, Enum):
    ACCESS_GRANTED = "ACCESS_GRANTED"
    ACCESS_DENIED = "ACCESS_DENIED"
