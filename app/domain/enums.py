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
