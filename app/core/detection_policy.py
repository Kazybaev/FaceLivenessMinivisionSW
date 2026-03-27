from __future__ import annotations

import re
from typing import Iterable

from app.core.models import ObjectDetection

_LABEL_ALIASES = {
    "cell phone": "phone",
    "mobile phone": "phone",
    "smart phone": "phone",
    "smartphone": "phone",
    "iphone": "phone",
    "android phone": "phone",
    "tablet computer": "tablet",
    "ipad": "tablet",
    "monitor screen": "monitor",
    "computer monitor": "monitor",
    "screen": "screen",
    "display": "screen",
    "tablet screen": "tablet",
    "laptop": "laptop screen",
    "laptop screen": "laptop screen",
    "notebook screen": "laptop screen",
    "television": "monitor",
    "photo": "photo",
    "paper photo": "paper photo",
    "printed photo": "printed photo",
    "printed photograph": "printed photo",
    "photo print": "printed photo",
    "photo sheet": "printed photo",
    "paper/photo sheet": "printed photo",
    "paper": "paper photo",
    "printed picture": "printed photo",
    "picture": "photo",
}


def normalize_detection_label(label: str) -> str:
    normalized = re.sub(r"[\s_\-]+", " ", label.strip().lower())
    normalized = _LABEL_ALIASES.get(normalized, normalized)

    if "phone" in normalized:
        return "phone"
    if "tablet" in normalized or "ipad" in normalized:
        return "tablet"
    if "laptop" in normalized:
        return "laptop screen"
    if "monitor" in normalized:
        return "monitor"
    if "screen" in normalized or "display" in normalized:
        return "screen"
    if "paper" in normalized and "photo" in normalized:
        return "paper photo"
    if "printed" in normalized or "sheet" in normalized:
        return "printed photo"
    if "photo" in normalized or "picture" in normalized:
        return "photo"
    return normalized


def normalize_suspicious_labels(labels: Iterable[str]) -> set[str]:
    return {normalize_detection_label(label) for label in labels}


def filter_suspicious_detections(
    detections: list[ObjectDetection],
    suspicious_labels: Iterable[str],
) -> list[ObjectDetection]:
    normalized_suspicious = normalize_suspicious_labels(suspicious_labels)
    suspicious_detections: list[ObjectDetection] = []
    for detection in detections:
        normalized_label = normalize_detection_label(detection.label)
        if normalized_label not in normalized_suspicious:
            continue
        suspicious_detections.append(
            ObjectDetection(
                label=normalized_label,
                confidence=detection.confidence,
                bbox=detection.bbox,
            )
        )
    return suspicious_detections


def should_block_session(
    detections: list[ObjectDetection],
    suspicious_labels: Iterable[str],
) -> tuple[bool, list[str]]:
    suspicious_detections = filter_suspicious_detections(detections, suspicious_labels)
    suspicious_types = sorted({detection.label for detection in suspicious_detections})
    return bool(suspicious_types), suspicious_types
