import cv2
import numpy as np

from app.config import Settings
from app.core.schemas import BoundingBox
from app.services.presentation_attack_service import PresentationAttackService


def _real_like_frame() -> tuple[np.ndarray, BoundingBox]:
    frame = np.full((240, 320, 3), 210, dtype=np.uint8)
    bbox = BoundingBox(x=115, y=55, width=90, height=110)
    cv2.ellipse(frame, (160, 110), (42, 54), 0, 0, 360, (170, 195, 220), -1)
    cv2.ellipse(frame, (160, 80), (44, 20), 0, 180, 360, (55, 55, 55), -1)
    cv2.rectangle(frame, (118, 170), (202, 235), (130, 130, 130), -1)
    return frame, bbox


def _phone_like_frame() -> tuple[np.ndarray, BoundingBox]:
    frame = np.full((240, 320, 3), 210, dtype=np.uint8)
    bbox = BoundingBox(x=118, y=58, width=84, height=104)
    cv2.rectangle(frame, (95, 38), (226, 182), (18, 18, 18), -1)
    cv2.rectangle(frame, (108, 50), (214, 170), (232, 232, 232), -1)
    cv2.ellipse(frame, (161, 108), (37, 46), 0, 0, 360, (170, 195, 220), -1)
    cv2.ellipse(frame, (161, 82), (39, 17), 0, 180, 360, (40, 40, 40), -1)
    cv2.line(frame, (108, 58), (214, 58), (255, 255, 255), 3)
    return frame, bbox


def test_presentation_attack_service_scores_phone_like_frame_higher_than_real_like_frame() -> None:
    service = PresentationAttackService(Settings())
    real_frame, real_bbox = _real_like_frame()
    phone_frame, phone_bbox = _phone_like_frame()

    real_evidence = service.analyze(real_frame, real_bbox)
    phone_evidence = service.analyze(phone_frame, phone_bbox)

    assert real_evidence.attack_score < 0.34
    assert phone_evidence.attack_score > real_evidence.attack_score + 0.20
    assert phone_evidence.rectangle_score > real_evidence.rectangle_score
