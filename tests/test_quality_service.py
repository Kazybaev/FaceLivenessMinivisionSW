import numpy as np

from app.config import Settings
from app.services.quality_service import FaceQualityService


def test_quality_service_flags_dark_backlit_face() -> None:
    service = FaceQualityService(Settings())
    frame = np.full((300, 300, 3), 220, dtype=np.uint8)
    face_crop = np.full((120, 120, 3), 35, dtype=np.uint8)
    assessment = service.assess(frame, face_crop)
    assert not assessment.ok
    assert any("dark" in reason.lower() or "backlight" in reason.lower() for reason in assessment.reasons)
