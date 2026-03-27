from __future__ import annotations

from typing import Protocol, Sequence

import cv2
import numpy as np

from app.config import AntiSpoofSettings
from app.core.models import AntiSpoofLabel, AntiSpoofResult, FramePacket, TrackedFace
from app.utils.image_utils import crop_bbox


class BaseAntiSpoofModel(Protocol):
    def predict(
        self,
        frames: Sequence[FramePacket],
        tracked_face: TrackedFace,
    ) -> AntiSpoofResult | None:
        ...


class MockTemporalAntiSpoofModel:
    """Simple temporal heuristic mock that accepts a buffer of frames."""

    def __init__(self, settings: AntiSpoofSettings):
        self._settings = settings

    def predict(
        self,
        frames: Sequence[FramePacket],
        tracked_face: TrackedFace,
    ) -> AntiSpoofResult | None:
        if len(frames) < self._settings.min_frames_for_inference:
            return None

        crops = [crop_bbox(packet.frame, tracked_face.bbox) for packet in frames]
        crops = [crop for crop in crops if crop.size > 0]
        if len(crops) < self._settings.min_frames_for_inference:
            return AntiSpoofResult(
                label=AntiSpoofLabel.UNCERTAIN,
                confidence=self._settings.uncertain_confidence,
                model_name="mock_temporal_v1",
                details={"reason": "not_enough_valid_crops"},
            )

        gray_crops = []
        for crop in crops:
            resized = cv2.resize(
                crop,
                (self._settings.crop_size, self._settings.crop_size),
                interpolation=cv2.INTER_LINEAR,
            )
            gray_crops.append(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY))
        textures = [float(cv2.Laplacian(gray, cv2.CV_64F).var()) for gray in gray_crops]
        motions: list[float] = []
        for prev, curr in zip(gray_crops, gray_crops[1:]):
            diff = cv2.absdiff(prev, curr)
            motions.append(float(np.mean(diff)))

        avg_texture = float(np.mean(textures)) if textures else 0.0
        avg_motion = float(np.mean(motions)) if motions else 0.0
        details = {
            "avg_texture": avg_texture,
            "avg_motion": avg_motion,
            "buffer_size": len(crops),
        }

        if (
            avg_texture <= self._settings.max_texture_for_spoof
            and avg_motion <= self._settings.max_motion_for_spoof
        ):
            return AntiSpoofResult(
                label=AntiSpoofLabel.SPOOF,
                confidence=0.88,
                model_name="mock_temporal_v1",
                details=details,
            )

        if (
            avg_texture >= self._settings.min_texture_for_real
            and self._settings.min_motion_for_real <= avg_motion <= self._settings.max_motion_for_real
        ):
            return AntiSpoofResult(
                label=AntiSpoofLabel.REAL,
                confidence=0.78,
                model_name="mock_temporal_v1",
                details=details,
            )

        return AntiSpoofResult(
            label=AntiSpoofLabel.UNCERTAIN,
            confidence=self._settings.uncertain_confidence,
            model_name="mock_temporal_v1",
            details=details,
        )
