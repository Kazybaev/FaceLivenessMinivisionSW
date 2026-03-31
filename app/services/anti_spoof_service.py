from __future__ import annotations

from typing import Protocol, Sequence

import cv2
import numpy as np

from app.config import AntiSpoofSettings
from app.adapters.analyzers.deep_learning_analyzer import DeepLearningAnalyzer
from app.adapters.repositories.filesystem_model_repo import FilesystemModelRepo
from app.infrastructure.config import DeepLearningConfig
from app.core.models import AntiSpoofLabel, AntiSpoofResult, FramePacket, TrackedFace
from app.utils.image_utils import crop_bbox


class BaseAntiSpoofModel(Protocol):
    backend_name: str

    def predict(
        self,
        frames: Sequence[FramePacket],
        tracked_face: TrackedFace,
    ) -> AntiSpoofResult | None:
        ...


class MockTemporalAntiSpoofModel:
    """Simple temporal heuristic mock that accepts a buffer of frames."""

    backend_name = "mock_temporal"

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


class MiniFASNetTemporalAntiSpoofModel:
    """Hybrid anti-spoof backend: MiniFASNet ensemble plus temporal heuristics."""

    backend_name = "minifasnet_temporal"

    def __init__(self, settings: AntiSpoofSettings):
        if not settings.model_dir:
            raise ValueError("Anti-spoof model_dir is required for MiniFASNet backend")

        self._settings = settings
        self._temporal = MockTemporalAntiSpoofModel(settings)
        repo = FilesystemModelRepo(settings.model_dir)
        dl_config = DeepLearningConfig(
            model_dir=settings.model_dir,
            confidence_threshold=settings.real_confidence_threshold,
            smoothing_frames=settings.frame_buffer_size,
        )
        self._deep_learning = DeepLearningAnalyzer(dl_config, repo)
        self._model_names = self._deep_learning.model_names

    def predict(
        self,
        frames: Sequence[FramePacket],
        tracked_face: TrackedFace,
    ) -> AntiSpoofResult | None:
        if len(frames) < self._settings.min_frames_for_inference:
            return None

        temporal_result = self._temporal.predict(frames, tracked_face)
        sampled_frames = self._sample_frames(frames)
        real_scores: list[float] = []
        fake_scores: list[float] = []

        for packet in sampled_frames:
            crop = crop_bbox(packet.frame, tracked_face.bbox)
            if crop.size == 0:
                continue
            prediction = self._deep_learning.predict_face(crop, smooth=False)
            real_prob = prediction["real_prob"]
            fake_prob = prediction["fake_prob"]
            if real_prob is None or fake_prob is None:
                continue
            real_scores.append(float(real_prob))
            fake_scores.append(float(fake_prob))

        if not real_scores:
            return temporal_result

        dl_real = float(np.mean(real_scores))
        dl_fake = float(np.mean(fake_scores))
        temporal_real = self._temporal_real_score(temporal_result)
        combined_real = (
            self._settings.deep_learning_weight * dl_real
            + self._settings.temporal_weight * temporal_real
        )
        details = {
            "dl_real": dl_real,
            "dl_fake": dl_fake,
            "temporal_real": temporal_real,
            "combined_real": combined_real,
            "sampled_frames": len(real_scores),
            "model_names": self._model_names,
        }
        if temporal_result is not None:
            details["temporal"] = {
                "label": temporal_result.label.value,
                "confidence": temporal_result.confidence,
                "details": temporal_result.details,
            }

        # Deep-learning fake override is the critical protection for close phone-screen photos.
        if dl_fake >= self._settings.strong_fake_threshold:
            return AntiSpoofResult(
                label=AntiSpoofLabel.SPOOF,
                confidence=dl_fake,
                model_name="minifasnet_ensemble",
                details=details,
            )

        if temporal_result is not None and temporal_result.label == AntiSpoofLabel.SPOOF and dl_fake >= self._settings.fake_threshold:
            return AntiSpoofResult(
                label=AntiSpoofLabel.SPOOF,
                confidence=max(dl_fake, temporal_result.confidence),
                model_name="minifasnet_ensemble",
                details=details,
            )

        if combined_real >= self._settings.real_threshold and dl_fake < 0.45:
            return AntiSpoofResult(
                label=AntiSpoofLabel.REAL,
                confidence=combined_real,
                model_name="minifasnet_ensemble",
                details=details,
            )

        if dl_fake >= self._settings.fake_threshold:
            return AntiSpoofResult(
                label=AntiSpoofLabel.SPOOF,
                confidence=dl_fake,
                model_name="minifasnet_ensemble",
                details=details,
            )

        return AntiSpoofResult(
            label=AntiSpoofLabel.UNCERTAIN,
            confidence=max(abs(combined_real - 0.5) * 2.0, self._settings.uncertain_confidence),
            model_name="minifasnet_ensemble",
            details=details,
        )

    def _sample_frames(self, frames: Sequence[FramePacket]) -> list[FramePacket]:
        frame_count = min(self._settings.frame_sample_count, len(frames))
        if frame_count >= len(frames):
            return list(frames)
        indices = np.linspace(0, len(frames) - 1, frame_count, dtype=int)
        return [frames[index] for index in indices.tolist()]

    @staticmethod
    def _temporal_real_score(result: AntiSpoofResult | None) -> float:
        if result is None or result.label == AntiSpoofLabel.UNCERTAIN:
            return 0.5
        if result.label == AntiSpoofLabel.REAL:
            return result.confidence
        return 1.0 - result.confidence


def create_anti_spoof_model(settings: AntiSpoofSettings) -> BaseAntiSpoofModel:
    if settings.backend == "minifasnet" and settings.model_dir:
        return MiniFASNetTemporalAntiSpoofModel(settings)
    return MockTemporalAntiSpoofModel(settings)
