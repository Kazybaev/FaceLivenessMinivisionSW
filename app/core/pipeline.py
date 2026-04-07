"""End-to-end frame pipeline."""
from __future__ import annotations

from collections import deque
import logging

import cv2
import numpy as np

from app.config import Settings
from app.core.decision_engine import DecisionEngine
from app.core.enums import AntiSpoofLabel
from app.core.schemas import AntiSpoofInference, Decision, FrameAnalysis, LandmarkEvidence, LandmarkObservation, ModelScore, PresentationAttackEvidence, TemporalEvidence
from app.services.anti_spoof_service import AntiSpoofService
from app.services.event_logger import EventLogger
from app.services.face_cropper import FaceCropper
from app.services.face_detector import FaceDetector
from app.services.landmark_liveness_service import LandmarkLivenessService
from app.services.overlay_service import OverlayService
from app.services.presentation_attack_service import PresentationAttackService
from app.services.quality_service import FaceQualityService
from app.utils.image_utils import resize_frame_if_needed
from app.utils.timers import FPSCounter


class AntiSpoofPipeline:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._logger = logging.getLogger(__name__)
        self._face_detector = FaceDetector(settings)
        self._face_cropper = FaceCropper(settings.face_margin_ratio)
        self._quality_service = FaceQualityService(settings)
        self._anti_spoof_service = AntiSpoofService(settings)
        self._landmark_liveness_service = LandmarkLivenessService(settings.face_landmarker_weights)
        self._presentation_attack_service = PresentationAttackService(settings)
        self._decision_engine = DecisionEngine(settings)
        self._overlay_service = OverlayService(settings)
        self._event_logger = EventLogger(settings.event_log_path)
        self._fps_counter = FPSCounter()
        self._recent_scores: deque[AntiSpoofInference] = deque(maxlen=max(1, settings.temporal_window))
        self._recent_patches: deque[np.ndarray] = deque(maxlen=max(1, settings.temporal_window))
        self._recent_landmarks: deque[LandmarkObservation] = deque(maxlen=max(1, settings.temporal_window))
        self._temporal_patch_size = max(24, settings.temporal_patch_size)

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, FrameAnalysis]:
        frame = resize_frame_if_needed(frame, self._settings.max_frame_width)
        fps = self._fps_counter.tick()
        detection = self._face_detector.detect(frame)

        if detection is None:
            self._recent_scores.clear()
            self._recent_patches.clear()
            self._recent_landmarks.clear()
            decision = self._decision_engine.no_face()
            self._event_logger.log_frame(
                decision=decision,
                inference=None,
                inference_time_ms=0.0,
                face_detected=False,
                bbox=None,
            )
            overlay = self._overlay_service.draw(
                frame,
                face_detection=None,
                decision=decision,
                inference=None,
                inference_time_ms=0.0,
                fps=fps,
            )
            return overlay, FrameAnalysis(decision=decision, face_detection=None, inference_time_ms=0.0, fps=fps)

        try:
            face_crop, crop_box = self._face_cropper.crop(frame, detection.bbox)
            quality = self._quality_service.assess(frame, face_crop)
            prepared_face = self._quality_service.prepare_for_inference(face_crop, quality)
            prepared_frame = frame.copy()
            prepared_frame[crop_box.y:crop_box.y2, crop_box.x:crop_box.x2] = prepared_face

            raw = self._anti_spoof_service.infer(prepared_frame, detection.bbox)
            raw.presentation = self._presentation_attack_service.analyze(frame, detection.bbox)
            landmark_observation = self._landmark_liveness_service.detect(prepared_face)
            if landmark_observation is None:
                self._recent_landmarks.clear()
            else:
                self._recent_landmarks.append(landmark_observation)
                raw.landmark_points = self._to_frame_landmark_points(landmark_observation, crop_box)
            self._recent_scores.append(raw)
            self._recent_patches.append(self._prepare_temporal_patch(prepared_face))

            inference = self._aggregate_recent_scores()
            inference.temporal = self._compute_temporal_evidence()
            inference.landmarks = self._compute_landmark_evidence()
            decision = self._decision_engine.decide(inference)
            if not quality.ok:
                decision = Decision(
                    label=decision.label,
                    confidence=decision.confidence,
                    real_score=decision.real_score,
                    spoof_score=decision.spoof_score,
                    reason=f"{decision.reason} Quality note: {quality.message}",
                )
            inference_ms = raw.inference_time_ms
            error = None
        except Exception as exc:
            self._logger.exception("pipeline_error")
            self._recent_scores.clear()
            self._recent_patches.clear()
            self._recent_landmarks.clear()
            decision = Decision(
                label=AntiSpoofLabel.FAKE,
                confidence=1.0,
                real_score=0.0,
                spoof_score=1.0,
                reason=f"Anti-spoof inference failed: {exc}",
            )
            inference = None
            inference_ms = 0.0
            error = str(exc)

        self._event_logger.log_frame(
            decision=decision,
            inference=inference,
            inference_time_ms=inference_ms,
            face_detected=True,
            bbox=detection.bbox,
            error=error,
        )
        overlay = self._overlay_service.draw(
            frame,
            face_detection=detection,
            decision=decision,
            inference=inference,
            inference_time_ms=inference_ms,
            fps=fps,
        )
        return overlay, FrameAnalysis(
            decision=decision,
            face_detection=detection,
            inference_time_ms=inference_ms,
            fps=fps,
            inference=inference,
        )

    def _aggregate_recent_scores(self) -> AntiSpoofInference:
        if not self._recent_scores:
            return AntiSpoofInference(
                real_score=0.0,
                spoof_score=0.0,
                inference_time_ms=0.0,
                raw_scores=None,
                window_size=0,
                model_scores=[],
            )

        weights = self._recent_weights(len(self._recent_scores))
        weight_sum = sum(weights)
        real_score = sum(item.real_score * weight for item, weight in zip(self._recent_scores, weights)) / weight_sum
        spoof_score = sum(item.spoof_score * weight for item, weight in zip(self._recent_scores, weights)) / weight_sum
        return AntiSpoofInference(
            real_score=float(real_score),
            spoof_score=float(spoof_score),
            inference_time_ms=float(self._recent_scores[-1].inference_time_ms),
            raw_scores=self._recent_scores[-1].raw_scores,
            window_size=len(self._recent_scores),
            model_scores=self._aggregate_model_scores(),
            presentation=self._aggregate_presentation(),
            landmark_points=self._latest_landmark_points(),
        )

    def _aggregate_model_scores(self) -> list[ModelScore]:
        grouped: dict[str, list[ModelScore]] = {}
        for inference in self._recent_scores:
            for score in inference.model_scores:
                grouped.setdefault(score.model_name, []).append(score)

        aggregated: list[ModelScore] = []
        for model_name, samples in grouped.items():
            weights = self._recent_weights(len(samples))
            weight_sum = sum(weights)
            aggregated.append(
                ModelScore(
                    model_name=model_name,
                    real_score=float(sum(sample.real_score * weight for sample, weight in zip(samples, weights)) / weight_sum),
                    spoof_score=float(sum(sample.spoof_score * weight for sample, weight in zip(samples, weights)) / weight_sum),
                    confidence=float(sum(sample.confidence * weight for sample, weight in zip(samples, weights)) / weight_sum),
                    weight=float(sum(sample.weight for sample in samples) / len(samples)),
                )
            )
        aggregated.sort(key=lambda item: item.model_name.lower())
        return aggregated

    def _aggregate_presentation(self) -> PresentationAttackEvidence | None:
        samples = [item.presentation for item in self._recent_scores if item.presentation is not None]
        if not samples:
            return None
        weights = self._recent_weights(len(samples))
        weight_sum = sum(weights)
        attack_score = sum(sample.attack_score * weight for sample, weight in zip(samples, weights)) / weight_sum
        bezel_score = sum(sample.bezel_score * weight for sample, weight in zip(samples, weights)) / weight_sum
        rectangle_score = sum(sample.rectangle_score * weight for sample, weight in zip(samples, weights)) / weight_sum
        glare_score = sum(sample.glare_score * weight for sample, weight in zip(samples, weights)) / weight_sum
        return PresentationAttackEvidence(
            attack_score=float(attack_score),
            bezel_score=float(bezel_score),
            rectangle_score=float(rectangle_score),
            glare_score=float(glare_score),
            detail=(
                f"screen={attack_score:.2f} bezel={bezel_score:.2f} "
                f"rect={rectangle_score:.2f} glare={glare_score:.2f}"
            ),
        )

    def _latest_landmark_points(self) -> np.ndarray | None:
        for item in reversed(self._recent_scores):
            if item.landmark_points is not None:
                return item.landmark_points
        return None

    def _compute_landmark_evidence(self) -> LandmarkEvidence | None:
        observations = list(self._recent_landmarks)
        if not observations:
            return None

        point_count = int(np.mean([item.points.shape[0] for item in observations]))
        depth_ranges = [float(np.ptp(item.depth_values)) for item in observations]
        depth_score = float(np.clip((float(np.mean(depth_ranges)) - 0.03) / 0.10, 0.0, 1.0))

        if len(observations) < 2:
            live_score = float(np.clip(0.48 + 0.18 * depth_score, 0.0, 1.0))
            return LandmarkEvidence(
                live_score=live_score,
                spoof_score=float(1.0 - live_score),
                motion_score=0.0,
                depth_score=depth_score,
                rigidity_score=0.5,
                frame_count=len(observations),
                point_count=point_count,
                detail=f"pts={point_count} live={live_score:.2f} depth={depth_score:.2f} rigid=0.50",
            )

        motion_samples: list[float] = []
        diversity_samples: list[float] = []
        residual_samples: list[float] = []
        for previous, current in zip(observations[:-1], observations[1:]):
            prev_points = previous.points.astype(np.float32)
            curr_points = current.points.astype(np.float32)
            displacement = np.linalg.norm(curr_points - prev_points, axis=1)
            motion_samples.append(float(displacement.mean()))
            diversity_samples.append(float(np.std(displacement) / max(np.mean(displacement), 1e-6)))

            transform, _ = cv2.estimateAffinePartial2D(prev_points, curr_points)
            if transform is None:
                residual_samples.append(0.0)
                continue
            ones = np.ones((prev_points.shape[0], 1), dtype=np.float32)
            predicted = np.hstack([prev_points, ones]) @ transform.T
            residual = np.linalg.norm(curr_points - predicted, axis=1)
            residual_samples.append(float(residual.mean()))

        motion_score = float(np.clip((float(np.mean(motion_samples)) - 0.002) / 0.030, 0.0, 1.0))
        diversity_score = float(np.clip(float(np.mean(diversity_samples)) / 0.60, 0.0, 1.0))
        non_rigid_score = float(np.clip((float(np.mean(residual_samples)) - 0.0015) / 0.020, 0.0, 1.0))
        live_score = float(np.clip(0.25 * depth_score + 0.25 * motion_score + 0.20 * diversity_score + 0.30 * non_rigid_score, 0.0, 1.0))
        if motion_score < 0.06 and non_rigid_score < 0.08:
            live_score = max(live_score, 0.48)

        rigidity_score = float(1.0 - non_rigid_score)
        return LandmarkEvidence(
            live_score=live_score,
            spoof_score=float(1.0 - live_score),
            motion_score=motion_score,
            depth_score=depth_score,
            rigidity_score=rigidity_score,
            frame_count=len(observations),
            point_count=point_count,
            detail=(
                f"pts={point_count} live={live_score:.2f} "
                f"motion={motion_score:.2f} depth={depth_score:.2f} rigid={rigidity_score:.2f}"
            ),
        )

    @staticmethod
    def _to_frame_landmark_points(observation: LandmarkObservation, crop_box) -> np.ndarray:
        if observation.points.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        points = observation.points.astype(np.float32).copy()
        points[:, 0] = crop_box.x + points[:, 0] * max(crop_box.width, 1)
        points[:, 1] = crop_box.y + points[:, 1] * max(crop_box.height, 1)
        return points

    @staticmethod
    def _recent_weights(length: int) -> list[float]:
        return [float(index + 1) for index in range(length)]

    def _prepare_temporal_patch(self, face_crop: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (self._temporal_patch_size, self._temporal_patch_size))
        return resized.astype(np.float32) / 255.0

    def _compute_temporal_evidence(self) -> TemporalEvidence:
        patches = list(self._recent_patches)
        if len(patches) < 2:
            return TemporalEvidence(
                live_score=0.5,
                spoof_score=0.5,
                motion_score=0.0,
                diversity_score=0.0,
                rigidity_score=0.5,
                frame_count=len(patches),
                detail="warming up",
            )

        motion_samples: list[float] = []
        diversity_samples: list[float] = []
        rigidity_samples: list[float] = []
        for previous, current in zip(patches[:-1], patches[1:]):
            diff = np.abs(current - previous)
            motion_samples.append(float(diff.mean()))
            cell_means = self._grid_cell_means(diff, grid_size=4)
            diversity = float(np.std(cell_means) / max(np.mean(cell_means), 1e-6))
            diversity_samples.append(diversity)
            rigidity_samples.append(self._estimate_rigidity(previous, current))

        motion_value = float(np.mean(motion_samples))
        diversity_value = float(np.mean(diversity_samples))
        rigidity_value = float(np.mean(rigidity_samples)) if rigidity_samples else 0.5
        motion_score = float(np.clip((motion_value - 0.003) / 0.028, 0.0, 1.0))
        diversity_score = float(np.clip(diversity_value / 0.65, 0.0, 1.0))
        if motion_score < 0.12:
            live_score = 0.53
            spoof_score = 0.47
        else:
            non_rigid_score = float(1.0 - rigidity_value)
            live_score = float(np.clip(0.20 * motion_score + 0.35 * diversity_score + 0.45 * non_rigid_score, 0.0, 1.0))
            spoof_score = float(1.0 - live_score)
        return TemporalEvidence(
            live_score=live_score,
            spoof_score=spoof_score,
            motion_score=motion_score,
            diversity_score=diversity_score,
            rigidity_score=rigidity_value,
            frame_count=len(patches),
            detail=f"motion={motion_score:.2f} diversity={diversity_score:.2f} rigid={rigidity_value:.2f}",
        )

    @staticmethod
    def _grid_cell_means(diff_frame: np.ndarray, grid_size: int) -> np.ndarray:
        height, width = diff_frame.shape[:2]
        cell_values: list[float] = []
        for row in range(grid_size):
            for col in range(grid_size):
                y1 = int(row * height / grid_size)
                y2 = int((row + 1) * height / grid_size)
                x1 = int(col * width / grid_size)
                x2 = int((col + 1) * width / grid_size)
                cell = diff_frame[y1:y2, x1:x2]
                if cell.size != 0:
                    cell_values.append(float(cell.mean()))
        return np.asarray(cell_values, dtype=np.float32)

    def _estimate_rigidity(self, previous: np.ndarray, current: np.ndarray) -> float:
        prev_u8 = np.clip(previous * 255.0, 0.0, 255.0).astype(np.uint8)
        curr_u8 = np.clip(current * 255.0, 0.0, 255.0).astype(np.uint8)
        points = cv2.goodFeaturesToTrack(prev_u8, maxCorners=24, qualityLevel=0.02, minDistance=4, blockSize=5)
        if points is None or len(points) < 8:
            return 0.5
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_u8, curr_u8, points, None)
        if next_points is None or status is None:
            return 0.5

        valid_prev = points[status.flatten() == 1].reshape(-1, 2)
        valid_next = next_points[status.flatten() == 1].reshape(-1, 2)
        if len(valid_prev) < 8:
            return 0.5

        transform, _ = cv2.estimateAffinePartial2D(valid_prev, valid_next)
        if transform is None:
            return 0.5

        ones = np.ones((valid_prev.shape[0], 1), dtype=np.float32)
        homogeneous = np.hstack([valid_prev.astype(np.float32), ones])
        predicted = homogeneous @ transform.T
        residual = np.linalg.norm(valid_next - predicted, axis=1)
        normalized_residual = float(np.mean(residual) / max(float(self._temporal_patch_size), 1.0))
        non_rigid_score = float(np.clip(normalized_residual / 0.025, 0.0, 1.0))
        return float(1.0 - non_rigid_score)
