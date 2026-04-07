"""Lightweight phone/screen presentation-attack heuristics."""
from __future__ import annotations

import cv2
import numpy as np

from app.config import Settings
from app.core.schemas import BoundingBox, PresentationAttackEvidence


class PresentationAttackService:
    def __init__(self, settings: Settings) -> None:
        self._context_scale = settings.presentation_context_scale
        self._ring_inner_scale = settings.presentation_ring_inner_scale
        self._ring_outer_scale = settings.presentation_ring_outer_scale
        self._dark_value_threshold = settings.presentation_dark_value_threshold
        self._low_saturation_threshold = settings.presentation_low_saturation_threshold
        self._glare_value_threshold = settings.presentation_glare_value_threshold
        self._glare_saturation_threshold = settings.presentation_glare_saturation_threshold

    def analyze(self, frame_bgr: np.ndarray, face_bbox: BoundingBox) -> PresentationAttackEvidence:
        context_box = self._expand_bbox(face_bbox, frame_bgr.shape[1], frame_bgr.shape[0], self._context_scale)
        context = frame_bgr[context_box.y:context_box.y2, context_box.x:context_box.x2]
        if context.size == 0:
            return PresentationAttackEvidence(
                attack_score=0.0,
                bezel_score=0.0,
                rectangle_score=0.0,
                glare_score=0.0,
                detail="context=empty",
            )

        local_face = BoundingBox(
            x=face_bbox.x - context_box.x,
            y=face_bbox.y - context_box.y,
            width=face_bbox.width,
            height=face_bbox.height,
        ).clamp(context.shape[1], context.shape[0])

        gray = cv2.cvtColor(context, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(context, cv2.COLOR_BGR2HSV)
        bezel_score = self._estimate_bezel_score(gray, hsv, local_face)
        rectangle_score = self._estimate_rectangle_score(gray, local_face)
        glare_score = self._estimate_glare_score(hsv, local_face)

        attack_score = float(np.clip(0.55 * rectangle_score + 0.30 * bezel_score + 0.15 * glare_score, 0.0, 1.0))
        if rectangle_score < 0.18:
            attack_score *= 0.65

        return PresentationAttackEvidence(
            attack_score=attack_score,
            bezel_score=bezel_score,
            rectangle_score=rectangle_score,
            glare_score=glare_score,
            detail=(
                f"screen={attack_score:.2f} bezel={bezel_score:.2f} "
                f"rect={rectangle_score:.2f} glare={glare_score:.2f}"
            ),
        )

    def _estimate_bezel_score(self, gray: np.ndarray, hsv: np.ndarray, face_bbox: BoundingBox) -> float:
        ring = self._build_side_strips(gray.shape[:2], face_bbox, self._ring_inner_scale, self._ring_outer_scale)
        if not ring:
            return 0.0

        saturation = hsv[:, :, 1]
        dark_mask = (gray <= self._dark_value_threshold) & (saturation <= self._low_saturation_threshold)
        strip_scores: list[float] = []
        for x1, y1, x2, y2 in ring:
            region = dark_mask[y1:y2, x1:x2]
            if region.size == 0:
                continue
            strip_scores.append(float(region.mean()))
        if len(strip_scores) < 2:
            return 0.0
        strongest = sorted(strip_scores, reverse=True)[:3]
        return float(np.clip((float(np.mean(strongest)) - 0.10) / 0.45, 0.0, 1.0))

    def _estimate_glare_score(self, hsv: np.ndarray, face_bbox: BoundingBox) -> float:
        ring = self._build_side_strips(hsv.shape[:2], face_bbox, self._ring_inner_scale, self._ring_outer_scale)
        if not ring:
            return 0.0
        value = hsv[:, :, 2]
        saturation = hsv[:, :, 1]
        glare_mask = (value >= self._glare_value_threshold) & (saturation <= self._glare_saturation_threshold)
        strip_scores: list[float] = []
        for x1, y1, x2, y2 in ring:
            region = glare_mask[y1:y2, x1:x2]
            if region.size == 0:
                continue
            strip_scores.append(float(region.mean()))
        if not strip_scores:
            return 0.0
        return float(np.clip((max(strip_scores) - 0.01) / 0.08, 0.0, 1.0))

    def _estimate_rectangle_score(self, gray: np.ndarray, face_bbox: BoundingBox) -> float:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 60, 140)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        face_center_x = face_bbox.x + (face_bbox.width / 2.0)
        face_center_y = face_bbox.y + (face_bbox.height / 2.0)
        face_area = max(face_bbox.area, 1)
        best_score = 0.0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < face_area * 1.15:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0.0:
                continue
            approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
            if len(approx) < 4 or len(approx) > 8:
                continue

            x, y, width, height = cv2.boundingRect(approx)
            rect_area = width * height
            if rect_area < face_area * 1.15 or rect_area > face_area * 6.0:
                continue
            if x > face_bbox.x or y > face_bbox.y or x + width < face_bbox.x2 or y + height < face_bbox.y2:
                continue
            if not (x <= face_center_x <= x + width and y <= face_center_y <= y + height):
                continue

            aspect_ratio = width / max(height, 1)
            if not 0.35 <= aspect_ratio <= 1.75:
                continue

            rectangularity = float(area / max(rect_area, 1))
            if rectangularity < 0.55:
                continue

            pad_x = (width - face_bbox.width) / max(face_bbox.width, 1)
            pad_y = (height - face_bbox.height) / max(face_bbox.height, 1)
            if pad_x < 0.02 or pad_y < 0.02:
                continue
            if pad_x > 1.60 or pad_y > 1.60:
                continue

            pad_score = float(
                np.clip(1.0 - abs(pad_x - 0.45) / 1.2, 0.0, 1.0) * 0.5
                + np.clip(1.0 - abs(pad_y - 0.45) / 1.2, 0.0, 1.0) * 0.5
            )
            rect_center_x = x + (width / 2.0)
            rect_center_y = y + (height / 2.0)
            center_distance = np.hypot(rect_center_x - face_center_x, rect_center_y - face_center_y)
            center_score = float(np.clip(1.0 - center_distance / max(np.hypot(width, height) * 0.35, 1.0), 0.0, 1.0))
            score = 0.45 * rectangularity + 0.30 * pad_score + 0.25 * center_score
            best_score = max(best_score, float(score))

        return float(np.clip(best_score, 0.0, 1.0))

    @staticmethod
    def _expand_bbox(bbox: BoundingBox, frame_width: int, frame_height: int, scale: float) -> BoundingBox:
        new_width = int(round(bbox.width * scale))
        new_height = int(round(bbox.height * scale))
        center_x = bbox.x + bbox.width / 2.0
        center_y = bbox.y + bbox.height / 2.0
        return BoundingBox(
            x=int(round(center_x - (new_width / 2.0))),
            y=int(round(center_y - (new_height / 2.0))),
            width=new_width,
            height=new_height,
        ).clamp(frame_width, frame_height)

    @staticmethod
    def _build_side_strips(
        shape: tuple[int, int],
        face_bbox: BoundingBox,
        inner_scale: float,
        outer_scale: float,
    ) -> list[tuple[int, int, int, int]]:
        frame_height, frame_width = shape
        inner_box = PresentationAttackService._expand_bbox(face_bbox, frame_width, frame_height, inner_scale)
        outer_box = PresentationAttackService._expand_bbox(face_bbox, frame_width, frame_height, outer_scale)
        strips = [
            (outer_box.x, outer_box.y, outer_box.x2, inner_box.y),
            (outer_box.x, inner_box.y2, outer_box.x2, outer_box.y2),
            (outer_box.x, inner_box.y, inner_box.x, inner_box.y2),
            (inner_box.x2, inner_box.y, outer_box.x2, inner_box.y2),
        ]
        return [
            (x1, y1, x2, y2)
            for x1, y1, x2, y2 in strips
            if x2 - x1 > 2 and y2 - y1 > 2
        ]
