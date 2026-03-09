"""Shared test fixtures."""
from __future__ import annotations

import numpy as np
import pytest

from app.domain.entities import BBox, FaceRegion


@pytest.fixture
def sample_image() -> np.ndarray:
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_face_image() -> np.ndarray:
    return np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)


@pytest.fixture
def sample_bbox() -> BBox:
    return BBox(x=100, y=100, width=200, height=200)


@pytest.fixture
def sample_face_region(sample_face_image, sample_bbox) -> FaceRegion:
    return FaceRegion(bbox=sample_bbox, image=sample_face_image)


@pytest.fixture
def sample_face_with_landmarks(sample_face_image, sample_bbox) -> FaceRegion:
    landmarks = np.random.rand(478, 2).astype(np.float32) * 80
    return FaceRegion(bbox=sample_bbox, image=sample_face_image, landmarks=landmarks)
