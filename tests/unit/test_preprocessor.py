"""Tests for image preprocessor."""
import numpy as np
import pytest

from app.adapters.preprocessors.opencv_preprocessor import OpenCVPreprocessor
from app.domain.entities import BBox


class TestOpenCVPreprocessor:
    def test_crop_face(self):
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bbox = BBox(x=100, y=100, width=200, height=200)
        result = OpenCVPreprocessor.crop_face(image, bbox, scale=2.7, out_w=80, out_h=80)
        assert result.shape == (80, 80, 3)

    def test_crop_face_no_scale(self):
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bbox = BBox(x=100, y=100, width=200, height=200)
        result = OpenCVPreprocessor.crop_face(image, bbox, scale=None, out_w=80, out_h=80)
        assert result.shape == (80, 80, 3)

    def test_to_tensor(self):
        image = np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)
        tensor = OpenCVPreprocessor.to_tensor(image)
        assert tensor.shape == (3, 80, 80)
        # CRITICAL: check that values are NOT divided by 255
        assert tensor.max() > 1.0

    def test_crop_face_boundary_clipping(self):
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        bbox = BBox(x=0, y=0, width=90, height=90)
        result = OpenCVPreprocessor.crop_face(image, bbox, scale=1.5, out_w=80, out_h=80)
        assert result.shape == (80, 80, 3)
