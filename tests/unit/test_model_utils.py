"""Tests for ML utility functions: parse_model_name, get_kernel."""
import pytest

from app.ml.utils import parse_model_name, get_kernel, get_width_height


class TestParseModelName:
    def test_standard_name(self):
        h, w, model_type, scale = parse_model_name("2.7_80x80_MiniFASNetV2.pth")
        assert h == 80
        assert w == 80
        assert model_type == "MiniFASNetV2"
        assert scale == 2.7

    def test_se_model(self):
        h, w, model_type, scale = parse_model_name("4_0_0_80x80_MiniFASNetV1SE.pth")
        assert h == 80
        assert w == 80
        assert model_type == "MiniFASNetV1SE"
        assert scale == 4.0

    def test_org_prefix(self):
        h, w, model_type, scale = parse_model_name("org_1_80x60_MiniFASNetV1.pth")
        assert h == 80
        assert w == 60
        assert model_type == "MiniFASNetV1"
        assert scale is None


class TestGetKernel:
    def test_80x80(self):
        k = get_kernel(80, 80)
        assert k == (5, 5)

    def test_80x60(self):
        k = get_kernel(80, 60)
        assert k == (5, 4)

    def test_32x32(self):
        k = get_kernel(32, 32)
        assert k == (2, 2)


class TestGetWidthHeight:
    def test_standard(self):
        w, h = get_width_height("1_80x80")
        assert w == 80
        assert h == 80

    def test_different_size(self):
        w, h = get_width_height("2.7_80x60")
        assert w == 60
        assert h == 80
