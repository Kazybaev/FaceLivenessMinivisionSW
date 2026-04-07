import numpy as np

from app.core.pipeline import AntiSpoofPipeline


def test_temporal_evidence_keeps_static_sequence_near_neutral() -> None:
    pipeline = AntiSpoofPipeline.__new__(AntiSpoofPipeline)
    pipeline._temporal_patch_size = 64
    patch = np.full((64, 64), 0.5, dtype=np.float32)
    pipeline._recent_patches = [patch, patch.copy(), patch.copy()]

    temporal = pipeline._compute_temporal_evidence()

    assert temporal.frame_count == 3
    assert round(temporal.live_score, 2) == 0.53
    assert round(temporal.spoof_score, 2) == 0.47
    assert temporal.rigidity_score >= 0.5


def test_temporal_evidence_marks_local_motion_as_more_live_like() -> None:
    pipeline = AntiSpoofPipeline.__new__(AntiSpoofPipeline)
    pipeline._temporal_patch_size = 64
    patch_a = np.full((64, 64), 0.5, dtype=np.float32)
    patch_b = patch_a.copy()
    patch_b[20:30, 18:28] = 0.9
    patch_c = patch_a.copy()
    patch_c[34:44, 30:40] = 0.1
    pipeline._recent_patches = [patch_a, patch_b, patch_c]

    temporal = pipeline._compute_temporal_evidence()

    assert temporal.frame_count == 3
    assert temporal.live_score > temporal.spoof_score
    assert temporal.rigidity_score < 0.8
