import numpy as np

from app.core.pipeline import AntiSpoofPipeline
from app.core.schemas import LandmarkObservation


def _obs(points: np.ndarray, depth_values: np.ndarray) -> LandmarkObservation:
    return LandmarkObservation(points=points.astype(np.float32), depth_values=depth_values.astype(np.float32))


def test_landmark_evidence_keeps_static_points_near_neutral() -> None:
    pipeline = AntiSpoofPipeline.__new__(AntiSpoofPipeline)
    points = np.asarray(
        [
            [0.50, 0.12],
            [0.50, 0.36],
            [0.50, 0.82],
            [0.32, 0.32],
            [0.40, 0.33],
            [0.68, 0.32],
            [0.60, 0.33],
            [0.38, 0.62],
            [0.62, 0.62],
            [0.50, 0.58],
            [0.50, 0.66],
            [0.26, 0.48],
            [0.74, 0.48],
        ],
        dtype=np.float32,
    )
    depth = np.asarray([0.01, -0.04, 0.06, -0.01, -0.02, -0.01, -0.02, 0.00, 0.00, 0.01, 0.02, 0.02, 0.02], dtype=np.float32)
    pipeline._recent_landmarks = [_obs(points, depth), _obs(points.copy(), depth.copy())]

    evidence = pipeline._compute_landmark_evidence()

    assert evidence is not None
    assert evidence.frame_count == 2
    assert evidence.live_score >= 0.48
    assert evidence.rigidity_score >= 0.8


def test_landmark_evidence_marks_non_rigid_points_as_more_live_like() -> None:
    pipeline = AntiSpoofPipeline.__new__(AntiSpoofPipeline)
    points_a = np.asarray(
        [
            [0.50, 0.12],
            [0.50, 0.36],
            [0.50, 0.82],
            [0.32, 0.32],
            [0.40, 0.33],
            [0.68, 0.32],
            [0.60, 0.33],
            [0.38, 0.62],
            [0.62, 0.62],
            [0.50, 0.58],
            [0.50, 0.66],
            [0.26, 0.48],
            [0.74, 0.48],
        ],
        dtype=np.float32,
    )
    points_b = points_a.copy()
    points_b[[3, 4, 7, 9], 0] -= 0.015
    points_b[[5, 6, 8, 10], 0] += 0.014
    points_b[[1, 9, 10], 1] += 0.010
    points_c = points_a.copy()
    points_c[[3, 4, 7], 1] += 0.012
    points_c[[5, 6, 8], 1] -= 0.010
    points_c[[1, 9, 10], 0] += 0.009
    depth = np.asarray([0.01, -0.05, 0.07, -0.01, -0.02, -0.01, -0.02, 0.00, 0.00, 0.01, 0.02, 0.03, 0.03], dtype=np.float32)
    pipeline._recent_landmarks = [_obs(points_a, depth), _obs(points_b, depth), _obs(points_c, depth)]

    evidence = pipeline._compute_landmark_evidence()

    assert evidence is not None
    assert evidence.frame_count == 3
    assert evidence.live_score > 0.5
    assert evidence.rigidity_score < 0.8
