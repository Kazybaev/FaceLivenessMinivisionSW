from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class RetinaFaceConfig(BaseModel):
    confidence_threshold: float = 0.6
    input_size: int = 192
    prototxt: str = "resources/detection_model/deploy.prototxt"
    caffemodel: str = "resources/detection_model/Widerface-RetinaFace.caffemodel"


class MediaPipeConfig(BaseModel):
    model_path: str = "resources/face_landmarker.task"
    min_face_detection_confidence: float = 0.5
    min_face_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    num_faces: int = 1


class DetectorConfig(BaseModel):
    retinaface: RetinaFaceConfig = RetinaFaceConfig()
    mediapipe: MediaPipeConfig = MediaPipeConfig()


class HeuristicConfig(BaseModel):
    blinks_needed: int = 3
    ear_threshold: float = 0.22
    min_blink_frames: int = 2
    moves_needed: int = 10
    move_pixels: int = 16
    move_max_jump: int = 35
    smooth_window: int = 6
    ratio_diff_min: float = 0.25
    ratio_checks_min: int = 6
    texture_min: float = 90.0
    texture_frames: int = 20
    left_eye: list[int] = [362, 385, 387, 263, 373, 380]
    right_eye: list[int] = [33, 160, 158, 133, 153, 144]
    nose_tip: int = 1
    left_corner: int = 263
    right_corner: int = 33


class DeepLearningConfig(BaseModel):
    model_dir: str = "resources/anti_spoof_models"
    confidence_threshold: float = 0.6
    smoothing_frames: int = 10
    num_classes: int = 3


class CombinedConfig(BaseModel):
    dl_weight: float = 0.6
    heuristic_weight: float = 0.4


class AnalyzerConfig(BaseModel):
    heuristic: HeuristicConfig = HeuristicConfig()
    deep_learning: DeepLearningConfig = DeepLearningConfig()
    combined: CombinedConfig = CombinedConfig()


class TrainingConfig(BaseModel):
    lr: float = 0.1
    milestones: list[int] = [10, 15, 22]
    gamma: float = 0.1
    epochs: int = 25
    momentum: float = 0.9
    weight_decay: float = 5e-4
    batch_size: int = 1024
    num_classes: int = 3
    input_channel: int = 3
    embedding_size: int = 128
    train_root_path: str = "./datasets/rgb_image"
    snapshot_dir_path: str = "./saved_logs/snapshot"
    log_path: str = "./saved_logs/jobs"
    board_loss_every: int = 10
    save_every: int = 30
    num_workers: int = 16
    cls_loss_weight: float = 0.5
    ft_loss_weight: float = 0.5


class CameraConfig(BaseModel):
    id: str = "camera-0"
    index: int = 0
    width: int = 1280
    height: int = 720


class TurnstileThresholdsConfig(BaseModel):
    min_face_ratio: float = 0.07
    max_face_ratio: float = 0.60
    min_brightness: float = 0.12
    max_brightness: float = 0.96
    min_blur_var: float = 22.0
    texture_real_min: float = 0.30
    flow_rigidity_spoof: float = 0.94
    parallax_live_min: float = 0.004
    motion_min: float = 0.45
    motion_max: float = 14.0
    edge_screen_high: float = 240.0
    flicker_min_hz: float = 7.0
    flicker_max_hz: float = 35.0
    flicker_amp_spoof: float = 0.010
    glare_ratio: float = 0.13
    live_score_threshold: float = 0.62
    spoof_score_threshold: float = 0.42
    dl_fake_override_threshold: float = 0.82
    dl_real_min_threshold: float = 0.68
    heuristic_weight: float = 0.55
    deep_learning_weight: float = 0.45
    dl_interval_frames: int = 3


class TurnstileConfig(BaseModel):
    device_id: str = "edge-device-001"
    decision_window_ms: int = 1200
    cooldown_ms: int = 1400
    min_good_frames: int = 8
    webhook_url: str | None = None
    webhook_timeout_ms: int = 800
    audit_log_dir: str = "logs/audit"
    control_plane_enabled: bool = True
    control_plane_host: str = "127.0.0.1"
    control_plane_port: int = 8000
    thresholds: TurnstileThresholdsConfig = TurnstileThresholdsConfig()


class ApiConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = 1
    cors_origins: list[str] = ["http://127.0.0.1", "http://localhost"]
    max_upload_size: int = 10_485_760


class AppSettings(BaseSettings):
    detector: DetectorConfig = DetectorConfig()
    analyzer: AnalyzerConfig = AnalyzerConfig()
    training: TrainingConfig = TrainingConfig()
    camera: CameraConfig = CameraConfig()
    turnstile: TurnstileConfig = TurnstileConfig()
    api: ApiConfig = ApiConfig()

    model_config = {"env_prefix": "LIVENESS_", "env_nested_delimiter": "__"}


def _find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "config" / "default.yaml").exists():
            return parent
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    root = _find_project_root()
    config_path = root / "config" / "default.yaml"
    yaml_data: dict = {}
    if config_path.exists():
        with open(config_path, encoding="utf-8") as file_obj:
            yaml_data = yaml.safe_load(file_obj) or {}
    return AppSettings(**yaml_data)
