from __future__ import annotations

from pathlib import Path
from functools import lru_cache

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class RetinaFaceConfig(BaseModel):
    confidence_threshold: float = 0.6
    input_size: int = 192
    prototxt: str = "resources/detection_model/deploy.prototxt"
    caffemodel: str = "resources/detection_model/Widerface-RetinaFace.caffemodel"


class MediaPipeConfig(BaseModel):
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


class ApiConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    cors_origins: list[str] = ["*"]
    max_upload_size: int = 10_485_760


class AppSettings(BaseSettings):
    detector: DetectorConfig = DetectorConfig()
    analyzer: AnalyzerConfig = AnalyzerConfig()
    training: TrainingConfig = TrainingConfig()
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
        with open(config_path) as f:
            yaml_data = yaml.safe_load(f) or {}
    return AppSettings(**yaml_data)
