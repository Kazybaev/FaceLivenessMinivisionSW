from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_YOLO_MODEL_CANDIDATES = (
    _PROJECT_ROOT / "models" / "yolo26n.pt",
    _PROJECT_ROOT / "models" / "yolo26s.pt",
    _PROJECT_ROOT / "models" / "yolo26m.pt",
    _PROJECT_ROOT / "models" / "yolo11n.pt",
    _PROJECT_ROOT / "models" / "yolov8n.pt",
)
_DEFAULT_ANTI_SPOOF_MODEL_DIR_CANDIDATES = (
    _PROJECT_ROOT / "resources" / "anti_spoof_models",
    _PROJECT_ROOT / "assets" / "anti_spoof_models",
)
_DEFAULT_ACTIVE_LIVENESS_MODEL_CANDIDATES = (
    _PROJECT_ROOT / "resources" / "face_landmarker.task",
    _PROJECT_ROOT / "face_landmarker.task",
)


class ApiSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: list[str] = ["*"]


class CameraSettings(BaseModel):
    index: int = 0
    source: str | None = None
    loop_video: bool = True
    width: int = 1280
    height: int = 720
    reconnect_interval_seconds: float = 2.0
    read_sleep_seconds: float = 0.0


class YoloSettings(BaseModel):
    backend: str = "mock"
    model_path: str | None = None
    confidence_threshold: float = 0.25
    inference_size: int = 640
    run_every_n_frames: int = 2
    suspicious_labels: list[str] = Field(
        default_factory=lambda: [
            "phone",
            "cell phone",
            "smartphone",
            "mobile phone",
            "tablet",
            "ipad",
            "monitor",
            "display",
            "laptop",
            "laptop screen",
            "screen",
            "photo",
            "printed photo",
            "paper photo",
            "photo sheet",
            "paper",
        ]
    )


class FaceSettings(BaseModel):
    min_face_size: int = 64
    tracker_iou_threshold: float = 0.3
    tracker_max_missing_seconds: float = 1.5
    detection_scale: float = 0.5


class AntiSpoofSettings(BaseModel):
    backend: str = "mock"
    model_dir: str | None = None
    frame_buffer_size: int = 4
    min_frames_for_inference: int = 3
    inference_interval_frames: int = 3
    max_cached_result_age_seconds: float = 0.35
    rerun_iou_threshold: float = 0.9
    real_confidence_threshold: float = 0.75
    crop_size: int = 96
    frame_sample_count: int = 3
    deep_learning_weight: float = 0.7
    temporal_weight: float = 0.3
    strong_fake_threshold: float = 0.70
    fake_threshold: float = 0.58
    real_threshold: float = 0.76
    active_assisted_real_threshold: float = 0.60
    active_assisted_max_fake_probability: float = 0.32
    min_texture_for_real: float = 18.0
    min_motion_for_real: float = 1.2
    max_motion_for_real: float = 24.0
    max_texture_for_spoof: float = 9.0
    max_motion_for_spoof: float = 0.8
    uncertain_confidence: float = 0.55


class ActiveLivenessSettings(BaseModel):
    enabled: bool = True
    require_for_allow: bool = True
    backend: str = "mock"
    model_path: str | None = None
    run_every_n_frames: int = 2
    challenge_timeout_seconds: float = 2.5
    min_face_iou: float = 0.20
    blinks_needed: int = 1
    ear_threshold: float = 0.22
    min_blink_frames: int = 2
    moves_needed: int = 2
    move_pixels: int = 10
    move_max_jump: int = 42
    smooth_window: int = 6
    ratio_diff_min: float = 0.14
    ratio_checks_min: int = 2
    texture_min: float = 70.0
    texture_frames: int = 16
    min_spoof_signal_seconds: float = 0.9
    parallax_live_min: float = 0.004
    flow_rigidity_spoof: float = 0.985
    flicker_min_hz: float = 7.0
    flicker_max_hz: float = 35.0
    flicker_amp_spoof: float = 0.010
    texture_spoof_max: float = 0.18


class SessionSettings(BaseModel):
    session_timeout_seconds: float = 10.0
    suspicious_cooldown_seconds: float = 0.0
    spoof_cooldown_seconds: float = 0.35
    sticky_suspicious_block: bool = False
    suspicious_hold_seconds: float = 0.12
    no_face_reset_seconds: float = 2.0
    state_display_seconds: float = 0.10


class LoggingSettings(BaseModel):
    max_events: int = 200


class RuntimeSettings(BaseModel):
    autostart: bool = True
    auto_enable_local_backends: bool = True
    require_real_backends: bool = True
    loop_sleep_seconds: float = 0.005
    metrics_window_size: int = 120
    preview_window_name: str = "Access Control Preview"
    preview_width: int = 1280
    preview_height: int = 720


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ACCESS_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    api: ApiSettings = ApiSettings()
    camera: CameraSettings = CameraSettings()
    yolo: YoloSettings = YoloSettings()
    face: FaceSettings = FaceSettings()
    anti_spoof: AntiSpoofSettings = AntiSpoofSettings()
    active_liveness: ActiveLivenessSettings = ActiveLivenessSettings()
    session: SessionSettings = SessionSettings()
    logging: LoggingSettings = LoggingSettings()
    runtime: RuntimeSettings = RuntimeSettings()


def enable_phone_detector(settings: Settings) -> Settings:
    if settings.yolo.backend != "mock" and settings.yolo.model_path:
        return settings

    for candidate in _DEFAULT_YOLO_MODEL_CANDIDATES:
        if candidate.exists():
            settings.yolo.backend = "ultralytics"
            settings.yolo.model_path = str(candidate)
            return settings
    return settings


def enable_anti_spoof_backend(settings: Settings) -> Settings:
    if settings.anti_spoof.backend != "mock" and settings.anti_spoof.model_dir:
        return settings

    for candidate in _DEFAULT_ANTI_SPOOF_MODEL_DIR_CANDIDATES:
        if candidate.is_dir() and any(candidate.glob("*.pth")):
            settings.anti_spoof.backend = "minifasnet"
            settings.anti_spoof.model_dir = str(candidate)
            return settings
    return settings


def enable_active_liveness_backend(settings: Settings) -> Settings:
    if not settings.active_liveness.enabled:
        return settings
    if settings.active_liveness.backend != "mock" and settings.active_liveness.model_path:
        return settings

    for candidate in _DEFAULT_ACTIVE_LIVENESS_MODEL_CANDIDATES:
        if candidate.is_file():
            settings.active_liveness.backend = "mediapipe"
            settings.active_liveness.model_path = str(candidate)
            return settings
    return settings


def prepare_runtime_settings(settings: Settings | None = None) -> Settings:
    runtime_settings = (settings or get_settings()).model_copy(deep=True)
    if not runtime_settings.runtime.auto_enable_local_backends:
        return runtime_settings
    runtime_settings = enable_phone_detector(runtime_settings)
    runtime_settings = enable_anti_spoof_backend(runtime_settings)
    runtime_settings = enable_active_liveness_backend(runtime_settings)
    return runtime_settings


def build_runtime_readiness(
    settings: Settings,
    *,
    runtime_running: bool,
    camera_running: bool,
    camera_error: str | None,
    yolo_backend: str,
    anti_spoof_backend: str,
    active_liveness_backend: str,
) -> dict[str, object]:
    issues: list[str] = []

    if settings.runtime.require_real_backends and yolo_backend == "mock":
        issues.append("real_yolo_backend_unavailable")
    if settings.runtime.require_real_backends and anti_spoof_backend.startswith("mock"):
        issues.append("real_anti_spoof_backend_unavailable")
    if (
        settings.runtime.require_real_backends
        and settings.active_liveness.enabled
        and settings.active_liveness.require_for_allow
        and active_liveness_backend.startswith("mock")
    ):
        issues.append("real_active_liveness_backend_unavailable")
    if runtime_running and not camera_running:
        issues.append("camera_not_streaming")
    if camera_error:
        issues.append(camera_error)

    return {
        "ready": not issues,
        "issues": issues,
        "require_real_backends": settings.runtime.require_real_backends,
    }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
