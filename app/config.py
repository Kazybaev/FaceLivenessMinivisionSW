from __future__ import annotations

from functools import lru_cache

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    frame_buffer_size: int = 4
    min_frames_for_inference: int = 3
    real_confidence_threshold: float = 0.75
    crop_size: int = 96
    min_texture_for_real: float = 18.0
    min_motion_for_real: float = 1.2
    max_motion_for_real: float = 24.0
    max_texture_for_spoof: float = 9.0
    max_motion_for_spoof: float = 0.8
    uncertain_confidence: float = 0.55


class SessionSettings(BaseModel):
    session_timeout_seconds: float = 10.0
    suspicious_cooldown_seconds: float = 0.0
    spoof_cooldown_seconds: float = 1.0
    sticky_suspicious_block: bool = False
    suspicious_hold_seconds: float = 0.12
    no_face_reset_seconds: float = 2.0
    state_display_seconds: float = 0.15


class LoggingSettings(BaseModel):
    max_events: int = 200


class RuntimeSettings(BaseModel):
    autostart: bool = True
    loop_sleep_seconds: float = 0.005
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
    session: SessionSettings = SessionSettings()
    logging: LoggingSettings = LoggingSettings()
    runtime: RuntimeSettings = RuntimeSettings()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
