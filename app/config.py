"""Application settings for the clean anti-spoof runtime."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(slots=True)
class Settings:
    camera_index: int = 1
    device: str = "auto"
    show_fps: bool = True
    window_name: str = "MiniFAS + DeepPixBiS Hard Anti-Spoof"
    max_frame_width: int = 960
    log_level: str = "INFO"
    log_path: str = "logs/anti_spoof_events.jsonl"

    mini_fas_v2_path: str = "resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth"
    mini_fas_v1se_path: str = "resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth"
    deeppixbis_path: str = "weights/DeePixBiS.pth"
    face_landmarker_path: str = "weights/face_landmarker_v2.task"

    mini_fas_v2_weight: float = 1.35
    mini_fas_v1se_weight: float = 1.0
    deeppixbis_weight: float = 0.80
    deeppixbis_context_weight: float = 0.55
    deeppixbis_crop_scale: float = 2.7
    deeppixbis_context_crop_scale: float = 4.0
    allow_minifas_only_fallback: bool = True
    enable_tta: bool = False

    real_threshold: float = 0.61
    spoof_threshold: float = 0.68
    minifas_primary_threshold: float = 0.65
    minifas_support_threshold: float = 0.52
    minifas_family_threshold: float = 0.57
    deeppixbis_family_threshold: float = 0.45
    deeppixbis_max_spoof_threshold: float = 0.68
    deeppixbis_hard_spoof_threshold: float = 0.88
    hard_spoof_threshold: float = 0.80
    super_real_threshold: float = 0.95
    assisted_real_threshold: float = 0.63
    assisted_minifas_primary_threshold: float = 0.60
    assisted_minifas_support_threshold: float = 0.45
    assisted_minifas_family_threshold: float = 0.53
    assisted_deeppixbis_family_threshold: float = 0.68
    assisted_deeppixbis_max_spoof_threshold: float = 0.40
    assisted_temporal_live_threshold: float = 0.50
    assisted_temporal_rigid_max: float = 0.50
    dominant_primary_real_threshold: float = 0.65
    dominant_primary_minifas_threshold: float = 0.84
    dominant_primary_family_threshold: float = 0.57
    dominant_primary_deeppixbis_family_threshold: float = 0.68
    dominant_primary_deeppixbis_max_spoof_threshold: float = 0.40
    dominant_primary_temporal_live_threshold: float = 0.50
    dominant_primary_temporal_rigid_max: float = 0.50

    minifas_only_primary_threshold: float = 0.63
    minifas_only_support_threshold: float = 0.52
    minifas_only_family_threshold: float = 0.57
    minifas_only_real_threshold: float = 0.59
    minifas_only_temporal_live_threshold: float = 0.30
    minifas_only_temporal_spoof_max: float = 0.66
    minifas_only_temporal_rigid_max: float = 0.82

    temporal_window: int = 5
    temporal_patch_size: int = 64
    temporal_min_frames: int = 3
    temporal_live_threshold: float = 0.34
    temporal_spoof_threshold: float = 0.72
    temporal_rigid_spoof_threshold: float = 0.72

    face_margin_ratio: float = 0.12
    min_face_brightness: float = 50.0
    min_face_contrast: float = 18.0
    min_face_size: int = 96
    max_backlight_delta: float = 45.0
    presentation_context_scale: float = 2.15
    presentation_ring_inner_scale: float = 1.06
    presentation_ring_outer_scale: float = 1.70
    presentation_dark_value_threshold: int = 70
    presentation_low_saturation_threshold: int = 85
    presentation_glare_value_threshold: int = 245
    presentation_glare_saturation_threshold: int = 40
    screen_attack_real_max: float = 0.22
    screen_attack_soft_threshold: float = 0.42
    screen_attack_hard_threshold: float = 0.68
    screen_attack_temporal_rigid_threshold: float = 0.48
    screen_rectangle_threshold: float = 0.46
    screen_bezel_threshold: float = 0.26
    landmark_quick_live_threshold: float = 0.54
    landmark_support_live_threshold: float = 0.46
    landmark_real_rigid_max: float = 0.78
    landmark_screen_rigid_threshold: float = 0.80
    landmark_min_points: int = 10

    detection_scale_factor: float = 1.1
    detection_min_neighbors: int = 5
    detection_min_size: int = 80

    @classmethod
    def from_env(cls) -> Settings:
        _load_dotenv()
        return cls(
            camera_index=int(os.getenv("CAMERA_INDEX", "1")),
            device=os.getenv("DEVICE", "auto").strip().lower(),
            show_fps=os.getenv("SHOW_FPS", "true").strip().lower() in {"1", "true", "yes", "on"},
            window_name=os.getenv("WINDOW_NAME", "MiniFAS + DeepPixBiS Hard Anti-Spoof"),
            max_frame_width=int(os.getenv("MAX_FRAME_WIDTH", "960")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_path=os.getenv("LOG_PATH", "logs/anti_spoof_events.jsonl"),
            mini_fas_v2_path=os.getenv("MINI_FAS_V2_PATH", "resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth"),
            mini_fas_v1se_path=os.getenv("MINI_FAS_V1SE_PATH", "resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth"),
            deeppixbis_path=os.getenv("DEEPIXBIS_PATH", "weights/DeePixBiS.pth"),
            face_landmarker_path=os.getenv("FACE_LANDMARKER_PATH", "weights/face_landmarker_v2.task"),
            mini_fas_v2_weight=float(os.getenv("MINI_FAS_V2_WEIGHT", "1.35")),
            mini_fas_v1se_weight=float(os.getenv("MINI_FAS_V1SE_WEIGHT", "1.0")),
            deeppixbis_weight=float(os.getenv("DEEPIXBIS_WEIGHT", "0.80")),
            deeppixbis_context_weight=float(os.getenv("DEEPIXBIS_CONTEXT_WEIGHT", "0.55")),
            deeppixbis_crop_scale=float(os.getenv("DEEPIXBIS_CROP_SCALE", "2.7")),
            deeppixbis_context_crop_scale=float(os.getenv("DEEPIXBIS_CONTEXT_CROP_SCALE", "4.0")),
            allow_minifas_only_fallback=os.getenv("ALLOW_MINIFAS_ONLY_FALLBACK", "true").strip().lower() in {"1", "true", "yes", "on"},
            enable_tta=os.getenv("ENABLE_TTA", "false").strip().lower() in {"1", "true", "yes", "on"},
            real_threshold=float(os.getenv("REAL_THRESHOLD", "0.61")),
            spoof_threshold=float(os.getenv("SPOOF_THRESHOLD", "0.68")),
            minifas_primary_threshold=float(os.getenv("MINIFAS_PRIMARY_THRESHOLD", "0.65")),
            minifas_support_threshold=float(os.getenv("MINIFAS_SUPPORT_THRESHOLD", "0.52")),
            minifas_family_threshold=float(os.getenv("MINIFAS_FAMILY_THRESHOLD", "0.57")),
            deeppixbis_family_threshold=float(os.getenv("DEEPIXBIS_FAMILY_THRESHOLD", "0.45")),
            deeppixbis_max_spoof_threshold=float(os.getenv("DEEPIXBIS_MAX_SPOOF_THRESHOLD", "0.68")),
            deeppixbis_hard_spoof_threshold=float(os.getenv("DEEPIXBIS_HARD_SPOOF_THRESHOLD", "0.88")),
            hard_spoof_threshold=float(os.getenv("HARD_SPOOF_THRESHOLD", "0.80")),
            super_real_threshold=float(os.getenv("SUPER_REAL_THRESHOLD", "0.95")),
            assisted_real_threshold=float(os.getenv("ASSISTED_REAL_THRESHOLD", "0.63")),
            assisted_minifas_primary_threshold=float(os.getenv("ASSISTED_MINIFAS_PRIMARY_THRESHOLD", "0.60")),
            assisted_minifas_support_threshold=float(os.getenv("ASSISTED_MINIFAS_SUPPORT_THRESHOLD", "0.45")),
            assisted_minifas_family_threshold=float(os.getenv("ASSISTED_MINIFAS_FAMILY_THRESHOLD", "0.53")),
            assisted_deeppixbis_family_threshold=float(os.getenv("ASSISTED_DEEPIXBIS_FAMILY_THRESHOLD", "0.68")),
            assisted_deeppixbis_max_spoof_threshold=float(os.getenv("ASSISTED_DEEPIXBIS_MAX_SPOOF_THRESHOLD", "0.40")),
            assisted_temporal_live_threshold=float(os.getenv("ASSISTED_TEMPORAL_LIVE_THRESHOLD", "0.50")),
            assisted_temporal_rigid_max=float(os.getenv("ASSISTED_TEMPORAL_RIGID_MAX", "0.50")),
            dominant_primary_real_threshold=float(os.getenv("DOMINANT_PRIMARY_REAL_THRESHOLD", "0.65")),
            dominant_primary_minifas_threshold=float(os.getenv("DOMINANT_PRIMARY_MINIFAS_THRESHOLD", "0.84")),
            dominant_primary_family_threshold=float(os.getenv("DOMINANT_PRIMARY_FAMILY_THRESHOLD", "0.57")),
            dominant_primary_deeppixbis_family_threshold=float(os.getenv("DOMINANT_PRIMARY_DEEPIXBIS_FAMILY_THRESHOLD", "0.68")),
            dominant_primary_deeppixbis_max_spoof_threshold=float(os.getenv("DOMINANT_PRIMARY_DEEPIXBIS_MAX_SPOOF_THRESHOLD", "0.40")),
            dominant_primary_temporal_live_threshold=float(os.getenv("DOMINANT_PRIMARY_TEMPORAL_LIVE_THRESHOLD", "0.50")),
            dominant_primary_temporal_rigid_max=float(os.getenv("DOMINANT_PRIMARY_TEMPORAL_RIGID_MAX", "0.50")),
            minifas_only_primary_threshold=float(os.getenv("MINIFAS_ONLY_PRIMARY_THRESHOLD", "0.63")),
            minifas_only_support_threshold=float(os.getenv("MINIFAS_ONLY_SUPPORT_THRESHOLD", "0.52")),
            minifas_only_family_threshold=float(os.getenv("MINIFAS_ONLY_FAMILY_THRESHOLD", "0.57")),
            minifas_only_real_threshold=float(os.getenv("MINIFAS_ONLY_REAL_THRESHOLD", "0.59")),
            minifas_only_temporal_live_threshold=float(os.getenv("MINIFAS_ONLY_TEMPORAL_LIVE_THRESHOLD", "0.30")),
            minifas_only_temporal_spoof_max=float(os.getenv("MINIFAS_ONLY_TEMPORAL_SPOOF_MAX", "0.66")),
            minifas_only_temporal_rigid_max=float(os.getenv("MINIFAS_ONLY_TEMPORAL_RIGID_MAX", "0.82")),
            temporal_window=int(os.getenv("TEMPORAL_WINDOW", "5")),
            temporal_patch_size=int(os.getenv("TEMPORAL_PATCH_SIZE", "64")),
            temporal_min_frames=int(os.getenv("TEMPORAL_MIN_FRAMES", "3")),
            temporal_live_threshold=float(os.getenv("TEMPORAL_LIVE_THRESHOLD", "0.34")),
            temporal_spoof_threshold=float(os.getenv("TEMPORAL_SPOOF_THRESHOLD", "0.72")),
            temporal_rigid_spoof_threshold=float(os.getenv("TEMPORAL_RIGID_SPOOF_THRESHOLD", "0.72")),
            face_margin_ratio=float(os.getenv("FACE_MARGIN_RATIO", "0.12")),
            min_face_brightness=float(os.getenv("MIN_FACE_BRIGHTNESS", "50")),
            min_face_contrast=float(os.getenv("MIN_FACE_CONTRAST", "18")),
            min_face_size=int(os.getenv("MIN_FACE_SIZE", "96")),
            max_backlight_delta=float(os.getenv("MAX_BACKLIGHT_DELTA", "45")),
            presentation_context_scale=float(os.getenv("PRESENTATION_CONTEXT_SCALE", "2.15")),
            presentation_ring_inner_scale=float(os.getenv("PRESENTATION_RING_INNER_SCALE", "1.06")),
            presentation_ring_outer_scale=float(os.getenv("PRESENTATION_RING_OUTER_SCALE", "1.70")),
            presentation_dark_value_threshold=int(os.getenv("PRESENTATION_DARK_VALUE_THRESHOLD", "70")),
            presentation_low_saturation_threshold=int(os.getenv("PRESENTATION_LOW_SATURATION_THRESHOLD", "85")),
            presentation_glare_value_threshold=int(os.getenv("PRESENTATION_GLARE_VALUE_THRESHOLD", "245")),
            presentation_glare_saturation_threshold=int(os.getenv("PRESENTATION_GLARE_SATURATION_THRESHOLD", "40")),
            screen_attack_real_max=float(os.getenv("SCREEN_ATTACK_REAL_MAX", "0.22")),
            screen_attack_soft_threshold=float(os.getenv("SCREEN_ATTACK_SOFT_THRESHOLD", "0.42")),
            screen_attack_hard_threshold=float(os.getenv("SCREEN_ATTACK_HARD_THRESHOLD", "0.68")),
            screen_attack_temporal_rigid_threshold=float(os.getenv("SCREEN_ATTACK_TEMPORAL_RIGID_THRESHOLD", "0.48")),
            screen_rectangle_threshold=float(os.getenv("SCREEN_RECTANGLE_THRESHOLD", "0.46")),
            screen_bezel_threshold=float(os.getenv("SCREEN_BEZEL_THRESHOLD", "0.26")),
            landmark_quick_live_threshold=float(os.getenv("LANDMARK_QUICK_LIVE_THRESHOLD", "0.54")),
            landmark_support_live_threshold=float(os.getenv("LANDMARK_SUPPORT_LIVE_THRESHOLD", "0.46")),
            landmark_real_rigid_max=float(os.getenv("LANDMARK_REAL_RIGID_MAX", "0.78")),
            landmark_screen_rigid_threshold=float(os.getenv("LANDMARK_SCREEN_RIGID_THRESHOLD", "0.80")),
            landmark_min_points=int(os.getenv("LANDMARK_MIN_POINTS", "10")),
            detection_scale_factor=float(os.getenv("DETECTION_SCALE_FACTOR", "1.1")),
            detection_min_neighbors=int(os.getenv("DETECTION_MIN_NEIGHBORS", "5")),
            detection_min_size=int(os.getenv("DETECTION_MIN_SIZE", "80")),
        )

    @property
    def mini_fas_v2_weights(self) -> Path:
        return Path(self.mini_fas_v2_path).resolve()

    @property
    def mini_fas_v1se_weights(self) -> Path:
        return Path(self.mini_fas_v1se_path).resolve()

    @property
    def deeppixbis_weights(self) -> Path:
        candidates = [
            Path(self.deeppixbis_path),
            Path("weights/DeePixBiS.pth"),
            Path("weights/deeppixbis.onnx"),
        ]
        seen: set[Path] = set()
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if resolved.exists():
                return resolved
        return Path(self.deeppixbis_path).resolve()

    @property
    def face_landmarker_weights(self) -> Path:
        return Path(self.face_landmarker_path).resolve()

    @property
    def event_log_path(self) -> Path:
        return Path(self.log_path).resolve()
