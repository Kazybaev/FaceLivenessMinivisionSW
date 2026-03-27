from __future__ import annotations

from pathlib import Path

from app.domain.entities import AssetManifest
from app.domain.exceptions import AssetValidationError
from app.infrastructure.config import AppSettings


def _project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "config" / "default.yaml").exists():
            return parent
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return _project_root() / path


def validate_assets(settings: AppSettings) -> AssetManifest:
    mediapipe_model = resolve_path(settings.detector.mediapipe.model_path)
    if not mediapipe_model.is_file():
        raise AssetValidationError(
            f"Missing MediaPipe model file: {mediapipe_model}"
        )

    retinaface_prototxt = resolve_path(settings.detector.retinaface.prototxt)
    if not retinaface_prototxt.is_file():
        raise AssetValidationError(
            f"Missing RetinaFace prototxt: {retinaface_prototxt}"
        )

    retinaface_caffemodel = resolve_path(settings.detector.retinaface.caffemodel)
    if not retinaface_caffemodel.is_file():
        raise AssetValidationError(
            f"Missing RetinaFace caffemodel: {retinaface_caffemodel}"
        )

    anti_spoof_dir = resolve_path(settings.analyzer.deep_learning.model_dir)
    if not anti_spoof_dir.is_dir():
        raise AssetValidationError(
            f"Missing anti-spoof model directory: {anti_spoof_dir}"
        )

    anti_spoof_models = sorted(p.name for p in anti_spoof_dir.glob("*.pth"))
    if not anti_spoof_models:
        raise AssetValidationError(
            f"No .pth anti-spoof models found in: {anti_spoof_dir}"
        )

    return AssetManifest(
        mediapipe_model_path=str(mediapipe_model),
        retinaface_prototxt=str(retinaface_prototxt),
        retinaface_caffemodel=str(retinaface_caffemodel),
        anti_spoof_model_dir=str(anti_spoof_dir),
        anti_spoof_model_names=anti_spoof_models,
    )
