"""Simple DI container with startup asset validation."""
from __future__ import annotations

from functools import cached_property

from app.adapters.analyzers.combined_analyzer import CombinedAnalyzer
from app.adapters.analyzers.deep_learning_analyzer import DeepLearningAnalyzer
from app.adapters.analyzers.heuristic_analyzer import HeuristicAnalyzer
from app.adapters.detectors.mediapipe_detector import MediaPipeDetector
from app.adapters.detectors.retinaface_detector import RetinaFaceDetector
from app.adapters.preprocessors.opencv_preprocessor import OpenCVPreprocessor
from app.adapters.repositories.filesystem_model_repo import FilesystemModelRepo
from app.domain.entities import AssetManifest
from app.domain.exceptions import AssetValidationError
from app.infrastructure.assets import validate_assets
from app.infrastructure.config import AppSettings
from app.turnstile.engine import TurnstileDecisionEngine
from app.use_cases.analyze_single_image import AnalyzeSingleImageUseCase
from app.use_cases.analyze_video_frame import AnalyzeVideoFrameUseCase


class Container:
    def __init__(self, settings: AppSettings):
        self._settings = settings
        self._asset_error: str | None = None
        self._asset_manifest: AssetManifest | None = None
        try:
            self._asset_manifest = validate_assets(settings)
        except AssetValidationError as exc:
            self._asset_error = str(exc)

    @property
    def asset_error(self) -> str | None:
        return self._asset_error

    @property
    def asset_manifest(self) -> AssetManifest:
        if self._asset_manifest is None:
            raise AssetValidationError(self._asset_error or "Assets are not ready")
        return self._asset_manifest

    @property
    def is_ready(self) -> bool:
        return self._asset_error is None

    @property
    def readiness(self) -> dict[str, object]:
        return {
            "status": "ready" if self.is_ready and self.models_loaded else "not_ready",
            "models_loaded": self.models_loaded,
            "asset_error": self._asset_error,
        }

    @cached_property
    def retinaface_detector(self) -> RetinaFaceDetector:
        self.asset_manifest
        return RetinaFaceDetector(self._settings.detector.retinaface)

    @cached_property
    def mediapipe_detector(self) -> MediaPipeDetector:
        self.asset_manifest
        return MediaPipeDetector(self._settings.detector.mediapipe, running_mode="IMAGE")

    @cached_property
    def mediapipe_video_detector(self) -> MediaPipeDetector:
        self.asset_manifest
        return MediaPipeDetector(self._settings.detector.mediapipe, running_mode="VIDEO")

    @cached_property
    def model_repo(self) -> FilesystemModelRepo:
        self.asset_manifest
        return FilesystemModelRepo(self._settings.analyzer.deep_learning.model_dir)

    @cached_property
    def dl_analyzer(self) -> DeepLearningAnalyzer:
        self.asset_manifest
        return DeepLearningAnalyzer(
            self._settings.analyzer.deep_learning,
            self.model_repo,
        )

    @cached_property
    def heuristic_analyzer(self) -> HeuristicAnalyzer:
        return HeuristicAnalyzer(self._settings.analyzer.heuristic)

    @cached_property
    def combined_analyzer(self) -> CombinedAnalyzer:
        return CombinedAnalyzer(
            self._settings.analyzer.combined,
            self.dl_analyzer,
            self.heuristic_analyzer,
        )

    @cached_property
    def preprocessor(self) -> OpenCVPreprocessor:
        return OpenCVPreprocessor()

    @cached_property
    def analyze_single_image(self) -> AnalyzeSingleImageUseCase:
        return AnalyzeSingleImageUseCase(
            detector=self.retinaface_detector,
            analyzer=self.dl_analyzer,
        )

    @cached_property
    def analyze_video_frame(self) -> AnalyzeVideoFrameUseCase:
        return AnalyzeVideoFrameUseCase(
            detector=self.mediapipe_detector,
            analyzer=self.heuristic_analyzer,
        )

    @cached_property
    def turnstile_engine(self) -> TurnstileDecisionEngine:
        self.asset_manifest
        return TurnstileDecisionEngine(
            camera_config=self._settings.camera,
            turnstile_config=self._settings.turnstile,
            detector=self.mediapipe_video_detector,
            dl_analyzer=self.dl_analyzer,
            assets=self.asset_manifest,
        )

    @property
    def models_loaded(self) -> bool:
        if not self.is_ready:
            return False
        try:
            return len(self.dl_analyzer.model_names) > 0
        except Exception:
            return False
