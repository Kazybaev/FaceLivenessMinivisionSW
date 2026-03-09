"""Simple DI container using @cached_property — models loaded once."""
from __future__ import annotations

from functools import cached_property
from pathlib import Path

from app.infrastructure.config import AppSettings
from app.adapters.detectors.retinaface_detector import RetinaFaceDetector
from app.adapters.detectors.mediapipe_detector import MediaPipeDetector
from app.adapters.analyzers.deep_learning_analyzer import DeepLearningAnalyzer
from app.adapters.analyzers.heuristic_analyzer import HeuristicAnalyzer
from app.adapters.analyzers.combined_analyzer import CombinedAnalyzer
from app.adapters.preprocessors.opencv_preprocessor import OpenCVPreprocessor
from app.adapters.repositories.filesystem_model_repo import FilesystemModelRepo
from app.use_cases.analyze_single_image import AnalyzeSingleImageUseCase
from app.use_cases.analyze_video_frame import AnalyzeVideoFrameUseCase


class Container:
    def __init__(self, settings: AppSettings):
        self._settings = settings

    @cached_property
    def retinaface_detector(self) -> RetinaFaceDetector:
        return RetinaFaceDetector(self._settings.detector.retinaface)

    @cached_property
    def mediapipe_detector(self) -> MediaPipeDetector:
        return MediaPipeDetector(self._settings.detector.mediapipe)

    @cached_property
    def model_repo(self) -> FilesystemModelRepo:
        return FilesystemModelRepo(self._settings.analyzer.deep_learning.model_dir)

    @cached_property
    def dl_analyzer(self) -> DeepLearningAnalyzer:
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

    @property
    def models_loaded(self) -> bool:
        try:
            return len(self.dl_analyzer._models) > 0
        except Exception:
            return False
