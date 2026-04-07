"""Strict fail-closed decision engine."""
from __future__ import annotations

from app.config import Settings
from app.core.enums import AntiSpoofLabel
from app.core.schemas import AntiSpoofInference, Decision, ModelScore


class DecisionEngine:
    def __init__(self, settings: Settings) -> None:
        self._real_threshold = settings.real_threshold
        self._spoof_threshold = settings.spoof_threshold
        self._minifas_primary_threshold = settings.minifas_primary_threshold
        self._minifas_support_threshold = settings.minifas_support_threshold
        self._minifas_family_threshold = settings.minifas_family_threshold
        self._deeppixbis_family_threshold = settings.deeppixbis_family_threshold
        self._deeppixbis_max_spoof_threshold = settings.deeppixbis_max_spoof_threshold
        self._deeppixbis_hard_spoof_threshold = settings.deeppixbis_hard_spoof_threshold
        self._hard_spoof_threshold = settings.hard_spoof_threshold
        self._super_real_threshold = settings.super_real_threshold
        self._assisted_real_threshold = settings.assisted_real_threshold
        self._assisted_minifas_primary_threshold = settings.assisted_minifas_primary_threshold
        self._assisted_minifas_support_threshold = settings.assisted_minifas_support_threshold
        self._assisted_minifas_family_threshold = settings.assisted_minifas_family_threshold
        self._assisted_deeppixbis_family_threshold = settings.assisted_deeppixbis_family_threshold
        self._assisted_deeppixbis_max_spoof_threshold = settings.assisted_deeppixbis_max_spoof_threshold
        self._assisted_temporal_live_threshold = settings.assisted_temporal_live_threshold
        self._assisted_temporal_rigid_max = settings.assisted_temporal_rigid_max
        self._dominant_primary_real_threshold = settings.dominant_primary_real_threshold
        self._dominant_primary_minifas_threshold = settings.dominant_primary_minifas_threshold
        self._dominant_primary_family_threshold = settings.dominant_primary_family_threshold
        self._dominant_primary_deeppixbis_family_threshold = settings.dominant_primary_deeppixbis_family_threshold
        self._dominant_primary_deeppixbis_max_spoof_threshold = settings.dominant_primary_deeppixbis_max_spoof_threshold
        self._dominant_primary_temporal_live_threshold = settings.dominant_primary_temporal_live_threshold
        self._dominant_primary_temporal_rigid_max = settings.dominant_primary_temporal_rigid_max
        self._screen_attack_real_max = settings.screen_attack_real_max
        self._screen_attack_soft_threshold = settings.screen_attack_soft_threshold
        self._screen_attack_hard_threshold = settings.screen_attack_hard_threshold
        self._screen_attack_temporal_rigid_threshold = settings.screen_attack_temporal_rigid_threshold
        self._screen_rectangle_threshold = settings.screen_rectangle_threshold
        self._screen_bezel_threshold = settings.screen_bezel_threshold
        self._landmark_quick_live_threshold = settings.landmark_quick_live_threshold
        self._landmark_support_live_threshold = settings.landmark_support_live_threshold
        self._landmark_real_rigid_max = settings.landmark_real_rigid_max
        self._landmark_screen_rigid_threshold = settings.landmark_screen_rigid_threshold
        self._landmark_min_points = settings.landmark_min_points
        self._minifas_only_primary_threshold = settings.minifas_only_primary_threshold
        self._minifas_only_support_threshold = settings.minifas_only_support_threshold
        self._minifas_only_family_threshold = settings.minifas_only_family_threshold
        self._minifas_only_real_threshold = settings.minifas_only_real_threshold
        self._minifas_only_temporal_live_threshold = settings.minifas_only_temporal_live_threshold
        self._minifas_only_temporal_spoof_max = settings.minifas_only_temporal_spoof_max
        self._minifas_only_temporal_rigid_max = settings.minifas_only_temporal_rigid_max
        self._temporal_min_frames = settings.temporal_min_frames
        self._temporal_live_threshold = settings.temporal_live_threshold
        self._temporal_spoof_threshold = settings.temporal_spoof_threshold
        self._temporal_rigid_spoof_threshold = settings.temporal_rigid_spoof_threshold

    def no_face(self) -> Decision:
        return Decision(
            label=AntiSpoofLabel.NO_FACE,
            confidence=0.0,
            real_score=0.0,
            spoof_score=0.0,
            reason="No face detected.",
        )

    def decide(self, inference: AntiSpoofInference) -> Decision:
        real_score = float(inference.real_score)
        spoof_score = float(inference.spoof_score)
        confidence = max(real_score, spoof_score)

        mini_primary = self._find_model(inference.model_scores, "MiniFASNetV2")
        mini_support = self._find_model(inference.model_scores, "MiniFASNetV1SE")
        mini_family = [score for score in inference.model_scores if "MiniFAS" in score.model_name]
        deeppix_family = [score for score in inference.model_scores if "DeepPixBiS" in score.model_name]
        deeppix_available = bool(deeppix_family)

        mini_family_real, mini_family_spoof = self._family_average(mini_family)
        deeppix_family_real, deeppix_family_spoof = self._family_average(deeppix_family)
        deeppix_max_spoof = max((score.spoof_score for score in deeppix_family), default=0.0)

        temporal = inference.temporal
        temporal_ready = temporal is not None and temporal.frame_count >= self._temporal_min_frames
        temporal_live = 0.5 if temporal is None else float(temporal.live_score)
        temporal_spoof = 0.5 if temporal is None else float(temporal.spoof_score)
        temporal_rigid = 0.5 if temporal is None else float(temporal.rigidity_score)
        presentation = inference.presentation
        screen_attack = 0.0 if presentation is None else float(presentation.attack_score)
        screen_rectangle = 0.0 if presentation is None else float(presentation.rectangle_score)
        screen_bezel = 0.0 if presentation is None else float(presentation.bezel_score)
        landmarks = inference.landmarks
        landmark_available = landmarks is not None and landmarks.point_count >= self._landmark_min_points
        landmark_live = 0.5 if not landmark_available else float(landmarks.live_score)
        landmark_rigid = 0.5 if not landmark_available else float(landmarks.rigidity_score)
        landmark_quick_live = (
            landmark_available
            and landmark_live >= self._landmark_quick_live_threshold
            and landmark_rigid < self._landmark_real_rigid_max
        )
        landmark_support_live = (
            landmark_available
            and landmark_live >= self._landmark_support_live_threshold
            and landmark_rigid < self._landmark_real_rigid_max
        )

        temporal_spoof_veto = (
            temporal_ready
            and temporal_spoof >= self._temporal_spoof_threshold
            and temporal_rigid >= self._temporal_rigid_spoof_threshold
            and temporal.motion_score >= 0.18
        )
        screen_temporal_veto = (
            screen_attack >= self._screen_attack_soft_threshold
            and temporal_ready
            and temporal_rigid >= self._screen_attack_temporal_rigid_threshold
        )
        screen_shape_veto = (
            screen_attack >= self._screen_attack_soft_threshold
            and screen_rectangle >= self._screen_rectangle_threshold
            and screen_bezel >= self._screen_bezel_threshold
        )
        screen_landmark_veto = (
            screen_attack >= self._screen_attack_soft_threshold
            and landmark_available
            and landmark_rigid >= self._landmark_screen_rigid_threshold
            and landmark_live < self._landmark_support_live_threshold
        )

        hard_spoof = (
            mini_primary.spoof_score >= self._hard_spoof_threshold
            or (
                mini_primary.spoof_score >= self._spoof_threshold
                and mini_support.spoof_score >= self._spoof_threshold
            )
            or deeppix_max_spoof >= self._deeppixbis_hard_spoof_threshold
            or temporal_spoof_veto
            or screen_attack >= self._screen_attack_hard_threshold
            or screen_temporal_veto
            or screen_shape_veto
            or screen_landmark_veto
        )

        super_real = (
            mini_primary.real_score >= self._super_real_threshold
            and mini_support.real_score >= 0.88
            and deeppix_available
            and deeppix_family_real >= 0.60
            and deeppix_max_spoof < 0.42
            and (not temporal_ready or temporal_live >= 0.42)
        )

        if deeppix_available:
            primary_real = (
                mini_primary.real_score >= self._minifas_primary_threshold
                and mini_support.real_score >= self._minifas_support_threshold
                and mini_family_real >= self._minifas_family_threshold
                and deeppix_family_real >= self._deeppixbis_family_threshold
                and deeppix_max_spoof < self._deeppixbis_max_spoof_threshold
                and screen_attack < self._screen_attack_real_max
                and real_score >= self._real_threshold
                and (
                    not temporal_ready
                    or temporal_live >= self._temporal_live_threshold
                    or super_real
                )
            )
            assisted_real = (
                temporal_ready
                and mini_primary.real_score >= self._assisted_minifas_primary_threshold
                and mini_support.real_score >= self._assisted_minifas_support_threshold
                and mini_family_real >= self._assisted_minifas_family_threshold
                and deeppix_family_real >= self._assisted_deeppixbis_family_threshold
                and deeppix_max_spoof < self._assisted_deeppixbis_max_spoof_threshold
                and screen_attack < self._screen_attack_real_max
                and real_score >= self._assisted_real_threshold
                and (
                    (
                        temporal_ready
                        and temporal_live >= self._assisted_temporal_live_threshold
                        and temporal_rigid < self._assisted_temporal_rigid_max
                    )
                    or landmark_quick_live
                )
            )
            dominant_primary_real = (
                mini_primary.real_score >= self._dominant_primary_minifas_threshold
                and mini_family_real >= self._dominant_primary_family_threshold
                and deeppix_family_real >= self._dominant_primary_deeppixbis_family_threshold
                and deeppix_max_spoof < self._dominant_primary_deeppixbis_max_spoof_threshold
                and screen_attack < self._screen_attack_real_max
                and real_score >= self._dominant_primary_real_threshold
                and (
                    landmark_quick_live
                    or (
                        temporal_ready
                        and temporal_live >= self._dominant_primary_temporal_live_threshold
                        and temporal_rigid < self._dominant_primary_temporal_rigid_max
                    )
                    or landmark_support_live
                )
            )
            strong_real = primary_real or assisted_real or dominant_primary_real
        else:
            strong_real = (
                mini_primary.real_score >= self._minifas_only_primary_threshold
                and mini_support.real_score >= self._minifas_only_support_threshold
                and mini_family_real >= self._minifas_only_family_threshold
                and screen_attack < self._screen_attack_real_max
                and real_score >= self._minifas_only_real_threshold
                and temporal_ready
                and temporal_live >= self._minifas_only_temporal_live_threshold
                and temporal_spoof < self._minifas_only_temporal_spoof_max
                and temporal_rigid < self._minifas_only_temporal_rigid_max
            )

        if strong_real and not hard_spoof:
            return Decision(
                label=AntiSpoofLabel.REAL,
                confidence=confidence,
                real_score=real_score,
                spoof_score=spoof_score,
                reason=(
                    "MiniFAS main gate is strongly live, DeepPixBiS stays below spoof veto, "
                    if deeppix_available and not assisted_real and not dominant_primary_real
                    else (
                        "MiniFASNetV2 dominated the decision and overruled a weak support model because DeepPixBiS and temporal evidence stayed strongly live, "
                        if deeppix_available and dominant_primary_real
                        else (
                        "DeepPixBiS and temporal evidence rescued a borderline MiniFAS result, while spoof vetoes stayed quiet, "
                        if deeppix_available
                        else "Running in MiniFAS-only strict fallback mode and live evidence is still strong enough, "
                        )
                    )
                    + (
                        f"temporal {temporal.detail}."
                        if temporal_ready and temporal is not None
                        else "temporal evidence is warming up."
                    )
                    + (
                        f" Points {landmarks.detail}."
                        if landmark_available and landmarks is not None
                        else ""
                    )
                    + (
                        f" Screen {presentation.detail}."
                        if presentation is not None
                        else ""
                    )
                ),
            )

        if hard_spoof:
            return Decision(
                label=AntiSpoofLabel.FAKE,
                confidence=confidence,
                real_score=real_score,
                spoof_score=spoof_score,
                reason=(
                    "Hard spoof veto triggered by MiniFAS, DeepPixBiS, temporal rigid presentation cues, or strong screen/phone suspicion."
                    + (f" Points {landmarks.detail}." if landmark_available and landmarks is not None else "")
                    + (f" Screen {presentation.detail}." if presentation is not None else "")
                ),
            )

        return Decision(
            label=AntiSpoofLabel.FAKE,
            confidence=confidence,
            real_score=real_score,
            spoof_score=spoof_score,
            reason=(
                "Live evidence is not strong enough for strict access control. "
                f"MiniFAS family real={mini_family_real:.2f}, spoof={mini_family_spoof:.2f}; "
                + (
                    f"DeepPixBiS family real={deeppix_family_real:.2f}, spoof={deeppix_family_spoof:.2f}; "
                    if deeppix_available
                    else "DeepPixBiS is not loaded, so MiniFAS-only fallback requires stronger temporal confirmation; "
                )
                + (
                    f"temporal {temporal.detail}."
                    if temporal is not None
                    else "temporal warming up."
                )
                + (
                    f" Points {landmarks.detail}."
                    if landmark_available and landmarks is not None
                    else ""
                )
                + (
                    f" Screen {presentation.detail}."
                    if presentation is not None
                    else ""
                )
            ),
        )

    @staticmethod
    def _find_model(scores: list[ModelScore], name: str) -> ModelScore:
        for score in scores:
            if score.model_name == name:
                return score
        raise ValueError(f"Required model score is missing: {name}")

    @staticmethod
    def _family_average(scores: list[ModelScore]) -> tuple[float, float]:
        if not scores:
            return 0.0, 0.0
        total_weight = sum(score.weight for score in scores) or float(len(scores))
        real = sum(score.real_score * score.weight for score in scores) / total_weight
        spoof = sum(score.spoof_score * score.weight for score in scores) / total_weight
        return float(real), float(spoof)
