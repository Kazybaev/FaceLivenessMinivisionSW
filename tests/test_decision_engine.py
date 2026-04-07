from app.config import Settings
from app.core.decision_engine import DecisionEngine
from app.core.enums import AntiSpoofLabel
from app.core.schemas import AntiSpoofInference, LandmarkEvidence, ModelScore, PresentationAttackEvidence, TemporalEvidence


def _score(model_name: str, real: float, spoof: float, weight: float = 1.0) -> ModelScore:
    return ModelScore(
        model_name=model_name,
        real_score=real,
        spoof_score=spoof,
        confidence=max(real, spoof),
        weight=weight,
    )


def _temporal(live: float, spoof: float, frame_count: int = 5, rigid: float | None = None) -> TemporalEvidence:
    return TemporalEvidence(
        live_score=live,
        spoof_score=spoof,
        motion_score=live,
        diversity_score=live,
        rigidity_score=spoof if rigid is None else rigid,
        frame_count=frame_count,
        detail="test",
    )


def _presentation(attack: float, bezel: float = 0.0, rect: float = 0.0, glare: float = 0.0) -> PresentationAttackEvidence:
    return PresentationAttackEvidence(
        attack_score=attack,
        bezel_score=bezel,
        rectangle_score=rect,
        glare_score=glare,
        detail="screen-test",
    )


def _landmarks(
    live: float,
    rigid: float,
    motion: float = 0.6,
    depth: float = 0.6,
    frame_count: int = 3,
    point_count: int = 13,
) -> LandmarkEvidence:
    return LandmarkEvidence(
        live_score=live,
        spoof_score=1.0 - live,
        motion_score=motion,
        depth_score=depth,
        rigidity_score=rigid,
        frame_count=frame_count,
        point_count=point_count,
        detail="landmark-test",
    )


def test_decision_engine_returns_real_for_strong_live_evidence() -> None:
    engine = DecisionEngine(Settings())
    decision = engine.decide(
        AntiSpoofInference(
            real_score=0.90,
            spoof_score=0.10,
            inference_time_ms=10.0,
            raw_scores=None,
            window_size=5,
            model_scores=[
                _score("MiniFASNetV2", 0.93, 0.07, 1.35),
                _score("MiniFASNetV1SE", 0.86, 0.14, 1.0),
                _score("DeepPixBiS", 0.70, 0.30, 0.8),
                _score("DeepPixBiSContext", 0.58, 0.42, 0.55),
            ],
            temporal=_temporal(0.48, 0.52, rigid=0.40),
        )
    )
    assert decision.label == AntiSpoofLabel.REAL


def test_decision_engine_can_use_deeppix_and_temporal_to_rescue_borderline_minifas() -> None:
    engine = DecisionEngine(Settings())
    decision = engine.decide(
        AntiSpoofInference(
            real_score=0.70,
            spoof_score=0.30,
            inference_time_ms=10.0,
            raw_scores=None,
            window_size=5,
            model_scores=[
                _score("MiniFASNetV2", 0.67, 0.33, 1.35),
                _score("MiniFASNetV1SE", 0.58, 0.42, 1.0),
                _score("DeepPixBiS", 0.78, 0.22, 0.8),
                _score("DeepPixBiSContext", 0.70, 0.30, 0.55),
            ],
            temporal=_temporal(0.74, 0.26, rigid=0.18),
        )
    )
    assert decision.label == AntiSpoofLabel.REAL


def test_decision_engine_can_pass_when_minifas_v2_is_strong_but_support_model_is_weak() -> None:
    engine = DecisionEngine(Settings())
    decision = engine.decide(
        AntiSpoofInference(
            real_score=0.71,
            spoof_score=0.29,
            inference_time_ms=10.0,
            raw_scores=None,
            window_size=5,
            model_scores=[
                _score("MiniFASNetV2", 0.95, 0.05, 1.35),
                _score("MiniFASNetV1SE", 0.31, 0.69, 1.0),
                _score("DeepPixBiS", 0.82, 0.18, 0.8),
                _score("DeepPixBiSContext", 0.77, 0.23, 0.55),
            ],
            temporal=_temporal(0.74, 0.26, rigid=0.44),
            presentation=_presentation(0.18, bezel=0.10, rect=0.05),
            landmarks=_landmarks(0.64, 0.42),
        )
    )
    assert decision.label == AntiSpoofLabel.REAL


def test_decision_engine_rejects_strong_screen_attack_even_with_live_scores() -> None:
    engine = DecisionEngine(Settings())
    decision = engine.decide(
        AntiSpoofInference(
            real_score=0.80,
            spoof_score=0.20,
            inference_time_ms=10.0,
            raw_scores=None,
            window_size=5,
            model_scores=[
                _score("MiniFASNetV2", 0.93, 0.07, 1.35),
                _score("MiniFASNetV1SE", 0.76, 0.24, 1.0),
                _score("DeepPixBiS", 0.80, 0.20, 0.8),
                _score("DeepPixBiSContext", 0.74, 0.26, 0.55),
            ],
            temporal=_temporal(0.70, 0.30, rigid=0.60),
            presentation=_presentation(0.88, bezel=0.74, rect=0.92, glare=0.35),
            landmarks=_landmarks(0.30, 0.88),
        )
    )
    assert decision.label == AntiSpoofLabel.FAKE


def test_decision_engine_can_use_points_for_fast_real_pass_while_temporal_warms_up() -> None:
    engine = DecisionEngine(Settings())
    decision = engine.decide(
        AntiSpoofInference(
            real_score=0.73,
            spoof_score=0.27,
            inference_time_ms=10.0,
            raw_scores=None,
            window_size=1,
            model_scores=[
                _score("MiniFASNetV2", 0.94, 0.06, 1.35),
                _score("MiniFASNetV1SE", 0.33, 0.67, 1.0),
                _score("DeepPixBiS", 0.83, 0.17, 0.8),
                _score("DeepPixBiSContext", 0.76, 0.24, 0.55),
            ],
            temporal=_temporal(0.50, 0.50, frame_count=1, rigid=0.50),
            presentation=_presentation(0.12, bezel=0.05, rect=0.04),
            landmarks=_landmarks(0.66, 0.40, frame_count=1),
        )
    )
    assert decision.label == AntiSpoofLabel.REAL


def test_decision_engine_returns_fake_for_hard_spoof() -> None:
    engine = DecisionEngine(Settings())
    decision = engine.decide(
        AntiSpoofInference(
            real_score=0.15,
            spoof_score=0.85,
            inference_time_ms=10.0,
            raw_scores=None,
            window_size=5,
            model_scores=[
                _score("MiniFASNetV2", 0.05, 0.95, 1.35),
                _score("MiniFASNetV1SE", 0.10, 0.90, 1.0),
                _score("DeepPixBiS", 0.12, 0.88, 0.8),
                _score("DeepPixBiSContext", 0.18, 0.82, 0.55),
            ],
            temporal=_temporal(0.10, 0.90, rigid=0.90),
        )
    )
    assert decision.label == AntiSpoofLabel.FAKE


def test_decision_engine_stays_fake_when_deeppixbis_is_suspicious() -> None:
    engine = DecisionEngine(Settings())
    decision = engine.decide(
        AntiSpoofInference(
            real_score=0.76,
            spoof_score=0.24,
            inference_time_ms=10.0,
            raw_scores=None,
            window_size=5,
            model_scores=[
                _score("MiniFASNetV2", 0.89, 0.11, 1.35),
                _score("MiniFASNetV1SE", 0.80, 0.20, 1.0),
                _score("DeepPixBiS", 0.41, 0.59, 0.8),
                _score("DeepPixBiSContext", 0.33, 0.67, 0.55),
            ],
            temporal=_temporal(0.44, 0.56, rigid=0.50),
        )
    )
    assert decision.label == AntiSpoofLabel.FAKE


def test_decision_engine_returns_real_in_minifas_only_fallback_when_temporal_is_strong() -> None:
    engine = DecisionEngine(Settings())
    decision = engine.decide(
        AntiSpoofInference(
            real_score=0.88,
            spoof_score=0.12,
            inference_time_ms=10.0,
            raw_scores=None,
            window_size=5,
            model_scores=[
                _score("MiniFASNetV2", 0.91, 0.09, 1.35),
                _score("MiniFASNetV1SE", 0.84, 0.16, 1.0),
            ],
            temporal=_temporal(0.58, 0.42, rigid=0.40),
        )
    )
    assert decision.label == AntiSpoofLabel.REAL


def test_decision_engine_keeps_fake_in_minifas_only_fallback_without_temporal_confirmation() -> None:
    engine = DecisionEngine(Settings())
    decision = engine.decide(
        AntiSpoofInference(
            real_score=0.90,
            spoof_score=0.10,
            inference_time_ms=10.0,
            raw_scores=None,
            window_size=2,
            model_scores=[
                _score("MiniFASNetV2", 0.94, 0.06, 1.35),
                _score("MiniFASNetV1SE", 0.86, 0.14, 1.0),
            ],
            temporal=_temporal(0.60, 0.40, frame_count=2, rigid=0.40),
        )
    )
    assert decision.label == AntiSpoofLabel.FAKE


def test_decision_engine_allows_borderline_real_in_minifas_only_fallback() -> None:
    engine = DecisionEngine(Settings())
    decision = engine.decide(
        AntiSpoofInference(
            real_score=0.67,
            spoof_score=0.33,
            inference_time_ms=10.0,
            raw_scores=None,
            window_size=5,
            model_scores=[
                _score("MiniFASNetV2", 0.72, 0.28, 1.35),
                _score("MiniFASNetV1SE", 0.60, 0.40, 1.0),
            ],
            temporal=_temporal(0.55, 0.45, rigid=0.70),
        )
    )
    assert decision.label == AntiSpoofLabel.REAL
