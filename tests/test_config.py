from pathlib import Path

from app.config import Settings


def test_settings_from_env_file(tmp_path: Path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        (
            "DEEPIXBIS_PATH=weights\\deeppixbis.onnx\n"
            "REAL_THRESHOLD=0.81\n"
            "TEMPORAL_WINDOW=7\n"
            "ENABLE_TTA=false\n"
            "ALLOW_MINIFAS_ONLY_FALLBACK=false\n"
            "MINIFAS_ONLY_REAL_THRESHOLD=0.64\n"
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    settings = Settings.from_env()
    assert settings.deeppixbis_path.endswith("deeppixbis.onnx")
    assert settings.real_threshold == 0.81
    assert settings.temporal_window == 7
    assert settings.enable_tta is False
    assert settings.allow_minifas_only_fallback is False
    assert settings.minifas_only_real_threshold == 0.64


def test_settings_auto_resolve_existing_deeppixbis_pth(tmp_path: Path, monkeypatch) -> None:
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    (weights_dir / "DeePixBiS.pth").write_bytes(b"stub")
    monkeypatch.delenv("DEEPIXBIS_PATH", raising=False)
    monkeypatch.chdir(tmp_path)

    settings = Settings.from_env()

    assert settings.deeppixbis_path.endswith("DeePixBiS.pth")
    assert settings.deeppixbis_weights.name == "DeePixBiS.pth"
