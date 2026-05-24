from __future__ import annotations

import importlib
import os
from pathlib import Path


def test_data_dir_moves_default_runtime_dirs(monkeypatch, tmp_path: Path):
    import app.core.config as config

    original = config
    data_dir = tmp_path / "portable-data"
    monkeypatch.setenv("STEMDECK_DATA_DIR", str(data_dir))
    monkeypatch.delenv("STEMDECK_JOBS_DIR", raising=False)
    monkeypatch.delenv("STEMDECK_CACHE_DIR", raising=False)
    monkeypatch.delenv("STEMDECK_DOWNLOADS_DIR", raising=False)
    monkeypatch.delenv("STEMDECK_MODELS_DIR", raising=False)
    monkeypatch.delenv("STEMDECK_LOGS_DIR", raising=False)
    try:
        reloaded = importlib.reload(config)
        assert data_dir.resolve() == reloaded.DATA_DIR
        assert data_dir.resolve() / "jobs" == reloaded.JOBS_DIR
        assert data_dir.resolve() / "cache" == reloaded.CACHE_DIR
        assert data_dir.resolve() / "downloads" == reloaded.DOWNLOADS_DIR
        assert data_dir.resolve() / "models" == reloaded.MODELS_DIR
        assert data_dir.resolve() / "logs" == reloaded.LOGS_DIR
    finally:
        monkeypatch.delenv("STEMDECK_DATA_DIR", raising=False)
        importlib.reload(original)


def test_jobs_dir_override_wins_over_data_dir(monkeypatch, tmp_path: Path):
    import app.core.config as config

    original = config
    data_dir = tmp_path / "portable-data"
    jobs_dir = tmp_path / "custom-jobs"
    monkeypatch.setenv("STEMDECK_DATA_DIR", str(data_dir))
    monkeypatch.setenv("STEMDECK_JOBS_DIR", str(jobs_dir))
    try:
        reloaded = importlib.reload(config)
        assert data_dir.resolve() == reloaded.DATA_DIR
        assert jobs_dir.resolve() == reloaded.JOBS_DIR
    finally:
        monkeypatch.delenv("STEMDECK_DATA_DIR", raising=False)
        monkeypatch.delenv("STEMDECK_JOBS_DIR", raising=False)
        importlib.reload(original)


def test_ffmpeg_executable_prefers_portable_binary(monkeypatch, tmp_path: Path):
    import app.core.config as config

    original = config
    ffmpeg = tmp_path / "ffmpeg" / "ffmpeg"
    ffmpeg.parent.mkdir()
    ffmpeg.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("STEMDECK_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("STEMDECK_FFMPEG", str(ffmpeg))
    try:
        reloaded = importlib.reload(config)
        assert reloaded.ffmpeg_executable() == str(ffmpeg.resolve())
    finally:
        monkeypatch.delenv("STEMDECK_DATA_DIR", raising=False)
        monkeypatch.delenv("STEMDECK_FFMPEG", raising=False)
        importlib.reload(original)


def test_configure_portable_environment_leaves_dev_cache_env_alone(monkeypatch):
    import app.core.config as config

    original = config
    monkeypatch.delenv("STEMDECK_DATA_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.delenv("TORCH_HOME", raising=False)
    try:
        reloaded = importlib.reload(config)
        reloaded.configure_portable_environment()
        assert "XDG_CACHE_HOME" not in os.environ
        assert "TORCH_HOME" not in os.environ
    finally:
        monkeypatch.delenv("STEMDECK_DATA_DIR", raising=False)
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        monkeypatch.delenv("TORCH_HOME", raising=False)
        importlib.reload(original)


def test_processing_mode_normalization_and_model_resolve(monkeypatch):
    import app.core.config as config

    original = config
    monkeypatch.setenv("STEMDECK_DEFAULT_MODE", "hq")
    monkeypatch.setenv("STEMDECK_DEMUCS_MODEL", "htdemucs_6s")
    monkeypatch.setenv("STEMDECK_DEMUCS_MODEL_FAST", "htdemucs")
    monkeypatch.setenv("STEMDECK_DEMUCS_MODEL_HQ", "htdemucs_ft")
    monkeypatch.setenv("STEMDECK_DEMUCS_MODEL_HQ_ENHANCED", "htdemucs_6s")
    try:
        reloaded = importlib.reload(config)
        assert reloaded.normalize_processing_mode("fast") == "fast"
        assert reloaded.normalize_processing_mode("HQ_ENHANCED") == "hq_enhanced"
        assert reloaded.normalize_processing_mode("unknown") == "hq"
        assert reloaded.resolve_demucs_model("fast") == "htdemucs"
        assert reloaded.resolve_demucs_model("hq") == "htdemucs_ft"
        assert reloaded.resolve_demucs_model("hq_enhanced") == "htdemucs_6s"
        assert reloaded.resolve_demucs_model("bad-mode") == "htdemucs_ft"
    finally:
        monkeypatch.delenv("STEMDECK_DEFAULT_MODE", raising=False)
        monkeypatch.delenv("STEMDECK_DEMUCS_MODEL", raising=False)
        monkeypatch.delenv("STEMDECK_DEMUCS_MODEL_FAST", raising=False)
        monkeypatch.delenv("STEMDECK_DEMUCS_MODEL_HQ", raising=False)
        monkeypatch.delenv("STEMDECK_DEMUCS_MODEL_HQ_ENHANCED", raising=False)
        importlib.reload(original)
