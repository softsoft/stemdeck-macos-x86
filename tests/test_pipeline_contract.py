from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from app.core.models import Job
from app.pipeline.runner import run_local_pipeline


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["fast", "hq", "hq_enhanced"])
async def test_local_pipeline_mode_contract_and_output_schema(tmp_path: Path, mode: str):
    job = Job(id=f"mode{mode.replace('_', '')[:8]}", mode=mode, selected_stems=["vocals", "drums"])
    job_dir = tmp_path / job.id
    job_dir.mkdir(parents=True, exist_ok=True)
    source = job_dir / "source.wav"
    source.write_bytes(b"RIFF")

    def fake_analyze(job_obj: Job, _source: Path) -> None:
        job_obj.title = "Smoke Track"
        job_obj.bpm = 120
        job_obj.key = "Am"
        job_obj.scale = "Natural Minor"
        job_obj.key_confidence = 87

    def fake_collect(_job_obj: Job, _stems_root: Path, _job_dir: Path) -> list[str]:
        stems_dir = _job_dir / "stems"
        stems_dir.mkdir(parents=True, exist_ok=True)
        (stems_dir / "vocals.wav").write_bytes(b"RIFF")
        (stems_dir / "drums.wav").write_bytes(b"RIFF")
        return ["vocals", "drums"]

    def fake_cleanup_report(_job_obj: Job, _stems_dir: Path, _found: list[str], _job_dir: Path) -> dict[str, object]:
        analysis_dir = _job_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        report_path = analysis_dir / "quality_report.json"
        report_path.write_text(
            json.dumps({"version": 1, "enhanced_applied": True, "cleaned_stems": ["vocals"]}) + "\n",
            encoding="utf-8",
        )
        return {
            "enhanced_applied": True,
            "cleaned_stems": ["vocals"],
            "quality_status": "improved",
        }

    with (
        patch("app.pipeline.runner._prepare_local_source", return_value=source),
        patch("app.pipeline.runner.analyze", side_effect=fake_analyze),
        patch("app.pipeline.runner.separate", return_value=job_dir / "fake-demucs"),
        patch("app.pipeline.runner.collect", side_effect=fake_collect),
        patch("app.pipeline.runner.compute_stem_presence", return_value={"vocals": 88, "drums": 72}),
        patch("app.pipeline.runner.cleanup_source", return_value=None),
        patch("app.pipeline.runner.make_original_track", return_value=None),
        patch("app.pipeline.runner.make_selected_mix", return_value=None),
        patch("app.pipeline.runner.run_enhanced_cleanup", side_effect=fake_cleanup_report),
    ):
        await run_local_pipeline(job, source, tmp_path)

    assert job.status == "done"
    state = job.to_state()
    assert state["mode"] == mode
    assert "enhanced_applied" in state
    assert "enhanced_quality_status" in state
    assert isinstance(state["stems"], list)

    metadata_path = job_dir / "metadata.json"
    assert metadata_path.is_file()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["mode"] == mode
    assert "enhanced_applied" in metadata
    assert "enhanced_quality_status" in metadata

    quality_report = job_dir / "analysis" / "quality_report.json"
    if mode == "hq_enhanced":
        assert quality_report.is_file()
    else:
        assert not quality_report.exists()
