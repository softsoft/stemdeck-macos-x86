from __future__ import annotations

import io
import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.core.config import MAX_PENDING_JOBS
from app.core.models import Job
from app.core.registry import _jobs


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Each test gets a fresh in-memory registry."""
    _jobs.clear()
    yield
    _jobs.clear()


@pytest.fixture
def client():
    async def _noop_pipeline(job, url, jobs_dir):
        return None

    with patch("app.api.jobs.run_pipeline", _noop_pipeline):
        from app.main import app

        with TestClient(app) as c:
            yield c


@pytest.fixture
def upload_client(tmp_path, monkeypatch):
    import app.core.config as cfg

    monkeypatch.setattr(cfg, "JOBS_DIR", tmp_path)

    async def _noop_local(job, source_path, jobs_dir):
        return None

    async def _noop_youtube(job, url, jobs_dir):
        return None

    with (
        patch("app.api.jobs.run_local_pipeline", _noop_local),
        patch("app.api.jobs.run_pipeline", _noop_youtube),
        patch("app.api.jobs._probe_duration", return_value=60.0),
    ):
        from app.main import app

        with TestClient(app) as c:
            yield c


def test_post_rejects_invalid_url(client):
    r = client.post("/api/jobs", json={"url": "https://example.com/foo"})
    assert r.status_code == 422
    assert "unsupported host" in r.json()["detail"]


def test_post_rejects_empty_url(client):
    r = client.post("/api/jobs", json={"url": ""})
    assert r.status_code == 422


def test_post_accepts_youtube_url(client):
    r = client.post("/api/jobs", json={"url": "https://youtu.be/dQw4w9WgXcQ"})
    assert r.status_code == 200
    assert "job_id" in r.json()
    assert len(r.json()["job_id"]) == 12


def test_post_accepts_mode(client):
    r = client.post(
        "/api/jobs",
        json={"url": "https://youtu.be/dQw4w9WgXcQ", "mode": "hq_enhanced"},
    )
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    assert _jobs[job_id].mode == "hq_enhanced"


def test_post_rejects_invalid_mode(client):
    r = client.post(
        "/api/jobs",
        json={"url": "https://youtu.be/dQw4w9WgXcQ", "mode": "turbo"},
    )
    assert r.status_code == 422
    assert "Unsupported mode" in r.json()["detail"]


def test_get_unknown_job_returns_404(client):
    r = client.get("/api/jobs/000000000000")
    assert r.status_code == 404


def test_cancel_unknown_job_returns_404(client):
    r = client.post("/api/jobs/000000000000/cancel")
    assert r.status_code == 404


def test_delete_running_job_rejected(client):
    r = client.post("/api/jobs", json={"url": "https://youtu.be/dQw4w9WgXcQ"})
    job_id = r.json()["job_id"]
    r = client.delete(f"/api/jobs/{job_id}")
    assert r.status_code == 409


def test_cancel_sets_flag_and_returns_state(client):
    r = client.post("/api/jobs", json={"url": "https://youtu.be/dQw4w9WgXcQ"})
    job_id = r.json()["job_id"]
    r = client.post(f"/api/jobs/{job_id}/cancel")
    assert r.status_code == 200
    assert _jobs[job_id].cancel_requested is True


def test_cancel_after_done_is_idempotent(client):
    r = client.post("/api/jobs", json={"url": "https://youtu.be/dQw4w9WgXcQ"})
    job_id = r.json()["job_id"]
    _jobs[job_id].status = "done"
    r = client.post(f"/api/jobs/{job_id}/cancel")
    assert r.status_code == 200
    assert _jobs[job_id].cancel_requested is False


# ─── Capacity (503) ───────────────────────────────────────────────────────────


def test_youtube_503_when_queue_full(client):
    for _ in range(MAX_PENDING_JOBS):
        r = client.post("/api/jobs", json={"url": "https://youtu.be/dQw4w9WgXcQ"})
        assert r.status_code == 200
    r = client.post("/api/jobs", json={"url": "https://youtu.be/dQw4w9WgXcQ"})
    assert r.status_code == 503


def test_upload_503_when_queue_full(upload_client):
    for _ in range(MAX_PENDING_JOBS):
        r = upload_client.post("/api/jobs", json={"url": "https://youtu.be/dQw4w9WgXcQ"})
        assert r.status_code == 200
    data = io.BytesIO(b"ID3" + b"\x00" * 128)
    r = upload_client.post(
        "/api/jobs",
        files={"file": ("track.mp3", data, "audio/mpeg")},
    )
    assert r.status_code == 503


# ─── File upload ─────────────────────────────────────────────────────────────


def test_upload_rejects_unsupported_extension(upload_client):
    data = io.BytesIO(b"OGG data")
    r = upload_client.post(
        "/api/jobs",
        files={"file": ("track.ogg", data, "audio/ogg")},
    )
    assert r.status_code == 422
    assert "Unsupported file type" in r.json()["detail"]


def test_upload_rejects_empty_file(upload_client):
    r = upload_client.post(
        "/api/jobs",
        files={"file": ("track.wav", io.BytesIO(b""), "audio/wav")},
    )
    assert r.status_code == 422
    assert "empty" in r.json()["detail"].lower()


def test_upload_mp3_returns_job_id(upload_client):
    data = io.BytesIO(b"ID3" + b"\x00" * 128)
    r = upload_client.post(
        "/api/jobs",
        files={"file": ("my_track.mp3", data, "audio/mpeg")},
    )
    assert r.status_code == 200
    assert "job_id" in r.json()
    assert len(r.json()["job_id"]) == 12


def test_upload_wav_returns_job_id(upload_client):
    data = io.BytesIO(b"RIFF" + b"\x00" * 128)
    r = upload_client.post(
        "/api/jobs",
        files={"file": ("my_track.wav", data, "audio/wav")},
    )
    assert r.status_code == 200
    assert "job_id" in r.json()


def test_upload_accepts_mode(upload_client):
    data = io.BytesIO(b"RIFF" + b"\x00" * 128)
    r = upload_client.post(
        "/api/jobs",
        data={"mode": "fast"},
        files={"file": ("my_track.wav", data, "audio/wav")},
    )
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    assert _jobs[job_id].mode == "fast"


def test_upload_rejects_invalid_mode(upload_client):
    data = io.BytesIO(b"RIFF" + b"\x00" * 128)
    r = upload_client.post(
        "/api/jobs",
        data={"mode": "turbo"},
        files={"file": ("my_track.wav", data, "audio/wav")},
    )
    assert r.status_code == 422
    assert "Unsupported mode" in r.json()["detail"]


# ─── Sections endpoint ────────────────────────────────────────────────────────


@pytest.fixture
def done_job(client, tmp_path, monkeypatch):
    import app.api.jobs as jobs_mod

    monkeypatch.setattr(jobs_mod, "JOBS_DIR", tmp_path)
    job = Job(id="abcdefabcdef")
    job.status = "done"
    _jobs[job.id] = job
    job_dir = tmp_path / job.id
    job_dir.mkdir(parents=True, exist_ok=True)
    return job


def test_sections_happy_path(client, done_job, tmp_path):
    payload = {
        "sections": [{"id": "sec1", "name": "Verse", "start": 0.0, "end": 30.0, "color": "#ff0000"}]
    }
    r = client.patch(f"/api/jobs/{done_job.id}/sections", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["job_id"] == done_job.id
    assert len(body["sections"]) == 1
    assert body["sections"][0]["name"] == "Verse"
    # Verify written to disk
    meta_path = tmp_path / done_job.id / "metadata.json"
    assert meta_path.is_file()
    meta = json.loads(meta_path.read_text())
    assert meta["sections"][0]["id"] == "sec1"


def test_quality_report_happy_path(client, done_job, tmp_path):
    analysis_dir = tmp_path / done_job.id / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    report = {"version": 1, "enhanced_applied": True, "cleaned_stems": ["guitar"]}
    (analysis_dir / "quality_report.json").write_text(json.dumps(report) + "\n", encoding="utf-8")
    r = client.get(f"/api/jobs/{done_job.id}/analysis/quality-report")
    assert r.status_code == 200
    assert r.json()["enhanced_applied"] is True


def test_vocals_reanalyze_happy_path(client, done_job, tmp_path, monkeypatch):
    import app.api.jobs as jobs_mod

    stems_dir = tmp_path / done_job.id / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)
    (stems_dir / "vocals.wav").write_bytes(b"RIFF")

    for stem in ["vocals", "drums"]:
        done_job.stems.append({"name": stem, "url": f"/api/jobs/{done_job.id}/stems/{stem}.wav"})

    def _fake_run(job_id, job_dir):
        (job_dir / "stems" / "lead_vocal.wav").write_bytes(b"RIFF")
        (job_dir / "stems" / "backing_vocals.wav").write_bytes(b"RIFF")
        analysis_dir = job_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        (analysis_dir / "vocals_report.json").write_text(
            json.dumps({"status": "created", "created_tracks": ["lead_vocal", "backing_vocals"]}) + "\n",
            encoding="utf-8",
        )
        return {"status": "created", "created_tracks": ["lead_vocal", "backing_vocals"]}

    monkeypatch.setattr(jobs_mod, "run_vocals_reanalyze", _fake_run)

    r = client.post(f"/api/jobs/{done_job.id}/vocals/reanalyze")
    assert r.status_code == 200
    assert r.json()["vocals_split_status"] == "created"
    assert "lead_vocal" in r.json()["created_tracks"]
    assert any(stem["name"] == "lead_vocal" for stem in done_job.stems)


def test_vocals_report_happy_path(client, done_job, tmp_path):
    analysis_dir = tmp_path / done_job.id / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    report = {"version": 1, "status": "skipped", "created_tracks": []}
    (analysis_dir / "vocals_report.json").write_text(json.dumps(report) + "\n", encoding="utf-8")
    r = client.get(f"/api/jobs/{done_job.id}/analysis/vocals-report")
    assert r.status_code == 200
    assert r.json()["status"] == "skipped"


def test_reanalyze_happy_path(client, done_job, tmp_path, monkeypatch):
    import app.api.jobs as jobs_mod

    stems_dir = tmp_path / done_job.id / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)
    (stems_dir / "vocals.wav").write_bytes(b"RIFF")
    (stems_dir / "drums.wav").write_bytes(b"RIFF")

    def _fake_cleanup(job_obj, _stems_dir, _found, _job_dir, **kwargs):
        analysis_dir = _job_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        (analysis_dir / "quality_report.json").write_text(
            json.dumps({"enhanced_applied": True, "cleaned_stems": ["vocals"]}) + "\n",
            encoding="utf-8",
        )
        return {"enhanced_applied": True, "cleaned_stems": ["vocals"], "quality_status": "improved"}

    monkeypatch.setattr(jobs_mod, "run_enhanced_cleanup", _fake_cleanup)
    r = client.post(f"/api/jobs/{done_job.id}/analysis/reanalyze", json={"profile": "strong"})
    assert r.status_code == 200
    assert r.json()["enhanced_quality_status"] == "improved"
    assert "vocals" in r.json()["cleaned_stems"]


def test_reclean_stem_happy_path(client, done_job, tmp_path, monkeypatch):
    import app.api.jobs as jobs_mod

    stems_dir = tmp_path / done_job.id / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)
    (stems_dir / "vocals.wav").write_bytes(b"RIFF")
    (stems_dir / "drums.wav").write_bytes(b"RIFF")

    def _fake_cleanup(job_obj, _stems_dir, _found, _job_dir, **kwargs):
        return {"enhanced_applied": True, "cleaned_stems": ["drums"], "quality_status": "improved"}

    monkeypatch.setattr(jobs_mod, "run_enhanced_cleanup", _fake_cleanup)
    r = client.post(f"/api/jobs/{done_job.id}/stems/drums/reclean", json={"profile": "conservative"})
    assert r.status_code == 200
    assert r.json()["stem"] == "drums"
    assert "drums" in r.json()["cleaned_stems"]


def test_quality_report_missing_returns_404(client, done_job):
    r = client.get(f"/api/jobs/{done_job.id}/analysis/quality-report")
    assert r.status_code == 404


def test_sections_unknown_job_returns_404(client):
    payload = {"sections": []}
    r = client.patch("/api/jobs/000000000000/sections", json=payload)
    assert r.status_code == 404


def test_sections_malformed_job_id_returns_404(client):
    # Job IDs must be 12 lowercase hex chars; anything else is rejected.
    r = client.patch("/api/jobs/BADID/sections", json={"sections": []})
    assert r.status_code == 404


def test_sections_invalid_color_returns_422(client, done_job):
    payload = {
        "sections": [
            {"id": "sec1", "name": "Intro", "start": 0.0, "end": 10.0, "color": "not-a-color"}
        ]
    }
    r = client.patch(f"/api/jobs/{done_job.id}/sections", json=payload)
    assert r.status_code == 422


def test_sections_invalid_id_returns_422(client, done_job):
    payload = {
        "sections": [{"id": "has space", "name": "x", "start": 0.0, "end": 5.0, "color": "#fff"}]
    }
    r = client.patch(f"/api/jobs/{done_job.id}/sections", json=payload)
    assert r.status_code == 422


# ─── SSE job_id validation ────────────────────────────────────────────────────


def test_sse_rejects_malformed_job_id(client):
    for bad_id in ("../etc", "ABC", "abcdefabcdef0"):
        r = client.get(f"/api/jobs/{bad_id}/events")
        assert r.status_code == 404, f"SSE should 404 for id {bad_id!r}"


def test_sse_503_when_connection_cap_reached(client):
    """#86/#88: SSE endpoint rejects with 503 when _MAX_SSE_CONNECTIONS is reached."""
    import app.api.events as events_mod

    original = events_mod._sse_active
    try:
        events_mod._sse_active = events_mod._MAX_SSE_CONNECTIONS
        job = Job(id="abcdefabcdef")
        job.status = "done"
        _jobs[job.id] = job
        r = client.get(f"/api/jobs/{job.id}/events")
        assert r.status_code == 503
    finally:
        events_mod._sse_active = original
