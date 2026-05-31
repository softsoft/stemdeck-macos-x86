from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import subprocess
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from app.core.config import (
    JOB_ID_RE,
    JOBS_DIR,
    MAX_DURATION_SEC,
    MAX_PENDING_JOBS,
    PROCESSING_MODES,
    STEM_NAMES,
    ffprobe_executable,
    normalize_processing_mode,
)
from app.core.models import Job
from app.core.registry import all_jobs as registry_all_jobs
from app.core.registry import get as registry_get
from app.core.registry import get_proc as registry_get_proc
from app.core.registry import persist as registry_persist
from app.core.registry import register_if_capacity as registry_register_if_capacity
from app.core.registry import remove as registry_remove
from app.core.stem_variants import cleanup_variant_cache, compose_variant_mix, list_variants, set_active_variant
from app.pipeline import run_local_pipeline, run_pipeline
from app.pipeline.download import InvalidYouTubeURL, validate_youtube_url
from app.pipeline.enhanced_cleanup import run_enhanced_cleanup
from app.pipeline.vocals_reanalyze import run_vocals_reanalyze

router = APIRouter(tags=["jobs"])
logger = logging.getLogger("stemdeck.api")

_ALLOWED_EXTS = frozenset((".mp3", ".wav"))
_MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB
_WS_RE = re.compile(r"\s+")


def _sanitize_title(filename: str) -> str:
    """Strip extension, normalize whitespace, cap at 120 chars."""
    stem = Path(filename).stem
    return _WS_RE.sub(" ", stem).strip()[:120]


def _probe_duration(path: Path) -> float:
    """Run ffprobe to get file duration in seconds."""
    result = subprocess.run(
        [
            ffprobe_executable(),
            "-v",
            "quiet",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr.strip()}")
    try:
        return float(result.stdout.strip())
    except ValueError as e:
        raise RuntimeError(f"ffprobe returned non-numeric duration: {result.stdout!r}") from e


def _check_file_size(file_obj: object) -> int:
    """Seek to end, return size, rewind. Operates on the SpooledTemporaryFile
    backing a starlette UploadFile — synchronous, suitable for to_thread."""
    file_obj.seek(0, 2)  # type: ignore[union-attr]
    size = file_obj.tell()  # type: ignore[union-attr]
    file_obj.seek(0)  # type: ignore[union-attr]
    return size


def _copy_to_dest(src_file: object, dest: Path) -> None:
    """Copy SpooledTemporaryFile contents to dest. Synchronous, run in thread."""
    with dest.open("wb") as out:
        shutil.copyfileobj(src_file, out)  # type: ignore[arg-type]


def _rmtree_job(job_id: str) -> None:
    job_dir = JOBS_DIR / job_id
    if not job_dir.is_dir():
        return
    try:
        shutil.rmtree(job_dir)
    except Exception:
        logger.warning("failed to remove job dir %s", job_dir, exc_info=True)


def _task_error_cb(task: asyncio.Task) -> None:
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("pipeline task raised unhandled exception", exc_info=exc)


class JobRequest(BaseModel):
    url: str
    # Subset of stems to include in the post-processing "selected mix"
    # audio file. None = all 6 (no extra mix produced; would equal the
    # original). Unknown stem names are dropped silently rather than
    # rejected, so a future model with extra stems doesn't break older
    # clients pinning the old set.
    stems: list[str] | None = None
    mode: str | None = None

    @field_validator("mode")
    @classmethod
    def _validate_mode(cls, value: str | None) -> str | None:
        if value is None:
            return value
        mode = value.strip().lower()
        if mode not in PROCESSING_MODES:
            raise ValueError(f"Unsupported mode '{value}'. Allowed: {', '.join(PROCESSING_MODES)}")
        return mode


@router.post("")
async def create_job(request: Request) -> dict[str, str]:
    """Submit a YouTube URL (JSON body) or upload an audio file (multipart/form-data)
    to start a stem-separation job. Returns the new job ID."""
    ct = request.headers.get("content-type", "")
    if "multipart/form-data" in ct:
        return await _create_local_job(request)
    return await _create_youtube_job(request)


async def _create_youtube_job(request: Request) -> dict[str, str]:
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid JSON: {e}") from e
    try:
        payload = JobRequest(**body)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    try:
        url = validate_youtube_url(payload.url)
    except InvalidYouTubeURL as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    selected = [s for s in payload.stems if s in STEM_NAMES] if payload.stems else list(STEM_NAMES)
    if not selected:
        selected = list(STEM_NAMES)

    mode = normalize_processing_mode(payload.mode)
    job = Job(id=uuid.uuid4().hex[:12], selected_stems=selected, source_url=url, mode=mode)
    if not registry_register_if_capacity(job, MAX_PENDING_JOBS):
        raise HTTPException(status_code=503, detail="Server busy, please try again later")
    task = asyncio.create_task(run_pipeline(job, url, JOBS_DIR))
    task.add_done_callback(_task_error_cb)
    return {"job_id": job.id}


async def _create_local_job(request: Request) -> dict[str, str]:
    # Fast pre-check: if already at capacity, reject before touching disk.
    # The real atomic check happens in register_if_capacity after the upload.
    if sum(1 for j in registry_all_jobs().values() if j.status == "queued") >= MAX_PENDING_JOBS:
        raise HTTPException(status_code=503, detail="Server busy, please try again later")

    # Quick pre-check on Content-Length to fail fast for obviously oversized
    # uploads without buffering the whole body first.
    cl_header = request.headers.get("content-length")
    if cl_header:
        try:
            if int(cl_header) > _MAX_UPLOAD_BYTES + 4096:
                raise HTTPException(status_code=422, detail="File exceeds 100 MB limit")
        except ValueError:
            pass

    form = await request.form()
    upload = form.get("file")
    stems_raw = form.get("stems", "[]")
    mode_raw = str(form.get("mode", "") or "")

    if upload is None or not hasattr(upload, "filename"):
        raise HTTPException(status_code=422, detail="No file provided")

    filename: str = getattr(upload, "filename", "") or ""
    ext = Path(filename).suffix.lower()
    if ext not in _ALLOWED_EXTS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{ext}': only .mp3 and .wav are accepted",
        )

    # Validate stems list from form field
    try:
        stems_list = json.loads(stems_raw)
        if not isinstance(stems_list, list):
            raise ValueError
    except (json.JSONDecodeError, ValueError):
        stems_list = []
    selected = [s for s in stems_list if s in STEM_NAMES] or list(STEM_NAMES)
    if mode_raw.strip() and mode_raw.strip().lower() not in PROCESSING_MODES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported mode '{mode_raw}'. Allowed: {', '.join(PROCESSING_MODES)}",
        )
    mode = normalize_processing_mode(mode_raw)

    # Check actual file size (SpooledTemporaryFile is already buffered at this
    # point; seek/tell are fast and don't re-read the body).
    file_obj = upload.file  # type: ignore[union-attr]
    file_size = await asyncio.to_thread(_check_file_size, file_obj)
    if file_size == 0:
        raise HTTPException(status_code=422, detail="Uploaded file is empty")
    if file_size > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=422, detail="File exceeds 100 MB limit")

    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    source_path = job_dir / f"source{ext}"

    job_dir.mkdir(parents=True, exist_ok=True)
    try:
        await asyncio.to_thread(_copy_to_dest, file_obj, source_path)

        # Duration check before registering the job so a violation leaves no
        # registered job and no leftover directory.
        try:
            duration = await asyncio.to_thread(_probe_duration, source_path)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Could not read file duration: {e}") from e

        if duration > MAX_DURATION_SEC:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"File is {int(duration // 60)} min — limit is {MAX_DURATION_SEC // 60} min"
                ),
            )
    except HTTPException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise

    title = _sanitize_title(filename)
    local_source_url = f"local:{title}"
    job = Job(
        id=job_id,
        selected_stems=selected,
        mode=mode,
        title=title,
        duration_sec=duration,
        source_url=local_source_url,
    )
    if not registry_register_if_capacity(job, MAX_PENDING_JOBS):
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=503, detail="Server busy, please try again later")
    task = asyncio.create_task(run_local_pipeline(job, source_path, JOBS_DIR))
    task.add_done_callback(_task_error_cb)
    return {"job_id": job.id}


@router.get("")
def list_jobs() -> list[dict]:
    """List all completed jobs in the library, sorted by creation time."""
    return [
        job.to_state()
        for job in sorted(registry_all_jobs().values(), key=lambda j: j.created_at)
        if job.status == "done"
    ]


@router.get("/{job_id}")
def get_job(job_id: str) -> dict:
    """Get the current state of a job by ID."""
    job = registry_get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job.to_state()


@router.get("/{job_id}/analysis/quality-report")
def get_quality_report(job_id: str) -> JSONResponse:
    if not JOB_ID_RE.match(job_id):
        raise HTTPException(status_code=404, detail="job not found")
    job = registry_get(job_id)
    if job is None or job.status != "done":
        raise HTTPException(status_code=404, detail="job not ready")
    report_path = (JOBS_DIR / job_id / "analysis" / "quality_report.json").resolve()
    if not report_path.is_file() or not report_path.is_relative_to(JOBS_DIR.resolve()):
        raise HTTPException(status_code=404, detail="quality report not found")
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=500, detail=f"failed to read quality report: {e}") from e
    return JSONResponse(payload)


@router.get("/{job_id}/analysis/vocals-report")
def get_vocals_report(job_id: str) -> JSONResponse:
    if not JOB_ID_RE.match(job_id):
        raise HTTPException(status_code=404, detail="job not found")
    job = registry_get(job_id)
    if job is None or job.status != "done":
        raise HTTPException(status_code=404, detail="job not ready")
    report_path = (JOBS_DIR / job_id / "analysis" / "vocals_report.json").resolve()
    if not report_path.is_file() or not report_path.is_relative_to(JOBS_DIR.resolve()):
        raise HTTPException(status_code=404, detail="vocals report not found")
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=500, detail=f"failed to read vocals report: {e}") from e
    return JSONResponse(payload)


@router.post("/{job_id}/vocals/reanalyze")
async def reanalyze_vocals(job_id: str) -> dict:
    if not JOB_ID_RE.match(job_id):
        raise HTTPException(status_code=404, detail="job not found")
    job = registry_get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.status != "done":
        raise HTTPException(status_code=409, detail="job is not ready")
    job_dir = (JOBS_DIR / job_id).resolve()
    if not job_dir.is_dir() or not job_dir.is_relative_to(JOBS_DIR.resolve()):
        raise HTTPException(status_code=404, detail="job not found")

    try:
        report = await asyncio.to_thread(run_vocals_reanalyze, job_id, job_dir)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logger.exception("vocals re-analyze failed for %s: %s", job_id, e)
        raise HTTPException(status_code=500, detail=f"vocals re-analyze failed: {e}") from e

    created_tracks = list(report.get("created_tracks") or [])
    status = str(report.get("status") or "skipped")
    job.vocals_split_status = status
    job.vocals_split_tracks = created_tracks

    stems_dir = job_dir / "stems"
    _refresh_job_stems_from_dir(job, stems_dir)
    registry_persist(JOBS_DIR)
    return {
        "job_id": job_id,
        "vocals_split_status": status,
        "created_tracks": created_tracks,
        "report_url": f"/api/jobs/{job_id}/analysis/vocals-report",
    }


@router.post("/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    """Request cancellation of a running job. Idempotent for terminal jobs."""
    job = registry_get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.status in ("done", "error", "cancelled"):
        return job.to_state()
    job.cancel_requested = True
    proc = registry_get_proc(job_id)
    if proc is not None and proc.poll() is None:
        proc.terminate()
    return job.to_state()


_SECTION_ID_RE = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")
_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{3,8}$")


class SectionItem(BaseModel):
    id: str
    name: str
    start: float
    end: float
    color: str

    @field_validator("id")
    @classmethod
    def _check_id(cls, v: str) -> str:
        if not _SECTION_ID_RE.match(v):
            raise ValueError("invalid section id")
        return v

    @field_validator("name")
    @classmethod
    def _check_name(cls, v: str) -> str:
        return v.strip()[:64] or "Section"

    @field_validator("color")
    @classmethod
    def _check_color(cls, v: str) -> str:
        if not _COLOR_RE.match(v):
            raise ValueError("invalid color")
        return v

    @field_validator("start", "end")
    @classmethod
    def _check_time(cls, v: float) -> float:
        if not (0 <= v < 86400):
            raise ValueError("time out of range")
        return round(v, 3)


class SectionsBody(BaseModel):
    sections: list[SectionItem]


class EnhancedActionBody(BaseModel):
    profile: str | None = None
    max_stems_to_clean: int | None = None
    min_improvement_frac: float | None = None

    @field_validator("profile")
    @classmethod
    def _validate_profile(cls, value: str | None) -> str | None:
        if value is None:
            return value
        normalized = value.strip().lower()
        if normalized not in {"balanced", "strong", "conservative"}:
            raise ValueError("Unsupported profile. Allowed: balanced, strong, conservative")
        return normalized


class VariantActivateBody(BaseModel):
    variant: str

    @field_validator("variant")
    @classmethod
    def _validate_variant(cls, value: str) -> str:
        v = (value or "").strip()
        if not v:
            raise ValueError("variant is required")
        return v


class VariantComposeBody(BaseModel):
    variants: list[str]
    include_original: bool = False

    @field_validator("variants")
    @classmethod
    def _validate_variants(cls, value: list[str]) -> list[str]:
        out = [str(v).strip() for v in (value or []) if str(v).strip()]
        if not out:
            raise ValueError("at least one variant is required")
        return out


def _refresh_job_stems_from_dir(job: Job, stems_dir: Path) -> None:
    stem_names = sorted(p.stem for p in stems_dir.glob("*.wav"))
    job.stems = [{"name": name, "url": f"/api/jobs/{job.id}/stems/{name}.wav"} for name in stem_names]


@router.patch("/{job_id}/sections")
def update_sections(job_id: str, body: SectionsBody) -> dict:
    """Save named timeline sections (intro, verse, chorus, etc.) for a done job."""
    if not JOB_ID_RE.match(job_id):
        raise HTTPException(status_code=404, detail="job not found")
    job = registry_get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    validated = [s.model_dump() for s in body.sections]
    job.sections = validated

    job_dir = (JOBS_DIR / job_id).resolve()
    if not job_dir.is_relative_to(JOBS_DIR.resolve()):
        raise HTTPException(status_code=404, detail="job not found")
    meta_path = job_dir / "metadata.json"

    meta: dict = {}
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    meta["sections"] = validated
    try:
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        logger.exception("failed to write sections for %s: %s", job_id, exc)
        raise HTTPException(status_code=500, detail="failed to save sections") from exc

    registry_persist(JOBS_DIR)

    return {"job_id": job_id, "sections": validated}


@router.post("/{job_id}/analysis/reanalyze")
async def rerun_enhanced_analysis(job_id: str, body: EnhancedActionBody | None = None) -> dict:
    if not JOB_ID_RE.match(job_id):
        raise HTTPException(status_code=404, detail="job not found")
    job = registry_get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.status != "done":
        raise HTTPException(status_code=409, detail="job is not ready")
    payload = body or EnhancedActionBody()
    job_dir = (JOBS_DIR / job_id).resolve()
    stems_dir = (job_dir / "stems").resolve()
    if not stems_dir.is_dir() or not stems_dir.is_relative_to(JOBS_DIR.resolve()):
        raise HTTPException(status_code=404, detail="stems not found")
    found = [name for name in STEM_NAMES if (stems_dir / f"{name}.wav").is_file()]
    if not found:
        raise HTTPException(status_code=404, detail="no base stems found")

    try:
        report = await asyncio.to_thread(
            run_enhanced_cleanup,
            job,
            stems_dir,
            found,
            job_dir,
            profile=payload.profile or "balanced",
            max_stems_to_clean=payload.max_stems_to_clean,
            min_improvement_frac=payload.min_improvement_frac,
        )
    except Exception as e:
        logger.exception("enhanced re-analyze failed for %s: %s", job_id, e)
        raise HTTPException(status_code=500, detail=f"enhanced re-analyze failed: {e}") from e

    job.enhanced_applied = bool(report.get("enhanced_applied"))
    job.enhanced_cleaned_stems = list(report.get("cleaned_stems") or [])
    job.enhanced_quality_status = str(report.get("quality_status") or "unchanged")
    job.enhanced_fallback_reason = None
    _refresh_job_stems_from_dir(job, stems_dir)
    registry_persist(JOBS_DIR)
    return {
        "job_id": job_id,
        "status": "ok",
        "enhanced_applied": job.enhanced_applied,
        "enhanced_quality_status": job.enhanced_quality_status,
        "cleaned_stems": job.enhanced_cleaned_stems,
        "report_url": f"/api/jobs/{job_id}/analysis/quality-report",
    }


@router.post("/{job_id}/stems/{stem_name}/reclean")
async def reclean_stem(job_id: str, stem_name: str, body: EnhancedActionBody | None = None) -> dict:
    if not JOB_ID_RE.match(job_id):
        raise HTTPException(status_code=404, detail="job not found")
    if stem_name not in STEM_NAMES:
        raise HTTPException(status_code=422, detail=f"unsupported stem '{stem_name}'")
    job = registry_get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.status != "done":
        raise HTTPException(status_code=409, detail="job is not ready")
    payload = body or EnhancedActionBody()
    job_dir = (JOBS_DIR / job_id).resolve()
    stems_dir = (job_dir / "stems").resolve()
    stem_path = stems_dir / f"{stem_name}.wav"
    if not stem_path.is_file() or not stems_dir.is_relative_to(JOBS_DIR.resolve()):
        raise HTTPException(status_code=404, detail="stem not found")
    found = [name for name in STEM_NAMES if (stems_dir / f"{name}.wav").is_file()]
    try:
        report = await asyncio.to_thread(
            run_enhanced_cleanup,
            job,
            stems_dir,
            found,
            job_dir,
            profile=payload.profile or "balanced",
            forced_stems=[stem_name],
            max_stems_to_clean=1,
            min_improvement_frac=payload.min_improvement_frac,
        )
    except Exception as e:
        logger.exception("stem reclean failed for %s/%s: %s", job_id, stem_name, e)
        raise HTTPException(status_code=500, detail=f"stem reclean failed: {e}") from e
    job.enhanced_applied = bool(report.get("enhanced_applied"))
    job.enhanced_cleaned_stems = list(report.get("cleaned_stems") or [])
    job.enhanced_quality_status = str(report.get("quality_status") or "unchanged")
    job.enhanced_fallback_reason = None
    stem_names = sorted(p.stem for p in stems_dir.glob("*.wav"))
    job.stems = [{"name": name, "url": f"/api/jobs/{job.id}/stems/{name}.wav"} for name in stem_names]
    registry_persist(JOBS_DIR)
    return {
        "job_id": job_id,
        "stem": stem_name,
        "status": "ok",
        "enhanced_applied": job.enhanced_applied,
        "enhanced_quality_status": job.enhanced_quality_status,
        "cleaned_stems": job.enhanced_cleaned_stems,
    }


@router.get("/{job_id}/stems/{stem_name}/variants")
def get_stem_variants(job_id: str, stem_name: str) -> dict:
    if not JOB_ID_RE.match(job_id):
        raise HTTPException(status_code=404, detail="job not found")
    if stem_name not in STEM_NAMES:
        raise HTTPException(status_code=422, detail=f"unsupported stem '{stem_name}'")
    job = registry_get(job_id)
    if job is None or job.status != "done":
        raise HTTPException(status_code=404, detail="job not ready")
    stems_dir = (JOBS_DIR / job_id / "stems").resolve()
    if not stems_dir.is_dir() or not stems_dir.is_relative_to(JOBS_DIR.resolve()):
        raise HTTPException(status_code=404, detail="stems not found")
    if not (stems_dir / f"{stem_name}.wav").is_file():
        raise HTTPException(status_code=404, detail="stem not found")
    return list_variants(stems_dir, stem_name)


@router.post("/{job_id}/stems/{stem_name}/variants/activate")
def activate_stem_variant(job_id: str, stem_name: str, body: VariantActivateBody) -> dict:
    if not JOB_ID_RE.match(job_id):
        raise HTTPException(status_code=404, detail="job not found")
    if stem_name not in STEM_NAMES:
        raise HTTPException(status_code=422, detail=f"unsupported stem '{stem_name}'")
    job = registry_get(job_id)
    if job is None or job.status != "done":
        raise HTTPException(status_code=404, detail="job not ready")
    stems_dir = (JOBS_DIR / job_id / "stems").resolve()
    if not stems_dir.is_dir() or not stems_dir.is_relative_to(JOBS_DIR.resolve()):
        raise HTTPException(status_code=404, detail="stems not found")
    if not (stems_dir / f"{stem_name}.wav").is_file():
        raise HTTPException(status_code=404, detail="stem not found")
    try:
        status = set_active_variant(stems_dir, stem_name, body.variant)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    _refresh_job_stems_from_dir(job, stems_dir)
    registry_persist(JOBS_DIR)
    return {"job_id": job_id, **status}


@router.post("/{job_id}/stems/{stem_name}/variants/cleanup")
def cleanup_stem_variants(job_id: str, stem_name: str) -> dict:
    if not JOB_ID_RE.match(job_id):
        raise HTTPException(status_code=404, detail="job not found")
    if stem_name not in STEM_NAMES:
        raise HTTPException(status_code=422, detail=f"unsupported stem '{stem_name}'")
    job = registry_get(job_id)
    if job is None or job.status != "done":
        raise HTTPException(status_code=404, detail="job not ready")
    stems_dir = (JOBS_DIR / job_id / "stems").resolve()
    if not stems_dir.is_dir() or not stems_dir.is_relative_to(JOBS_DIR.resolve()):
        raise HTTPException(status_code=404, detail="stems not found")
    if not (stems_dir / f"{stem_name}.wav").is_file():
        raise HTTPException(status_code=404, detail="stem not found")
    status = cleanup_variant_cache(stems_dir, stem_name, keep_active=True)
    _refresh_job_stems_from_dir(job, stems_dir)
    registry_persist(JOBS_DIR)
    return {"job_id": job_id, **status}


@router.post("/{job_id}/stems/{stem_name}/variants/compose")
def compose_stem_variants(job_id: str, stem_name: str, body: VariantComposeBody) -> dict:
    if not JOB_ID_RE.match(job_id):
        raise HTTPException(status_code=404, detail="job not found")
    if stem_name not in STEM_NAMES:
        raise HTTPException(status_code=422, detail=f"unsupported stem '{stem_name}'")
    job = registry_get(job_id)
    if job is None or job.status != "done":
        raise HTTPException(status_code=404, detail="job not ready")
    stems_dir = (JOBS_DIR / job_id / "stems").resolve()
    if not stems_dir.is_dir() or not stems_dir.is_relative_to(JOBS_DIR.resolve()):
        raise HTTPException(status_code=404, detail="stems not found")
    if not (stems_dir / f"{stem_name}.wav").is_file():
        raise HTTPException(status_code=404, detail="stem not found")
    try:
        created = compose_variant_mix(
            stems_dir,
            stem_name,
            variant_names=body.variants,
            include_original=bool(body.include_original),
            source="ui_compose",
            auto_activate=True,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    _refresh_job_stems_from_dir(job, stems_dir)
    registry_persist(JOBS_DIR)
    status = list_variants(stems_dir, stem_name)
    return {"job_id": job_id, "created": created, **status}


@router.delete("/{job_id}")
def delete_job(job_id: str) -> dict[str, str]:
    """Delete a completed or failed job and remove its stem files from disk."""
    if not JOB_ID_RE.match(job_id):
        raise HTTPException(status_code=404, detail="job not found")
    job = registry_get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.status not in ("done", "error", "cancelled"):
        raise HTTPException(status_code=409, detail="job is still running")
    _rmtree_job(job_id)
    registry_remove(job_id)
    registry_persist(JOBS_DIR)
    return {"job_id": job_id, "status": "deleted"}
