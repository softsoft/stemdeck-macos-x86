from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger("stemdeck.vocals")


def _load_wav(path: Path) -> tuple[object, int]:
    import soundfile as sf

    audio, sr = sf.read(path, always_2d=False)
    return audio, int(sr)


def _to_mono(audio: object) -> object:
    import numpy as np

    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    return arr.mean(axis=1)


def _rms(y: object) -> float:
    import numpy as np

    arr = np.asarray(y, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr * arr)))


def run_vocals_reanalyze(job_id: str, job_dir: Path) -> dict[str, object]:
    import numpy as np
    import soundfile as sf
    import librosa

    stems_dir = job_dir / "stems"
    vocals_path = stems_dir / "vocals.wav"
    if not vocals_path.is_file():
        raise FileNotFoundError("vocals stem not found")

    audio, sr = _load_wav(vocals_path)
    vocals = _to_mono(audio)
    if len(vocals) < sr:
        raise RuntimeError("vocals track is too short for re-analysis")

    harmonic, _ = librosa.effects.hpss(vocals)
    lead = harmonic.astype(np.float32)
    backing = (vocals - lead).astype(np.float32)

    orig_rms = _rms(vocals)
    lead_rms = _rms(lead)
    backing_rms = _rms(backing)
    lead_ratio = (lead_rms / orig_rms) if orig_rms > 0 else 0.0
    backing_ratio = (backing_rms / orig_rms) if orig_rms > 0 else 0.0

    recon = lead + backing
    denom = float(np.linalg.norm(vocals) + 1e-9)
    recon_error = float(np.linalg.norm(vocals - recon) / denom)
    corr = float(np.corrcoef(lead, backing)[0, 1]) if len(lead) > 16 else 1.0
    confidence = max(0.0, min(1.0, 1.0 - min(1.0, abs(corr))))

    should_create = (
        recon_error <= 0.02
        and 0.15 <= lead_ratio <= 1.15
        and 0.05 <= backing_ratio <= 0.8
        and confidence >= 0.2
    )

    created: list[str] = []
    if should_create:
        lead_path = stems_dir / "lead_vocal.wav"
        backing_path = stems_dir / "backing_vocals.wav"
        sf.write(lead_path, lead, sr, subtype="PCM_16")
        sf.write(backing_path, backing, sr, subtype="PCM_16")
        created = ["lead_vocal", "backing_vocals"]

    analysis_dir = job_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "version": 1,
        "job_id": job_id,
        "status": "created" if created else "skipped",
        "created_tracks": created,
        "metrics": {
            "confidence": round(confidence, 4),
            "reconstruction_error": round(recon_error, 6),
            "lead_ratio": round(lead_ratio, 4),
            "backing_ratio": round(backing_ratio, 4),
            "lead_backing_corr": round(corr, 4),
        },
    }
    report_path = analysis_dir / "vocals_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.info("[%s] vocals re-analyze report: %s", job_id, report_path)
    return report
