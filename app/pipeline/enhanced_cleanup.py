from __future__ import annotations

import json
import logging
from pathlib import Path

from app.core.config import ENHANCED_MAX_STEMS_TO_CLEAN, ENHANCED_MIN_IMPROVEMENT_FRAC
from app.core.models import Job

logger = logging.getLogger("stemdeck.enhanced")


def _load_audio(path: Path) -> tuple[object, int]:
    import librosa

    y, sr = librosa.load(path, sr=44100, mono=True)
    return y, sr


def _save_audio(path: Path, y: object, sr: int) -> None:
    import soundfile as sf

    sf.write(path, y, sr, subtype="PCM_16")


def _corr_abs(a: object, b: object) -> float:
    import numpy as np

    n = min(len(a), len(b))
    if n < 2048:
        return 0.0
    av = np.asarray(a[:n], dtype=np.float32)
    bv = np.asarray(b[:n], dtype=np.float32)
    if float(np.std(av)) < 1e-7 or float(np.std(bv)) < 1e-7:
        return 0.0
    c = float(np.corrcoef(av, bv)[0, 1])
    return abs(c)


def _build_corr_matrix(loaded: dict[str, tuple[object, int]]) -> dict[str, dict[str, float]]:
    names = list(loaded.keys())
    matrix: dict[str, dict[str, float]] = {}
    for a in names:
        matrix[a] = {}
        for b in names:
            if a == b:
                continue
            c = _corr_abs(loaded[a][0], loaded[b][0])
            matrix[a][b] = round(c, 4)
    return matrix


def _mean_suspicion(matrix: dict[str, dict[str, float]], stem: str) -> float:
    row = matrix.get(stem, {})
    if not row:
        return 0.0
    return float(sum(row.values()) / max(1, len(row)))


def _adaptive_threshold(matrix: dict[str, dict[str, float]]) -> float:
    vals: list[float] = []
    for row in matrix.values():
        vals.extend(row.values())
    if not vals:
        return 0.35
    vals.sort()
    p70 = vals[min(len(vals) - 1, int(len(vals) * 0.7))]
    mean = sum(vals) / len(vals)
    # Robust adaptive threshold: max of percentile and mean-based floor.
    return float(max(0.22, min(0.7, max(p70, mean + 0.03))))


def _clean_stem(name: str, y: object, profile: str = "balanced") -> object:
    import librosa
    import numpy as np

    profile = (profile or "balanced").strip().lower()
    if profile not in {"balanced", "strong", "conservative"}:
        profile = "balanced"

    # Stem-specific cleanup strategy.
    y_h, y_p = librosa.effects.hpss(y)
    if name == "drums":
        cleaned = 0.9 * y_p + 0.1 * y
    elif name == "vocals":
        cleaned = 0.25 * y_h + 0.75 * y
    elif name == "bass":
        cleaned = 0.3 * y_h + 0.7 * y
    else:
        cleaned = 0.7 * y_h + 0.3 * y

    if profile == "strong":
        cleaned = 0.55 * cleaned + 0.45 * y_h
    elif profile == "conservative":
        cleaned = 0.9 * y + 0.1 * cleaned

    # Preserve rough loudness/peak scale to avoid surprising gain jumps.
    src_peak = float(np.max(np.abs(y))) if len(y) else 0.0
    out_peak = float(np.max(np.abs(cleaned))) if len(cleaned) else 0.0
    if src_peak > 1e-6 and out_peak > 1e-6:
        cleaned = cleaned * (src_peak / out_peak)
    return cleaned


def _rms(y: object) -> float:
    import numpy as np

    if len(y) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.asarray(y, dtype=np.float32) ** 2)))


def _peak(y: object) -> float:
    import numpy as np

    if len(y) == 0:
        return 0.0
    return float(np.max(np.abs(np.asarray(y, dtype=np.float32))))


def _clip_ratio(y: object) -> float:
    import numpy as np

    if len(y) == 0:
        return 0.0
    arr = np.abs(np.asarray(y, dtype=np.float32))
    # Soft practical clipping threshold in normalized audio domain.
    return float(np.mean(arr >= 0.999))


def run_enhanced_cleanup(
    job: Job,
    stems_dir: Path,
    found: list[str],
    job_dir: Path,
    *,
    profile: str = "balanced",
    forced_stems: list[str] | None = None,
    max_stems_to_clean: int | None = None,
    min_improvement_frac: float | None = None,
) -> dict[str, object]:
    analysis_dir = job_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    loaded: dict[str, tuple[object, int]] = {}
    for name in found:
        p = stems_dir / f"{name}.wav"
        if p.is_file():
            loaded[name] = _load_audio(p)

    # Build adaptive bleed suspicion matrix.
    matrix = _build_corr_matrix(loaded)
    threshold = _adaptive_threshold(matrix)
    suspicion_scores = {name: _mean_suspicion(matrix, name) for name in loaded}
    suspicious = [name for name, score in suspicion_scores.items() if score >= threshold]
    suspicious.sort(key=lambda n: suspicion_scores.get(n, 0.0), reverse=True)
    cap = ENHANCED_MAX_STEMS_TO_CLEAN if max_stems_to_clean is None else max(1, min(6, max_stems_to_clean))
    if forced_stems:
        allowed = {name for name in forced_stems if name in loaded}
        suspicious = [name for name in suspicious if name in allowed]
        if not suspicious:
            suspicious = [name for name in loaded if name in allowed]
    suspicious = suspicious[:cap]
    min_improvement = (
        ENHANCED_MIN_IMPROVEMENT_FRAC
        if min_improvement_frac is None
        else float(max(0.0, min(0.9, min_improvement_frac)))
    )

    cleaned_stems: list[str] = []
    reverted_stems: list[str] = []
    comparisons: dict[str, dict[str, object]] = {}
    for name in suspicious:
        y, sr = loaded[name]
        cleaned = _clean_stem(name, y, profile=profile)
        baseline_score = suspicion_scores.get(name, 0.0)
        # Auto-rollback gate: cleaned version must improve mean cross-correlation.
        candidate_loaded = dict(loaded)
        candidate_loaded[name] = (cleaned, sr)
        candidate_matrix = _build_corr_matrix(candidate_loaded)
        candidate_score = _mean_suspicion(candidate_matrix, name)
        improved = candidate_score <= baseline_score * (1.0 - min_improvement)
        before_rms = _rms(y)
        after_rms = _rms(cleaned)
        before_peak = _peak(y)
        after_peak = _peak(cleaned)
        before_clip = _clip_ratio(y)
        after_clip = _clip_ratio(cleaned)
        corr_before_after = _corr_abs(y, cleaned)
        decision = "improved" if improved else "reverted"
        comparisons[name] = {
            "decision": decision,
            "suspicion_before": round(baseline_score, 6),
            "suspicion_after": round(candidate_score, 6),
            "corr_before_after": round(corr_before_after, 6),
            "rms_before": round(before_rms, 8),
            "rms_after": round(after_rms, 8),
            "energy_shift_frac": round(abs(after_rms - before_rms) / max(before_rms, 1e-9), 6),
            "peak_before": round(before_peak, 8),
            "peak_after": round(after_peak, 8),
            "clip_ratio_before": round(before_clip, 8),
            "clip_ratio_after": round(after_clip, 8),
        }
        if not improved:
            reverted_stems.append(name)
            continue
        out = stems_dir / f"{name}_clean.wav"
        _save_audio(out, cleaned, sr)
        cleaned_stems.append(name)

    report = {
        "version": 1,
        "mode": job.mode,
        "cleanup_profile": profile,
        "enhanced_applied": bool(cleaned_stems),
        "bleed_correlation_threshold": threshold,
        "suspicious_stems": suspicious,
        "suspicion_scores": {k: round(v, 4) for k, v in suspicion_scores.items()},
        "max_stems_to_clean": cap,
        "min_improvement_frac": min_improvement,
        "cleaned_stems": cleaned_stems,
        "reverted_stems": reverted_stems,
        "bleed_matrix": matrix,
    }
    report_path = analysis_dir / "quality_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    quality_status = (
        "improved" if cleaned_stems else ("reverted" if reverted_stems else "unchanged")
    )
    compare = {
        "version": 1,
        "mode": job.mode,
        "status": quality_status,
        "comparisons": comparisons,
    }
    compare_path = analysis_dir / "quality_compare.json"
    compare_path.write_text(json.dumps(compare, indent=2) + "\n", encoding="utf-8")
    logger.info("[%s] enhanced cleanup report: %s", job.id, report_path)
    logger.info("[%s] enhanced cleanup compare: %s", job.id, compare_path)
    report["quality_status"] = quality_status
    return report
