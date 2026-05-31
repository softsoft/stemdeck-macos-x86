from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def _stem_dir(stems_dir: Path, stem_name: str) -> Path:
    return stems_dir / f"{stem_name}.waves"


def _meta_path(stems_dir: Path, stem_name: str) -> Path:
    return _stem_dir(stems_dir, stem_name) / "metadata.json"


def _default_meta(stem_name: str) -> dict[str, Any]:
    return {
        "version": 1,
        "stem": stem_name,
        "active": "original",
        "variants": [],
        "updated_at": int(time.time()),
    }


def _load_meta(stems_dir: Path, stem_name: str) -> dict[str, Any]:
    path = _meta_path(stems_dir, stem_name)
    if not path.is_file():
        return _default_meta(stem_name)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _default_meta(stem_name)
    if not isinstance(data, dict):
        return _default_meta(stem_name)
    data.setdefault("version", 1)
    data.setdefault("stem", stem_name)
    data.setdefault("active", "original")
    data.setdefault("variants", [])
    data.setdefault("updated_at", int(time.time()))
    if not isinstance(data.get("variants"), list):
        data["variants"] = []
    return data


def _save_meta(stems_dir: Path, stem_name: str, meta: dict[str, Any]) -> None:
    d = _stem_dir(stems_dir, stem_name)
    d.mkdir(parents=True, exist_ok=True)
    meta["updated_at"] = int(time.time())
    _meta_path(stems_dir, stem_name).write_text(
        json.dumps(meta, indent=2) + "\n",
        encoding="utf-8",
    )


def _next_suffix(meta: dict[str, Any], category: str) -> int:
    max_n = 0
    for item in meta.get("variants", []):
        if not isinstance(item, dict):
            continue
        if str(item.get("category") or "") != category:
            continue
        n = int(item.get("index") or 0)
        if n > max_n:
            max_n = n
    return max_n + 1


def store_variant_audio(
    stems_dir: Path,
    stem_name: str,
    *,
    category: str,
    audio: object,
    sr: int,
    source: str,
    auto_activate: bool = True,
) -> dict[str, Any]:
    import soundfile as sf

    meta = _load_meta(stems_dir, stem_name)
    idx = _next_suffix(meta, category)
    variant_name = f"{category}_{idx:03d}"
    rel_file = f"{variant_name}.wav"
    out = _stem_dir(stems_dir, stem_name) / rel_file
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, audio, sr, subtype="PCM_16")
    entry = {
        "name": variant_name,
        "file": rel_file,
        "category": category,
        "index": idx,
        "created_at": int(time.time()),
        "source": source,
    }
    meta.setdefault("variants", []).append(entry)
    if auto_activate:
        meta["active"] = variant_name
    _save_meta(stems_dir, stem_name, meta)
    return entry


def list_variants(stems_dir: Path, stem_name: str) -> dict[str, Any]:
    meta = _load_meta(stems_dir, stem_name)
    base_exists = (stems_dir / f"{stem_name}.wav").is_file()
    changed = False
    present: list[dict[str, Any]] = []
    for item in meta.get("variants", []):
        if not isinstance(item, dict):
            continue
        rel = str(item.get("file") or "").strip()
        if not rel:
            continue
        p = (_stem_dir(stems_dir, stem_name) / rel).resolve()
        if p.is_file():
            present.append(item)
    if present != meta.get("variants", []):
        changed = True
    meta["variants"] = present
    active = str(meta.get("active") or "original")
    if active != "original" and all(v.get("name") != active for v in present):
        active = "original"
        meta["active"] = "original"
        changed = True
    if changed:
        _save_meta(stems_dir, stem_name, meta)
    return {
        "stem": stem_name,
        "base_exists": base_exists,
        "active": active,
        "variants": present,
    }


def set_active_variant(stems_dir: Path, stem_name: str, variant: str) -> dict[str, Any]:
    meta = _load_meta(stems_dir, stem_name)
    wanted = (variant or "").strip()
    if wanted == "original":
        meta["active"] = "original"
        _save_meta(stems_dir, stem_name, meta)
        return list_variants(stems_dir, stem_name)
    names = {str(v.get("name")) for v in meta.get("variants", []) if isinstance(v, dict)}
    if wanted not in names:
        raise ValueError(f"unknown variant '{wanted}'")
    meta["active"] = wanted
    _save_meta(stems_dir, stem_name, meta)
    return list_variants(stems_dir, stem_name)


def resolve_stem_audio_path(stems_dir: Path, stem_name: str) -> Path:
    base = (stems_dir / f"{stem_name}.wav").resolve()
    meta = _load_meta(stems_dir, stem_name)
    active = str(meta.get("active") or "original")
    if active == "original":
        return base
    for item in meta.get("variants", []):
        if not isinstance(item, dict) or str(item.get("name") or "") != active:
            continue
        rel = str(item.get("file") or "").strip()
        if not rel:
            break
        candidate = (_stem_dir(stems_dir, stem_name) / rel).resolve()
        if candidate.is_file():
            return candidate
        break
    return base


def cleanup_variant_cache(stems_dir: Path, stem_name: str, *, keep_active: bool = True) -> dict[str, Any]:
    meta = _load_meta(stems_dir, stem_name)
    active = str(meta.get("active") or "original")
    kept: list[dict[str, Any]] = []
    removed = 0
    stem_folder = _stem_dir(stems_dir, stem_name)
    for item in meta.get("variants", []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        rel = str(item.get("file") or "")
        if not rel:
            continue
        keep = keep_active and active != "original" and name == active
        p = stem_folder / rel
        if keep:
            if p.is_file():
                kept.append(item)
            continue
        if p.is_file():
            p.unlink(missing_ok=True)
            removed += 1
    meta["variants"] = kept
    if active != "original" and all(str(v.get("name")) != active for v in kept):
        meta["active"] = "original"
    _save_meta(stems_dir, stem_name, meta)
    status = list_variants(stems_dir, stem_name)
    status["removed"] = removed
    return status


def compose_variant_mix(
    stems_dir: Path,
    stem_name: str,
    *,
    variant_names: list[str],
    include_original: bool = False,
    source: str = "variant_compose",
    auto_activate: bool = True,
) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf

    meta = _load_meta(stems_dir, stem_name)
    by_name = {
        str(v.get("name")): v
        for v in meta.get("variants", [])
        if isinstance(v, dict) and v.get("name")
    }
    selected_paths: list[Path] = []
    for name in variant_names:
        item = by_name.get(name)
        if not item:
            continue
        rel = str(item.get("file") or "").strip()
        if not rel:
            continue
        p = _stem_dir(stems_dir, stem_name) / rel
        if p.is_file():
            selected_paths.append(p)
    if include_original:
        base = stems_dir / f"{stem_name}.wav"
        if base.is_file():
            selected_paths.append(base)
    if not selected_paths:
        raise ValueError("no valid variants selected")

    signals: list[np.ndarray] = []
    max_len = 0
    out_sr = 44100
    for p in selected_paths:
        y, sr = sf.read(p, always_2d=False)
        arr = np.asarray(y, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if sr != out_sr:
            # Lightweight resample fallback.
            old_x = np.linspace(0.0, 1.0, num=max(1, len(arr)), endpoint=False)
            new_n = int(round(len(arr) * (out_sr / max(1, sr))))
            new_x = np.linspace(0.0, 1.0, num=max(1, new_n), endpoint=False)
            arr = np.interp(new_x, old_x, arr).astype(np.float32)
        max_len = max(max_len, len(arr))
        signals.append(arr)
    if max_len <= 0:
        raise ValueError("selected variants are empty")

    acc = np.zeros(max_len, dtype=np.float32)
    for sig in signals:
        if len(sig) < max_len:
            pad = np.zeros(max_len, dtype=np.float32)
            pad[: len(sig)] = sig
            sig = pad
        acc += sig
    mix = acc / float(max(1, len(signals)))
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 0.999:
        mix = mix * (0.999 / peak)

    return store_variant_audio(
        stems_dir,
        stem_name,
        category=f"{stem_name}_stack",
        audio=mix,
        sr=out_sr,
        source=source,
        auto_activate=auto_activate,
    )
