#!/usr/bin/env bash
set -euo pipefail

ARCH="${ARCH:-arm64}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD_DIR="${REPO_ROOT}/.build"
STAGING="${BUILD_DIR}/runtime-staging-${ARCH}"
RUNTIME_DIR="${STAGING}/runtime"
PYTHON_DIR="${RUNTIME_DIR}/python"
BACKEND_DIR="${RUNTIME_DIR}/backend"

if [[ -z "${VERSION:-}" ]]; then
  VERSION="$(
    python3 - "$REPO_ROOT/pyproject.toml" <<'PY'
import pathlib, re, sys
text = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
m = re.search(r'(?m)^version\s*=\s*"([^"]+)"\s*$', text)
print(m.group(1) if m else "0.6.0-alpha.2")
PY
  )"
fi
VERSION="${VERSION#v}"
RELEASE_BASE_URL="${RELEASE_BASE_URL:-https://github.com/stemdeckapp/stemdeck/releases/download/v${VERSION}}"

if [[ "$(uname)" != "Darwin" ]]; then
  echo "ERROR: make-runtime-pack.sh must run on macOS" >&2
  exit 1
fi

if [[ "$ARCH" != "arm64" && "$ARCH" != "x64" ]]; then
  echo "ERROR: ARCH must be arm64 or x64, got '${ARCH}'" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi

for cmd in ditto shasum tar "$PYTHON_BIN"; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "ERROR: required command not found on PATH: $cmd" >&2
    exit 1
  fi
done

# Homebrew Python is marked as externally managed (PEP 668) and is not suitable
# as the source for our bundled runtime. Auto-switch to a uv-managed standalone
# Python unless the caller already provided a non-Homebrew interpreter.
if "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import pathlib, sys
ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
marker = pathlib.Path(sys.base_prefix) / "lib" / ver / "EXTERNALLY-MANAGED"
raise SystemExit(0 if marker.is_file() else 1)
PY
then
  echo "==> Detected externally-managed Python at $PYTHON_BIN; switching to uv-managed Python"
  uv python install 3.12
  UV_LIST_JSON="$(mktemp)"
  uv python list --only-installed --managed-python --output-format json > "$UV_LIST_JSON"
  UV_MANAGED_BIN="$("$PYTHON_BIN" - "$UV_LIST_JSON" <<'PY'
import json, pathlib, sys
json_path = pathlib.Path(sys.argv[1])
items = json.loads(json_path.read_text(encoding="utf-8"))
pick = None
for item in items:
    if item.get("implementation") == "cpython" and item.get("arch") == "x86_64":
        ver = item.get("version", "")
        if ver.startswith("3.12"):
            pick = item.get("path")
            break
if not pick and items:
    pick = items[0].get("path")
print(pick or "")
PY
)"
  rm -f "$UV_LIST_JSON"
  if [[ -z "$UV_MANAGED_BIN" || ! -x "$UV_MANAGED_BIN" ]]; then
    echo "ERROR: failed to resolve uv-managed Python 3.12 executable" >&2
    exit 1
  fi
  PYTHON_BIN="$UV_MANAGED_BIN"
fi

PYTHON_VERSION="$("$PYTHON_BIN" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
PYTHON_MACHINE="$("$PYTHON_BIN" - <<'PY'
import platform
print(platform.machine())
PY
)"
case "$PYTHON_VERSION" in
  3.10|3.11|3.12|3.13) ;;
  *)
    echo "ERROR: ${PYTHON_BIN} is Python ${PYTHON_VERSION}; Torch 2.6 runtime builds require Python 3.10-3.13." >&2
    echo "Set PYTHON_BIN=/path/to/python3.12 or PYTHON_BIN=/path/to/python3.11." >&2
    exit 1
    ;;
esac

HOST_ARCH="$(uname -m)"
if [[ "$ARCH" == "arm64" && "$PYTHON_MACHINE" != "arm64" ]]; then
  echo "ERROR: arm64 runtime requires an arm64 Python, got ${PYTHON_MACHINE}" >&2
  exit 1
fi
if [[ "$ARCH" == "x64" && "$PYTHON_MACHINE" != "x86_64" ]]; then
  echo "ERROR: x64 runtime requires an x86_64 Python, got ${PYTHON_MACHINE}" >&2
  exit 1
fi
if [[ "$ARCH" == "x64" && "$HOST_ARCH" == "arm64" ]]; then
  if ! arch -x86_64 /usr/bin/true >/dev/null 2>&1; then
    echo "ERROR: x64 runtime on arm64 hosts requires Rosetta 2." >&2
    exit 1
  fi
fi

rm -rf "$STAGING"
mkdir -p "$PYTHON_DIR" "$BACKEND_DIR" "$BUILD_DIR"

echo "==> Bundling Python installation (${ARCH})"
echo "==> Python: $("$PYTHON_BIN" --version)"
echo "==> Python architecture: ${PYTHON_MACHINE}"

# Get the full PBS (python-build-standalone) installation root.
# This directory has the complete stdlib in lib/pythonX.Y/ — unlike a venv,
# which only creates site-packages/ and relies on the original base_prefix
# (a path that won't exist on user machines) for stdlib.
PYTHON_BASE_PREFIX="$("$PYTHON_BIN" - <<'PY'
import sys
print(sys.base_prefix)
PY
)"
echo "==> Python base prefix: ${PYTHON_BASE_PREFIX}"

if [[ ! -d "${PYTHON_BASE_PREFIX}/lib" ]]; then
  echo "ERROR: Python base prefix has no lib/ dir: ${PYTHON_BASE_PREFIX}" >&2
  echo "  Make sure PYTHON_BIN points to a python-build-standalone (UV) Python." >&2
  exit 1
fi

# Verify the stdlib is actually present in base_prefix before copying.
STDLIB_CHECK="$("$PYTHON_BIN" - <<'PY'
import sys, pathlib
ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
p = pathlib.Path(sys.base_prefix) / "lib" / ver / "encodings" / "__init__.py"
print("ok" if p.is_file() else f"missing:{p}")
PY
)"
if [[ "$STDLIB_CHECK" != "ok" ]]; then
  echo "ERROR: stdlib not found in base_prefix (${STDLIB_CHECK})" >&2
  echo "  base_prefix: ${PYTHON_BASE_PREFIX}" >&2
  exit 1
fi

# Copy the entire PBS Python installation into the runtime bundle.
# ditto preserves symlinks, HFS+ metadata, and extended attributes.
ditto "$PYTHON_BASE_PREFIX" "$PYTHON_DIR"

# PBS Python ships an EXTERNALLY-MANAGED marker that blocks uv from installing
# packages into it. Remove it so we can treat this copy as our own install.
find "$PYTHON_DIR/lib" -name "EXTERNALLY-MANAGED" -delete

# Resolve executable name inside the copied runtime. Homebrew/PBS layouts vary:
# some ship bin/python, others only python3 / python3.x.
PYTHON_RUNTIME_BIN=""
for candidate in "$PYTHON_DIR/bin/python" "$PYTHON_DIR/bin/python3" "$PYTHON_DIR/bin/python${PYTHON_VERSION}"; do
  if [[ -x "$candidate" ]]; then
    PYTHON_RUNTIME_BIN="$candidate"
    break
  fi
done
if [[ -z "$PYTHON_RUNTIME_BIN" ]]; then
  echo "ERROR: no runnable python executable found under $PYTHON_DIR/bin" >&2
  exit 1
fi

echo "==> Installing packages into bundled Python"
# Prefer uv (fast), but fall back to stdlib ensurepip/pip when uv refuses to
# treat the copied runtime as a "system" installation.
if uv pip install --system --python "$PYTHON_RUNTIME_BIN" pip setuptools wheel; then
  uv pip install --system --python "$PYTHON_RUNTIME_BIN" "$REPO_ROOT"
else
  echo "==> uv install path failed; falling back to bundled pip"
  PYTHONHOME="$PYTHON_DIR" "$PYTHON_RUNTIME_BIN" -m ensurepip --upgrade
  PYTHONHOME="$PYTHON_DIR" "$PYTHON_RUNTIME_BIN" -m pip install --upgrade pip setuptools wheel
  PYTHONHOME="$PYTHON_DIR" "$PYTHON_RUNTIME_BIN" -m pip install "$REPO_ROOT"
fi

echo "==> Verifying stdlib and imports"
PYTHON_DIR="$PYTHON_DIR" PYTHONHOME="$PYTHON_DIR" "$PYTHON_RUNTIME_BIN" - <<'PY'
import importlib, os, pathlib, sys

ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
stdlib = pathlib.Path(os.environ["PYTHON_DIR"]) / "lib" / ver
if not (stdlib / "encodings" / "__init__.py").is_file():
    print(f"ERROR: encodings not found in {stdlib}", file=sys.stderr)
    sys.exit(1)
print(f"  stdlib OK at {stdlib}")

packages = [
    "fastapi", "uvicorn", "yt_dlp", "demucs", "torch", "torchaudio",
    "librosa", "pyloudnorm", "soundfile",
]
for package in packages:
    importlib.import_module(package)
    print(f"  OK {package}")
PY

echo "==> Staging backend"
cp -R "$REPO_ROOT/app" "$BACKEND_DIR/app"
cp -R "$REPO_ROOT/static" "$BACKEND_DIR/static"
cp "$REPO_ROOT/pyproject.toml" "$BACKEND_DIR/pyproject.toml"
cp "$REPO_ROOT/uv.lock" "$BACKEND_DIR/uv.lock"

cat > "$BACKEND_DIR/static/version.json" <<JSON
{
  "version": "${VERSION}",
  "arch": "${ARCH}"
}
JSON

echo "==> Capturing dependency inventory"
mkdir -p "$RUNTIME_DIR/licenses"
if ! uv pip list --system --python "$PYTHON_RUNTIME_BIN" --format=json > "$RUNTIME_DIR/licenses/pip-list.json"; then
  PYTHONHOME="$PYTHON_DIR" "$PYTHON_RUNTIME_BIN" -m pip list --format=json > "$RUNTIME_DIR/licenses/pip-list.json"
fi

cat > "$RUNTIME_DIR/runtime-manifest.json" <<JSON
{
  "version": "${VERSION}",
  "arch": "${ARCH}",
  "createdBy": "scripts/macos/make-runtime-pack.sh"
}
JSON

echo "==> Stripping Python caches"
find "$PYTHON_DIR" -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
find "$PYTHON_DIR" -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete

ARCHIVE_NAME="StemDeck-runtime-macOS-${ARCH}.tar.zst"
ARCHIVE_PATH="${BUILD_DIR}/${ARCHIVE_NAME}"
if command -v zstd >/dev/null 2>&1; then
  tar --zstd -cf "$ARCHIVE_PATH" -C "$STAGING" runtime
else
  ARCHIVE_NAME="StemDeck-runtime-macOS-${ARCH}.tar.gz"
  ARCHIVE_PATH="${BUILD_DIR}/${ARCHIVE_NAME}"
  tar -czf "$ARCHIVE_PATH" -C "$STAGING" runtime
fi

SIZE="$(stat -f%z "$ARCHIVE_PATH")"
SHA256="$(shasum -a 256 "$ARCHIVE_PATH" | awk '{print $1}')"
RUNTIME_URL="${RELEASE_BASE_URL}/${ARCHIVE_NAME}"

cat > "${BUILD_DIR}/runtime-manifest-${ARCH}.json" <<JSON
{
  "version": "${VERSION}",
  "arch": "${ARCH}",
  "runtimeUrl": "${RUNTIME_URL}",
  "runtimeSha256": "${SHA256}",
  "runtimeSize": ${SIZE},
  "archiveName": "${ARCHIVE_NAME}"
}
JSON

echo "==> Runtime pack ready"
echo "Archive:  ${ARCHIVE_PATH}"
echo "Size:     ${SIZE}"
echo "SHA256:   ${SHA256}"
echo "Manifest: ${BUILD_DIR}/runtime-manifest-${ARCH}.json"
