#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
NEST_CONFIG="${NEST_CONFIG:-$(command -v nest-config)}"

if [[ -z "${NEST_CONFIG}" ]]; then
  echo "nest-config not found. Set NEST_CONFIG or add nest-config to PATH." >&2
  exit 1
fi

NEST_COMPILER="$("${NEST_CONFIG}" --compiler)"
if [[ -z "${NEST_COMPILER}" ]]; then
  echo "Could not determine the C++ compiler from ${NEST_CONFIG} --compiler." >&2
  exit 1
fi
if [[ ! -x "${NEST_COMPILER}" ]]; then
  echo "The compiler reported by ${NEST_CONFIG} does not exist or is not executable: ${NEST_COMPILER}" >&2
  exit 1
fi

# CMake caches absolute source/build paths. If the same checkout is reached via
# a different path (for example a symlink or a moved workspace), reconfigure
# from a clean build directory instead of failing with a cache mismatch.
if [[ -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
  CACHED_SOURCE_DIR="$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "${BUILD_DIR}/CMakeCache.txt" | head -n 1)"
  CACHED_CXX_COMPILER="$(sed -n 's/^CMAKE_CXX_COMPILER:FILEPATH=//p' "${BUILD_DIR}/CMakeCache.txt" | head -n 1)"
  if [[ -z "${CACHED_CXX_COMPILER}" ]]; then
    CACHED_COMPILER_FILE="$(find "${BUILD_DIR}/CMakeFiles" -path '*/CMakeCXXCompiler.cmake' -type f 2>/dev/null | head -n 1)"
    if [[ -n "${CACHED_COMPILER_FILE}" ]]; then
      CACHED_CXX_COMPILER="$(sed -n 's/^set(CMAKE_CXX_COMPILER \"\\(.*\\)\")/\\1/p' "${CACHED_COMPILER_FILE}" | head -n 1)"
    fi
  fi
  if [[ -n "${CACHED_SOURCE_DIR}" && "${CACHED_SOURCE_DIR}" != "${ROOT_DIR}" ]]; then
    echo "Removing stale build cache from ${CACHED_SOURCE_DIR}" >&2
    rm -rf "${BUILD_DIR}"
  elif [[ -n "${CACHED_CXX_COMPILER}" && "${CACHED_CXX_COMPILER}" != "${NEST_COMPILER}" ]]; then
    echo "Removing build cache configured with ${CACHED_CXX_COMPILER}; expected ${NEST_COMPILER}" >&2
    rm -rf "${BUILD_DIR}"
  fi
fi

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
  -Dwith-nest="${NEST_CONFIG}" \
  -DCMAKE_CXX_COMPILER="${NEST_COMPILER}"
cmake --build "${BUILD_DIR}"

INSTALLED=1
if ! cmake --install "${BUILD_DIR}"; then
  echo "Install into the active NEST environment failed; continuing with local build verification." >&2
  INSTALLED=0
fi

INSTALLED="${INSTALLED}" BUILD_DIR="${BUILD_DIR}" python - <<'PY'
import os
from pathlib import Path

import nest

nest.ResetKernel()

installed = os.environ["INSTALLED"] == "1"
build_dir = Path(os.environ["BUILD_DIR"])

if installed:
    nest.Install("cavallari_module")
else:
    for candidate in (
        build_dir / "src" / "cavallari_module.so",
        build_dir / "cavallari_module.so",
        build_dir / "libcavallari_module.so",
    ):
        if candidate.exists():
            nest.Install(str(candidate))
            break
    else:
        raise RuntimeError("Built module not found in build directory.")

if "iaf_bw_2003" not in nest.node_models:
    raise RuntimeError("iaf_bw_2003 was not registered after installing cavallari_module.")

if installed:
    print("Built, installed, and verified: cavallari_module")
else:
    print("Built and locally verified without installation: cavallari_module")
PY
