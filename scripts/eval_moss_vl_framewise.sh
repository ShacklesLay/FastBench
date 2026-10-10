#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."
exec python -m moss_native.infer \
  --model-path "${MODEL_PATH:?Set MODEL_PATH to the same pinned snapshot as the server}" \
  --trace-dir "${TRACE_DIR:?Set TRACE_DIR to the matching server trace directory}" \
  --output "${OUTPUT_DIR:?Set OUTPUT_DIR to a new run directory}" \
  --api-base "${API_BASE:-http://127.0.0.1:8000/v1}" \
  --video-root "${VIDEO_ROOT:-$PWD}" "$@"
