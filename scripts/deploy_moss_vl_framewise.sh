#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."
exec python -m moss_native.server \
  --model-path "${MODEL_PATH:?Set MODEL_PATH to the pinned local MOSS-VL-Realtime snapshot}" \
  --trace-dir "${TRACE_DIR:?Set TRACE_DIR to a fresh directory shared with the client}" \
  --model-type moss_vl --torch-dtype bf16 --attn-implementation "${ATTN_IMPL:-sdpa}" \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" "$@"
