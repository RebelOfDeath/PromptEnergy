#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-config/experiment.yaml}"
OUTPUT_DIR="${2:-}"

ARGS=(--config "${CONFIG_PATH}")
if [[ -n "${OUTPUT_DIR}" ]]; then
  ARGS+=(--output "${OUTPUT_DIR}")
fi

python3 scripts/run_eval.py "${ARGS[@]}"
