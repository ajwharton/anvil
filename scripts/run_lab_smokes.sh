#!/usr/bin/env bash
# Lab live smokes entrypoint for forge/hammer (cron-friendly). Not used by GitHub CI.
#
# Usage:
#   ./scripts/run_lab_smokes.sh              # profile=nightly
#   ./scripts/run_lab_smokes.sh quick
#   ./scripts/run_lab_smokes.sh full
#
# Cron example (forge, 03:15 local daily):
#   15 3 * * * /mnt/data/anvil/scripts/run_lab_smokes.sh nightly >>/mnt/data/anvil-runs/lab-smokes/cron.log 2>&1

set -euo pipefail

PROFILE="${1:-nightly}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="${ANVIL_VENV:-/mnt/data/anvil-venv}/bin:${PATH}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export ANVIL_OBSERVE_ROOT="${ANVIL_OBSERVE_ROOT:-/mnt/data/anvil-observe}"
export ANVIL_MEDIA_ROOT="${ANVIL_MEDIA_ROOT:-/mnt/data/anvil-media}"
export ANVIL_LAB_SMOKE_DIR="${ANVIL_LAB_SMOKE_DIR:-/mnt/data/anvil-runs/lab-smokes}"
export HF_HOME="${HF_HOME:-/mnt/data/hf_cache}"

mkdir -p "${ANVIL_LAB_SMOKE_DIR}"
cd "${ROOT}"

# Prefer lab tree on NVMe when present
if [[ -d /mnt/data/anvil/.git ]]; then
  cd /mnt/data/anvil
  # non-fatal sync to main so smokes track shipped code
  git fetch origin main 2>/dev/null || true
  git merge --ff-only origin/main 2>/dev/null || true
fi

ENDPOINT="${ANVIL_LAB_ENDPOINT:-local://}"
# quick profile never needs GPU
if [[ "${PROFILE}" == "quick" ]]; then
  ENDPOINT="fake://"
fi

echo "=== lab_smokes $(date -u +%Y-%m-%dT%H:%M:%SZ) profile=${PROFILE} endpoint=${ENDPOINT} ==="
exec python scripts/lab_smokes.py \
  --profile "${PROFILE}" \
  --endpoint "${ENDPOINT}" \
  --model "${ANVIL_LAB_MODEL:-/mnt/data/models/qwen2.5-1.5b-instruct}" \
  --vlm-model "${ANVIL_LAB_VLM:-/mnt/data/models/Qwen2.5-VL-3B-Instruct}" \
  --observe-root "${ANVIL_OBSERVE_ROOT}" \
  --media-root "${ANVIL_MEDIA_ROOT}" \
  --report-dir "${ANVIL_LAB_SMOKE_DIR}"
