#!/usr/bin/env bash
# Overnight J-Lens fit+solve on forge for Qwen2.5-7B-Instruct (mixed WikiText+math).
#
# Spike is **parked** (2026-07-19) — this script is the resume recipe, not a
# scheduled product path. See docs/spikes/jlens-math.md §Fifth pass.
#
# Prerequisites (forge anvil-venv):
#   pip install 'git+https://github.com/anthropics/jacobian-lens.git'
#   pip install 'datasets>=2.14'   # required for --fit-corpus mixed (WikiText-103)
# Without `datasets`, mixed silently falls back to math-only prompts.
#
# Usage (on forge)::
#
#   source /mnt/data/anvil-venv/bin/activate
#   cd /mnt/data/anvil && git pull
#   nohup bash scripts/run_jlens_7b_mixed.sh &
#   tail -f /mnt/data/anvil/results/jlens-solve-7b-mixed/run.log
#
# Env overrides: MODEL, LENS_DIR, LENS, OUT, ANVIL_VENV, FIT_N, DIM_BATCH

set -euo pipefail

ANVIL_ROOT="${ANVIL_ROOT:-/mnt/data/anvil}"
ANVIL_VENV="${ANVIL_VENV:-/mnt/data/anvil-venv}"
MODEL="${MODEL:-/mnt/data/models/qwen2.5-7b-instruct}"
LENS_DIR="${LENS_DIR:-/mnt/data/models/lenses/qwen2.5-7b-instruct-v0}"
LENS="${LENS:-$LENS_DIR/jacobian_lens.pt}"
OUT="${OUT:-/mnt/data/anvil/results/jlens-solve-7b-mixed}"
FIT_N="${FIT_N:-128}"
DIM_BATCH="${DIM_BATCH:-64}"

# shellcheck source=/dev/null
source "$ANVIL_VENV/bin/activate"
cd "$ANVIL_ROOT"
mkdir -p "$LENS_DIR" "$OUT"
export PYTHONUNBUFFERED=1
LOG="$OUT/run.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== $(date -Is) START 7B mixed fit+solve ==="
echo "model=$MODEL"
echo "lens=$LENS"
echo "out=$OUT"
echo "corpus=mixed (wikitext + math), fit_n=$FIT_N, dim_batch=$DIM_BATCH, bf16"

if ! python -c "import datasets" 2>/dev/null; then
  echo "ERROR: Python package 'datasets' is required for --fit-corpus mixed (WikiText)."
  echo "  pip install 'datasets>=2.14'"
  exit 1
fi
python -c "import datasets; print('datasets', getattr(datasets, '__version__', '?'))"
if ! python -c "import jlens" 2>/dev/null; then
  echo "ERROR: jlens not installed. pip install 'git+https://github.com/anthropics/jacobian-lens.git'"
  exit 1
fi

if [ ! -f "$MODEL/config.json" ]; then
  echo "MODEL MISSING: $MODEL"
  echo "  Pull with: python scripts/pull_base_model.py --repo Qwen/Qwen2.5-7B-Instruct  # or local snapshot path"
  exit 1
fi

if [ ! -f "$LENS" ]; then
  echo "=== $(date -Is) FIT begin ==="
  python scripts/jlens_spike.py fit \
    --model-path "$MODEL" \
    --lens-path "$LENS" \
    --device cuda \
    --dtype bf16 \
    --fit-n "$FIT_N" \
    --fit-corpus mixed \
    --dim-batch "$DIM_BATCH" \
    --seq-len 128
  echo "=== $(date -Is) FIT end ==="
  ls -lh "$LENS" "$LENS_DIR"/*.meta.json 2>/dev/null || true
else
  echo "=== $(date -Is) using existing lens $LENS ==="
fi

echo "=== $(date -Is) APPLY solve begin ==="
python scripts/jlens_spike.py apply \
  --model-path "$MODEL" \
  --lens-path "$LENS" \
  --device cuda \
  --dtype bf16 \
  --protocol solve \
  --inter-score-mode boundary \
  --max-new-tokens 64 \
  --out "$OUT"
echo "=== $(date -Is) DONE exit=0 ==="
