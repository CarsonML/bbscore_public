#!/usr/bin/env bash
# Smoke tests: NSD parietal assembly, Qwen language layer 4, three modes.
# Requires: NSD data available via sklearn data home (or SCIKIT_LEARN_DATA),
#           GPU + torch/transformers, and valid caption JSONLs.
#
# Usage:
#   cd /path/to/bbscore_public
#   ./scripts/smoke_qwen_parietal_layer4.sh
#
# Override caption paths if needed:
#   CAPTIONS_SHORT=/path/to/captions_Describe_this_image.jsonl \
#   CAPTIONS_LONG=/path/to/captions_Describe_every_part.jsonl \
#   ./scripts/smoke_qwen_parietal_layer4.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SMOKE_ROOT="${SMOKE_ROOT:-$(mktemp -d "${TMPDIR:-/tmp}/qwen_parietal_layer4_smoke.XXXXXX")}"
# Optional (cluster): export SCIKIT_LEARN_DATA=/scratch/users/carsonml/bbscore_data

CAPTIONS_SHORT="${CAPTIONS_SHORT:-$REPO_ROOT/captions/nsd_qwen3vl/captions_Describe_this_image.jsonl}"
CAPTIONS_LONG="${CAPTIONS_LONG:-$REPO_ROOT/captions/nsd_qwen3vl/captions_Describe_every_part.jsonl}"

LAYER="model.language_model.layers.4"
METRIC="torch_ridge"
BS="4"

echo "Smoke root (RESULTS_PATH parent dirs): $SMOKE_ROOT"
if [[ -n "${SCIKIT_LEARN_DATA:-}" ]]; then echo "SCIKIT_LEARN_DATA=$SCIKIT_LEARN_DATA"; fi

run_one() {
  local name="$1"
  shift
  export RESULTS_PATH="$SMOKE_ROOT/$name"
  mkdir -p "$RESULTS_PATH"
  echo ""
  echo "========== $name =========="
  echo "RESULTS_PATH=$RESULTS_PATH"
  echo "$@"
  "$@"
}

# 1) Caption-only (short) — same benchmark family as long-caption sweeps
if [[ ! -f "$CAPTIONS_SHORT" ]]; then
  echo "WARN: missing CAPTIONS_SHORT=$CAPTIONS_SHORT — skip short-caption task"
else
  export NSD_CAPTIONS_PATH="$CAPTIONS_SHORT"
  run_one "caption_short" \
    python run.py \
      --model qwen3_vl_8b_txt \
      --layer "$LAYER" \
      --benchmark NSDParietalCaptionShared \
      --metric "$METRIC" \
      --batch-size "$BS"
fi

# 2) Long captions (same benchmark ID; different JSONL)
if [[ ! -f "$CAPTIONS_LONG" ]]; then
  echo "WARN: missing CAPTIONS_LONG=$CAPTIONS_LONG — skip long-caption task"
else
  export NSD_CAPTIONS_PATH="$CAPTIONS_LONG"
  run_one "caption_long" \
    python run.py \
      --model qwen3_vl_8b_txt \
      --layer "$LAYER" \
      --benchmark NSDParietalCaptionShared \
      --metric "$METRIC" \
      --batch-size "$BS"
fi

# 3) Multimodal (image + caption, single forward)
if [[ ! -f "$CAPTIONS_SHORT" ]]; then
  echo "WARN: missing CAPTIONS_SHORT — skip multimodal task"
else
  export NSD_CAPTIONS_PATH="$CAPTIONS_SHORT"
  run_one "multimodal_joint" \
    python run.py \
      --model qwen3_vl_8b_joint \
      --layer "$LAYER" \
      --benchmark NSDParietalSharedMultimodal \
      --metric "$METRIC" \
      --batch-size "$BS"
fi

echo ""
echo "Done. Pickles under each $SMOKE_ROOT/*/results/"
