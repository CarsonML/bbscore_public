#!/bin/bash
#SBATCH --job-name=qwen3-v1v2-caption-vs-img
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=175G
#SBATCH --time=12:00:00
#SBATCH --mail-user=carsonml@stanford.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=qwen3_v1v2_caption_vs_img_%j.out
#SBATCH --error=qwen3_v1v2_caption_vs_img_%j.err

set -euo pipefail

# V1/V2 comparison: short captions (text-only pathway) vs NSD images (language_image:
# image + fixed prompt, readout at language layers). Same layer indices for both.
#
# Short captions JSONL: {"index", "caption"} per line (e.g. Describe_this_image).

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"

if [ -f "./activate_bbscore.sh" ]; then
  source "./activate_bbscore.sh"
fi

export SCIKIT_LEARN_DATA="${SCIKIT_LEARN_DATA:-/scratch/users/carsonml/bbscore_data}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BASE="${RESULTS_PATH:-$REPO_ROOT/slurm_runs/qwen_v1v2_short_caption_vs_language_image_${TIMESTAMP}}"
mkdir -p "$BASE"

export NSD_CAPTIONS_PATH="${NSD_CAPTIONS_PATH:-$REPO_ROOT/captions/nsd_qwen3vl/captions_Describe_this_image.jsonl}"

LAYERS=(4 12 24 35)

echo "Repo root: $REPO_ROOT"
echo "Base results: $BASE"
echo "SCIKIT_LEARN_DATA: $SCIKIT_LEARN_DATA"
echo "NSD_CAPTIONS_PATH (caption leg only): $NSD_CAPTIONS_PATH"
echo "Layers: ${LAYERS[*]}"
echo "Starting V1/V2 short-caption vs language-image sweeps at $(date)"

# 1) Short captions only — qwen3_vl_8b_txt, caption benchmarks
export RESULTS_PATH="$BASE/short_caption"
mkdir -p "$RESULTS_PATH"
python qwen_visual_sweep.py \
  --sweep-mode language_caption \
  --config qwen_sweep_config.json \
  --benchmarks NSDV1CaptionShared NSDV2CaptionShared \
  --layer-indices "${LAYERS[@]}" \
  --metric torch_ridge \
  --continue-on-error \
  --output-root "$RESULTS_PATH"

# 2) NSD images — qwen3_vl_8b_img, language layers (language_image)
export RESULTS_PATH="$BASE/language_image"
mkdir -p "$RESULTS_PATH"
python qwen_visual_sweep.py \
  --sweep-mode language_image \
  --config qwen_sweep_config.json \
  --benchmarks NSDV1Shared NSDV2Shared \
  --layer-indices "${LAYERS[@]}" \
  --metric torch_ridge \
  --continue-on-error \
  --output-root "$RESULTS_PATH"

echo "Finished at $(date)"
echo "Results under: $BASE/short_caption and $BASE/language_image"
