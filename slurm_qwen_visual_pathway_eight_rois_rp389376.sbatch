#!/bin/bash
#SBATCH --job-name=qwen3-vis-8roi-rp389376
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=249G
#SBATCH --time=12:00:00
#SBATCH --mail-user=carsonml@stanford.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=qwen3_visual_eight_rois_rp389376_%j.out
#SBATCH --error=qwen3_visual_eight_rois_rp389376_%j.err

set -euo pipefail

# Qwen3-VL **vision pathway** (model.visual.blocks.*): NSD image stimuli, readout from
# visual encoder layers. Dense random projection to match prior runs named like
#   qwen_visual_vs_language_subset_rp389376/results
# RP target dim **389376** is taken from that folder name (leading digit 3).

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"

if [ -f "./activate_bbscore.sh" ]; then
  source "./activate_bbscore.sh"
fi

export SCIKIT_LEARN_DATA="${SCIKIT_LEARN_DATA:-/scratch/users/carsonml/bbscore_data}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
export RESULTS_PATH="${RESULTS_PATH:-$REPO_ROOT/slurm_runs/qwen_visual_pathway_eight_rois_rp389376_${TIMESTAMP}}"
mkdir -p "$RESULTS_PATH"

# Well-spaced vision blocks (0..26): extend prior {12,24} with {6,18} → four layers.
LAYERS=(6 12 18 24)

# Full NSD “shared” set: V1–V4, parietal, lateral, high ventral, high lateral.
BENCHMARKS=(
  NSDV1Shared NSDV2Shared NSDV3Shared NSDV4Shared
  NSDParietalShared NSDLateralShared
  NSDHighVentralShared NSDHighLateralShared
)

echo "Repo root: $REPO_ROOT"
echo "Results path: $RESULTS_PATH"
echo "SCIKIT_LEARN_DATA: $SCIKIT_LEARN_DATA"
echo "Vision layers (blocks): ${LAYERS[*]}"
echo "Random projection: dense -> 389376"
echo "Starting Qwen visual-pathway sweep at $(date)"

python qwen_visual_sweep.py \
  --sweep-mode visual_image \
  --config qwen_sweep_config.json \
  --benchmarks "${BENCHMARKS[@]}" \
  --layer-indices "${LAYERS[@]}" \
  --metric torch_ridge \
  --batch-size 2 \
  --use-ridge-smart-memory \
  --random-projection dense \
  --random-projection-target-dim 389376 \
  --continue-on-error \
  --output-root "$RESULTS_PATH"

echo "Finished at $(date)"
