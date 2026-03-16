#!/bin/bash
#SBATCH --job-name=qwen3-ventral-torch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=240G
#SBATCH --time=12:00:00
#SBATCH --mail-user=carsonml@stanford.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=qwen3_ventral_torch_%j.out
#SBATCH --error=qwen3_ventral_torch_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"

if [ -f "./activate_bbscore.sh" ]; then
  source "./activate_bbscore.sh"
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
export RESULTS_PATH="${RESULTS_PATH:-$REPO_ROOT/slurm_runs/qwen3_ventral_torch_${TIMESTAMP}}"
mkdir -p "$RESULTS_PATH"

echo "Repo root: $REPO_ROOT"
echo "Results path: $RESULTS_PATH"
echo "Starting Qwen ventral torch_ridge test at $(date)"

python run.py \
  --model qwen3_vl_8b_img \
  --layer model.visual.blocks.12 model.visual.blocks.18 model.visual.blocks.24 \
  --benchmark NSDVentralShared \
  --metric torch_ridge \
  --batch-size 2 \
  --use-ridge-smart-memory

echo "Finished at $(date)"

