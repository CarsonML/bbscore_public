#!/bin/bash
#SBATCH --job-name=qwen3-caption-long-sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=175G
#SBATCH --time=12:00:00
#SBATCH --mail-user=carsonml@stanford.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=qwen3_caption_long_sweep_%j.out
#SBATCH --error=qwen3_caption_long_sweep_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"

if [ -f "./activate_bbscore.sh" ]; then
  source "./activate_bbscore.sh"
fi

export SCIKIT_LEARN_DATA="${SCIKIT_LEARN_DATA:-/scratch/users/carsonml/bbscore_data}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
export RESULTS_PATH="${RESULTS_PATH:-$REPO_ROOT/slurm_runs/qwen3_caption_long_sweep_${TIMESTAMP}}"
mkdir -p "$RESULTS_PATH"

export NSD_CAPTIONS_PATH="${NSD_CAPTIONS_PATH:-$REPO_ROOT/captions/nsd_qwen3vl/captions_Describe_every_part.jsonl}"

echo "Repo root: $REPO_ROOT"
echo "Results path: $RESULTS_PATH"
echo "SCIKIT_LEARN_DATA: $SCIKIT_LEARN_DATA"
echo "NSD_CAPTIONS_PATH: $NSD_CAPTIONS_PATH"
echo "Starting Qwen long-caption sweep at $(date)"

python qwen_visual_sweep.py \
  --sweep-mode language_caption_long \
  --config qwen_sweep_config.json \
  --metric torch_ridge \
  --continue-on-error \
  --output-root "$RESULTS_PATH"

echo "Finished at $(date)"
