#!/bin/bash
#SBATCH --job-name=qwen3-joint-language-sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=150G
#SBATCH --time=12:00:00
#SBATCH --mail-user=carsonml@stanford.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=qwen3_joint_language_sweep_%j.out
#SBATCH --error=qwen3_joint_language_sweep_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"

if [ -f "./activate_bbscore.sh" ]; then
  source "./activate_bbscore.sh"
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
export RESULTS_PATH="${RESULTS_PATH:-$REPO_ROOT/slurm_runs/qwen3_joint_language_sweep_${TIMESTAMP}}"
mkdir -p "$RESULTS_PATH"

export NSD_CAPTIONS_PATH="${NSD_CAPTIONS_PATH:-$REPO_ROOT/captions/nsd_qwen3vl/captions_Describe_this_image.jsonl}"

echo "Repo root: $REPO_ROOT"
echo "Results path: $RESULTS_PATH"
echo "Captions path: $NSD_CAPTIONS_PATH"
echo "Starting Qwen joint language sweep at $(date)"

python qwen_joint_language_sweep.py \
  --metric joint_ridge \
  --batch-size 2 \
  --continue-on-error \
  --output-root "$RESULTS_PATH"

echo "Finished at $(date)"
