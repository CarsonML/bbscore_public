#!/bin/bash
#SBATCH --job-name=qwen3-visual-sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --mail-user=carsonml@stanford.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=qwen3_visual_sweep_%j.out
#SBATCH --error=qwen3_visual_sweep_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"

if [ -f "./activate_bbscore.sh" ]; then
  source "./activate_bbscore.sh"
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
export RESULTS_PATH="${RESULTS_PATH:-$REPO_ROOT/slurm_runs/qwen3_visual_sweep_${TIMESTAMP}}"
mkdir -p "$RESULTS_PATH"

echo "Repo root: $REPO_ROOT"
echo "Results path: $RESULTS_PATH"
echo "Starting Qwen visual sweep at $(date)"

python qwen_visual_sweep.py \
  --model qwen3_vl_8b_img \
  --benchmarks \
    NSDV1Shared NSDV2Shared NSDV3Shared NSDV4Shared \
    NSDLateralShared NSDVentralShared NSDParietalShared \
    NSDHighLateralShared NSDHighVentralShared NSDHighParietalShared \
    NSDMidLateralShared NSDMidVentralShared NSDMidParietalShared \
  --metric ridge \
  --batch-size 2 \
  --step 6 \
  --use-ridge-smart-memory \
  --continue-on-error \
  --output-root "$RESULTS_PATH"

echo "Finished at $(date)"
