#!/bin/bash
#SBATCH --job-name=nsd-batch-captions
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=150G
#SBATCH --time=12:00:00
#SBATCH --mail-user=carsonml@stanford.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=slurm_batch_captions_%j.out
#SBATCH --error=slurm_batch_captions_%j.err

set -euo pipefail

# Optional: override before sbatch (e.g. PROMPT="What objects are in this image?" sbatch slurm_batch_captions_nsd.sbatch)
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"

if [ -f "./activate_bbscore.sh" ]; then
  source "./activate_bbscore.sh"
fi

OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/captions/nsd_qwen3vl}"
PROMPT="${PROMPT:-Describe this image in one sentence.}"
# Start from scratch by default. To resume after a failure: RESUME=1 sbatch slurm_batch_captions_nsd.sbatch
RESUME="${RESUME:-0}"

echo "Repo root: $REPO_ROOT"
echo "Output dir: $OUTPUT_DIR"
echo "Prompt: $PROMPT"
echo "Max new tokens: 750"
echo "Resume: $RESUME (start from scratch). To resume after failure: RESUME=1 sbatch ..."
echo "Starting batch NSD captions at $(date)"

RESUME_ARGS=""
if [ "$RESUME" = "1" ]; then
  RESUME_ARGS="--resume"
fi

python batch_captions_nsd.py \
  --prompt "$PROMPT" \
  --output_dir "$OUTPUT_DIR" \
  --max_new_tokens 750 \
  $RESUME_ARGS

echo "Finished at $(date)"
