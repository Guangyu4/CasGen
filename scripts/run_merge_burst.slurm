#!/bin/bash
#SBATCH --job-name=merge-burst
#SBATCH --partition=sfscai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a800:1
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/gw2556/ODE/CasGen/logs/merge_burst_%j.out
#SBATCH --error=/scratch/gw2556/ODE/CasGen/logs/merge_burst_%j.err

set -e
export PYTHONUNBUFFERED=1

CASGEN=/scratch/gw2556/ODE/CasGen
SCRIPT=$CASGEN/LLMasPro

echo "=== merge-burst started: $(date) ==="
echo "=== Node: $SLURMD_NODENAME | JobID: $SLURM_JOB_ID ==="

# Step 1: validate all records (full scan, no sampling)
echo "--- Step 1: validate ---"
conda run -n T2S python "$SCRIPT/validate_burst.py" APS WeiboCov RedditM

# Step 2: merge into pt files (missing scores → zeros)
echo "--- Step 2: merge ---"
conda run -n T2S python "$SCRIPT/merge_burst.py" APS WeiboCov RedditM

echo "=== merge-burst finished: $(date) ==="
