#!/bin/bash
#SBATCH --job-name=burst-score
#SBATCH --partition=sfscai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:a800:4
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/gw2556/ODE/CasGen/logs/score_burst_%j.out
#SBATCH --error=/scratch/gw2556/ODE/CasGen/logs/score_burst_%j.err
#SBATCH --open-mode=append

set -e
module purge 2>/dev/null || true

export PYTHONUNBUFFERED=1

CASGEN=/scratch/gw2556/ODE/CasGen
PYTHON=/scratch/gw2556/ODE/BatchInf/vllm_env/bin/python

mkdir -p "$CASGEN/logs"

echo "=== Job started:  $(date) ==="
echo "=== Node: $SLURMD_NODENAME | JobID: $SLURM_JOB_ID ==="
echo "=== Targets: ${*:-APS WeiboCov RedditM} ==="

$PYTHON -u "$CASGEN/LLMasPro/extract_texts.py" "$@"

echo "=== Job finished: $(date) ==="
