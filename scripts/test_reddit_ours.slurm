#!/bin/bash
#SBATCH --job-name=test-ours
#SBATCH --partition=sfscai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/test_ours_%j.out
#SBATCH --error=logs/test_ours_%j.err

source ~/.bashrc
conda activate T2S

cd /scratch/gw2556/ODE/CasGen
mkdir -p logs
export PYTHONUNBUFFERED=1

# Set OUTDIR to the run directory that contains best.pt
OUTDIR="${1:-runs_ours/reddit_latest}"

echo "============================================"
echo "  TEST: OURS on RedditM"
echo "  Outdir: ${OUTDIR}"
echo "============================================"

python test.py \
    --model OURS \
    --variant motive \
    --cond_data dataset/RedditM_burst.pt \
    --outdir "${OUTDIR}" \
    --max_events 500 \
    --batch_size 128 \
    --n_sample_steps 100 \
    2>&1 | tee "${OUTDIR}/test.log"

echo "============================================"
echo "  Done. Results in ${OUTDIR}/test_results.json"
echo "============================================"
