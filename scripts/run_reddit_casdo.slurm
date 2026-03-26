#!/bin/bash
#SBATCH --job-name=reddit-casdo
#SBATCH --partition=sfscai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/reddit_casdo_%j.out
#SBATCH --error=logs/reddit_casdo_%j.err

source ~/.bashrc
conda activate T2S

cd /scratch/gw2556/ODE/CasGen
mkdir -p logs
export PYTHONUNBUFFERED=1

RUN_DIR="runs_casdo/reddit_${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"

echo "============================================"
echo "  CasDO on RedditM"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Run dir: ${RUN_DIR}"
echo "============================================"

python train.py \
    --model CASDO \
    --data dataset/RedditM_burst.pt \
    --bert_model pretrained/bert-base-chinese \
    --bert_device cuda \
    --bert_cache dataset/RedditM_bert.pt \
    --outdir "${RUN_DIR}" \
    --z_dim 64 \
    --ode_units 128 \
    --diff_weight 0.1 \
    --dropout 0.1 \
    --max_events 500 \
    --max_steps 30000 \
    --batch_size 64 \
    --lr 1e-3 \
    --eval_every 2000 \
    --n_val_samples 200 \
    2>&1 | tee "${RUN_DIR}/train.log"

echo "============================================"
echo "  Done. Run dir: ${RUN_DIR}"
echo "============================================"
