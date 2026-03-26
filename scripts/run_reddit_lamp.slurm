#!/bin/bash
#SBATCH --job-name=reddit-lamp
#SBATCH --partition=sfscai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/reddit_lamp_%j.out
#SBATCH --error=logs/reddit_lamp_%j.err

source ~/.bashrc
conda activate T2S

cd /scratch/gw2556/ODE/CasGen
mkdir -p logs
export PYTHONUNBUFFERED=1

RUN_DIR="runs_lamp/reddit_${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"

echo "============================================"
echo "  LAMP on RedditM"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Run dir: ${RUN_DIR}"
echo "============================================"

python train.py \
    --model LAMP \
    --data dataset/RedditM_burst.pt \
    --bert_model pretrained/bert-base-chinese \
    --bert_device cuda \
    --bert_cache dataset/RedditM_bert.pt \
    --outdir "${RUN_DIR}" \
    --context_size 64 \
    --mark_emb_size 32 \
    --n_mix 16 \
    --rnn_type GRU \
    --dropout 0.1 \
    --max_events 500 \
    --max_steps 30000 \
    --batch_size 128 \
    --lr 1e-3 \
    --weight_decay 1e-5 \
    --eval_every 1000 \
    --n_val_samples 500 \
    --ebm_hidden 128 \
    --ebm_epochs 30 \
    --n_candidates 16 \
    --llm_weight 0.3 \
    2>&1 | tee "${RUN_DIR}/train.log"

echo "============================================"
echo "  Done. Run dir: ${RUN_DIR}"
echo "============================================"
