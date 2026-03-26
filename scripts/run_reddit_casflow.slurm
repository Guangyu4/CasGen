#!/bin/bash
#SBATCH --job-name=reddit-casflow
#SBATCH --partition=sfscai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/reddit_casflow_%j.out
#SBATCH --error=logs/reddit_casflow_%j.err

source ~/.bashrc
conda activate T2S

cd /scratch/gw2556/ODE/CasGen
mkdir -p logs
export PYTHONUNBUFFERED=1

RUN_DIR="runs_casflow/reddit_${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"

echo "============================================"
echo "  CASFLOW on RedditM"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Run dir: ${RUN_DIR}"
echo "============================================"

python train.py \
    --model CASFLOW \
    --data dataset/RedditM_burst.pt \
    --bert_model pretrained/bert-base-chinese \
    --bert_device cuda \
    --bert_cache dataset/RedditM_bert.pt \
    --outdir "${RUN_DIR}" \
    --emb_dim 128 \
    --z_dim 64 \
    --rnn_units 128 \
    --n_flows 8 \
    --kl_weight 0.1 \
    --nf_weight 0.1 \
    --max_events 500 \
    --max_steps 30000 \
    --batch_size 64 \
    --lr 3e-4 \
    --eval_every 2000 \
    --n_val_samples 200 \
    2>&1 | tee "${RUN_DIR}/train.log"

echo "============================================"
echo "  Done. Run dir: ${RUN_DIR}"
echo "============================================"
