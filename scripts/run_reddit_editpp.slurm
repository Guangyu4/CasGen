#!/bin/bash
#SBATCH --job-name=reddit-editpp
#SBATCH --partition=sfscai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/reddit_editpp_%j.out
#SBATCH --error=logs/reddit_editpp_%j.err

source ~/.bashrc
conda activate T2S

cd /scratch/gw2556/ODE/CasGen
mkdir -p logs
export PYTHONUNBUFFERED=1

RUN_DIR="runs_editpp/reddit_${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"

echo "============================================"
echo "  EditFlows on RedditM"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Run dir: ${RUN_DIR}"
echo "============================================"

python train.py \
    --model EDITPP \
    --data dataset/RedditM_burst.pt \
    --bert_model pretrained/bert-base-chinese \
    --bert_device cuda \
    --bert_cache dataset/RedditM_bert.pt \
    --outdir "${RUN_DIR}" \
    --hidden_dim 128 \
    --n_heads 4 \
    --n_layers 4 \
    --n_ins_bins 32 \
    --dropout 0.1 \
    --n_euler_steps 50 \
    --max_events 100 \
    --max_steps 30000 \
    --batch_size 64 \
    --lr 1e-3 \
    --weight_decay 1e-5 \
    --eval_every 1000 \
    --n_val_samples 200 \
    2>&1 | tee "${RUN_DIR}/train.log"

echo "============================================"
echo "  Done. Run dir: ${RUN_DIR}"
echo "============================================"
