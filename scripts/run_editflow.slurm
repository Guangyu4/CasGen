#!/bin/bash
#SBATCH --job-name=cas-editflow
#SBATCH --partition=sfscai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/editflow_%j.out
#SBATCH --error=logs/editflow_%j.err

source ~/.bashrc
conda activate T2S

cd /scratch/gw2556/ODE/CasGen
mkdir -p logs runs_editflow
export PYTHONUNBUFFERED=1

RUN_DIR="runs_editflow/${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"

echo "============================================"
echo "  Edit Flow Training"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "============================================"

python train_editflow.py \
    --data dataset/processed.pt \
    --outdir "${RUN_DIR}" \
    --hidden_dim 128 \
    --n_heads 4 \
    --n_layers 4 \
    --n_ins_bins 64 \
    --delta 0.05 \
    --gamma 0.3 \
    --rate_penalty 0.75 \
    --max_events 500 \
    --max_steps 30000 \
    --batch_size 64 \
    --lr 2e-4 \
    --eval_every 2000 \
    --n_sample_steps 100 \
    --n_val_samples 200 \
    2>&1 | tee "${RUN_DIR}/train.log"

echo "============================================"
echo "  Training Complete"
echo "============================================"
