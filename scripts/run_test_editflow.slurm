#!/bin/bash
#SBATCH --job-name=cas-ef-test
#SBATCH --partition=sfscai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/ef_test_%j.out
#SBATCH --error=logs/ef_test_%j.err

source ~/.bashrc
conda activate T2S

cd /scratch/gw2556/ODE/CasGen
mkdir -p logs
export PYTHONUNBUFFERED=1

CKPT="${1:-runs_editflow/1861864/best.pt}"

echo "============================================"
echo "  Edit Flow Test"
echo "  Checkpoint: ${CKPT}"
echo "============================================"

python test_editflow.py \
    --ckpt "${CKPT}" \
    --data dataset/processed.pt \
    --n_steps 100

echo "============================================"
echo "  Test Complete"
echo "============================================"
