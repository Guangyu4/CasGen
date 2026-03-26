#!/bin/bash
#SBATCH --job-name=aps-langtpp
#SBATCH --partition=sfscai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/aps_langtpp_%j.out
#SBATCH --error=logs/aps_langtpp_%j.err

source ~/.bashrc
conda activate T2S

cd /scratch/gw2556/ODE/CasGen
mkdir -p logs
export PYTHONUNBUFFERED=1

if [ ! -f "pretrained/Qwen2.5-0.5B/config.json" ]; then
    echo "=== Downloading Qwen2.5-0.5B weights ==="
    bash scripts/download_qwen.sh hfmirror
fi

RUN_DIR="runs_langtpp/aps_${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"

echo "============================================"
echo "  LangTPP on APS"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Run dir: ${RUN_DIR}"
echo "============================================"

python train.py \
    --model LANGTPP \
    --data dataset/APS_burst.pt \
    --qwen_model pretrained/Qwen2.5-0.5B \
    --outdir "${RUN_DIR}" \
    --max_events 150 \
    --max_text_tokens 64 \
    --max_seq_len 1152 \
    --max_steps 500 \
    --batch_size 1 \
    --grad_accum 32 \
    --lr 2e-5 \
    --warmup_steps 50 \
    --eval_every 100 \
    --n_val_samples 200 \
    --gradient_checkpointing \
    2>&1 | tee "${RUN_DIR}/train.log"

echo "============================================"
echo "  Done. Run dir: ${RUN_DIR}"
echo "============================================"
