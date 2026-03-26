#!/bin/bash
#SBATCH --job-name=cas-ablation
#SBATCH --partition=sfscai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ablation_%x_%j.out
#SBATCH --error=logs/ablation_%x_%j.err

# VARIANT must be set via: sbatch --export=VARIANT=uncond|bert|motive|flat|ddpm ...
VARIANT="${VARIANT:-uncond}"

source ~/.bashrc
conda activate T2S

cd /scratch/gw2556/ODE/CasGen
mkdir -p logs
export PYTHONUNBUFFERED=1

RUN_DIR="runs_ablation/${VARIANT}_${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"

echo "============================================"
echo "  CasGen Ablation: variant=${VARIANT}"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Run dir: ${RUN_DIR}"
echo "============================================"

# ---- Variant-specific extra args ----
EXTRA_ARGS=""

if [ "${VARIANT}" = "ddpm" ]; then
    EXTRA_ARGS="--lambda_hpp 50.0 --n_sample_steps 200"
fi

if [ "${VARIANT}" = "bert" ]; then
    # Use GPU for fast BERT pre-computation; BERT is freed before training starts
    EXTRA_ARGS="--bert_device cuda --bert_model pretrained/bert-base-chinese"
fi

if [ "${VARIANT}" = "bert" ] || [ "${VARIANT}" = "motive" ]; then
    DATA_ARGS="--cond_data dataset/socialnet_cascades_with_motivation.pt"
else
    DATA_ARGS="--data dataset/socialnet_processed.pt"
fi

# ---- Training ----
python train_editflow.py \
    --variant "${VARIANT}" \
    ${DATA_ARGS} \
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
    --n_val_samples 200 \
    ${EXTRA_ARGS} \
    2>&1 | tee "${RUN_DIR}/train.log"

echo "Training complete."

# ---- Evaluation ----
CKPT="${RUN_DIR}/best.pt"
if [ ! -f "${CKPT}" ]; then
    echo "ERROR: checkpoint not found at ${CKPT}"
    exit 1
fi

if [ "${VARIANT}" = "ddpm" ]; then
    N_STEPS=200
else
    N_STEPS=100
fi

python test_editflow.py \
    --ckpt "${CKPT}" \
    --data dataset/socialnet_processed.pt \
    --n_steps "${N_STEPS}" \
    2>&1 | tee "${RUN_DIR}/eval.log"

echo "============================================"
echo "  Done: variant=${VARIANT}"
echo "  Metrics: ${RUN_DIR}/eval/metrics.json"
echo "============================================"
