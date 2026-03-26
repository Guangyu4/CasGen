#!/bin/bash
#SBATCH --job-name=aps-all
#SBATCH --partition=sfscai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/aps_all_%j.out
#SBATCH --error=logs/aps_all_%j.err

source ~/.bashrc
conda activate T2S

cd /scratch/gw2556/ODE/CasGen
mkdir -p logs
export PYTHONUNBUFFERED=1

DATA="dataset/APS_burst.pt"
BERT_MODEL="pretrained/bert-base-uncased"
BERT_CACHE="dataset/APS_bert.pt"

echo "============================================"
echo "  Training all 10 models on APS"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "============================================"

# ---- 1. OURS ----
echo ""
echo ">>> [1/10] OURS"
RUN_DIR="runs_editflow/aps_ours_${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"
python train.py \
    --model OURS \
    --variant motive \
    --cond_data "${DATA}" \
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

# ---- 2. CasFlow ----
echo ""
echo ">>> [2/10] CASFLOW"
RUN_DIR="runs_casflow/aps_${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"
python train.py \
    --model CASFLOW \
    --data "${DATA}" \
    --bert_model "${BERT_MODEL}" \
    --bert_device cuda \
    --bert_cache "${BERT_CACHE}" \
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

# ---- 3. CasCN ----
echo ""
echo ">>> [3/10] CASCN"
RUN_DIR="runs_cascn/aps_${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"
python train.py \
    --model CASCN \
    --data "${DATA}" \
    --bert_model "${BERT_MODEL}" \
    --bert_device cuda \
    --bert_cache "${BERT_CACHE}" \
    --outdir "${RUN_DIR}" \
    --hidden_dim 256 \
    --rnn_units 128 \
    --n_time_bins 6 \
    --max_events 500 \
    --max_steps 30000 \
    --batch_size 64 \
    --lr 3e-4 \
    --eval_every 2000 \
    --n_val_samples 200 \
    2>&1 | tee "${RUN_DIR}/train.log"

# ---- 4. CTCP ----
echo ""
echo ">>> [4/10] CTCP"
RUN_DIR="runs_ctcp/aps_${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"
python train.py \
    --model CTCP \
    --data "${DATA}" \
    --bert_model "${BERT_MODEL}" \
    --bert_device cuda \
    --bert_cache "${BERT_CACHE}" \
    --outdir "${RUN_DIR}" \
    --hidden_dim 256 \
    --rnn_units 128 \
    --time_enc_dim 16 \
    --dropout 0.1 \
    --max_events 500 \
    --max_steps 30000 \
    --batch_size 64 \
    --lr 1e-4 \
    --eval_every 2000 \
    --n_val_samples 200 \
    2>&1 | tee "${RUN_DIR}/train.log"

# ---- 5. CasDO ----
echo ""
echo ">>> [5/10] CASDO"
RUN_DIR="runs_casdo/aps_${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"
python train.py \
    --model CASDO \
    --data "${DATA}" \
    --bert_model "${BERT_MODEL}" \
    --bert_device cuda \
    --bert_cache "${BERT_CACHE}" \
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

# ---- 6. CasFT ----
echo ""
echo ">>> [6/10] CASFT"
RUN_DIR="runs_casft/aps_${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"
python train.py \
    --model CASFT \
    --data "${DATA}" \
    --bert_model "${BERT_MODEL}" \
    --bert_device cuda \
    --bert_cache "${BERT_CACHE}" \
    --outdir "${RUN_DIR}" \
    --cond_dim 64 \
    --interval_num 10 \
    --rnn_units 128 \
    --T_diff 200 \
    --sample_T 50 \
    --diff_weight 0.5 \
    --dropout 0.1 \
    --max_events 500 \
    --max_steps 30000 \
    --batch_size 64 \
    --lr 2e-3 \
    --weight_decay 1e-6 \
    --eval_every 2000 \
    --n_val_samples 200 \
    2>&1 | tee "${RUN_DIR}/train.log"

# ---- 7. IFLTPP ----
echo ""
echo ">>> [7/10] IFLTPP"
RUN_DIR="runs_ifltpp/aps_${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"
python train.py \
    --model IFLTPP \
    --data "${DATA}" \
    --bert_model "${BERT_MODEL}" \
    --bert_device cuda \
    --bert_cache "${BERT_CACHE}" \
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
    --n_val_samples 200 \
    2>&1 | tee "${RUN_DIR}/train.log"

# ---- 8. AddThin ----
echo ""
echo ">>> [8/10] ADDTHIN"
RUN_DIR="runs_addthin/aps_${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"
python train.py \
    --model ADDTHIN \
    --data "${DATA}" \
    --bert_model "${BERT_MODEL}" \
    --bert_device cuda \
    --bert_cache "${BERT_CACHE}" \
    --outdir "${RUN_DIR}" \
    --hidden_dim 64 \
    --n_mix 10 \
    --n_cnn_layers 4 \
    --kernel_size 16 \
    --steps 100 \
    --dropout 0.1 \
    --max_events 500 \
    --max_steps 30000 \
    --batch_size 64 \
    --lr 1e-3 \
    --weight_decay 1e-5 \
    --eval_every 1000 \
    --n_val_samples 200 \
    2>&1 | tee "${RUN_DIR}/train.log"

# ---- 9. PSDiff ----
echo ""
echo ">>> [9/10] PSDIFF"
RUN_DIR="runs_psdiff/aps_${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"
python train.py \
    --model PSDIFF \
    --data "${DATA}" \
    --bert_model "${BERT_MODEL}" \
    --bert_device cuda \
    --bert_cache "${BERT_CACHE}" \
    --outdir "${RUN_DIR}" \
    --hidden_dim 64 \
    --n_mix 10 \
    --n_blocks 4 \
    --steps 100 \
    --dropout 0.1 \
    --max_events 500 \
    --max_steps 30000 \
    --batch_size 64 \
    --lr 1e-3 \
    --weight_decay 1e-5 \
    --eval_every 1000 \
    --n_val_samples 200 \
    2>&1 | tee "${RUN_DIR}/train.log"

# ---- 10. EditPP ----
echo ""
echo ">>> [10/10] EDITPP"
RUN_DIR="runs_editpp/aps_${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"
python train.py \
    --model EDITPP \
    --data "${DATA}" \
    --bert_model "${BERT_MODEL}" \
    --bert_device cuda \
    --bert_cache "${BERT_CACHE}" \
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

echo ""
echo "============================================"
echo "  All 10 models done on APS."
echo "  Job ID: ${SLURM_JOB_ID}"
echo "============================================"
