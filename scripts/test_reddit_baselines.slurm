#!/bin/bash
#SBATCH --job-name=test-baselines
#SBATCH --partition=sfscai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/test_baselines_%j.out
#SBATCH --error=logs/test_baselines_%j.err

# Test all six BERT-based baseline models on RedditM.
# Pass the base directory containing each model's run folder as positional args:
#   sbatch test_reddit_baselines.slurm \
#       runs_casflow/reddit_NNN \
#       runs_cascn/reddit_NNN  \
#       runs_ctcp/reddit_NNN   \
#       runs_casdo/reddit_NNN  \
#       runs_casft/reddit_NNN
# Or edit DIRS below directly.

source ~/.bashrc
conda activate T2S

cd /scratch/gw2556/ODE/CasGen
mkdir -p logs
export PYTHONUNBUFFERED=1

BERT_MODEL="pretrained/bert-base-chinese"
BERT_CACHE="dataset/RedditM_bert.pt"
DATA="dataset/RedditM_burst.pt"

# ---- CasFlow ----
CASFLOW_DIR="${1:-runs_casflow/reddit_latest}"
echo "=== CASFLOW  outdir=${CASFLOW_DIR} ==="
python test.py \
    --model CASFLOW \
    --data "${DATA}" \
    --bert_model "${BERT_MODEL}" \
    --bert_device cuda \
    --bert_cache "${BERT_CACHE}" \
    --outdir "${CASFLOW_DIR}" \
    --max_events 500 --batch_size 128 \
    2>&1 | tee "${CASFLOW_DIR}/test.log"

# ---- CasCN ----
CASCN_DIR="${2:-runs_cascn/reddit_latest}"
echo "=== CASCN  outdir=${CASCN_DIR} ==="
python test.py \
    --model CASCN \
    --data "${DATA}" \
    --bert_model "${BERT_MODEL}" \
    --bert_device cuda \
    --bert_cache "${BERT_CACHE}" \
    --outdir "${CASCN_DIR}" \
    --max_events 500 --batch_size 128 \
    2>&1 | tee "${CASCN_DIR}/test.log"

# ---- CTCP ----
CTCP_DIR="${3:-runs_ctcp/reddit_latest}"
echo "=== CTCP  outdir=${CTCP_DIR} ==="
python test.py \
    --model CTCP \
    --data "${DATA}" \
    --bert_model "${BERT_MODEL}" \
    --bert_device cuda \
    --bert_cache "${BERT_CACHE}" \
    --outdir "${CTCP_DIR}" \
    --max_events 500 --batch_size 128 \
    2>&1 | tee "${CTCP_DIR}/test.log"

# ---- CasDO ----
CASDO_DIR="${4:-runs_casdo/reddit_latest}"
echo "=== CASDO  outdir=${CASDO_DIR} ==="
python test.py \
    --model CASDO \
    --data "${DATA}" \
    --bert_model "${BERT_MODEL}" \
    --bert_device cuda \
    --bert_cache "${BERT_CACHE}" \
    --outdir "${CASDO_DIR}" \
    --max_events 500 --batch_size 128 \
    2>&1 | tee "${CASDO_DIR}/test.log"

# ---- CasFT ----
CASFT_DIR="${5:-runs_casft/reddit_latest}"
echo "=== CASFT  outdir=${CASFT_DIR} ==="
python test.py \
    --model CASFT \
    --data "${DATA}" \
    --bert_model "${BERT_MODEL}" \
    --bert_device cuda \
    --bert_cache "${BERT_CACHE}" \
    --outdir "${CASFT_DIR}" \
    --max_events 500 --batch_size 128 \
    2>&1 | tee "${CASFT_DIR}/test.log"

echo "============================================"
echo "  All tests complete."
echo "  Results in each outdir/test_results.json"
echo "============================================"
