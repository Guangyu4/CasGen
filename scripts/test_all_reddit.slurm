#!/bin/bash
#SBATCH --job-name=test-all
#SBATCH --partition=sfscai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/test_all_%j.out
#SBATCH --error=logs/test_all_%j.err

source ~/.bashrc
conda activate T2S

cd /scratch/gw2556/ODE/CasGen
mkdir -p logs
export PYTHONUNBUFFERED=1

BERT_MODEL="pretrained/bert-base-chinese"
BERT_CACHE="dataset/RedditM_bert.pt"
DATA="dataset/RedditM_burst.pt"

echo "============================================"
echo "  Testing all 6 models on RedditM"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "============================================"

# ---- 1. OURS (motive variant) ----
echo ""
echo ">>> [1/6] OURS"
python test.py \
    --model OURS \
    --variant motive \
    --cond_data "${DATA}" \
    --bert_model "${BERT_MODEL}" \
    --outdir runs_editflow/reddit_ours_1953482 \
    --max_events 500 \
    --batch_size 128 \
    --n_sample_steps 100

# ---- 2. CasFlow ----
echo ""
echo ">>> [2/6] CASFLOW"
python test.py \
    --model CASFLOW \
    --data "${DATA}" \
    --bert_model "${BERT_MODEL}" \
    --bert_device cuda \
    --bert_cache "${BERT_CACHE}" \
    --outdir runs_casflow/reddit_1953483 \
    --max_events 500 \
    --batch_size 128

# ---- 3. CasCN ----
echo ""
echo ">>> [3/6] CASCN"
python test.py \
    --model CASCN \
    --data "${DATA}" \
    --bert_model "${BERT_MODEL}" \
    --bert_device cuda \
    --bert_cache "${BERT_CACHE}" \
    --outdir runs_cascn/reddit_1953503 \
    --max_events 500 \
    --batch_size 128

# ---- 4. CTCP ----
echo ""
echo ">>> [4/6] CTCP"
python test.py \
    --model CTCP \
    --data "${DATA}" \
    --bert_model "${BERT_MODEL}" \
    --bert_device cuda \
    --bert_cache "${BERT_CACHE}" \
    --outdir runs_ctcp/reddit_1953505 \
    --max_events 500 \
    --batch_size 128

# ---- 5. CasDO ----
echo ""
echo ">>> [5/6] CASDO"
python test.py \
    --model CASDO \
    --data "${DATA}" \
    --bert_model "${BERT_MODEL}" \
    --bert_device cuda \
    --bert_cache "${BERT_CACHE}" \
    --outdir runs_casdo/reddit_1953507 \
    --max_events 500 \
    --batch_size 128

# ---- 6. CasFT ----
echo ""
echo ">>> [6/6] CASFT"
python test.py \
    --model CASFT \
    --data "${DATA}" \
    --bert_model "${BERT_MODEL}" \
    --bert_device cuda \
    --bert_cache "${BERT_CACHE}" \
    --outdir runs_casft/reddit_1953512 \
    --max_events 500 \
    --batch_size 128

# ---- Summary ----
echo ""
echo "============================================"
echo "  All done. Results:"
echo "============================================"
conda run -n T2S python - <<'PYEOF'
import json, os

models = [
    ("OURS",    "runs_editflow/reddit_ours_1953482"),
    ("CASFLOW", "runs_casflow/reddit_1953483"),
    ("CASCN",   "runs_cascn/reddit_1953503"),
    ("CTCP",    "runs_ctcp/reddit_1953505"),
    ("CASDO",   "runs_casdo/reddit_1953507"),
    ("CASFT",   "runs_casft/reddit_1953512"),
]

print(f"\n{'Model':<10} {'MMD':>10} {'W1(l)':>10} {'W1(t)':>10} {'gen_mean':>10} {'ref_mean':>10}")
print("-" * 62)
for name, outdir in models:
    p = os.path.join(outdir, "test_results.json")
    if os.path.exists(p):
        r = json.load(open(p))
        print(f"{name:<10} {r['mmd']:>10.6f} {r['w1_l']:>10.6f} {r['w1_t']:>10.6f} "
              f"{r.get('gen_mean_len',0):>10.1f} {r.get('ref_mean_len',0):>10.1f}")
    else:
        print(f"{name:<10} {'N/A':>10}")
PYEOF
