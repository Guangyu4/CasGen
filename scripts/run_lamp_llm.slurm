#!/bin/bash
# Standalone LLM inference step for LAMP, using single GPU.
# Use this to re-run ONLY the LLM step when it fails or needs retrying,
# without re-running IFLTPP or EBM training.
#
# Usage:
#   sbatch scripts/run_lamp_llm.slurm <run_dir> [llm_inp.jsonl] [llm_out.jsonl]
#
# Examples:
#   sbatch scripts/run_lamp_llm.slurm runs_lamp/reddit_1234567
#   sbatch scripts/run_lamp_llm.slurm runs_lamp/reddit_1234567 \
#          runs_lamp/reddit_1234567/llm_inp.jsonl \
#          runs_lamp/reddit_1234567/llm_out.jsonl

#SBATCH --job-name=lamp-llm
#SBATCH --partition=sfscai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/lamp_llm_%j.out
#SBATCH --error=logs/lamp_llm_%j.err

set -e
cd /scratch/gw2556/ODE/CasGen
mkdir -p logs
export PYTHONUNBUFFERED=1

RUN_DIR="${1:?Usage: sbatch run_lamp_llm.slurm <run_dir>}"
INP="${2:-${RUN_DIR}/llm_inp.jsonl}"
OUT="${3:-${RUN_DIR}/llm_out.jsonl}"

VLLM_PY="/scratch/gw2556/ODE/BatchInf/vllm_env/bin/python"
SCRIPT="scripts/lamp_llm_infer.py"

echo "============================================"
echo "  LAMP LLM inference (single GPU)"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Run dir: ${RUN_DIR}"
echo "  Input:   ${INP}"
echo "  Output:  ${OUT}"
echo "============================================"

if [ ! -f "${INP}" ]; then
    echo "ERROR: Input file not found: ${INP}"
    echo "Run the main LAMP training first to generate the LLM input file."
    exit 1
fi

# Delete output to force re-run (remove this line to resume from partial output)
# rm -f "${OUT}"

echo "Starting LLM inference..."
"${VLLM_PY}" "${SCRIPT}" "${INP}" "${OUT}"

echo ""
echo "LLM inference complete."
echo "Output: ${OUT}"
echo "Lines: $(wc -l < "${OUT}")"
echo "============================================"
echo "  To resume LAMP evaluation with this LLM output, run:"
echo "  python test.py --model LAMP --data <dataset> --outdir ${RUN_DIR} \\"
echo "    --llm_out ${OUT} [other args]"
echo "============================================"
