#!/bin/bash
# Submit all ablation variants and print job IDs.
# Usage: bash submit_ablation.sh

set -e
cd "$(dirname "$0")"
mkdir -p logs runs_ablation

# Already completed (runs_ablation/full_1862940 → uncond baseline)
# JOB_UNCOND=$(sbatch --job-name=cas-uncond --export=VARIANT=uncond  run_ablation_variant.slurm | awk '{print $NF}')
# Already completed (runs_ablation/flat_1862941)
# JOB_FLAT=$(  sbatch --job-name=cas-flat   --export=VARIANT=flat    run_ablation_variant.slurm | awk '{print $NF}')
# Already completed (runs_ablation/ddpm_1862942)
# JOB_DDPM=$(  sbatch --job-name=cas-ddpm   --export=VARIANT=ddpm    run_ablation_variant.slurm | awk '{print $NF}')

JOB_BERT=$(  sbatch --job-name=cas-bert   --export=VARIANT=bert    run_ablation_variant.slurm | awk '{print $NF}')
JOB_MOTIVE=$(sbatch --job-name=cas-motive --export=VARIANT=motive  run_ablation_variant.slurm | awk '{print $NF}')

echo "Submitted jobs:"
echo "  bert   : ${JOB_BERT}"
echo "  motive : ${JOB_MOTIVE}"

echo "${JOB_BERT} ${JOB_MOTIVE}" > runs_ablation/job_ids.txt
echo "Job IDs saved to runs_ablation/job_ids.txt"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Summary: python summarize_ablation.py runs_ablation/"
