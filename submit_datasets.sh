#!/bin/bash
# Submit APS and WeiboCov training jobs simultaneously (one GPU each).
cd "$(dirname "$0")"
mkdir -p logs

JOB1=$(sbatch scripts/run_aps_all.slurm | awk '{print $4}')
JOB2=$(sbatch scripts/run_weibocov_all.slurm | awk '{print $4}')

echo "Submitted APS job:      ${JOB1}  (logs/aps_all_${JOB1}.out)"
echo "Submitted WeiboCov job: ${JOB2}  (logs/weibocov_all_${JOB2}.out)"
