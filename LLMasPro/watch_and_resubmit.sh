#!/bin/bash
# 后台监控 burst 打分任务，任务退出后自动重提，直到所有数据集处理完成
# 用法: bash watch_and_resubmit.sh [WeiboCov RedditM ...]
# 后台运行: nohup bash watch_and_resubmit.sh WeiboCov RedditM > watch.log 2>&1 &

SLURM_SCRIPT=/scratch/gw2556/ODE/CasGen/LLMasPro/run_score_burst.slurm
TEXTS_DIR=/scratch/gw2556/ODE/CasGen/LLMasPro/texts
PYTHON=/scratch/gw2556/ODE/BatchInf/vllm_env/bin/python
CHECK_INTERVAL=60   # seconds between status polls
TARGETS="${*:-WeiboCov RedditM}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

check_progress() {
    # prints "done total" for a given dataset name
    local name=$1
    $PYTHON -c "
import json, sys
try:
    d = json.load(open('$TEXTS_DIR/${name}_texts.json'))
    done = sum(1 for r in d if 'burst_scores' in r)
    print(done, len(d))
except Exception as e:
    print(0, 1, file=sys.stderr)
    print('0 1')
"
}

all_done() {
    local all=1
    for name in $TARGETS; do
        result=$(check_progress "$name")
        done_n=$(echo $result | awk '{print $1}')
        total=$(echo $result | awk '{print $2}')
        log "$name: ${done_n} / ${total} scored"
        [ "${done_n}" -lt "${total}" ] && all=0
    done
    return $((1 - all))
}

JOBID=""

submit() {
    JOBID=$(sbatch $SLURM_SCRIPT $TARGETS 2>&1 | grep -oP '(?<=job )\d+')
    log "Submitted job $JOBID (targets: $TARGETS)"
}

log "=== Watch-and-resubmit started (targets: $TARGETS) ==="

# 检查是否已经全部完成
if all_done; then
    log "All targets already complete. Exiting."
    exit 0
fi

# 检测是否已有同名任务在跑，有则复用
JOB_NAME=$(grep '#SBATCH --job-name' "$SLURM_SCRIPT" | awk -F= '{print $2}' | tr -d ' ')
EXISTING=$(squeue -u "$USER" -h -o '%i %j' 2>/dev/null | awk -v n="$JOB_NAME" '$2==n{print $1}' | head -1)
if [ -n "$EXISTING" ]; then
    JOBID=$EXISTING
    log "Detected existing job $JOBID — attaching to it."
else
    submit
fi

while true; do
    sleep $CHECK_INTERVAL

    if all_done; then
        log "All targets complete!"
        # 如果任务还在跑，取消它
        if [ -n "$JOBID" ] && squeue -j "$JOBID" &>/dev/null; then
            scancel "$JOBID"
            log "Cancelled remaining job $JOBID"
        fi
        log "=== Done ==="
        exit 0
    fi

    # 检查任务是否还在队列中
    if [ -z "$JOBID" ] || ! squeue -j "$JOBID" -h &>/dev/null 2>&1; then
        log "Job $JOBID not found in queue — resubmitting..."
        submit
    else
        state=$(squeue -j "$JOBID" -h -o '%T' 2>/dev/null)
        log "Job $JOBID state: $state"
    fi
done
