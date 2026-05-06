#!/bin/bash
# Launch one Stage-A worker per GPU specified, each in a detached tmux session.
# Run this on each lab machine, naming the GPU IDs available there.
#
# Usage: bash launch_stageA_workers.sh <cohort|all> <gpu_id> [<gpu_id> ...]
# Example (3 GPUs on this machine):
#     bash launch_stageA_workers.sh all 0 1 2
# Example (just GPU 0, just bronch cohort, for a test run):
#     bash launch_stageA_workers.sh bronch 0
#
# Verify: tmux ls   |   reattach: tmux attach -t <session_name>

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <cohort|all> <gpu_id> [<gpu_id> ...]"
    exit 1
fi

COHORT="$1"; shift
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RUNNER="${SCRIPT_DIR}/run_finetune_stageA_distributed.sh"

if [ ! -x "${RUNNER}" ]; then
    echo "ERROR: runner not executable: ${RUNNER}"
    exit 1
fi

# Optional: pin the python env if the user has set ENVPY (kept for parity with
# earlier runs).  When unset, falls back to whatever python3 is on PATH.
ENV_EXPORT=""
if [ -n "${ENVPY:-}" ]; then
    ENV_EXPORT="export ENVPY=${ENVPY}; "
fi

for GPU in "$@"; do
    SESSION="stageA_${COHORT}_${HOSTNAME}_gpu${GPU}"
    if tmux has-session -t "${SESSION}" 2>/dev/null; then
        echo "  ${SESSION}: already running, skipping"
        continue
    fi
    CMD="${ENV_EXPORT}export CUDA_VISIBLE_DEVICES=${GPU}; bash '${RUNNER}' '${COHORT}'"
    tmux new-session -d -s "${SESSION}" "${CMD}"
    echo "  launched ${SESSION}"
done

echo ""
echo "Active stageA sessions on ${HOSTNAME}:"
tmux ls 2>/dev/null | grep '^stageA_' || echo "  (none)"
echo ""
echo "Reattach:    tmux attach -t <session>"
echo "Live logs:   tail -f /valiant02/masi/zuol1/projects/biodesix/DeepLungScreen/data/finetune_harmonized/logs/<worker>__<chunk>.log"
