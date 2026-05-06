#!/bin/bash
# Launch one Stage-A worker per GPU on THIS machine, each in a detached tmux
# session.  Auto-detects GPUs by default; can be overridden manually.
# Workers from different machines / different GPUs cooperate on the same
# shared NFS chunks/ dir (atomic lockfile claim — see runner script).
#
# Typical use (one command per machine, regardless of GPU count):
#     bash launch_stageA_workers.sh all
#
# Other usage examples:
#     bash launch_stageA_workers.sh all auto             # same as above
#     bash launch_stageA_workers.sh all 0 1 2            # explicit GPU IDs
#     bash launch_stageA_workers.sh bronch 0             # specific cohort, single GPU
#     bash launch_stageA_workers.sh all auto --dry-run   # show plan, don't launch
#     bash launch_stageA_workers.sh all auto --max-chunks 5
#         (each worker exits after 5 chunks — useful for splitting work
#          across heterogeneous machines manually)
#
# Verify:    tmux ls
# Reattach:  tmux attach -t <session_name>
# Kill all:  tmux ls | awk -F: '/^stageA_/ {print $1}' | xargs -n1 tmux kill-session -t

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: launch_stageA_workers.sh <cohort|all> [<gpu_id>... | auto] [flags]

Positional:
  cohort   Cohort name (bronch, veritas, ...) or 'all' to iterate every cohort.
  gpu_ids  Optional list of GPU IDs (0 1 2). If omitted or 'auto', detect via
           nvidia-smi -L.

Flags:
  --dry-run        Print the launch plan, do not start any tmux session.
  --max-chunks N   Each worker stops after processing N chunks (default: until
                   no chunks remain).  Useful for manually capping the share of
                   work a particular machine takes.
  -h | --help      Show this message.
EOF
}

if [ $# -lt 1 ]; then usage; exit 1; fi
case "${1:-}" in -h|--help) usage; exit 0;; esac

COHORT="$1"; shift

# Collect remaining args: numeric GPU ids, 'auto', and flags.
GPUS=()
DRY_RUN=0
MAX_CHUNKS=""
AUTO=0
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)      DRY_RUN=1; shift;;
        --max-chunks)   MAX_CHUNKS="$2"; shift 2;;
        --max-chunks=*) MAX_CHUNKS="${1#*=}"; shift;;
        -h|--help)      usage; exit 0;;
        auto)           AUTO=1; shift;;
        [0-9]*)         GPUS+=("$1"); shift;;
        *)              echo "Unknown arg: $1"; usage; exit 1;;
    esac
done

# If no explicit GPUs given, auto-detect.
if [ "${#GPUS[@]}" -eq 0 ] || [ "${AUTO}" -eq 1 ]; then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "ERROR: nvidia-smi not on PATH; specify GPU ids explicitly."
        exit 1
    fi
    mapfile -t DETECTED < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null)
    if [ "${#DETECTED[@]}" -eq 0 ]; then
        echo "ERROR: nvidia-smi found 0 GPUs on ${HOSTNAME}."
        exit 1
    fi
    if [ "${#GPUS[@]}" -eq 0 ]; then
        GPUS=("${DETECTED[@]}")
    fi
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RUNNER="${SCRIPT_DIR}/run_finetune_stageA_distributed.sh"
if [ ! -x "${RUNNER}" ]; then
    echo "ERROR: runner not executable: ${RUNNER}"
    exit 1
fi

ENV_EXPORT=""
# Pin BLAS to 1 thread per joblib worker by default, otherwise numpy/scipy
# spawn one BLAS thread per core inside each joblib worker, leading to
# severe oversubscription on wide-CPU boxes.  Override only if you know why.
ENV_EXPORT+="export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}; "
ENV_EXPORT+="export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}; "
ENV_EXPORT+="export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}; "
[ -n "${ENVPY:-}"      ] && ENV_EXPORT+="export ENVPY=${ENVPY}; "
[ -n "${MAX_CHUNKS}"   ] && ENV_EXPORT+="export MAX_CHUNKS=${MAX_CHUNKS}; "
[ -n "${N_JOBS:-}"     ] && ENV_EXPORT+="export N_JOBS=${N_JOBS}; "
[ -n "${CHUNK_SIZE:-}" ] && ENV_EXPORT+="export CHUNK_SIZE=${CHUNK_SIZE}; "

# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------
echo "============================================================"
echo "Stage A worker launch plan"
echo "  host:        ${HOSTNAME}"
echo "  cohort:      ${COHORT}"
echo "  gpus:        ${GPUS[*]}    (${#GPUS[@]} workers)"
[ -n "${MAX_CHUNKS}" ] && echo "  max_chunks:  ${MAX_CHUNKS} per worker"
[ -n "${ENVPY:-}"      ] && echo "  envpy:       ${ENVPY}"
echo "  runner:      ${RUNNER}"
echo "  dry-run:     ${DRY_RUN}"
echo "============================================================"

# Show how many chunks remain so the user can sanity-check before launch.
ROOT=/valiant02/masi/zuol1/projects/biodesix/DeepLungScreen/data/finetune_harmonized
CHUNK_ROOT=${ROOT}/chunks
count_glob () {
    # Count files matching a glob, robust to set -e + pipefail when nothing matches.
    local n=0
    for f in $1; do
        [ -e "$f" ] && n=$((n + 1))
    done
    echo "$n"
}

if [ -d "${CHUNK_ROOT}" ] && [ -n "$(ls -A "${CHUNK_ROOT}" 2>/dev/null || true)" ]; then
    echo "Current chunk state on shared NFS:"
    for d in "${CHUNK_ROOT}"/*/; do
        [ -d "${d}" ] || continue
        c=$(basename "${d}")
        if [ "${COHORT}" != "all" ] && [ "${c}" != "${COHORT}" ]; then continue; fi
        total=$(count_glob "${d}chunk_*.csv")
        [ "${total}" -eq 0 ] && continue
        claimed=$(count_glob "${d}chunk_*.lock")
        done=$(count_glob "${d}chunk_*.lock/done")
        printf "  %-12s total=%d  claimed=%d  done=%d  pending=%d\n" \
            "${c}" "${total}" "${claimed}" "${done}" "$((total - claimed))"
    done
else
    echo "(chunks dir not yet created — first worker will populate it)"
fi
echo ""

if [ "${DRY_RUN}" -eq 1 ]; then
    echo "Dry-run: no sessions started."
    exit 0
fi

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
for GPU in "${GPUS[@]}"; do
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
echo "Status:      bash $(dirname "${RUNNER}")/stageA_status.sh"
echo "Logs:        tail -f ${ROOT}/logs/<worker>__<chunk>.log"
