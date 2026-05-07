#!/bin/bash
# Distributed Stage A worker.
# - Reads per-cohort CSVs from cohorts/finetune_harmonized/.
# - Each invocation is one worker; multiple workers can run concurrently on the
#   same shared NFS output dir, cooperating via lockfile-based chunk claiming.
# - Pin to a specific GPU by exporting CUDA_VISIBLE_DEVICES before invoking.
#
# Usage:
#     bash run_finetune_stageA_distributed.sh <cohort>           # one cohort
#     bash run_finetune_stageA_distributed.sh all                # all cohorts in COHORTS_ALL
#
# Examples:
#     CUDA_VISIBLE_DEVICES=0 bash run_finetune_stageA_distributed.sh bronch
#     CUDA_VISIBLE_DEVICES=1 bash run_finetune_stageA_distributed.sh all
#
# Steps 1/2/3 are idempotent (they skip existing output files), so a crashed or
# killed worker is safe — re-running will resume.

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT=/valiant02/masi/zuol1/projects/biodesix/DeepLungScreen/data/finetune_harmonized
COHORT_CSV_DIR_DEFAULT=cohorts/finetune_harmonized      # relative to SCRIPT_DIR
COHORTS_ALL=("bronch" "veritas" "reliant1" "reliant2" "1496" "vlsp" "mcl" "nodulevu" "nlst")
# Order = smallest-first so problems surface fast on early cohorts.
# NLST is largest (~65k rows after dropping un-harmonized) and goes last.

CHUNK_SIZE=${CHUNK_SIZE:-2000}     # rows per chunk
N_JOBS=${N_JOBS:-8}                # Step 1 parallelism per worker (CPU)
STALE_HOURS=${STALE_HOURS:-6}      # claim recovered if lock older than this and no done file
MAX_CHUNKS=${MAX_CHUNKS:-0}        # exit after this many chunks (0 = no limit)

RUN_STEP0=${RUN_STEP0:-1}
RUN_STEP1=${RUN_STEP1:-1}
RUN_STEP2=${RUN_STEP2:-1}
RUN_STEP3=${RUN_STEP3:-1}

# ---------------------------------------------------------------------------
# Derived
# ---------------------------------------------------------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

COHORT_CSV_DIR="${SCRIPT_DIR}/${COHORT_CSV_DIR_DEFAULT}"

# Per-cohort output dirs are set inside the cohort loop (see below).
# Only chunks/ and logs/ are shared at ROOT level.
CHUNK_ROOT=${ROOT}/chunks
LOG_DIR=${ROOT}/logs
mkdir -p "${ROOT}" "${CHUNK_ROOT}" "${LOG_DIR}"

# Worker identity (used in lock metadata)
WORKER_ID="${HOSTNAME}_gpu${CUDA_VISIBLE_DEVICES:-cpu}_$$"
echo "Worker: ${WORKER_ID}"
echo "Root:   ${ROOT}"
echo "GPU:    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

# ---------------------------------------------------------------------------
# Cohort selection
# ---------------------------------------------------------------------------
ARG=${1:-}
if [ -z "${ARG}" ]; then
    echo "Usage: $0 <cohort|all>"
    echo "Available cohorts: ${COHORTS_ALL[*]}"
    exit 1
fi
if [ "${ARG}" = "all" ]; then
    SELECTED=("${COHORTS_ALL[@]}")
else
    SELECTED=("${ARG}")
fi

# ---------------------------------------------------------------------------
# Helper: split a cohort CSV into CHUNK_SIZE row chunks (idempotent)
# ---------------------------------------------------------------------------
ensure_chunks () {
    local cohort=$1
    local src_csv="${COHORT_CSV_DIR}/finetune_stageA_harmonized_${cohort}.csv"
    local out_dir="${CHUNK_ROOT}/${cohort}"

    # Special: smoketest synthesizes a 5-row CSV from veritas the first time
    # it's invoked.  Lets you sanity-check the whole pipeline end-to-end fast
    # without polluting any real cohort's outputs.
    if [ "${cohort}" = "smoketest" ] && [ ! -f "${src_csv}" ]; then
        local seed_csv="${COHORT_CSV_DIR}/finetune_stageA_harmonized_veritas.csv"
        if [ ! -f "${seed_csv}" ]; then
            echo "  SKIP smoketest: seed CSV missing at ${seed_csv} (run inventory script first)"
            return 1
        fi
        echo "  Building smoketest cohort from first 5 rows of veritas"
        head -6 "${seed_csv}" > "${src_csv}"
    fi

    if [ ! -f "${src_csv}" ]; then
        echo "  SKIP cohort ${cohort}: no CSV at ${src_csv}"
        return 1
    fi
    mkdir -p "${out_dir}"
    if compgen -G "${out_dir}/chunk_*.csv" > /dev/null; then
        local existing=$(ls "${out_dir}"/chunk_*.csv 2>/dev/null | wc -l)
        echo "  Cohort ${cohort}: ${existing} chunks already split (reusing)"
        return 0
    fi
    echo "  Cohort ${cohort}: splitting ${src_csv} -> ${out_dir}"
    python3 - "${src_csv}" "${out_dir}" "${CHUNK_SIZE}" <<'PYEOF'
import os, sys, pandas as pd
src, dst, sz = sys.argv[1], sys.argv[2], int(sys.argv[3])
df = pd.read_csv(src, dtype={'id': str})
n = len(df); ndigits = max(3, len(str((n + sz - 1) // sz)))
for i, start in enumerate(range(0, n, sz)):
    df.iloc[start:start+sz].to_csv(
        os.path.join(dst, f"chunk_{i:0{ndigits}d}.csv"), index=False)
print(f"    split {n} rows -> {(n + sz - 1)//sz} chunks of <= {sz}")
PYEOF
    return 0
}

# ---------------------------------------------------------------------------
# Helper: claim the next available chunk for a cohort.
# Atomic via mkdir on the .lock directory (POSIX guarantees only one wins).
# Stale locks (older than STALE_HOURS, no done file) are reclaimed.
# Echos the claimed CSV path on stdout, or empty string if nothing left.
# ---------------------------------------------------------------------------
claim_next_chunk () {
    # Echoes the claimed CSV path (or empty) on stdout. ALWAYS exits 0 so that
    # `var=$(claim_next_chunk ...)` doesn't trip `set -e` when nothing's left.
    local cohort=$1
    local out_dir="${CHUNK_ROOT}/${cohort}"
    local stale_min=$((STALE_HOURS * 60))

    for csv in "${out_dir}"/chunk_*.csv; do
        [ -f "${csv}" ] || continue
        local lockdir="${csv%.csv}.lock"

        # Already done — skip permanently.
        [ -f "${lockdir}/done" ] && continue

        # Try fresh claim.
        if mkdir "${lockdir}" 2>/dev/null; then
            echo "${WORKER_ID}" > "${lockdir}/worker.txt"
            date -Iseconds > "${lockdir}/started.txt"
            echo "${csv}"; return 0
        fi

        # Lock exists; check staleness.
        if [ -d "${lockdir}" ] && [ ! -f "${lockdir}/done" ]; then
            local stale=$(find "${lockdir}" -maxdepth 0 -mmin +${stale_min} 2>/dev/null | wc -l)
            if [ "${stale}" -gt 0 ]; then
                echo "  reclaiming stale lock: ${lockdir}" >&2
                rm -rf "${lockdir}" 2>/dev/null || true
                if mkdir "${lockdir}" 2>/dev/null; then
                    echo "${WORKER_ID}" > "${lockdir}/worker.txt"
                    date -Iseconds > "${lockdir}/started.txt"
                    echo "${csv}"; return 0
                fi
            fi
        fi
    done
    # No chunk available — emit empty string and return success so that
    # `set -e` doesn't bring the whole script down.
    echo ""
    return 0
}

# ---------------------------------------------------------------------------
# Helper: run all 4 steps on a single chunk.
# ---------------------------------------------------------------------------
process_chunk () {
    local CHUNK_CSV=$1
    local TAG=$2
    local LOG=${LOG_DIR}/${WORKER_ID}__${TAG}.log

    echo "=========================================================="
    echo "Worker ${WORKER_ID}  Chunk: ${TAG}"
    echo "  CSV: ${CHUNK_CSV}"
    echo "  Log: ${LOG}"
    echo "=========================================================="

    if [ "${RUN_STEP0}" = "1" ]; then
        echo "  Step 0: symlinks"
        python3 - "${CHUNK_CSV}" "${ORI_ROOT}" >>"${LOG}" 2>&1 <<'PYEOF'
import os, sys, pandas as pd
csv_path, ori_root = sys.argv[1], sys.argv[2]
df = pd.read_csv(csv_path, dtype={'id': str})
made = skipped = missing = 0
for _, row in df.iterrows():
    src, sid = row['fpath'], row['id']
    dst = os.path.join(ori_root, f"{sid}.nii.gz")
    if not os.path.exists(src):
        missing += 1; continue
    if os.path.islink(dst) or os.path.exists(dst):
        skipped += 1; continue
    try:
        os.symlink(src, dst); made += 1
    except FileExistsError:
        skipped += 1
print(f"    created={made} existed={skipped} missing_src={missing}")
PYEOF
    fi

    if [ "${RUN_STEP1}" = "1" ]; then
        echo "  Step 1: lung segmentation (n_jobs=${N_JOBS})"
        python3 ./1_preprocess/step1_main.py \
            --sess_csv  "${CHUNK_CSV}" \
            --prep_root "${PREP_ROOT}" \
            --ori_root  "${ORI_ROOT}" \
            --n_jobs    "${N_JOBS}" \
            >>"${LOG}" 2>&1
    fi

    # Filter the chunk CSV down to IDs that actually have {id}_clean.npy in prep/.
    # Step 1 silently drops IDs whose lung segmentation fails ("Unable to segment
    # image"); Step 2 would otherwise crash on the first missing _clean.npy.
    # Survivor CSVs live INSIDE the lock dir so they don't match chunk_*.csv globs.
    local LOCKDIR="${CHUNK_CSV%.csv}.lock"
    local CHUNK_CSV_OK="${LOCKDIR}/step1ok.csv"
    echo "  Filter: keep only IDs with {id}_clean.npy in ${PREP_ROOT}"
    python3 - "${CHUNK_CSV}" "${PREP_ROOT}" "${CHUNK_CSV_OK}" >>"${LOG}" 2>&1 <<'PYEOF'
import os, sys, pandas as pd
src, prep_root, dst = sys.argv[1], sys.argv[2], sys.argv[3]
df = pd.read_csv(src, dtype={'id': str})
keep = df['id'].apply(lambda i: os.path.isfile(os.path.join(prep_root, f"{i}_clean.npy")))
df_ok = df[keep]
df_ok.to_csv(dst, index=False)
print(f"    step1ok filter: {len(df_ok)}/{len(df)} kept", flush=True)
PYEOF
    local NUM_OK
    NUM_OK=$(($(wc -l < "${CHUNK_CSV_OK}") - 1))
    echo "    -> ${NUM_OK} IDs survived Step 1"

    if [ "${NUM_OK}" -le 0 ]; then
        echo "  WARN: 0 IDs left after Step 1; skipping Steps 2/3 for this chunk"
        return 0
    fi

    if [ "${RUN_STEP2}" = "1" ]; then
        echo "  Step 2: nodule detection (GPU)"
        python3 ./2_nodule_detection/step2_main.py \
            --sess_csv  "${CHUNK_CSV_OK}" \
            --bbox_root "${BBOX_ROOT}" \
            --prep_root "${PREP_ROOT}" \
            >>"${LOG}" 2>&1
    fi

    # Same defensive filter between Steps 2 and 3 — Step 2 can also produce no
    # bbox for some IDs (rare, but possible).  Use _pbb.npy as the survival signal.
    local CHUNK_CSV_OK2="${LOCKDIR}/step2ok.csv"
    python3 - "${CHUNK_CSV_OK}" "${BBOX_ROOT}" "${CHUNK_CSV_OK2}" >>"${LOG}" 2>&1 <<'PYEOF'
import os, sys, pandas as pd
src, bbox_root, dst = sys.argv[1], sys.argv[2], sys.argv[3]
df = pd.read_csv(src, dtype={'id': str})
keep = df['id'].apply(lambda i: os.path.isfile(os.path.join(bbox_root, f"{i}_pbb.npy")))
df_ok = df[keep]
df_ok.to_csv(dst, index=False)
print(f"    step2ok filter: {len(df_ok)}/{len(df)} kept", flush=True)
PYEOF
    local NUM_OK2
    NUM_OK2=$(($(wc -l < "${CHUNK_CSV_OK2}") - 1))
    echo "    -> ${NUM_OK2} IDs survived Step 2"

    if [ "${NUM_OK2}" -le 0 ]; then
        echo "  WARN: 0 IDs left after Step 2; skipping Step 3 for this chunk"
        return 0
    fi

    if [ "${RUN_STEP3}" = "1" ]; then
        echo "  Step 3: feature extraction (GPU)"
        python3 ./3_feature_extraction/step3_main.py \
            --sess_csv  "${CHUNK_CSV_OK2}" \
            --bbox_root "${BBOX_ROOT}" \
            --prep_root "${PREP_ROOT}" \
            --feat64    "${FEAT64}" \
            --feat128   "${FEAT128}" \
            >>"${LOG}" 2>&1
    fi
}

# ---------------------------------------------------------------------------
# Main loop: per cohort, claim and process chunks until none remain.
# ---------------------------------------------------------------------------
for COHORT in "${SELECTED[@]}"; do
    echo ""
    echo "##########################################################"
    echo "# Cohort: ${COHORT}    Worker: ${WORKER_ID}"
    echo "##########################################################"

    # Per-cohort output dirs (used by process_chunk via dynamic scoping).
    ORI_ROOT=${ROOT}/${COHORT}/nifti
    PREP_ROOT=${ROOT}/${COHORT}/prep
    BBOX_ROOT=${ROOT}/${COHORT}/bbox
    FEAT64=${ROOT}/${COHORT}/feat64
    FEAT128=${ROOT}/${COHORT}/feat128
    mkdir -p "${ORI_ROOT}" "${PREP_ROOT}" "${BBOX_ROOT}" "${FEAT64}" "${FEAT128}"

    if ! ensure_chunks "${COHORT}"; then
        continue
    fi

    processed=0
    while true; do
        if [ "${MAX_CHUNKS}" -gt 0 ] && [ "${processed}" -ge "${MAX_CHUNKS}" ]; then
            echo "  Worker ${WORKER_ID} reached MAX_CHUNKS=${MAX_CHUNKS}; exiting cohort ${COHORT}"
            break
        fi
        CHUNK_CSV=$(claim_next_chunk "${COHORT}")
        if [ -z "${CHUNK_CSV}" ]; then
            echo "  No more chunks in cohort ${COHORT} for worker ${WORKER_ID}"
            break
        fi
        TAG=${COHORT}_$(basename "${CHUNK_CSV}" .csv)
        process_chunk "${CHUNK_CSV}" "${TAG}"
        # Mark this chunk done so future workers / re-runs skip it.
        touch "${CHUNK_CSV%.csv}.lock/done"
        processed=$((processed + 1))
        echo "  Chunk ${TAG} done at $(date) (worker total: ${processed})"
    done
done

echo ""
echo "Worker ${WORKER_ID} exiting."
