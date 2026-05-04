#!/bin/bash
# Stage A on the fine-tuning cohorts (NoduleVU first, then NLST).
# Splits each cohort CSV into fixed-size chunks and runs Steps 0-3 per chunk
# so a crash mid-cohort doesn't lose progress (Steps 1/2/3 all skip existing files).

set -euo pipefail

# ---------------------------------------------------------------------------
# EDIT THESE
# ---------------------------------------------------------------------------
ROOT=/valiant02/masi/zuol1/projects/biodesix/DeepLungScreen/data/finetune

# Per-cohort CSVs (committed under deeplungscreen/cohorts/finetune/).
COHORTS=("nodulevu")

CHUNK_SIZE=2000        # rows per chunk
N_JOBS=8               # parallelism for Step 1 (CPU)

RUN_STEP0=1
RUN_STEP1=1
RUN_STEP2=1
RUN_STEP3=1

# ---------------------------------------------------------------------------
# Derived
# ---------------------------------------------------------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

ORI_ROOT=${ROOT}/nifti
PREP_ROOT=${ROOT}/prep
BBOX_ROOT=${ROOT}/bbox
FEAT64=${ROOT}/feat64
FEAT128=${ROOT}/feat128
CHUNK_DIR=${ROOT}/chunks
LOG_DIR=${ROOT}/logs

mkdir -p "${ORI_ROOT}" "${PREP_ROOT}" "${BBOX_ROOT}" "${FEAT64}" "${FEAT128}" "${CHUNK_DIR}" "${LOG_DIR}"

run_chunk () {
    local CHUNK_CSV=$1
    local TAG=$2
    local LOG=${LOG_DIR}/${TAG}.log

    echo "=========================================================="
    echo "Chunk: ${TAG}   ($(wc -l < "${CHUNK_CSV}") lines incl. header)"
    echo "Log:   ${LOG}"
    echo "=========================================================="

    if [ "${RUN_STEP0}" = "1" ]; then
        echo "  Step 0: symlinks"
        python3 - "${CHUNK_CSV}" "${ORI_ROOT}" >>"${LOG}" 2>&1 <<'PYEOF'
import os, sys, pandas as pd
csv_path, ori_root = sys.argv[1], sys.argv[2]
df = pd.read_csv(csv_path, dtype={'id': str})
made = skipped = missing = 0
miss_examples = []
for _, row in df.iterrows():
    src, sid = row['fpath'], row['id']
    dst = os.path.join(ori_root, f"{sid}.nii.gz")
    if not os.path.exists(src):
        missing += 1
        if len(miss_examples) < 5: miss_examples.append(src)
        continue
    if os.path.islink(dst) or os.path.exists(dst):
        skipped += 1; continue
    os.symlink(src, dst); made += 1
print(f"    created={made} existed={skipped} missing_src={missing}")
for p in miss_examples: print(f"    MISS: {p}")
PYEOF
    fi

    if [ "${RUN_STEP1}" = "1" ]; then
        echo "  Step 1: lung segmentation (n_jobs=${N_JOBS})"
        python3 ./1_preprocess/step1_main.py \
            --sess_csv "${CHUNK_CSV}" \
            --prep_root "${PREP_ROOT}" \
            --ori_root  "${ORI_ROOT}" \
            --n_jobs    "${N_JOBS}" \
            >>"${LOG}" 2>&1
    fi

    if [ "${RUN_STEP2}" = "1" ]; then
        echo "  Step 2: nodule detection (GPU)"
        python3 ./2_nodule_detection/step2_main.py \
            --sess_csv  "${CHUNK_CSV}" \
            --bbox_root "${BBOX_ROOT}" \
            --prep_root "${PREP_ROOT}" \
            >>"${LOG}" 2>&1
    fi

    if [ "${RUN_STEP3}" = "1" ]; then
        echo "  Step 3: feature extraction (GPU)"
        python3 ./3_feature_extraction/step3_main.py \
            --sess_csv  "${CHUNK_CSV}" \
            --bbox_root "${BBOX_ROOT}" \
            --prep_root "${PREP_ROOT}" \
            --feat64    "${FEAT64}" \
            --feat128   "${FEAT128}" \
            >>"${LOG}" 2>&1
    fi

    echo "  Chunk ${TAG} done at $(date)"
}

# ---------------------------------------------------------------------------
# Main loop: per cohort, split into chunks, run sequentially.
# ---------------------------------------------------------------------------
for COHORT in "${COHORTS[@]}"; do
    SRC_CSV=${SCRIPT_DIR}/cohorts/finetune/finetune_stageA_${COHORT}_unharmonized.csv
    if [ ! -f "${SRC_CSV}" ]; then
        echo "SKIP: cohort CSV not found: ${SRC_CSV}"
        continue
    fi

    echo ""
    echo "##########################################################"
    echo "# Cohort: ${COHORT}"
    echo "# Source: ${SRC_CSV}"
    echo "##########################################################"

    # Split into chunks of CHUNK_SIZE rows (header preserved per chunk).
    COHORT_CHUNK_DIR=${CHUNK_DIR}/${COHORT}
    mkdir -p "${COHORT_CHUNK_DIR}"
    python3 - "${SRC_CSV}" "${COHORT_CHUNK_DIR}" "${CHUNK_SIZE}" <<'PYEOF'
import os, sys, pandas as pd
src, dst, sz = sys.argv[1], sys.argv[2], int(sys.argv[3])
df = pd.read_csv(src, dtype={'id': str})
n = len(df); ndigits = max(3, len(str((n + sz - 1) // sz)))
written = 0
for i, start in enumerate(range(0, n, sz)):
    chunk = df.iloc[start:start+sz]
    fname = os.path.join(dst, f"chunk_{i:0{ndigits}d}.csv")
    chunk.to_csv(fname, index=False)
    written += 1
print(f"  split {n} rows -> {written} chunks of <= {sz}")
PYEOF

    # Iterate chunk files.
    for CHUNK_CSV in "${COHORT_CHUNK_DIR}"/chunk_*.csv; do
        TAG=${COHORT}_$(basename "${CHUNK_CSV}" .csv)
        run_chunk "${CHUNK_CSV}" "${TAG}"
    done
done

echo ""
echo "=========================================================="
echo "ALL DONE.  Outputs:"
echo "  prep:    ${PREP_ROOT}"
echo "  bbox:    ${BBOX_ROOT}"
echo "  feat64:  ${FEAT64}"
echo "  feat128: ${FEAT128}"
echo "  logs:    ${LOG_DIR}"
echo "=========================================================="
