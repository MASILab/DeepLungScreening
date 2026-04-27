#!/bin/bash
# Stage A driver: NIfTI CT -> nodule features (.npy)
# Edit the paths below for your cohort, then run from the DeepLungScreen root.

set -euo pipefail

# ---------------------------------------------------------------------------
# EDIT THESE
# ---------------------------------------------------------------------------
# Root for all stage A outputs of this cohort.
ROOT=/valiant02/masi/zuol1/data/mycohort/DeepLungScreening

# Where your input NIfTI volumes live, named "{id}.nii.gz".
ORI_ROOT=${ROOT}/nifti

# Session CSV: must have a column "id" matching the NIfTI basenames.
SPLIT_CSV=/valiant02/masi/zuol1/projects/biodesix/DeepLungScreen/cohorts/mycohort/sessions.csv

# Parallelism for stage 1 (CPU-bound).
N_JOBS=8

# Which steps to run (set to 0 to skip).
RUN_STEP1=1
RUN_STEP2=1
RUN_STEP3=1

# ---------------------------------------------------------------------------
# Derived (no edits needed below this line for typical use)
# ---------------------------------------------------------------------------
PREP_ROOT=${ROOT}/prep
BBOX_ROOT=${ROOT}/bbox
FEAT64=${ROOT}/feat64
FEAT128=${ROOT}/feat128

mkdir -p "${PREP_ROOT}" "${BBOX_ROOT}" "${FEAT64}" "${FEAT128}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

if [ "${RUN_STEP1}" = "1" ]; then
    echo "=== Step 1: lung segmentation + crop ==="
    python3 ./1_preprocess/step1_main.py \
        --sess_csv "${SPLIT_CSV}" \
        --prep_root "${PREP_ROOT}" \
        --ori_root "${ORI_ROOT}" \
        --n_jobs "${N_JOBS}"
fi

if [ "${RUN_STEP2}" = "1" ]; then
    echo "=== Step 2: nodule detection (GPU) ==="
    python3 ./2_nodule_detection/step2_main.py \
        --sess_csv "${SPLIT_CSV}" \
        --bbox_root "${BBOX_ROOT}" \
        --prep_root "${PREP_ROOT}"
fi

if [ "${RUN_STEP3}" = "1" ]; then
    echo "=== Step 3: feature extraction (GPU) ==="
    python3 ./3_feature_extraction/step3_main.py \
        --sess_csv "${SPLIT_CSV}" \
        --bbox_root "${BBOX_ROOT}" \
        --prep_root "${PREP_ROOT}" \
        --feat64 "${FEAT64}" \
        --feat128 "${FEAT128}"
fi

echo "=== Done. Features at: ${FEAT128} ==="
