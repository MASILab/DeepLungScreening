#!/bin/bash
# Stage A driver: NIfTI CT -> nodule features (.npy)
# Cohort: NLST harmonized-to-B50f, one image per session
# Source CSV: cohorts/nlst/nlst_test_nodule.csv

set -euo pipefail

# ---------------------------------------------------------------------------
# EDIT THESE
# ---------------------------------------------------------------------------
# Root for all stage A outputs of this run.
ROOT=/valiant02/masi/zuol1/projects/biodesix/DeepLungScreen/data/nlst_test_nodule

# Session CSV (must have columns: id, fpath).
SPLIT_CSV=/valiant02/masi/zuol1/projects/biodesix/DeepLungScreen/cohorts/nlst/nlst_test_nodule_step1ok.csv

# Parallelism for stage 1 (CPU-bound).
N_JOBS=8

# Which steps to run (set to 0 to skip).
RUN_STEP0=1   # symlink setup
RUN_STEP1=1   # lung segmentation + crop (CPU)
RUN_STEP2=1   # nodule detection (GPU)
RUN_STEP3=1   # feature extraction (GPU)

# ---------------------------------------------------------------------------
# Derived
# ---------------------------------------------------------------------------
ORI_ROOT=${ROOT}/nifti        # symlinks live here, named {id}.nii.gz
PREP_ROOT=${ROOT}/prep
BBOX_ROOT=${ROOT}/bbox
FEAT64=${ROOT}/feat64
FEAT128=${ROOT}/feat128

mkdir -p "${ORI_ROOT}" "${PREP_ROOT}" "${BBOX_ROOT}" "${FEAT64}" "${FEAT128}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

# ---------------------------------------------------------------------------
# Step 0: symlink each harmonized NIfTI into ORI_ROOT as {id}.nii.gz
# Stage 1 reads ${ORI_ROOT}/{id}.nii.gz; the actual files live elsewhere.
# Symlinks are essentially free and avoid copying ~hundreds of GB.
# ---------------------------------------------------------------------------
if [ "${RUN_STEP0}" = "1" ]; then
    echo "=== Step 0: setting up symlinks in ${ORI_ROOT} ==="
    python3 - <<PYEOF
import os, sys, pandas as pd

csv_path  = "${SPLIT_CSV}"
ori_root  = "${ORI_ROOT}"

df = pd.read_csv(csv_path, dtype={'pid': str}, low_memory=False)
required = {'id', 'fpath'}
missing  = required - set(df.columns)
if missing:
    sys.exit(f"CSV missing required columns: {missing}")

made, skipped, missing_src = 0, 0, []
for _, row in df.iterrows():
    src = row['fpath']
    dst = os.path.join(ori_root, f"{row['id']}.nii.gz")
    if not os.path.exists(src):
        missing_src.append(src)
        continue
    if os.path.islink(dst) or os.path.exists(dst):
        skipped += 1
        continue
    os.symlink(src, dst)
    made += 1

print(f"  created: {made}, already-existed: {skipped}, missing-source: {len(missing_src)}")
if missing_src:
    print("  WARNING: first 5 missing source files:")
    for p in missing_src[:5]:
        print(f"    {p}")
PYEOF
fi

# ---------------------------------------------------------------------------
# Step 1: lung segmentation + lumen crop  -> {id}_clean.npy
# ---------------------------------------------------------------------------
if [ "${RUN_STEP1}" = "1" ]; then
    echo "=== Step 1: lung segmentation + crop (CPU, n_jobs=${N_JOBS}) ==="
    python3 ./1_preprocess/step1_main.py \
        --sess_csv "${SPLIT_CSV}" \
        --prep_root "${PREP_ROOT}" \
        --ori_root "${ORI_ROOT}" \
        --n_jobs "${N_JOBS}"
fi

# ---------------------------------------------------------------------------
# Step 2: nodule detection -> {id}_pbb.npy, {id}_lbb.npy
# ---------------------------------------------------------------------------
if [ "${RUN_STEP2}" = "1" ]; then
    echo "=== Step 2: nodule detection (GPU) ==="
    python3 ./2_nodule_detection/step2_main.py \
        --sess_csv "${SPLIT_CSV}" \
        --bbox_root "${BBOX_ROOT}" \
        --prep_root "${PREP_ROOT}"
fi

# ---------------------------------------------------------------------------
# Step 3: CaseNet feature extraction -> {id}.npy (top-5 nodules x 128-d)
# ---------------------------------------------------------------------------
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
