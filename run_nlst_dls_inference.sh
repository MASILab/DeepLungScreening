#!/bin/bash
# Stage A.5: DLS inference (DeepLungScreening step 4) on the harmonized B50f cohort.
# Uses the .npy nodule features from Stage A and pretrain.pth.

set -euo pipefail

ROOT=/valiant02/masi/zuol1/projects/biodesix/DeepLungScreen/data/nlst_test_nodule
SPLIT_CSV=/valiant02/masi/zuol1/projects/biodesix/DeepLungScreen/cohorts/nlst/nlst_test_nodule_step1ok.csv
FEAT_ROOT=${ROOT}/feat128
PRED_DIR=${ROOT}/pred
PRED_CSV=${PRED_DIR}/dls_pred.csv

ENVPY="${ENVPY:-/valiant02/masi/zuol1/envs/lungbl/bin/python}"

mkdir -p "${PRED_DIR}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/4_co_learning"

"${ENVPY}" step4_main.py \
    --sess_csv "${SPLIT_CSV}" \
    --feat_root "${FEAT_ROOT}" \
    --save_csv_path "${PRED_CSV}"

echo "=== Done. Predictions at: ${PRED_CSV} ==="
