#!/bin/bash
# Held-out test inference: match held-out subjects against existing Stage A
# features, then run the fine-tuned DLS model on them and report AUC.
#
# Usage:
#     bash run_holdout_inference.sh                 # default checkpoint = stageB/best.pth
#     CKPT=/path/to/custom.pth bash run_holdout_inference.sh
#     CUDA_VISIBLE_DEVICES=1 bash run_holdout_inference.sh

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

ROOT=/valiant02/masi/zuol1/projects/biodesix/DeepLungScreen
COHORT_DIR=${ROOT}/cohorts/finetune_harmonized
STAGEB_DIR=${ROOT}/data/finetune_harmonized/stageB

HOLDOUT_XLSX=${HOLDOUT_XLSX:-${COHORT_DIR}/selected_subjects_exclude_20260515.xlsx}
MATCHED_CSV=${COHORT_DIR}/biodesix_holdout_matched.csv
PRED_CSV=${PRED_CSV:-${STAGEB_DIR}/holdout_pred.csv}
CKPT=${CKPT:-${STAGEB_DIR}/best.pth}
LABEL_COL=${LABEL_COL:-lung_cancer}

ENVPY="${ENVPY:-/valiant02/masi/zuol1/envs/lungbl/bin/python}"

echo "============================================================"
echo "Held-out inference"
echo "  spreadsheet: ${HOLDOUT_XLSX}"
echo "  matched_csv: ${MATCHED_CSV}"
echo "  checkpoint:  ${CKPT}"
echo "  pred_out:    ${PRED_CSV}"
echo "  label_col:   ${LABEL_COL}"
echo "  python:      ${ENVPY}"
echo "  gpu:         CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "============================================================"

# 1. Match held-out subjects to existing Stage A features
HOLDOUT_XLSX="${HOLDOUT_XLSX}" "${ENVPY}" \
    cohorts/finetune_harmonized/build_holdout_matched.py

if [ ! -f "${MATCHED_CSV}" ]; then
    echo "ERROR: expected matched CSV not produced: ${MATCHED_CSV}"
    exit 1
fi

# 2. Run inference with the fine-tuned model
"${ENVPY}" 4_co_learning/step4_predict.py \
    --input_csv  "${MATCHED_CSV}" \
    --checkpoint "${CKPT}" \
    --output_csv "${PRED_CSV}" \
    --label_col  "${LABEL_COL}" \
    2>&1 | tee "${PRED_CSV}.log"

echo "=== Done. ==="
echo "  predictions: ${PRED_CSV}"
echo "  metrics:     ${PRED_CSV}.metrics.json"
echo "  log:         ${PRED_CSV}.log"
