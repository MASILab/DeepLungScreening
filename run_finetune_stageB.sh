#!/bin/bash
# Stage B fine-tuning driver: trains MultipathModelBL on the matched
# biodesix train/valid CSVs and saves the best checkpoint.
#
# Usage:
#     bash run_finetune_stageB.sh                 # uses defaults below
#     EPOCHS=100 LR=5e-5 bash run_finetune_stageB.sh
#     CUDA_VISIBLE_DEVICES=1 bash run_finetune_stageB.sh
#
# Output: data/finetune_harmonized/stageB/{best.pth, val_pred_best.csv, train_log.csv, tb/}

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

ROOT=/valiant02/masi/zuol1/projects/biodesix/DeepLungScreen
COHORT_DIR=${ROOT}/cohorts/finetune_harmonized
OUTPUT_DIR=${ROOT}/data/finetune_harmonized/stageB

TRAIN_CSV=${COHORT_DIR}/biodesix_finetune_train.csv
VALID_CSV=${COHORT_DIR}/biodesix_finetune_valid.csv
PRETRAIN=${SCRIPT_DIR}/4_co_learning/pretrain.pth

# Tunable defaults — override via env vars before launch.
BATCH_SIZE=${BATCH_SIZE:-256}
LR=${LR:-1e-4}
EPOCHS=${EPOCHS:-50}
PATIENCE=${PATIENCE:-10}
LABEL_COL=${LABEL_COL:-lung_cancer}
AUX_WEIGHT=${AUX_WEIGHT:-0.5}
NUM_WORKERS=${NUM_WORKERS:-4}

ENVPY="${ENVPY:-/valiant02/masi/zuol1/envs/lungbl/bin/python}"

mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "Stage B fine-tuning"
echo "  train_csv:  ${TRAIN_CSV}  ($(wc -l < ${TRAIN_CSV}) lines)"
echo "  valid_csv:  ${VALID_CSV}  ($(wc -l < ${VALID_CSV}) lines)"
echo "  pretrain:   ${PRETRAIN}"
echo "  output_dir: ${OUTPUT_DIR}"
echo "  hparams:    bs=${BATCH_SIZE} lr=${LR} epochs=${EPOCHS} patience=${PATIENCE} aux_w=${AUX_WEIGHT}"
echo "  label_col:  ${LABEL_COL}"
echo "  python:     ${ENVPY}"
echo "  gpu:        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "============================================================"

"${ENVPY}" 4_co_learning/step4_train.py \
    --train_csv    "${TRAIN_CSV}" \
    --valid_csv    "${VALID_CSV}" \
    --output_dir   "${OUTPUT_DIR}" \
    --pretrain_pth "${PRETRAIN}" \
    --label_col    "${LABEL_COL}" \
    --batch_size   "${BATCH_SIZE}" \
    --lr           "${LR}" \
    --epochs       "${EPOCHS}" \
    --patience     "${PATIENCE}" \
    --aux_weight   "${AUX_WEIGHT}" \
    --num_workers  "${NUM_WORKERS}" \
    2>&1 | tee "${OUTPUT_DIR}/run.log"

echo "=== Done. Outputs at: ${OUTPUT_DIR} ==="
