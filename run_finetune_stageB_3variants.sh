#!/bin/bash
# Three Stage-B variants for nodule-focused deployment:
#   M1: image-only            (4-30mm cohort, imgPred head)
#   M2: image + nodule_size   (4-30mm cohort, bothPred head)
#   M3: image + nodule_size   (all subjects from original QA-yes cohort, bothPred head)
#
# Then evaluates each on the held-out 550.

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

ROOT=/valiant02/masi/zuol1/projects/biodesix/DeepLungScreen
COHORT_DIR=${ROOT}/cohorts/finetune_harmonized
STAGEB_BASE=${ROOT}/data/finetune_harmonized/stageB
PRETRAIN=${SCRIPT_DIR}/4_co_learning/pretrain.pth

# Tunables (env overrides supported)
BATCH_SIZE=${BATCH_SIZE:-256}
LR=${LR:-1e-4}
EPOCHS=${EPOCHS:-50}
PATIENCE=${PATIENCE:-10}
LABEL_COL=${LABEL_COL:-lung_cancer}
AUX_WEIGHT=${AUX_WEIGHT:-0.5}
NUM_WORKERS=${NUM_WORKERS:-4}
ENVPY="${ENVPY:-/valiant02/masi/zuol1/envs/lungbl/bin/python}"

# Cohort CSV inputs
TRAIN_4TO30=${COHORT_DIR}/biodesix_finetune_train_4to30.csv
VALID_4TO30=${COHORT_DIR}/biodesix_finetune_valid_4to30.csv
TRAIN_FULL=${COHORT_DIR}/biodesix_finetune_train.csv
VALID_FULL=${COHORT_DIR}/biodesix_finetune_valid.csv
HOLDOUT_CSV=${COHORT_DIR}/biodesix_holdout_matched.csv

# ---------------------------------------------------------------------------
# 0. Build the 4-30mm matched CSVs (idempotent — does nothing if already built)
# ---------------------------------------------------------------------------
if [ ! -f "${TRAIN_4TO30}" ] || [ ! -f "${VALID_4TO30}" ]; then
    echo "Building 4-30mm matched CSVs..."
    "${ENVPY}" cohorts/finetune_harmonized/build_finetune_4to30_matched.py
fi

if [ ! -f "${HOLDOUT_CSV}" ]; then
    echo "Building held-out matched CSV..."
    "${ENVPY}" cohorts/finetune_harmonized/build_holdout_matched.py
fi

# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------
train_one () {
    local NAME=$1     # M1 / M2 / M3
    local TRAIN=$2
    local VALID=$3
    local MODE=$4     # image_only / nodule_size_only / all_biomarkers
    local OUT=${STAGEB_BASE}/${NAME}
    mkdir -p "${OUT}"
    echo ""
    echo "================================================================"
    echo "${NAME}: clinical_mode=${MODE}"
    echo "  train: ${TRAIN}"
    echo "  valid: ${VALID}"
    echo "  out:   ${OUT}"
    echo "================================================================"
    "${ENVPY}" 4_co_learning/step4_train.py \
        --train_csv     "${TRAIN}" \
        --valid_csv     "${VALID}" \
        --output_dir    "${OUT}" \
        --pretrain_pth  "${PRETRAIN}" \
        --label_col     "${LABEL_COL}" \
        --batch_size    "${BATCH_SIZE}" \
        --lr            "${LR}" \
        --epochs        "${EPOCHS}" \
        --patience      "${PATIENCE}" \
        --aux_weight    "${AUX_WEIGHT}" \
        --num_workers   "${NUM_WORKERS}" \
        --clinical_mode "${MODE}" \
        2>&1 | tee "${OUT}/train.log"
}

# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------
eval_one () {
    local NAME=$1     # M1 / M2 / M3
    local MODE=$2
    local OUT=${STAGEB_BASE}/${NAME}
    local CKPT=${OUT}/best.pth
    local PRED=${OUT}/holdout_pred.csv
    if [ ! -f "${CKPT}" ]; then
        echo "  ERR ${NAME}: no checkpoint at ${CKPT}, skipping eval"
        return
    fi
    echo ""
    echo "----- Eval ${NAME} on held-out 550 (clinical_mode=${MODE}) -----"
    "${ENVPY}" 4_co_learning/step4_predict.py \
        --input_csv     "${HOLDOUT_CSV}" \
        --checkpoint    "${CKPT}" \
        --output_csv    "${PRED}" \
        --label_col     "${LABEL_COL}" \
        --clinical_mode "${MODE}" \
        2>&1 | tee "${PRED}.log"
}

# ---------------------------------------------------------------------------
# Train + eval each variant
# ---------------------------------------------------------------------------
train_one M1 "${TRAIN_4TO30}" "${VALID_4TO30}" image_only
train_one M2 "${TRAIN_4TO30}" "${VALID_4TO30}" nodule_size_only
train_one M3 "${TRAIN_FULL}"  "${VALID_FULL}"  nodule_size_only

eval_one M1 image_only
eval_one M2 nodule_size_only
eval_one M3 nodule_size_only

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "SUMMARY"
echo "================================================================"
"${ENVPY}" - <<PYEOF
import os, json
base = "${STAGEB_BASE}"
print(f"{'variant':<6} {'best_val_auc':>12} {'holdout_auc':>12} {'holdout_ap':>12} {'n':>6}  description")
print("-" * 80)
for v in ("M1", "M2", "M3"):
    args_p = os.path.join(base, v, "args.json")
    pred_m = os.path.join(base, v, "holdout_pred.csv.metrics.json")
    val_auc = "-"; hauc = "-"; hap = "-"; n = "-"; desc = ""
    if os.path.isfile(args_p):
        with open(args_p) as f: a = json.load(f)
        desc = f"mode={a.get('clinical_mode','?')}, train={os.path.basename(a.get('train_csv',''))}"
    # best val AUC is encoded in train_log.csv last best row; easiest from checkpoint
    import torch
    ckpt_p = os.path.join(base, v, "best.pth")
    if os.path.isfile(ckpt_p):
        try:
            c = torch.load(ckpt_p, map_location="cpu", weights_only=False)
            val_auc = f"{c.get('val_auc', float('nan')):.4f}"
        except Exception: pass
    if os.path.isfile(pred_m):
        with open(pred_m) as f: m = json.load(f)
        hauc = f"{m.get('auc', float('nan')):.4f}" if 'auc' in m else "-"
        hap  = f"{m.get('ap',  float('nan')):.4f}" if 'ap'  in m else "-"
        n    = str(m.get('n_scored', m.get('n_total', '-')))
    print(f"{v:<6} {val_auc:>12} {hauc:>12} {hap:>12} {n:>6}  {desc}")
PYEOF

echo ""
echo "All outputs under: ${STAGEB_BASE}/{M1,M2,M3}/"
