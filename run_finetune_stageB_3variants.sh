#!/bin/bash
# Stage-B variants for nodule-focused deployment:
#   M1: image-only                 (4-30mm cohort, imgPred head)
#   M2: image + nodule_size        (4-30mm cohort, bothPred head; size always present)
#   M3: image + nodule_size        (all subjects from QA-yes cohort, bothPred head)
#   M4: image + OPTIONAL size      (full cohort, single shared-trunk model with
#                                   modality dropout). This is the deployment model:
#                                   one checkpoint serves both real-world cases
#                                   (image-only OR image+size), with the head chosen
#                                   deterministically by size availability. M4 is
#                                   evaluated on the holdout under BOTH routes so it
#                                   can be compared head-to-head against M1 (image-only)
#                                   and M2/M3 (image+size).
#
# M1/M2/M3 are kept as baselines; M4 is the one to ship if it matches or beats them.
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
DROPOUT_P=${DROPOUT_P:-0.3}     # M4 only: size modality-dropout probability
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
        --train_csv      "${TRAIN}" \
        --valid_csv      "${VALID}" \
        --output_dir     "${OUT}" \
        --pretrain_pth   "${PRETRAIN}" \
        --label_col      "${LABEL_COL}" \
        --batch_size     "${BATCH_SIZE}" \
        --lr             "${LR}" \
        --epochs         "${EPOCHS}" \
        --patience       "${PATIENCE}" \
        --aux_weight     "${AUX_WEIGHT}" \
        --num_workers    "${NUM_WORKERS}" \
        --clinical_mode  "${MODE}" \
        --size_dropout_p "${DROPOUT_P}" \
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

# Eval a single model on the holdout under a forced routing (image_only|both|natural).
# Writes holdout_pred_<route>.csv so a single image_plus_optional_size model can be
# scored in each deployment mode and compared against the M1/M2 baselines.
eval_route () {
    local NAME=$1     # e.g. M4
    local MODE=$2     # clinical_mode used at train (image_plus_optional_size)
    local ROUTE=$3    # image_only | both | natural
    local OUT=${STAGEB_BASE}/${NAME}
    local CKPT=${OUT}/best.pth
    local PRED=${OUT}/holdout_pred_${ROUTE}.csv
    if [ ! -f "${CKPT}" ]; then
        echo "  ERR ${NAME}: no checkpoint at ${CKPT}, skipping eval (${ROUTE})"
        return
    fi
    echo ""
    echo "----- Eval ${NAME} on held-out 550 (mode=${MODE}, force_route=${ROUTE}) -----"
    "${ENVPY}" 4_co_learning/step4_predict.py \
        --input_csv     "${HOLDOUT_CSV}" \
        --checkpoint    "${CKPT}" \
        --output_csv    "${PRED}" \
        --label_col     "${LABEL_COL}" \
        --clinical_mode "${MODE}" \
        --force_route   "${ROUTE}" \
        2>&1 | tee "${PRED}.log"
}

# ---------------------------------------------------------------------------
# Train + eval each variant
# ---------------------------------------------------------------------------
train_one M1 "${TRAIN_4TO30}" "${VALID_4TO30}" image_only
train_one M2 "${TRAIN_4TO30}" "${VALID_4TO30}" nodule_size_only
train_one M3 "${TRAIN_FULL}"  "${VALID_FULL}"  nodule_size_only
train_one M4 "${TRAIN_FULL}"  "${VALID_FULL}"  image_plus_optional_size

eval_one M1 image_only
eval_one M2 nodule_size_only
eval_one M3 nodule_size_only

# M4 is the deployment model: score the same holdout under each routing.
#   natural    = realistic mixed deployment (size-present -> joint, size-absent -> image)
#   image_only = every subject through the image-only head  (apples-to-apples vs M1)
#   both       = every subject through the joint head       (apples-to-apples vs M2/M3)
eval_route M4 image_plus_optional_size natural
eval_route M4 image_plus_optional_size image_only
eval_route M4 image_plus_optional_size both

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

def fmt_metrics(pred_m):
    hauc = hap = n = "-"
    if os.path.isfile(pred_m):
        with open(pred_m) as f: m = json.load(f)
        hauc = f"{m['auc']:.4f}" if 'auc' in m else "-"
        hap  = f"{m['ap']:.4f}"  if 'ap'  in m else "-"
        n    = str(m.get('n_scored', m.get('n_total', '-')))
    return hauc, hap, n

def best_val(ckpt_p):
    if not os.path.isfile(ckpt_p):
        return "-"
    try:
        import torch
        c = torch.load(ckpt_p, map_location="cpu", weights_only=False)
        return f"{c.get('sel_metric', c.get('val_auc', float('nan'))):.4f}"
    except Exception:
        return "-"

print(f"{'variant':<14} {'best_val':>9} {'holdout_auc':>12} {'holdout_ap':>11} {'n':>6}  description")
print("-" * 90)
# Baselines M1/M2/M3 (single natural-route eval).
for v in ("M1", "M2", "M3"):
    args_p = os.path.join(base, v, "args.json")
    desc = ""
    if os.path.isfile(args_p):
        with open(args_p) as f: a = json.load(f)
        desc = f"mode={a.get('clinical_mode','?')}, train={os.path.basename(a.get('train_csv',''))}"
    hauc, hap, n = fmt_metrics(os.path.join(base, v, "holdout_pred.csv.metrics.json"))
    print(f"{v:<14} {best_val(os.path.join(base, v, 'best.pth')):>9} {hauc:>12} {hap:>11} {n:>6}  {desc}")

# M4 deployment model: report each route.
v = "M4"
bv = best_val(os.path.join(base, v, "best.pth"))
for route, label, desc in (
    ("natural",    "M4/natural",    "realistic mixed deployment"),
    ("image_only", "M4/image-only", "compare vs M1"),
    ("both",       "M4/image+size", "compare vs M2/M3"),
):
    hauc, hap, n = fmt_metrics(os.path.join(base, v, f"holdout_pred_{route}.csv.metrics.json"))
    print(f"{label:<14} {bv:>9} {hauc:>12} {hap:>11} {n:>6}  {desc}")
PYEOF

echo ""
echo "All outputs under: ${STAGEB_BASE}/{M1,M2,M3}/"
