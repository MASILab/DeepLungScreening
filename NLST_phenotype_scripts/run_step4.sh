#!/bin/bash 
ROOT="/valiant02/masi/krishar1/NLST_phenotype_journal_extension_from_SPIE/paired_NLST/DeepLungScreening" 
ORI_ROOT=${ROOT}/nifti
PREP_ROOT=${ROOT}/prep
BBOX_ROOT=${ROOT}/bbox
FEAT64=${ROOT}/feat64
FEAT128=${ROOT}/feat128
PRED_DIR=${ROOT}/predictions
PRED_CSV=${ROOT}/dls_predictions_paired_cancer.csv

mkdir -p ${ORI_ROOT}
mkdir -p ${PREP_ROOT}
mkdir -p ${BBOX_ROOT}
mkdir -p ${FEAT64}
mkdir -p ${FEAT128}
mkdir -p ${PRED_DIR}

echo "Starting step 4 DLS prediction..."

python /home-local/krishar1/DeepLungScreening/4_dls_prediction/step4_main.py \
     --sess_csv /home-local/krishar1/DeepLungScreening/NLST_T0_paired_cancer_for_DLS_step4_pred_9_8_26.csv\
    --feat_root ${FEAT128} \
    --save_csv_path ${PRED_CSV} 