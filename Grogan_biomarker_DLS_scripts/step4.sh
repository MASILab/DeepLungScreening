#!/bin/bash 
ROOT="/nfs/masi/krishar1/MASLAB_Grogan_biomarker_9_29_26/rerun_for_MASLAB_10_6_26" 
ORI_ROOT=${ROOT}/nifti
PREP_ROOT=${ROOT}/prep
BBOX_ROOT=${ROOT}/bbox
FEAT64=${ROOT}/feat64
FEAT128=${ROOT}/feat128
PRED_DIR=${ROOT}/predictions
PRED_CSV=${PRED_DIR}/DLS_imageonly_predictions.csv

mkdir -p ${ORI_ROOT}
mkdir -p ${PREP_ROOT}
mkdir -p ${BBOX_ROOT}
mkdir -p ${FEAT64}
mkdir -p ${FEAT128}
mkdir -p ${PRED_DIR}

echo "Starting step 4 DLS prediction..."

python /home-local/krishar1/DeepLungScreening/4_co_learning/step4_main_imagepred_only.py \
     --sess_csv /home-local/krishar1/MASLAB_projects_with_MASI_8_25_26/MASLAB_combined_biomarker_model/spreadsheets_latest_10_6_26/combined_vandy_colorado_cohort_available_filepaths_final_mcl_ids_10_6_26.csv\
    --feat_root ${FEAT128} \
    --save_csv_path ${PRED_CSV} 

echo "Step 4 DLS prediction completed. Predictions saved to ${PRED_CSV}"