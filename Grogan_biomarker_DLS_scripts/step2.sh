#!/bin/bash 
ROOT="/nfs/masi/krishar1/MASLAB_Grogan_biomarker_9_29_26"
ORI_ROOT=${ROOT}/nifti
PREP_ROOT=${ROOT}/prep
BBOX_ROOT=${ROOT}/bbox


mkdir -p ${ORI_ROOT}
mkdir -p ${PREP_ROOT}
mkdir -p ${BBOX_ROOT}

echo "Starting step 2 nodule detection..."
echo "Using original root: ${ORI_ROOT}"
echo "Using prep root: ${PREP_ROOT}"
echo "Using bbox root: ${BBOX_ROOT}"

python /home-local/krishar1/DeepLungScreening/2_nodule_detection/step2_main.py \
 --sess_csv /home-local/krishar1/MASLAB_projects_with_MASI_8_25_26/MASLAB_combined_biomarker_model/spreadsheets_to_run_dls/9_29_26_concatenated_vandy_colorado_cohort_for_DLS.csv \
 --prep_root ${PREP_ROOT} \
 --bbox_root ${BBOX_ROOT} \
