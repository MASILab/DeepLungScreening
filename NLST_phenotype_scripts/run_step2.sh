#!/bin/bash 
ROOT="/valiant02/masi/krishar1/NLST_phenotype_journal_extension_from_SPIE/paired_NLST/DeepLungScreening"
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
echo "Using log directory: ${ROOT}"


python /home-local/krishar1/DeepLungScreening/2_nodule_detection/step2_main.py \
 --sess_csv /home-local/krishar1/NLST_nodule_phenotype_extension/NLST_phenotype_journal_extension/spreadsheets_analysis/cancer_nlst_for_dls_preparation_9_7_26.csv \
 --prep_root ${PREP_ROOT} \
 --bbox_root ${BBOX_ROOT} \
 --gpu 1 \

