#!/bin/bash 
# Trial run on subjects with cancer-labels
ROOT="/valiant02/masi/krishar1/NLST_phenotype_journal_extension_from_SPIE/paired_NLST/DeepLungScreening"
ORI_ROOT=${ROOT}/nifti
PREP_ROOT=${ROOT}/prep

mkdir -p ${ORI_ROOT}
mkdir -p ${PREP_ROOT}
echo "Starting step 1 preprocessing..."
echo "Using original root: ${ORI_ROOT}"
echo "Using prep root: ${PREP_ROOT}"
echo "Using log directory: ${ROOT}"


python /home-local/krishar1/DeepLungScreening/1_preprocess/step1_main.py \
 --sess_csv /home-local/krishar1/NLST_nodule_phenotype_extension/NLST_phenotype_journal_extension/spreadsheets_analysis/cancer_nlst_for_dls_preparation_9_7_26.csv \
 --prep_root ${PREP_ROOT} \
 --ori_root ${ORI_ROOT} \
 --log_dir ${ROOT} \
 --n_jobs 30