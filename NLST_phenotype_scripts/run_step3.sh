#!/bin/bash 
ROOT="/valiant02/masi/krishar1/NLST_phenotype_journal_extension_from_SPIE/paired_NLST/DeepLungScreening"
ORI_ROOT=${ROOT}/nifti
PREP_ROOT=${ROOT}/prep
BBOX_ROOT=${ROOT}/bbox
FEAT64=${ROOT}/feat64
FEAT128=${ROOT}/feat128

mkdir -p ${ORI_ROOT}
mkdir -p ${PREP_ROOT}
mkdir -p ${BBOX_ROOT}
mkdir -p ${FEAT64}
mkdir -p ${FEAT128}

echo "Starting step 2 nodule detection..."
echo "Using original root: ${ORI_ROOT}"
echo "Using prep root: ${PREP_ROOT}"
echo "Using bbox root: ${BBOX_ROOT}"
echo "Using feat64 root: ${FEAT64}"
echo "Using feat128 root: ${FEAT128}"

echo "Step 3 feature extraction...."

python /home-local/krishar1/DeepLungScreening/3_feature_extraction/step3_main.py \
     --sess_csv /home-local/krishar1/DeepLungScreening/cancer_nlst_for_dls_preparation_9_7_26.csv \
    --prep_root ${PREP_ROOT} \
    --bbox_root ${BBOX_ROOT} \
    --feat64 ${FEAT64} \
    --feat128 ${FEAT128} \