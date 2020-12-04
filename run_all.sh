#!/bin/bash

IN_ROOT=/INPUTS
OUT_ROOT=/OUTPUTS

PREP_ROOT=${OUT_ROOT}/prep
BBOX_ROOT=${OUT_ROOT}/bbox
FEAT_ROOT=${OUT_ROOT}/feat
PRED_CSV=${OUT_ROOT}/pred.csv

mkdir -p ${PREP_ROOT}
mkdir -p ${BBOX_ROOT}
mkdir -p ${FEAT_ROOT}

ORI_ROOT=${IN_ROOT}/NIfTI
SPLIT_CSV=${IN_ROOT}/test.csv

echo "Run step 1 data preprocessing ..."

python3 ./1_preprocess/step1_main.py --sess_csv ${SPLIT_CSV} --prep_root ${PREP_ROOT} --ori_root ${ORI_ROOT}

echo " step 1 data preprocess finished !"

echo "Run step 2 nodule detection ... (CPU version, 3 - 4 mins per scan needed)"

python3 ./2_nodule_detection/step2_main.py --sess_csv ${SPLIT_CSV} --bbox_root ${BBOX_ROOT} --prep_root ${PREP_ROOT} 

echo "step 2 nodule detection finished ! "

echo "Run step 3 feat extract ... "

python3 ./3_feature_extraction/step3_main.py --sess_csv ${SPLIT_CSV} --bbox_root ${BBOX_ROOT} --prep_root ${PREP_ROOT} --feat_root ${FEAT_ROOT}

echo "step 3 feat extract finished ! "

echo "Run step 4 co-predicting ... "

python3 ./4_co_learning/step4_main.py --sess_csv ${SPLIT_CSV} --feat_root ${FEAT_ROOT} --save_csv_path ${PRED_CSV}

echo "all finished "