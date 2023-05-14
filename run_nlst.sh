#!/bin/bash
ROOT=/media/litz/Elements/litz/data/nlst/DeepLungScreening
ORI_ROOT=${ROOT}/nifti
PREP_ROOT=${ROOT}/prep
BBOX_ROOT=${ROOT}/bbox
FEAT_ROOT=${ROOT}/feat
PRED_CSV=${ROOT}/pred/nlst_train_pred.csv
TEST_SET=/home/local/VANDERBILT/litz/github/MASILab/DeepLungScreening/cohorts/nlst/nlst_train.csv

echo "Run step 1 data preprocessing ..."

python3 ./1_preprocess/step1_main.py --sess_csv ${TEST_SET} --prep_root ${PREP_ROOT} --ori_root ${ORI_ROOT}

echo " step 1 data preprocess finished !"

echo "Run step 2 nodule detection ... (CPU version, 3 - 4 mins per scan needed)"

# python3 ./2_nodule_detection/step2_main.py --sess_csv ${TEST_SET} --bbox_root ${BBOX_ROOT} --prep_root ${PREP_ROOT} 

echo "step 2 nodule detection finished ! "

echo "Run step 3 feat extract ... "

# python3 ./3_feature_extraction/step3_main.py --sess_csv ${TEST_SET} --bbox_root ${BBOX_ROOT} --prep_root ${PREP_ROOT} --feat_root ${FEAT_ROOT}

echo "step 3 feat extract finished ! "

echo "Run step 4 co-predicting ... "

# python3 ./4_co_learning/step4_main.py --sess_csv ${TEST_SET} --feat_root ${FEAT_ROOT} --save_csv_path ${PRED_CSV}

echo "all finished "

