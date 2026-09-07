#!/bin/bash

 python 1_preprocess/step1_main.py \
 --sess_csv /home-local/krishar1/Nodule_phenotype_SPIE/analysis_spreadsheets/nlst_1999_cohort_b50f_with_totalseg_nodule_paths_2026_06_30.csv\
 --prep_root /home-local/krishar1/DeepLungScreening/mcl_adenocarcinoma_no_nodule_size

 python 2_nodule_detection/step2_main.py \
 --sess_csv /home-local/krishar1/Nodule_phenotype_SPIE/analysis_spreadsheets/nlst_cohort_b50f_with_totalseg_nodule_paths_2026_06_30.csv \
 --bbox_root /home-local/krishar1/DeepLungScreening/bbox_root_mcl_adenocarcinoma \
 --prep_root /home-local/krishar1/DeepLungScreening/mcl_adenocarcinoma_no_nodule_size