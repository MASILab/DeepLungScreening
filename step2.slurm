#!/bin/bash
#SBATCH --mail-user=thomas.z.li@vanderbilt.edu
#SBATCH --mail-type=FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --array=1-8%10
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/litz/DeepLungScreening/logs/log_%A_%a.out

ml Anaconda3/2021.05
source env/bin/activate
python3 ./2_nodule_detection/step2_main.py --sess_csv ${SPLIT_CSV} --bbox_root ${BBOX_ROOT} --prep_root ${PREP_ROOT}
