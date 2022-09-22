import os
import sys
import shutil
import argparse
import pandas as pd
from tqdm import tqdm

def copy_cohort(src_dir, dst_dir, cohort_path):
    cohort_df = pd.read_csv(cohort_path)
    cohort_ids = cohort_df['id'].tolist()
    for pid in tqdm(os.listdir(src_dir)):
        for year in os.listdir(os.path.join(src_dir, pid)):
            for fname in os.listdir(os.path.join(src_dir, pid, year)):
                scanid = fname.split(".nii.gz")[0]
                if scanid in cohort_ids:
                    os.symlink(os.path.join(src_dir, pid, year, fname), os.path.join(dst_dir, fname))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--src_dir', type=str, default='/nfs/masi/NLST/nifti/NIFTI_pending')
    parser.add_argument('--dst_dir', type=str, default='/home/local/VANDERBILT/litz/data/nlst/DeepLungScreening/nifti')
    parser.add_argument('--cohort_df', type=str, default='/home/local/VANDERBILT/litz/github/MASILab/DeepLungScreening/nlst_cohort_v1.csv')
    args = parser.parse_args()
    copy_cohort(args.src_dir, args.dst_dir, args.cohort_df)
