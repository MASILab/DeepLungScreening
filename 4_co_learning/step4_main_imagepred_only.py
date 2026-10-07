import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from model import *
import argparse
import os   # CHANGED

parser = argparse.ArgumentParser()

parser.add_argument('--sess_csv', type=str, default='100004time1999',
                    help='sessions want to be tested')
parser.add_argument('--feat_root', type=str, default='/nfs/masi/gaor2/tmp/justtest/bbox',
                    help='the root for save feat data (use the feat128 folder from step 3)')
parser.add_argument('--save_csv_path', type=str, default='/nfs/masi/gaor2/tmp/justtest/prep',
                    help='the root for save result data')
parser.add_argument('--model_pth', type=str, default='./4_co_learning/pretrain.pth',
                    help='co-learning weights')  # CHANGED: was hardcoded to litz's home dir

args = parser.parse_args()

# CHANGED: image-only prediction, so clinical factors (need_factor) are not read.
# The original need_factor / sess_mark_dict loop is removed; only 'id' is needed in sess_csv.

df = pd.read_csv(args.sess_csv, dtype={'id': str})   # CHANGED: keep ids as strings
sess_splits = df['id'].dropna().tolist()
testsplit = sess_splits

data_path = args.feat_root

model = MultipathModelBL(1)

# model.load_state_dict(torch.load(args.model_pth, map_location=lambda storage, location: storage))
model.load_state_dict(torch.load(args.model_pth))
model.eval()   # CHANGED: turn off Dropout(0.2) on image features at inference

# CHANGED: empty inputs so the clinical and fusion paths are skipped inside forward()
empty_factor = torch.zeros((0, 12))
empty_both_img = torch.zeros((0, 5, 128))

pred_list = []
status_list = []   # CHANGED: record why a scan has no prediction

with torch.no_grad():   # CHANGED
    for i in range(len(testsplit)):
        sess_id = testsplit[i]
        feat_file = data_path + '/' + sess_id + '.npy'

        # CHANGED: scans that failed in step 1/2/3 have no (or a bad) feature file -> skip, don't crash
        if not os.path.isfile(feat_file):
            pred_list.append(np.nan)
            status_list.append('no_feature_file')
            continue
        try:
            test_imgfeat = np.load(feat_file).astype('float32')
        except Exception:
            pred_list.append(np.nan)
            status_list.append('unreadable_feature_file')
            continue
        if test_imgfeat.ndim != 2 or test_imgfeat.shape[1] != 128:
            pred_list.append(np.nan)
            status_list.append(f'bad_feature_shape_{test_imgfeat.shape}')
            continue
        if not np.isfinite(test_imgfeat).all():
            pred_list.append(np.nan)
            status_list.append('nan_in_features')
            continue

        test_imgfeat = torch.from_numpy(test_imgfeat).unsqueeze(0)
        imgPred, clicPred, bothImgPred, bothClicPred, bothPred = model(test_imgfeat, empty_factor, empty_both_img, empty_factor)
        pred_list.append(float(imgPred.data.numpy()[0]))   # CHANGED: was bothPred
        status_list.append('ok')

data = pd.DataFrame()
data['id'] = testsplit
data['pred_image_only'] = pred_list   # CHANGED: was 'pred'
data['status'] = status_list          # CHANGED

data.to_csv(args.save_csv_path, index = False)

# CHANGED: summary instead of printing every prediction
print(data['status'].value_counts().to_string())
failed = data[data['status'] != 'ok']
if len(failed) > 0:
    print(f'{len(failed)} scans without a prediction (see status column), e.g.:')
    print(failed.head(10).to_string(index=False))