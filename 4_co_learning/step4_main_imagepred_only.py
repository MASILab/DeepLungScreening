import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from model import *
import argparse

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

with torch.no_grad():   # CHANGED
    for i in range(len(testsplit)):
        sess_id = testsplit[i]

        test_imgfeat = np.load(data_path + '/' + sess_id + '.npy').astype('float32')
        assert test_imgfeat.shape[-1] == 128, f'{sess_id}: got {test_imgfeat.shape}, need 128-dim features'  # CHANGED
        test_imgfeat = torch.from_numpy(test_imgfeat).unsqueeze(0)
        imgPred, clicPred, bothImgPred, bothClicPred, bothPred = model(test_imgfeat, empty_factor, empty_both_img, empty_factor)
        pred_list += list(imgPred.data.numpy())   # CHANGED: was bothPred (Return only image pred)

data = pd.DataFrame()
data['id'] = testsplit
data['pred_image_only'] = pred_list   # CHANGED: was 'pred'

data.to_csv(args.save_csv_path, index = False)

print (pred_list)