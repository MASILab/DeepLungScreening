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
                    help='the root for save feat data')
parser.add_argument('--save_csv_path', type=str, default='/nfs/masi/gaor2/tmp/justtest/prep',
                    help='the root for save result data')

args = parser.parse_args()

need_factor = ['with_image', 'with_marker',  'age',  'education',  'bmi',  'phist', 'fhist', 'smo_status', 'quit_time', 'pkyr', 'plco', 'kaggle_cancer']

sess_mark_dict = {}

df = pd.read_csv(args.sess_csv)
sess_splits = df['id'].tolist()
testsplit = sess_splits

for i, item in df.iterrows():
    test_biomarker = np.zeros(12).astype('float32')
    for j in range(len(need_factor)):
        test_biomarker[j] = item[need_factor[j]]
    sess_mark_dict[item['id']] = test_biomarker

data_path = args.feat_root



model = MultipathModelBL(1)

model_pth = './4_co_learning/pretrain.pth'

model.load_state_dict(torch.load(model_pth, map_location=lambda storage, location: storage))
#model.load_state_dict(torch.load(model_pth))

pred_list = []

for i in range(len(testsplit)):
    sess_id = testsplit[i]
    test_biomarker = sess_mark_dict[sess_id]
    
    test_imgfeat = np.load(data_path + '/' + sess_id + '.npy')
    test_biomarker = torch.from_numpy(test_biomarker).unsqueeze(0)
    test_imgfeat = torch.from_numpy(test_imgfeat).unsqueeze(0)
    imgPred, clicPred, bothImgPred, bothClicPred, bothPred = model(test_imgfeat, test_biomarker, test_imgfeat, test_biomarker)
    pred_list += list(bothPred.data.numpy())

data = pd.DataFrame()
data['id'] = testsplit
data['pred'] = pred_list

data.to_csv(args.save_csv_path, index = False)

print (pred_list)

