import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
from importlib import import_module
from data_classifier import DataBowl3Classifier
import argparse
import pandas as pd
from tqdm import tqdm

from cls_config import config as config2

parser = argparse.ArgumentParser()

parser.add_argument('--sess_csv', type=str, default='./test.csv',
                    help='sessions want to be tested')
parser.add_argument('--prep_root', type=str, default='/nfs/masi/gaor2/tmp/justtest',
                    help='the root for save preprocessed data')
parser.add_argument('--bbox_root', type=str, default='/nfs/masi/gaor2/tmp/justtest',
                    help='the root of original data')
# parser.add_argument('--feat_root', type=str, default='/nfs/masi/gaor2/tmp/justtest',
#                     help='the root of original data')
parser.add_argument('--feat64', type=str, default='/nfs/masi/gaor2/tmp/justtest',
                    help='the root of original data')
parser.add_argument('--feat128', type=str, default='/nfs/masi/gaor2/tmp/justtest',
                    help='the root of original data')
parser.add_argument('--job', type=int, default=0)


args = parser.parse_args()


device = torch.device("cuda:0")
casemodel = import_module('net_classifier')
casenet = casemodel.CaseNet(topk=5)
# load_state_dict
config2 = casemodel.config
state_dict = torch.load('/home/local/VANDERBILT/litz/github/MASILab/DeepLungScreening/3_feature_extraction/classifier_state_dictpy3.ckpt')

model_dict = casenet.state_dict()
pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}

casenet.load_state_dict(state_dict)
casenet = casenet.to(device) #.cuda()

config2['bboxpath'] = args.bbox_root
config2['datadir'] = args.prep_root
# config2['feat_root'] = args.feat_root
config2['feat64'] = args.feat64
config2['feat128'] = args.feat128

def job_split(sess_splits, job, max_job=4):
    n = int(len(sess_splits)/max_job) # size of each job
    jobs = []
    for i in range(0, len(sess_splits), n):
        if i==(n*(max_job-1)):
            jobs.append(sess_splits[i:])
            break
        else:
            jobs.append(sess_splits[i:i+n])
    return jobs[job]

sess_splits = pd.read_csv(args.sess_csv, dtype={'id':str})
sess_splits = sess_splits[~sess_splits['id'].isnull()]['id'].tolist()
# sess_splits = pd.read_csv(args.sess_csv)['id'].tolist()
# testsplit = job_split(sess_splits, args.job)

def test_casenet(model,testset, device):
    data_loader = DataLoader(
        testset,
        batch_size = 1,
        shuffle = False,
        num_workers = 4,
        pin_memory=False)
    #model = model.cuda()
    model.eval()
    predlist = []
    
    #     weight = torch.from_numpy(np.ones_like(y).float().cuda()
    for i,(x,coord, subj_name) in tqdm(enumerate(data_loader)):
        fname64 = os.path.join(config2['feat64'], f"{subj_name[0]}.npy")
        fname128 = os.path.join(config2['feat128'], f"{subj_name[0]}.npy")
        # fname = os.path.join(config2['feat_root'], f"{subj_name[0]}.npy")
        if os.path.isfile(fname64) and os.path.isfile(fname128):
            print(f"{fname64} and {fname128} existed")
            continue
        else:
            print (i, subj_name[0])   

            coord = Variable(coord).to(device) #.cuda()
            x = Variable(x).to(device) #.cuda()
            nodulePred,casePred, feat128, feat64 = model(x,coord)
            predlist.append(casePred.data.cpu().numpy())

            np.save(fname128, feat128.data.cpu().numpy())
            np.save(fname64, feat64.data.cpu().numpy())
    #        if 'feat128' in config_submit['save_feat_mode']:
    #            np.save(fname128, feat128.data.cpu().numpy())
#         if 'feat64' in config_submit['save_feat_mode'] :
#             np.save(fname64, feat64.data.cpu().numpy())

        #print([i,data_loader.dataset.split[i,1],casePred.data.cpu().numpy()])
    # predlist = np.concatenate(predlist)
    return predlist



dataset = DataBowl3Classifier(sess_splits, config2, phase = 'test')
test_casenet(casenet,dataset, device)
#print (predlist)