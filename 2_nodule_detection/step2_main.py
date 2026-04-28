
import torch

from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable

#from layers import acc
from data_loader import DataBowl3Detector,collate
#from data_classifier import DataBowl3Classifier

from utils import *
from split_combine import SplitComb
from test_detect import test_detect
from importlib import import_module
import pandas as pd
import pdb
import argparse
from joblib import Parallel, delayed
from detect_config import config
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--sess_csv', type=str, default='test.csv',
                    help='sessions want to be tested')
parser.add_argument('--bbox_root', type=str, default='/nfs/masi/gaor2/tmp/justtest/bbox',
                    help='the root for save preprocessed data')
parser.add_argument('--prep_root', type=str, default='/nfs/masi/gaor2/tmp/justtest/prep',
                    help='the root for save preprocessed data')
parser.add_argument('--n_jobs', type=int, default=1)

args = parser.parse_args()
config['datadir'] = args.prep_root

sessions = pd.read_csv(args.sess_csv, dtype={'id':str})
sessions = sessions[~sessions['id'].isnull()]['id'].tolist()

# job_size = len(sessions) // args.n_jobs
# sess_splits = [sessions[i: i+job_size] for i in range(0, len(sessions), job_size)]
sess_splits = sessions

# def detect(sess_splits):
config['testsplit'] = sess_splits
# config['testsplit'] = ['100529time2001']

nodmodel = import_module('net_detector')
config1, nod_net, loss, get_pbb = nodmodel.get_model()
_ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'detector.ckpt')
checkpoint = torch.load(_ckpt_path, weights_only=False)
nod_net.load_state_dict(checkpoint['state_dict'])

device = torch.device("cuda:0")
nod_net = nod_net.to(device)

bbox_result_path = args.bbox_root
if not os.path.exists(bbox_result_path):
    os.mkdir(bbox_result_path)

split_comber = SplitComb(config1['sidelen'],config1['max_stride'],config1['stride'],config1['margin'],pad_value= config1['pad_value'])

dataset = DataBowl3Detector(config['testsplit'],config1,phase='test',split_comber=split_comber)
test_loader = DataLoader(dataset, batch_size = 1,
    shuffle = False, num_workers = 1, pin_memory=False, collate_fn =collate)

test_detect(test_loader, nod_net, get_pbb, bbox_result_path, config1, device)

# Parallel(n_jobs=args.n_jobs, prefer="threads")(
#     delayed(detect)(sess_split) for sess_split in tqdm(sess_splits, total=len(sess_splits))
# )
