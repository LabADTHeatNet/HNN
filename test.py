# %% imports and main parameters
import os
import os.path as osp
import torch

from exp import test_exp
from src.utils import get_str_timestamp

CASCADE = False

if CASCADE:
    server_name = 'CASCADE'
    TORCH_HUB_DIR = '/storage0/pia/python/hub/'
    torch.hub.set_dir(TORCH_HUB_DIR)
    root_dir = '/storage0/pia/python/heatnet/'
else:
    server_name = 'seth'
    root_dir = '.'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

exp_dir_path = 'out_Yasn_Q/ParametricGCN_GlobalPool_SAGEConv_mean_20241224_152937'
out_dir_path = osp.join(exp_dir_path, 'results')
test_exp(exp_dir_path, out_dir_path)