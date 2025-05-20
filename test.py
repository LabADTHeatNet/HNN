# %% imports and main parameters
import os
import os.path as osp
import torch

from exp import test_exp, test_exp_cls
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

# exp_dir_path = 'out_Yasn_Q/StandardScaler_EdgeRegressorNetwork_Attr_bs16_20250514_160523'
# exp_dir_path = 'out_Yasn_Q/StandardScaler_EdgeRegressorNetwork_Attr_bs64_20250515_175846'
# exp_dir_path = 'out_Yasn_Q/StandardScaler_EdgeRegressorNetwork_Attr_bs64_20250516_112315'
# exp_dir_path = 'out_Yasn_Q/StandardScaler_EdgeRegressorNetwork_Attr_bs64_20250516_115329'
# exp_dir_path = 'out_Yasn_Q/StandardScaler_EdgeRegressorNetwork_Attr_bs8_20250516_115725'
# exp_dir_path = 'out_Yasn_Q/StandardScaler_EdgeRegressorNetwork_Attr_bs32_20250516_134513'
# exp_dir_path = 'out_Yasn_Q/StandardScaler_EdgeRegressorNetwork_Attr_bs8_20250516_151723'

# # === Default experiments ===
# exp_dir_path_list = [
#     # 'out_Yasn_Q/StandardScaler_EdgeRegressorNetwork_Attr_bs32_20250516_184323',
#     'out_Yasn_Q/StandardScaler_EdgeRegressorNetwork_Attr_bs16_20250516_184401',  # best default
#     # 'out_Yasn_Q/StandardScaler_EdgeRegressorNetwork_Attr_bs16_20250518_230055',
# ]

# === Experiments with addition edge attributes ===
exp_dir_path_list = [
    # 'out_Yasn_Q/StandardScaler_EdgeRegressorNetwork_Attr_bs16_20250516_184401',
    'out_Yasn_Q/StandardScaler_EdgeRegressorNetwork_Attr_bs64_20250520_144527',  # best default
    # 'out_Yasn_Q/eaL_StandardScaler_EdgeRegressorNetwork_Attr_bs256_20250519_173355',
    'out_Yasn_Q/eaLdPQ_StandardScaler_EdgeRegressorNetwork_Attr_bs64_20250520_103717',
    # 'out_Yasn_Q/eaLdp_StandardScaler_EdgeRegressorNetwork_Attr_bs64_20250520_105029',
]

for exp_dir_path in exp_dir_path_list:
    out_dir_path = osp.join(exp_dir_path, 'results')
    test_exp(exp_dir_path, out_dir_path, num_samples_to_draw=0)