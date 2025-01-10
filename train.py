# %% imports and main parameters
import copy
import os.path as osp
import torch

from exp import (
    exp,
    test_exp
)
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

# %% experiment configs
utils = dict(
    server_name=server_name,
    out_dir='out_Yasn_Q',
    device=device,
    seed=42
)

dataset = dict(
    datasets_dir=osp.join(root_dir, 'datasets'),
    name='database_Yasn_Q',
    load=False,
    fp='pyg_dataset_Yasn_Q.pt',
    # Пример: другие атрибуты добавьте сюда
    node_attr=['pos_x', 'pos_y', 'P', 'types'],
    # Пример: укажите ваши атрибуты рёбер
    edge_attr=['dP'],
    # Имя колонки, используемой для edge_label
    edge_label=['d'],
    num_samples=None
)

dataloader = dict(
    train_ratio=0.7,  # 70% - обучающая выборка
    val_ratio=0.15,   # 15% - валидационная выборка
    batch_size=32,
)

optimizer = dict(
    name='AdamW',
    kwargs=dict(
        lr=1e-3,
        weight_decay=1e-5
    )
)

scheduler = dict(
    name='StepLR',
    kwargs=dict(
        step_size=10,
        gamma=0.85
    )
)

criterion = dict(
    name='MSELoss',
    kwargs=dict()
)

train = dict(
    num_epochs=250,
    score_metric='MSE'
)

# %% experiment
if __name__ == '__main__':
    debug_run = False
    run_clear_ml = True

    model_list = list()

    # model_name_list = ['FlatGCN', 'ParametricGCN', 'ParametricGCN_GlobalPool']
    # model_name_list = ['ParametricGCN', 'ParametricGCN_GlobalPool']
    # conv_type_list = ['GCNConv', 'SAGEConv', 'GATConv']
    # global_pool_list = ['global_mean_pool', 'global_max_pool']

    node_conv_layer_list = [4*2**i for i in range(6)] + [256] * 10
    edge_fc_layer_list = [8*2**i for i in range(4)]
    out_fc_layer_list = [32*4**i for i in list(reversed(range(4)))]

    model = dict(
        name='ParametricGCN_GlobalPool',
        kwargs=dict(
            node_conv_layer_type='SAGEConv',
            node_conv_layer_list=node_conv_layer_list,
            node_conv_heads=1,
            node_conv_layer_kwargs=dict(
                aggr='mean'),
            node_global_pool_type='global_mean_pool',
            edge_fc_layer_list=edge_fc_layer_list,
            out_fc_layer_list=out_fc_layer_list
        )
    )
    model_main_params = [
        "RobustScaler",
        f"{model['name']}",
        f"{model['kwargs']['node_conv_layer_type']}",
        f"{model['kwargs']['node_conv_layer_kwargs']['aggr']}",
    ]
    model_list.append([model, model_main_params])
    batch_size_list = [384, 256, 128, 64, 32]

    first_exp = True
    for model, model_main_params in model_list:
        for batch_size in batch_size_list:

            cfg = dict(
                utils=utils,
                dataset=dataset,
                dataloader=dataloader,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
                train=train
            )

            cfg['dataloader']['batch_size'] = batch_size

            if debug_run:
                run_clear_ml = False
                cfg['utils']['out_dir'] += '_test'
                cfg['dataset']['num_samples'] = 6000
                cfg['train']['num_epochs'] = 100

            if not first_exp:
                cfg['dataset']['load'] = True

            ts = get_str_timestamp()
            model_main_params_cur = copy.copy(model_main_params)
            model_main_params_cur.append(
                f"bs{cfg['dataloader']['batch_size']}")
            model_main_params_cur.append(ts)
            exp_name = '_'.join(model_main_params_cur)
            exp_dir_path = osp.join(cfg['utils']['out_dir'], exp_name)
            exp(cfg,
                project_name='HeatNet',
                run_clear_ml=run_clear_ml,
                log_dir=exp_dir_path)

            first_exp = False

            out_dir_path = osp.join(exp_dir_path, 'results')
            test_exp(exp_dir_path, out_dir_path, num_samples_to_draw=100)
