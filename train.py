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
    edge_label=['d', 'vel'],
    # scaler_fn="RobustScaler",
    scaler_fn="MinMaxScaler",
    num_samples=None
)

dataloader = dict(
    train_ratio=0.7,  # 70% - обучающая выборка
    val_ratio=0.15,   # 15% - валидационная выборка
    batch_size=None,
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
    num_epochs=500,
    score_metric='MSE'
)

# %% experiment
if __name__ == '__main__':
    debug_run = False
    run_clear_ml = True
    debug_out_dir = utils['out_dir'] + '_test'
    num_samples_to_draw = 20
    
    model_list = list()

    # model_name_list = ['FlatGCN', 'ParametricGCN', 'ParametricGCN_GlobalPool', 'uGCN']
    # conv_type_list = ['GCNConv', 'SAGEConv', 'GATConv']
    # global_pool_list = ['global_mean_pool', 'global_max_pool']

    node_conv_layer_list = [4*2**i for i in range(6)] + [256] * 10
    edge_fc_layer_list = [8*2**i for i in range(4)]
    out_fc_layer_list = [32*4**i for i in list(reversed(range(4)))]

    model = dict(
        name='uGCN',
        kwargs=dict(
            node_conv_layer_type='SAGEConv',
            node_conv_layer_list=node_conv_layer_list,
            node_conv_heads=1,
            node_conv_layer_kwargs=dict(
                aggr='mean'),
            node_global_pool_type='global_mean_pool',
            edge_fc_layer_list=edge_fc_layer_list,
            out_fc_layer_list=out_fc_layer_list,
            split_out_fc=None,
        )
    )
    # model = dict(
    #     name='uGCN',
    #     kwargs=dict(
    #         node_conv_layer_type='GATConv',
    #         node_conv_layer_list=node_conv_layer_list,
    #         node_conv_layer_kwargs=dict(
    #             heads=4),
    #         node_global_pool_type='global_mean_pool',
    #         edge_fc_layer_list=edge_fc_layer_list,
    #         out_fc_layer_list=out_fc_layer_list,
    #         split_out_fc=None,
    #     )
    # )
    model_main_params = [
        f"{dataset['scaler_fn']}",
        f"{model['name']}",
    ]
    if 'aggr' in model['kwargs']['node_conv_layer_kwargs']:
        model_main_params.append(f"{model['kwargs']['node_conv_layer_kwargs']['aggr']}")
    if 'heads' in model['kwargs']['node_conv_layer_kwargs']:
        model_main_params.append(f"{model['kwargs']['node_conv_layer_kwargs']['heads']}heads")

    for split_out_fc in [True]:
        model_cur = copy.copy(model)
        model_main_params_cur = copy.copy(model_main_params)

        model['kwargs']['split_out_fc'] = split_out_fc
        if model['kwargs']['split_out_fc']:
            model_main_params_cur.append('splited_out')
        model_list.append([model_cur, model_main_params_cur])

    # batch_size_list = [64, 32, 16, 8, 4]
    batch_size_list = [64, 32, 16, 8]

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
                cfg['utils']['out_dir'] = debug_out_dir
                cfg['dataset']['num_samples'] = 60
                cfg['train']['num_epochs'] = 10
                num_samples_to_draw = 0

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
            test_exp(exp_dir_path, out_dir_path, num_samples_to_draw=num_samples_to_draw)
