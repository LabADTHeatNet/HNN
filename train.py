# %% imports and main parameters
import os.path as osp
import torch

from exp import exp
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
    batch_size=8,
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
    num_epochs=1000,
    score_metric='MSE'
)

# %% experiment
if __name__ == '__main__':
    test = False
    run_clear_ml = False

    model_list = list()

    # model_name_list = ['FlatGCN', 'ParametricGCN', 'ParametricGCN_GlobalPool']
    model_name_list = ['ParametricGCN', 'ParametricGCN_GlobalPool']
    conv_type_list = ['GCNConv', 'SAGEConv', 'GATConv']
    global_pool_list = ['global_mean_pool', 'global_max_pool']

    node_conv_layer_list = [4*2**i for i in range(6)] + [256] * 10
    edge_fc_layer_list = [8*2**i for i in range(4)]
    out_fc_layer_list = [32*4**i for i in list(reversed(range(4)))]

    # for model_name in model_name_list:
    #     for node_conv_layer_type in conv_type_list:
    #         if node_conv_layer_type == 'SAGEConv':
    #             node_conv_layer_kwargs = dict(
    #                 aggr='mean'
    #             )
    #             node_conv_heads = 1
    #         elif node_conv_layer_type == 'GATConv':
    #             node_conv_layer_kwargs = dict(
    #                 heads=4,
    #                 dropout=0.5
    #             )
    #             node_conv_heads = 4
    #         else:
    #             node_conv_layer_kwargs = dict()
    #             node_conv_heads = 1
    #
    #         if model_name == 'FlatGCN':
    #             model = dict(
    #                 name=model_name,
    #                 kwargs=dict(
    #                     hidden_dim=128,
    #                     node_conv_layer_type=node_conv_layer_type,
    #                     node_conv_heads=node_conv_heads,
    #                     node_conv_layer_kwargs=node_conv_layer_kwargs,
    #                     node_conv_layer_num=8,
    #                     edge_fc_layer_num=8,
    #                     out_fc_layers_num=4
    #                 )
    #             )
    #             model_main_params = [
    #                 f"{model['name']}",
    #                 f"{model['kwargs']['node_conv_layer_type']}",
    #             ]
    #             model_list.append([model, model_main_params])
    #         elif model_name == 'ParametricGCN':
    #             model = dict(
    #                 name=model_name,
    #                 kwargs=dict(
    #                     node_conv_layer_type=node_conv_layer_type,
    #                     node_conv_layer_list=node_conv_layer_list,
    #                     node_conv_heads=node_conv_heads,
    #                     node_conv_layer_kwargs=node_conv_layer_kwargs,
    #                     edge_fc_layer_list=edge_fc_layer_list,
    #                     out_fc_layer_list=out_fc_layer_list
    #                 )
    #             )
    #             model_main_params = [
    #                 f"{model['name']}",
    #                 f"{model['kwargs']['node_conv_layer_type']}",
    #             ]
    #             model_list.append([model, model_main_params])
    #         elif model_name == 'ParametricGCN_GlobalPool':
    #             for node_global_pool_type in global_pool_list:
    #                 model = dict(
    #                     name=model_name,
    #                     kwargs=dict(
    #                         node_conv_layer_type=node_conv_layer_type,
    #                         node_conv_layer_list=node_conv_layer_list,
    #                         node_conv_heads=node_conv_heads,
    #                         node_conv_layer_kwargs=node_conv_layer_kwargs,
    #                         node_global_pool_type=node_global_pool_type,
    #                         edge_fc_layer_list=edge_fc_layer_list,
    #                         out_fc_layer_list=out_fc_layer_list
    #                     )
    #                 )
    #                 model_main_params = [
    #                     f"{model['name']}",
    #                     f"{model['kwargs']['node_conv_layer_type']}",
    #                     f"{model['kwargs']['node_global_pool_type']}"
    #                 ]
    #                 model_list.append([model, model_main_params])

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
        f"{model['name']}",
        f"{model['kwargs']['node_conv_layer_type']}",
        f"{model['kwargs']['node_conv_layer_kwargs']['aggr']}",
    ]
    model_list.append([model, model_main_params])

    for model, model_main_params in model_list:
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

        if test:
            run_clear_ml = False
            cfg['train']['num_epochs'] = 10
            cfg['dataloader']['batch_size'] = 32
            cfg['utils']['out_dir'] += '_test'

        ts = get_str_timestamp()

        model_main_params.append(ts)
        exp_name = '_'.join(model_main_params)
        log_dir = osp.join(cfg['utils']['out_dir'], exp_name)
        exp(cfg,
            project_name='HeatNet',
            run_clear_ml=run_clear_ml,
            log_dir=log_dir)
