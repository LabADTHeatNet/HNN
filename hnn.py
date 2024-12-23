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

if __name__ == '__main__':
    test = False

    # Set config
    run_clear_ml = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    utils = dict(
        server_name=server_name,
        out_dir='out',
        device=device,
        seed=42
    )

    dataset = dict(
        datasets_dir=osp.join(root_dir, 'datasets'),
        name='straight_base',
        load=False,
        fp='pyg_dataset.pt',
        # Пример: другие атрибуты добавьте сюда
        node_attr=['pos_x', 'pos_y', 'P', 'types'],
        # Пример: укажите ваши атрибуты рёбер
        edge_attr=['dP'],
        # Имя колонки, используемой для edge_label
        edge_label='d',
    )

    dataloader = dict(
        train_ratio=0.7,  # 70% - обучающая выборка
        val_ratio=0.15,   # 15% - валидационная выборка
        batch_size=4,
    )

    model = dict(
        name='EdgeRegressionNetSAGE',
        kwargs=dict(
            hidden_dim=8,
            heads=4,
            aggr='lstm'
        ),
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
        cfg['dataloader']['batch_size'] = 1
        cfg['utils']['out_dir'] = 'out_test'
        
    ts = get_str_timestamp()
    log_dir = osp.join(cfg['utils']['out_dir'], f"{cfg['model']['name']}_{ts}")
    exp(cfg,
        project_name='HeatNet',
        run_clear_ml=run_clear_ml,
        log_dir=log_dir)
