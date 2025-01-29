from pathlib import Path
import importlib
import copy
import json
import tqdm
from pprint import pprint

import torch
from torchinfo import summary

from torch.utils.tensorboard import SummaryWriter
from clearml import (
    Task,
    OutputModel
)

import matplotlib.pyplot as plt

from src.datasets import (
    prepare_data,
    data_to_tables
)
from src.utils import (
    train,
    valid
)
from src.plots import (
    draw_data
)


def exp(cfg, project_name='HeatNet', run_clear_ml=False, log_dir=None):

    if log_dir is None:
        log_dir = 'tmp'
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    pprint(cfg)
    cfg_dump = copy.copy(cfg)
    with open(log_dir / 'params.json', 'w') as f:
        json.dump(cfg_dump, f, indent=4)

    device = torch.device(cfg['utils']['device'])

    dataset, scalers, train_loader, val_loader, test_loader = prepare_data(
        cfg['dataset'], cfg['dataloader'], cfg['utils']['seed'])

    # Пример вывода батча
    for batch in train_loader:
        print("Пример батча:")
        print(batch)
        break  # Один батч для примера

    # Параметры модели
    in_node_dim = dataset[0].x.shape[1]
    in_edge_dim = dataset[0].edge_attr.shape[1]
    # Регрессия: одно значение для edge_label
    out_dim = dataset[0].edge_label.shape[1]
    # Инициализация модели, оптимизатора и функции потерь

    model_fn = getattr(
        importlib.import_module(f"src.models.{cfg['model']['name']}"),
        cfg['model']['name'])

    def create_model():
        return model_fn(in_node_dim=in_node_dim,
                        in_edge_dim=in_edge_dim,
                        out_dim=out_dim,
                        **cfg['model']['kwargs'])
    model = create_model()
    with torch.no_grad():
        pred_tmp = model(batch)
    print("Размер вывода:")
    print(pred_tmp.shape)

    print(model)
    # summary(model)

    optimizer_fn = getattr(importlib.import_module(
        'torch.optim'), cfg['optimizer']['name'])
    optimizer = optimizer_fn(model.parameters(), **cfg['optimizer']['kwargs'])

    if cfg['scheduler']['name'] is not None:
        scheduler_fn = getattr(importlib.import_module(
            'torch.optim.lr_scheduler'), cfg['scheduler']['name'])
        scheduler = scheduler_fn(optimizer, **cfg['scheduler']['kwargs'])
    else:
        scheduler = None

    if cfg['criterion']['name'] is not None:
        criterion_fn = getattr(importlib.import_module(
            'torch.nn'), cfg['criterion']['name'])
        criterion = criterion_fn(**cfg['criterion']['kwargs'])
    else:
        criterion = None

    if run_clear_ml:
        task = Task.init(project_name=project_name,
                         task_name=str(log_dir),
                         output_uri=False)

        model_p_dump = cfg_dump.get('model')
        task.connect(cfg_dump)
        output_model = OutputModel(task=task)
        output_model.update_design(config_dict=model_p_dump)
    else:
        task = None
    writer = SummaryWriter(log_dir=log_dir)

    model = model.to(device)

    edge_label_scaler = scalers['edge_label_scaler']

    best_score = torch.inf
    with tqdm.tqdm(total=cfg['train']['num_epochs'], desc="Epochs", unit="epoch") as pbar:
        for epoch in range(0, cfg['train']['num_epochs']):
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

            train_metrics = train(model, train_loader,
                                  optimizer, criterion, device, scaler=edge_label_scaler)
            for k, v in train_metrics.items():
                writer.add_scalar(f'{k}/train', v, epoch)

            valid_metrics = valid(
                model, val_loader, criterion, device, scaler=edge_label_scaler)
            for k, v in valid_metrics.items():
                writer.add_scalar(f'{k}/val', v, epoch)

            # do something (save model, change lr, etc.)
            if best_score > valid_metrics[cfg['train']['score_metric']]:
                best_epoch = epoch
                best_score = valid_metrics[cfg['train']['score_metric']]
                torch.save(model.state_dict(), log_dir / 'best_model.pth')

            lr = scheduler.get_last_lr()[0]
            if scheduler is not None:
                scheduler.step()

            pbar.set_postfix({
                'best': f'{best_epoch+1:04d}',
                'LR': f'{lr:7.1e}',
                'Train': '|'.join([f'{k} {v:7.1e}' for k, v in train_metrics.items()]),
                'Val': '|'.join([f'{k} {v:7.1e}' for k, v in valid_metrics.items()]),
            })
            pbar.update(1)  # Обновляем прогресс на одну эпоху

    state_dict = torch.load(log_dir / 'best_model.pth', weights_only=True)
    model = create_model()
    model.load_state_dict(state_dict)
    model = model.to(device)

    # Оценка на тестовом датасете
    test_metrics = valid(model, test_loader, criterion,
                         device, scaler=edge_label_scaler)
    for k, v in test_metrics.items():
        writer.add_scalar(f'{k}/test', v, 0)
    print(
        f"Test: {'|'.join([f'{k} {v:7.1e}' for k, v in test_metrics.items()])}")

    writer.close()
    if run_clear_ml:
        task.close()


def test_exp(exp_dir_path, out_dir_path, num_samples_to_draw=None):
    exp_dir_path = Path(exp_dir_path)
    out_dir_path = Path(out_dir_path)

    with open(exp_dir_path / 'params.json', 'r') as f:
        cfg = json.load(f)

    cfg['dataset']['load'] = True
    # cfg['dataset']['load'] = False
    # cfg['dataset']['num_samples'] = None
    # cfg['dataset']['fp'] = 'pyg_dataset_Yasn_Q_fp.pt'

    device = torch.device(cfg['utils']['device'])

    dataset, scalers, train_loader, val_loader, test_loader = prepare_data(
        cfg['dataset'], cfg['dataloader'], cfg['utils']['seed'])

    # Пример вывода батча
    for batch in test_loader:
        print("Пример батча:")
        print(batch)
        break  # Один батч для примера

    # Параметры модели
    in_node_dim = dataset[0].x.shape[1]
    in_edge_dim = dataset[0].edge_attr.shape[1]
    out_dim = dataset[0].edge_label.shape[1]
    # Инициализация модели, оптимизатора и функции потерь

    model_fn = getattr(
        importlib.import_module(f"src.models.{cfg['model']['name']}"),
        cfg['model']['name'])

    def create_model():
        return model_fn(in_node_dim=in_node_dim,
                        in_edge_dim=in_edge_dim,
                        out_dim=out_dim,
                        **cfg['model']['kwargs'])
    model = create_model()
    with torch.no_grad():
        pred_tmp = model(batch)
    print("Размер вывода:")
    print(pred_tmp.shape)

    print(model)

    if cfg['criterion']['name'] is not None:
        criterion_fn = getattr(importlib.import_module(
            'torch.nn'), cfg['criterion']['name'])
        criterion = criterion_fn(**cfg['criterion']['kwargs'])
    else:
        criterion = None

    state_dict = torch.load(exp_dir_path / 'best_model.pth', weights_only=True)
    model = create_model()
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    edge_label_scaler = scalers['edge_label_scaler']
    # Оценка на тестовом датасете
    test_metrics = valid(model, test_loader, criterion,
                         device, scaler=edge_label_scaler)
    print(
        f"Test: {'|'.join([f'{k} {v:7.1e}' for k, v in test_metrics.items()])}")

    all_data = list()
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            edge_pred = model(data)
            for idx, d in enumerate(data.to_data_list()):
                s = d.edge_label.shape[0]
                d.edge_label_pred = edge_pred[s*idx:s*(idx+1)]
                all_data.append(d)

    k = 2
    figsize = (int(16*k), int(8*k))

    def draw_data_formated(data, pos_idxs):
        return draw_data(data,
                         pos_idxs,
                         # node_size_idx=0,
                         # node_size_label='P',
                         node_color_idx=0,
                         node_color_label='P',
                         edge_data_from_label=True,
                         # edge_size_idx=0,
                         # edge_size_label='d',
                         edge_color_idx=-1,
                         edge_color_label='d',
                         additional_node_label_idx=0,
                         figsize=figsize,
                         no_draw=False,
                         font_size=10,
                         arrows=True,
                         arrowstyle='-|>',
                         alt_pos=True)

    out_dir_path.mkdir(exist_ok=True)

    for idx, d in enumerate(tqdm.tqdm(all_data)):

        nodes_df, edges_df = data_to_tables(d,
                                            node_attr=cfg['dataset']['node_attr'],
                                            edge_attr=cfg['dataset']['edge_attr'],
                                            edge_label=cfg['dataset']['edge_label'],
                                            scalers=scalers,
                                            edge_label_pred=[f'{v}_pred' for v in cfg['dataset']['edge_label']])
        out_nodes_path = out_dir_path / \
            Path(d.nodes_fp).with_suffix('.csv').name
        out_edges_path = out_dir_path / \
            Path(d.edges_fp).with_suffix('.csv').name

        nodes_df.to_csv(out_nodes_path)
        edges_df.to_csv(out_edges_path)

        if idx < num_samples_to_draw:
            moded_idx_list, fig, ax, log_str_list = draw_data_formated(
                d.cpu(), pos_idxs=2)

            fp = Path(d.nodes_fp)
            out_fig_path = out_dir_path / fp.with_suffix('.png').name
            fig.suptitle(fp.stem)
            fig.savefig(out_fig_path, bbox_inches='tight', pad_inches=0.1)

            # plt.show()
            plt.close(fig)
