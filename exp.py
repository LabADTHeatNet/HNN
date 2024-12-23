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

from src.datasets import (
    prepare_data
)
from src.utils import (
    train,
    valid
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

    dataset, train_loader, val_loader, test_loader = prepare_data(
        cfg['dataset'], cfg['dataloader'], cfg['utils']['seed'])

    # Пример вывода батча
    for batch in train_loader:
        print("Пример батча:")
        print(batch)
        break  # Один батч для примера

    # Параметры модели
    node_in_dim = dataset[0].x.shape[1]
    edge_in_dim = dataset[0].edge_attr.shape[1]
    out_dim = 1  # Регрессия: одно значение для edge_label
    # Инициализация модели, оптимизатора и функции потерь

    model_fn = getattr(
        importlib.import_module(f'src.models.{cfg['model']['name']}'),
        cfg['model']['name'])
    model = model_fn(node_in_dim=node_in_dim,
                     edge_in_dim=edge_in_dim,
                     out_dim=out_dim,
                     **cfg['model']['kwargs'])

    with torch.no_grad():
        pred_tmp = model(batch.x, batch.edge_index, batch.edge_attr)
    print("Размер вывода:")
    print(pred_tmp.shape)

    print(model)
    summary(model)

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
        criterion_fn = getattr(importlib.import_module('torch.nn'), cfg['criterion']['name'])
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
    
    best_score = torch.inf
    with tqdm.tqdm(total=cfg['train']['num_epochs'], desc="Epochs", unit="epoch") as pbar:
        for epoch in range(0, cfg['train']['num_epochs']):
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

            train_metrics = train(model, train_loader, optimizer, criterion, device)
            for k, v in train_metrics.items():
                writer.add_scalar(f'{k}/train', v, epoch)
                # print(f"{' '.join(k.split('_')).title()}: {v:.2f}")

            valid_metrics = valid(model, val_loader, criterion, device)
            for k, v in valid_metrics.items():
                writer.add_scalar(f'{k}/val', v, epoch)
                # print(f"{' '.join(k.split('_')).title()}: {v:.2f}")

            # do something (save model, change lr, etc.)
            if best_score > valid_metrics[cfg['train']['score_metric']]:
                best_epoch = epoch
                best_score = valid_metrics[cfg['train']['score_metric']]
                torch.save(model, log_dir / 'best_model.pth')
                
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
    
    # Оценка на тестовом датасете
    test_metrics = valid(model, test_loader, criterion, device)
    print(f"Test: {'|'.join([f'{k} {v:7.1e}' for k, v in test_metrics.items()])}")