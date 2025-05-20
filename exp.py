from pathlib import Path
import os.path as osp
import importlib
import copy
import json
import tqdm

import numpy as np
import pandas as pd
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
    valid,
    weighted_mse_loss,
    FocalRegressionLoss,
    FocalLoss
)
from src.plots import (
    draw_data
)


def _log_metrics(metrics, suffix, writer, epoch):
    """Логирует метрики в TensorBoard с указанным префиксом."""
    for key, value in metrics.items():
        writer.add_scalar(f"{key}/{suffix}", value, epoch)


def exp(cfg, project_name='HeatNet', run_clear_ml=False, log_dir=None):
    """Основная функция запуска эксперимента: обучение и валидация модели."""
    # Создание директории для логов
    if log_dir is None:
        log_dir = 'tmp'
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Сохранение конфигурации в файл
    cfg_dump = copy.copy(cfg)
    with open(log_dir / 'params.json', 'w') as f:
        json.dump(cfg_dump, f, indent=4)

    device = torch.device(cfg['utils']['device'])  # Устройство для вычислений (GPU/CPU)

    # Подготовка данных
    dataset, scalers, train_loader, val_loader, test_loader = prepare_data(cfg['dataset'], cfg['dataloader'], cfg['utils']['seed'])

    # Пример вывода информации о батче
    for batch in train_loader:
        print("Пример батча:")
        print(batch)
        break

    # Инициализация модели
    in_node_dim = dataset[0].x.shape[1]  # Размерность признаков узлов
    in_edge_dim = dataset[0].edge_attr.shape[1]  # Размерность признаков ребер
    out_dim = dataset[0].edge_label.shape[1]  # Размерность целевых меток

    # Динамический импорт класса модели
    model_fn = getattr(
        importlib.import_module(f"src.models.{cfg['model']['name']}"),
        cfg['model']['name'])

    def create_model():
        return model_fn(
            in_node_dim=in_node_dim,
            in_edge_dim=in_edge_dim,
            out_dim=out_dim,
            **cfg['model']['kwargs']
        )
    model = create_model()
    model = model.to(device)

    # state = torch.load('/home/ivan/python/heatnet/out_Yasn_Q/StandardScaler_EdgeRegressorNetwork_Attr_bs16_20250516_184401/best_model.pth', weights_only=True)
    # model.load_state_dict(state)

    # Инициализация оптимизатора и планировщика
    optimizer_fn = getattr(importlib.import_module('torch.optim'), cfg['optimizer']['name'])
    optimizer = optimizer_fn(model.parameters(), **cfg['optimizer']['kwargs'])

    if cfg['scheduler']['name'] is not None:
        scheduler_fn = getattr(importlib.import_module('torch.optim.lr_scheduler'), cfg['scheduler']['name'])
        scheduler = scheduler_fn(optimizer, **cfg['scheduler']['kwargs'])
    else:
        scheduler = None

    # Функция потерь
    if cfg['criterion']['name'] is not None:
        if cfg['criterion']['name'] == 'FocalRegressionLoss':
            criterion_fn = FocalRegressionLoss
        if cfg['criterion']['name'] == 'FocalLoss':
            criterion_fn = FocalLoss
        elif cfg['criterion']['name'] == 'weighted_mse_loss':
            criterion_fn = weighted_mse_loss
        else:
            criterion_fn = getattr(importlib.import_module('torch.nn'), cfg['criterion']['name'])
        if 'pos_weight' in cfg['criterion']['kwargs']:
            cfg['criterion']['kwargs']['pos_weight'] = torch.Tensor(
                cfg['criterion']['kwargs']['pos_weight']
            ).to(device)
        if 'weight' in cfg['criterion']['kwargs']:
            cfg['criterion']['kwargs']['weight'] = torch.Tensor(
                cfg['criterion']['kwargs']['weight']
            ).to(device)
        criterion = criterion_fn(**cfg['criterion']['kwargs'])
    else:
        criterion = None

    # Проверка формы вывода модели
    with torch.no_grad():
        pred_tmp = model(batch.to(device))
        tmp_loss = criterion(pred_tmp, batch.edge_label)
    print("Размер вывода модели:", pred_tmp.shape)
    print("Тестовый лосс:", tmp_loss)

    # Интеграция с ClearML
    if run_clear_ml:
        task = Task.init(
            project_name=project_name,
            task_name=str(log_dir),
            output_uri=False
        )
        task.connect(cfg_dump)  # Логирование параметров
        output_model = OutputModel(task=task)
        output_model.update_design(config_dict=cfg_dump.get('model'))
    else:
        task = None

    # Логирование в TensorBoard
    writer = SummaryWriter(log_dir=log_dir)

    summary(model)
    print(model)

    edge_label_scaler = scalers['edge_label_scaler']  # Скейлер для меток

    # Обучение модели
    best_score = torch.inf  # Лучшее значение метрики
    with tqdm.tqdm(total=cfg['train']['num_epochs'], desc="Epochs", unit="epoch") as pbar:
        for epoch in range(cfg['train']['num_epochs']):
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)  # Логирование lr

            # Обучение на тренировочных данных
            train_metrics = train(model, train_loader, optimizer, criterion, device, scaler=edge_label_scaler, max_norm=1e-2)
            # Валидация
            valid_metrics = valid(model, val_loader, criterion, device, scaler=edge_label_scaler)

            # Логирование метрик
            _log_metrics(train_metrics, "train", writer, epoch)
            _log_metrics(valid_metrics, "val", writer, epoch)

            # Сохранение лучшей модели
            if best_score > valid_metrics[cfg['train']['score_metric']]:
                best_epoch = epoch
                best_score = valid_metrics[cfg['train']['score_metric']]
                torch.save(model.state_dict(), log_dir / 'best_model.pth')

            # Обновление lr
            if scheduler is not None:
                scheduler.step()

            # Обновление прогресс-бара
            pbar.set_postfix({
                'best': f'{best_epoch+1:04d}',
                'LR': f'{optimizer.param_groups[0]["lr"]:7.1e}',
                'Train': '|'.join([f'{k} {v:7.1e}' for k, v in train_metrics.items()]),
                'Val': '|'.join([f'{k} {v:7.1e}' for k, v in valid_metrics.items()]),
            })
            pbar.update(1)

    # Загрузка лучшей модели для тестирования
    state_dict = torch.load(log_dir / 'best_model.pth', weights_only=True)
    model = create_model()
    model.load_state_dict(state_dict)
    model = model.to(device)

    # Оценка на тестовых данных
    test_metrics = valid(model, test_loader, criterion, device, scaler=edge_label_scaler)
    _log_metrics(test_metrics, "test", writer, 0)

    print(f"Тест: {'|'.join([f'{k} {v:7.1e}' for k, v in test_metrics.items()])}")

    writer.close()
    if run_clear_ml:
        task.close()  # Завершение задачи ClearML


def test_exp(exp_dir_path, out_dir_path, num_samples_to_draw=None):
    """Тестирование модели и сохранение результатов."""
    exp_dir_path = Path(exp_dir_path)
    out_dir_path = Path(out_dir_path)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # 1) Загрузка конфигурации
    with open(exp_dir_path / 'params.json', 'r') as f:
        cfg = json.load(f)
    device = torch.device(cfg['utils']['device'])

    # 2) Подготовка данных
    cfg['dataset']['load'] = True
    # cfg['dataset']['fp'] = osp.join(exp_dir_path, 'data.pt')
    dataset, scalers, _, _, test_loader = prepare_data(
        cfg['dataset'],
        cfg['dataloader'],
        cfg['utils']['seed']
    )

    # 3) Пример батча
    for batch in test_loader:
        print("Пример тестового батча:")
        print(batch)
        break

    # 4) Инициализация модели
    in_node_dim = dataset[0].x.shape[1]
    in_edge_dim = dataset[0].edge_attr.shape[1]
    out_dim = dataset[0].edge_label.shape[1]
    model_module = importlib.import_module(f"src.models.{cfg['model']['name']}")
    ModelClass = getattr(model_module, cfg['model']['name'])

    def create_model():
        return ModelClass(
            in_node_dim=in_node_dim,
            in_edge_dim=in_edge_dim,
            out_dim=out_dim,
            **cfg['model']['kwargs']
        )

    model = create_model().to(device)

    # 5) Функция потерь
    if cfg['criterion']['name'] is not None:
        if cfg['criterion']['name'] == 'FocalRegressionLoss':
            criterion_fn = FocalRegressionLoss
        if cfg['criterion']['name'] == 'FocalLoss':
            criterion_fn = FocalLoss
        elif cfg['criterion']['name'] == 'weighted_mse_loss':
            criterion_fn = weighted_mse_loss
        else:
            criterion_fn = getattr(importlib.import_module('torch.nn'), cfg['criterion']['name'])
        if 'pos_weight' in cfg['criterion']['kwargs']:
            cfg['criterion']['kwargs']['pos_weight'] = torch.Tensor(
                cfg['criterion']['kwargs']['pos_weight']
            ).to(device)
        if 'weight' in cfg['criterion']['kwargs']:
            cfg['criterion']['kwargs']['weight'] = torch.Tensor(
                cfg['criterion']['kwargs']['weight']
            ).to(device)
        criterion = criterion_fn(**cfg['criterion']['kwargs'])
    else:
        criterion = None

    # 6) Загрузка весов
    state = torch.load(exp_dir_path / 'best_model.pth', weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # 7) Регрессионная оценка
    edge_label_scaler = scalers['edge_label_scaler']
    test_metrics = valid(model, test_loader, criterion, device,
                         scaler=edge_label_scaler)
    print("Тест (регрессия): " +
          "|".join(f"{k}={v:.3e}" for k, v in test_metrics.items()))

    # 8) Собираем предсказания по-ребру
    all_data = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            preds = model(batch)
            offset = 0
            for d in batch.to_data_list():
                n_e = d.edge_label.shape[0]
                d.edge_label_pred = preds[offset:offset+n_e].cpu()
                offset += n_e
                all_data.append(d.cpu())

    # 9) Вычисляем идеальные диаметры для каждого ребра (где edge_moded==0)
    E = all_data[0].edge_label.shape[0]
    sum_dia = torch.zeros(E)
    count_dia = torch.zeros(E)
    for d in tqdm.tqdm(all_data, desc="Обработка классов ребер"):
        _, edges_df = data_to_tables(
            d,
            node_attr=cfg['dataset']['node_attr'],
            edge_attr=cfg['dataset']['edge_attr'],
            edge_label=cfg['dataset']['edge_label'],
            scalers=scalers,
            edge_label_pred=[f"{v}_pred" for v in cfg['dataset']['edge_label']]
        )
        true_vals = torch.tensor(
            edges_df[cfg['dataset']['edge_label']].values,
            dtype=torch.float
        )
        mask0 = (d.edge_moded == 0).squeeze()
        sum_dia[mask0] += true_vals[mask0, 0]
        count_dia[mask0] += 1
    ideal_dia = (sum_dia / count_dia.clamp(min=1)).numpy()

    # 10) Переменные для graph-level метрик
    nn_list = []         # N | N
    nd_list = []         # N | D
    dn_list = []         # D | N
    dd_list = []         # D | D (correct)
    dd_wrong_list = []   # D | D (wrong)

    # 11) Обработка каждого примера
    for idx, d in enumerate(tqdm.tqdm(all_data, desc="Обработка примеров")):
        sample_name = Path(d.nodes_fp).stem

        # a) получаем таблицы
        nodes_df, edges_df = data_to_tables(
            d,
            node_attr=cfg['dataset']['node_attr'],
            edge_attr=cfg['dataset']['edge_attr'],
            edge_label=cfg['dataset']['edge_label'],
            scalers=scalers,
            edge_label_pred=[f"{v}_pred" for v in cfg['dataset']['edge_label']]
        )

        denorm = d.clone().cpu()
        denorm.x = torch.tensor(
            nodes_df[cfg['dataset']['node_attr']].values, dtype=torch.float)
        denorm.edge_attr = torch.tensor(
            edges_df[cfg['dataset']['edge_attr']].values, dtype=torch.float)
        denorm.edge_label = torch.tensor(
            edges_df[cfg['dataset']['edge_label']].values, dtype=torch.float)
        denorm.edge_label_pred = torch.tensor(
            edges_df[[f'{v}_pred' for v in cfg['dataset']['edge_label']]].values,
            dtype=torch.float
        )
        d = denorm

        # b) true и pred значения
        true_vals = d.edge_label[..., 0].numpy()
        pred_vals = d.edge_label_pred[..., 0].numpy()
        true_dev = np.abs(true_vals - ideal_dia) / ideal_dia * 100.0
        pred_dev = np.abs(pred_vals - ideal_dia) / ideal_dia * 100.0

        # true_dev = np.abs(true_vals - ideal_dia)
        # pred_dev = np.abs(pred_vals - ideal_dia)

        # c) edge-level модед
        pm = np.zeros_like(pred_dev, dtype=int)
        # pm[pred_dev > 5.0] = 1
        pm[pred_dev > 5.0] = 2

        # d) сохраняем CSV
        edges_df['moded'] = d.edge_moded.numpy()
        edges_df['dev'] = true_dev
        edges_df['pred_dev'] = pred_dev
        edges_df['pred_moded'] = pm
        out_nodes_path = out_dir_path / Path(d.nodes_fp).with_suffix('.csv').name
        out_edges_path = out_dir_path / Path(d.edges_fp).with_suffix('.csv').name
        nodes_df.to_csv(out_nodes_path, index=False)
        edges_df.to_csv(out_edges_path, index=False)

        # e) визуализация
        k = 2
        figsize = (int(16*k), int(8*k))

        def draw_data_formated(data, pos_idxs):
            return draw_data(
                data,
                pos_idxs,
                node_color_idx=0,
                node_color_label='P',
                edge_data_from_label=True,
                edge_color_idx=-1,
                edge_color_label='d',
                additional_node_label_idx=0,
                figsize=figsize,
                no_draw=False,
                font_size=10,
                arrows=True,
                arrowstyle='-|>',
                alt_pos=True
            )

        if num_samples_to_draw and idx < num_samples_to_draw:
            moded_idx_list, fig, ax, log_str_list = draw_data_formated(
                d,
                pos_idxs=2,
            )
            fig.savefig(out_dir_path / f"{sample_name}.png", bbox_inches='tight')
            plt.close(fig)

        # f) graph-level классификация
        true_idxs = np.where(d.edge_moded.numpy() == 2)[0].tolist()
        pred_idxs = np.where(pm == 2)[0].tolist()

        def fmt(i):
            u, v = int(d.edge_index[0, i]), int(d.edge_index[1, i])
            itdev, ipdev = true_dev[i], pred_dev[i]
            return i, u, v, itdev, ipdev

        def print_fmt(i, u, v, itdev, ipdev):
            str_fmt = f'{i}({u}-{v}), true_dev={itdev:.2f}, pred_dev={ipdev:.2f}'
            return str_fmt

        if not true_idxs and not pred_idxs:
            # N | N
            nn_list.append([sample_name, [], []])

        elif not true_idxs and pred_idxs:
            # N | D
            nd_list.append([sample_name, [], [fmt(i) for i in pred_idxs]])

        elif true_idxs and not pred_idxs:
            # D | N
            dn_list.append([sample_name, [fmt(i) for i in true_idxs], []])

        else:
            hit = set(true_idxs) & set(pred_idxs)
            if hit:
                # D | D (correct)
                dd_list.append([sample_name, [fmt(i) for i in true_idxs], [fmt(i) for i in pred_idxs]])
            else:
                # D | D (wrong)
                dd_wrong_list.append([sample_name, [fmt(i) for i in true_idxs], [fmt(i) for i in pred_idxs]])

    nn_list.sort(key=lambda v: v[0])
    nd_list.sort(key=lambda v: v[0])
    dn_list.sort(key=lambda v: v[0])
    dd_list.sort(key=lambda v: v[0])
    dd_wrong_list.sort(key=lambda v: v[0])

    # После цикла печатаем все вместе
    # print()
    print(f"=== N | N ({len(nn_list)} samples)===")
    # for sample_name, true_idxs, pred_idxs in nn_list:
    #     print(f'{sample_name}: '
    #           f'OK'
    #           )

    # print()
    print(f"=== N | D ({len(nd_list)} samples)===")
    # for sample_name, true_idxs, pred_idxs in nd_list:
    #     print(f'{sample_name}: '
    #           f'pred_defect {[print_fmt(*v) for v in pred_idxs]} '
    #           )

    # print()
    print(f"=== D | N ({len(dn_list)} samples)===")
    # for sample_name, true_idxs, pred_idxs in dn_list:
    #     print(f'{sample_name}: '
    #           f'true_defect {[print_fmt(*v) for v in true_idxs]} '
    #           )

    # print()
    print(f"=== D | D ({len(dd_list)} samples)===")
    # for sample_name, true_idxs, pred_idxs in dd_list:
    #     if len(true_idxs) != len(pred_idxs):
    #         print(f'{sample_name}: '
    #               f'true_defect {[print_fmt(*v) for v in true_idxs]} '
    #               f'pred_defect {[print_fmt(*v) for v in pred_idxs]} '
    #               )

    # print()
    print(f"=== D | D wrong ({len(dd_wrong_list)} samples)===")
    # for sample_name, true_idxs, pred_idxs in dd_wrong_list:
    #     print(f'{sample_name}: '
    #           f'true_defect {[print_fmt(*v) for v in true_idxs]} '
    #           f'pred_defect {[print_fmt(*v) for v in pred_idxs]} '
    #           )

    cm = [[len(nn_list), len(nd_list), 0],
          [len(dn_list), len(dd_list), len(dd_wrong_list)]]
    cm = np.array(cm, dtype=np.int32)

    def print_cm(cm, y_labels=None, x_labels=None, col_width=5):
        fr = 2
        if np.issubdtype(cm.dtype, np.integer):
            format_str = '{val:>{col_width}}'
        else:
            format_str = '{val:{col_width}.{fr}f}'
        format_h_str = '{val:>{col_width}}'
        print_str = ''
        if y_labels is None:
            y_labels = [''] * cm.shape[0]
        for row, yl in zip(cm, y_labels):
            row_str = f'{format_h_str.format(val=str(yl)[:col_width], col_width=col_width)} | '
            for col in row:
                row_str += f'{format_str.format(val=col, col_width=col_width, fr=fr)} '
            print_str += f'{row_str}\n'
        if x_labels is not None:
            header_str = f'{format_h_str.format(val="", col_width=col_width)}   '
            for xl in x_labels:
                header_str += f'{format_h_str.format(val=str(xl)[:col_width], col_width=col_width)} '
            print_str += header_str
        print(print_str)

    print('CM')
    print_cm(cm, ['N', 'D'], ['N', 'D', 'Dwr'], col_width=5)

    cm_norm = cm.copy().astype(np.float32)
    cm_norm /= cm.sum(axis=1)[:, np.newaxis]

    print('CM (norm, row)')
    print_cm(cm_norm, ['N', 'D'], ['N', 'D', 'Dwr'], col_width=5)

    cm_norm_all = cm.copy().astype(np.float32)
    cm_norm_all /= cm.sum()

    print('CM (norm, all)')
    print_cm(cm_norm_all, ['N', 'D'], ['N', 'D', 'Dwr'], col_width=5)

    errors = []
    # i, u, v, itdev, ipdev
    errors += [['N', true_idxs[0][3], true_idxs[0][4]] for _, true_idxs, _ in dn_list]
    errors += [['D', true_idxs[0][3], true_idxs[0][4]] for _, true_idxs, _ in dd_list]

    errors_df = pd.DataFrame(errors, columns=['class', 'true_dev', 'pred_dev'])
    en_df = errors_df[errors_df['class'] == 'N']
    ed_df = errors_df[errors_df['class'] == 'D']
    
    # Определяем общий диапазон и одинаковые бины
    min_dev = errors_df['true_dev'].min()
    max_dev = errors_df['true_dev'].max()
    n_bins  = 100
    bins    = np.linspace(min_dev, max_dev, n_bins + 1)
    
    fig, ax = plt.subplots(figsize=(8,5))
    ax.hist(en_df['true_dev'], bins=bins, color='red',   alpha=0.5, label='N')
    ax.hist(ed_df['true_dev'], bins=bins, color='green', alpha=0.5, label='D')

    ax.set_xlabel('true_dev')
    ax.set_ylabel('Frequency')
    ax.set_title('Распределение true_dev по классам')
    ax.legend()
    
    plt.show()

