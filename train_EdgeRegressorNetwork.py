from pathlib import Path
import importlib
import copy
import json
import tqdm
import torch
from torchinfo import summary

import copy
import os.path as osp
import torch

import pprint

from exp import (
    exp,
    test_exp
)
from src.utils import get_str_timestamp


server_name = 'seth'
root_dir = '.'

if __name__ == '__main__':
    # %% Конфигурация эксперимента
    debug_run = False  # Режим отладки (уменьшает размер данных и длительность обучения)
    run_clear_ml = False  # Интеграция с ClearML для трекинга экспериментов
    num_samples_to_draw = 5  # Количество примеров для визуализации после теста

    # Определение устройства (GPU/CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # %% Конфигурации компонентов эксперимента
    # Общие утилиты
    utils = dict(
        server_name=server_name,
        out_dir='out_Yasn_Q',  # Выходная директория для результатов
        device=device,
        seed=42  # Фиксация случайности для воспроизводимости
    )

    # # === default ===
    # exp_mode = None
    # fp = 'data.pt'
    # edge_attr = ['dP']

    # === eaL ===
    exp_mode = 'eaL'
    fp = 'data_eaL.pt'
    edge_attr = ['l']

    # # === eaLdPQ ===
    # exp_mode = 'eaLdpQ'
    # fp = 'data_eaLdPQ.pt'
    # edge_attr = ['l', 'dP', 'Q']

    # # === eaLdP ===
    # exp_mode = 'eaLdp'
    # fp = 'data_eaLdP.pt'
    # edge_attr = ['l', 'dP']

    # Настройки датасета
    dataset = dict(
        datasets_dir=osp.join(root_dir, 'datasets'),  # Путь к данным
        name='database_Yasn_Q',  # Имя датасета
        load=False,  # Загружать предобработанный датасет из файла
        fp=fp,  # Файл предобработанного датасета
        node_attr=['pos_x', 'pos_y', 'P', 'types'],  # Атрибуты узлов
        edge_attr=edge_attr,  # Атрибуты ребер
        # edge_label=['d', 'vel'],  # Целевые метки ребер
        edge_label=['d'],  # Целевые метки ребер
        # edge_label=['mod'],  # Целевые метки ребер
        # scaler_fn=None,  # Метод нормализации данных (None/MinMaxScaler/RobustScaler)
        num_samples=None  # Ограничение количества выборок (None для всех)
    )

    # Настройки загрузчиков данных
    dataloader = dict(
        train_ratio=0.7,  # Доля обучающих данных
        val_ratio=0.15,  # Доля валидационных данных
        batch_size=None,  # Размер батча (будет задан позже)
    )
    # Параметры обучения

    init_lr = 1e-3
    final_lr = 1e-6
    epochs_num = 2000

    # Настройки оптимизатора
    # optimizer = dict(
    #     name='AdamW',  # Название оптимизатора
    #     kwargs=dict(
    #         lr=init_lr,  # Скорость обучения
    #         weight_decay=1e-3  # L2-регуляризация
    #     )
    # )

    optimizer = dict(
        name='RAdam',  # Название оптимизатора
        kwargs=dict(
            lr=init_lr,  # Скорость обучения
            betas=(0.9, 0.99),  # стандартные моменты
            eps=1e-8,            # небольшая цифра для числовой стабильности
            weight_decay=1e-6    # чуть поменьше, чем у AdamW — чтобы не переточить сеть
        )
    )

    # Настройки планировщика скорости обучения
    scheduler = dict(
        name='StepLR',  # Стратегия изменения lr
        kwargs=dict(
            step_size=1,  # Шаг уменьшения lr
            gamma=pow(final_lr / init_lr, 1 / epochs_num)  # Множитель уменьшения lr
        )
    )

    # scheduler = dict(
    #     name='CosineAnnealingLR',  # Стратегия изменения lr
    #     kwargs=dict(
    #         T_max=epochs_num,
    #     )
    # )

    # Функция потерь
    criterion = dict(
        name='MSELoss',  # 'MSELoss',  # Среднеквадратичная ошибка
        kwargs=dict()
    )

    # # Функция потерь
    # criterion = dict(
    #     name='FocalRegressionLoss',
    #     kwargs=dict(
    #         alpha=2.0,
    #         gamma=2.0
    #     )
    # )

    # Параметры обучения
    train = dict(
        num_epochs=epochs_num,  # Количество эпох
        score_metric='Loss'  # Метрика для выбора лучшей модели
    )

    node_hidden_channels = 64
    num_node_layers = 4
    edge_hidden_channels = 64
    num_edge_layers = 4
    heads = 4
    dropout = 0.0
    jump_mode = 'cat'

    EdgeRegressorNetwork_Attr_model = dict(
        name='EdgeRegressorNetwork_Attr',
        kwargs=dict(
            # in_node_dim=in_channels,
            # in_edge_dim=edge_in_channels,
            node_hidden_channels=node_hidden_channels,
            num_node_layers=num_node_layers,
            edge_hidden_channels=edge_hidden_channels,
            num_edge_layers=num_edge_layers,
            heads=heads,
            dropout=dropout,
            # out_dim=out_channels,
            jump_mode=jump_mode)
    )
    EdgeRegressorNetwork_Attr_models_list = [EdgeRegressorNetwork_Attr_model]

    # Формирование списка конфигураций моделей
    model_list = list()

    model_list.extend(EdgeRegressorNetwork_Attr_models_list)

    # Параметры для перебора: размер батча и методы нормализации
    batch_size_list = [64]
    # scalers_list = ['MinMaxScaler', 'RobustScaler']
    # scalers_list = ['MinMaxScaler']
    scalers_list = ['StandardScaler']
    # scalers_list = [None]

    # Формирование всех комбинаций конфигураций
    cfg_list = [
        {
            "utils": copy.deepcopy(utils),
            "dataset": {**dataset, "scaler_fn": scaler_fn},
            "dataloader": {**dataloader, "batch_size": batch_size},
            "model": copy.deepcopy(model),
            "optimizer": copy.deepcopy(optimizer),
            "scheduler": copy.deepcopy(scheduler),
            "criterion": copy.deepcopy(criterion),
            "train": copy.deepcopy(train),
        }
        for model in model_list
        for batch_size in batch_size_list
        for scaler_fn in scalers_list
    ]

    # Запуск экспериментов
    for idx, cfg in enumerate(cfg_list):

        if idx != 0:
            cfg['dataset']['load'] = True  # Загружать датасет после первого эксперимента

        # Настройки для отладки
        if debug_run:
            run_clear_ml = False
            cfg['utils']['out_dir'] += '_test'
            cfg['dataset']['load'] = False
            cfg['dataset']['num_samples'] = 600  # Ограничение данных
            cfg['train']['num_epochs'] = 100  # Сокращение эпох
            num_samples_to_draw = 5  # Отключение визуализации

        # Формирование уникальных имен экспериментов

        exp_params = [
            f"{cfg['dataset']['scaler_fn']}",
            f"{cfg['model']['name']}",
        ]

        if exp_mode is not None:
            exp_params.insert(0, exp_mode)

        #  Добавление параметра размера батча
        exp_params.append(f"bs{cfg['dataloader']['batch_size']}")

        # Генерация уникального имени эксперимента с временной меткой
        ts = get_str_timestamp()
        exp_params.append(ts)
        exp_name = '_'.join(exp_params)

        # Вывод конфигурации
        pprint.pprint(cfg)

        # Запуск эксперимента
        exp_dir_path = osp.join(cfg['utils']['out_dir'], exp_name)
        if cfg['dataset']['load'] is False:
            cfg['dataset']['fp'] = osp.join(exp_dir_path, 'data.pt')
        exp(cfg,
            project_name='HeatNet',
            run_clear_ml=run_clear_ml,
            log_dir=exp_dir_path)

        # Тестирование модели
        out_dir_path = osp.join(exp_dir_path, 'results')
        test_exp(exp_dir_path,
                 out_dir_path,
                 num_samples_to_draw=num_samples_to_draw)
