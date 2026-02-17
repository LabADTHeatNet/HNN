import copy
import torch

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
    debug_run = False  # Режим отладки (уменьшает размер данных и длительность обучения)
    run_clear_ml = False  # Интеграция с ClearML для трекинга экспериментов
    num_samples_to_draw = 0  # Количество примеров для визуализации после теста

    # Определение устройства (GPU/CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'

    # Утилитарные параметры
    utils = dict(
        server_name=server_name,
        out_dir='out_Termo_final',  # Выходная директория для всех результатов
        device=device,
        seed=42  # Фиксация случайности для воспроизводимости
    )

    node_attr = ['pos_x', 'pos_y', 'P', 'types']

    # === Termo ===
    exp_mode = 'Termo'  # Режим эксперимента
    fp = 'data_Termo_Heat.pt'
    node_attr = ['pos_x', 'pos_y', 'types_def', 'types_usr', 'types_src', 'P', 'Temp', 'P_ideal', 'Temp_ideal']  # Атрибуты узлов
    edge_attr = ['d', 'l', 'Vid_fwd', 'Vid_bwd', 'Vid_usr']  # Атрибуты ребер
    in_global_dim = 5  # Размерность глобальных параметров (например, для Termo: [t_outside, q_out_node, t_out_node, t_in_node])

    # Параметры датасета
    dataset = dict(
        datasets_dir=osp.join(root_dir, 'datasets'),  # Путь к данным
        name='Termo_model_fwd_and_bwd',  # Имя датасета\
        load=True,  # Загружать предобработанный датасет из файла
        fp=fp,  # Файл предобработанного датасета
        node_attr=node_attr,  # Атрибуты узлов
        edge_attr=edge_attr,  # Атрибуты ребер
        edge_label=['graph_label'],  # Целевые метки ребер
        scaler_fn='StandardScaler',  # Метод нормализации данных (None/MinMaxScaler/RobustScaler/StandardScaler)
        num_samples=None,  # Ограничение количества выборок (None для всех)
        add_ideal=True # Добавление идеального датасета (True/False)
    )

    # Параметры загрузчиков данных
    dataloader = dict(
        train_ratio=0.7,  # Доля обучающих данных
        val_ratio=0.15,  # Доля валидационных данных
        batch_size=16,  # Размер батча
    )

    # Параметры модели
    # node_hidden_channels = 64
    # num_node_layers = 4
    # edge_hidden_channels = 64
    # num_edge_layers = 4
    # heads = 4
    # dropout = 0.0
    # jump_mode = 'cat'

    node_hidden_channels = 128
    num_node_layers = 8
    edge_hidden_channels = 128
    num_edge_layers = 8
    heads = 4
    dropout = 0.2
    jump_mode = 'cat'
    out_dim= 44
    
    EdgeClassifierNetwork_Attr_model = dict(
        name='EdgeClassifierNetwork_Attr',
        kwargs=dict(
            # node_in_channels=node_in_channels,   # устанавливается в exp_cls, = размеру входным данных
            # edge_in_channels=edge_in_channels,   # устанавливается в exp_cls, = размеру входных данных
            out_dim=out_dim,           # устанавливается в exp_cls, = размеру выходных данных
            in_global_dim=in_global_dim,
            node_hidden_channels=node_hidden_channels,
            num_node_layers=num_node_layers,
            edge_hidden_channels=edge_hidden_channels,
            num_edge_layers=num_edge_layers,
            heads=heads,
            dropout=dropout,
            jump_mode=jump_mode,
            use_edge_attention=True,
            )
    )
    model = EdgeClassifierNetwork_Attr_model

    # Параметры обучения
    init_lr = 1e-3
    final_lr = 1e-6
    epochs_num = 100
    defect_weight = 1.0    # базовый вес для классов с дефектами
    no_defect_weight = 0.5 # меньший вес для "нет дефекта"
    class_weights = [defect_weight for i in range(44)]
    class_weights[43] = no_defect_weight


    # defect_weight = 0.5    # базовый вес для классов с дефектами
    # no_defect_weight = 1.0 # меньший вес для "нет дефекта"
    # class_weights = [defect_weight for i in range(73)]
    # class_weights[72] = no_defect_weight
    # # Параметры оптимизатора
    # optimizer = dict(
    #     name='RAdam',  # Название оптимизатора
    #     kwargs=dict(
    #         lr=init_lr,  # Скорость обучения
    #         betas=(0.9, 0.99),  # стандартные моменты
    #         eps=1e-8,            # небольшая цифра для числовой стабильности
    #         weight_decay=1e-6    # чуть поменьше, чем у AdamW — чтобы не переточить сеть
    #     )
    # )

    # Параметры оптимизатора
    optimizer = dict(
        name='Adam',  # Название оптимизатора
        kwargs=dict(
            lr=init_lr,  # Скорость обучения
            betas=(0.9, 0.99),  # стандартные моменты
            eps=1e-8,            # небольшая цифра для числовой стабильности
            weight_decay=1e-6    # чуть поменьше, чем у AdamW — чтобы не переточить сеть
        )
    )
    
    # Параметры планировщика скорости обучения
    scheduler = dict(
        name='StepLR',  # Стратегия изменения lr
        kwargs=dict(
            step_size=1,  # Шаг уменьшения lr
            gamma=pow(final_lr / init_lr, 1 / epochs_num)  # Множитель уменьшения lr
        )
    )

    # Функция потерь
    criterion = dict(
        name='CrossEntropyLoss',
        kwargs=dict(
            weight= class_weights,
            label_smoothing=0.1,
            # gamma= 2.0,
        )
    )

    # criterion = dict(
    #     name='L1Loss',  # 'L1Loss',  # Средняя абсолютная ошибка
    #     kwargs=dict()
    # )
    
    # Параметры обучения
    train = dict(
        num_epochs=epochs_num,  # Количество эпох
        score_metric='Loss'  # Метрика для выбора лучшей модели
    )

    # Формирование конфигураций
    cfg = {
        "utils": copy.deepcopy(utils),
        "dataset": copy.deepcopy(dataset),
        "dataloader": copy.deepcopy(dataloader),
        "model": copy.deepcopy(model),
        "optimizer": copy.deepcopy(optimizer),
        "scheduler": copy.deepcopy(scheduler),
        "criterion": copy.deepcopy(criterion),
        "train": copy.deepcopy(train),
    }

    # Настройки для отладки
    if debug_run:
        run_clear_ml = False
        cfg['utils']['out_dir'] += '_test'
        cfg['dataset']['load'] = False  # Создаем свой уменьшеный датасет
        cfg['dataset']['fp'] = 'data_test.pt'  # Путь к тестовому датасету
        cfg['dataset']['num_samples'] = 60  # Ограничение данных
        cfg['train']['num_epochs'] = 10  # Сокращение эпох
        num_samples_to_draw = 0  # Отключение визуализации

    # Формирование уникальных имен экспериментов
    exp_params = [
        f"{cfg['dataset']['scaler_fn']}",
        f"{cfg['model']['name']}",
    ]
    if exp_mode is not None:
        exp_params.insert(0, exp_mode)
    #  Добавление параметра размера батча
    exp_params.append(f"bs{cfg['dataloader']['batch_size']}")
    exp_params.append(get_str_timestamp())  # Генерация уникального имени эксперимента с временной меткой
    exp_name = '_'.join(exp_params)

    # Вывод конфигурации
    pprint.pprint(cfg)

    # Запуск эксперимента
    exp_dir_path = osp.join(cfg['utils']['out_dir'], exp_name)
    # if cfg['dataset']['load'] is False:
    #     cfg['dataset']['fp'] = osp.join(exp_dir_path, 'data.pt')
    exp(cfg,
        project_name='HeatNet',
        run_clear_ml=run_clear_ml,
        log_dir=exp_dir_path)

    # Тестирование модели
    results_dir_path = osp.join(exp_dir_path, 'results')
    test_exp(exp_dir_path,
             results_dir_path,
             cfg,
             num_samples_to_draw=num_samples_to_draw)
