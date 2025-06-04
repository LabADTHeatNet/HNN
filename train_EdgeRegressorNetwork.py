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

    # Утилитарные параметры
    utils = dict(
        server_name=server_name,
        out_dir='out_Yasn_Q',  # Выходная директория для всех результатов
        device=device,
        seed=42  # Фиксация случайности для воспроизводимости
    )

    node_attr = ['pos_x', 'pos_y', 'P', 'types']

    # Набор конифгураций датасета  #TODO: переделать выбор
    # === default ===
    exp_mode = None
    fp = 'data.pt'
    edge_attr = ['dP']

    # # === eaL ===
    # exp_mode = 'eaL'
    # fp = 'data_eaL.pt'
    # edge_attr = ['l']

    # # === eaLdPQ ===
    # exp_mode = 'eaLdpQ'
    # fp = 'data_eaLdPQ.pt'
    # edge_attr = ['l', 'dP', 'Q']

    # # === eaLdP ===
    # exp_mode = 'eaLdp'
    # fp = 'data_eaLdP.pt'
    # edge_attr = ['l', 'dP']

    # # === naQ ===
    # exp_mode = 'naQ'
    # fp = 'data_naQ.pt'
    # node_attr = ['pos_x', 'pos_y', 'P', 'Q', 'types']
    # edge_attr = ['dP']

    # # === eadQ ===
    # exp_mode = 'eadQ'
    # fp = 'data_eadQ.pt'
    # node_attr = ['pos_x', 'pos_y', 'P', 'types_def', 'types_usr', 'types_src']
    # edge_attr = ['l', 'dP', 'dQ', 'Q_out', 'Q_in']

    # Параметры датасета
    dataset = dict(
        datasets_dir=osp.join(root_dir, 'datasets'),  # Путь к данным
        name='database_Yasn_Q',  # Имя датасета
        load=True,  # Загружать предобработанный датасет из файла
        fp=fp,  # Файл предобработанного датасета
        node_attr=node_attr,  # Атрибуты узлов
        edge_attr=edge_attr,  # Атрибуты ребер
        edge_label=['d'],  # Целевые метки ребер
        scaler_fn='StandardScaler',  # Метод нормализации данных (None/MinMaxScaler/RobustScaler/StandardScaler)
        num_samples=None  # Ограничение количества выборок (None для всех)
    )

    # Параметры загрузчиков данных
    dataloader = dict(
        train_ratio=0.7,  # Доля обучающих данных
        val_ratio=0.15,  # Доля валидационных данных
        batch_size=16,  # Размер батча (будет задан позже)
    )

    # Параметры модели
    node_hidden_channels = 128
    num_node_layers = 8
    edge_hidden_channels = 128
    num_edge_layers = 8
    heads = 8
    dropout = 0.0
    jump_mode = 'cat'

    EdgeRegressorNetwork_Attr_model = dict(
        name='EdgeRegressorNetwork_Attr',
        kwargs=dict(
            # node_in_channels=node_in_channels,   # устанавливается в exp_cls, = размеру входным данных
            # edge_in_channels=edge_in_channels,   # устанавливается в exp_cls, = размеру входных данных
            # out_channels=out_channels,           # устанавливается в exp_cls, = размеру выходных данных
            node_hidden_channels=node_hidden_channels,
            num_node_layers=num_node_layers,
            edge_hidden_channels=edge_hidden_channels,
            num_edge_layers=num_edge_layers,
            heads=heads,
            dropout=dropout,
            jump_mode=jump_mode)
    )
    model = EdgeRegressorNetwork_Attr_model
    
    # Параметры обучения
    init_lr = 1e-3
    final_lr = 1e-6
    epochs_num = 2000

    # Параметры оптимизатора
    optimizer = dict(
        name='RAdam',  # Название оптимизатора
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
        name='MSELoss',  # 'MSELoss',  # Среднеквадратичная ошибка
        kwargs=dict()
    )

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
        cfg['dataset']['num_samples'] = 600  # Ограничение данных
        cfg['train']['num_epochs'] = 100  # Сокращение эпох
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
    if cfg['dataset']['load'] is False:
        cfg['dataset']['fp'] = osp.join(exp_dir_path, 'data.pt')
    exp(cfg,
        project_name='HeatNet',
        run_clear_ml=run_clear_ml,
        log_dir=exp_dir_path)

    # Тестирование модели
    results_dir_path = osp.join(exp_dir_path, 'results')
    test_exp(exp_dir_path,
             results_dir_path,
             num_samples_to_draw=num_samples_to_draw)
