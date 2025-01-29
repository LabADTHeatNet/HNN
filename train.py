# %% Импорты и основные параметры
import copy
import os.path as osp
import torch

import pprint

from exp import (
    exp,
    test_exp
)
from src.utils import get_str_timestamp

# Флаг для работы на кластере CASCADE
CASCADE = False  # если запускать на кластере CASCADE, установить True

if CASCADE:
    server_name = 'CASCADE'
    TORCH_HUB_DIR = '/storage0/pia/python/hub/'  # Директория для хранения моделей torch.hub
    torch.hub.set_dir(TORCH_HUB_DIR)
    root_dir = '/storage0/pia/python/heatnet/'  # Корневая директория проекта на кластере
else:
    server_name = 'seth'  # Локальный сервер
    root_dir = '.'  # Текущая директория для локального запуска

# %% Конфигурация эксперимента
if __name__ == '__main__':
    debug_run = False  # Режим отладки (уменьшает размер данных и длительность обучения)
    run_clear_ml = True  # Интеграция с ClearML для трекинга экспериментов
    num_samples_to_draw = 20  # Количество примеров для визуализации после теста

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

    # Настройки датасета
    dataset = dict(
        datasets_dir=osp.join(root_dir, 'datasets'),  # Путь к данным
        name='database_Yasn_Q',  # Имя датасета
        load=False,  # Загружать предобработанный датасет из файла
        fp='pyg_dataset_Yasn_Q.pt',  # Файл предобработанного датасета
        node_attr=['pos_x', 'pos_y', 'P', 'types'],  # Атрибуты узлов
        edge_attr=['dP'],  # Атрибуты ребер
        edge_label=['d', 'vel'],  # Целевые метки ребер
        scaler_fn=None,  # Метод нормализации данных (None/MinMaxScaler/RobustScaler)
        num_samples=None  # Ограничение количества выборок (None для всех)
    )

    # Настройки загрузчиков данных
    dataloader = dict(
        train_ratio=0.7,  # Доля обучающих данных
        val_ratio=0.15,  # Доля валидационных данных
        batch_size=None,  # Размер батча (будет задан позже)
    )

    # Настройки оптимизатора
    optimizer = dict(
        name='AdamW',  # Название оптимизатора
        kwargs=dict(
            lr=1e-3,  # Скорость обучения
            weight_decay=1e-5  # L2-регуляризация
        )
    )

    # Настройки планировщика скорости обучения
    scheduler = dict(
        name='StepLR',  # Стратегия изменения lr
        kwargs=dict(
            step_size=10,  # Шаг уменьшения lr
            gamma=0.85  # Множитель уменьшения lr
        )
    )

    # Функция потерь
    criterion = dict(
        name='MSELoss',  # Среднеквадратичная ошибка
        kwargs=dict()
    )

    # Параметры обучения
    train = dict(
        num_epochs=500,  # Количество эпох
        score_metric='MSE'  # Метрика для выбора лучшей модели
    )

    model_list = list()  # Список конфигураций моделей

    # Генерация архитектурных параметров модели
    node_conv_layer_list = [4*2**i for i in range(6)] + [256] * 10  # Слои для конволюций узлов
    edge_fc_layer_list = [8*2**i for i in range(4)]  # Слои для ребер
    out_fc_layer_list = [32*4**i for i in list(reversed(range(4)))]  # Выходные слои

    # Базовая конфигурация модели uGCN
    model = dict(
        name='uGCN',
        kwargs=dict(
            node_conv_layer_type='SAGEConv',  # Тип слоя для узлов
            node_conv_layer_list=node_conv_layer_list,
            node_conv_heads=1,  # Количество голов внимания (для GAT)
            node_conv_layer_kwargs=dict(aggr='mean'),  # Агрегация признаков узлов
            node_global_pool_type='global_mean_pool',  # Глобальный пулинг
            edge_fc_layer_list=edge_fc_layer_list,
            out_fc_layer_list=out_fc_layer_list,
            split_out_fc=None,  # Разделение выходных слоев
        )
    )

    # Генерация вариантов моделей с разными параметрами
    for split_out_fc in [True]:
        model_cur = copy.copy(model)
        model['kwargs']['split_out_fc'] = split_out_fc
        model_list.append(model_cur)

    # Параметры для перебора: размер батча и методы нормализации
    batch_size_list = [64, 32, 16, 8]
    scalers_list = ['MinMaxScaler', 'RobustScaler']

    # Формирование всех комбинаций конфигураций
    cfg_list = list()
    for model in model_list:
        for batch_size in batch_size_list:
            for scaler_fn in scalers_list:
                cfg = dict(
                    utils=copy.copy(utils),
                    dataset=copy.copy(dataset),
                    dataloader=copy.copy(dataloader),
                    model=copy.copy(model),
                    optimizer=copy.copy(optimizer),
                    scheduler=copy.copy(scheduler),
                    criterion=copy.copy(criterion),
                    train=copy.copy(train)
                )
                cfg['dataset']['scaler_fn'] = scaler_fn
                cfg['dataloader']['batch_size'] = batch_size
        
                cfg_list.append(cfg)

    # Запуск экспериментов
    for idx, cfg in enumerate(cfg_list):

        if idx != 0:
            cfg['dataset']['load'] = True  # Загружать датасет после первого эксперимента

        # Настройки для отладки
        if debug_run:
            run_clear_ml = False
            cfg['utils']['out_dir'] += '_test'
            cfg['dataset']['num_samples'] = 60  # Ограничение данных
            cfg['train']['num_epochs'] = 10  # Сокращение эпох
            num_samples_to_draw = 0  # Отключение визуализации

        # Формирование уникальных имен экспериментов
        exp_params = [
            f"{cfg['dataset']['scaler_fn']}",
            f"{cfg['model']['name']}",
        ]
        # Добавление параметров агрегации и голов
        if 'aggr' in cfg['model']['kwargs']['node_conv_layer_kwargs']:
            exp_params.append(f"{cfg['model']['kwargs']['node_conv_layer_kwargs']['aggr']}")
        if 'heads' in cfg['model']['kwargs']['node_conv_layer_kwargs']:
            exp_params.append(f"heads{cfg['model']['kwargs']['node_conv_layer_kwargs']['heads']}")
        if cfg['model']['kwargs']['split_out_fc']:
            exp_params.append('split_out')
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
        exp(cfg,
            project_name='HeatNet',
            run_clear_ml=run_clear_ml,
            log_dir=exp_dir_path)

        # Тестирование модели
        out_dir_path = osp.join(exp_dir_path, 'results')
        test_exp(exp_dir_path,
                 out_dir_path,
                 num_samples_to_draw=num_samples_to_draw)
