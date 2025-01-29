import os
import os.path as osp
import importlib
import pandas as pd
import numpy as np
import random

import torch
from torch.utils.data import random_split

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import tqdm

def find_file_pairs(root_dir):
    """Поиск пар файлов nodes и edges в директории и поддиректориях."""
    files_list = []
    for subdir, _, files in os.walk(root_dir):
        nodes_files = [f for f in files if 'nodes' in f]

        for nodes_file in nodes_files:
            if '-checkpoint' not in nodes_file:  # Игнорирование временных файлов
                edges_file = nodes_file.replace('nodes', 'tubes')
                nodes_path = os.path.join(subdir, nodes_file)
                edges_path = os.path.join(subdir, edges_file)
                # Проверка существования парных файлов
                if os.path.exists(nodes_path) and os.path.exists(edges_path):
                    files_list.append([nodes_path, edges_path])
    return files_list

def load_dataframes(files_list):
    """Загрузка данных узлов и ребер из CSV-файлов."""
    nodes_dataframes = []
    edges_dataframes = []

    for nodes_path, edges_path in tqdm.tqdm(files_list):
        nodes_df = pd.read_csv(nodes_path, sep='\t')
        edges_df = pd.read_csv(edges_path, sep='\t')

        # Исправление пропущенных идентификаторов узлов (например, id=129)
        edges_df = edges_df[edges_df.id_in != 129]
        edges_df = edges_df[edges_df.id_out != 129]
        nodes_df = nodes_df[nodes_df.id != 129]
        # Корректировка идентификаторов после удаления
        nodes_df.loc[nodes_df['id'] >= 129, 'id'] -= 1
        edges_df.loc[edges_df['id_in'] >= 129, 'id_in'] -= 1
        edges_df.loc[edges_df['id_out'] >= 129, 'id_out'] -= 1

        nodes_dataframes.append(nodes_df)
        edges_dataframes.append(edges_df)

    return nodes_dataframes, edges_dataframes

def fit_global_scalers(nodes_dataframes, edges_dataframes,
                       node_attr, edge_attr, edge_label, scaler_fn=None):
    """Обучение скейлеров на всех данных для согласованной нормализации."""
    if scaler_fn is not None:
        # Динамический импорт класса скейлера из sklearn
        scaler_fn = getattr(importlib.import_module(f"sklearn.preprocessing"), scaler_fn)
        
        # Инициализация скейлеров
        node_attr_scaler = scaler_fn()
        edge_attr_scaler = scaler_fn()
        edge_label_scaler = scaler_fn()

        # Объединение данных из всех файлов
        all_node_attr_data = pd.concat([df[node_attr] for df in nodes_dataframes], ignore_index=True)
        all_edge_attr_data = pd.concat([df[edge_attr] for df in edges_dataframes], ignore_index=True)
        all_edge_label_data = pd.concat([df[edge_label] for df in edges_dataframes], ignore_index=True)

        # Обучение скейлеров
        node_attr_scaler.fit(all_node_attr_data)
        edge_attr_scaler.fit(all_edge_attr_data)
        edge_label_scaler.fit(all_edge_label_data)
    else:
        node_attr_scaler = None
        edge_attr_scaler = None
        edge_label_scaler = None

    return {
        'node_attr_scaler': node_attr_scaler,
        'edge_attr_scaler': edge_attr_scaler,
        'edge_label_scaler': edge_label_scaler
    }

def normalize_dataframes(nodes_dataframes, edges_dataframes,
                         node_attr, edge_attr, edge_label,
                         scalers, edge_label_pred=None):
    """Применение обученных скейлеров к данным."""
    for nodes_df, edges_df in tqdm.tqdm(zip(nodes_dataframes, edges_dataframes), total=len(nodes_dataframes)):
        # Нормализация признаков узлов, ребер и меток
        nodes_df[node_attr] = scalers['node_attr_scaler'].transform(nodes_df[node_attr])
        edges_df[edge_attr] = scalers['edge_attr_scaler'].transform(edges_df[edge_attr])
        edges_df[edge_label] = scalers['edge_label_scaler'].transform(edges_df[edge_label])
        if edge_label_pred is not None:  # Нормализация предсказаний, если заданы
            edges_df[edge_label_pred] = scalers['edge_label_scaler'].transform(edges_df[edge_label_pred])

    return nodes_dataframes, edges_dataframes

def denormalize_dataframes(nodes_dataframes, edges_dataframes,
                           node_attr, edge_attr, edge_label,
                           scalers, edge_label_pred=None):
    """Обратное преобразование данных (денормализация)."""
    for nodes_df, edges_df in zip(nodes_dataframes, edges_dataframes):
        nodes_df[node_attr] = scalers['node_attr_scaler'].inverse_transform(nodes_df[node_attr])
        edges_df[edge_attr] = scalers['edge_attr_scaler'].inverse_transform(edges_df[edge_attr])
        edges_df[edge_label] = scalers['edge_label_scaler'].inverse_transform(edges_df[edge_label])
        if edge_label_pred is not None:
            edges_df[edge_label_pred] = scalers['edge_label_scaler'].inverse_transform(edges_df[edge_label_pred])

    return nodes_dataframes, edges_dataframes

def process_dataframes(nodes_df, edges_df,
                       node_attr, edge_attr, edge_label,
                       nodes_fp, edges_fp):
    """Преобразование DataFrame в объект PyG Data."""
    # Извлечение признаков узлов
    x = torch.tensor(nodes_df[node_attr].values, dtype=torch.float)

    # Построение edge_index (связи между узлами)
    t_edge_index = torch.tensor(np.array([edges_df['id_in'].values, edges_df['id_out'].values]), dtype=torch.long)

    # Извлечение признаков и меток ребер
    t_edge_attr = torch.tensor(edges_df[edge_attr].values, dtype=torch.float)
    t_edge_label = torch.tensor(edges_df[edge_label].values, dtype=torch.float)
    t_edge_moded = torch.tensor(edges_df[['moded']].values, dtype=torch.float)

    # Создание объекта Data для PyTorch Geometric
    data = Data(
        x=x,
        edge_index=t_edge_index,
        edge_attr=t_edge_attr,
        edge_label=t_edge_label,
        edge_moded=t_edge_moded,
        nodes_fp=nodes_fp,  # Пути к исходным файлам для трассировки
        edges_fp=edges_fp
    )
    return data

def create_dataset(root_dir, node_attr, edge_attr, edge_label, num_samples=None, seed=42, scaler_fn=None):
    """Создание датасета из файлов с нормализацией и преобразованием в графы."""
    print("Поиск пар файлов...")
    files_list = find_file_pairs(root_dir)
    print(f"Найдено {len(files_list)} пар файлов.")
    
    # Фиксация случайности для воспроизводимости
    random.Random(seed).shuffle(files_list)
    files_list = files_list[:num_samples]  # Ограничение количества выборок

    print("Считывание таблиц...")
    nodes_dataframes, edges_dataframes = load_dataframes(files_list)

    print("Обучение глобальных скейлеров...")
    scalers = fit_global_scalers(nodes_dataframes, edges_dataframes,
                                 node_attr, edge_attr, edge_label, scaler_fn=scaler_fn)

    print("Нормализация таблиц...")
    nodes_dataframes, edges_dataframes = normalize_dataframes(
        nodes_dataframes, edges_dataframes, node_attr, edge_attr, edge_label, scalers)

    print("Конвертация в PyG Data...")
    dataset = []
    for nodes_df, edges_df, (nodes_fp, edges_fp) in tqdm.tqdm(zip(nodes_dataframes, edges_dataframes, files_list), total=len(nodes_dataframes)):
        try:
            data = process_dataframes(nodes_df, edges_df, node_attr, edge_attr, edge_label, nodes_fp, edges_fp)
            dataset.append(data)
        except Exception as e:
            print(f"Ошибка обработки: {e}")

    return dataset, scalers

def split_dataset(dataset, train_ratio, val_ratio, seed=42):
    """Разделение датасета на обучающую, валидационную и тестовую выборки."""
    total_len = len(dataset)
    train_len = int(train_ratio * total_len)
    val_len = int(val_ratio * total_len)
    test_len = total_len - train_len - val_len  # Оставшиеся данные для теста

    # Фиксация случайности
    torch.manual_seed(seed)
    return random_split(dataset, [train_len, val_len, test_len])

def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=16):
    """Создание DataLoader для обучения, валидации и тестирования."""
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def prepare_data(dataset_config, dataloader_config, seed=42):
    """Основная функция подготовки данных: загрузка или создание датасета."""
    if dataset_config['load']:  # Загрузка предобработанного датасета
        dataset_dict = torch.load(dataset_config['fp'])
        dataset = dataset_dict['dataset']
        scalers = dataset_dict['scalers']
        print(f"Датасет загружен из файла: {dataset_config['fp']}")
    else:  # Создание нового датасета
        print("Создание датасета...")
        dataset_path = osp.join(dataset_config['datasets_dir'], dataset_config['name'])
        dataset, scalers = create_dataset(
            str(dataset_path),
            dataset_config['node_attr'],
            dataset_config['edge_attr'],
            dataset_config['edge_label'],
            num_samples=dataset_config['num_samples'],
            seed=seed,
            scaler_fn=dataset_config['scaler_fn']
        )
        # Сохранение датасета для повторного использования
        torch.save({'dataset': dataset, 'scalers': scalers}, dataset_config['fp'])
        print(f"Датасет сохранен в файл: {dataset_config['fp']}")

    print(f"Готово! Количество графов: {len(dataset)}")

    # Разделение на выборки
    train_dataset, val_dataset, test_dataset = split_dataset(
        dataset, dataloader_config['train_ratio'], dataloader_config['val_ratio'], seed=seed)
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Создание DataLoader
    return create_dataloaders(train_dataset, val_dataset, test_dataset, dataloader_config['batch_size'])

def data_to_tables(in_data,
                   node_attr, edge_attr, edge_label,
                   scalers=None, edge_label_pred=None):
    """Обратное преобразование Data в таблицы с денормализацией."""
    data = in_data.cpu()
    nodes_df = pd.DataFrame(data.x.numpy(), columns=node_attr)
    nodes_df['id'] = range(len(nodes_df))

    # Восстановление связей
    edge_index = data.edge_index.numpy()
    edges_df = pd.DataFrame({'id_in': edge_index[0], 'id_out': edge_index[1]})
    # Восстановление признаков и меток
    for i, col_name in enumerate(edge_attr):
        edges_df[col_name] = data.edge_attr[:, i].numpy()
    edges_df[edge_label] = data.edge_label.numpy()
    
    if edge_label_pred is not None:
        edges_df[edge_label_pred] = data.edge_label_pred.numpy()

    # Денормализация
    if scalers:
        nodes_df, edges_df = denormalize_dataframes(
            [nodes_df], [edges_df], node_attr, edge_attr, edge_label, scalers, edge_label_pred=edge_label_pred)
        nodes_df = nodes_df[0]
        edges_df = edges_df[0]

    return nodes_df, edges_df
