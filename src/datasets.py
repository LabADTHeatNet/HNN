import os
import os.path as osp
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

from torch_geometric.loader import DataLoader
from torch.utils.data import random_split

import tqdm


def process_pair(nodes_path, edges_path, node_attr, edge_attr, edge_label):
    """
    Создает граф PyG из пары файлов nodes и tubes.
    """
    # Загружаем файлы
    nodes_df = pd.read_csv(nodes_path, sep='\t')
    edges_df = pd.read_csv(edges_path, sep='\t')

    # Fix: missing node id 129
    edges_df = edges_df[edges_df.id_in != 129]
    edges_df = edges_df[edges_df.id_out != 129]
    nodes_df = nodes_df[nodes_df.id != 129]
    nodes_df.loc[nodes_df['id'] >= 129, 'id'] -= 1
    edges_df.loc[edges_df['id_in'] >= 129, 'id_in'] -= 1
    edges_df.loc[edges_df['id_out'] >= 129, 'id_out'] -= 1

    # Извлекаем node attributes (x)
    x = torch.tensor(nodes_df[node_attr].values, dtype=torch.float)

    # Строим edge_index (индексы рёбер)
    edge_index = torch.tensor(
        np.array([edges_df['id_in'].values, edges_df['id_out'].values]),
        dtype=torch.long
    )

    # Извлекаем edge attributes (edge_attr)
    edge_attrs = torch.tensor(edges_df[edge_attr].values, dtype=torch.float)

    # Извлекаем edge labels (edge_label)
    edge_labels = torch.tensor(edges_df[edge_label].values, dtype=torch.float)

    edge_moded_list = torch.tensor(edges_df['moded'].values, dtype=torch.float)

    # Создаем объект Data
    data = Data(x=x,
                edge_index=edge_index,
                edge_attr=edge_attrs,
                edge_label=edge_labels,
                edge_moded=edge_moded_list,
                nodes_fp=nodes_path,
                edges_fp=edges_path,)
    return data


def create_dataset(root_dir, node_attr, edge_attr, edge_label, num_samples=None):
    """
    Создает список графов PyG из вложенных папок.
    """   
    files_list = list()
    for subdir, _, files in os.walk(root_dir):
        # Находим все файлы с nodes и tubes
        nodes_files = [f for f in files if 'nodes' in f]
        edges_files = [f for f in files if 'tubes' in f]

        # Сопоставляем пары nodes и tubes
        for nodes_file in nodes_files:
            if '-checkpoint' not in nodes_file:
                # Общий префикс до "nodes"
                prefix = nodes_file.split('nodes')[0]
                edges_file = next(
                    (f for f in edges_files if f.startswith(prefix)), None)
                if edges_file:
                    nodes_path = os.path.join(subdir, nodes_file)
                    edges_path = os.path.join(subdir, edges_file)
                    files_list.append([nodes_path, edges_path])

    dataset = []
    for nodes_path, edges_path in tqdm.tqdm(files_list[:num_samples]):
        try:
            data = process_pair(nodes_path, edges_path, node_attr, edge_attr, edge_label)
            dataset.append(data)
        except Exception as e:
            print(f"Ошибка обработки пары {nodes_path}, {edges_path}: {e}")
    return dataset


def split_dataset(dataset, train_ratio, val_ratio, seed=42):
    """
    Делит датасет на train, val и test с заданными долями.
    """
    total_len = len(dataset)
    train_len = int(train_ratio * total_len)
    val_len = int(val_ratio * total_len)
    test_len = total_len - train_len - val_len  # Оставшиеся данные идут в test

    # Фиксируем случайность для воспроизводимости
    torch.manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_len, val_len, test_len])
    return train_dataset, val_dataset, test_dataset


def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=16):
    """
    Создаёт DataLoader для train, val и test.
    """
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def prepare_data(dataset_config, dataloader_config, seed):
    # Создаем датасет
    if dataset_config['load']:
        # Загружаем датасет из файла
        dataset = torch.load(dataset_config['fp'])
        print(f'Датасет загружен из файла: {dataset_config['fp']}')
    else:
        print("Создание датасета...")
        dataset_path = osp.join(dataset_config['datasets_dir'], dataset_config['name'])
        dataset = create_dataset(str(dataset_path),
                                 dataset_config['node_attr'],
                                 dataset_config['edge_attr'],
                                 dataset_config['edge_label'],
                                 num_samples=dataset_config['num_samples'])
        torch.save(dataset, dataset_config['fp'])
        print(f'Датасет сохранен в файл: {dataset_config['fp']}')

    print(f"Готово! Количество графов: {len(dataset)}")

    # Разделяем датасет
    train_dataset, val_dataset, test_dataset = split_dataset(
        dataset, dataloader_config['train_ratio'], dataloader_config['val_ratio'], seed=seed)
    print(f'Train графов: {len(train_dataset)}, Val графов: {
          len(val_dataset)}, Test графов: {len(test_dataset)}')

    # Создаем DataLoader'ы
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset, dataloader_config['batch_size'])

    return dataset, train_loader, val_loader, test_loader
