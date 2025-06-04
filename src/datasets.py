from pathlib import Path
import os.path as osp
import importlib
import pandas as pd
import numpy as np
import random
from collections import deque

import torch
from torch.utils.data import random_split

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

import tqdm

from src.utils import get_str_timestamp


def find_file_pairs(root_dir):
    """Поиск пар файлов nodes и edges в директории и поддиректориях с использованием pathlib."""
    root = Path(root_dir)
    file_pairs = []

    # Рекурсивно ищем файлы, содержащие 'nodes' и оканчивающиеся на .csv
    for nodes_path in root.rglob("*nodes*.csv"):
        # Пропускаем временные файлы (например, Excel временные файлы начинающиеся с '~$'
        # или файлы, содержащие '-checkpoint' в имени)
        if nodes_path.name.startswith("~$") or "-checkpoint" in nodes_path.name:
            continue

        # Определяем соответствующий файл, заменяя 'nodes' на 'tubes'
        edges_name = nodes_path.name.replace("nodes", "tubes")
        edges_path = nodes_path.parent / edges_name

        if edges_path.exists():
            file_pairs.append([str(nodes_path), str(edges_path)])

    return file_pairs


def load_dataframes(files_list):
    """Загрузка данных узлов и ребер из CSV-файлов."""
    nodes_dataframes = []
    edges_dataframes = []

    for nodes_path, edges_path in tqdm.tqdm(files_list):
        nodes_df = pd.read_csv(nodes_path, sep='\t')
        edges_df = pd.read_csv(edges_path, sep='\t')
        # edges_df['mod'] = edges_df['moded']
        # edges_df.loc[edges_df['mod'] == 1, "mod"] = 0
        # edges_df.loc[edges_df['mod'] == 2, "mod"] = 1
        
        nodes_df['types_def'] = nodes_df['types'] == 0
        nodes_df['types_usr'] = nodes_df['types'] == 1
        nodes_df['types_src'] = nodes_df['types'] == 2

        # создаём отображение id → Q
        q_map = nodes_df.set_index('id')['Q']
        # добавляем dQ прямо по map
        edges_df['Q_out'] = edges_df['id_out'].map(q_map)
        edges_df['Q_in'] = edges_df['id_in'].map(q_map)
        edges_df['dQ'] = edges_df['id_out'].map(q_map) - edges_df['id_in'].map(q_map)
        
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
        from src.utils import IdealValueScaler
        # Динамический импорт класса скейлера из sklearn
        scaler_fn = getattr(importlib.import_module(f"sklearn.preprocessing"), scaler_fn)

        # Инициализация скейлеров
        node_attr_scaler = scaler_fn()
        edge_attr_scaler = scaler_fn()
        edge_label_scaler = scaler_fn()
        # edge_label_scaler = IdealValueScaler(edges_dataframes[0][edge_label])


        # Объединение данных из всех файлов
        all_node_attr_data = pd.concat([df[node_attr] for df in nodes_dataframes], ignore_index=True)
        all_edge_attr_data = pd.concat([df[edge_attr] for df in edges_dataframes], ignore_index=True)
        all_edge_label_data = pd.concat([df[edge_label] for df in edges_dataframes], ignore_index=True)

        # Обучение скейлеров
        node_attr_scaler.fit(all_node_attr_data)
        edge_attr_scaler.fit(all_edge_attr_data)
        edge_label_scaler.fit(all_edge_label_data)
        
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
        if scalers['node_attr_scaler'] is not None:
            nodes_df[node_attr] = scalers['node_attr_scaler'].transform(nodes_df[node_attr])
        if scalers['edge_attr_scaler'] is not None:
            edges_df[edge_attr] = scalers['edge_attr_scaler'].transform(edges_df[edge_attr])
        if scalers['edge_label_scaler'] is not None:
            edges_df[edge_label] = scalers['edge_label_scaler'].transform(edges_df[edge_label])
        if edge_label_pred is not None:  # Нормализация предсказаний, если заданы
            if scalers['edge_label_scaler'] is not None:
                edges_df[edge_label_pred] = scalers['edge_label_scaler'].transform(edges_df[edge_label_pred])

    return nodes_dataframes, edges_dataframes


def denormalize_dataframes(nodes_dataframes, edges_dataframes,
                           node_attr, edge_attr, edge_label,
                           scalers, edge_label_pred=None):
    """Обратное преобразование данных (денормализация)."""
    for nodes_df, edges_df in zip(nodes_dataframes, edges_dataframes):
        if scalers['node_attr_scaler'] is not None:
            nodes_df[node_attr] = scalers['node_attr_scaler'].inverse_transform(nodes_df[node_attr])
        if scalers['edge_attr_scaler'] is not None:
            edges_df[edge_attr] = scalers['edge_attr_scaler'].inverse_transform(edges_df[edge_attr])
        if scalers['edge_label_scaler'] is not None:
            edges_df[edge_label] = scalers['edge_label_scaler'].inverse_transform(edges_df[edge_label])
        if edge_label_pred is not None:
            if scalers['edge_label_scaler'] is not None:
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
            error_msg = f"Ошибка обработки файлов:\n- Узлы: {nodes_fp}\n- Ребра: {edges_fp}\nПричина: {str(e)}"
            print(error_msg)
            with open("data_processing_errors.log", "a") as log_file:
                log_file.write(f"{get_str_timestamp()} | {error_msg}\n")

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


def prepare_data(dataset_config, dataloader_config, seed=42, prepare_dataloaders=True):
    """Основная функция подготовки данных: загрузка или создание датасета.
    Параметр prepare_dataloaders управляет разделением и созданием DataLoader'ов."""
    # Загрузка или создание датасета
    if dataset_config['load']:
        try:
            dataset_dict = torch.load(dataset_config['fp'])
        except FileNotFoundError:
            raise FileNotFoundError(f"Файл датасета не найден: {dataset_config['fp']}")
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки датасета из {dataset_config['fp']}: {e}")
        dataset = dataset_dict['dataset']
        scalers = dataset_dict['scalers']
        print(f"Датасет загружен из файла: {dataset_config['fp']}")
    else:
        print("Создание датасета...")
        dataset_path = osp.join(dataset_config['datasets_dir'], dataset_config['name'])
        dataset, scalers = create_dataset(
            str(dataset_path),
            dataset_config['node_attr'],
            dataset_config['edge_attr'],
            dataset_config['edge_label'],
            num_samples=dataset_config.get('num_samples'),
            seed=seed,
            scaler_fn=dataset_config.get('scaler_fn')
        )
        torch.save({'dataset': dataset, 'scalers': scalers}, dataset_config['fp'])
        print(f"Датасет сохранен в файл: {dataset_config['fp']}")

    print(f"Готово! Количество графов: {len(dataset)}")

    # Если не требуется создание DataLoader'ов, возвращаем только dataset и scalers
    if not prepare_dataloaders:
        return dataset, scalers

    # Разделение на выборки
    train_dataset, val_dataset, test_dataset = split_dataset(
        dataset,
        dataloader_config['train_ratio'],
        dataloader_config['val_ratio'],
        seed=seed
    )
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Создание DataLoader'ов
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=dataloader_config.get('batch_size', 1),
        **dataloader_config.get('kwargs', {})
    )

    return dataset, scalers, train_loader, val_loader, test_loader


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


def refine(dataset, k=1):
    '''
    Че-то делает, возвращает новый датасет с приписанным таргетом к рёбрам,
    ВАЖНО: столбец атрибута в вершине должен быть последним, костыль, работает для dataset_f.py
    '''
    class GraphDatasetFromList(InMemoryDataset):
        def __init__(self, data_list):
            super().__init__()
            self.data, self.slices = self.collate(data_list)

    # Функция для поиска всех вершин на расстоянии <= k от заданной вершины

    def get_nodes_within_k_hop(edge_index, node, k):
        visited = set()
        queue = deque([(node, 0)])  # (vertex, current_distance)

        while queue:
            current_node, current_distance = queue.popleft()
            if current_node in visited:
                continue
            visited.add(current_node)

            if current_distance < k:
                # Добавляем соседей текущей вершины (ГРАФ НАПРАВЛЕННЫЙ!!)
                neighbors = torch.cat((edge_index[1, edge_index[0] == current_node], edge_index[0, edge_index[1] == current_node]))
                for neighbor in neighbors:
                    queue.append((neighbor.item(), current_distance + 1))

        return visited

    def get_new_data(data: Data, k=1):
        '''
        Преобразует полученный Data объект, приписывая столбец в тензор edge_target, являющийся суммой всех вершин-соседей рёбер на расстоянии не более k
        '''
        # Инициализируем edge_target
        edge_target = torch.zeros(data.edge_index.size(1), dtype=torch.float)

        # Для каждого ребра
        for i in range(data.edge_index.size(1)):
            src, dst = data.edge_index[:, i]  # Исходная и целевая вершины ребра

            # Находим все вершины на расстоянии <= k от исходной и целевой вершин
            src_nodes = get_nodes_within_k_hop(data.edge_index, src.item(), k)
            dst_nodes = get_nodes_within_k_hop(data.edge_index, dst.item(), k)

            # Объединяем вершины и убираем дубликаты
            all_nodes = src_nodes.union(dst_nodes)

            # Суммируем атрибуты этих вершин (если не сделать [:, -1], будут суммироваться ещё и координаты вершин, см. костыль)
            edge_target[i] = data.x[list(all_nodes)][:, -1].sum()

        # Добавляем edge_target в data
        data.edge_target = torch.cat([data.edge_target, edge_target.unsqueeze(1)], 1)
        return data
    new_data_list = [get_new_data(data, k) for data in tqdm(dataset)]
    # Интересно, что поле edge_target_cols делят все объекты типа Data (если они из одного Dataset?)
    new_data_list[0].edge_target_cols.append(f'{k}_sum')
    return GraphDatasetFromList(new_data_list)

import torch

def detect_defects(all_data):
    """
    all_data: list of PyG Data-объектов из test_loader, у каждого есть
      - edge_label      (Tensor[E]) — истинные диаметры (де-факто текущие, но мы берём только те, что не модифицированы)
      - edge_label_pred (Tensor[E]) — предсказанные диаметры
      - edge_moded      (Tensor[E], int) — true класс (0,1,2)
    Возвращает:
      - ideal_dia   Tensor[E]: «идеальный» диаметр каждого ребра (среднее по всем графам, где edge_moded==0)
      - acc1, acc2  float: accuracy по классам 1 и 2
      - dev_list    list of Tensor[E]: список всех pred_dev для каждого графа (необязательно)
      - pred_moded_list list of Tensor[E]: предсказанные классы для каждого графа
    """
    # № графа не важен: E одно и то же для всех
    E = all_data[0].edge_label.shape[0]
    sum_dia = torch.zeros(E)
    count_dia = torch.zeros(E)

    # 1) Считаем идеальный диаметр: усредняем все НЕ изменённые ребра (edge_moded==0)
    for d in all_data:
        labels = d.edge_label.cpu()
        mask0 = (d.edge_moded.cpu() == 0)
        sum_dia[mask0]   += labels[mask0]
        count_dia[mask0] += 1
    # Чтобы не делить на 0, можно оставить идеал там, где count_dia==0 равным 0
    ideal_dia = sum_dia / count_dia.clamp(min=1)

    # 2) Для каждого графа считаем отклонение и предсказанный класс
    total1 = total2 = 0
    correct1 = correct2 = 0
    dev_list = []
    pred_moded_list = []

    for d in all_data:
        pred   = d.edge_label_pred.cpu()
        actual = d.edge_moded.cpu()

        # процент отклонения от идеала
        dev = (pred - ideal_dia).abs() / ideal_dia * 100.0
        dev_list.append(dev)

        # pred_moded: 1 если dev<=5%, 2 если dev>10%
        # (между 5 и 10% редких случаев нет, но их можно отнести к «1»)
        pred_moded = torch.where(dev > 10.0, 2, 1)
        pred_moded_list.append(pred_moded)

        # 3) аккумулируем accuracy для классов 1 и 2
        mask1 = (actual == 1)
        mask2 = (actual == 2)

        total1  += mask1.sum().item()
        total2  += mask2.sum().item()
        correct1 += (pred_moded[mask1] == 1).sum().item()
        correct2 += (pred_moded[mask2] == 2).sum().item()

    acc1 = correct1 / total1 if total1>0 else 0.0
    acc2 = correct2 / total2 if total2>0 else 0.0

    return ideal_dia, acc1, acc2, dev_list, pred_moded_list
