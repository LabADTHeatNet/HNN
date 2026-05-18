from pathlib import Path
import os.path as osp
import importlib
import pandas as pd
import numpy as np
import random
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import torch
from torch.utils.data import random_split, Subset
from torch_geometric.data import Data, Dataset, InMemoryDataset, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected
from sklearn.preprocessing import LabelEncoder
import tqdm
from sklearn.model_selection import train_test_split
from src.utils import get_str_timestamp

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PairedGraphDataset(Dataset):
    def __init__(self, fwd, bwd):
        self.fwd = [g.to(device) for g in fwd]
        self.bwd = [g.to(device) for g in bwd]

    def __len__(self):
        return len(self.fwd)

    def __getitem__(self, idx):
        return self.fwd[idx], self.bwd[idx]
    
def paired_collate(batch):
    G1 = [g[0] for g in batch]
    G2 = [g[1] for g in batch]
    return Batch.from_data_list(G1), Batch.from_data_list(G2)

def find_file_pairs(root_dir, ideal=False):
    """Поиск пар файлов nodes и edges в директории и поддиректориях с использованием pathlib."""
    root = Path(root_dir)
    file_pairs = []

    # Рекурсивно ищем файлы, содержащие 'nodes' и оканчивающиеся на .csv
    for nodes_path in root.rglob("*nodes*.csv"):
        # Пропускаем временные файлы (например, Excel временные файлы начинающиеся с '~$'
        # или файлы, содержащие '-checkpoint' в имени)
        if nodes_path.name.startswith("~$") or "-checkpoint" in nodes_path.name:
            continue

        # Фильтрация файлов по наличию '_ideal' в имени в зависимости от параметра ideal
        if ("_ideal" in nodes_path.name) != ideal:
            continue

        # Определяем соответствующий файл, заменяя 'nodes' на 'tubes'
        edges_name = nodes_path.name.replace("nodes", "tubes")
        edges_path = nodes_path.parent / edges_name

        if edges_path.exists():
            file_pairs.append([str(nodes_path), str(edges_path)])

    return sorted(file_pairs)

def get_global_parameters(file_pairs):
    global_dataframes = []
    for nodes_path, edges_path in file_pairs:
        global_path = nodes_path.replace("nodes", "global")
        if Path(global_path).exists():
            global_df = pd.read_csv(global_path, sep= '\t')
            global_dataframes.append(global_df)
    return global_dataframes        

        

def get_node_degrees (edges_df):
    node_deg_out = {}
    node_deg_in = {}
    for edge in edges_df.iterrows():
        node_deg_in[edge[1]['id_out']] = node_deg_in.get(edge[1]['id_out'], 0) + 1
        node_deg_out[edge[1]['id_in']] = node_deg_out.get(edge[1]['id_in'], 0) + 1
    return node_deg_out, node_deg_in

def add_sections(nodes_df, edges_df):
    '''
    Функция, добавляющая в edge_df колонку id_section,
    отвечающую за принадлежность рёбер компоненте "участка",
    разделяющей рёбра между разными рёбрами-потребителями и вершинами-развилками. 
    '''
    # Определяем вершины-источники и степени вершин в целом
    node_deg_out = {}
    node_deg_in = {}
    for edge in edges_df.iterrows():
        node_deg_in[edge[1]['id_out']] = node_deg_in.get(edge[1]['id_out'], 0) + 1
        node_deg_out[edge[1]['id_in']] = node_deg_out.get(edge[1]['id_in'], 0) + 1
    start_vertices_id = (set(nodes_df['id'])).difference(set(node_deg_in.keys()))
    
    if len(start_vertices_id) > 2:
        print("Found more than 2 source nodes!\nAre you sure this is correct behaviour?")
        
    # Определяем рёбра, исходящие из вершин-источников
    start_edges = []
    for vertice in start_vertices_id:
        start_edges.append(edges_df.loc[edges_df['id_in'] == vertice])
    if len(start_edges) == 0:
        return edges_df
    
    edge_queue = deque()
    visited_id = set()
    
    # На данное значение будет инкрементироваться id_section,
    # подразумевается, что для двух компонент связности нечётные id_section будут относиться к 1-й, а чётные -- ко 2-й компоненте
    num_sources = len(start_edges) 
    
    
    # Добавляем стартовые рёбра в очередь
    for i, edge in enumerate(start_edges):
        # Внутренний цикл -- костыль (добавлять строки df в list, вероятно, не лучшая идея)
        for edge_t in edge.itertuples(): 
            id_section = i + num_sources if edge_t.Vid_usr else i
            edge_queue.append((edge_t.Index, edge_t.id_in, edge_t.id_out, id_section))
            visited_id.add(edge_t.Index)
            
    next_section_ids = [i for i in range(num_sources)]   
     
    # Запускаем BFS
    while (edge_queue):
        id, id_in, id_out, id_section = edge_queue.popleft()
        edges_df.loc[id,'id_section'] = id_section
        next_edges = edges_df.loc[edges_df['id_in'] == id_out]
        if len(next_edges) > 0:
            for next_edge in next_edges.itertuples():
                if next_edge.Index not in visited_id:
                    
                    # Если ребро -- потребитель, то увеличиваем id_section всех последующих рёбер
                    # Или если ребро выходит из вершины-развилки, то пусть оно тоже имеет новый id_section
                    if (next_edge.Vid_usr) or node_deg_in[id_out] + node_deg_out[id_out] > 2:

                        next_section_ids[id_section % num_sources] += num_sources 
                        edge_queue.append((next_edge.Index, next_edge.id_in, next_edge.id_out, next_section_ids[id_section % num_sources]))
                    else:
                        edge_queue.append((next_edge.Index, next_edge.id_in, next_edge.id_out, id_section))
                    
                    # Добавляем смежное ребро в очередь
                    
                    visited_id.add(next_edge.Index)
    
    if visited_id != set(edges_df.index):
        for id in set(edges_df.index):
            if id not in visited_id:
                print("Edge ", id, " was not visited!")
        print("Not all edges were visited!")
        # TO DO: подумать надо ли бросать исключение
        raise Exception('RuntimeError')
    edges_df['id_section'] = LabelEncoder().fit_transform(edges_df['id_section'])
    return edges_df
            
    
            
    

def load_dataframes(files_list, zero_data = True):
    """Загрузка данных узлов и ребер из CSV-файлов."""
    nodes_dataframes = []
    edges_dataframes = []

    id_section = None
    junction_nodes = None

    for nodes_path, edges_path in tqdm.tqdm(files_list):
        fwd_subgraph = False
        bwd_subgraph = False
        if 'fwd' in edges_path or 'bwd' in edges_path:
            if 'fwd' in edges_path:
                fwd_subgraph = True
            else:
                bwd_subgraph = True
        nodes_df = pd.read_csv(nodes_path, sep='\t')
        edges_df = pd.read_csv(edges_path, sep='\t')
        # edges_df['mod'] = edges_df['moded']
        # edges_df.loc[edges_df['mod'] == 1, "mod"] = 0
        # edges_df.loc[edges_df['mod'] == 2, "mod"] = 1

        nodes_df['types_def'] = nodes_df['types'] == 0
        nodes_df['types_usr'] = nodes_df['types'] == 1
        nodes_df['types_src'] = nodes_df['types'] == 2

        edges_df['Vid_fwd'] = edges_df['Vid'] == 0
        edges_df['Vid_bwd'] = edges_df['Vid'] == 1
        edges_df['Vid_usr'] = edges_df['Vid'] == 2
        
        # создаём отображение id → Q
        q_map = nodes_df.set_index('id')['Q']
        # добавляем dQ прямо по map
        edges_df['Q_out'] = edges_df['id_out'].map(q_map)
        edges_df['Q_in'] = edges_df['id_in'].map(q_map)
        edges_df['dQ'] = edges_df['id_out'].map(q_map) - edges_df['id_in'].map(q_map)

        # ids_to_correct = [129]
        ids_to_correct = []
        for id in ids_to_correct:
            # Исправление пропущенных идентификаторов узлов
            edges_df = edges_df[edges_df.id_in != id]
            edges_df = edges_df[edges_df.id_out != id]
            nodes_df = nodes_df[nodes_df.id != id]
            # Корректировка идентификаторов после удаления
            nodes_df.loc[nodes_df['id'] >= id, 'id'] -= 1
            edges_df.loc[edges_df['id_in'] >= id, 'id_in'] -= 1
            edges_df.loc[edges_df['id_out'] >= id, 'id_out'] -= 1
        
        # Добавление параметра секции труб
        edges_df['id_section'] = -1
        if id_section is None and not fwd_subgraph and not bwd_subgraph:
            edges_df = add_sections(nodes_df, edges_df)
            id_section = edges_df['id_section']
        else:
            if fwd_subgraph:
                id_section = np.array([ 0,  0, 41, 24,  0, 13, 18, 18, 18, 30, 31, 10, 11, 11, 20, 21, 21,
                        1,  1,  0,  0, 11,  9,  9, 10,  5,  3,  3,  3,  3,  3,  3,  3, 37,
                        8,  9, 14, 25, 25, 25, 32, 19, 11,  6,  2, 15, 15, 16,  4, 15, 16,
                    14, 25, 25, 35, 10, 19, 18, 32, 36, 36, 36, 36, 36, 38,  8,  9, 14,
                    35, 36, 32, 32, 32, 18, 11,  6,  2,  4, 15, 16, 16, 14, 25, 35, 10,
                    10, 19, 18, 32, 32, 32, 36, 36, 42, 39, 29, 17, 12, 33, 34, 22, 23,
                        7, 40, 28, 27, 26])
            if bwd_subgraph:
                id_section = np.array([ 0, 41, 24,  0, 13, 18, 18, 18, 30, 31, 10, 11, 11, 20, 21, 21,  1,
        1,  0,  0, 11,  9,  9, 10,  5,  3,  3,  3,  3,  3,  3,  3, 37,  8,
        9, 14, 25, 25, 25, 32, 19, 11,  6,  2, 15, 15, 16,  4, 15, 16, 14,
       25, 25, 35, 10, 19, 18, 32, 36, 36, 36, 36, 36, 38,  8,  9, 14, 35,
       36, 32, 32, 32, 18, 11,  6,  2,  4, 15, 16, 16, 14, 25, 35, 10, 10,
       19, 18, 32, 32, 32, 36, 36,  0, 42, 39, 29, 17, 12, 33, 34, 22, 23,
        7, 40, 28, 27, 26])
            edges_df['id_section'] = id_section
        
        users = edges_df.loc[edges_df['Vid_usr'], ['id_in', 'id_out']]
        nodes_usr =set(pd.concat([users['id_in'], users['id_out']]))
        hardcoded_id = 186 if not bwd_subgraph else 107
        nodes_src = set(nodes_df.loc[nodes_df['types_src'] | nodes_df['types_usr'] | (nodes_df['id'] == hardcoded_id)].index) 
        if junction_nodes is None:
            deg_out, deg_in = get_node_degrees(edges_df)
            mapped_degrees = nodes_df.index.map(lambda x : deg_out.get(x, 0)) + nodes_df.index.map(lambda x : deg_in.get(x, 0))
            junction_nodes = set(nodes_df.loc[mapped_degrees > 2].index)
            
        # Обнуляем большую часть данных исходя из того, что в реальной жизни их не будет
        if zero_data:    
            nodes_df.loc[~nodes_df.index.isin(nodes_usr | nodes_src), ['P', 'Temp']] = 0
          
        deviation = np.abs(edges_df['moded'] - 1.0)
        
        # ВЫНЕСТИ КУДА-ТО ЭТОТ ПАРАМЕТР
        defect_threshold = 0.05
        
        defect = deviation[deviation > defect_threshold]
        if len(defect) > 0:
            # Берем секцию с самым сильным дефектом
            defect_section = edges_df.loc[defect.idxmax(), 'id_section']
        else:
            # Если дефекта нет, присваиваем отдельную метку
            defect_section = max(edges_df['id_section'].unique()) + 1 
        edges_df['graph_label'] = defect_section
        nodes_dataframes.append(nodes_df)
        edges_dataframes.append(edges_df)
    return nodes_dataframes, edges_dataframes


def fit_global_scalers(nodes_dataframes, edges_dataframes, global_dataframes,
                       node_attr, edge_attr, global_attr, edge_label, scaler_fn=None):
    """Обучение скейлеров на всех данных для согласованной нормализации."""
    if scaler_fn is not None:
        # Динамический импорт класса скейлера из sklearn
        scaler_fn = getattr(importlib.import_module(f"sklearn.preprocessing"), scaler_fn)

        # Инициализация скейлеров
        node_attr_scaler = scaler_fn()
        edge_attr_scaler = scaler_fn()
        global_scaler = scaler_fn()
        # edge_label_scaler = scaler_fn()
        # edge_label_scaler = IdealValueScaler(edges_dataframes[0][edge_label])

        # Объединение данных из всех файлов
        all_node_attr_data = pd.concat([df[node_attr] for df in nodes_dataframes], ignore_index=True)
        all_edge_attr_data = pd.concat([df[edge_attr] for df in edges_dataframes], ignore_index=True)
        all_global_data = pd.concat([df[global_attr] for df in global_dataframes], ignore_index=True)
        # all_edge_label_data = pd.concat([df[edge_label] for df in edges_dataframes], ignore_index=True)

        # Обучение скейлеров
        node_attr_scaler.fit(all_node_attr_data)
        edge_attr_scaler.fit(all_edge_attr_data)
        global_scaler.fit(all_global_data)
        # edge_label_scaler.fit(all_edge_label_data)

        # # Обучение скейлеров
        # node_attr_scaler.fit(all_node_attr_data)
        # edge_attr_scaler.fit(all_edge_attr_data)
        # # edge_label_scaler.fit(all_edge_label_data)

    else:
        node_attr_scaler = None
        edge_attr_scaler = None
        edge_label_scaler = None
        global_scaler = None
    return {
        'node_attr_scaler': node_attr_scaler,
        'edge_attr_scaler': edge_attr_scaler,
        # 'edge_label_scaler': edge_label_scaler,
        'global_scaler' : global_scaler
    }


def normalize_dataframes(nodes_dataframes, edges_dataframes, global_dataframes,
                         node_attr, edge_attr, global_attr, edge_label,
                         scalers, edge_label_pred=None, num_workers=4):
    """Применение обученных скейлеров к данным с использованием многопоточности."""

    def normalize_pair(args):
        nodes_df, edges_df, global_df = args
        # Нормализация признаков узлов, ребер и меток
        if scalers['node_attr_scaler'] is not None:
            nodes_df[node_attr] = scalers['node_attr_scaler'].transform(nodes_df[node_attr])
        if scalers['edge_attr_scaler'] is not None:
            edges_df[edge_attr] = scalers['edge_attr_scaler'].transform(edges_df[edge_attr])
        if scalers['global_scaler'] is not None:
            global_df[global_attr] = scalers['global_scaler'].transform(global_df[global_attr])
        # if scalers['edge_label_scaler'] is not None:
        #     edges_df[edge_label] = scalers['edge_label_scaler'].transform(edges_df[edge_label])
        # if edge_label_pred is not None:
        #     if scalers['edge_label_scaler'] is not None:
        #         edges_df[edge_label_pred] = scalers['edge_label_scaler'].transform(edges_df[edge_label_pred])
        return nodes_df, edges_df, global_df

    triplets = list(zip(nodes_dataframes, edges_dataframes, global_dataframes))
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for res in tqdm.tqdm(executor.map(normalize_pair, triplets), total=len(triplets)):
            results.append(res)

    nodes_dataframes, edges_dataframes, global_dataframes = zip(*results)
    return list(nodes_dataframes), list(edges_dataframes), list(global_dataframes)


def denormalize_dataframes(nodes_dataframes, edges_dataframes, global_dataframes,
                           node_attr, edge_attr, global_attr, edge_label,
                           scalers, edge_label_pred=None):
    """Обратное преобразование данных (денормализация)."""
    if global_dataframes is not None:
        raise NotImplementedError()
    for nodes_df, edges_df in zip(nodes_dataframes, edges_dataframes):
        if scalers['node_attr_scaler'] is not None:
            nodes_df[node_attr] = scalers['node_attr_scaler'].inverse_transform(nodes_df[node_attr])
        if scalers['edge_attr_scaler'] is not None:
            edges_df[edge_attr] = scalers['edge_attr_scaler'].inverse_transform(edges_df[edge_attr])
        # if scalers['global_scaler'] is not None:
        #     global_df[global_attr] = scalers['global_scaler'].inverse_transform(global_df[global_attr])
        if scalers['edge_label_scaler'] is not None:
            edges_df[edge_label] = scalers['edge_label_scaler'].inverse_transform(edges_df[edge_label])
        if edge_label_pred is not None:
            if scalers['edge_label_scaler'] is not None:
                edges_df[edge_label_pred] = scalers['edge_label_scaler'].inverse_transform(edges_df[edge_label_pred])

    return nodes_dataframes, edges_dataframes


def get_t_outside(fp):
    """Извлечение температуры воздуха снаружи из имени файла."""
    # Ищем часть пути, начинающуюся с 'Tout_'
    tout_part = next((p for p in Path(fp).parts if p.startswith('Tout_')), None)
    if tout_part is None:
        raise ValueError(f"Не найдено части пути, начинающейся с 'Tout_' в {fp}")

    return int(tout_part[len('Tout_'):])  # Возвращаем температуру как целое число


def process_dataframes(nodes_df, edges_df, global_df,
                       node_attr, edge_attr, edge_label,
                       nodes_fp, edges_fp):
    """Преобразование DataFrame в объект PyG Data."""

    # Извлечение глобальных параметров
    t_outside = get_t_outside(nodes_fp)  # Температура воздуха снаружи
    

    q_out_node = global_df.loc[0, 'Q']  # Расход на первой котельной
    t_out_node = global_df.loc[0, 'Temp']  # Температура на узле с id == 4
    t_in_node = global_df.loc[1, 'Temp'] # Температура на узле с id == 186
    graph_type = global_df.loc[0, 'channel']
    global_attrs = torch.tensor([t_outside, q_out_node, t_out_node, t_in_node, graph_type], dtype=torch.float).unsqueeze(0)  # [1, global_dim]
    # global_attrs = torch.tensor([t_outside], dtype=torch.float).unsqueeze(0)  # [1, global_dim]
    
    # Извлечение признаков узлов
    x = torch.tensor(nodes_df[node_attr].values, dtype=torch.float)

    # Построение edge_index (связи между узлами)
    t_edge_index = torch.tensor( np.array([edges_df['id_out'].values, edges_df['id_in'].values]), dtype=torch.long)
    
    # t_edge_index = torch.tensor(np.hstack([np.array([edges_df['id_in'].values, edges_df['id_out'].values]), np.array([edges_df['id_out'].values, edges_df['id_in'].values])]), dtype=torch.long)
    
    

    # Извлечение признаков и меток ребер
    t_edge_attr = torch.tensor(edges_df[edge_attr].values, dtype=torch.float)
    # ИЗМЕНЕНО ДЛЯ КЛАССИФИКАЦИИ
    t_edge_label = torch.tensor(edges_df[edge_label].values[0], dtype=torch.long)
    
    t_edge_moded = torch.tensor(edges_df[['moded']].values, dtype=torch.float)

    # Ненаправленный:
    
    # t_edge_index, t_edge_attr = to_undirected(edge_index=t_edge_index, edge_attr=t_edge_attr, reduce= "add")
    
    # Создание объекта Data для PyTorch Geometric
    data = Data(
        global_attrs=global_attrs,
        x=x,
        edge_index=t_edge_index,
        edge_attr=t_edge_attr,
        edge_label=t_edge_label,
        edge_moded=t_edge_moded,
        nodes_fp=nodes_fp,  # Пути к исходным файлам для трассировки
        edges_fp=edges_fp
    )
    return data


def create_dataset(root_dir, node_attr, edge_attr, edge_label, num_samples=None, seed=42, scaler_fn=None, add_ideal=False, global_attr = ['Q', 'Temp']):
    """Создание датасета из файлов с нормализацией и преобразованием в графы."""
    
    if add_ideal:
        print("[IDEAL] Поиск пар файлов...")
        ideal_files_list = find_file_pairs(root_dir, ideal=True)
        print(f"[IDEAL] Найдено {len(ideal_files_list)} пар файлов.")

        print("[IDEAL] Считывание таблиц...")
        ideal_nodes_dataframes, ideal_edges_dataframes = load_dataframes(ideal_files_list, zero_data=True)
        ideal_global_dataframes = get_global_parameters(ideal_files_list)
        
        if len(ideal_global_dataframes) != len(ideal_nodes_dataframes):
            ideal_global_dataframes = ideal_nodes_dataframes
        
        ideal_ne_df_list = dict()
        for (n_fp, _), in_df, ie_df  in zip(ideal_files_list,  ideal_nodes_dataframes, ideal_edges_dataframes):
            ideal_ne_df_list[get_t_outside(n_fp)] = (in_df, ie_df)
            
    print("Поиск пар файлов...")
    files_list = find_file_pairs(root_dir)
    print(f"Найдено {len(files_list)} пар файлов.")

    # Фиксация случайности для воспроизводимости
    random.Random(seed).shuffle(files_list)
    files_list = files_list[:num_samples]  # Ограничение количества выборок

    print("Считывание таблиц...")
    nodes_dataframes, edges_dataframes = load_dataframes(files_list, zero_data=True)
    
    global_dataframes = get_global_parameters(files_list)
    if len(global_dataframes) != len(nodes_dataframes):
        global_dataframes = nodes_dataframes
        
    if add_ideal:
        tag = '_ideal'
        nodes_ideal_attrs = []
        for na in node_attr:
            if tag in na:
                nodes_ideal_attrs.append(na[:-len(tag)])  # Удаляем '_ideal' из имени атрибута
        edges_ideal_attrs = []
        for ea in edge_attr:
            if tag in ea:
                edges_ideal_attrs.append(ea[:-len(tag)])  # Удаляем '_ideal' из имени атрибута

        print("[IDEAL] Добавление идеальных данных в таблицы...")
        for nodes_df, edges_df, (nodes_fp, edges_fp)  in tqdm.tqdm(zip(nodes_dataframes, edges_dataframes, files_list), total=len(nodes_dataframes)):
            # Если есть идеальные данные, добавляем их в глобальные параметры
            t_outside = get_t_outside(nodes_fp)
            if t_outside in ideal_ne_df_list:
                ideal_nodes_df, ideal_edges_df = ideal_ne_df_list[t_outside]
                for k in nodes_ideal_attrs:
                    nodes_df[f'{k}_ideal'] = ideal_nodes_df[k]
                    if k in node_attr:
                        if k in ['P', 'Temp']:
                            mask_k = nodes_df[k] != 0.
                            assert len(mask_k[mask_k == True]) == 29
                            nodes_df.loc[mask_k, k] =  nodes_df.loc[mask_k, k] - ideal_nodes_df.loc[mask_k, k]
                        else:
                            nodes_df[k] -= ideal_nodes_df[k]  # вычитание идеальных значений
                for k in edges_ideal_attrs:
                    edges_df[f'{k}_ideal'] = ideal_edges_df[k]
                    if k in edge_attr or k in edge_label:
                        edges_df[k] -= ideal_edges_df[k]  # вычитание идеальных значений

            else:
                print(f"[IDEAL] Предупреждение: Идеальные данные для nodes_fp={nodes_fp}, t_outside={t_outside} не найдены. Используются исходные данные.")

        for ideal_nodes_df in ideal_nodes_dataframes:
            for k in ideal_nodes_df.columns:
                ideal_nodes_df[f'{k}_ideal'] = ideal_nodes_df[k]
        for ideal_edges_df in ideal_edges_dataframes:
                for k in ideal_edges_df.columns:
                    ideal_edges_df[f'{k}_ideal'] = ideal_edges_df[k]

    print("Обучение скейлеров...")
    # if add_ideal: # я не помню что здесь происходило
    #     scalers = fit_global_scalers(nodes_dataframes + ideal_nodes_dataframes, edges_dataframes + ideal_edges_dataframes, global_dataframes + ideal_global_dataframes,
    #                                 node_attr, edge_attr, global_attr, edge_label, scaler_fn=scaler_fn)
    # else:
    scalers = fit_global_scalers(nodes_dataframes, edges_dataframes, global_dataframes,
                            node_attr, edge_attr, global_attr, edge_label, scaler_fn=scaler_fn)

    if add_ideal:
        print("[IDEAL] Нормализация таблиц...")
        ideal_nodes_dataframes, ideal_edges_dataframes, ideal_global_dataframes = normalize_dataframes(
            ideal_nodes_dataframes, ideal_edges_dataframes, ideal_global_dataframes, node_attr, edge_attr, global_attr, edge_label, scalers)

    print("Нормализация таблиц...")
    nodes_dataframes, edges_dataframes, global_dataframes = normalize_dataframes(
        nodes_dataframes, edges_dataframes, global_dataframes, node_attr, edge_attr, global_attr, edge_label, scalers)

    ideal_dataset = []
    if add_ideal:
        print("[IDEAL] Конвертация в PyG Data...")
        for nodes_df, edges_df, global_df,  (nodes_fp, edges_fp) in tqdm.tqdm(zip(ideal_nodes_dataframes, ideal_edges_dataframes, ideal_global_dataframes, ideal_files_list), total=len(ideal_nodes_dataframes)):
            try:
                data = process_dataframes(nodes_df, edges_df, global_df,  node_attr, edge_attr, edge_label, nodes_fp, edges_fp)
                ideal_dataset.append(data)
            except Exception as e:
                error_msg = f"[IDEAL] Ошибка обработки файлов:\n- Узлы: {nodes_fp}\n- Ребра: {edges_fp}\nПричина: {str(e)}"
                print(error_msg)
                with open("data_processing_errors.log", "a") as log_file:
                    log_file.write(f"{get_str_timestamp()} | {error_msg}\n")

    print("Конвертация в PyG Data...")
    dataset = []
    for nodes_df, edges_df, global_df, (nodes_fp, edges_fp) in tqdm.tqdm(zip(nodes_dataframes, edges_dataframes, global_dataframes, files_list), total=len(nodes_dataframes)):
        try:
            data = process_dataframes(nodes_df, edges_df, global_df, node_attr, edge_attr, edge_label, nodes_fp, edges_fp)
            dataset.append(data)
        except Exception as e:
            error_msg = f"Ошибка обработки файлов:\n- Узлы: {nodes_fp}\n- Ребра: {edges_fp}\nПричина: {str(e)}"
            print(error_msg)
            with open("data_processing_errors.log", "a") as log_file:
                log_file.write(f"{get_str_timestamp()} | {error_msg}\n")

    return dataset, scalers, ideal_dataset


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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=paired_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=paired_collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=paired_collate)
    return train_loader, val_loader, test_loader


def prepare_data(dataset_config, dataloader_config, seed=42, prepare_dataloaders=True):
    """Основная функция подготовки данных: загрузка или создание датасета.
    Параметр prepare_dataloaders управляет разделением и созданием DataLoader'ов."""
    # Загрузка или создание датасета
    if dataset_config['load'] and Path(dataset_config['fp']).exists():
        if dataset_config['name'] != 'Termo_model_fwd_and_bwd':
            try:
                dataset_dict = torch.load(dataset_config['fp'])
            except FileNotFoundError:
                raise FileNotFoundError(f"Файл датасета не найден: {dataset_config['fp']}")
            except Exception as e:
                raise RuntimeError(f"Ошибка загрузки датасета из {dataset_config['fp']}: {e}")
            dataset = dataset_dict.get('dataset', [])
            scalers = dataset_dict.get('scalers', [])
            ideal_dataset = dataset_dict.get('ideal_dataset', [])
            print(f"Датасет загружен из файла: {dataset_config['fp']}")
        else:
            try:
                dataset_dict = torch.load(dataset_config['fp'], weights_only=False)
            except FileNotFoundError:
                raise FileNotFoundError(f"Файл датасета не найден: {dataset_config['fp']}")
            except Exception as e:
                raise RuntimeError(f"Ошибка загрузки датасета из {dataset_config['fp']}: {e}")
            fwd_dict = dataset_dict.get('fwd', {})
            bwd_dict = dataset_dict.get('bwd', {})
            dataset_fwd = fwd_dict.get('dataset', [])
            scalers_fwd = fwd_dict.get('scalers', [])
            ideal_dataset_fwd = fwd_dict.get('ideal_dataset', [])
            dataset_bwd = bwd_dict.get('dataset', [])
            scalers_bwd = bwd_dict.get('scalers', [])
            ideal_dataset_bwd = bwd_dict.get('ideal_dataset', [])
            dataset = PairedGraphDataset(dataset_fwd, dataset_bwd)
            
            scalers, ideal_dataset = [scalers_fwd, scalers_bwd], [ideal_dataset_fwd, ideal_dataset_bwd]
            
            print(f"Датасет загружен из файла: {dataset_config['fp']}")
    else:
        print("Создание датасета...")
        if dataset_config['name'] != 'Termo_model_fwd_and_bwd':
            dataset_path = osp.join(dataset_config['datasets_dir'], dataset_config['name'])
            dataset, scalers, ideal_dataset = create_dataset(
                str(dataset_path),
                dataset_config['node_attr'],
                dataset_config['edge_attr'],
                dataset_config['edge_label'],
                num_samples=dataset_config.get('num_samples'),
                seed=seed,
                scaler_fn=dataset_config.get('scaler_fn'),
                add_ideal=dataset_config.get('add_ideal', False)
            )
            torch.save({'dataset': dataset, 'scalers': scalers, 'ideal_dataset': ideal_dataset}, dataset_config['fp'])
            print(f"Датасет сохранен в файл: {dataset_config['fp']}")
        if dataset_config['name'] == 'Termo_model_fwd_and_bwd':
            dataset_path = osp.join(dataset_config['datasets_dir'], 'Termo_model_fwd')
            dataset_fwd, scalers_fwd, ideal_dataset_fwd = create_dataset(
                str(dataset_path),
                dataset_config['node_attr'],
                dataset_config['edge_attr'],
                dataset_config['edge_label'],
                num_samples=dataset_config.get('num_samples'),
                seed=seed,
                scaler_fn=dataset_config.get('scaler_fn'),
                add_ideal=dataset_config.get('add_ideal', False)
            )
            dataset_path = osp.join(dataset_config['datasets_dir'], 'Termo_model_bwd')
            dataset_bwd, scalers_bwd, ideal_dataset_bwd = create_dataset(
                str(dataset_path),
                dataset_config['node_attr'],
                dataset_config['edge_attr'],
                dataset_config['edge_label'],
                num_samples=dataset_config.get('num_samples'),
                seed=seed,
                scaler_fn=dataset_config.get('scaler_fn'),
                add_ideal=dataset_config.get('add_ideal', False)
            )
            dataset = PairedGraphDataset(dataset_fwd, dataset_bwd)
            fwd_dict = {'dataset': dataset_fwd, 'scalers': scalers_fwd, 'ideal_dataset': ideal_dataset_fwd}
            bwd_dict = {'dataset': dataset_bwd, 'scalers': scalers_bwd, 'ideal_dataset': ideal_dataset_bwd}
            
            scalers, ideal_dataset = [scalers_fwd, scalers_bwd], [ideal_dataset_fwd, ideal_dataset_bwd]
        
            torch.save({'fwd' : fwd_dict, 'bwd' : bwd_dict}, dataset_config['fp'])
            print(f"Датасет сохранен в файл: {dataset_config['fp']}")

    print(f"Готово! Количество графов: {len(dataset)}, идеальных графов: {len(ideal_dataset)}")

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

    return dataset, scalers, train_loader, val_loader, test_loader, ideal_dataset


def data_to_tables(in_data,
                   node_attr, edge_attr, edge_label,
                   scalers=None, edge_label_pred=None, global_attr= ['Q', 'Temp']):
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
            [nodes_df], [edges_df], None,  node_attr, edge_attr, global_attr, [], scalers)
        nodes_df = nodes_df[0]
        edges_df = edges_df[0]

    return nodes_df, edges_df


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
        sum_dia[mask0] += labels[mask0]
        count_dia[mask0] += 1
    # Чтобы не делить на 0, можно оставить идеал там, где count_dia==0 равным 0
    ideal_dia = sum_dia / count_dia.clamp(min=1)

    # 2) Для каждого графа считаем отклонение и предсказанный класс
    total1 = total2 = 0
    correct1 = correct2 = 0
    dev_list = []
    pred_moded_list = []

    for d in all_data:
        pred = d.edge_label_pred.cpu()
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

        total1 += mask1.sum().item()
        total2 += mask2.sum().item()
        correct1 += (pred_moded[mask1] == 1).sum().item()
        correct2 += (pred_moded[mask2] == 2).sum().item()

    acc1 = correct1 / total1 if total1 > 0 else 0.0
    acc2 = correct2 / total2 if total2 > 0 else 0.0

    return ideal_dia, acc1, acc2, dev_list, pred_moded_list
