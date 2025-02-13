import os
import glob
import tqdm

import numpy as np
import pandas as pd
import pandapower as pp
from pandapower.results import reset_results

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, InMemoryDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from tqdm.contrib.concurrent import process_map

from dataset_pp_vis import (
    visualize_data_as_net,
    visualize_data_as_net_plotly
)


##########################
# Функции кодирования категориальных признаков
##########################


def one_hot_encode_column(df, column, categories):
    """
    Преобразует столбец с категориальными значениями в one-hot представление.
    Если некоторые из ожидаемых категорий отсутствуют, добавляет их со значением 0.

    Аргументы:
      df: DataFrame, содержащий столбец column.
      column: имя столбца для преобразования.
      categories: список всех возможных категорий (фиксированный порядок).

    Возвращает:
      DataFrame с one-hot признаками с именами вида <column>_<cat>.
    """
    # Применяем get_dummies для получения one-hot кодирования указанного столбца
    one_hot = pd.get_dummies(df[column], prefix=column)
    # Для каждой ожидаемой категории проверяем наличие соответствующего столбца и добавляем его, если отсутствует
    for cat in categories:
        col_name = f"{column}_{cat}"
        if col_name not in one_hot.columns:
            one_hot[col_name] = 0
    # Упорядочиваем столбцы в соответствии с заданным порядком категорий
    one_hot = one_hot[[f"{column}_{cat}" for cat in categories]]
    return one_hot


def one_hot_decode_column(one_hot_df, column, categories):
    """
    Преобразует one-hot представление обратно в категориальные значения.

    Аргументы:
      one_hot_df: DataFrame, содержащий one-hot столбцы с именами вида <column>_<cat>.
      column: исходное имя столбца (например, 'node_type' или 'edge_type').
      categories: список категорий в том же порядке, что использовался при кодировании.

    Возвращает:
      Series с декодированными строковыми значениями.
    """
    # Внутренняя функция для декодирования одной строки
    def decode_row(row):
        # Ищем, какая из категорий имеет значение 1
        for cat in categories:
            col_name = f"{column}_{cat}"
            if row[col_name] == 1:
                return cat
        # Если ни одна категория не найдена, возвращаем None
        return None
    # Применяем функцию декодирования к каждой строке DataFrame
    decoded = one_hot_df.apply(decode_row, axis=1)
    return decoded

##########################
# Генерация примера (CSV-файлов)
##########################


class PPDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 normalize=True,
                 scaler_type='standard',
                 num_samples=None,
                 *,
                 nodes_numeric_cols,
                 nodes_cat_cols,
                 edges_numeric_cols,
                 edges_cat_cols,
                 edges_index_cols,
                 edges_label_cols,
                 edges_moded_cols):
        # Инициализация базовых параметров набора данных
        self.root = root
        self.normalize = normalize
        self.num_samples = num_samples

        # Списки колонок для числовых и категориальных признаков узлов и ребер
        self.nodes_numeric_cols = nodes_numeric_cols
        self.nodes_cat_cols = nodes_cat_cols
        self.edges_numeric_cols = edges_numeric_cols
        self.edges_cat_cols = edges_cat_cols

        # Списки колонок, содержащих индексы ребер, метки ребер и информацию о модификации
        self.edges_index_cols = edges_index_cols
        self.edges_label_cols = edges_label_cols
        self.edges_moded_cols = edges_moded_cols

        # Инициализация скейлеров для нормализации данных на основе выбранного типа
        if scaler_type == 'standard':
            self.nodes_numeric_scaler = StandardScaler()
            self.edges_numeric_scaler = StandardScaler()
            self.edges_label_scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.nodes_numeric_scaler = MinMaxScaler()
            self.edges_numeric_scaler = MinMaxScaler()
            self.edges_label_scaler = MinMaxScaler()
        else:
            raise ValueError("Unsupported scaler_type. Use 'standard' or 'minmax'")

        # Обработка набора данных и формирование списка объектов Data
        self.data_list = self.process_dataset()

    def process_dataset(self):
        # Инициализация списков для хранения данных
        data_list = []
        # Получение списка файлов с данными узлов и ребер
        nodes_files = sorted(glob.glob(os.path.join(self.root, 'nodes_*.csv')))
        edges_files = sorted(glob.glob(os.path.join(self.root, 'tubes_*.csv')))
        # Проверка соответствия количества файлов узлов и ребер
        if len(nodes_files) != len(edges_files):
            raise ValueError("Количество файлов узлов и ребер не совпадает!")
        # Ограничение количества образцов, если задано
        if self.num_samples is not None:
            nodes_files = nodes_files[:self.num_samples]
            edges_files = edges_files[:self.num_samples]

        # Сбор уникальных категорий узлов из всех файлов
        global_nodes_types = set()
        for nf in nodes_files:
            df = pd.read_csv(nf)
            global_nodes_types.update(df['node_type'].unique())
        global_nodes_types = sorted(list(global_nodes_types))
        self.global_nodes_types = global_nodes_types

        # Сбор уникальных категорий ребер из всех файлов
        global_edges_types = set()
        for ef in edges_files:
            df = pd.read_csv(ef)
            global_edges_types.update(df['edge_type'].unique())
        global_edges_types = sorted(list(global_edges_types))
        self.global_edges_types = global_edges_types

        # Инициализация списков для хранения признаков из всех образцов
        all_nodes_numeric_feats = []
        all_edges_numeric_feats = []
        all_nodes_cat_feats = []
        all_edges_cat_feats = []
        all_edges_index = []
        all_edges_label = []
        all_edges_moded = []

        # Итерация по парам файлов узлов и ребер с отображением прогресса
        for nf, ef in tqdm.tqdm(zip(nodes_files, edges_files), total=len(nodes_files)):
            nodes_df = pd.read_csv(nf)
            edges_df = pd.read_csv(ef)

            # Заполнение пропущенных значений для числовых признаков узлов и ребер
            nodes_df[self.nodes_numeric_cols] = nodes_df[self.nodes_numeric_cols].fillna(0)
            edges_df[self.edges_numeric_cols] = edges_df[self.edges_numeric_cols].fillna(0)
            # Заполнение пропущенных значений для меток ребер
            edges_df[self.edges_label_cols] = edges_df[self.edges_label_cols].fillna(0)

            # Извлечение числовых признаков
            nodes_numeric = nodes_df[self.nodes_numeric_cols]
            edges_numeric = edges_df[self.edges_numeric_cols]
            # Преобразование категориальных признаков в one-hot представление
            nodes_cat = one_hot_encode_column(nodes_df, 'node_type', global_nodes_types)
            edges_cat = one_hot_encode_column(edges_df, 'edge_type', global_edges_types)

            # Извлечение индексных и меточных данных для ребер
            edges_index_arr = edges_df[self.edges_index_cols].values
            edges_label_arr = edges_df[self.edges_label_cols].values
            edges_moded_arr = edges_df[self.edges_moded_cols].values

            # Добавление извлечённых данных в соответствующие списки
            all_nodes_numeric_feats.append(nodes_numeric.values)
            all_nodes_cat_feats.append(nodes_cat.values)
            all_edges_numeric_feats.append(edges_numeric.values)
            all_edges_cat_feats.append(edges_cat.values)
            all_edges_index.append(edges_index_arr)
            all_edges_label.append(edges_label_arr)
            all_edges_moded.append(edges_moded_arr)

        # Конкатенация данных для масштабирования по всем образцам
        all_nodes_numeric_cat = np.concatenate(all_nodes_numeric_feats, axis=0)
        all_edges_numeric_cat = np.concatenate(all_edges_numeric_feats, axis=0)
        all_edges_label_cat = np.concatenate(all_edges_label, axis=0)

        # Применение нормализации, если она включена
        if self.normalize:
            self.nodes_numeric_scaler.fit(all_nodes_numeric_cat)
            self.edges_numeric_scaler.fit(all_edges_numeric_cat)
            self.edges_label_scaler.fit(all_edges_label_cat)

        # Формирование объектов Data для каждого образца
        for i in range(len(all_nodes_numeric_feats)):
            # Нормализация числовых признаков узлов и ребер для текущего образца
            nodes_numeric = (self.nodes_numeric_scaler.transform(all_nodes_numeric_feats[i])
                             if self.normalize else all_nodes_numeric_feats[i])
            edges_numeric = (self.edges_numeric_scaler.transform(all_edges_numeric_feats[i])
                             if self.normalize else all_edges_numeric_feats[i])
            edges_label = (self.edges_label_scaler.transform(all_edges_label[i])
                           if self.normalize else all_edges_label[i])

            # Преобразование данных в тензоры PyTorch
            x = torch.tensor(nodes_numeric, dtype=torch.float)
            nodes_cats = torch.tensor(all_nodes_cat_feats[i], dtype=torch.float)
            edges_attr = torch.tensor(edges_numeric, dtype=torch.float)
            edges_cats = torch.tensor(all_edges_cat_feats[i], dtype=torch.float)
            edges_index_tensor = torch.tensor(all_edges_index[i].T, dtype=torch.long)
            edges_label_tensor = torch.tensor(edges_label, dtype=torch.float)
            edges_moded_tensor = torch.tensor(all_edges_moded[i], dtype=torch.float)

            # Создание объекта Data, содержащего все признаки для текущего образца
            data = Data(x=x,
                        nodes_cats=nodes_cats,
                        edges_index=edges_index_tensor,
                        edges_attr=edges_attr,
                        edges_cats=edges_cats,
                        edges_label=edges_label_tensor,
                        edges_moded=edges_moded_tensor)
            data_list.append(data)

        return data_list

    def __len__(self):
        # Возвращает количество образцов в наборе данных
        return len(self.data_list)

    def __getitem__(self, idx):
        # Позволяет обращаться к образцам по индексу
        return self.data_list[idx]

    def denormalize_data(self, data):
        # Функция обратного масштабирования данных (денормализация) для визуализации или анализа
        if self.normalize:
            x = torch.tensor(self.nodes_numeric_scaler.inverse_transform(data.x), dtype=data.x.dtype)
            edges_attr = torch.tensor(self.edges_numeric_scaler.inverse_transform(data.edges_attr), dtype=data.edges_attr.dtype)
            edges_label = torch.tensor(self.edges_label_scaler.inverse_transform(data.edges_label), dtype=data.edges_label.dtype)
            denorm_data = Data(x=x,
                               nodes_cats=data.nodes_cats,
                               edges_index=data.edges_index,
                               edges_attr=edges_attr,
                               edges_cats=data.edges_cats,
                               edges_label=edges_label,
                               edges_moded=data.edges_moded)
            return denorm_data
        else:
            return data

    def sample_to_dfs(self, data):
        # Преобразует объект Data в два DataFrame: один для узлов и один для ребер

        # Формирование DataFrame для узлов на основе числовых признаков
        nodes_numeric_df = pd.DataFrame(data.x.numpy(), columns=self.nodes_numeric_cols)
        # Формирование DataFrame для категориальных признаков узлов (one-hot)
        nodes_cat_columns = [f"node_type_{cat}" for cat in self.global_nodes_types]
        nodes_cat_df = pd.DataFrame(data.nodes_cats.numpy(), columns=nodes_cat_columns)
        # Декодирование one-hot представления обратно в строковые значения
        decoded_types = one_hot_decode_column(nodes_cat_df, 'node_type', self.global_nodes_types)
        nodes_numeric_df['node_type'] = decoded_types
        # Добавление идентификаторов узлов
        nodes_numeric_df['id'] = np.arange(len(nodes_numeric_df))
        nodes_df = nodes_numeric_df

        # Формирование DataFrame для ребер: извлечение индексных столбцов
        edges_index_np = data.edges_index.numpy()
        edges_index_df = pd.DataFrame({self.edges_index_cols[i]: edges_index_np[i] for i in range(edges_index_np.shape[0])})
        # Формирование DataFrame для числовых признаков ребер
        edges_numeric_df = pd.DataFrame(data.edges_attr.numpy(), columns=self.edges_numeric_cols)
        # Формирование DataFrame для категориальных признаков ребер (one-hot)
        edges_cat_columns = [f"edge_type_{cat}" for cat in self.global_edges_types]
        edges_cat_df = pd.DataFrame(data.edges_cats.numpy(), columns=edges_cat_columns)
        # Декодирование one-hot представления ребер обратно в строковые значения
        decoded_edges_types = one_hot_decode_column(edges_cat_df, 'edge_type', self.global_edges_types)
        edges_numeric_df['edge_type'] = decoded_edges_types
        # Добавление метки ребра (например, r_ohm_per_km) в числовой DataFrame
        edges_label_np = data.edges_label.numpy()
        edges_numeric_df[self.edges_label_cols[0]] = edges_label_np.flatten()
        # Формирование DataFrame для информации о модификации ребер
        edges_moded_np = data.edges_moded.numpy().T
        edges_moded_df = pd.DataFrame({self.edges_moded_cols[i]: edges_moded_np[i] for i in range(edges_moded_np.shape[0])})

        # Объединение всех данных ребер в один DataFrame
        edges_df = pd.concat([edges_index_df, edges_numeric_df, edges_moded_df], axis=1)

        return nodes_df, edges_df


def data_to_net(nodes_df, edges_df):
    """
    Преобразует объект Data (sample) в pandapower-сеть.
    Использует информацию из узловой и реберной таблиц (dataset.sample_to_dfs(sample)).
    При создании линий сохраняет значение edges_label в net.line.
    """
    # Создание пустой электросети с использованием pandapower
    net = pp.create_empty_network()
    # Инициализация DataFrame для геоданных шин
    net.bus_geodata = pd.DataFrame(columns=['x', 'y'])
    # Сброс результатов предыдущих расчетов
    reset_results(net)
    # Словарь для сопоставления оригинальных идентификаторов узлов с новыми id, созданными в pandapower
    bus_map = {}
    # Список для сохранения значений напряжения на шинах
    vm_pu_list = []
    # Итерация по строкам DataFrame узлов
    for idx, row in nodes_df.iterrows():
        # Создание шины в сети с заданными параметрами (номинальное напряжение, статус работы)
        bus_id = pp.create_bus(net, vn_kv=row['vn_kv'], name=f"bus_{row['id']}",
                               in_service=bool(row['in_service']))
        vm_pu_list.append(row['vm_pu'])
        # Заполнение геоданных для созданной шины
        net.bus_geodata.loc[bus_id] = [row['pos_x'], row['pos_y']]
        # Сохранение сопоставления между исходным id и новым id шины
        bus_map[row['id']] = bus_id
        # Добавление нагрузки, если таковая присутствует
        if 'load_count' in row and row['load_count'] > 1e-1:
            pp.create_load(net, bus=bus_id, p_mw=row['p_load_mw'], q_mvar=row['q_load_mvar'])
        # Добавление генератора, если таковой присутствует
        if 'gen_count' in row and row['gen_count'] > 1e-1:
            pp.create_gen(net, bus=bus_id, p_mw=row['p_gen_mw'], q_mvar=row['q_gen_mvar'])
        # Добавление шунта, если таковой присутствует
        if 'shunt_count' in row and row['shunt_count'] > 1e-1:
            pp.create_shunt(net, bus=bus_id, q_mvar=row['shunt_q_mvar'])
        # Добавление внешней сети, если таковая присутствует
        if 'ext_grid_count' in row and row['ext_grid_count'] > 1e-1:
            pp.create_ext_grid(net, bus=bus_id, vm_pu=row['ext_grid_vm_pu'])
    # Сохранение рассчитанных значений напряжения шин
    net.res_bus['vm_pu'] = vm_pu_list

    # Добавление линий и трансформаторов в сеть
    loading_percent_list = []
    for idx, row in edges_df.iterrows():
        # Получение новых id шин для начальной и конечной точек ребра
        from_bus = bus_map.get(row['from_bus'], None)
        to_bus = bus_map.get(row['to_bus'], None)
        if from_bus is None or to_bus is None:
            continue
        if row['edge_type'] == 'line':
            # Создание линии в сети с заданной длиной и стандартным типом линии
            new_line = pp.create_line(net, from_bus=from_bus, to_bus=to_bus,
                                      length_km=row['length_km'], std_type="NAYY 4x50 SE", name=f"line_{idx}")
            # Сохранение значений сопротивления линии и флага модификации в соответствующих столбцах
            net.line.at[new_line, 'r_ohm_per_km'] = row.get('r_ohm_per_km', 0)
            net.line.at[new_line, 'r_ohm_per_km_ideal'] = row.get('r_ohm_per_km_ideal', 0)
            net.line.at[new_line, 'moded'] = row.get('moded', 0)

        elif row['edge_type'] == 'trafo':
            # Создание трансформатора в сети с заданным стандартным типом
            new_trafo = pp.create_transformer(net, hv_bus=from_bus, lv_bus=to_bus,
                                              std_type="25 MVA 110/20 kV", name=f"trafo_{idx}")
            loading_percent_list.append(row['loading_percent'])
    # Сохранение информации о загрузке трансформаторов
    net.res_trafo['loading_percent'] = loading_percent_list

    return net

##########################
# Пример использования
##########################

##########################
# (For plotly) Execute before start:
# export LD_PRELOAD=/lib/x86_64-linux-gnu/libpthread.so.0
##########################


if __name__ == '__main__':
    # Указание директории с набором данных
    dataset_dir = 'datasets/case14'
    # dataset_dir = 'datasets/case118'

    # Задание количества образцов для обработки
    num_samples = 10

    # Определение списка колонок для числовых признаков узлов

    # Оставляем только физически значимые признаки для GNN (убраны строковые идентификаторы и ненужный poly_cost_count)
    nodes_numeric_cols = ['pos_x',
                          'pos_y',
                          'vn_kv',
                          'in_service',
                          'vm_pu',
                          'va_degree',
                          'p_load_mw',
                          'q_load_mvar',
                          'load_count',
                          'p_gen_mw',
                          'q_gen_mvar',
                          'gen_count']

    # nodes_numeric_cols = ['pos_x',
    #                       'pos_y',
    #                       'vn_kv',
    #                       'in_service',
    #                       'moded',
    #                       'vm_pu',
    #                       'va_degree',
    #                       'p_load_mw',
    #                       'q_load_mvar',
    #                       'load_count',
    #                       'load_ids',
    #                       'p_gen_mw',
    #                       'q_gen_mvar',
    #                       'gen_count',
    #                       'shunt_q_mvar',
    #                       'shunt_count',
    #                       'shunt_ids',
    #                       'ext_grid_vm_pu',
    #                       'ext_grid_count',
    #                       'ext_grid_ids',
    #                       'poly_cost_count']

    # Определение списка колонок для категориальных признаков узлов
    nodes_cat_cols = ['node_type']
    # Определение списка колонок для числовых признаков ребер
    edges_numeric_cols = ['length_km',
                          'r_ohm_per_km_ideal',
                          'loading_percent']

    # Определение списка колонок для категориальных признаков ребер
    edges_cat_cols = ['edge_type']
    # Определение списка колонок, содержащих индексы ребер
    edges_index_cols = ['from_bus', 'to_bus']
    # Определение списка колонок, содержащих метки ребер
    edges_label_cols = ['r_ohm_per_km']

    # Глобальные категории для узлов и ребер
    nodes_cats = ['b', 'slack', 'PQ', 'PV']
    edges_cats = ['line', 'trafo']

    # Определение списка колонок для информации о модификации ребер
    edges_moded_cols = ['moded']

    # Задание параметров для батч-обработки данных
    batch_size = 4

    # Параметры нормализации и настройки визуализации
    normalize = True
    scaler_type = 'standard'
    figsize = (20, 20)

    # Инициализация набора данных с использованием ранее заданных параметров
    dataset = PPDataset(root=dataset_dir,
                        normalize=normalize,
                        scaler_type=scaler_type,
                        num_samples=num_samples,
                        nodes_numeric_cols=nodes_numeric_cols,
                        nodes_cat_cols=nodes_cat_cols,
                        edges_numeric_cols=edges_numeric_cols,
                        edges_cat_cols=edges_cat_cols,
                        edges_index_cols=edges_index_cols,
                        edges_label_cols=edges_label_cols,
                        edges_moded_cols=edges_moded_cols)

    # Вывод количества образцов в наборе данных
    print(f"Dataset size: {len(dataset)} samples")

    # Инициализация загрузчика данных для формирования батчей
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Получение первого батча данных
    batch = next(iter(loader))
    data_list = batch.to_data_list()
    # Обработка каждого образца из батча
    for idx in range(len(data_list)):
        sample = data_list[idx]

        # Преобразование образца в DataFrame для узлов и ребер
        nodes_df, edges_df = dataset.sample_to_dfs(sample)
        # (Опционально) Вывод DataFrame для проверки содержимого
        # print("nodes DataFrame:")
        # print(nodes_df)
        # print("\nedges DataFrame:")
        # print(edges_df)

        # Денормализация данных (при необходимости) для интерпретации в исходных масштабах
        sample = loader.dataset.denormalize_data(sample)
        print(f"Denormalized sample data: {sample}")
        nodes_df, edges_df = dataset.sample_to_dfs(sample)
        print("nodes DataFrame (denorm):")
        print(nodes_df)
        print("\nedges DataFrame (denorm):")
        print(edges_df)

        # (Опционально) Визуализация графа с использованием цветовой кодировки
        # visualize_sample(sample, title="Граф с цветовой кодировкой",
        #                  nodes_categories=nodes_cats, edges_categories=edges_cats)

        # Преобразование объекта Data в pandapower-сеть для последующего анализа или визуализации
        net = data_to_net(nodes_df, edges_df)
        # (Опционально) Визуализация сети с использованием pandapower
        # visualize_data_as_net(net,
        #                       figsize=figsize,
        #                       title="Pandapower Visualization of Sample")

        # Визуализация сети с использованием Plotly
        visualize_data_as_net_plotly(net,
                                     figsize=figsize,
                                     title="Pandapower Visualization of Sample")
