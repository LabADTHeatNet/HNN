import os
import shutil
import random
import math
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset

#############################
# Вспомогательные функции
#############################


def euclidean_distance(p1, p2):
    '''Евклидово расстояние между точками p1 и p2.'''
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def orientation(p, q, r):
    '''Определяет ориентацию трёх точек.
       Возвращает 0 – коллинеарны, 1 – по часовой, 2 – против.
    '''
    val = (q[1]-p[1])*(r[0]-q[0]) - (q[0]-p[0])*(r[1]-q[1])
    if abs(val) < 1e-9:
        return 0
    return 1 if val > 0 else 2


def on_segment(p, q, r):
    '''Проверяет, лежит ли точка q на отрезке pr.'''
    if min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1]):
        return True
    return False


def do_segments_intersect(a, b, c, d):
    '''
    Проверяет, пересекаются ли отрезки ab и cd.
    Если отрезки имеют общую точку (концы) – пересечение не считается.
    '''
    # Если отрезки имеют общую точку, разрешаем пересечение:
    if a == c or a == d or b == c or b == d:
        return False

    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)

    # Общий случай
    if o1 != o2 and o3 != o4:
        return True
    # Обработка коллинеарных случаев
    if o1 == 0 and on_segment(a, c, b):
        return True
    if o2 == 0 and on_segment(a, d, b):
        return True
    if o3 == 0 and on_segment(c, a, d):
        return True
    if o4 == 0 and on_segment(c, b, d):
        return True
    return False


def edge_segments_intersect(nodes, edge1, edge2):
    '''
    Проверяет, пересекаются ли два ребра, заданные индексами узлов.
    Узлы берутся из списка nodes, где каждый узел – словарь с ключами 'x' и 'y'.
    Если ребра имеют общий узел, пересечение не считается.
    '''
    i1, j1 = edge1
    i2, j2 = edge2
    p1 = (nodes[i1]['x'], nodes[i1]['y'])
    p2 = (nodes[j1]['x'], nodes[j1]['y'])
    p3 = (nodes[i2]['x'], nodes[i2]['y'])
    p4 = (nodes[j2]['x'], nodes[j2]['y'])
    return do_segments_intersect(p1, p2, p3, p4)

#############################
# Функция сортировки ребер
#############################


def sort_edges(edges):
    '''
    Сортирует список ребер по возрастанию: сначала по исходному узлу, затем по целевому.
    '''
    return sorted(edges, key=lambda e: (e[0], e[1]))

#############################
# Генерация структуры графа
#############################


def generate_nodes(node_num, coord_range):
    '''Генерирует список узлов с фиксированными координатами.'''
    nodes = []
    for i in range(node_num):
        node = {
            'id': i,
            'x': random.uniform(*coord_range),
            'y': random.uniform(*coord_range)
        }
        nodes.append(node)
    return nodes


def generate_tree_edges(nodes):
    '''
    Генерирует остовное дерево с корнем в узле 0 по алгоритму Прима.
    Данный алгоритм выбирает на каждом шаге минимальное ребро,
    соединяющее уже выбранные узлы с оставшимися.
    MST, вычисленный таким образом, является планарным (ребра не пересекаются).
    '''
    n = len(nodes)
    in_tree = [False] * n
    in_tree[0] = True
    tree_edges = []

    while sum(in_tree) < n:
        best_edge = None
        best_dist = float('inf')
        for i in range(n):
            if in_tree[i]:
                for j in range(n):
                    if not in_tree[j]:
                        p1 = (nodes[i]['x'], nodes[i]['y'])
                        p2 = (nodes[j]['x'], nodes[j]['y'])
                        d = euclidean_distance(p1, p2)
                        if d < best_dist:
                            best_dist = d
                            best_edge = (i, j)
        if best_edge is None:
            break
        tree_edges.append(best_edge)
        in_tree[best_edge[1]] = True
    return tree_edges


def generate_random_edges(nodes, edge_num):
    '''
    Генерирует случайный набор ребер (из всех возможных пар)
    с проверкой на отсутствие пересечений. Если ребро пересекается с уже выбранными,
    оно не добавляется. Если добавить требуемое число ребер невозможно,
    функция выводит предупреждение.
    '''
    n = len(nodes)
    candidate_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(candidate_edges)

    selected_edges = []
    for edge in candidate_edges:
        intersect = False
        for exist_edge in selected_edges:
            if edge_segments_intersect(nodes, edge, exist_edge):
                intersect = True
                break
        if not intersect:
            selected_edges.append(edge)
            if len(selected_edges) == edge_num:
                break
    if len(selected_edges) < edge_num:
        print('Warning: Не удалось создать требуемое количество ребер без пересечений. Сгенерировано:', len(selected_edges))
    return selected_edges

####################################
# Функции для генерации параметров
####################################


def generate_node_parameters(node_attr_dict):
    '''
    Генерирует параметры для узла по словарю диапазонов.
    Например, если node_attr_dict = {'attr': (0, 10)},
    то будет сгенерировано значение attr в диапазоне [0, 10].
    '''
    params = {}
    for attr, attr_range in node_attr_dict.items():
        params[attr] = random.uniform(*attr_range)
    return params


def generate_edge_parameters(edge_attr_dict):
    '''
    Генерирует параметры для ребра по заданному словарю диапазонов.
    '''
    params = {}
    for attr, attr_range in edge_attr_dict.items():
        params[attr] = random.uniform(*attr_range)
    return params

####################################
# Генерация сэмплов (используем общую структуру)
####################################


def generate_samples(samples_num, nodes_fixed, edges_fixed, node_attr_dict, edge_attr_dict, edge_target_fn, dataset_dir):
    '''
    Для каждого сэмпла:
      1. Структура узлов (id, x, y) берётся из nodes_fixed.
      2. Для каждого узла генерируются случайные параметры из node_attr_dict.
      3. Для каждого ребра генерируются случайные параметры из edge_attr_dict.
      4. Для ребра вычисляются целевые значения с помощью edge_target_fn
         (с использованием параметров узлов, полученных на данном сэмпле).
      5. Результаты сохраняются в CSV-файлы.
    '''
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir, ignore_errors=True)
    os.makedirs(dataset_dir)

    for sample_idx in range(samples_num):
        # Генерация параметров для узлов
        sample_nodes = []
        node_params_sample = {}  # для удобства доступа по id
        for node in nodes_fixed:
            params = generate_node_parameters(node_attr_dict)
            node_sample = {
                'id': node['id'],
                'x': node['x'],
                'y': node['y']
            }
            node_sample.update(params)
            sample_nodes.append(node_sample)
            node_params_sample[node['id']] = params

        # Генерация параметров для ребер и вычисление целевых значений
        sample_edges = []
        for edge in edges_fixed:
            src, tgt = edge
            edge_sample = {'source': src, 'target': tgt}
            edge_params = generate_edge_parameters(edge_attr_dict)
            edge_sample.update(edge_params)
            targets = edge_target_fn(node_params_sample[src], node_params_sample[tgt])
            edge_sample.update(targets)
            sample_edges.append(edge_sample)

        # Сохранение в CSV
        nodes_df = pd.DataFrame(sample_nodes)
        edges_df = pd.DataFrame(sample_edges)
        nodes_file = os.path.join(dataset_dir, f'nodes_{sample_idx:06d}.csv')
        edges_file = os.path.join(dataset_dir, f'edges_{sample_idx:06d}.csv')
        nodes_df.to_csv(nodes_file, index=False)
        edges_df.to_csv(edges_file, index=False)
        print(f'Сгенерирован сэмпл {sample_idx:06d}')

####################################
# Класс для формирования датасета с помощью PyTorch Geometric
####################################


class MyGraphDataset(InMemoryDataset):
    '''
    При создании датасета необходимо передать:
      root: путь к директории с CSV-файлами,
      node_attr_cols: список столбцов узловых признаков,
      edge_attr_cols: список столбцов параметров ребра,
      edge_target_cols: список столбцов целевых значений ребра.
    '''

    def __init__(self, root, node_attr_cols=None, edge_attr_cols=None, edge_target_cols=None, transform=None, pre_transform=None):
        self.node_attr_cols = node_attr_cols
        self.edge_attr_cols = edge_attr_cols
        self.edge_target_cols = edge_target_cols
        super(MyGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        files = os.listdir(self.root)
        node_files = sorted([f for f in files if f.startswith('nodes_') and f.endswith('.csv')])
        for node_file in node_files:
            sample_idx = node_file.split('_')[1].split('.')[0]
            edge_file = f'edges_{sample_idx}.csv'
            node_path = os.path.join(self.root, node_file)
            edge_path = os.path.join(self.root, edge_file)
            nodes_df = pd.read_csv(node_path)
            edges_df = pd.read_csv(edge_path)

            # Формирование признаков узлов – выбираем только нужные столбцы
            x = torch.tensor(nodes_df[self.node_attr_cols].values, dtype=torch.float)

            # Формирование индексов ребер
            edge_index = torch.tensor(
                [edges_df['source'].values, edges_df['target'].values],
                dtype=torch.long
            )

            # Получаем столбцы ребер по разделению: первые edge_attr_cols, остальные – целевые
            edge_all = edges_df.drop(columns=['source', 'target'])
            edge_attr = torch.tensor(edge_all[self.edge_attr_cols].values, dtype=torch.float)
            edge_target = torch.tensor(edge_all[self.edge_target_cols].values, dtype=torch.float)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_target=edge_target,
                        x_cols=self.node_attr_cols, edge_attr_cols=self.edge_attr_cols, edge_target_cols=self.edge_target_cols)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

####################################
# Основная функция
####################################


def main():
    # Параметры генерации структуры
    node_num = 40              # количество узлов
    coord_range = (0, 100)     # диапазон координат (x и y)
    mode = 'tree'              # режим формирования графа: 'tree' или 'random'
    edge_num = 80              # число ребер для режима 'random' (игнорируется для tree)

    # Параметры для генерации атрибутов
    node_attr_dict = {
        'attr': (0, 10)       # для узлов генерируется параметр 'attr' в диапазоне [0, 10]
    }
    edge_attr_dict = {
        'weight': (1, 5)      # для ребер генерируется параметр 'weight' в диапазоне [1, 5]
    }

    samples_num = 100           # число сэмплов
    dataset_dir = 'datasets/sum_prod'    # папка для сохранения CSV файлов

    def edge_target_fn(node_params1, node_params2):
        '''
        Вычисляет целевые параметры для ребра:
        для каждого атрибута узла вычисляется сумма и произведение значений.
        Например, если атрибут узла называется 'attr', то возвращаются ключи 'sum' и 'prod'.
        '''
        targets = {}
        targets['sum'] = node_params1['attr'] + node_params2['attr']
        targets['prod'] = node_params1['attr'] * node_params2['attr']
        return targets

    node_attr_cols = ['x', 'y', 'attr']
    edge_attr_cols = ['weight']
    edge_target_cols = ['sum', 'prod']

    # Генерация фиксированной структуры графа
    nodes_fixed = generate_nodes(node_num, coord_range)
    if mode == 'tree':
        edges_fixed = generate_tree_edges(nodes_fixed)
    elif mode == 'random':
        edges_fixed = generate_random_edges(nodes_fixed, edge_num)
    else:
        raise ValueError('Некорректный режим. Выберите "tree" или "random".')

    # Вызов функции сортировки ребер после генерации структуры графа
    edges_fixed = sort_edges(edges_fixed)

    print('Структура графа сгенерирована:')
    print('Количество узлов:', len(nodes_fixed))
    print('Количество ребер:', len(edges_fixed))

    # Генерация сэмплов (у каждого свои параметры, структура общая)
    generate_samples(samples_num, nodes_fixed, edges_fixed, node_attr_dict, edge_attr_dict, edge_target_fn, dataset_dir)

    # Создание датасета PyTorch Geometric.
    dataset = MyGraphDataset(root=dataset_dir,
                             node_attr_cols=node_attr_cols,
                             edge_attr_cols=edge_attr_cols,
                             edge_target_cols=edge_target_cols)
    print('Количество графов в датасете:', len(dataset))
    print(dataset[0])


if __name__ == '__main__':
    main()
