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


def generate_nodes(node_num, coord_range, min_distance=5.0, max_attempts=1000):
    """
    Генерирует список узлов с фиксированными координатами так, чтобы между любыми двумя точками
    расстояние было не меньше min_distance.

    Аргументы:
      node_num: требуемое число узлов.
      coord_range: диапазон координат (например, (0, 100)).
      min_distance: минимальное расстояние между точками.
      max_attempts: максимальное число попыток для генерации каждой точки.

    Возвращает:
      nodes: список узлов в формате [{'id': 0, 'x': ..., 'y': ...}, ...]
    """
    nodes = []
    attempts = 0
    while len(nodes) < node_num and attempts < max_attempts * node_num:
        x = random.uniform(*coord_range)
        y = random.uniform(*coord_range)
        valid = True
        for node in nodes:
            dist = ((node['x'] - x)**2 + (node['y'] - y)**2)**0.5
            if dist < min_distance:
                valid = False
                break
        if valid:
            nodes.append({'id': len(nodes), 'x': x, 'y': y})
        attempts += 1
    if len(nodes) < node_num:
        print("Warning: Не удалось сгенерировать заданное число узлов при условии минимального расстояния.")
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


def compute_angle_between_edges(nodes, edge1, edge2):
    """
    Вычисляет угол между двумя ребрами (edge1 и edge2), если они имеют общую вершину.
    Если общей вершины нет, возвращает None.
    Аргументы:
      nodes: список узлов, где каждый узел – словарь с ключами 'x' и 'y'
      edge1, edge2: кортежи (i, j) с индексами узлов
    Возвращает угол в радианах.
    """
    common = set(edge1).intersection(set(edge2))
    if not common:
        return None
    v = common.pop()

    def other_vertex(edge, v):
        return edge[0] if edge[1] == v else edge[1]
    v1 = other_vertex(edge1, v)
    v2 = other_vertex(edge2, v)
    p_v = (nodes[v]['x'], nodes[v]['y'])
    p_v1 = (nodes[v1]['x'], nodes[v1]['y'])
    p_v2 = (nodes[v2]['x'], nodes[v2]['y'])
    a = (p_v1[0] - p_v[0], p_v1[1] - p_v[1])
    b = (p_v2[0] - p_v[0], p_v2[1] - p_v[1])
    norm_a = math.sqrt(a[0]**2 + a[1]**2)
    norm_b = math.sqrt(b[0]**2 + b[1]**2)
    if norm_a == 0 or norm_b == 0:
        return 0
    dot = a[0]*b[0] + a[1]*b[1]
    cos_angle = max(min(dot/(norm_a*norm_b), 1.0), -1.0)
    angle = math.acos(cos_angle)
    return angle


def generate_connected_edges(nodes, edge_num, min_edge_angle, k_neighbors=5):
    """
    Генерирует набор ребер для одного связного графа.

    Сначала строится минимальное остовное дерево (MST) по алгоритму Прима для обеспечения связности (все узлы участвуют).
    Затем, если требуется больше ребер (edge_num > n-1), добавляются дополнительные ребра из кандидатного набора,
    выбранного среди k ближайших соседей для каждого узла.

    Дополнительное ребро добавляется, если:
      - оно не пересекается с уже выбранными ребрами (функция edge_segments_intersect),
      - для ребер с общей вершиной угол между ними не меньше min_edge_angle (в радианах).

    Возвращает отсортированный список ребер (каждое ребро – кортеж (i, j), i < j).
    """
    # Сначала строим MST для обеспечения связности
    tree_edges = generate_tree_edges(nodes)
    selected_edges = list(tree_edges)

    n = len(nodes)
    # Если MST уже содержит требуемое число ребер, возвращаем его
    if len(selected_edges) >= edge_num:
        return sort_edges(selected_edges[:edge_num])

    # Формируем кандидатный набор дополнительных ребер.
    # Для каждого узла рассматриваем k ближайших соседей.
    extra_candidates = set()
    for i in range(n):
        distances = []
        for j in range(n):
            if i == j:
                continue
            d = euclidean_distance((nodes[i]['x'], nodes[i]['y']), (nodes[j]['x'], nodes[j]['y']))
            distances.append((d, j))
        distances.sort(key=lambda x: x[0])
        for _, j in distances[:k_neighbors]:
            edge = (i, j) if i < j else (j, i)
            if edge in selected_edges:
                continue
            extra_candidates.add(edge)
    extra_candidates = list(extra_candidates)
    random.shuffle(extra_candidates)

    # Добавляем кандидатов, проверяя условия
    for edge in extra_candidates:
        valid = True
        # Проверка пересечения с уже выбранными ребрами
        for exist_edge in selected_edges:
            if edge_segments_intersect(nodes, edge, exist_edge):
                valid = False
                break
        if not valid:
            continue
        # Проверка минимального угла для ребер, имеющих общую вершину
        for exist_edge in selected_edges:
            if set(edge).intersection(set(exist_edge)):
                angle = compute_angle_between_edges(nodes, edge, exist_edge)
                if angle is not None and angle < min_edge_angle:
                    valid = False
                    break
        if valid:
            selected_edges.append(edge)
            if len(selected_edges) == edge_num:
                break
    if len(selected_edges) < edge_num:
        print("Warning: Не удалось создать требуемое количество ребер с учетом минимального угла и близости. Сгенерировано:", len(selected_edges))
    return sort_edges(selected_edges)

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
         Параметры ребер сохраняются в словаре для доступа при вычислении target в режиме 4.
      4. Для ребра вычисляются целевые значения с помощью edge_target_fn,
         которой передаётся информация о смежности графа и словарь sample_edge_params.
      5. Результаты сохраняются в CSV-файлы.
    '''
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir, ignore_errors=True)
    os.makedirs(dataset_dir)

    # Формирование списка смежности для фиксированной структуры графа
    adjacency = {node['id']: set() for node in nodes_fixed}
    for edge in edges_fixed:
        src, tgt = edge
        adjacency[src].add(tgt)
        adjacency[tgt].add(src)

    for sample_idx in range(samples_num):
        # Генерация параметров для узлов
        sample_nodes = []
        node_params_sample = {}  # для удобного доступа по id
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

        # Для всех ребер генерируем параметры и сохраняем их в словаре
        sample_edge_params = {}
        for edge in edges_fixed:
            edge_key = tuple(sorted(edge))
            edge_params = generate_edge_parameters(edge_attr_dict)
            sample_edge_params[edge_key] = edge_params

        # Генерация параметров для ребер и вычисление целевых значений
        sample_edges = []
        for edge in edges_fixed:
            src, tgt = edge
            edge_key = tuple(sorted(edge))
            edge_params = sample_edge_params[edge_key]
            targets = edge_target_fn(src, tgt, node_params_sample, adjacency, sample_edge_params)
            edge_sample = {'source': src, 'target': tgt}
            edge_sample.update(edge_params)
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
        
    return list(targets.keys())

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

            # Разделяем параметры ребер на признаки и целевые значения
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
    node_num = 100              # количество узлов
    coord_range = (0, 100)     # диапазон координат (x и y)
    mode = 'tree'              # режим формирования графа: 'tree' или 'random'
    edge_num = node_num * 2    # число ребер для режима 'random' (игнорируется для tree)
    min_edge_angle = math.radians(30)  # 30 градусов в радианах
    k_neighbors = 5

    # Параметры для генерации атрибутов
    node_attr_dict = {
        'attr': (0.1, 1.0)       # для узлов генерируется параметр 'attr' в диапазоне [0, 10]
    }
    edge_attr_dict = {
        'weight': (0.1, 1.0)      # для ребер генерируется параметр 'weight' в диапазоне [1, 5]
    }

    samples_num = 2           # число сэмплов

    # Параметры для вычисления целевых значений ребра
    # target_mode:
    # k_sum – сумма атрибутов в k-hop окрестности (без затухания),
    # k_sum_att – с затуханием,
    # k_sum_att_w – с затуханием, но при накоплении каждый вклад домножается на вес ребра, которое соединяет текущую вершину с соседом.
    target_modes = ['k_sum', 'sum_prod']
    # target_mode = 'k_sum_att'
    # target_mode = 'k_sum_att_w'
    # target_mode = 'sum_prod'

    k_hop = 5         # количество шагов для окрестности
    att_coef = 0.5

    def edge_target_fn(src, tgt, node_params, adjacency, sample_edge_params=None):
        # Внутренняя функция для обхода k-hop окрестности.
        # Для режима 4 используется параметр use_edge_weight=True, при этом для каждого перехода
        # вес ребра извлекается из sample_edge_params по ключу (min(вершина, сосед), max(вершина, сосед)).
        def get_k_hop_sum(node_id, k, attenuation=False, coef=0.5, use_edge_weight=False, edge_weight_dict=None):
            # visited хранит для каждой вершины кортеж (расстояние, вес ребра, с которым она была достигнута)
            visited = {node_id: (0, None)}

            queue = [node_id]
            total = 0.0
            while queue:
                current = queue.pop(0)
                d, _ = visited[current]
                if d >= k:
                    continue
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        w = 1.0
                        if use_edge_weight and edge_weight_dict is not None:
                            edge_key = tuple(sorted((current, neighbor)))
                            if edge_key in edge_weight_dict:
                                w = edge_weight_dict[edge_key]['weight']
                        visited[neighbor] = (d + 1, w)
                        factor = (coef ** d) if attenuation else 1.0
                        total += node_params[neighbor]['attr'] * factor * w
                        queue.append(neighbor)
            return total

        ret_dict = dict()
        if 'k_sum' in target_modes:
            for k in range(k_hop+1):
                sum_src = get_k_hop_sum(src, k, attenuation=False)
                sum_tgt = get_k_hop_sum(tgt, k, attenuation=False)
                target_value = sum_src + sum_tgt
                ret_dict[f'{k}_sum'] =  target_value
        if 'k_sum_att' in target_modes:
            for k in range(k_hop+1):
                sum_src = get_k_hop_sum(src, k, attenuation=True, coef=att_coef)
                sum_tgt = get_k_hop_sum(tgt, k, attenuation=True, coef=att_coef)
                target_value = sum_src + sum_tgt
                ret_dict[f'{k}_att_sum'] =  target_value
        if 'k_sum_att_w' in target_modes:
            for k in range(k_hop+1):
                sum_src = get_k_hop_sum(src, k, attenuation=True, coef=att_coef, use_edge_weight=True, edge_weight_dict=sample_edge_params)
                sum_tgt = get_k_hop_sum(tgt, k, attenuation=True, coef=att_coef, use_edge_weight=True, edge_weight_dict=sample_edge_params)
                target_value = sum_src + sum_tgt
                ret_dict[f'{k}_att_w_sum'] =  target_value
        if 'sum_prod' in target_modes:
            a = node_params[src]['attr']
            b = node_params[tgt]['attr']
            ret_dict['sum'] = a + b
            ret_dict['prod'] = a * b
        return ret_dict

    node_attr_cols = ['x', 'y', 'attr']
    edge_attr_cols = ['weight']

    # Генерация фиксированной структуры графа
    nodes_fixed = generate_nodes(node_num, coord_range, min_distance=5)
    if mode == 'tree':
        edges_fixed = generate_tree_edges(nodes_fixed)
    elif mode == 'random':
        edges_fixed = generate_connected_edges(nodes_fixed, edge_num, min_edge_angle, k_neighbors=k_neighbors)
    else:
        raise ValueError('Некорректный режим. Выберите "tree" или "random".')

    edges_fixed = sort_edges(edges_fixed)

    print('Структура графа сгенерирована:')
    print('Количество узлов:', len(nodes_fixed))
    print('Количество ребер:', len(edges_fixed))

    dataset_dir = f'datasets/{'__'.join(target_modes)}__{samples_num}s'    # папка для сохранения CSV файлов

    # Генерация сэмплов (общая структура, но у каждого свои параметры)
    edge_target_cols = generate_samples(samples_num, nodes_fixed, edges_fixed, node_attr_dict, edge_attr_dict, edge_target_fn, dataset_dir)
    print(f'edge_target_cols: {edge_target_cols}')
    
    # Создание датасета PyTorch Geometric
    dataset = MyGraphDataset(root=dataset_dir,
                             node_attr_cols=node_attr_cols,
                             edge_attr_cols=edge_attr_cols,
                             edge_target_cols=edge_target_cols)
    print('Количество графов в датасете:', len(dataset))
    print(dataset[0])


if __name__ == '__main__':
    main()
