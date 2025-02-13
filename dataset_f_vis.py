import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import torch
from dataset_f import MyGraphDataset


def visualize_data_sample(data, node_pos,
                          node_attr_label_list, node_attr_color,
                          edge_attr_label_list, edge_attr_color,
                          figsize=(10, 8)):
    '''
    Визуализирует граф, представленный объектом Data из PyTorch Geometric.

    Аргументы:
      data: объект Data, содержащий:
            - data.x: тензор узлов [num_nodes, num_features]. Первые 2 столбца – координаты (x, y).
            - data.edge_index: тензор [2, num_edges] с индексами ребер.
            - data.edge_attr: тензор [num_edges, num_edge_features] с признаками ребер.
      node_attr_label_list: список имен узловых признаков для подписи (например, ['attr']).
      node_attr_color: имя узлового признака для цветовой градации (например, 'attr').
      edge_attr_label_list: список имен реберных признаков для подписи (например, ['weight', 'attr_sum', 'attr_prod']).
      edge_attr_color: имя реберного признака для цветовой градации (например, 'weight').
    '''

    data = data.clone()
    data.edge_attr = torch.concat([data.edge_attr, data.edge_target], axis=1)
    data.edge_attr_cols = data.edge_attr_cols + data.edge_target_cols
    print(data.edge_attr_cols)
    # Создаем граф NetworkX
    G = nx.Graph()

    num_nodes = data.x.size(0)
    # Добавляем узлы: позиция определяется первыми двумя столбцами data.x
    for i in range(num_nodes):
        pos_x = data.x[i, node_pos.index('x')].item()
        pos_y = data.x[i, node_pos.index('y')].item()
        node_attrs = {}
        # Для подписей запишем те признаки, которые указаны в node_attr_label_list
        for key in node_attr_label_list:
            if key in data.x_cols:
                value = data.x[i, data.x_cols.index(key)].item()
                node_attrs[key] = value
        G.add_node(i, pos=(pos_x, pos_y), **node_attrs)

    # Список для цветовой градации узлов: значение берется из признака node_attr_color
    node_colors = []
    for i in range(num_nodes):
        if node_attr_color in data.x_cols:
            node_colors.append(data.x[i, data.x_cols.index(node_attr_color)].item())
        else:
            node_colors.append(0)

    # Формируем подписи для узлов
    node_labels = {}
    for i in range(num_nodes):
        label_lines = []
        for key in node_attr_label_list:
            if key in data.x_cols:
                value = data.x[i, data.x_cols.index(key)].item()
                label_lines.append(f'{key}: {value:.2f}')
        node_labels[i] = f'{i}\n' + '\n'.join(label_lines)

    # Добавляем ребра: используем data.edge_index и данные из data.edge_attr
    num_edges = data.edge_index.size(1)
    for j in range(num_edges):
        u = int(data.edge_index[0, j].item())
        v = int(data.edge_index[1, j].item())
        edge_attrs = {}
        for key in edge_attr_label_list:
            if key in data.edge_attr_cols:
                value = data.edge_attr[j, data.edge_attr_cols.index(key)].item()
                edge_attrs[key] = value
        G.add_edge(u, v, **edge_attrs)

    # Список для цветовой градации ребер
    edge_colors = []
    for j in range(num_edges):
        if edge_attr_color in data.edge_attr_cols:
            edge_colors.append(data.edge_attr[j, data.edge_attr_cols.index(edge_attr_color)].item())
        else:
            edge_colors.append(0)

    # Позиции узлов (для отрисовки)
    pos = {i: (data.x[i, 0].item(), data.x[i, 1].item()) for i in range(num_nodes)}

    # Формируем подписи для ребер
    edge_labels = {}
    for j in range(num_edges):
        u = int(data.edge_index[0, j].item())
        v = int(data.edge_index[1, j].item())
        label_lines = []
        for key in edge_attr_label_list:
            if key in data.edge_attr_cols:
                value = data.edge_attr[j, data.edge_attr_cols.index(key)].item()
                label_lines.append(f'{key}: {value:.2f}')
        edge_labels[(u, v)] = '\n'.join(label_lines)

    # Настраиваем colormap для узлов
    cmap_nodes = cm.gist_gray
    vmin_nodes, vmax_nodes = min(node_colors), max(node_colors)

    # Настраиваем colormap для ребер
    cmap_edges = cm.plasma
    vmin_edges, vmax_edges = min(edge_colors), max(edge_colors) if edge_colors else (0, 1)

    # Создаем фигуру и ось
    fig, ax = plt.subplots(figsize=figsize)

    # Отрисовка узлов с учетом цветовой градации
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        cmap=cmap_nodes,
        vmin=vmin_nodes,
        vmax=vmax_nodes,
        node_size=500,
        edgecolors='k',
        ax=ax
    )

    # Отрисовка подписей узлов
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_color='k', font_weight='normal', bbox=dict(alpha=0.5, color='white'), ax=ax)

    # Отрисовка ребер с цветовой градацией
    nx.draw_networkx_edges(
        G, pos,
        edge_color=edge_colors,
        edge_cmap=cmap_edges,
        edge_vmin=vmin_edges,
        edge_vmax=vmax_edges,
        width=2,
        ax=ax
    )

    # Отрисовка подписей ребер
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color='k', ax=ax)

    # Добавляем colorbar для узлов
    sm_nodes = cm.ScalarMappable(cmap=cmap_nodes, norm=mcolors.Normalize(vmin=vmin_nodes, vmax=vmax_nodes))
    sm_nodes.set_array([])
    cbar_nodes = plt.colorbar(sm_nodes, ax=ax, fraction=0.046, pad=0.04)
    cbar_nodes.set_label(node_attr_color, fontsize=10)

    # Добавляем colorbar для ребер
    sm_edges = cm.ScalarMappable(cmap=cmap_edges, norm=mcolors.Normalize(vmin=vmin_edges, vmax=vmax_edges))
    sm_edges.set_array([])
    cbar_edges = plt.colorbar(sm_edges, ax=ax, fraction=0.046, pad=0.04)
    cbar_edges.set_label(edge_attr_color, fontsize=10)

    ax.set_title('Визуализация Data sample', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# Пример использования:
if __name__ == '__main__':
    dataset_dir = 'datasets/sum_prod'
    dataset = MyGraphDataset(root=dataset_dir)
    node_pos_list = ['x', 'y']
    # Задаем, какие признаки выводить в подписи и по какому проводить цветовую градацию
    node_attr_label_list = ['attr']
    node_attr_color = 'attr'
    edge_attr_label_list = ['prod', 'sum']
    edge_attr_color = 'prod'

    for idx in range(2):
        visualize_data_sample(dataset[idx],
                              node_pos_list,
                              node_attr_label_list, node_attr_color,
                              edge_attr_label_list, edge_attr_color,
                              figsize=(16, 9))
