import copy
import numpy as np
import torch_geometric
import matplotlib.pyplot as plt
import networkx as nx


def draw_data(data,
              pos_idxs,
              node_size_idx=None,
              node_size_label=None,
              node_color_idx=None,
              node_color_label=None,
              edge_size_idx=None,
              edge_size_label=None,
              edge_color_idx=None,
              edge_color_label=None,
              edge_data_from_label=False,
              figsize=(16, 8),
              additional_node_label_idx=None,
              font_size=10,
              no_draw=False,
              arrows=True,
              arrowstyle='-|>',
              alt_pos=False
              ):

    log_str_list = list()

    # colors
    node_cmap = plt.cm.coolwarm
    edge_cmap = plt.cm.coolwarm

    # # Шаг 1: Обеспечиваем, чтобы первый элемент в каждом столбце был меньше второго
    # data.edge_index = torch.where(data.edge_index[0] > data.edge_index[1], data.edge_index.flip(0), data.edge_index)

    # # Шаг 2: Получаем индексы для сортировки столбцов по первой строке, а затем по второй
    # sort_indices = (data.edge_index[0] * data.edge_index.shape[1] + data.edge_index[1]).argsort()

    # data.edge_index = data.edge_index[:, sort_indices]
    # data.edge_label = data.edge_label[sort_indices]
    # data.edge_attr = data.edge_attr[sort_indices]
    # data.edge_moded = data.edge_moded[sort_indices]

    # data.edge_index = data.edge_index[:,::2]
    # data.edge_label = data.edge_label[::2]
    # data.edge_attr = data.edge_attr[::2]
    # data.edge_moded = data.edge_moded[::2]

    for i in range(data.edge_index.shape[1]):
        for j in range(i+1, data.edge_index.shape[1]):
            if (data.edge_index[:, i][0] == data.edge_index[:, j][0] and data.edge_index[:, i][1] == data.edge_index[:, j][1]) or (
                    data.edge_index[:, i][0] == data.edge_index[:, j][1] and data.edge_index[:, i][1] == data.edge_index[:, j][0]):
                data.edge_attr[j] = data.edge_attr[i]
                data.edge_label[j] = data.edge_label[i]

    if edge_data_from_label:
        edge_data = data.edge_label
    else:
        edge_data = data.edge_attr

    # values
    pos = data.x[..., :pos_idxs].cpu().numpy()
    edge_index = data.edge_index[..., :].squeeze().cpu().numpy()

    if node_color_idx is not None:
        node_color = data.x[..., pos_idxs +
                             node_color_idx:pos_idxs+node_color_idx+1].cpu().numpy()
    else:
        node_color = [0] * data.x.shape[0]
    if node_size_idx is not None:
        node_size = data.x[..., pos_idxs+node_size_idx:pos_idxs +
                            node_size_idx+1].cpu().numpy() * 100 + 100
    else:
        node_size = [100] * data.x.shape[0]

    if edge_color_idx is not None and edge_color_idx != -1:
        edge_color = edge_data[...,
                               edge_color_idx:edge_color_idx+1].squeeze().cpu().numpy()
        # edge_color_zs = data.edge_attr[..., -2+edge_color_idx:-2+edge_color_idx+1].squeeze().cpu().numpy()
        # edge_color_diff = edge_color - edge_color_zs

    else:
        edge_color = [1e-1] * edge_index.shape[1]
        # edge_color_diff = [-1] * len(edge_color)

    edge_color = [edge_cmap(v) for v in edge_color]

    if edge_size_idx is not None:
        edge_size = edge_data[..., edge_size_idx:edge_size_idx +
                              1].squeeze().cpu().numpy() * 1 + 1
        # edge_size_zs = data.edge_attr[..., -1+edge_size_idx:-1+edge_size_idx+1].squeeze().cpu().numpy()
        # edge_size_diff = edge_size - edge_size_zs
    else:
        edge_size = [2] * edge_index.shape[1]
        # edge_size_diff = [-1] * len(edge_color)

    edge_moded = data.edge_moded[..., :].squeeze().cpu().numpy()

    # mark modded edge
    moded_idx_list = np.where(edge_moded != 0)[0].tolist()
    moded_idx_list.sort()
    pred_moded_idx_list = list()
    # if 'edge_label_pred' in data.keys():
    #     pred_moded_idx_list = np.where(data.edge_label_pred != 0)[0].tolist()
    #     pred_moded_idx_list.sort()
    #     for moded_idx in pred_moded_idx_list:
    #         moded_in_out = edge_index[..., moded_idx].squeeze()
    #         log_str_list.append(f'Pred moded: {moded_idx}, {moded_in_out}')

    for moded_idx in moded_idx_list:
        moded_in_out = edge_index[..., moded_idx].squeeze()
        # if edge_moded[moded_idx] == 1:
        #     color = 'orange'
        # elif edge_moded[moded_idx] == 2:
        #     color = 'red'
        # else:
        #     color = 'grey'

        if edge_moded[moded_idx] > 1.05:
            color = 'red'
        elif edge_moded[moded_idx] > 1:
            color = 'orange'
        else:
            color = 'grey'
        
        log_str_list.append(f'moded: edge {moded_idx}, type = {
                            edge_moded[moded_idx]}, nodes {moded_in_out}')

        # if moded_idx in pred_moded_idx_list:
        #     color = 'green'
        edge_color[moded_idx] = color

    # del modded edge from colormap
    if edge_color_idx is not None:
        edge_color_cmap = copy.copy(edge_color)
        moded_idx_list = list(set(moded_idx_list + pred_moded_idx_list))
        moded_idx_list.sort()
        moded_idx_list.reverse()
        for moded_idx in moded_idx_list:
            del edge_color_cmap[moded_idx]

    if not no_draw:
        plt.ioff()
        fig, ax = plt.subplots(figsize=figsize)  # Добавляем оси для графа
        # data = torch_geometric.data.Data(
        #     x=data.x, edge_index=data.edge_index, num_nodes=data.x.shape[0])

        g = torch_geometric.utils.to_networkx(data, to_undirected=True)

        pos = {i: pos[i] for i in range(pos.shape[0])}
        # fixed = [i for i in range(data.x.shape[0]) if data.x[i, -5] == 0 and not(130 < i < 140) ]
        # print(fixed)
        # pos = nx.spring_layout(g, pos=pos, fixed=fixed, iterations=100, threshold=1e-4, seed=42, k=20)
        if alt_pos:
            pos = nx.arf_layout(g, pos=pos, a=1.1, etol=1e-1, max_iter=50, dt=1e-3)

        # nx.draw(g,
        #         pos,
        #         with_labels=True,
        #         node_color=node_color,  # Цвет узлов
        #         cmap=node_cmap,
        #         node_size=node_size,    # Размер узлов
        #         font_size=font_size//2,
        #         font_color='black',
        #         font_weight='normal',
        #         ax=ax,
        #         )
        # print(node_color)
        nodes = nx.draw_networkx_nodes(g,
                                       pos=pos,
                                       node_color=node_color,  # Цвет узлов
                                       cmap=node_cmap,
                                       node_size=node_size,    # Размер узлов
                                       ax=ax,
                                       )
        nodes.set_zorder(-100)

        nodes_labels = nx.draw_networkx_labels(g,
                                               pos=pos,
                                               labels={i: f'{i}|{node_color[i][0]:.1e}' for i in range(
                                                   node_color.shape[0])},
                                               font_size=font_size//1.5,
                                               font_color='black',
                                               font_weight='normal',
                                               ax=ax,
                                               )
        # for nl in nodes_labels:
        #     nl.set_zorder(1)

        edge_labels_list = list()
        for i in range(edge_index.shape[1]):
            edge_str_list = list()
            if edge_color_idx is not None:
                # if i in moded_idx_list:
                edge_color_value = edge_data[i, edge_color_idx]
                edge_str_list.append(f'{edge_color_label}={
                    edge_color_value:.3f}')
            if edge_size_idx is not None:
                edge_size_value = edge_data[i, edge_size_idx]
                edge_str_list.append(f'{edge_size_label}={
                                     edge_size_value:.3f}')
            if 'edge_label_pred' in data.keys():
                # if i in moded_idx_list:
                pred_values = '|'.join([f'{v:.3f}' for v in data.edge_label_pred[i]])
                edge_str_list.append(f'pr={pred_values}')
            # if i in moded_idx_list:
            #     edge_str_list.append(f'moded: {int(edge_moded[i])}')
            # if i in pred_moded_idx_list:
            #     edge_str_list.append(f'pred_moded: {int(y_pred[i])}')
            if len(edge_str_list) > 0:
                if len(edge_str_list) % 2 == 1:
                    edge_str_list.append('')

            edge_str = '\n'.join(edge_str_list)
            edge_labels = {(edge_index[0, i], edge_index[1, i]): edge_str}

            font_color = edge_color[i]
            font_color = font_color if type(font_color) == str else 'black'
            edgelist = (edge_index[0, i].item(), edge_index[1, i].item())
            edge_labels_list.append(
                [edge_labels, edgelist, font_color, edge_color[i], edge_size[i]])

        for edge_labels, edgelist, font_color, edge_color_cur, edge_size_cur in edge_labels_list:
            # Рисуем ребра с указанным цветом и размером
            edges = nx.draw_networkx_edges(g,
                                           pos=pos,
                                           edgelist=[edgelist],
                                           width=edge_size_cur,
                                           edge_color=edge_color_cur,
                                           # edge_vmin=0.25, edge_vmax=2,
                                           ax=ax,
                                           alpha=1,
                                           arrows=arrows,
                                           arrowsize=edge_size_cur*3,
                                           arrowstyle=arrowstyle,
                                           #    zorder=2,
                                           )
            for e in edges:
                e.set_zorder(3)

        for edge_labels, edgelist, font_color, edge_color_cur, edge_size_cur in edge_labels_list:
            # Draw edge labels
            edges_labels = nx.draw_networkx_edge_labels(g,
                                                        pos=pos,
                                                        edge_labels=edge_labels,
                                                        font_size=font_size//1.5,
                                                        #  font_weight='bold',
                                                        font_color=font_color,
                                                        ax=ax,
                                                        alpha=1,
                                                        #  zorder=3,
                                                        )
            # for el in edges_labels:
            #     print(el)
            #     el.set_zorder(3)

        # Add additional node labels if specified
        if additional_node_label_idx is not None:
            additional_labels = {i: f"{data.x[i, additional_node_label_idx].item():.2f}" for i in range(data.num_nodes)}
            nx.draw_networkx_labels(g, pos, labels=additional_labels, font_size=8, font_color='blue', ax=ax)

        sm_nodes = plt.cm.ScalarMappable(cmap=node_cmap)
        sm_nodes.set_array(node_color)  # Добавляем значения для узлов
        # Указываем ось для цветовой шкалы
        fig.colorbar(sm_nodes, ax=ax, label=node_color_label)

        if edge_color_idx:
            sm_edges = plt.cm.ScalarMappable(cmap=edge_cmap)
            sm_edges.set_array(edge_color_cmap)  # Добавляем значения для ребер
            # Указываем ось для цветовой шкалы
            fig.colorbar(sm_edges, ax=ax, label=edge_color_label)

    return moded_idx_list, fig, ax, log_str_list
