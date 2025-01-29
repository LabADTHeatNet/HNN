import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn

class FlatGCN(torch.nn.Module):
    def __init__(self, in_node_dim, in_edge_dim, out_dim,
                 hidden_dim,
                 node_conv_layer_type='GCNConv',
                 node_conv_layer_num=4,
                 node_conv_heads=1,
                 node_conv_layer_kwargs=dict(),
                 edge_fc_layer_num=8,
                 out_fc_layer_num=4,
                 **kwargs):
        """Архитектура с фиксированным числом слоев GCN и полносвязных слоев.
        
        Args:
            in_node_dim (int): Размерность признаков узлов.
            in_edge_dim (int): Размерность признаков ребер.
            out_dim (int): Размерность выходных меток.
            hidden_dim (int): Размерность скрытых слоев.
            node_conv_layer_type (str): Тип графового слоя (например, GCNConv).
            node_conv_layer_num (int): Количество графовых слоев.
            edge_fc_layer_num (int): Количество FC-слоев для признаков ребер.
            out_fc_layer_num (int): Количество FC-слоев для объединенных признаков.
        """
        super(FlatGCN, self).__init__()

        # Инициализация графовых слоев для обработки узлов
        conv = getattr(importlib.import_module(f'torch_geometric.nn'), node_conv_layer_type)
        self.node_conv_list = nn.ModuleList()
        # Первый слой: входная размерность -> hidden_dim
        self.node_conv_list.append(conv(in_node_dim, hidden_dim, **node_conv_layer_kwargs))
        # Последующие слои: hidden_dim * heads -> hidden_dim
        for i in range(1, node_conv_layer_num):
            self.node_conv_list.append(conv(hidden_dim * node_conv_heads, hidden_dim, **node_conv_layer_kwargs))

        # Полносвязные слои для признаков ребер
        self.edge_fc_list = nn.ModuleList()
        self.edge_fc_list.append(nn.Linear(in_edge_dim, hidden_dim))
        for i in range(1, edge_fc_layer_num):
            self.edge_fc_list.append(nn.Linear(hidden_dim, hidden_dim))

        # Слои для объединения признаков узлов и ребер
        self.out_fc_list = nn.ModuleList()
        # Вход: признаки узлов (src и dst) + признаки ребер
        self.out_fc_list.append(nn.Linear(hidden_dim * (2 * node_conv_heads + 1), hidden_dim))
        for i in range(1, out_fc_layer_num):
            self.out_fc_list.append(nn.Linear(hidden_dim, hidden_dim))
        # Финальный слой для предсказания
        self.final_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        """Прямой проход данных через модель."""
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        src, dst = edge_index  # Индексы исходных и целевых узлов для ребер

        # Обработка узлов через графовые слои
        for node_conv in self.node_conv_list:
            x = F.relu(node_conv(x, edge_index))  # ReLU активация после каждого слоя

        # Обработка признаков ребер через FC
        edge_emb = edge_attr
        for edge_fc in self.edge_fc_list:
            edge_emb = F.relu(edge_fc(edge_emb))

        # Конкатенация признаков узлов и ребер для каждого ребра
        edge_features = torch.cat([x[src], x[dst], edge_emb], dim=1)

        # Прогон через выходные слои
        for out_fc in self.out_fc_list:
            edge_features = F.relu(out_fc(edge_features))
        edge_pred = self.final_layer(edge_features)  # Финальное предсказание

        return edge_pred
