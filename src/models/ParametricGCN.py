import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn

class ParametricGCN(nn.Module):
    def __init__(self, in_node_dim, in_edge_dim, out_dim,
                 node_conv_layer_type='GCNConv',
                 node_conv_layer_kwargs=dict(),
                 node_conv_layer_list=[16, 8],
                 node_conv_heads=1,
                 edge_fc_layer_list=[8],
                 out_fc_layer_list=[8],
                 **kwargs):
        """Гибкая архитектура с настраиваемым числом и размером слоев.
        
        Args:
            in_node_dim (int): Размерность признаков узлов.
            in_edge_dim (int): Размерность признаков ребер.
            out_dim (int): Размерность выходных меток.
            node_conv_layer_type (str): Тип графового слоя (например, GCNConv).
            node_conv_layer_list (list): Список размерностей графовых слоев.
            edge_fc_layer_list (list): Список размерностей FC-слоев для ребер.
            out_fc_layer_list (list): Список размерностей выходных FC-слоев.
        """
        super(ParametricGCN, self).__init__()

        # Графовые слои для узлов с динамической настройкой размеров
        conv = getattr(importlib.import_module(f'torch_geometric.nn'), node_conv_layer_type)

        # Графовые слои для узлов
        self.node_conv_list = nn.ModuleList()
        in_channels = in_node_dim
        for dim in node_conv_layer_list:
            self.node_conv_list.append(conv(in_channels, dim, **node_conv_layer_kwargs))
            in_channels = dim * node_conv_heads  # Обновление входной размерности

        # FC-слои для признаков ребер
        self.edge_fc_list = nn.ModuleList()
        in_channels = in_edge_dim
        for dim in edge_fc_layer_list:
            self.edge_fc_list.append(nn.Linear(in_channels, dim))
            in_channels = dim

        # Слои для объединения признаков
        self.out_fc_list = nn.ModuleList()
        # Вход: признаки узлов (src и dst) + признаки ребер
        in_channels = node_conv_layer_list[-1] * 2 * node_conv_heads + edge_fc_layer_list[-1]
        for dim in out_fc_layer_list:
            self.out_fc_list.append(nn.Linear(in_channels, dim))
            in_channels = dim
        self.final_layer = nn.Linear(in_channels, out_dim)

    def forward(self, data):
        """Прямой проход данных через модель."""
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        src, dst = edge_index

        # Обработка узлов через графовые слои
        for node_conv in self.node_conv_list:
            x = F.relu(node_conv(x, edge_index))

        # Обработка признаков ребер через FC
        edge_emb = edge_attr
        for edge_fc in self.edge_fc_list:
            edge_emb = F.relu(edge_fc(edge_emb))

        # Конкатенация признаков узлов и ребер
        edge_features = torch.cat([x[src], x[dst], edge_emb], dim=1)

        # Прогон через выходные слои
        for out_fc in self.out_fc_list:
            edge_features = F.relu(out_fc(edge_features))
        edge_pred = self.final_layer(edge_features)

        return edge_pred
