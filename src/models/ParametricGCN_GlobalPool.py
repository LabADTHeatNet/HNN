import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn


class ParametricGCN_GlobalPool(nn.Module):
    def __init__(self, in_node_dim, in_edge_dim, out_dim,
                 node_conv_layer_type='GCNConv',
                 node_conv_layer_kwargs=dict(),
                 node_conv_layer_list=[16, 8],
                 node_conv_heads=1,
                 node_global_pool_type='global_mean_pool',
                 edge_fc_layer_list=[8],
                 out_fc_layer_list=[8],
                 **kwargs):
        super(ParametricGCN_GlobalPool, self).__init__()

        # GCN для обработки x
        conv = getattr(importlib.import_module(f'torch_geometric.nn'), node_conv_layer_type)
        self.global_pool = getattr(importlib.import_module(f'torch_geometric.nn'), node_global_pool_type)
        self.node_conv_list = nn.ModuleList()
        in_channels = in_node_dim * 2
        for dim in node_conv_layer_list:
            self.node_conv_list.append(conv(in_channels, dim, **node_conv_layer_kwargs))
            in_channels = dim * node_conv_heads * 2

        # FC для обработки edge_attr
        self.edge_fc_list = nn.ModuleList()
        in_channels = in_edge_dim
        for dim in edge_fc_layer_list:
            self.edge_fc_list.append(nn.Linear(in_channels, dim))
            in_channels = dim

        # FC для объединения информации из узлов и рёбер
        self.out_fc_list = nn.ModuleList()
        in_channels = node_conv_layer_list[-1] * 2 * node_conv_heads + edge_fc_layer_list[-1]
        for dim in out_fc_layer_list:
            self.out_fc_list.append(nn.Linear(in_channels, dim))
            in_channels = dim
        self.final_layer = nn.Linear(in_channels, out_dim)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        src, dst = edge_index

        # Обработка узлов через GCN
        for node_conv in self.node_conv_list:
            graph_pool = self.global_pool(x, data.batch)[data.batch]
            x = torch.cat([x, graph_pool], dim=1)
            x = F.relu(node_conv(x, edge_index))

        # Обработка edge_attr через FC
        edge_emb = edge_attr
        for edge_fc in self.edge_fc_list:
            edge_emb = F.relu(edge_fc(edge_emb))

        # Извлечение узлов для каждого ребра
        edge_features = torch.cat([x[src], x[dst], edge_emb], dim=1)

        # Предсказание для рёбер
        for out_fc in self.out_fc_list:
            edge_features = F.relu(out_fc(edge_features))
        edge_pred = self.final_layer(edge_features)

        return edge_pred
