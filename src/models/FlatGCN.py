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
        super(FlatGCN, self).__init__()

        # GCN для обработки x
        conv = getattr(importlib.import_module(f'torch_geometric.nn'), node_conv_layer_type)
        self.node_conv_list = nn.ModuleList()
        self.node_conv_list.append(conv(in_node_dim, hidden_dim, **node_conv_layer_kwargs))
        for i in range(1, node_conv_layer_num):
            self.node_conv_list.append(conv(hidden_dim * node_conv_heads, hidden_dim,  **node_conv_layer_kwargs))

        # FC для обработки edge_attr
        self.edge_fc_list = nn.ModuleList()
        self.edge_fc_list.append(nn.Linear(in_edge_dim, hidden_dim))
        for i in range(1, edge_fc_layer_num):
            self.edge_fc_list.append(nn.Linear(hidden_dim, hidden_dim))

        # FC для объединения информации из узлов и рёбер
        self.out_fc_list = nn.ModuleList()
        self.out_fc_list.append(nn.Linear(hidden_dim * (2 * node_conv_heads + 1), hidden_dim))
        for i in range(1, out_fc_layer_num):
            self.out_fc_list.append(nn.Linear(hidden_dim, hidden_dim))
        self.final_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        src, dst = edge_index

        # Обработка узлов через GCN
        for node_conv in self.node_conv_list:
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