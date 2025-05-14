import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch


class MLPEdgeAttributePredictor(nn.Module):
    def __init__(self, node_in_channels, fc_channels, edge_out_channels, num_nodes, num_edges):
        super(MLPEdgeAttributePredictor, self).__init__()
        self.num_nodes = num_nodes
        self.num_edges = num_edges

        # Начальный размер входного вектора: число узлов * размер признака узла
        input_dim = num_nodes * node_in_channels

        # Построение последовательности слоёв MLP согласно списку fc_channels
        layers = []
        for hidden_dim in fc_channels:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LeakyReLU(negative_slope=1))
            input_dim = hidden_dim

        # Финальный линейный слой, который выдаёт вектор нужного размера: num_edges * edge_out_channels
        layers.append(nn.Linear(input_dim, num_edges * edge_out_channels))
        self.mlp = nn.Sequential(*layers)

    def forward(self, data):
        batch_size = data.num_graphs

        # Преобразуем данные в плотный батч:
        # x_dense имеет размер [batch_size, num_nodes, node_in_channels]
        x_dense, mask = to_dense_batch(data.x, data.batch)

        # Выравниваем признаки узлов для каждого графа в вектор размерности [num_nodes * node_in_channels]
        x_flat = x_dense.view(batch_size, -1)

        # Передаём через MLP
        out = self.mlp(x_flat)  # [batch_size, num_edges * edge_out_channels]

        # Приводим к виду [batch_size * num_edges, edge_out_channels]
        out = out.view(batch_size * self.num_edges, -1)
        return out
