import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Sequential, Linear, ReLU


class EdgeRegressionNet(torch.nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim, out_dim, **kwargs):
        super(EdgeRegressionNet, self).__init__()

        # GCN для обработки node features
        self.node_conv1 = GCNConv(node_in_dim, hidden_dim)
        self.node_conv2 = GCNConv(hidden_dim, hidden_dim)

        # MLP для обработки edge_attr
        self.edge_mlp = Sequential(
            Linear(edge_in_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU()
        )

        # Слои для объединения информации из узлов и рёбер
        self.edge_predictor = Sequential(
            Linear(hidden_dim * 3, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, out_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        # Обработка узлов через GCN
        x = F.relu(self.node_conv1(x, edge_index))
        x = F.relu(self.node_conv2(x, edge_index))

        # Обработка edge_attr через MLP
        edge_emb = self.edge_mlp(edge_attr)

        # Извлечение узлов для каждого ребра
        row, col = edge_index
        edge_features = torch.cat([x[row], x[col], edge_emb], dim=1)

        # Предсказание для рёбер
        edge_pred = self.edge_predictor(edge_features)

        return edge_pred
