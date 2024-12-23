import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.nn import Sequential, Linear, ReLU


class EdgeRegressionNetWithGAT(torch.nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim, out_dim, heads=8):
        super(EdgeRegressionNetWithGAT, self).__init__()

        # GAT для обработки node features с вниманием
        self.gat1 = GATConv(node_in_dim, hidden_dim, heads=heads, dropout=0.5)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim,
                            heads=heads, dropout=0.5)

        # MLP для обработки edge_attr
        self.edge_mlp = Sequential(
            Linear(edge_in_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU()
        )

        # Слои для объединения информации из узлов и рёбер
        self.edge_predictor = Sequential(
            # Изменено на правильную размерность
            Linear(hidden_dim * (heads + heads + 1), hidden_dim),
            ReLU(),
            Linear(hidden_dim, out_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        # Применение GAT к узлам
        x = F.relu(self.gat1(x, edge_index))
        x = F.relu(self.gat2(x, edge_index))

        # Применение MLP к рёбрам
        edge_emb = self.edge_mlp(edge_attr)

        # Извлечение узлов для каждого ребра
        row, col = edge_index
        # print(x[row].shape, x[col].shape, edge_emb.shape)
        edge_features = torch.cat([x[row], x[col], edge_emb], dim=1)

        # Предсказание для рёбер
        edge_pred = self.edge_predictor(edge_features)

        return edge_pred
