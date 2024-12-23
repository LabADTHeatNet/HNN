import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeRegressionNetWithMLP(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim, out_dim):
        super(EdgeRegressionNetWithMLP, self).__init__()

        # Линейные слои для приведения данных из разных источников к общей размерности
        self.node_fc = nn.Linear(node_in_dim, hidden_dim)
        self.edge_fc = nn.Linear(edge_in_dim, hidden_dim)

        # MLP для объединенных признаков
        self.mlp = nn.Sequential(
            # Объединяем преобразованные node_attr и edge_attr
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim*4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            # Выходной слой для предсказания диаметра
            nn.Linear(hidden_dim * 2, out_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        # Преобразуем признаки узлов и ребер в единую размерность
        node_features = F.relu(self.node_fc(x))
        edge_features = F.relu(self.edge_fc(edge_attr))

        # Объединяем преобразованные признаки
        combined_features = torch.cat(
            [node_features[src], node_features[dst], edge_features], dim=1)

        # Пропускаем объединенные данные через MLP
        predictions = self.mlp(combined_features)

        return predictions
