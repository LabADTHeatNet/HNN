import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn


class GCN5_KHop(nn.Module):
    def __init__(self, num_node_features=1, num_edge_input_features=1, out_channels=1):
        super(GCN5_KHop, self).__init__()

        # Первая проекция узловых признаков
        self.input_proj1 = nn.Linear(num_node_features, 128)
        nn1 = nn.Sequential(
            nn.Linear(num_edge_input_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128 * 128)
        )
        self.conv1 = gnn.NNConv(128, 128, nn=nn1, aggr='mean')

        # Вторая проекция
        self.input_proj2 = nn.Linear(128, 64)
        nn2 = nn.Sequential(
            nn.Linear(num_edge_input_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64 * 64)
        )
        self.conv2 = gnn.NNConv(64, 64, nn=nn2, aggr='mean')

        # Третья проекция
        self.input_proj3 = nn.Linear(64, 32)
        nn3 = nn.Sequential(
            nn.Linear(num_edge_input_features, 128),
            nn.ReLU(),
            nn.Linear(128, 32 * 32)
        )
        self.conv3 = gnn.NNConv(32, 32, nn=nn3, aggr='mean')

        # Четвертая проекция
        self.input_proj4 = nn.Linear(32, 16)
        nn4 = nn.Sequential(
            nn.Linear(num_edge_input_features, 128),
            nn.ReLU(),
            nn.Linear(128, 16 * 16)
        )
        self.conv4 = gnn.NNConv(16, 16, nn=nn4, aggr='mean')

        # Линейные слои для формирования предсказания ребра
        self.edge_lin1 = nn.Linear(16, 16)
        self.edge_lin2 = nn.Linear(16, out_channels)

    def forward(self, data):
        x, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index

        # Первый блок
        x_proj1 = self.input_proj1(x)
        x1 = F.relu(self.conv1(x_proj1, edge_index, edge_attr))
        x = x1 + x_proj1  # остаточное соединение

        # Второй блок
        x_proj2 = self.input_proj2(x)
        x2 = F.relu(self.conv2(x_proj2, edge_index, edge_attr))
        x = x2 + x_proj2

        # Третий блок
        x_proj3 = self.input_proj3(x)
        x3 = F.relu(self.conv3(x_proj3, edge_index, edge_attr))
        x = x3 + x_proj3

        # Четвертый блок
        x_proj4 = self.input_proj4(x)
        x4 = F.relu(self.conv4(x_proj4, edge_index, edge_attr))
        x = x4 + x_proj4

        # Формирование представления ребра как суммы эмбеддингов двух конечных узлов
        row, col = edge_index
        edge_embeddings = x[row] + x[col]
        edge_embeddings = F.relu(self.edge_lin1(edge_embeddings))
        edge_out = self.edge_lin2(edge_embeddings)

        # Так как out_channels=1, возвращаем скаляр для каждого ребра
        return edge_out.squeeze(-1)
