import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn


class ResidualGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualGCNLayer, self).__init__()
        self.conv = gnn.SAGEConv(in_channels, out_channels)
        self.bn = gnn.BatchNorm(out_channels)
        if in_channels != out_channels:
            self.res_linear = nn.Linear(in_channels, out_channels)
        else:
            self.res_linear = None

    def forward(self, x, edge_index):
        out = self.conv(x, edge_index)
        out = self.bn(out)
        out = F.relu(out)
        if self.res_linear is not None:
            x = self.res_linear(x)
        return out + x  # residual connection


class EdgePredictorResGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, mlp_hidden):
        super(EdgePredictorResGCN, self).__init__()
        self.layers = nn.ModuleList()
        # Первый слой
        self.layers.append(ResidualGCNLayer(in_channels, hidden_channels))
        # Остальные слои
        for _ in range(num_layers - 1):
            self.layers.append(ResidualGCNLayer(hidden_channels, hidden_channels))

        # Более сложный MLP для ребер
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1)  # предсказание скалярного значения
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for layer in self.layers:
            x = layer(x, edge_index)

        # Для каждого ребра конкатенация эмбеддингов источника и целевого узла
        src, tgt = edge_index
        edge_repr = torch.cat([x[src], x[tgt]], dim=1)
        edge_pred = self.edge_mlp(edge_repr)
        return edge_pred
