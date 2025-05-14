import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn


class DynamicGCN5(nn.Module):
    def __init__(self,
                 in_channels,         # размер исходных признаков узла
                 in_edge_channels,    # размер признаков ребра (например, 1)
                 hidden_dims=[128, 64, 32, 16],  # список размерностей для каждого слоя
                 edge_mlp_hidden=16,  # размер скрытого слоя для MLP ребра
                 out_channels=1,      # размер выхода (скалярное предсказание)
                 aggr='mean',         # функция агрегации для NNConv
                 edge_nn_hidden=128,  # размер скрытого слоя внутри edge-нейронной сети
                 edge_nn_activation=nn.ReLU,  # функция активации внутри edge-нейронной сети
                 use_skip=True,       # использовать ли skip-соединения внутри слоев
                 raw_skip=False):     # использовать ли skip-соединения для необработанных данных (data.x и data.edge_attr)
        super(DynamicGCN5, self).__init__()

        self.num_layers = len(hidden_dims)
        self.use_skip = use_skip
        self.raw_skip = raw_skip

        # Модули для проекции входных признаков и NNConv слои
        self.input_projs = nn.ModuleList()
        self.convs = nn.ModuleList()

        for i in range(self.num_layers):
            in_dim = in_channels if i == 0 else hidden_dims[i-1]
            out_dim = hidden_dims[i]
            # Линейная проекция признаков узлов в нужное пространство
            self.input_projs.append(nn.Linear(in_dim, out_dim))
            # Формирование edge-нейронной сети для NNConv: она преобразует признаки ребра в матрицу размерности out_dim x out_dim.
            edge_nn = nn.Sequential(
                nn.Linear(in_edge_channels, edge_nn_hidden),
                edge_nn_activation(),
                nn.Linear(edge_nn_hidden, out_dim * out_dim)
            )
            self.convs.append(gnn.NNConv(out_dim, out_dim, nn=edge_nn, aggr=aggr))

        # МЛП для формирования представления ребра.
        # На вход подаётся представление ребра, сформированное как сумма эмбеддингов конечных узлов (с дополнительными raw skip, если включено)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dims[-1], edge_mlp_hidden),
            nn.ReLU(),
            nn.Linear(edge_mlp_hidden, out_channels)
        )

        # Если включена опция raw_skip, создаём проекции для необработанных признаков узлов и ребер
        if self.raw_skip:
            self.raw_node_proj = nn.Linear(in_channels, hidden_dims[-1])
            self.raw_edge_proj = nn.Linear(in_edge_channels, hidden_dims[-1])

    def forward(self, data):
        x, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index

        # Проходим по всем слоям: проекция, NNConv, активация и (опциональное) skip-соединение
        for proj, conv in zip(self.input_projs, self.convs):
            x_proj = proj(x)
            x_conv = F.relu(conv(x_proj, edge_index, edge_attr))
            if self.use_skip:
                x = x_conv + x_proj
            else:
                x = x_conv

        # Формирование представления ребра на основе эмбеддингов конечных узлов
        row, col = edge_index
        edge_repr = x[row] + x[col]  # базовый вариант: сумма эмбеддингов узлов

        # Если включена опция raw_skip, дополнительно добавляем информацию от необработанных данных:
        if self.raw_skip:
            raw_node = self.raw_node_proj(data.x)
            # Для ребра добавляем сумму проекций raw-данных его конечных узлов
            raw_node_repr = raw_node[row] + raw_node[col]
            # Для ребра также учитываем обработанную проекцию исходных признаков ребра
            raw_edge_repr = self.raw_edge_proj(edge_attr)
            edge_repr = edge_repr + raw_node_repr + raw_edge_repr

        edge_out = self.edge_mlp(edge_repr)
        return edge_out
