import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn


class DynamicGCN5_depr(nn.Module):
    def __init__(self,
                 in_channels,         # размер исходных признаков узла
                 in_edge_channels,   # размер признаков ребра (например, 1)
                 hidden_dims=[128, 64, 32, 16],  # список размерностей для каждого слоя
                 edge_mlp_hidden=16,        # размер скрытого слоя для MLP ребра
                 out_channels=1,            # размер выхода (скалярное предсказание)
                 aggr='mean',               # функция агрегации для NNConv
                 edge_nn_hidden=128,        # размер скрытого слоя внутри edge-нейронной сети
                 edge_nn_activation=nn.ReLU,  # функция активации внутри edge-нейронной сети
                 use_skip=True):            # использовать ли skip-соединения (остаточные связи)
        super(DynamicGCN5_depr, self).__init__()

        self.num_layers = len(hidden_dims)
        self.use_skip = use_skip

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
            self.convs.append(NNConv(out_dim, out_dim, nn=edge_nn, aggr=aggr))

        # МЛП для формирования представления ребра.
        # На вход подаётся конкатенация эмбеддингов двух конечных узлов (в нашем случае мы складываем их)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dims[-1], edge_mlp_hidden),
            nn.ReLU(),
            nn.Linear(edge_mlp_hidden, out_channels)
        )

    def forward(self, data):
        x, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index

        # Проходим по всем слоям: проекция, NNConv, активация и skip-соединение
        for proj, conv in zip(self.input_projs, self.convs):
            x_proj = proj(x)
            x_conv = F.relu(conv(x_proj, edge_index, edge_attr))
            if self.use_skip:
                x = x_conv + x_proj
            else:
                x = x_conv

        # Для каждого ребра агрегируем информацию двух конечных узлов
        # Здесь можно экспериментировать: например, можно использовать конкатенацию, разность или их комбинацию.
        row, col = edge_index
        edge_repr = x[row] + x[col]  # простой вариант: сумма эмбеддингов обоих узлов
        edge_out = self.edge_mlp(edge_repr)
        return edge_out.squeeze(-1)
