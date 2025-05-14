import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn


class NodeEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, heads=1):
        super(NodeEncoder, self).__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels * heads
            conv = gnn.GATv2Conv(in_dim, hidden_channels, heads=heads, add_self_loops=True, residual=True)
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_channels * heads))

    def forward(self, x, edge_index):
        x_all = []
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x_all.append(x)
        # Можно использовать JumpingKnowledge, если требуется агрегация нескольких слоёв
        return x_all[-1]

# Пример модифицированного EdgePredictor для работы с объединёнными признаками


class EdgePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_channels, use_attention=False):
        super(EdgePredictor, self).__init__()
        self.use_attention = use_attention
        if use_attention:
            self.att_fc = nn.Linear(input_dim, 1)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_channels)
        )

    def forward_with_features(self, edge_feat):
        if self.use_attention:
            weights = torch.sigmoid(self.att_fc(edge_feat))
            edge_feat = edge_feat * weights
        return self.mlp(edge_feat)


class GraphEdgeRegressor(nn.Module):
    def __init__(self, in_channels, node_hidden_dim, num_node_layers, edge_hidden_dim, out_channels, heads=1, use_attention=False):
        super(GraphEdgeRegressor, self).__init__()
        self.node_encoder = NodeEncoder(in_channels, node_hidden_dim, num_node_layers, heads=heads)
        # Увеличиваем размер входного вектора для EdgePredictor:
        # локальное представление ребра имеет размер 2 * (node_hidden_dim * heads),
        # и глобальный контекст имеет размер node_hidden_dim * heads.
        self.edge_predictor = EdgePredictor(2 * node_hidden_dim * heads + node_hidden_dim * heads,
                                            edge_hidden_dim, out_channels, use_attention=use_attention)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # Получаем представления узлов
        node_repr = self.node_encoder(x, edge_index)

        # Вычисляем глобальный контекст для каждого графа в батче
        global_context = gnn.global_add_pool(node_repr, batch)

        # Привязываем глобальный контекст к ребрам.
        # Предполагаем, что для ребра оба узла принадлежат одному графу, берем batch из исходного узла
        edge_global_context = global_context[batch[edge_index[0]]]

        # Формируем локальное представление ребра как конкатенацию представлений исходного и целевого узлов
        src, dst = edge_index
        edge_local = torch.cat([node_repr[src], node_repr[dst]], dim=-1)

        # Объединяем локальные признаки с глобальным контекстом
        edge_feat = torch.cat([edge_local, edge_global_context], dim=-1)

        # Передаём объединённые признаки в предсказатель ребер
        edge_out = self.edge_predictor.forward_with_features(edge_feat)
        return edge_out
