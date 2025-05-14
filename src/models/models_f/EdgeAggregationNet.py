import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_scatter import scatter


class EdgeAttentionAggregator(nn.Module):
    """
    Модуль обновления признаков ребер с использованием механизма внимания.
    Для каждого ребра вычисляются веса, на основе которых происходит
    агрегирование соседних сообщений, и затем обновляется представление ребра.
    """

    def __init__(self, in_channels, hidden_channels):
        super(EdgeAttentionAggregator, self).__init__()
        # Линейный слой для вычисления скалярного веса (attention score)
        self.att_fc = nn.Linear(in_channels, 1)
        # MLP для обновления признаков ребра
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, in_channels)
        )

    def forward(self, edge_features, edge_index, num_nodes):
        src, _ = edge_index
        # Вычисляем attention score для каждого ребра
        att_scores = torch.sigmoid(self.att_fc(edge_features))
        # Взвешиваем признаки ребра
        weighted_edges = edge_features * att_scores
        # Агрегируем сообщения для каждого узла (например, суммируя взвешенные признаки ребер)
        aggregated = scatter(weighted_edges, src, dim=0, dim_size=num_nodes, reduce='sum')
        # Для каждого ребра получаем сообщение от исходящего узла
        messages = aggregated[src]
        # Обновляем признаки ребра с остаточным соединением
        updated_edge = self.mlp(edge_features + messages)
        return updated_edge


class EdgeAggregationNet(nn.Module):
    """
    Архитектура, комбинирующая узловой энкодер и модуль агрегации ребер.
    """

    def __init__(self, in_channels, hidden_channels, out_channels,
                 conv_channels, mp_edge_steps=2, dropout_rate=0.1, use_jump=True):
        super(EdgeAggregationNet, self).__init__()
        self.dropout_rate = dropout_rate
        self.use_jump = use_jump

        # Узловой энкодер на основе GATv2
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        heads = 1
        for idx, out_c in enumerate(conv_channels):
            in_c = in_channels if idx == 0 else conv_channels[idx-1] * heads
            conv_layer = gnn.GATv2Conv(in_c, out_c, heads=heads, add_self_loops=True)
            self.convs.append(conv_layer)
            self.norms.append(gnn.norm.BatchNorm(out_c * heads))
        if self.use_jump:
            self.jump = gnn.JumpingKnowledge(mode='cat')
            node_out_dim = sum([c * heads for c in conv_channels])
        else:
            node_out_dim = conv_channels[-1] * heads

        # Инициализация признаков ребер через конкатенацию представлений начального и конечного узла
        self.edge_init = nn.Linear(2 * node_out_dim, hidden_channels)

        # Модули обновления признаков ребер с вниманием (итерационный этап)
        self.mp_edge_steps = mp_edge_steps
        self.edge_mp_layers = nn.ModuleList([
            EdgeAttentionAggregator(hidden_channels, hidden_channels) for _ in range(mp_edge_steps)
        ])

        # Финальный регрессионный блок для предсказания атрибутов ребер
        self.edge_regressor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        src, dst = edge_index

        # Узловой энкодер: применение последовательности GATv2 слоёв с нормализацией и активацией
        x_all = []
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = torch.relu(x)
            x_all.append(x)
        if self.use_jump:
            x_node = self.jump(x_all)
        else:
            x_node = x_all[-1]

        # Формирование начальных признаков ребер путём конкатенации представлений исходного и конечного узлов
        edge_features = torch.cat([x_node[src], x_node[dst]], dim=-1)
        edge_features = self.edge_init(edge_features)

        # Итеративное обновление признаков ребер через модуль внимания
        num_nodes = x_node.size(0)
        for layer in self.edge_mp_layers:
            edge_features = layer(edge_features, edge_index, num_nodes)

        # Предсказание итогового значения для ребра
        out = self.edge_regressor(edge_features)
        return out
