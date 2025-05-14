import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_scatter import scatter


class EdgeMessagePassing(nn.Module):
    """
    Модуль для message passing по рёбрам.
    Для каждого ребра (u,v) агрегируются признаки всех рёбер, инцидентных с u и v 
    (исключая само ребро), и затем с помощью MLP происходит обновление признаков ребра.
    """

    def __init__(self, in_channels, out_channels, fc_channels):
        super(EdgeMessagePassing, self).__init__()
        fc_layers = []
        for idx, out_dim in enumerate(fc_channels):
            in_c = in_channels * (1 + 4) if idx == 0 else fc_channels[idx-1]
            fc_layers.append(nn.Linear(in_c, out_dim))
            fc_layers.append(nn.ReLU())
        fc_layers.append(nn.Linear(fc_channels[-1] if len(fc_channels) else in_channels, out_channels))
        self.mlp = nn.Sequential(*fc_layers)

    def forward(self, edge_features, edge_index, num_nodes):
        E, D = edge_features.size()
        node_indices = torch.cat([edge_index[0], edge_index[1]], dim=0)
        edge_features_repeated = torch.cat([edge_features, edge_features], dim=0)
        aggregated_sum = scatter(edge_features_repeated, node_indices, dim=0, dim_size=num_nodes, reduce='sum')
        aggregated_mul = scatter(edge_features_repeated, node_indices, dim=0, dim_size=num_nodes, reduce='mul')

        message_sum_u = aggregated_sum[edge_index[0]] - edge_features
        message_sum_v = aggregated_sum[edge_index[1]] - edge_features
        message_mul_u = aggregated_mul[edge_index[0]]
        message_mul_v = aggregated_mul[edge_index[1]]
        # message = message_u + message_v

        updated_edge = self.mlp(torch.cat([edge_features, message_sum_u, message_sum_v, message_mul_u, message_mul_v], dim=-1))
        return updated_edge


class EdgeMessagePassingV2(nn.Module):
    def __init__(self, in_channels, out_channels, fc_channels, dropout=0.1):
        super(EdgeMessagePassingV2, self).__init__()
        fc_layers = []
        for idx, out_dim in enumerate(fc_channels):
            in_c = in_channels * 3 if idx == 0 else fc_channels[idx-1]
            fc_layers.append(nn.Linear(in_c, out_dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout))
        fc_layers.append(nn.Linear(fc_channels[-1] if fc_channels else in_channels, out_channels))
        self.mlp = nn.Sequential(*fc_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, edge_features, edge_index, num_nodes):
        # Сохраняем исходные признаки ребра для skip-соединения
        residual = edge_features

        # Агрегируем признаки рёбер, инцидентных с узлами
        node_indices = torch.cat([edge_index[0], edge_index[1]], dim=0)
        edge_features_repeated = torch.cat([edge_features, edge_features], dim=0)
        aggregated_sum = scatter(edge_features_repeated, node_indices, dim=0, dim_size=num_nodes, reduce='sum')

        # Получаем сообщения для каждого ребра
        message_sum_u = aggregated_sum[edge_index[0]] - edge_features
        message_sum_v = aggregated_sum[edge_index[1]] - edge_features

        # Можно попробовать добавить взвешивание с вниманием или другие операции
        messages = torch.cat([edge_features, message_sum_u, message_sum_v], dim=-1)
        messages = self.dropout(messages)
        updated_edge = self.mlp(messages)
        # Добавляем skip-соединение
        return updated_edge + residual


class EdgeAttributePredictorConvMP(nn.Module):
    def __init__(self, in_channels, out_channels, fc_channels, mp_fc_channels, conv_channels,
                 prod_mode=False, rescon_mode=False, gat_args={}, num_edge_mp_steps=1, use_jump_knowledge=True):
        """
        in_channels: размерность входных признаков узлов
        out_channels: размерность предсказываемых атрибутов ребра
        conv_channels: список размерностей скрытых представлений для сверточных (GAT) слоёв
        fc_channels: список размерностей полносвязных слоёв
        prod_mode: если True – представление ребра получаем поэлементным умножением признаков конечных узлов,
                   иначе – их конкатенацией
        rescon_mode: если True – на каждом слое FC добавляются остаточные признаки (исходное объединение признаков узлов)
        gat_args: дополнительные аргументы для GATv2Conv (например, число голов 'heads')
        num_edge_mp_steps: число итераций message passing по рёбрам (чтобы собрать информацию от соседей на расстоянии K шагов)
        use_jump_knowledge: если True – применяется слой JumpingKnowledge для агрегации выходов сверточных слоёв,
                            иначе используется только выход последнего слоя
        """
        super(EdgeAttributePredictorConvMP, self).__init__()

        self.act = nn.ReLU()
        self.prod_mode = prod_mode
        self.rescon_mode = rescon_mode
        self.num_edge_mp_steps = num_edge_mp_steps
        self.use_jump_knowledge = use_jump_knowledge
        heads = gat_args.get('heads', 1)
        node_repr_dim = 2 * in_channels

        # Слои message passing по узлам
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for idx, out_c in enumerate(conv_channels):
            in_c = in_channels if idx == 0 else conv_channels[idx-1] * heads
            conv_layer = gnn.GATv2Conv(in_c, out_c, **gat_args)
            self.convs.append(conv_layer)
            self.norms.append(gnn.norm.PairNorm())

        # Определяем размерность выходного представления узлов в зависимости от use_jump_knowledge
        if self.use_jump_knowledge:
            # Агрегируем выходы всех слоёв
            self.jump = gnn.JumpingKnowledge(mode='cat')
            total_conv_dim = sum([c * heads for c in conv_channels])
            if self.prod_mode:
                conv_node_repr_dim = total_conv_dim
            else:
                conv_node_repr_dim = 2 * total_conv_dim
        else:
            # Используем только выход последнего слоя
            last_dim = conv_channels[-1] * heads
            if self.prod_mode:
                conv_node_repr_dim = last_dim
            else:
                conv_node_repr_dim = 2 * last_dim

        # Модуль message passing по рёбрам
        self.edge_mp = EdgeMessagePassingV2(conv_node_repr_dim, conv_node_repr_dim, mp_fc_channels)

        # Полносвязная сеть для предсказания атрибутов ребра
        fc_layers = []
        for idx, out_dim in enumerate(fc_channels):
            in_dim = conv_node_repr_dim if idx == 0 else fc_channels[idx-1]
            if self.rescon_mode:
                in_dim += node_repr_dim
            fc_layers.append(nn.Linear(in_dim, out_dim))
        self.fc_net = nn.Sequential(*fc_layers)

        in_dim = fc_channels[-1]
        if self.rescon_mode:
            in_dim += node_repr_dim
        self.fc_out = nn.Linear(in_dim, out_channels)

    def forward(self, data):
        x, _, edge_index = data.x, data.edge_attr, data.edge_index
        src, dst = edge_index

        # Сохраняем исходное объединение признаков узлов для остаточных связей
        node_repr = torch.cat([x[src], x[dst]], dim=-1)

        # Message passing по узлам
        x_all = []
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.norms[i](x)
            x = self.act(x)
            x_all.append(x)
        # Агрегация выходов сверточных слоёв:
        if self.use_jump_knowledge:
            x_agg = self.jump(x_all)
        else:
            x_agg = x_all[-1]

        # Формирование начального представления ребра
        if self.prod_mode:
            edge_feat = x_agg[src] * x_agg[dst]
        else:
            edge_feat = torch.cat([x_agg[src], x_agg[dst]], dim=-1)

        # Итерационный этап message passing по рёбрам
        num_nodes = x_agg.size(0)
        for _ in range(self.num_edge_mp_steps):
            edge_feat = self.edge_mp(edge_feat, edge_index, num_nodes)

        # Полносвязная сеть для предсказания атрибутов ребра
        edge_features = edge_feat
        for layer in self.fc_net:
            if self.rescon_mode:
                edge_features = torch.cat([edge_features, node_repr], dim=-1)
            edge_features = self.act(layer(edge_features))
        if self.rescon_mode:
            edge_features = torch.cat([edge_features, node_repr], dim=-1)
        edge_features = self.fc_out(edge_features)

        return edge_features
