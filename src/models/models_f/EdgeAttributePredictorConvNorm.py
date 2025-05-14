import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn


class EdgeAttributePredictorConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, fc_channels, conv_channels,
                 prod_mode=False, rescon_mode=False, gat_args={}):
        super(EdgeAttributePredictorConvNorm, self).__init__()

        self.act = nn.LeakyReLU(negative_slope=0.1)  # Используем ReLU в качестве функции активации

        self.prod_mode = prod_mode
        self.rescon_mode = rescon_mode
        heads = gat_args.get('heads', 1)
        node_repr_dim = 2 * in_channels  # Умножаем на 2 из-за объединения признаков двух узлов

        # Списки для сверточных слоев и PairNorm слоев
        self.conv_net = nn.ModuleList()
        self.pair_norms = nn.ModuleList()

        for idx in range(len(conv_channels)):
            if idx == 0:
                in_c = in_channels
            else:
                in_c = conv_channels[idx-1] * heads
            out_c = conv_channels[idx]
            conv_layer = gnn.SAGEConv(in_c, out_c, aggr='mean')
            # conv_layer = gnn.GATv2Conv(in_c, out_c, **gat_args)
            self.conv_net.append(conv_layer)
            self.pair_norms.append(gnn.norm.PairNorm())

        # Определяем размер выходного представления после сверточных слоев
        if self.prod_mode:
            conv_node_repr_dim = out_c * heads
        else:
            conv_node_repr_dim = 2 * out_c * heads  # Умножаем на 2 из-за объединения признаков двух узлов

        # Полносвязная сеть (FC)
        fc_layers = []
        for idx in range(len(fc_channels)):
            if idx == 0:
                prev_out_dim = conv_node_repr_dim
            else:
                prev_out_dim = fc_channels[idx-1]
            in_c = prev_out_dim
            if self.rescon_mode:
                in_c += node_repr_dim
            fc_layers.append(nn.Linear(in_c, fc_channels[idx]))
        self.fc_net = nn.Sequential(*fc_layers)

        in_c = fc_channels[-1]
        if self.rescon_mode:
            in_c += node_repr_dim
        self.fc_out = nn.Linear(in_c, out_channels)

    def forward(self, data):
        x, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index
        src, dst = edge_index

        # Объединяем признаки начального и конечного узлов для каждого ребра
        node_repr = torch.cat([x[src], x[dst]], dim=-1)

        # Применяем последовательность сверточных слоев с нормализацией PairNorm
        for i, layer in enumerate(self.conv_net):
            x = layer(x, edge_index)
            x = self.pair_norms[i](x)
            x = self.act(x)

        # Объединяем представления конечных узлов для каждого ребра
        if self.prod_mode:
            conv_node_repr = x[src] * x[dst]
        else:
            conv_node_repr = torch.cat([x[src], x[dst]], dim=-1)

        # Применяем полносвязную сеть для предсказания атрибутов рёбер
        edge_attr = conv_node_repr
        for layer in self.fc_net:
            if self.rescon_mode:
                edge_attr = torch.cat([edge_attr, node_repr], dim=-1)
            edge_attr = self.act(layer(edge_attr))
        if self.rescon_mode:
            edge_attr = torch.cat([edge_attr, node_repr], dim=-1)
        edge_attr = self.fc_out(edge_attr)  # Финальный линейный слой без активации

        return edge_attr
