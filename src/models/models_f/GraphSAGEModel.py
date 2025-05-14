import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn


class GraphSAGEModel(nn.Module):
    def __init__(self, in_channels, out_channels, fc_channels, conv_channels,
                 prod_mode=False, rescon_mode=False, sage_args={}):
        super(GraphSAGEModel, self).__init__()

        self.act = nn.LeakyReLU(negative_slope=1)

        self.prod_mode = prod_mode
        self.rescon_mode = rescon_mode
        heads = 1
        node_repr_dim = 2 * in_channels  # Умножаем на 2 из-за объединения признаков двух узлов

        self.conv_net = list()
        out_c = in_channels
        for idx in range(len(conv_channels)):
            if idx == 0:
                prev_out_dim = in_channels  # пока нет предыдущего слоя
                in_c = in_channels
            else:
                prev_out_dim = conv_channels[idx-1]
                in_c = prev_out_dim * heads
            out_c = conv_channels[idx]
            self.conv_net.append(gnn.GraphSAGE(in_c, out_c, **sage_args))

        self.conv_net = nn.Sequential(*self.conv_net)

        if self.prod_mode:
            conv_node_repr_dim = 1 * out_c * heads
        else:
            conv_node_repr_dim = 2 * out_c * heads  # Умножаем на 2 из-за объединения признаков двух узлов

        self.fc_net = list()
        for idx in range(len(fc_channels)):
            if idx == 0:
                prev_out_dim = conv_node_repr_dim  # пока нет предыдущего слоя
            else:
                prev_out_dim = fc_channels[idx-1]
            in_c = prev_out_dim
            if self.rescon_mode:
                in_c += node_repr_dim
            out_c = fc_channels[idx]
            self.fc_net.append(nn.Linear(in_c, out_c))
        self.fc_net = nn.Sequential(*self.fc_net)

        in_c = out_c
        if self.rescon_mode:
            in_c += node_repr_dim
        self.fc_out = nn.Linear(in_c, out_channels)

    def forward(self, data):
        x, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index
        src, dst = edge_index

        # Объединяем признаки начального и конечного узлов для каждого ребра
        node_repr = torch.cat([x[src], x[dst]], dim=-1)

        # Применяем слой передачи сообщений
        for layer in self.conv_net:
            x = self.act(layer(x, edge_index))

        if self.prod_mode:
            conv_node_repr = (x[src] * x[dst])
        else:
            conv_node_repr = torch.cat([x[src], x[dst]], dim=-1)

        # Применяем полносвязный слой для предсказания атрибутов рёбер
        edge_attr = conv_node_repr
        for layer in self.fc_net:
            if self.rescon_mode:
                edge_attr = torch.cat([edge_attr, node_repr], dim=-1)
            edge_attr = self.act(layer(edge_attr))
        if self.rescon_mode:
            edge_attr = torch.cat([edge_attr, node_repr], dim=-1)
        edge_attr = self.fc_out(edge_attr)  # без relu

        return edge_attr
