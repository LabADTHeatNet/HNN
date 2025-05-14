import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn


class EdgeAttributeMessagePassing(gnn.MessagePassing):
    def __init__(self, in_channels, out_channels, fc_channels, aggr=['sum']):
        super(EdgeAttributeMessagePassing, self).__init__(aggr=aggr)  # Используем агрегацию по сумме
        self.fc_net = list()
        for idx in range(len(fc_channels)):
            if idx == 0:
                prev_out_dim = in_channels  # пока нет предыдущего слоя
            else:
                prev_out_dim = fc_channels[idx-1]
            out_c = fc_channels[idx]
            self.fc_net.append(nn.Linear(prev_out_dim, out_c))
        self.fc_net = nn.Sequential(*self.fc_net)
        self.fc_out = nn.Linear(fc_channels[-1], out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.fc_net:
            layer.reset_parameters()
        self.fc_out.reset_parameters()

    def forward(self, x, edge_index):
        # Применяем линейное преобразование к признакам узлов
        for layer in self.fc_net:
            x = torch.relu(layer(x))
        x = self.fc_out(x)

        # Применяем передачу сообщений
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out, x):
        return aggr_out


class EdgeAttributePredictor(nn.Module):
    def __init__(self, in_channels, out_channels, fc_channels, msg_pass_channels, msg_pass_aggr=[]):
        super(EdgeAttributePredictor, self).__init__()

        node_repr_dim = 2 * in_channels  # Умножаем на 2 из-за объединения признаков двух узлов

        msg_pass_out_channels = 4 * in_channels  # Можно поставить любое
        if len(msg_pass_aggr) != 0:
            self.message_passing = EdgeAttributeMessagePassing(in_channels, msg_pass_out_channels, msg_pass_channels, aggr=msg_pass_aggr)
        else:
            self.message_passing = None
        aggr_node_repr_dim = 2 * msg_pass_out_channels * len(msg_pass_aggr)  # Аналогично и умножаем на кол-во функций MessagePassing

        self.fc_net = list()
        for idx in range(len(fc_channels)):
            if idx == 0:
                prev_out_dim = 0  # пока нет предыдущего слоя
            else:
                prev_out_dim = fc_channels[idx-1]
            in_c = prev_out_dim + node_repr_dim + aggr_node_repr_dim  # edge_attr, node_repr и aggr_node_repr
            out_c = fc_channels[idx]
            self.fc_net.append(nn.Linear(in_c, out_c))
        self.fc_net = nn.Sequential(*self.fc_net)

        in_c = fc_channels[-1] + node_repr_dim + aggr_node_repr_dim
        self.fc_out = nn.Linear(in_c, out_channels)

    def forward(self, data):
        x, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index
        src, dst = edge_index

        # Объединяем признаки начального и конечного узлов для каждого ребра
        node_repr = torch.cat([x[src], x[dst]], dim=-1)

        # Применяем слой передачи сообщений
        if self.message_passing is not None:
            x_aggr = self.message_passing(x, edge_index)
            aggr_node_repr = torch.cat([x_aggr[src], x_aggr[dst]], dim=-1)
        else:
            aggr_node_repr = torch.Tensor()

        # Применяем полносвязный слой для предсказания атрибутов рёбер
        edge_attr = torch.Tensor()  # prev_out_dim = 0
        for layer in self.fc_net:
            edge_attr = torch.cat([edge_attr, node_repr, aggr_node_repr], dim=-1)
            edge_attr = torch.relu(layer(edge_attr))
        edge_attr = torch.cat([edge_attr, node_repr, aggr_node_repr], dim=-1)
        edge_attr = self.fc_out(edge_attr)  # без relu

        return edge_attr
