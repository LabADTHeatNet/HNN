import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn


class EdgeMessagePassingNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, aggregator='mean'):
        """
        Параметры:
         - in_channels: размер входных признаков узла.
         - hidden_channels: размер скрытого представления.
         - out_channels: размер выхода (например, скалярное предсказание для ребра).
         - num_layers: число message passing слоёв.
         - aggregator: тип агрегации в SAGEConv ('mean', 'max', 'lstm' и т.д.).
        """
        super(EdgeMessagePassingNet, self).__init__()
        self.convs = nn.ModuleList()
        # Первый слой преобразует входные признаки в скрытое пространство.
        self.convs.append(gnn.SAGEConv(in_channels, hidden_channels, aggr=aggregator))
        # Остальные слои сохраняют размерность hidden_channels.
        for _ in range(num_layers - 1):
            self.convs.append(gnn.SAGEConv(hidden_channels, hidden_channels, aggr=aggregator))

        # Используем Jumping Knowledge для объединения представлений с разных слоёв.
        self.jump = gnn.JumpingKnowledge(mode='cat')  # Конкатенация выходов всех слоёв.

        # Размер объединённого представления для узла равен hidden_channels * num_layers.
        node_emb_dim = hidden_channels * num_layers

        # МЛП для предсказания атрибута ребра на основе объединённых представлений двух узлов.
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_emb_dim, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # Сохраняем выходы каждого слоя.
        out_list = []
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            out_list.append(x)
        # Объединяем представления из всех слоёв.
        x_cat = self.jump(out_list)

        # Для каждого ребра получаем представления исходного и конечного узлов.
        src, dst = edge_index
        edge_repr = torch.cat([x_cat[src], x_cat[dst]], dim=-1)

        # Предсказываем атрибут ребра.
        edge_pred = self.edge_mlp(edge_repr)
        return edge_pred.squeeze(-1)
