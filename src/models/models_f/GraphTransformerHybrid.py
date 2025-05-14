import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn


class GraphLocalEncoder(nn.Module):
    """
    Локальный энкодер с использованием графовых сверточных слоёв (например, GAT).
    """

    def __init__(self, in_channels, hidden_channels, num_layers, heads=1, dropout=0.1):
        super(GraphLocalEncoder, self).__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            in_c = in_channels if i == 0 else hidden_channels * heads
            conv = gnn.GATv2Conv(in_c, hidden_channels, heads=heads, dropout=dropout)
            self.convs.append(conv)
            self.norms.append(gnn.norm.PairNorm())

    def forward(self, x, edge_index):
        # Применяем графовые слои с нормализацией и активацией
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
        return x


class GraphTransformerHybrid(nn.Module):
    """
    Гибридная архитектура, комбинирующая локальное кодирование с последующим трансформер-блоком.
    """

    def __init__(self, in_channels, hidden_channels, num_gnn_layers,
                 num_transformer_layers, nhead, out_channels, dropout=0.1):
        super(GraphTransformerHybrid, self).__init__()
        # Локальное кодирование: графовый энкодер
        self.local_encoder = GraphLocalEncoder(in_channels, hidden_channels, num_gnn_layers, heads=nhead, dropout=dropout)

        # Проекция в пространство трансформера
        self.input_proj = nn.Linear(hidden_channels * nhead, hidden_channels)

        # Трансформер-блок
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_channels, nhead=nhead, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # Предсказание атрибутов ребра
        # Представление ребра формируется как конкатенация представлений исходного и конечного узлов
        edge_input_dim = hidden_channels * 2
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, edge_input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(edge_input_dim, out_channels)
        )

    def forward(self, data):
        # data.x - признаки узлов, data.edge_index - топология графа
        x, edge_index = data.x, data.edge_index

        # 1. Локальное кодирование
        x_local = self.local_encoder(x, edge_index)
        # Если используется GAT с несколькими головами, размерность может быть [num_nodes, hidden_channels * nhead]

        # 2. Проекция для трансформера
        x_proj = self.input_proj(x_local)

        # 3. Трансформер-блок
        # Трансформер ожидает вход в форме [sequence_length, batch_size, d_model].
        # Если обрабатываем один граф, можно транспонировать:
        x_transformed = self.transformer(x_proj.unsqueeze(1)).squeeze(1)

        # 4. Формирование представления ребра
        src, dst = edge_index
        edge_repr = torch.cat([x_transformed[src], x_transformed[dst]], dim=-1)

        # 5. Предсказание атрибута ребра
        edge_out = self.edge_mlp(edge_repr)

        return edge_out
