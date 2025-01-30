import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn

class MultiScaleEdgeGCN(nn.Module):
    def __init__(self, in_node_dim, in_edge_dim, out_dim, hidden_dim=64, scales=3):
        """Иерархическая GCN для регрессии атрибутов ребер.
        
        Args:
            in_node_dim (int): Размерность признаков узлов.
            in_edge_dim (int): Размерность признаков ребер.
            out_dim (int): Размерность выходных меток (для каждого ребра).
            hidden_dim (int): Размерность скрытых слоев.
            scales (int): Количество уровней иерархии.
        """
        super().__init__()
        self.scales = scales
        self.hidden_dim = hidden_dim

        # Графовые слои для узлов
        self.convs = nn.ModuleList([
            gnn.GCNConv(in_node_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(scales)
        ])

        # Обработка признаков ребер
        self.edge_processor = nn.Sequential(
            nn.Linear(in_edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Выходные слои для ребер
        self.edge_fc = nn.Sequential(
            nn.Linear(hidden_dim * (2 * scales + 1), hidden_dim),  # [src, dst, edge]
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, data):
        """Прямой проход для предсказания атрибутов ребер."""
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        src, dst = edge_index  # Индексы исходных и целевых узлов ребер

        # Иерархическая обработка узлов
        combined = []
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            combined.append(x[src])  # Сохраняем признаки исходных узлов
            combined.append(x[dst])  # Сохраняем целевых исходных узлов

        # Обработка признаков ребер
        edge_emb = self.edge_processor(edge_attr)  # [num_edges, hidden_dim]
        combined = torch.cat([*combined, edge_emb], dim=1)  # [num_edges, hidden_dim * (2 * scales + 1)]

        # Предсказание для ребер
        return self.edge_fc(combined)  # [num_edges, out_dim]
