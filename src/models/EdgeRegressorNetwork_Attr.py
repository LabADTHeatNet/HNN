import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_scatter import scatter_softmax, scatter_sum


##############################################
# 1. Node Encoder with JumpingKnowledge
##############################################
class NodeEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, heads=1, jump_mode='cat'):
        super(NodeEncoder, self).__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            in_c = in_channels if i == 0 else hidden_channels * heads
            # Многослойные внимательные GNN (GATv2Conv)
            self.convs.append(gnn.GATv2Conv(in_c, hidden_channels, heads=heads, residual=True, add_self_loops=True))
            self.norms.append(nn.LayerNorm(hidden_channels * heads))
        # JumpingKnowledge агрегирует выходы всех слоёв (concat)
        self.jump = gnn.JumpingKnowledge(mode=jump_mode)

    def forward(self, x, edge_index):
        x_list = []
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)           # Передача сообщений
            x = norm(x)                       # Нормализация
            x = F.relu(x)                     # Активация
            x_list.append(x)                  # Сохраняем выход слоя
        x = self.jump(x_list)                # Объединяем выходы всех слоёв
        return x

##############################################
# 2. Optimized Edge Attention Layer
##############################################


class EdgeAttentionLayerFast(nn.Module):
    def __init__(self, in_channels, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.d_head = in_channels // heads
        self.scale = self.d_head ** -0.5

        # Линейные проекции для Q, K, V
        self.q_proj = nn.Linear(in_channels, in_channels)
        self.k_proj = nn.Linear(in_channels, in_channels)
        self.v_proj = nn.Linear(in_channels, in_channels)
        self.out_proj = nn.Linear(in_channels, in_channels)

        self.norm = nn.LayerNorm(in_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, edge_features, edge_index, num_nodes):
        E, d = edge_features.size()
        device = edge_features.device
        H, D = self.heads, self.d_head

        # Формируем многоголовочные представления Q, K, V
        Q = self.q_proj(edge_features).view(E, H, D)
        K = self.k_proj(edge_features).view(E, H, D)
        V = self.v_proj(edge_features).view(E, H, D)

        # Связь: каждому ребру соответствует два узла (src, dst)
        node2edge = torch.cat([edge_index[0], edge_index[1]], dim=0)
        edge2idx = torch.arange(E, device=device).repeat(2)

        # Выбираем Q, K, V для всех рёбер, связанных с узлами
        Q_n = Q[edge2idx]  # [2E, H, D]
        K_n = K[edge2idx]
        V_n = V[edge2idx]

        # Вычисляем скалярные произведения между Q и K
        attn_scores = (Q_n * K_n).sum(dim=-1) * self.scale  # [2E, H]
        attn_weights = scatter_softmax(attn_scores, node2edge, dim=0)  # Взвешивание по узлу
        attn_weights = self.dropout(attn_weights)

        # Агрегируем значения V по весам внимания
        out = attn_weights.unsqueeze(-1) * V_n
        edge_msg = scatter_sum(out, node2edge, dim=0, dim_size=num_nodes)  # [N, H, D]

        # Переносим обратно к ребрам: усредняем src и dst
        edge_msg_src = edge_msg[edge_index[0]]
        edge_msg_dst = edge_msg[edge_index[1]]
        edge_msg = (edge_msg_src + edge_msg_dst) / 2

        edge_msg = edge_msg.reshape(E, H * D)
        out = self.out_proj(edge_msg)  # Финальное обновление ребра
        return self.norm(out + edge_features)  # Skip-соединение + нормализация

##########################################################
# 3. Full Edge Regressor Network
##########################################################


class EdgeRegressorNetwork_Attr(nn.Module):
    def __init__(self,
                 in_node_dim,
                 in_edge_dim,
                 node_hidden_channels,
                 num_node_layers,
                 edge_hidden_channels,
                 num_edge_layers,
                 heads=4,
                 dropout=0.1,
                 out_dim=1,
                 jump_mode='cat'):
        super().__init__()
        self.edge_in_channels = in_edge_dim
        # Кодировщик узлов (GAT + JK)
        self.jump_mode = jump_mode
        self.node_encoder = NodeEncoder(in_node_dim, node_hidden_channels, num_node_layers, heads=1, jump_mode=self.jump_mode)
        node_repr_dim = num_node_layers * node_hidden_channels  # Из-за JK (concat всех слоёв)

        # Инициализация признаков рёбер из признаков узлов (h_u ⊕ h_v)

        node_repr_dim = node_hidden_channels
        if self.jump_mode == 'cat':
            node_repr_dim = num_node_layers * node_hidden_channels   # Из-за JK (concat всех слоёв)
        edge_init_repr_dim = 2 * node_repr_dim + self.edge_in_channels

        # self.edge_init = nn.Linear(2 * node_repr_dim, edge_hidden_channels)
        self.edge_init = nn.Sequential(
            nn.Linear(edge_init_repr_dim, edge_hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(edge_hidden_channels, edge_hidden_channels)
        )
        
        # Итеративные attention-обновления рёбер
        self.edge_attention_layers = nn.ModuleList([
            EdgeAttentionLayerFast(edge_hidden_channels, heads=heads, dropout=dropout)
            for _ in range(num_edge_layers)
        ])

        # Финальный регрессионный MLP по рёбрам
        self.mlp_out = nn.Sequential(
            nn.Linear(edge_hidden_channels + edge_init_repr_dim, edge_hidden_channels),
            nn.ReLU(),
            nn.Linear(edge_hidden_channels, out_dim)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        node_features = self.node_encoder(x, edge_index)  # [N, node_repr_dim]

        # Формируем признаки рёбер: h_u ⊕ h_v
        src, dst = edge_index
        edge_init_feat = torch.cat([node_features[src], node_features[dst]], dim=-1)
        if self.edge_in_channels != 0:
            edge_init_feat = torch.cat([edge_init_feat, data.edge_attr], dim=-1)

        edge_feat = self.edge_init(edge_init_feat)

        # Обновление признаков рёбер через attention
        for layer in self.edge_attention_layers:
            edge_feat = layer(edge_feat, edge_index, num_nodes=node_features.size(0))

        # Финальное объединение и предсказание
        fused_edge_feat = torch.cat([edge_feat, edge_init_feat], dim=-1)
        # fused_edge_feat = torch.cat([edge_feat, edge_init_feat, edge_feat - edge_init_feat], dim=-1)
        output = self.mlp_out(fused_edge_feat)
        return output
