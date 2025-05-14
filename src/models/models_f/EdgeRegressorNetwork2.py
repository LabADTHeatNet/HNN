import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_scatter import scatter_softmax, scatter_sum

class NodeEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, heads=1, jump_mode='cat'):
        super(NodeEncoder, self).__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            in_c = in_channels if i == 0 else hidden_channels * heads
            self.convs.append(gnn.GATv2Conv(in_c, hidden_channels, heads=heads, residual=True, add_self_loops=True))
            self.norms.append(nn.LayerNorm(hidden_channels * heads))
        self.jump = gnn.JumpingKnowledge(mode=jump_mode)

    def forward(self, x, edge_index):
        x_list = []
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x_list.append(x)
        x = self.jump(x_list)
        return x

class EdgeAttentionLayerFast(nn.Module):
    def __init__(self, in_channels, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.d_head = in_channels // heads
        self.scale = self.d_head ** -0.5

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

        Q = self.q_proj(edge_features).view(E, H, D)
        K = self.k_proj(edge_features).view(E, H, D)
        V = self.v_proj(edge_features).view(E, H, D)

        node2edge = torch.cat([edge_index[0], edge_index[1]], dim=0)
        edge2idx = torch.arange(E, device=device).repeat(2)

        Q_n = Q[edge2idx]
        K_n = K[edge2idx]
        V_n = V[edge2idx]

        attn_scores = (Q_n * K_n).sum(dim=-1) * self.scale
        attn_weights = scatter_softmax(attn_scores, node2edge, dim=0)
        attn_weights = self.dropout(attn_weights)

        out = attn_weights.unsqueeze(-1) * V_n
        edge_msg = scatter_sum(out, node2edge, dim=0, dim_size=num_nodes)

        edge_msg_src = edge_msg[edge_index[0]]
        edge_msg_dst = edge_msg[edge_index[1]]
        edge_msg = (edge_msg_src + edge_msg_dst) / 2

        edge_msg = edge_msg.reshape(E, H * D)
        out = self.out_proj(edge_msg)
        return self.norm(out + edge_features)

class EdgeRegressorNetwork2(nn.Module):
    def __init__(self,
                 node_in_channels,
                 node_hidden_channels,
                 num_node_layers,
                 edge_hidden_channels,
                 num_edge_layers,
                 heads=4,
                 dropout=0.1,
                 out_channels=1,
                 jump_mode='cat'):
        super().__init__()
        self.jump_mode = jump_mode
        self.node_encoder = NodeEncoder(node_in_channels, node_hidden_channels, num_node_layers, heads=1, jump_mode=self.jump_mode)
        node_repr_dim = node_hidden_channels
        if self.jump_mode == 'cat':
            node_repr_dim = num_node_layers * node_hidden_channels

        self.edge_init = nn.Sequential(
            nn.Linear(4 * node_repr_dim, edge_hidden_channels),
            nn.ReLU(),
            nn.LayerNorm(edge_hidden_channels),
            nn.Dropout(dropout),
            nn.Linear(edge_hidden_channels, edge_hidden_channels),
            nn.LayerNorm(edge_hidden_channels)
        )

        self.edge_attention_layers = nn.ModuleList([
            EdgeAttentionLayerFast(edge_hidden_channels, heads=heads, dropout=dropout)
            for _ in range(num_edge_layers)
        ])

        self.mlp_out = nn.Sequential(
            nn.Linear(edge_hidden_channels + 4 * node_repr_dim, edge_hidden_channels),
            nn.ReLU(),
            nn.LayerNorm(edge_hidden_channels),
            nn.Linear(edge_hidden_channels, out_channels)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        node_features = self.node_encoder(x, edge_index)

        src, dst = edge_index
        h_src, h_dst = node_features[src], node_features[dst]

        edge_init_feat = torch.cat([
            h_src,
            h_dst,
            h_src - h_dst,
            h_src * h_dst
        ], dim=-1)

        edge_feat = self.edge_init(edge_init_feat)

        for layer in self.edge_attention_layers:
            edge_feat = layer(edge_feat, edge_index, num_nodes=node_features.size(0))

        fused_edge_feat = torch.cat([edge_feat, edge_init_feat], dim=-1)
        output = self.mlp_out(fused_edge_feat)
        return output

if __name__ == '__main__':
    in_channels = 3
    out_channels = 1

    model = EdgeRegressorNetwork(
        node_in_channels=in_channels,
        node_hidden_channels=192,
        num_node_layers=4,
        edge_hidden_channels=384,
        num_edge_layers=4,
        heads=8,
        dropout=0.15,
        out_channels=out_channels,
        jump_mode='cat'
    ).to("cuda")

    import torch.optim as optim

    criterion = lambda pred, target: F.huber_loss(pred, target, delta=0.01)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)

    # Пример использования в цикле обучения:
    # for batch in train_loader:
    #     batch = batch.to("cuda")
    #     pred = model(batch)
    #     loss = criterion(pred, batch.edge_target)
    #     loss.backward()
    #     optimizer.step()
    #     optimizer.zero_grad()
    #     scheduler.step()