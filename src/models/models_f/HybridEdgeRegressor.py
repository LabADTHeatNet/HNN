import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_scatter import scatter_mean

class LocalContextGather(nn.Module):
    def __init__(self, node_dim, local_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(node_dim, local_dim),
            nn.ReLU(),
            nn.Linear(local_dim, local_dim)
        )

    def forward(self, x, edge_index):
        local_context = scatter_mean(x[edge_index[1]], edge_index[0], dim=0, dim_size=x.size(0))
        return self.encoder(local_context)

class HybridEdgeRegressor(nn.Module):
    def __init__(self,
                 node_in_dim,
                 node_hidden_dim,
                 edge_hidden_dim,
                 num_node_layers,
                 num_edge_layers,
                 local_context_dim=64,
                 heads=4,
                 dropout=0.1,
                 out_dim=1):
        super().__init__()

        self.node_gnn = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_node_layers):
            in_dim = node_in_dim if i == 0 else node_hidden_dim * heads
            self.node_gnn.append(
                gnn.GATv2Conv(in_dim, node_hidden_dim, heads=heads, add_self_loops=True, residual=True)
            )
            self.norms.append(nn.LayerNorm(node_hidden_dim * heads))

        self.local_context = LocalContextGather(node_hidden_dim * heads, local_context_dim)

        self.edge_init = nn.Sequential(
            nn.Linear(4 * (node_hidden_dim * heads + local_context_dim), edge_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(edge_hidden_dim, edge_hidden_dim)
        )

        self.edge_updater = nn.ModuleList([
            nn.Sequential(
                nn.Linear(edge_hidden_dim, edge_hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(edge_hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(edge_hidden_dim, edge_hidden_dim),
                nn.LayerNorm(edge_hidden_dim)
            ) for _ in range(num_edge_layers)
        ])

        self.edge_out = nn.Sequential(
            nn.Linear(edge_hidden_dim, edge_hidden_dim),
            nn.ReLU(),
            nn.Linear(edge_hidden_dim, out_dim)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for conv, norm in zip(self.node_gnn, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)

        x_local = self.local_context(x, edge_index)
        x_combined = torch.cat([x, x_local], dim=-1)

        src, dst = edge_index
        x_src, x_dst = x_combined[src], x_combined[dst]

        edge_features = torch.cat([
            x_src,
            x_dst,
            x_src - x_dst,
            x_src * x_dst
        ], dim=-1)

        edge_features = self.edge_init(edge_features)

        for layer in self.edge_updater:
            edge_features = edge_features + layer(edge_features)

        out = self.edge_out(edge_features)
        return out


# Пример создания модели и параметров обучения
if __name__ == '__main__':
    in_channels = 3
    out_channels = 1

    model = HybridEdgeRegressor(
        node_in_dim=in_channels,
        node_hidden_dim=128,
        edge_hidden_dim=256,
        num_node_layers=3,
        num_edge_layers=3,
        local_context_dim=64,
        heads=4,
        dropout=0.15,
        out_dim=out_channels
    ).to("cuda")

    # Пример инициализации обучения
    import torch.optim as optim

    criterion = lambda pred, target: F.huber_loss(pred, target, delta=0.01)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)

    # Пример вызова:
    # for batch in train_loader:
    #     batch = batch.to("cuda")
    #     pred = model(batch)
    #     loss = criterion(pred, batch.edge_target)
    #     loss.backward()
    #     optimizer.step()
    #     optimizer.zero_grad()
    #     scheduler.step()
