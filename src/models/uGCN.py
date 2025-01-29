import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn

class uGCN(nn.Module):
    def __init__(self, in_node_dim, in_edge_dim, out_dim,
                 node_conv_layer_type='GCNConv',
                 node_conv_layer_kwargs=dict(),
                 node_conv_layer_list=[16, 8],
                 node_global_pool_type='global_mean_pool',
                 edge_fc_layer_list=[8],
                 out_fc_layer_list=[8],
                 split_out_fc=False,
                 **kwargs):
        """Универсальная GCN с опцией раздельных выходных веток.
        
        Args:
            in_node_dim (int): Размерность признаков узлов.
            in_edge_dim (int): Размерность признаков ребер.
            out_dim (int): Размерность выходных меток.
            node_conv_layer_type (str): Тип графового слоя (например, GCNConv).
            node_conv_layer_list (list): Список размерностей графовых слоев.
            edge_fc_layer_list (list): Список размерностей FC-слоев для ребер.
            out_fc_layer_list (list): Список размерностей выходных FC-слоев.
            split_out_fc (bool): Если True, для каждой метки создается отдельная ветка.
        """
        super(uGCN, self).__init__()

        # Графовые слои для узлов с динамической настройкой размеров
        conv = getattr(importlib.import_module(f'torch_geometric.nn'), node_conv_layer_type)
        self.global_pool = (
            getattr(importlib.import_module(f'torch_geometric.nn'), node_global_pool_type) 
            if node_global_pool_type else None
        )
        global_pool_concat_coef = 2 if self.global_pool else 1  # Коэффициент конкатенации
        node_conv_heads = node_conv_layer_kwargs.get('heads', 1)

        # Графовые слои для узлов
        self.node_conv_list = nn.ModuleList()
        in_channels = in_node_dim * global_pool_concat_coef
        for dim in node_conv_layer_list:
            self.node_conv_list.append(conv(in_channels, dim, **node_conv_layer_kwargs))
            in_channels = dim * node_conv_heads * global_pool_concat_coef  # Обновление входной размерности

        # FC-слои для признаков ребер
        self.edge_fc_list = nn.ModuleList()
        in_channels = in_edge_dim
        for dim in edge_fc_layer_list:
            self.edge_fc_list.append(nn.Linear(in_channels, dim))
            in_channels = dim

        # Слои для объединения признаков: раздельные или общие
        self.split_out_fc = split_out_fc
        if self.split_out_fc:
            # Отдельные ветки для каждой выходной метки
            self.out_branches = nn.ModuleList()
            for _ in range(out_dim):
                out_fc_list = nn.ModuleList()
                # Вход: признаки узлов (src и dst) + признаки ребер
                in_channels = node_conv_layer_list[-1] * 2 * node_conv_heads + edge_fc_layer_list[-1]
                for dim in out_fc_layer_list:
                    out_fc_list.append(nn.Linear(in_channels, dim))
                    in_channels = dim
                final_layer = nn.Linear(in_channels, 1)
                self.out_branches.append(nn.ModuleList([out_fc_list, final_layer]))
        else:
            # Общая ветка для всех меток
            self.out_fc_list = nn.ModuleList()
            # Вход: признаки узлов (src и dst) + признаки ребер
            in_channels = node_conv_layer_list[-1] * 2 * node_conv_heads + edge_fc_layer_list[-1]
            for dim in out_fc_layer_list:
                self.out_fc_list.append(nn.Linear(in_channels, dim))
                in_channels = dim
            self.final_layer = nn.Linear(in_channels, out_dim)

    def forward(self, data):
        """Прямой проход данных через модель."""
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        src, dst = edge_index

        # Обработка узлов через графовые слои
        for node_conv in self.node_conv_list:
            if self.global_pool is not None:
                graph_pool = self.global_pool(x, data.batch)[data.batch]  # Глобальный пулинг
                x = torch.cat([x, graph_pool], dim=1)  # Объединение локальных и глобальных признаков
            x = F.relu(node_conv(x, edge_index))

        # Обработка признаков ребер через FC
        edge_emb = edge_attr
        for edge_fc in self.edge_fc_list:
            edge_emb = F.relu(edge_fc(edge_emb))

        # Формирование выходов
        if self.split_out_fc:
            # Раздельные ветки для каждой метки
            edge_feature_pred_list = []
            for out_fc_list, final_layer in self.out_branches:
                # Конкатенация признаков узлов и ребер
                edge_features = torch.cat([x[src], x[dst], edge_emb], dim=1)
                # Прогон через выходные слои
                for out_fc in out_fc_list:
                    edge_features = F.relu(out_fc(edge_features))
                edge_feature_pred = final_layer(edge_features)
                edge_feature_pred_list.append(edge_feature_pred)
            edge_pred = torch.cat(edge_feature_pred_list, dim=1)  # Объединение предсказаний
        else:
            # Общая ветка
            # Конкатенация признаков узлов и ребер
            edge_features = torch.cat([x[src], x[dst], edge_emb], dim=1)
            # Прогон через выходные слои
            for out_fc in self.out_fc_list:
                edge_features = F.relu(out_fc(edge_features))
            edge_pred = self.final_layer(edge_features)

        return edge_pred