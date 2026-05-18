"""Extract flat feature vectors from PyG Data objects for sklearn baselines.

Each graph is converted to a 78-dimensional vector using per-dimension
aggregation statistics (mean, std, min, max) on node features, edge features,
and global attributes, plus graph-level metadata (num_nodes, num_edges).
"""

import numpy as np
import torch
from torch_geometric.data import Data


def _stats(arr: np.ndarray) -> np.ndarray:
    """Compute per-column mean, std, min, max and return as flat array."""
    return np.concatenate([
        arr.mean(axis=0),
        arr.std(axis=0),
        arr.min(axis=0),
        arr.max(axis=0),
    ])


def extract_single(data: Data) -> np.ndarray:
    """Convert a single PyG Data object to a flat 78-dim numpy vector."""
    x = data.x.cpu().numpy()                 # [N, in_node_dim]
    edge_attr = data.edge_attr.cpu().numpy() # [E, in_edge_dim]
    global_ = data.global_attrs.cpu().numpy() # [1, global_dim]

    features = np.concatenate([
        _stats(x),
        _stats(edge_attr),
        _stats(global_),
        np.array([x.shape[0], edge_attr.shape[0]], dtype=np.float32),
    ])
    return features


def extract_dataset(dataset, subset_indices):
    """Extract features and labels for a list of indices into *dataset*.

    Handles both paired (each index returns a tuple of two Data objects)
    and single (each index returns one Data object) modes.

    Args:
        dataset: PairedGraphDataset or plain list/Dataset of Data objects.
        subset_indices: list of integer indices (e.g. Subset.indices).

    Returns:
        X: np.ndarray of shape [n_samples, 78]
        y: np.ndarray of shape [n_samples]  ]
    """
    X, y = [], []
    for idx in subset_indices:
        item = dataset[idx]
        if isinstance(item, (tuple, list)):
            fwd, bwd = item
            X.append(extract_single(fwd))
            y.append(fwd.edge_label.item())
            X.append(extract_single(bwd))
            y.append(bwd.edge_label.item())
        else:
            X.append(extract_single(item))
            y.append(item.edge_label.item())
    return np.array(X), np.array(y)


def extract_dataset_from_list(data_list):
    """Extract features from a plain list of Data objects (single mode only).

    Args:
        data_list: list of Data objects.

    Returns:
        X: np.ndarray of shape [len(data_list), 78]
        y: np.ndarray of shape [len(data_list)]
    """
    X, y = [], []
    for d in data_list:
        X.append(extract_single(d))
        y.append(d.edge_label.item())
    return np.array(X), np.array(y)
