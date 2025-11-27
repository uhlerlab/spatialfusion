"""
Utility functions for building graphs, generating subgraphs, and plotting losses for GCN models.

This module provides:
- plot_training_losses: Plot training loss curves for GCN models.
- build_knn_graph: Build k-NN graph from spatial coordinates.
- generate_overlapping_subgraphs: Generate overlapping subgraphs using spatial clustering.
- split_index: Split index strings into sample IDs and corrected indices.
"""

from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import dgl
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors


def plot_training_losses(loss_history: dict, title: str = "Training Losses") -> None:
    """
    Plot total, feature, edge, and regularization losses over training epochs.

    Args:
        loss_history (dict): Dictionary with keys 'total', 'feat', 'edge', 'reg' and corresponding loss lists.
        title (str): Plot title.
    """
    df = pd.DataFrame(loss_history)
    df['epoch'] = range(1, len(df['total']) + 1)
    df_melted = df.melt(id_vars='epoch', var_name='loss_type', value_name='loss_value')

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df_melted, x='epoch', y='loss_value', hue='loss_type')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def build_knn_graph(coords: np.ndarray, k: int = 30) -> dgl.DGLGraph:
    """
    Build a k-nearest neighbors (k-NN) graph from coordinate data.

    Args:
        coords (np.ndarray): Node coordinates.
        k (int): Number of neighbors.

    Returns:
        dgl.DGLGraph: Constructed graph.
    """
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(coords)
    _, indices = nbrs.kneighbors(coords)

    src, dst = [], []
    for i, neighbors in enumerate(indices):
        for j in neighbors[1:]:  # skip self
            src.append(i)
            dst.append(j)

    return dgl.graph((src, dst), num_nodes=len(coords))


def generate_overlapping_subgraphs(full_graph: dgl.DGLGraph,
                                   coords: np.ndarray,
                                   subgraph_size: int = 5000,
                                   stride: int = 2500) -> List[dgl.DGLGraph]:
    """
    Generate overlapping subgraphs from a large graph using spatial clustering.

    Args:
        full_graph (dgl.DGLGraph): Full input graph.
        coords (np.ndarray): Node coordinates.
        subgraph_size (int): Max number of nodes in each subgraph.
        stride (int): Distance between cluster centers.

    Returns:
        List[dgl.DGLGraph]: List of subgraphs.
    """
    n_cells = coords.shape[0]
    n_clusters = max(1, n_cells // stride)
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=10000)
    labels = kmeans.fit_predict(coords)

    subgraphs = []
    for cluster_id in np.unique(labels):
        idx = np.where(labels == cluster_id)[0]
        if len(idx) > subgraph_size:
            idx = np.random.choice(idx, subgraph_size, replace=False)

        sg = dgl.node_subgraph(full_graph, idx)
        sg.ndata["feat"] = full_graph.ndata["feat"][idx]
        subgraphs.append(sg)

    return subgraphs


def split_index(index: List[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Splits index values into sample IDs and corrected indices.

    Args:
        index (List[str]): List of index strings.

    Returns:
        tuple[np.ndarray, np.ndarray]: Arrays of sample_ids and corrected indices.
    """
    sample_ids = []
    corrected_index = []

    for idx in index:
        parts = idx.split("_")
        if len(parts) == 2:
            sample_ids.append(parts[0])
            corrected_index.append(parts[1])
        elif len(parts) > 2:
            sample_ids.append("_".join(parts[:3]))
            corrected_index.append("_".join(parts[3:]))
        else:
            sample_ids.append(idx)
            corrected_index.append("")  # fallback if unexpected

    return np.array(sample_ids), np.array(corrected_index)
