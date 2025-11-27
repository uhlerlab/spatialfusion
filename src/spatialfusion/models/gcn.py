"""
GCN-based autoencoder model for node feature reconstruction and optional classification.

This module provides:
- GCNAutoencoder: Graph convolutional autoencoder with masked node input and optional classification head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from sklearn.neighbors import NearestNeighbors


class GCNAutoencoder(nn.Module):
    """
    GCN-based Autoencoder with optional classification head and masked node input.

    Args:
        in_dim (int): Input feature dimension.
        hidden_dim (int): Hidden layer size.
        out_dim (int): Output feature dimension (reconstruction).
        node_mask_ratio (float): Fraction of nodes to mask during training.
        num_layers (int): Number of GCN layers.
        dropout (float): Dropout probability.
        noise_std (float): Standard deviation of Gaussian noise added to inputs.
        n_classes (int): Number of classes (if > 0, enables classification).
    """
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 out_dim: int,
                 node_mask_ratio: float = 0.1,
                 num_layers: int = 1,
                 dropout: float = 0.2,
                 noise_std: float = 0.1,
                 n_classes: int = 0):
        """
        Initializes the GCNAutoencoder.

        Args:
            in_dim (int): Input feature dimension.
            hidden_dim (int): Hidden layer size.
            out_dim (int): Output feature dimension (reconstruction).
            node_mask_ratio (float): Fraction of nodes to mask during training.
            num_layers (int): Number of GCN layers.
            dropout (float): Dropout probability.
            noise_std (float): Standard deviation of Gaussian noise added to inputs.
            n_classes (int): Number of classes (if > 0, enables classification).
        """
        super().__init__()
        self.node_mask_ratio = node_mask_ratio
        self.num_layers = num_layers
        self.dropout = dropout
        self.noise_std = noise_std
        self.n_classes = n_classes

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(in_dim))
        nn.init.xavier_uniform_(self.mask_token.unsqueeze(0))

        # GCN encoder layers
        self.gcn_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.gcn_layers.append(dgl.nn.GraphConv(in_dim, hidden_dim))
        self.norm_layers.append(nn.LayerNorm(hidden_dim))

        for _ in range(num_layers - 1):
            self.gcn_layers.append(dgl.nn.GraphConv(hidden_dim, hidden_dim))
            self.norm_layers.append(nn.LayerNorm(hidden_dim))

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

        # Optional classifier
        self.classifier = nn.Linear(hidden_dim, n_classes) if n_classes > 0 else None

    def encode(self, g: dgl.DGLGraph, feat: torch.Tensor) -> torch.Tensor:
        """
        Encodes node features through GCN layers.

        Args:
            g (dgl.DGLGraph): Input graph.
            feat (torch.Tensor): Input node features.

        Returns:
            torch.Tensor: Encoded node representations.
        """
        h = feat
        for i, (gcn, norm) in enumerate(zip(self.gcn_layers, self.norm_layers)):
            h = gcn(g, h)
            h = norm(h)
            if i != self.num_layers - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def forward(self, g: dgl.DGLGraph):
        """
        Forward pass: masked node encoding, decoding, optional classification.

        Args:
            g (dgl.DGLGraph): Input graph with node features in `g.ndata["feat"]`.

        Returns:
            Tuple: (x_recon, x_original, node_mask, latent_representation, logits or None)
        """
        x = g.ndata["feat"]
        num_nodes = g.num_nodes()
        device = x.device

        # Node masking
        num_mask = max(1, int(self.node_mask_ratio * num_nodes))
        mask_idx = torch.randperm(num_nodes, device=device)[:num_mask]
        node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        node_mask[mask_idx] = True

        x_masked = x.clone()
        x_masked[node_mask] = self.mask_token
        x_masked += torch.randn_like(x_masked) * self.noise_std
        x_masked = F.dropout(x_masked, p=self.dropout, training=self.training)

        # Encode and decode
        z = self.encode(g, x_masked)
        z = F.dropout(z, p=self.dropout, training=self.training)
        x_recon = self.decoder(z)

        logits = self.classifier(z) if self.classifier is not None else None
        return x_recon, x, node_mask, z, logits
