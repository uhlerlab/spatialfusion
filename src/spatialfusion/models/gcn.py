"""
GCN-based autoencoder model for node feature reconstruction and optional classification.

This module provides:
- GCNAutoencoder: Graph convolutional autoencoder with masked node input and optional classification head.
"""
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from sklearn.neighbors import NearestNeighbors


class GCNAutoencoder(nn.Module):
    """
    Graph Convolutional Network (GCN) autoencoder with optional classification head.

    The model performs masked node feature reconstruction using stacked GCN layers.
    During training, a fraction of node features is replaced by a learnable mask
    token and corrupted with noise. The latent node embeddings can optionally be
    passed through a classification head.

    Args:
        in_dim (int): Input feature dimension.
        hidden_dim (int): Hidden layer dimensionality.
        out_dim (int): Output feature dimension for reconstruction.
        node_mask_ratio (float): Fraction of nodes to mask during training.
        num_layers (int): Number of GCN layers in the encoder.
        dropout (float): Dropout probability applied during training.
        noise_std (float): Standard deviation of Gaussian noise added to masked inputs.
        n_classes (int): Number of classes for node classification.
            If 0, no classification head is used.
    """

    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 out_dim: int,
                 node_mask_ratio: float = 0.1,
                 num_layers: int = 1,
                 dropout: float = 0.2,
                 noise_std: float = 0.1,
                 n_classes: int = 0,
                 combine_mode: str = "average"):
        """
        Initialize the GCN autoencoder.

        Args:
            in_dim: Input node feature dimension.
            hidden_dim: Hidden layer size for GCN encoder.
            out_dim: Output feature dimension for reconstruction.
            node_mask_ratio: Fraction of nodes to mask during training.
            num_layers: Number of GCN layers.
            dropout: Dropout probability.
            noise_std: Standard deviation of Gaussian noise applied to inputs.
            n_classes: Number of output classes for classification.
                If greater than zero, a classification head is created.
        """
        super().__init__()
        self.node_mask_ratio = node_mask_ratio
        self.num_layers = num_layers
        self.dropout = dropout
        self.noise_std = noise_std
        self.n_classes = n_classes
        self.combine_mode = combine_mode

        if combine_mode == "gated":

            self.modal_dim = in_dim // 2

            self.fusion_gate = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU(),
                nn.Linear(in_dim, self.modal_dim),
                nn.Sigmoid()
            )

            first_gcn_dim = self.modal_dim

        else:
            first_gcn_dim = in_dim

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(first_gcn_dim))
        nn.init.xavier_uniform_(self.mask_token.unsqueeze(0))

        # GCN encoder layers
        self.gcn_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.gcn_layers.append(dgl.nn.GraphConv(first_gcn_dim, hidden_dim))
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
        self.classifier = nn.Linear(
            hidden_dim, n_classes) if n_classes > 0 else None

    def encode(self, g, feat):

        h = feat

        if self.combine_mode == "gated":

            z1, z2 = torch.chunk(h, 2, dim=1)

            gates = self.fusion_gate(h)

            h = gates * z1 + (1.0 - gates) * z2

        for i, (gcn, norm) in enumerate(
            zip(self.gcn_layers, self.norm_layers)
        ):
            h = gcn(g, h)
            h = norm(h)

            if i != self.num_layers - 1:
                h = F.relu(h)
                h = F.dropout(
                    h,
                    p=self.dropout,
                    training=self.training,
                )

        return h

    def forward(self, g: dgl.DGLGraph):
        """
        Forward pass of the GCN autoencoder.

        The forward pass performs:
        1. Random node masking
        2. Noise injection and dropout
        3. GCN encoding
        4. Feature reconstruction
        5. Optional node classification

        Args:
            g: Input DGL graph with node features stored in `g.ndata["feat"]`.

        Returns:
            A tuple `(x_recon, x_original, node_mask, z, logits)` where:
                - x_recon: Reconstructed node features.
                - x_original: Original input node features.
                - node_mask: Boolean mask indicating masked nodes.
                - z: Latent node embeddings.
                - logits: Classification logits if `n_classes > 0`,
                  otherwise `None`.
        """

        x_input = g.ndata["feat"]
        x = x_input

        if self.combine_mode == "gated":

            z1, z2 = torch.chunk(x, 2, dim=1)

            gates = self.fusion_gate(x)

            x = gates * z1 + (1 - gates) * z2

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
        return x_recon, x_input, node_mask, z, logits


class SpatialSmoothingBaseline(nn.Module):

    def __init__(
        self,
        node_mask_ratio=0.1,
        num_smoothing_steps=5,
        alpha=0.8,
    ):
        super().__init__()

        self.node_mask_ratio = node_mask_ratio
        self.num_smoothing_steps = num_smoothing_steps
        self.alpha = alpha

    def smooth(self, g, x0):

        h = x0

        with g.local_scope():

            for _ in range(self.num_smoothing_steps):

                g.ndata["h"] = h

                g.update_all(
                    fn.copy_u("h", "m"),
                    fn.mean("m", "h_neigh")
                )

                h_neigh = g.ndata["h_neigh"]

                # residual diffusion
                h = (
                    self.alpha * h_neigh
                    + (1 - self.alpha) * x0
                )

        return h

    def forward(self, g):

        x = g.ndata["feat"]

        num_nodes = g.num_nodes()
        device = x.device

        # SAME masking strategy as GCMAE
        num_mask = max(
            1,
            int(self.node_mask_ratio * num_nodes)
        )

        mask_idx = torch.randperm(
            num_nodes,
            device=device
        )[:num_mask]

        node_mask = torch.zeros(
            num_nodes,
            dtype=torch.bool,
            device=device
        )

        node_mask[mask_idx] = True

        # zero-mask nodes
        x_masked = x.clone()
        x_masked[node_mask] = 0.0

        # pure neighbor averaging
        x_recon = self.smooth(g, x_masked)

        # Match reconstruction target dimension in concat mode
        if "feat_orig" in g.ndata:
            out_dim = g.ndata["feat_orig"].shape[1]
            x_recon = x_recon[:, :out_dim]

        # no latent representation
        z = x_recon

        logits = None

        return (
            x_recon,
            x,
            node_mask,
            z,
            logits
        )

    def encode(self, g, feat):
        return self.smooth(g, feat)
