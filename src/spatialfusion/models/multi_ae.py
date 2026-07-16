"""
Multi-modal autoencoder models for paired datasets.

This module provides:
- PairedDataset: PyTorch Dataset for paired samples.
- PairedAE: Standard autoencoder for paired modalities.
- EncoderAE, Decoder: Building blocks for autoencoders.
- build_mlp: Utility to build MLP networks.
"""
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# -------- Dataset and Model Classes -------- #


class PairedDataset(Dataset):
    """
    PyTorch Dataset for paired samples from two modalities.

    Each item returned by the dataset is a tuple `(X1, X2)` corresponding
    to aligned samples from two feature matrices.

    Args:
        df1 (pd.DataFrame): First modality data (samples × features).
        df2 (pd.DataFrame): Second modality data (samples × features).

    Raises:
        AssertionError: If the indices of `df1` and `df2` do not match.
    """

    def __init__(self, df1: pd.DataFrame, df2: pd.DataFrame):
        assert all(df1.index == df2.index), "Indices must match"
        self.X1 = torch.tensor(df1.values, dtype=torch.float32)
        self.X2 = torch.tensor(df2.values, dtype=torch.float32)

    def __len__(self):
        """Return the number of samples."""
        return len(self.X1)

    def __getitem__(self, idx):
        """Return a tuple of paired samples (X1, X2) at index idx."""
        return self.X1[idx], self.X2[idx]


def build_mlp(layer_dims, activation_fn=nn.ReLU):
    """
    Build a multi-layer perceptron (MLP) network.

    The MLP consists of linear layers with an activation function applied
    after each hidden layer. No activation is applied to the output layer.

    Args:
        layer_dims: List of layer dimensions, including input and output.
        activation_fn: Activation function class used between layers.

    Returns:
        A `torch.nn.Sequential` MLP model.
    """
    layers = []
    for i in range(len(layer_dims) - 1):
        layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
        if i < len(layer_dims) - 2:
            layers.append(activation_fn())
    return nn.Sequential(*layers)


class EncoderAE(nn.Module):
    """
    Encoder network for autoencoder models.

    This encoder maps input features to a latent representation
    using a feedforward neural network.

    Args:
        input_dim: Input feature dimensionality.
        hidden_dims: List of hidden layer sizes.
        latent_dim: Dimensionality of the latent space.
    """

    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        self.model = build_mlp([input_dim] + hidden_dims + [latent_dim])

    def forward(self, x):
        """
        Forward pass for AE encoder.

        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            z (torch.Tensor): Latent vector.
        """
        z = self.model(x)
        return z


class Decoder(nn.Module):
    """
    Decoder network for autoencoders.

    Args:
        latent_dim (int): Latent space dimension.
        hidden_dims (list): Hidden layer sizes.
        output_dim (int): Output feature dimension.
    """

    def __init__(self, latent_dim, hidden_dims, output_dim):
        super().__init__()
        self.net = build_mlp([latent_dim] + hidden_dims + [output_dim])

    def forward(self, z):
        """
        Forward pass for decoder.

        Args:
            z (torch.Tensor): Latent vector.
        Returns:
            torch.Tensor: Reconstructed output.
        """
        return self.net(z)


class PairedAE(nn.Module):
    """
    Autoencoder model for paired modalities.

    This model learns a shared latent representation for two input
    modalities and supports:
    - Within-modality reconstruction
    - Cross-modality reconstruction
    - Single-modality operation (modality 1 only or modality 2 only)

    Args:
        d1_dim: Input feature dimension for modality 1.
        d2_dim: Input feature dimension for modality 2.
        latent_dim: Dimensionality of the latent space.
        enc_hidden_dims: Hidden layer sizes for the encoders.
        dec_hidden_dims: Hidden layer sizes for the decoders.
    """

    def __init__(self, d1_dim, d2_dim, latent_dim, enc_hidden_dims=None, dec_hidden_dims=None):
        super().__init__()
        enc_hidden_dims = enc_hidden_dims or [64]
        dec_hidden_dims = dec_hidden_dims or [64]

        self.encoder1 = EncoderAE(d1_dim, enc_hidden_dims, latent_dim)
        self.encoder2 = EncoderAE(d2_dim, enc_hidden_dims, latent_dim)
        self.decoder1 = Decoder(latent_dim, dec_hidden_dims, d1_dim)
        self.decoder2 = Decoder(latent_dim, dec_hidden_dims, d2_dim)

    def forward(self, d1=None, d2=None):
        """
        Forward pass of the paired autoencoder.

        If both modalities are provided, the model computes:
        - Latent representations for both modalities
        - Within-modality reconstructions
        - Cross-modality reconstructions

        If only `d1` is provided, the model runs in modality-1-only mode.
        If only `d2` is provided, the model runs in modality-2-only mode.

        Args:
            d1: Optional input tensor for modality 1.
            d2: Optional input tensor for modality 2.

        Returns:
            Dictionary containing:
                - "z1": Latent embedding for modality 1 (or None)
                - "z2": Latent embedding for modality 2 (or None)
                - "recon1": Reconstruction of modality 1 (or None)
                - "recon2": Reconstruction of modality 2 (or None)
                - "cross12": Reconstruction of modality 2 from z1 (or None)
                - "cross21": Reconstruction of modality 1 from z2 (or None)

        Raises:
            ValueError: If both `d1` and `d2` are None.
        """
        if d1 is None and d2 is None:
            raise ValueError("At least one of d1 or d2 must be provided.")

        if d1 is not None and d2 is not None:
            # Full paired AE
            z1 = self.encoder1(d1)
            z2 = self.encoder2(d2)
            recon1 = self.decoder1(z1)
            recon2 = self.decoder2(z2)
            cross12 = self.decoder2(z1)
            cross21 = self.decoder1(z2)
        elif d1 is not None:
            # Modality-1-only mode
            z1 = self.encoder1(d1)
            recon1 = self.decoder1(z1)
            z2 = None
            recon2 = None
            cross12 = None
            cross21 = None
        else:
            # Modality-2-only mode
            z2 = self.encoder2(d2)
            recon2 = self.decoder2(z2)
            z1 = None
            recon1 = None
            cross12 = None
            cross21 = None

        return {
            "z1": z1,
            "z2": z2,
            "recon1": recon1,
            "recon2": recon2,
            "cross12": cross12,
            "cross21": cross21
        }