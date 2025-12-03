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

    Args:
        df1 (pd.DataFrame): First modality data (samples x features).
        df2 (pd.DataFrame): Second modality data (samples x features).
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
    Build a multi-layer perceptron (MLP) with specified layer dimensions and activation.

    Args:
        layer_dims (list): List of layer sizes.
        activation_fn (nn.Module): Activation function class.

    Returns:
        nn.Sequential: MLP network.
    """
    layers = []
    for i in range(len(layer_dims) - 1):
        layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
        if i < len(layer_dims) - 2:
            layers.append(activation_fn())
    return nn.Sequential(*layers)


class EncoderAE(nn.Module):
    """
    Standard encoder for autoencoder models.

    Args:
        input_dim (int): Input feature dimension.
        hidden_dims (list): Hidden layer sizes.
        latent_dim (int): Latent space dimension.
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
    Standard autoencoder for paired modalities.

    Args:
        d1_dim (int): Input dimension for modality 1.
        d2_dim (int): Input dimension for modality 2.
        latent_dim (int): Latent space dimension.
        enc_hidden_dims (list, optional): Encoder hidden layer sizes.
        dec_hidden_dims (list, optional): Decoder hidden layer sizes.
    """

    def __init__(self, d1_dim, d2_dim, latent_dim, enc_hidden_dims=None, dec_hidden_dims=None):
        super().__init__()
        enc_hidden_dims = enc_hidden_dims or [64]
        dec_hidden_dims = dec_hidden_dims or [64]

        self.encoder1 = EncoderAE(d1_dim, enc_hidden_dims, latent_dim)
        self.encoder2 = EncoderAE(d2_dim, enc_hidden_dims, latent_dim)
        self.decoder1 = Decoder(latent_dim, dec_hidden_dims, d1_dim)
        self.decoder2 = Decoder(latent_dim, dec_hidden_dims, d2_dim)

    def forward(self, d1, d2=None):
        """
        Forward pass for paired AE.

        Args:
            d1 (torch.Tensor): Modality 1 input.
            d2 (torch.Tensor): Modality 2 input.
        Returns:
            dict: Outputs including latent vectors and reconstructions.
        """
        # Encoder for modality 1
        z1 = self.encoder1(d1)

        # If d2 is provided, run full paired AE
        if d2 is not None:
            z2 = self.encoder2(d2)
            recon1 = self.decoder1(z1)
            recon2 = self.decoder2(z2)
            cross12 = self.decoder2(z1)
            cross21 = self.decoder1(z2)
        else:
            # UNI-only mode
            z2 = None
            recon1 = self.decoder1(z1)
            recon2 = None
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
