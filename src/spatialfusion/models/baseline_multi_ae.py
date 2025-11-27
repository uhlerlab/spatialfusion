"""
Baseline multi-modal autoencoder models for paired image and gene expression data.

This module provides:
- PairedDatasetBaseline: PyTorch Dataset for paired tensors.
- PairedAE: Autoencoder for image and GEX modalities.
- ResNetEncoder, CNNDecoder: Image encoder/decoder using ResNet and ConvTranspose layers.
- MLPEncoder, MLPDecoder: MLP-based encoder/decoder for gene expression.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset

class PairedDatasetBaseline(Dataset):
    """
    PyTorch Dataset for paired tensors (image and GEX modalities).

    Args:
        X1 (torch.Tensor): Image modality tensor.
        X2 (torch.Tensor): GEX modality tensor.
    """
    def __init__(self, X1: torch.Tensor, X2: torch.Tensor):
        assert len(X1) == len(X2), "Tensor lengths must match"
        self.X1 = X1
        self.X2 = X2

    def __len__(self):
        """Return the number of samples."""
        return len(self.X1)

    def __getitem__(self, idx):
        """Return a tuple of paired samples (X1, X2) at index idx."""
        return self.X1[idx], self.X2[idx]

class ResNetEncoder(nn.Module):
    """
    Image encoder using a pretrained ResNet backbone.

    Args:
        latent_dim (int): Latent space dimension.
        backbone (str): ResNet backbone name (e.g., 'resnet18').
        freeze (bool): If True, freeze backbone weights.
    """
    def __init__(self, latent_dim, backbone='resnet18', freeze=False):
        super().__init__()
        resnet = getattr(models, backbone)(pretrained=True)

        # Remove classification head
        modules = list(resnet.children())[:-1]
        self.backbone = nn.Sequential(*modules)  # Output shape: (B, F, 1, 1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(resnet.fc.in_features, latent_dim)

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        Forward pass for image encoder.

        Args:
            x (torch.Tensor): Input image tensor.
        Returns:
            torch.Tensor: Latent vector.
        """
        x = self.backbone(x)  # shape: (B, F, 1, 1)
        x = self.flatten(x)   # shape: (B, F)
        return self.fc(x)     # shape: (B, latent_dim)


class CNNDecoder(nn.Module):
    """
    Image decoder using ConvTranspose layers to reconstruct images from latent vectors.

    Args:
        latent_dim (int): Latent space dimension.
        output_size (tuple): Output image size (channels, height, width).
    """
    def __init__(self, latent_dim, output_size=(3, 224, 224)):
        super().__init__()
        self.output_size = output_size
        self.fc = nn.Linear(latent_dim, 512 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # 7 → 14
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 14 → 28
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),   # 28 → 56
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),    # 56 → 112
            nn.ReLU(),
            nn.ConvTranspose2d(32, output_size[0], kernel_size=3, stride=2, padding=1, output_padding=1),  # 112 → 224
            nn.Sigmoid(),  # Optional: use Tanh() if inputs are normalized to [-1, 1]
        )

    def forward(self, z):
        """
        Forward pass for image decoder.

        Args:
            z (torch.Tensor): Latent vector.
        Returns:
            torch.Tensor: Reconstructed image tensor.
        """
        x = self.fc(z)
        x = x.view(-1, 512, 7, 7)
        return self.decoder(x)

class MLPEncoder(nn.Module):
    """
    MLP-based encoder for gene expression data.

    Args:
        input_dim (int): Input feature dimension.
        latent_dim (int): Latent space dimension.
    """
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )

    def forward(self, x):
        """
        Forward pass for GEX encoder.

        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Latent vector.
        """
        return self.net(x)


class MLPDecoder(nn.Module):
    """
    MLP-based decoder for gene expression data.

    Args:
        latent_dim (int): Latent space dimension.
        output_dim (int): Output feature dimension.
    """
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, z):
        """
        Forward pass for GEX decoder.

        Args:
            z (torch.Tensor): Latent vector.
        Returns:
            torch.Tensor: Reconstructed GEX tensor.
        """
        return self.net(z)


class PairedAE(nn.Module):
    """
    Autoencoder for paired image and gene expression modalities.

    Args:
        d2_dim (int): GEX input dimension.
        latent_dim (int): Latent space dimension.
        resnet_backbone (str): ResNet backbone name for image encoder.
        freeze_resnet (bool): If True, freeze image encoder weights.
    """
    def __init__(self, d2_dim, latent_dim, resnet_backbone='resnet18', freeze_resnet=False):
        super().__init__()

        # Modality 1 (image)
        self.encoder1 = ResNetEncoder(latent_dim, backbone=resnet_backbone, freeze=freeze_resnet)
        self.decoder1 = CNNDecoder(latent_dim)

        # Modality 2 (GEX)
        self.encoder2 = MLPEncoder(d2_dim, latent_dim)
        self.decoder2 = MLPDecoder(latent_dim, d2_dim)

    def forward(self, d1, d2):
        """
        Forward pass for paired autoencoder.

        Args:
            d1 (torch.Tensor): Image modality input.
            d2 (torch.Tensor): GEX modality input.
        Returns:
            dict: Outputs including latent vectors, reconstructions, and cross-modal predictions.
        """
        z1 = self.encoder1(d1)
        z2 = self.encoder2(d2)

        recon1 = self.decoder1(z1)
        recon2 = self.decoder2(z2)

        cross12 = self.decoder2(z1)  # image -> gex
        cross21 = self.decoder1(z2)  # gex -> image

        return {
            "z1": z1,
            "z2": z2,
            "recon1": recon1,
            "recon2": recon2,
            "cross12": cross12,
            "cross21": cross21
        }
