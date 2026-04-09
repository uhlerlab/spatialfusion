# finetune_models_modular.py

from spatialfusion.utils.ae_data_loader import load_and_preprocess_sample, safe_standardize
from typing import Optional, Tuple
import os
import pathlib as pl
from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import pandas as pd
import numpy as np
import scanpy as sc
from tqdm import tqdm
import dgl
from dgl.dataloading import GraphDataLoader

# --- Project imports ---
from spatialfusion.utils.ae_data_loader import load_and_preprocess_sample
from spatialfusion.models.multi_ae import PairedDataset, PairedAE
from spatialfusion.utils.embed_ae_utils import (
    extract_embeddings_for_all_samples,
    save_embeddings_separately,
    safe_standardize,
)
from spatialfusion.utils.gcn_utils import build_knn_graph, generate_overlapping_subgraphs
from spatialfusion.models.gcn import GCNAutoencoder
from spatialfusion.utils.pkg_ckpt import resolve_pkg_ckpt

DEFAULT_AE_CKPT_RELPATH = "spatialfusion-multimodal-ae.pt"
DEFAULT_GCN_CKPT_RELPATH = "spatialfusion-full-gcn.pt"


# ---------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------

def get_device() -> torch.device:
    """Select an available computation device."""
    return (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )


# ---------------------------------------------------------------
# Autoencoder (AE) functions
# ---------------------------------------------------------------


def _to_str_index(index):
    """Convert indices to canonical string form."""
    return index.astype(str).str.strip()


def build_ae_dataset(
    samples,
    base_path=None,
    preloaded_data=None,
    batch_size=128,
    max_cells=10**6,
):
    """
    Build a PyTorch DataLoader for autoencoder fine-tuning.

    This function supports two data sources:
    - Disk-based loading using `base_path`
    - In-memory loading using `preloaded_data`

    In both cases, the same preprocessing steps are applied to ensure
    consistency with the original AE training pipeline.

    Args:
        samples: List of sample identifiers.
        base_path: Root directory containing sample data on disk.
        preloaded_data: Optional dict mapping sample name to
            `(feat1_df, feat2_df)` DataFrames.
        batch_size: Number of cell pairs per batch.
        max_cells: Maximum number of cells loaded per sample (disk mode).

    Returns:
        A tuple `(loader, d1_dim, d2_dim)` where:
            - loader: PyTorch DataLoader over paired features.
            - d1_dim: Feature dimension of modality 1.
            - d2_dim: Feature dimension of modality 2.

    Raises:
        ValueError: If neither `base_path` nor `preloaded_data` is provided.
        KeyError: If a requested sample is missing from `preloaded_data`.
    """

    datasets = []
    d1_dim = d2_dim = None

    if preloaded_data is not None:
        # --- In-memory preprocessing path
        for sample_name in samples:
            if sample_name not in preloaded_data:
                raise KeyError(f"{sample_name} missing from preloaded_data")

            feat1, feat2 = preloaded_data[sample_name]

            # Clean index names
            feat1.index = _to_str_index(feat1.index)
            feat2.index = _to_str_index(feat2.index)

            # Standardize like disk pipeline
            feat1 = safe_standardize(feat1)
            feat2 = safe_standardize(feat2)

            if d1_dim is None:
                d1_dim, d2_dim = feat1.shape[1], feat2.shape[1]
            datasets.append(PairedDataset(feat1, feat2))

    elif base_path is not None:
        # --- On-disk preprocessing path (original logic)
        for sample_name in samples:
            std_feat1, std_feat2, _ = load_and_preprocess_sample(
                sample_name, base_path, max_cells=max_cells
            )
            if d1_dim is None:
                d1_dim, d2_dim = std_feat1.shape[1], std_feat2.shape[1]
            datasets.append(PairedDataset(std_feat1, std_feat2))
    else:
        raise ValueError(
            "Either preloaded_data or base_path must be provided.")

    loader = DataLoader(
        ConcatDataset(datasets),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    return loader, d1_dim, d2_dim


def ae_from_arrays_finetune(
    model,
    feat1: pd.DataFrame,
    feat2: pd.DataFrame,
    adata: Optional[sc.AnnData] = None,
    device: str = "cuda:0",
    batch_size: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run a PairedAE model on in-memory feature matrices for fine-tuning.

    This function mirrors the preprocessing used in disk-based embedding
    extraction:
    - Converts indices to strings
    - Aligns features across modalities (and AnnData if provided)
    - Applies safe standardization
    - Uses encoder networks directly (no reconstruction loss)

    Args:
        model: Pretrained PairedAE model.
        feat1: Feature matrix for modality 1 (cells × features).
        feat2: Feature matrix for modality 2 (cells × features).
        adata: Optional AnnData object for additional index alignment.
        device: Torch device used for inference.
        batch_size: Optional batch size for processing. If None, it will be auto-determined based on input size and available memory.

    Returns:
        A tuple `(z1_df, z2_df, z_joint_df)` where:
            - z1_df: Latent embeddings from encoder1.
            - z2_df: Latent embeddings from encoder2.
            - z_joint_df: Average of z1 and z2 embeddings.

    Raises:
        ValueError: If no overlapping cell identifiers are found.
    """

    # --- Clean and align indices
    feat1.index = _to_str_index(feat1.index)
    feat2.index = _to_str_index(feat2.index)
    idx = set(feat1.index) & set(feat2.index)

    if adata is not None:
        adata.obs_names = _to_str_index(adata.obs_names)
        idx &= set(adata.obs_names)

    if len(idx) == 0:
        raise ValueError("No overlapping cells between inputs / adata.")

    common = sorted(idx)

    # --- Standardize each feature matrix
    std_feat1 = safe_standardize(feat1.loc[common])
    std_feat2 = safe_standardize(feat2.loc[common])

    X1_np = std_feat1.values.astype(np.float32)
    X2_np = std_feat2.values.astype(np.float32)

    n_samples = X1_np.shape[0]

    # Auto batch size if not provided
    if batch_size is None:
        est_mem_per_sample = (X1_np.shape[1] + X2_np.shape[1]) * 4 * 2
        batch_size = max(1, int(300 * 1024 * 1024 / est_mem_per_sample))
        batch_size = min(batch_size, 5000)
        batch_size = max(batch_size, 100)

    z1_list, z2_list = [], []

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, n_samples, batch_size), desc="Encoding AE", leave=False):
            end = min(i + batch_size, n_samples)

            x1_batch = torch.from_numpy(X1_np[i:end]).to(device)
            x2_batch = torch.from_numpy(X2_np[i:end]).to(device)

            z1_batch = model.encoder1(x1_batch)
            z2_batch = model.encoder2(x2_batch)

            z1_list.append(z1_batch.cpu().numpy())
            z2_list.append(z2_batch.cpu().numpy())

            del x1_batch, x2_batch, z1_batch, z2_batch
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()

    z1 = np.vstack(z1_list)
    z2 = np.vstack(z2_list)
    # --- Combine
    z_joint = (z1 + z2) / 2.0

    # --- Return DataFrames indexed by cell IDs
    z1_df = pd.DataFrame(z1, index=common)
    z2_df = pd.DataFrame(z2, index=common)
    z_joint_df = pd.DataFrame(z_joint, index=common)

    return z1_df, z2_df, z_joint_df


def finetune_autoencoder(
    loader: DataLoader,
    d1_dim: int,
    d2_dim: int,
    pretrained_ae: str,
    save_dir: pl.Path,
    device: torch.device,
    latent_dim: int = 64,
    enc_hidden_dims: List[int] = [64],
    dec_hidden_dims: List[int] = [64],
    epochs: int = 5,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    grad_clip: Optional[float] = 1.0,
    lambda_recon1: float = 0.5,
    lambda_recon2: float = 0.5,
    lambda_cross12: float = 0.25,
    lambda_cross21: float = 0.25,
    lambda_align: float = 1.0,
) -> torch.nn.Module:
    """
    Fine-tune a pretrained paired autoencoder.

    The training objective includes reconstruction losses for each modality,
    cross-modality reconstruction losses, and a latent alignment loss.

    Args:
        loader: DataLoader yielding paired feature tensors.
        d1_dim: Input dimension of modality 1.
        d2_dim: Input dimension of modality 2.
        pretrained_ae: Path to pretrained AE checkpoint.
        save_dir: Directory to save fine-tuned model and loss history.
        device: Torch device for training.
        latent_dim: Latent space dimensionality.
        enc_hidden_dims: Hidden layer sizes for encoders.
        dec_hidden_dims: Hidden layer sizes for decoders.
        epochs: Number of training epochs.
        lr: Learning rate.
        weight_decay: Weight decay coefficient.
        grad_clip: Optional gradient clipping norm.
        lambda_recon1: Weight for modality 1 reconstruction loss.
        lambda_recon2: Weight for modality 2 reconstruction loss.
        lambda_cross12: Weight for cross reconstruction (1→2).
        lambda_cross21: Weight for cross reconstruction (2→1).
        lambda_align: Weight for latent alignment loss.

    Returns:
        The fine-tuned PairedAE model.
    """
    model = PairedAE(d1_dim, d2_dim, latent_dim,
                     enc_hidden_dims, dec_hidden_dims).to(device)
    state = torch.load(pretrained_ae, map_location=device)
    model.load_state_dict(state, strict=True)
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    criterion = nn.MSELoss()

    hist = {k: [] for k in ["epoch", "total", "recon1",
                            "recon2", "cross12", "cross21", "align"]}
    model.train()

    for epoch in range(1, epochs + 1):
        metrics = {k: 0.0 for k in hist if k != "epoch"}
        for d1, d2 in tqdm(loader, desc=f"AE Epoch {epoch}/{epochs}", leave=False):
            d1, d2 = d1.to(device), d2.to(device)
            optimizer.zero_grad()
            out = model(d1, d2)

            loss_recon1 = criterion(out["recon1"], d1)
            loss_recon2 = criterion(out["recon2"], d2)
            loss_cross12 = criterion(out["cross12"], d2)
            loss_cross21 = criterion(out["cross21"], d1)
            loss_align = criterion(out["z1"], out["z2"])

            loss = (
                lambda_recon1 * loss_recon1 +
                lambda_recon2 * loss_recon2 +
                lambda_cross12 * loss_cross12 +
                lambda_cross21 * loss_cross21 +
                lambda_align * loss_align
            )

            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            for k, v in zip(metrics.keys(), [loss, loss_recon1, loss_recon2, loss_cross12, loss_cross21, loss_align]):
                metrics[k] += float(v.item())

        n = len(loader)
        for k in metrics:
            metrics[k] /= n
        hist["epoch"].append(epoch)
        for k in metrics:
            hist[k].append(metrics[k])
        print(f"[AE {epoch}] total={metrics['total']:.4f}")

    torch.save(model.state_dict(), save_dir / "paired_model_finetuned.pt")
    pd.DataFrame(hist).to_csv(save_dir / "ae_finetune_loss.csv", index=False)
    return model


# ---------------------------------------------------------------
# Graph + GCN functions
# ---------------------------------------------------------------

def standardize_pathways(df: pd.DataFrame, method: str = "robust_z", eps: float = 1e-6, tol: float = 1e-3) -> pd.DataFrame:
    """
    Standardize pathway activation scores column-wise.

    Supported methods:
    - 'robust_z': (x - median) / IQR
    - 'z': (x - mean) / std

    Columns with near-zero variation are set to zero, and all NaN or
    infinite values are replaced with 0.

    Args:
        df: Pathway activation DataFrame.
        method: Standardization method ('robust_z' or 'z').
        eps: Numerical stability constant.
        tol: Threshold for detecting near-zero columns.

    Returns:
        Standardized pathway activation DataFrame (float32).
    """
    df = df.copy()
    all_near_zero = (df.abs().max(axis=0) < tol)

    if method == "z":
        mu = df.mean(axis=0)
        sigma = df.std(axis=0).replace(0, np.nan)
        out = (df - mu) / (sigma + eps)
    else:
        med = df.median(axis=0)
        q1 = df.quantile(0.25, axis=0)
        q3 = df.quantile(0.75, axis=0)
        iqr = (q3 - q1).replace(0, np.nan)
        out = (df - med) / (iqr + eps)

    out.loc[:, all_near_zero] = 0.0
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    return out


def get_coords(adata, eps=1e-6, key="spatial"):
    """
    Extract and standardize spatial coordinates from an AnnData object.

    Args:
        adata: AnnData containing spatial coordinates.
        eps: Small constant to avoid division by zero.
        key: Key in `adata.obsm` storing spatial coordinates.

    Returns:
        Standardized coordinate array.

    Raises:
        KeyError: If the specified spatial key is not present.
    """
    if key in adata.obsm_keys():
        coords = adata.obsm[key]
    else:
        raise KeyError(f"No spatial coords found in adata.obsm (tried {key})")
    coords = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + eps)
    return coords


def build_graphs(
    samples: List[str],
    z_joint_df: pd.DataFrame,
    base_path: Optional[str] = None,
    adatas: Optional[Dict[str, Any]] = None,
    pathway_data: Optional[Dict[str, pd.DataFrame]] = None,
    knn_k: int = 30,
    subgraph_size: int = 5000,
    stride: int = 2500,
    use_cls_loss: bool = True,
    spatial_key: str = "spatial_he",
) -> List[dgl.DGLGraph]:
    """
    Construct spatial graphs (and overlapping subgraphs) for GCN fine-tuning.

    For each sample, this function:
    - Loads or uses preloaded AnnData
    - Aligns cells with joint AE embeddings
    - Builds a kNN spatial graph
    - Optionally attaches pathway activation labels
    - Generates overlapping subgraphs

    Args:
        samples: List of sample identifiers.
        z_joint_df: Joint AE embedding indexed by cell ID.
        base_path: Root directory for loading AnnData and labels.
        adatas: Optional dict of preloaded AnnData objects.
        pathway_data: Optional dict of pathway activation DataFrames.
        knn_k: Number of neighbors for kNN graph.
        subgraph_size: Number of nodes per subgraph.
        stride: Stride between subgraph centers.
        use_cls_loss: Whether classification labels are used.
        spatial_key: Key in AnnData.obsm for spatial coordinates.

    Returns:
        List of DGL graphs ready for GCN training.

    Raises:
        RuntimeError: If no graphs are successfully built.
    """
    graphs = []
    for sample in samples:
        # --- Load AnnData ---
        if adatas and sample in adatas:
            adata = adatas[sample]
        elif base_path:
            adata = sc.read_h5ad(pl.Path(base_path) / sample / "adata.h5ad")
        else:
            raise ValueError("Either adatas or base_path must be provided.")

        new_idx = z_joint_df.index.intersection(adata.obs_names.astype(str))
        if len(new_idx) == 0:
            print(
                f"[{sample}] No overlapping cells between z_joint and adata. Skipping.")
            continue

        adata = adata[new_idx].copy()

        # --- Spatial features ---
        if spatial_key not in adata.obsm:
            raise KeyError(
                f"'{spatial_key}' not found in adata.obsm keys: {list(adata.obsm.keys())}"
            )
        coords = get_coords(adata, eps=1e-6, key=spatial_key)
        joint_emb_np = safe_standardize(
            z_joint_df.loc[new_idx], fill_value=0.0, min_std=1e-6
        ).to_numpy(dtype=np.float32)

        full_graph = build_knn_graph(coords, k=knn_k)
        full_graph.ndata["feat"] = torch.tensor(
            joint_emb_np, dtype=torch.float32)

        # --- Load or assign labels ---
        labels = None
        if use_cls_loss:
            if pathway_data and sample in pathway_data:
                df_labels = pathway_data[sample].loc[adata.obs_names]
                print(f"[{sample}] Using preloaded pathway activation data.")
            elif base_path and (pl.Path(base_path) / sample / "pathway_activation.parquet").exists():
                df_labels = pd.read_parquet(
                    pl.Path(base_path) / sample / "pathway_activation.parquet"
                ).loc[adata.obs_names]
                print(f"[{sample}] Loaded pathway activation labels from disk.")
            else:
                print(
                    f"[{sample}] ⚠️ use_cls_loss=True but no label data found. Skipping classification loss.")
                df_labels = None

            if df_labels is not None:
                labels = torch.tensor(
                    standardize_pathways(df_labels, method="robust_z").values,
                    dtype=torch.float32,
                )
                print(f"[{sample}] Attached {labels.shape[1]} label dimensions.")

        # --- Generate overlapping subgraphs ---
        subgraphs = generate_overlapping_subgraphs(
            full_graph, coords, subgraph_size, stride)
        for sg in subgraphs:
            if labels is not None:
                sg.ndata["label"] = labels[sg.ndata[dgl.NID].numpy()]
            sg = dgl.add_self_loop(sg)
            graphs.append(sg)

    if len(graphs) == 0:
        raise RuntimeError(
            "No graphs were built. Check input data and indices.")
    return graphs


def finetune_gcn(
    graphs: List[dgl.DGLGraph],
    pretrained_gcn: str,
    save_dir: pl.Path,
    device: torch.device,
    hidden_dim: int = 10,
    num_layers: int = 2,
    node_mask_ratio: float = 0.9,
    epochs: int = 10,
    batch_size: int = 2,
    lr: float = 1e-4,
    lambda_reg: float = 1e-3,
    lambda_cls: float = 1.0,
    use_cls_loss: bool = True,
    use_huber: bool = True,
) -> torch.nn.Module:
    """
    Fine-tune a pretrained GCN autoencoder on spatial graphs.

    The loss includes feature reconstruction, latent regularization,
    and optional pathway classification.

    Args:
        graphs: List of DGL graphs for training.
        pretrained_gcn: Path to pretrained GCN checkpoint.
        save_dir: Directory to save fine-tuned model and loss history.
        device: Torch device for training.
        hidden_dim: Hidden layer dimensionality.
        num_layers: Number of GCN layers.
        node_mask_ratio: Fraction of masked nodes.
        epochs: Number of training epochs.
        batch_size: Graph batch size.
        lr: Learning rate.
        lambda_reg: Weight for latent regularization loss.
        lambda_cls: Weight for classification loss.
        use_cls_loss: Whether to include classification loss.
        use_huber: Whether to use Huber loss for classification.

    Returns:
        Fine-tuned GCNAutoencoder model.
    """
    in_dim = graphs[0].ndata["feat"].shape[1]
    n_classes = graphs[0].ndata["label"].shape[1] if use_cls_loss and "label" in graphs[0].ndata else 0

    loader = GraphDataLoader(graphs, batch_size=batch_size, shuffle=True)
    model = GCNAutoencoder(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=in_dim,
        node_mask_ratio=node_mask_ratio,
        num_layers=num_layers,
        n_classes=n_classes,
    ).to(device)
    state = torch.load(pretrained_gcn, map_location=device)
    model.load_state_dict({k: v for k, v in state.items()
                          if k in model.state_dict()}, strict=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse, huber = nn.MSELoss(), nn.SmoothL1Loss(beta=0.5)
    hist = {"total": [], "feat": [], "cls": [], "reg": []}

    for epoch in range(1, epochs + 1):
        tot = feat = cls = reg = 0
        for batch in tqdm(loader, desc=f"GCN Epoch {epoch}/{epochs}", leave=False):
            batch = batch.to(device)
            x_recon, x_true, mask, z, logits = model(batch)
            loss_feat = mse(x_recon[mask], x_true[mask])
            loss_reg = (z ** 2).mean()
            loss_cls = 0
            if use_cls_loss and "label" in batch.ndata:
                targets = batch.ndata["label"]
                loss_cls = huber(logits, targets) if use_huber else mse(
                    logits, targets)
            loss = loss_feat + lambda_reg * loss_reg + lambda_cls * loss_cls

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tot += loss.item()
            feat += loss_feat.item()
            reg += loss_reg.item()
            cls += float(loss_cls)

        n = len(loader)
        for k, v in zip(["total", "feat", "cls", "reg"], [tot, feat, cls, reg]):
            hist[k].append(v / n)
        print(f"[GCN {epoch}] total={tot/n:.4f}")

    torch.save(model.state_dict(), save_dir / "model.pt")
    pd.DataFrame(hist).to_csv(save_dir / "loss.csv", index=False)
    return model


# ---------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------

def finetune_models(
    samples: List[str],
    base_path: Optional[str] = None,
    pretrained_ae: str = "",
    pretrained_gcn: str = "",
    save_dir: str = "./finetuned_outputs",
    preloaded_data: Optional[Dict[str,
                                  Tuple[pd.DataFrame, pd.DataFrame]]] = None,
    adatas: Optional[Dict[str, Any]] = None,
    preloaded_pathway_data: Optional[Dict[str, pd.DataFrame]] = None,
    # AE parameters
    latent_dim: int = 64,
    enc_hidden_dims: List[int] = [64],
    dec_hidden_dims: List[int] = [64],
    ae_epochs: int = 5,
    ae_batch_size: int = 128,
    ae_lr: float = 1e-4,
    ae_weight_decay: float = 0.0,
    ae_grad_clip: Optional[float] = 1.0,
    lambda_recon1: float = 0.5,
    lambda_recon2: float = 0.5,
    lambda_cross12: float = 0.25,
    lambda_cross21: float = 0.25,
    lambda_align: float = 1.0,
    # Graph + GCN parameters
    knn_k: int = 30,
    subgraph_size: int = 5000,
    stride: int = 2500,
    gcn_hidden_dim: int = 10,
    gcn_num_layers: int = 2,
    node_mask_ratio: float = 0.9,
    gcn_epochs: int = 10,
    gcn_batch_size: int = 2,
    gcn_lr: float = 1e-4,
    lambda_reg: float = 1e-3,
    lambda_cls: float = 1.0,
    use_cls_loss: bool = True,
    use_huber: bool = True,
    spatial_key: str = "spatial",
):
    """
    End-to-end fine-tuning pipeline for SpatialFusion models.

    This function fine-tunes a paired autoencoder followed by a GCN,
    using either disk-based inputs or fully preloaded data.

    Args:
        samples: List of sample identifiers.
        base_path: Root directory for disk-based data loading.
        pretrained_ae: Path to pretrained AE checkpoint.
            If omitted, the packaged pretrained AE checkpoint is used.
        pretrained_gcn: Path to pretrained GCN checkpoint.
            If omitted, the packaged pretrained GCN checkpoint is used.
        save_dir: Output directory for fine-tuned models and logs.
        preloaded_data: Optional in-memory feature data per sample.
        adatas: Optional preloaded AnnData objects.
        preloaded_pathway_data: Optional pathway activation labels.
        latent_dim: AE latent dimensionality.
        enc_hidden_dims: Encoder hidden layer sizes.
        dec_hidden_dims: Decoder hidden layer sizes.
        ae_epochs: Number of AE fine-tuning epochs.
        ae_batch_size: AE batch size.
        ae_lr: AE learning rate.
        ae_weight_decay: AE weight decay.
        ae_grad_clip: AE gradient clipping norm.
        lambda_recon1: AE reconstruction loss weight (modality 1).
        lambda_recon2: AE reconstruction loss weight (modality 2).
        lambda_cross12: AE cross reconstruction loss weight (1→2).
        lambda_cross21: AE cross reconstruction loss weight (2→1).
        lambda_align: AE latent alignment loss weight.
        knn_k: Number of neighbors for spatial graph construction.
        subgraph_size: Size of spatial subgraphs.
        stride: Stride between subgraphs.
        gcn_hidden_dim: GCN hidden layer size.
        gcn_num_layers: Number of GCN layers.
        node_mask_ratio: GCN node masking ratio.
        gcn_epochs: Number of GCN fine-tuning epochs.
        gcn_batch_size: GCN batch size.
        gcn_lr: GCN learning rate.
        lambda_reg: GCN latent regularization weight.
        lambda_cls: GCN classification loss weight.
        use_cls_loss: Whether to use pathway classification loss.
        use_huber: Whether to use Huber loss for classification.
        spatial_key: Key for spatial coordinates in AnnData.

    Returns:
        A tuple `(ae_model, gcn_model)` containing the fine-tuned models.
    """
    if not pretrained_ae:
        pretrained_ae = resolve_pkg_ckpt(
            f"checkpoint_dir_ae/{DEFAULT_AE_CKPT_RELPATH}"
        )
        print(f"[finetune_models] Using packaged pretrained AE checkpoint: {pretrained_ae}")
    if not pretrained_gcn:
        pretrained_gcn = resolve_pkg_ckpt(
            f"checkpoint_dir_gcn/{DEFAULT_GCN_CKPT_RELPATH}"
        )
        print(f"[finetune_models] Using packaged pretrained GCN checkpoint: {pretrained_gcn}")

    SAVE_DIR = pl.Path(save_dir)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()

    # --------------------------
    # 1️⃣ Build AE dataset
    # --------------------------
    ae_loader, d1_dim, d2_dim = build_ae_dataset(
        samples=samples,
        base_path=base_path,
        preloaded_data=preloaded_data,
        batch_size=ae_batch_size,
    )

    # --------------------------
    # 2️⃣ Fine-tune AE
    # --------------------------
    ae_model = finetune_autoencoder(
        loader=ae_loader,
        d1_dim=d1_dim,
        d2_dim=d2_dim,
        pretrained_ae=pretrained_ae,
        save_dir=SAVE_DIR,
        device=device,
        latent_dim=latent_dim,
        enc_hidden_dims=enc_hidden_dims,
        dec_hidden_dims=dec_hidden_dims,
        epochs=ae_epochs,
        lr=ae_lr,
        weight_decay=ae_weight_decay,
        grad_clip=ae_grad_clip,
        lambda_recon1=lambda_recon1,
        lambda_recon2=lambda_recon2,
        lambda_cross12=lambda_cross12,
        lambda_cross21=lambda_cross21,
        lambda_align=lambda_align,
    )

    # --------------------------
    # 3️⃣ Extract embeddings
    # --------------------------
    if preloaded_data is not None:
        z1_all, z2_all, zjoint_all = [], [], []
        for sample_name in samples:
            feat1, feat2 = preloaded_data[sample_name]
            adata = adatas[sample_name] if adatas and sample_name in adatas else None
            z1_df, z2_df, z_joint_df = ae_from_arrays_finetune(
                ae_model, feat1, feat2, adata, device=device
            )
            z1_all.append(z1_df)
            z2_all.append(z2_df)
            zjoint_all.append(z_joint_df)

        # merge all samples into global frames
        z1_df = pd.concat(z1_all)
        z2_df = pd.concat(z2_all)
        z_joint_df = pd.concat(zjoint_all)

        # placeholders for metadata
        celltypes_sr = pd.Series([], dtype=str)
        samples_sr = pd.Series([], dtype=str)

    else:
        z1_df, z2_df, z_joint_df, celltypes_sr, samples_sr = extract_embeddings_for_all_samples(
            ae_model, samples, pl.Path(base_path), device=device,
        )

    # --------------------------
    # 4️⃣ Build graphs
    # --------------------------
    graphs = build_graphs(
        samples=samples,
        z_joint_df=z_joint_df,
        base_path=base_path,
        adatas=adatas,
        pathway_data=preloaded_pathway_data if 'preloaded_pathway_data' in locals() else None,
        knn_k=knn_k,
        subgraph_size=subgraph_size,
        stride=stride,
        use_cls_loss=use_cls_loss,
        spatial_key=spatial_key,
    )

    # --------------------------
    # 5️⃣ Fine-tune GCN
    # --------------------------
    gcn_out_dir = SAVE_DIR / "gcn_finetuned"
    gcn_out_dir.mkdir(exist_ok=True)
    gcn_model = finetune_gcn(
        graphs=graphs,
        pretrained_gcn=pretrained_gcn,
        save_dir=gcn_out_dir,
        device=device,
        hidden_dim=gcn_hidden_dim,
        num_layers=gcn_num_layers,
        node_mask_ratio=node_mask_ratio,
        epochs=gcn_epochs,
        batch_size=gcn_batch_size,
        lr=gcn_lr,
        lambda_reg=lambda_reg,
        lambda_cls=lambda_cls,
        use_cls_loss=use_cls_loss,
        use_huber=use_huber,
    )

    print("✓ Finetuning complete.")
    return ae_model, gcn_model
