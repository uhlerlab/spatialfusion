"""Autoencoder (AE) and Graph Convolutional Network (GCN) embedding pipeline
for SpatialFusion.

This module provides utilities for:
- Running a paired autoencoder on UNI / scGPT embeddings
- Combining embeddings across modalities
- Constructing spatial graphs
- Running a GCN to produce final embeddings
- Orchestrating the full end-to-end embedding workflow
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from tqdm import tqdm

from spatialfusion.models.multi_ae import PairedAE
from spatialfusion.models.gcn import GCNAutoencoder
from spatialfusion.utils.embed_ae_utils import (
    extract_embeddings_for_all_samples,   # still available if you want disk-based AE
    save_embeddings_separately,
    safe_standardize,
)
from spatialfusion.utils.embed_gcn_utils import extract_gcn_embeddings_with_metadata
from spatialfusion.utils.gcn_utils import build_knn_graph
from spatialfusion.utils.pkg_ckpt import resolve_pkg_ckpt

# ---------------------------
# Small utilities
# ---------------------------

DEFAULT_AE_CKPT_RELPATH = "spatialfusion-ae-uni-scgpt.pt"
DEFAULT_GCN_CKPT_RELPATH = "spatialfusion-gcn-uni-scgpt.pt"
DEFAULT_HE_ONLY_GCN_CKPT_RELPATH = "spatialfusion-he-gcn-uni-scgpt.pt"
DEFAULT_RNA_ONLY_GCN_CKPT_RELPATH = "spatialfusion-rna-gcn-uni-scgpt.pt"


def _default_gcn_ckpt_relpath(combine_mode: str) -> str:
    """Return the packaged GCN checkpoint appropriate for a fusion mode."""
    mode = combine_mode.lower()
    if mode == "z1":
        return DEFAULT_HE_ONLY_GCN_CKPT_RELPATH
    if mode == "z2":
        return DEFAULT_RNA_ONLY_GCN_CKPT_RELPATH
    return DEFAULT_GCN_CKPT_RELPATH


def _combine_embeddings(z1: pd.DataFrame, z2: pd.DataFrame, mode: Literal["average", "concat", "z1", "z2", "gated"]) -> pd.DataFrame:
    """
    Combine two embedding matrices according to a specified mode.

    Args:
        z1: First embedding DataFrame (cells × features).
        z2: Second embedding DataFrame (cells × features).
        mode: Combination strategy.

    Returns:
        Combined embedding DataFrame.

    Raises:
        ValueError: If the mode is invalid or embeddings are incompatible.
    """
    mode = mode.lower()
    if mode not in {"average", "concat", "z1", "z2", "gated"}:
        raise ValueError(
            "combine_mode must be one of: 'average', 'concat', 'z1', 'z2', 'gated'")
    if mode == "z1":
        return z1.copy()
    if mode == "z2":
        return z2.copy()

    common_idx = z1.index.intersection(z2.index)
    if len(common_idx) == 0:
        raise ValueError(
            f"{mode} mode: z1 and z2 have no overlapping cells (index).")
    z1c = z1.loc[common_idx]
    z2c = z2.loc[common_idx]

    if mode in ["concat", "gated"]:
        z1c = z1c.copy()
        z2c = z2c.copy()
        z1c.columns = [f"z1_{c}" for c in z1c.columns]
        z2c.columns = [f"z2_{c}" for c in z2c.columns]
        return pd.concat([z1c, z2c], axis=1)

    # average
    shared_cols = [c for c in z1c.columns if c in set(z2c.columns)]
    if not shared_cols:
        raise ValueError(
            "average mode requires overlapping columns between z1 and z2.")
    return (z1c[shared_cols] + z2c[shared_cols]) / 2.0


def _read_tabular_one_row(path: Union[str, Path]) -> pd.DataFrame:
    """
    Read a single-row tabular embedding file.

    Args:
        path: Path to a CSV or Parquet file.

    Returns:
        DataFrame containing the first row.
    """
    p = str(path)
    if p.endswith(".csv"):
        return pd.read_csv(p, index_col=0, nrows=1)
    return pd.read_parquet(p, engine="pyarrow")


def infer_input_dims_from_files(uni_path: Union[str, Path], scgpt_path: Union[str, Path]) -> Tuple[int, int]:
    """
    Infer embedding dimensions from UNI and scGPT files.

    Args:
        uni_path: Path to UNI embedding file.
        scgpt_path: Path to scGPT embedding file.

    Returns:
        Tuple of (UNI dimension, scGPT dimension).
    """
    uni = _read_tabular_one_row(uni_path)
    scgpt = _read_tabular_one_row(scgpt_path)
    return uni.shape[1], scgpt.shape[1]


def infer_input_dims(sample_list: Iterable[str], base_path: Union[str, Path],
                     uni_path: Optional[Union[str, Path]] = None,
                     scgpt_path: Optional[Union[str, Path]] = None) -> Tuple[int, int]:
    """
    Infer AE input dimensions from disk.

    Args:
        sample_list: Iterable of sample identifiers.
        base_path: Base directory containing sample subfolders.
        uni_path: Optional explicit UNI file path.
        scgpt_path: Optional explicit scGPT file path.

    Returns:
        Tuple of (UNI dimension, scGPT dimension).

    Raises:
        ValueError: If no valid embeddings are found.
    """
    if uni_path and scgpt_path:
        return infer_input_dims_from_files(uni_path, scgpt_path)

    base_path = Path(base_path)
    for sample in sample_list:
        embed_dir = base_path / sample / "embeddings"
        up = next((embed_dir / f"UNI{ext}" for ext in [".csv", ".parquet"] if (
            embed_dir / f"UNI{ext}").exists()), None)
        sp = next((embed_dir / f"scGPT{ext}" for ext in [".csv", ".parquet"] if (
            embed_dir / f"scGPT{ext}").exists()), None)
        if up and sp:
            try:
                return infer_input_dims_from_files(up, sp)
            except Exception as e:
                print(f"Skipping {sample} due to read error: {e}")
                continue
    raise ValueError(
        "No valid samples found with both UNI and scGPT embeddings (.csv or .parquet).")


# ---------------------------
# AE: modular API
# ---------------------------

@dataclass
class AEInputs:
    """
    Container for in-memory autoencoder inputs.

    Attributes:
        adata: AnnData object containing spatial metadata.
        z_he: Optional H&E foundation model embeddings indexed by cell
            (UNI or Virchow).
        z_rna: Optional RNA foundation model embeddings indexed by cell
            (scGPT or Nicheformer).
    """
    adata: sc.AnnData
    z_he: Optional[pd.DataFrame] = None   # ← allow None
    z_rna: Optional[pd.DataFrame] = None   # ← allow None


def load_paired_ae(ae_ckpt: Union[str, Path], d1_dim: int, d2_dim: int,
                   latent_dim: int = 64, device: str = "cuda:0") -> PairedAE:
    """
    Load a pretrained PairedAE model from disk.

    Args:
        ae_ckpt: Path to the AE checkpoint.
        d1_dim: Input dimension of modality 1.
        d2_dim: Input dimension of modality 2.
        latent_dim: Latent dimension size.
        device: Torch device string.

    Returns:
        Loaded PairedAE model in evaluation mode.
    """
    model = PairedAE(d1_dim, d2_dim, latent_dim=latent_dim)
    state = torch.load(ae_ckpt, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model

# 4) build joint according to combine_mode


def _combine(z1: pd.DataFrame, z2: pd.DataFrame, mode: str) -> pd.DataFrame:
    mode = mode.lower()
    if mode == "z1":
        if z1.empty:
            raise ValueError("combine_mode='z1' but z1 is not available.")
        return z1.copy()
    if mode == "z2":
        if z2.empty:
            raise ValueError("combine_mode='z2' but z2 is not available.")
        return z2.copy()
    if z1.empty or z2.empty:
        raise ValueError(
            f"combine_mode='{mode}' requires both z1 and z2, but one is missing.")

    common_idx = z1.index.intersection(z2.index)
    z1c, z2c = z1.loc[common_idx], z2.loc[common_idx]

    if mode == "concat":
        z1c = z1c.copy()
        z2c = z2c.copy()
        z1c.columns = [f"z1_{c}" for c in z1c.columns]
        z2c.columns = [f"z2_{c}" for c in z2c.columns]
        return pd.concat([z1c, z2c], axis=1)

    # average
    shared = [c for c in z1c.columns if c in set(z2c.columns)]
    if not shared:
        raise ValueError(
            "average mode requires overlapping columns between z1 and z2.")
    return (z1c[shared] + z2c[shared]) / 2.0


def ae_from_arrays(
    model: PairedAE,
    inputs: AEInputs,
    device: str = "cuda:0",
    combine_mode: Literal["average", "concat",
                          "z1", "z2", "gated"] = "average",
    batch_size: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run a pretrained PairedAE on in-memory UNI and scGPT embeddings.

    This function standardizes the input embeddings using the same preprocessing
    applied during training, runs the paired autoencoder, and combines the
    resulting latent representations according to `combine_mode`.

    Args:
        model: Pretrained PairedAE model.
        inputs: In-memory inputs containing AnnData and modality embeddings.
        device: Torch device used for inference.
        combine_mode: Strategy for combining modality embeddings.
            One of {"average", "concat", "z1", "z2", "gated"}.
        batch_size: Optional batch size for processing. If None, it will be auto-determined based on input size and available memory.

    Returns:
        A tuple `(z1_df, z2_df, z_joint_df)` where:
            - z1_df: Latent embeddings for UNI modality (cells × latent_dim),
              or empty if not produced.
            - z2_df: Latent embeddings for scGPT modality (cells × latent_dim),
              or empty if not produced.
            - z_joint_df: Combined latent embedding according to `combine_mode`.

        All DataFrames are indexed by cell IDs aligned to `inputs.adata.obs_names`.

    Raises:
        ValueError: If required embeddings are missing or no overlapping cells
            are found between AnnData and embeddings.
    """
    # determine which inputs are required based on combine_mode
    needs_z1 = (combine_mode in ["average", "concat", "z1", "gated"])
    needs_z2 = (combine_mode in ["average", "concat", "z2", "gated"])

    idx = inputs.adata.obs_names.astype(str)

    common = idx
    if needs_z1:
        if inputs.z_he is None:
            raise ValueError(
                "combine_mode requires H&E input, but z_he=None")
        common = common.intersection(inputs.z_he.index)
    if needs_z2:
        if inputs.z_rna is None:
            raise ValueError(
                "combine_mode requires RNA input, but z_rna=None")
        common = common.intersection(inputs.z_rna.index)

    if len(common) == 0:
        raise ValueError(
            "No overlapping cells between adata and required embeddings.")

    # --- Standardize before feeding to AE
    if needs_z1:
        x1_df = safe_standardize(inputs.z_he.loc[common])
        x1_np = x1_df.astype(np.float32).values
    else:
        x1_np = None

    if needs_z2:
        x2_df = safe_standardize(inputs.z_rna.loc[common])
        x2_np = x2_df.astype(np.float32).values
    else:
        x2_np = None

    n_samples = len(common)

    # Auto-determine batch size if not provided
    if batch_size is None:
        est_mem_per_sample = 0
        if needs_z1:
            est_mem_per_sample += x1_np.shape[1] * 4 * 2
        if needs_z2:
            est_mem_per_sample += x2_np.shape[1] * 4 * 2
        batch_size = max(1, int(300 * 1024 * 1024 / est_mem_per_sample))
        batch_size = min(batch_size, 5000)
        batch_size = max(batch_size, 100)

    if n_samples > batch_size:
        z1_list, z2_list = [], []

        model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, n_samples, batch_size), desc="Processing AE in batches"):
                end_idx = min(i + batch_size, n_samples)

                if needs_z1:
                    batch_x1_np = x1_np[i:end_idx]
                    x1_batch = torch.from_numpy(batch_x1_np).to(device)
                else:
                    x1_batch = None

                if needs_z2:
                    batch_x2_np = x2_np[i:end_idx]
                    x2_batch = torch.from_numpy(batch_x2_np).to(device)
                else:
                    x2_batch = None

                out = model(x1_batch, x2_batch)

                z1_t = out.get("z1")
                z2_t = out.get("z2")

                if z1_t is not None:
                    z1_list.append(z1_t.cpu().numpy())
                if z2_t is not None:
                    z2_list.append(z2_t.cpu().numpy())

                del x1_batch, x2_batch, out, z1_t, z2_t
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()

        z1_np = np.vstack(z1_list) if z1_list else np.array([])
        z2_np = np.vstack(z2_list) if z2_list else np.array([])

        z1_df = pd.DataFrame(z1_np, index=common) if len(
            z1_np) > 0 else pd.DataFrame(index=common)
        z2_df = pd.DataFrame(z2_np, index=common) if len(
            z2_np) > 0 else pd.DataFrame(index=common)

    else:
        with torch.no_grad():
            x1 = torch.from_numpy(x1_np).to(device) if needs_z1 else None
            x2 = torch.from_numpy(x2_np).to(device) if needs_z2 else None

            out = model(x1, x2)
            z1_t = out.get("z1")
            z2_t = out.get("z2")

        z1_df = pd.DataFrame(z1_t.cpu().numpy(
        ), index=common) if z1_t is not None else pd.DataFrame(index=common)
        z2_df = pd.DataFrame(z2_t.cpu().numpy(
        ), index=common) if z2_t is not None else pd.DataFrame(index=common)

    z_joint_df = _combine_embeddings(z1_df, z2_df, combine_mode)
    return z1_df, z2_df, z_joint_df


def ae_from_disk_for_samples(
    model: PairedAE,
    sample_list: Iterable[str],
    base_path: Union[str, Path],
    device: str = "cuda:0",
    combine_mode: Literal["average", "concat", "z1", "z2"] = "average",
    batch_size: Optional[int] = None,
    save_dir: Optional[Union[str, Path]] = None,
):
    """
    Run the paired autoencoder using disk-based inputs for multiple samples.

    This function loads UNI and scGPT embeddings from disk for each sample,
    runs the PairedAE model, combines modality embeddings, and optionally
    saves the outputs to disk.

    Args:
        model: Pretrained PairedAE model.
        sample_list: Iterable of sample identifiers.
        base_path: Base directory containing sample subfolders.
        device: Torch device used for inference.
        combine_mode: Strategy for combining modality embeddings.
            One of {"average", "concat", "z1", "z2"}. Note: "gated" is not
            supported for disk-based loading; use the in-memory path
            (``ae_from_arrays``) if you need the "gated" mode.
        batch_size: Optional batch size for AE inference from disk.
        save_dir: Optional directory in which to save AE outputs.

    Returns:
        A tuple `(z1_df, z2_df, z_joint_df)` containing:
            - z1_df: UNI latent embeddings.
            - z2_df: scGPT latent embeddings.
            - z_joint_df: Combined latent embeddings.
    """
    z1, z2, z_joint_unused, celltypes, samples = extract_embeddings_for_all_samples(
        model, sample_list, base_path, device, batch_size=batch_size
    )
    if combine_mode in {"z1", "z2"}:
        z_joint = _combine_embeddings(z1, z2, combine_mode)
    else:
        z_joint = _combine_embeddings(z1, z2, combine_mode)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_embeddings_separately(
            z1, z2, z_joint, celltypes, samples, save_dir, "ae_outputs")

    return z1, z2, z_joint


# ---------------------------
# GCN: modular API
# ---------------------------

@dataclass
class GCNInputs:
    """
    Container for inputs to the GCN stage.

    Attributes:
        z_joint: Joint AE embedding indexed by cell ID.
        adata_by_sample: Mapping from sample ID to AnnData objects,
            each containing spatial coordinates and metadata.
    """
    z_joint: pd.DataFrame                        # indexed by cell, columns = features
    # each AnnData contains obs_names that intersect z_joint.index
    adata_by_sample: Dict[str, sc.AnnData]


def load_gcn(gcn_ckpt: Union[str, Path], in_dim: int, hidden_dim: int = 10,
             num_layers: int = 2, combine_mode='average',
             device: str = "cuda:0") -> GCNAutoencoder:
    """
    Load a pretrained GCN autoencoder from disk.

    Args:
        gcn_ckpt: Path to the GCN checkpoint file.
        in_dim: Input feature dimensionality.
        hidden_dim: Hidden layer dimensionality of the GCN. Must match the
            value used when the checkpoint was trained (default: 10).
        num_layers: Number of GCN layers. Must match the checkpoint (default: 2).
        combine_mode: Embedding combination strategy the GCN was trained with.
            One of {"average", "concat", "z1", "z2", "gated"}.
            Only use gated mode with checkpoints trained with gated fusion
            parameters (not provided with the package).
        device: Torch device on which to load the model.

    Returns:
        A GCNAutoencoder instance in evaluation mode.
    """
    model = GCNAutoencoder(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=in_dim,
        node_mask_ratio=0.9,
        num_layers=num_layers,
        n_classes=0,
        combine_mode=combine_mode,
    ).to(device)
    state = torch.load(gcn_ckpt, map_location=device)
    state = {k: v for k, v in state.items() if not k.startswith("classifier.")}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def graphs_from_embeddings_and_adata(
    z_joint: pd.DataFrame,
    adata_by_sample: Dict[str, sc.AnnData],
    spatial_key: str = "spatial",
    k: int = 30,
) -> Tuple[List["dgl.DGLGraph"], List[str]]:
    """
    Construct spatial KNN graphs from joint embeddings and AnnData objects.

    For each sample, this function:
    - Aligns cells between `z_joint` and AnnData
    - Standardizes joint embeddings
    - Builds a k-nearest-neighbor graph using spatial coordinates
    - Attaches node features to the graph

    Args:
        z_joint: Joint AE embeddings indexed by cell ID.
        adata_by_sample: Mapping from sample ID to AnnData.
        spatial_key: Key in ``adata.obsm`` containing spatial coordinates
            (default: ``"spatial"``). Set this to whichever key holds the
            X/Y centroid coordinates in your AnnData.
        k: Number of nearest neighbors for graph construction.

    Returns:
        A tuple `(graphs, keep_samples)` where:
            - graphs: List of DGL graphs, one per retained sample.
            - keep_samples: List of sample IDs corresponding to the graphs.
    """
    import dgl  # lazy import to keep this modular

    graphs: List[dgl.DGLGraph] = []
    keep_samples: List[str] = []

    for sample, adata in tqdm(adata_by_sample.items(), desc="Building graphs"):
        new_idx = z_joint.index.intersection(adata.obs_names.astype(str))
        if len(new_idx) == 0:
            print(f"[{sample}] No index overlap. Skipping.")
            continue

        # features
        joint_emb = z_joint.loc[new_idx]
        joint_emb = safe_standardize(joint_emb, fill_value=0.0, min_std=1e-5)
        feats = torch.tensor(joint_emb.astype(np.float32).values)

        # spatial graph
        coords = adata[new_idx].obsm[spatial_key]
        coords = (coords - coords.mean(axis=0)) / coords.std(axis=0)
        g = build_knn_graph(coords, k=k)
        g.ndata["feat"] = feats
        graphs.append(g)
        keep_samples.append(sample)

    return graphs, keep_samples


def gcn_embeddings_from_joint(
    gcn_model: GCNAutoencoder,
    z_joint: pd.DataFrame,
    adata_by_sample: Dict[str, sc.AnnData],
    base_path: Union[str, Path],
    device: str = "cuda:0",
    spatial_key: str = "spatial",
    celltype_key: str = "celltypes",
    k: int = 30,
    batch_size: Optional[int] = None,
    k_hop: int = 2,
) -> pd.DataFrame:
    """
    Generate GCN embeddings from joint AE embeddings and spatial graphs.

    This function constructs spatial graphs for each sample and applies
    a pretrained GCN model to produce final embeddings with associated
    metadata.

    Args:
        gcn_model: Pretrained GCN autoencoder.
        z_joint: Joint AE embedding indexed by cell ID.
        adata_by_sample: Mapping from sample ID to AnnData.
        base_path: Base path used for metadata resolution.
        device: Torch device used for inference.
        spatial_key: Key in ``adata.obsm`` containing spatial coordinates
            (default: ``"spatial"``). Set this to whichever key holds the
            X/Y centroid coordinates in your AnnData.
        celltype_key: Key in ``adata.obs`` containing cell type annotations
            (default: ``"celltypes"``). Set this to the column name used in
            your AnnData.
        k: Number of neighbors for KNN graph construction.
        batch_size: Optional batch size for GCN inference.
        k_hop: Number of hops in the spatial graph for batching.

    Returns:
        DataFrame containing GCN embeddings and associated metadata.
    """
    full_graphs, ordered_samples = graphs_from_embeddings_and_adata(
        z_joint, adata_by_sample, spatial_key=spatial_key, k=k
    )
    emb_df = extract_gcn_embeddings_with_metadata(
        gcn_model, full_graphs, ordered_samples, Path(base_path), z_joint,
        device=device, spatial_key=spatial_key, celltype_key=celltype_key,
        adata_by_sample=adata_by_sample, batch_size=batch_size, k_hop=k_hop,
    )
    return emb_df


# ---------------------------
# Orchestration: flexible runner
# ---------------------------

def run_full_embedding(
    *,
    # either provide in-memory inputs...
    ae_inputs_by_sample: Optional[Dict[str, AEInputs]] = None,
    # ...or provide sample names + paths (disk-based)
    sample_list: Optional[Iterable[str]] = None,
    base_path: Optional[Union[str, Path]] = None,

    # models (paths or preloaded)
    ae_model_path: Optional[Union[str, Path]] = None,
    gcn_model_path: Optional[Union[str, Path]] = None,
    ae_model: Optional[PairedAE] = None,
    gcn_model: Optional[GCNAutoencoder] = None,

    # dims / config
    latent_dim: int = 64,
    device: str = "cuda:0",
    spatial_key: str = "spatial_px",
    k: int = 30,
    celltype_key: str = "celltypes",
    combine_mode: Literal["average", "concat",
                          "z1", "z2", "gated"] = "average",
    hidden_dim: int = 10,
    num_layers: int = 2,

    ae_batch_size: Optional[int] = None,
    gcn_batch_size: Optional[int] = None,
    k_hop: int = 2,

    # optional explicit input-dim inference from files for disk path mode
    uni_path: Optional[Union[str, Path]] = None,
    scgpt_path: Optional[Union[str, Path]] = None,

    # outputs
    save_ae_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Run the full SpatialFusion embedding pipeline.

    This function supports two execution modes:
    1. In-memory mode using `ae_inputs_by_sample`
    2. Disk-based mode using `sample_list` and `base_path`

    In both cases, it:
    - Runs the paired autoencoder (AE)
    - Combines modality embeddings
    - Constructs spatial graphs
    - Runs the GCN to produce final embeddings

    Args:
        ae_inputs_by_sample: Optional in-memory AE inputs per sample.
        sample_list: Sample identifiers for disk-based execution.
        base_path: Base directory containing sample data.
        ae_model_path: Path to AE checkpoint (if AE model not provided).
            If omitted, the packaged pretrained AE checkpoint is used.
        gcn_model_path: Path to GCN checkpoint (if GCN model not provided).
            If omitted, the packaged checkpoint is selected from
            ``combine_mode``: H&E-only for ``"z1"``, RNA-only for ``"z2"``,
            and multimodal for paired fusion modes.
        ae_model: Optional preloaded AE model.
        gcn_model: Optional preloaded GCN model.
        latent_dim: Latent dimensionality of the AE.
        device: Torch device used for inference.
        spatial_key: Key in ``adata.obsm`` containing spatial coordinates
            (default: ``"spatial_px"``). Set this to whichever key holds the
            X/Y centroid coordinates in your AnnData (e.g. check with
            ``list(adata.obsm.keys())``).
        k: Number of neighbors for spatial graph construction.
        celltype_key: Key in ``adata.obs`` containing cell type annotations
            (default: ``"celltypes"``). Set this to the column name used in
            your AnnData (e.g. check with ``adata.obs.columns.tolist()``).
        combine_mode: Strategy for combining modality embeddings.
            Only use gated mode with checkpoints trained with gated fusion
            parameters (not provided with the package).
        ae_batch_size: Optional batch size for AE inference.
            If None, an effective batch size is auto-determined for AE
            processing.
        gcn_batch_size: Optional batch size for GCN inference.
            If None, GCN runs full-graph inference.
            If set, GCN uses memory-efficient subgraph batching.
        k_hop: Number of hops for subgraph expansion during batched GCN
            inference; used only when `gcn_batch_size` is set.
        uni_path: Optional UNI file path for dimension inference.
        scgpt_path: Optional scGPT file path for dimension inference.
        save_ae_dir: Optional directory to save AE outputs.

    Returns:
        DataFrame containing final GCN embeddings with metadata.
    """
    # Fallback to packaged pretrained AE checkpoint when no path is provided
    # and a checkpoint is needed for loading or shape inference.
    if ae_model_path is None and (ae_model is None or ae_inputs_by_sample is not None):
        ae_model_path = resolve_pkg_ckpt(
            f"checkpoint_dir_ae/{DEFAULT_AE_CKPT_RELPATH}"
        )
        print(
            f"[run_full_embedding] Using packaged pretrained AE checkpoint: {ae_model_path}")
    # Fallback to packaged pretrained GCN checkpoint when no path/model is provided.
    if gcn_model_path is None and gcn_model is None:
        gcn_ckpt_relpath = _default_gcn_ckpt_relpath(combine_mode)
        gcn_model_path = resolve_pkg_ckpt(
            f"checkpoint_dir_gcn/{gcn_ckpt_relpath}"
        )
        print(
            f"[run_full_embedding] Using packaged pretrained GCN checkpoint: {gcn_model_path}")

    # --- AE stage ---
    if ae_inputs_by_sample is not None:
        # In-memory branch
        # infer dims from the first sample’s matrices
        first = next(iter(ae_inputs_by_sample.values()))
        tmp_state = None
        if first.z_he is not None:
            d1_dim = first.z_he.shape[1]
        else:
            # load temporarily to inspect expected input dim
            tmp_state = torch.load(ae_model_path, map_location="cpu")
            # encoder1 layers always start with a weight of shape (latent_dim, d1_dim)
            for key, v in tmp_state.items():
                if key.startswith("encoder1.model.0.weight"):
                    d1_dim = v.shape[1]
                    break
        if first.z_rna is not None:
            d2_dim = first.z_rna.shape[1]
        else:
            # load temporarily to inspect expected input dim
            if tmp_state is None:
                tmp_state = torch.load(ae_model_path, map_location="cpu")
            # encoder2 layers always start with a weight of shape (latent_dim, d2_dim)
            for key, v in tmp_state.items():
                if key.startswith("encoder2.model.0.weight"):
                    d2_dim = v.shape[1]
                    break

        if ae_model is None:
            if ae_model_path is None:
                raise ValueError("Provide ae_model or ae_model_path.")
            ae_model = load_paired_ae(
                ae_model_path, d1_dim, d2_dim, latent_dim=latent_dim, device=device)

        z1_all, z2_all, zjoint_all = [], [], []
        adata_by_sample: Dict[str, sc.AnnData] = {}

        for sample, inputs in ae_inputs_by_sample.items():
            z1, z2, z_joint = ae_from_arrays(
                ae_model, inputs, device=device, combine_mode=combine_mode, batch_size=ae_batch_size)
            # collect
            z1["sample"] = sample
            z2["sample"] = sample
            z_joint["sample"] = sample
            z1_all.append(z1)
            z2_all.append(z2)
            zjoint_all.append(z_joint)
            adata_by_sample[sample] = inputs.adata

        z1_df = pd.concat(z1_all).drop(columns=["sample"])
        z2_df = pd.concat(z2_all).drop(columns=["sample"])
        z_joint_df = pd.concat(zjoint_all).drop(columns=["sample"])

        if save_ae_dir is not None:
            save_dir = Path(save_ae_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            # celltypes/samples optional here; pass minimal placeholders
            celltypes = pd.Series(index=z_joint_df.index, dtype="category")
            samples = pd.Series(index=z_joint_df.index, dtype="category")
            save_embeddings_separately(
                z1_df, z2_df, z_joint_df, celltypes, samples, save_dir, "ae_outputs")

    else:
        # Disk-based branch
        if sample_list is None or base_path is None:
            raise ValueError(
                "When not providing ae_inputs_by_sample, you must provide sample_list and base_path.")
        d1_dim, d2_dim = infer_input_dims(
            sample_list, base_path, uni_path, scgpt_path)
        if ae_model is None:
            if ae_model_path is None:
                raise ValueError("Provide ae_model or ae_model_path.")
            ae_model = load_paired_ae(
                ae_model_path, d1_dim, d2_dim, latent_dim=latent_dim, device=device)

        z1_df, z2_df, z_joint_df = ae_from_disk_for_samples(
            ae_model,
            sample_list,
            base_path,
            device=device,
            combine_mode=combine_mode,
            batch_size=ae_batch_size,
            save_dir=save_ae_dir,
        )

        # read adatas for GCN stage
        adata_by_sample = {
            s: sc.read_h5ad(Path(base_path) / s / "adata.h5ad")
            for s in sample_list
            if (Path(base_path) / s / "adata.h5ad").exists()
        }

    # --- GCN stage ---
    if gcn_model is None:
        if gcn_model_path is None:
            raise ValueError("Provide gcn_model or gcn_model_path.")
        gcn_model = load_gcn(
            gcn_model_path, in_dim=z_joint_df.shape[1], hidden_dim=hidden_dim,
            num_layers=num_layers, combine_mode=combine_mode, device=device)

    emb_df = gcn_embeddings_from_joint(
        gcn_model=gcn_model,
        z_joint=z_joint_df,
        adata_by_sample=adata_by_sample,
        base_path=base_path if base_path is not None else ".",
        device=device,
        spatial_key=spatial_key,
        celltype_key=celltype_key,
        k=k,
        batch_size=gcn_batch_size,
        k_hop=k_hop,
    )
    return emb_df
