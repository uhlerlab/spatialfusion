"""
Utility functions for extracting GCN embeddings and metadata for downstream analysis.

This module provides:
- extract_gcn_embeddings_with_metadata: Extracts GCN embeddings and merges with cell type, spatial, and ligand-receptor metadata.
"""
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pathlib as pl
import scanpy as sc
import dgl
from typing import Optional, Dict


@torch.no_grad()
def extract_gcn_embeddings_with_metadata(
    model,
    graphs,
    sample_list,
    base_path,
    z_joint: pd.DataFrame,
    device: str = "cuda:0",
    spatial_key: str = "spatial_px",
    celltype_key: str = "celltypes",
    adata_by_sample: Optional[Dict[str, sc.AnnData]] = None,
) -> pd.DataFrame:
    """
    Extracts GCN embeddings along with metadata such as spatial coordinates and cell types.
    Supports both in-memory AnnData inputs and on-disk loading via base_path.

    Args:
        model (torch.nn.Module):
            Trained GCN model with an `encode` method that takes (graph, features) as input.
        graphs (list[dgl.DGLGraph]):
            List of DGL graphs, one per sample, containing node features in `ndata['feat']`.
        sample_list (list[str]):
            List of sample IDs corresponding to the graphs.
        base_path (str or Path):
            Root directory containing per-sample subdirectories. Used only if `adata_by_sample`
            is not provided.
        z_joint (pd.DataFrame):
            Joint embeddings from the autoencoder step. Used to align indices.
        device (str):
            Device string (e.g., `'cuda:0'` or `'cpu'`) for model inference.
        spatial_key (str):
            Key name for spatial coordinates stored in `adata.obsm`.
        celltype_key (str):
            Column name in `adata.obs` or celltypes.csv for cell type annotation.
        adata_by_sample (Optional[Dict[str, sc.AnnData]]):
            Optional mapping from sample IDs to AnnData objects already loaded in memory.
            If provided, this takes precedence over disk loading.

    Returns:
        pd.DataFrame:
            A concatenated DataFrame of GCN embeddings across all samples, including metadata:
            - sample_id, cell_id
            - celltype (and optional subtype/niche labels)
            - spatial coordinates (X_coord, Y_coord)
            - optional ligand–receptor features if present
    """
    model.eval()
    all_dfs = []

    # Normalize base path if used
    base_path = None if base_path in (None, "", ".") else pl.Path(base_path)

    for g, sample in tqdm(zip(graphs, sample_list), total=len(sample_list), desc="Running GCN inference"):
        # --- Load AnnData ---
        if adata_by_sample is not None and sample in adata_by_sample:
            adata = adata_by_sample[sample]
            datapath = (base_path / sample) if base_path is not None else None
        else:
            if base_path is None:
                raise FileNotFoundError(
                    f"AnnData for sample '{sample}' not provided in-memory and base_path is None."
                )
            datapath = base_path / sample
            adata = sc.read_h5ad(datapath / "adata.h5ad")

        # --- Forward pass through GCN ---
        g = g.to(device)
        x = g.ndata["feat"].to(device)
        z = model.encode(dgl.add_self_loop(g), x)
        p_drop = getattr(model, "dropout", 0.0)
        z = F.dropout(z, p=p_drop, training=False)
        z_np = z.detach().cpu().numpy()

        # --- Construct DataFrame ---
        cell_ids = adata.obs_names.astype(str).intersection(z_joint.index)
        if len(cell_ids) != g.num_nodes():
            raise ValueError(
                f"[{sample}] node/cell mismatch: {g.num_nodes()} nodes vs {len(cell_ids)} matched cells."
            )

        df = pd.DataFrame(z_np, index=cell_ids)
        df["sample_id"] = sample
        df["cell_id"] = cell_ids

        # --- Cell type annotations ---
        added_celltype = False
        if datapath is not None:
            celltypes_path = datapath / "celltypes.csv"
            if celltypes_path.exists():
                cdf = pd.read_csv(celltypes_path, index_col=0)
                cdf.index = cdf.index.astype(str)
                if celltype_key in cdf.columns:
                    df["celltype"] = cdf.reindex(df.index)[celltype_key].values
                    added_celltype = True

        if not added_celltype and celltype_key in adata.obs.columns:
            df["celltype"] = adata.obs.loc[df.index, celltype_key].values
            added_celltype = True

        if not added_celltype:
            df["celltype"] = "unknown"

        # --- Optional extra obs columns ---
        for key in ["cellsubtypes", "CNiche", "TNiche"]:
            if key in adata.obs.columns:
                df[key] = adata.obs.loc[df.index, key].values

        # --- Spatial coordinates ---
        if spatial_key in adata.obsm:
            coords = adata.obsm[spatial_key]
            spatial_df = pd.DataFrame(coords, index=adata.obs_names, columns=[
                                      "X_coord", "Y_coord"])
            df[["X_coord", "Y_coord"]] = spatial_df.loc[df.index,
                                                        ["X_coord", "Y_coord"]].values
        else:
            df[["X_coord", "Y_coord"]] = np.nan

        # --- Ligand–receptor matrix (optional) ---
        if datapath is not None:
            lr_path = datapath / "training_LR_matrix.csv"
            if lr_path.exists():
                lr_df = pd.read_csv(lr_path, index_col=0)
                lr_df.index = lr_df.index.astype(str)
                lr_df = lr_df.reindex(df.index)
                df = pd.concat([df, lr_df], axis=1)

        all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True)
