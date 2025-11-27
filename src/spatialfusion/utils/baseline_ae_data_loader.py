"""
Utility functions for loading and preprocessing baseline AE data (image and GEX modalities).

This module provides:
- reindex_adata_genes: Reindex AnnData to a target gene list, filling missing genes with zeros.
- load_and_preprocess_sample_baseline: Load, match, and preprocess paired image and GEX data for baseline AE training.
"""
import pathlib as pl
import pandas as pd
import warnings
import random
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm

import tifffile
import shapely.wkb
import scanpy as sc
import scipy.sparse

def reindex_adata_genes(adata, target_genes):
    """
    Reindex AnnData object to a target gene list, filling missing genes with zeros.

    Args:
        adata (AnnData): Input AnnData object.
        target_genes (list): List of target gene names.

    Returns:
        AnnData: Reindexed AnnData object with all target genes.
    """
    # Ensure adata is backed by dense or sparse matrix
    X_df = pd.DataFrame(
        adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X,
        index=adata.obs_names,
        columns=adata.var_names
    )

    # Reindex to target gene list — missing genes filled with 0s
    X_df = X_df.reindex(columns=target_genes, fill_value=0)

    # Create new AnnData
    new_adata = sc.AnnData(
        X=X_df.values,
        obs=adata.obs.copy(),
        var=pd.DataFrame(index=target_genes)
    )
    new_adata.obsm = adata.obsm.copy() if hasattr(adata, "obsm") else {}

    return new_adata


def load_and_preprocess_sample_baseline(sample_name, base_path, raw_path, SOFT_UNION_GENE_LIST, max_cells=30000, image_size=224):
    """
    Loads and preprocesses paired image and GEX data for baseline AE training.
    - Loads AnnData and reindexes to a common gene list.
    - Loads segmentation and WSI image.
    - Matches cells, extracts image patches and GEX vectors.
    - Applies normalization and standardization.

    Args:
        sample_name (str): Sample identifier.
        base_path (str or Path): Directory containing sample data.
        raw_path (str or Path): Directory containing raw segmentation and WSI data.
        SOFT_UNION_GENE_LIST (list): List of genes to use for GEX.
        max_cells (int): Maximum number of cells to sample.
        image_size (int): Size to resize image patches to.

    Returns:
        tuple: (img_tensor, gex_tensor, valid_ids)
            img_tensor (torch.Tensor): Image patches (N, 3, image_size, image_size).
            gex_tensor (torch.Tensor): Standardized GEX data (N, genes).
            valid_ids (list): List of cell IDs used.
    """
    datapath = pl.Path(base_path) / sample_name
    segpath = pl.Path(raw_path) / "xenium_seg"
    wsipath = pl.Path(raw_path) / "wsis"

    # Load adata with raw GEX
    adata = sc.read_h5ad(datapath / "adata.h5ad")
    adata.obs_names = adata.obs_names.astype(str)

    # This ensures all GEX has the same genes to start off with, fills with 0 when the gene is not present
    adata = reindex_adata_genes(adata, SOFT_UNION_GENE_LIST)

    # Perform log1p(CP10k) normalization
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Load segmentation and WSI
    df_seg = pd.read_parquet(segpath / f"{sample_name}_xenium_cell_seg.parquet")
    with tifffile.TiffFile(wsipath / f"{sample_name}.tif") as tif:
        wsi = tif.pages[0].asarray()

    # Extract centroid coordinates
    df_seg["geometry"] = df_seg["geometry"].apply(
        lambda g: g if hasattr(g, "centroid") else shapely.wkb.loads(g)
    )
    df_seg["he_x"] = df_seg["geometry"].apply(lambda g: g.centroid.x)
    df_seg["he_y"] = df_seg["geometry"].apply(lambda g: g.centroid.y)
    df_seg.index = df_seg.index.astype(str)
    # this is to ensure that the cell IDs are consistent with the .h5ad saved earlier
    df_seg.index = [f"{sample_name}_{cid}" for cid in df_seg.index]


    # Match cells
    common_ids = list(set(adata.obs_names) & set(df_seg.index))
    if not common_ids:
        raise ValueError(f"No overlapping cell IDs in {sample_name}.")

    n_cells = min(len(common_ids), max_cells)
    selected_ids = random.sample(common_ids, n_cells)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    img_tensors = []
    gex_tensors = []
    valid_ids = []

    for cid in selected_ids:
        x, y = df_seg.loc[cid, ["he_x", "he_y"]]
        x, y = int(x), int(y)

        x0, x1 = x - 128, x + 128
        y0, y1 = y - 128, y + 128

        pad_x0 = max(0, -x0)
        pad_x1 = max(0, x1 - wsi.shape[1])
        pad_y0 = max(0, -y0)
        pad_y1 = max(0, y1 - wsi.shape[0])

        patch = np.pad(
            wsi[max(0, y0):min(wsi.shape[0], y1), max(0, x0):min(wsi.shape[1], x1)],
            ((pad_y0, pad_y1), (pad_x0, pad_x1), (0, 0)),
            mode="constant"
        )

        if patch.shape[:2] != (256, 256):
            continue  # Skip bad patch

        try:
            img_tensor = transform(Image.fromarray(patch))
            gex = adata[cid].X.A.squeeze() if hasattr(adata.X, "A") else adata[cid].X.squeeze()
        except Exception as e:
            print(f"[DEBUG] Skipping cell {cid} due to error: {e}")
            continue

        img_tensors.append(img_tensor)
        gex_tensors.append(np.asarray(gex, dtype=np.float32))
        valid_ids.append(cid)

    if not img_tensors:
        raise ValueError(f"No valid cells found for sample {sample_name} after filtering.")

    img_tensor = torch.stack(img_tensors)  # (N, 3, 224, 224)
    gex_tensor = torch.tensor(np.stack(gex_tensors), dtype=torch.float32)

    # Standardize GEX
    eps = 1e-6
    gex_tensor = (gex_tensor - gex_tensor.mean(dim=0)) / (gex_tensor.std(dim=0) + eps)

    return img_tensor, gex_tensor, valid_ids