"""
Utility functions for extracting and saving AE embeddings and metadata.

This module provides:
- safe_standardize: Robust z-score standardization for DataFrames.
- extract_embeddings_for_all_samples: Extract embeddings for all samples using a trained AE model.
- save_embeddings_separately: Save embeddings and metadata to disk.
"""
import os
import torch
from tqdm import tqdm
import pathlib as pl
import pandas as pd
import scanpy as sc
import numpy as np
import h5py
from collections.abc import Mapping
import warnings

LABEL_CANDIDATES = [
    "celltypes", "cellsubtypes", "celltype", "CellType", "cell_type",
    "label", "labels", "annotation", "Annotation", 'major_celltype',
]


def _to_str_index(idx_like):
    """
    Cast index-like object to pandas Index of strings, stripping whitespace.

    Args:
        idx_like (iterable): Index or iterable of IDs.
    Returns:
        pd.Index: String index.
    """
    return pd.Index([str(x).strip() for x in idx_like], dtype="object")


def _extract_labels_from_df(df: pd.DataFrame, ids) -> np.ndarray:
    """
    Extract a 1-D array of labels for the given ids from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with label columns.
        ids (iterable): IDs to extract labels for.
    Returns:
        np.ndarray: Array of labels as strings.
    """
    df = df.loc[ids]
    # Pick a column by name if available
    for col in LABEL_CANDIDATES:
        if col in df.columns:
            return df[col].astype(str).to_numpy()
    # If exactly one column, use it
    if df.shape[1] == 1:
        return df.iloc[:, 0].astype(str).to_numpy()
    # Otherwise warn and use the first column
    warnings.warn(
        f"Multiple label columns found ({list(df.columns)}); using the first one."
    )
    return df.iloc[:, 0].astype(str).to_numpy()


def safe_standardize(df: pd.DataFrame, fill_value: float = 0.0, min_std: float = 1e-5) -> pd.DataFrame:
    """
    Standardizes a DataFrame (z-score per column) while avoiding NaNs and large numbers.
    Handles unsafe float16 input by casting to float32 first.

    Any column with std < min_std is filled with `fill_value`.

    Args:
        df (pd.DataFrame): Input DataFrame.
        fill_value (float): Value to fill for low-variance columns.
        min_std (float): Minimum allowed std for columns.

    Returns:
        pd.DataFrame: Standardized DataFrame (float32), no NaNs.
    """
    # Ensure float32 to prevent float16 overflow and float64 memory bloat
    df = df.astype(np.float32, copy=False)

    means = df.mean()
    stds = df.std()

    low_variance_mask = stds < min_std
    stds_clipped = stds.clip(lower=min_std)

    standardized = (df - means) / stds_clipped
    standardized.loc[:, low_variance_mask] = fill_value

    if low_variance_mask.any():
        print(
            f"⚠️ Columns with std < {min_std} set to {fill_value}: {list(df.columns[low_variance_mask])}")

    # Final safety check
    assert np.isfinite(standardized.values).all(
    ), "Non-finite values in standardized data"

    return standardized.astype(np.float32)


def extract_embeddings_for_all_samples(model, sample_list, base_path, device='cpu'):
    """
    Extract embeddings for all samples using a trained AE model.
    Loads UNI and scGPT embeddings, matches cell IDs, standardizes features, and computes model embeddings.
    Also extracts cell type labels and sample names.

    Args:
        model: Trained AE model with encoder1 and encoder2.
        sample_list (list): List of sample info (str or dict).
        base_path (str or Path): Base directory for samples.
        device (str): Device for model inference.

    Returns:
        tuple: (z1_df, z2_df, z_joint_df, celltypes, samples)
            z1_df (pd.DataFrame): Embeddings from encoder1.
            z2_df (pd.DataFrame): Embeddings from encoder2.
            z_joint_df (pd.DataFrame): Averaged joint embeddings.
            celltypes (np.ndarray): Cell type labels.
            samples (np.ndarray): Sample names.
    """
    all_z1, all_z2, all_zjoint = [], [], []
    all_celltypes, all_samples = [], []

    model.eval()

    with torch.no_grad():
        for sample_info in tqdm(sample_list, desc="Processing samples"):
            # Support legacy string or new dict format with custom path
            if isinstance(sample_info, Mapping):
                sample = str(sample_info["name"])
                sample_path = str(sample_info.get("path", base_path))
            else:
                sample = str(sample_info)
                sample_path = str(base_path)

            datapath = pl.Path(sample_path) / sample
            embeddings_path = datapath / "embeddings"
            celltypes_path = datapath / "celltypes.csv"

            print(f"🔍 Looking for embeddings in: {embeddings_path}")

            # Detect UNI
            uni_path = None
            for ext in ('.csv', '.parquet'):
                p = embeddings_path / f"UNI{ext}"
                if p.exists():
                    uni_path = p
                    break

            # Detect scGPT
            scgpt_path = None
            for ext in ('.csv', '.parquet'):
                p = embeddings_path / f"scGPT{ext}"
                if p.exists():
                    scgpt_path = p
                    break

            if not (uni_path and scgpt_path):
                print(f"⚠️ Missing embedding files for {sample}, skipping.")
                continue

            try:
                uni = (pd.read_csv(uni_path, index_col=0) if uni_path.suffix == '.csv'
                       else pd.read_parquet(uni_path)).astype('float32')
                scgpt = (pd.read_csv(scgpt_path, index_col=0) if scgpt_path.suffix == '.csv'
                         else pd.read_parquet(scgpt_path)).astype('float32')

                # 🔧 Canonicalize indices to strings (trim whitespace)
                uni.index = _to_str_index(uni.index)
                scgpt.index = _to_str_index(scgpt.index)

            except Exception as e:
                print(f"❌ Skipping {sample} due to read error: {e}")
                continue

            # Determine available IDs
            adata = None
            try:
                if celltypes_path.exists():
                    ct_df = pd.read_csv(celltypes_path, index_col=0)
                    # <- make labels index strings
                    ct_df.index = _to_str_index(ct_df.index)
                    base_ids = set(ct_df.index)
                else:
                    adata = sc.read_h5ad(datapath / "adata.h5ad")
                    # Ensure obs_names are strings
                    adata.obs_names = _to_str_index(adata.obs_names)
                    base_ids = set(adata.obs_names)
            except Exception as e:
                print(
                    f"⚠️ No celltypes.csv or unreadable adata.h5ad for {sample} ({e}), skipping.")
                continue

            # Intersect on string IDs
            cell_ids = set(uni.index) & set(scgpt.index) & base_ids

            if not cell_ids:
                # Helpful debug: show small samples of IDs
                u0 = list(uni.index)[:3]
                s0 = list(scgpt.index)[:3]
                b0 = list(base_ids)[:3]
                print(f"⚠️ No overlapping cell IDs for {sample}, skipping."
                      f"\n   UNI sample IDs: {u0}"
                      f"\n   scGPT sample IDs: {s0}"
                      f"\n   Base (labels/adata) sample IDs: {b0}")
                continue

            common_ids = sorted(cell_ids)
            patho_feat = uni.loc[common_ids]
            transcr_feat = scgpt.loc[common_ids]

            std_feat_1 = safe_standardize(patho_feat)
            std_feat_2 = safe_standardize(transcr_feat)

            X1 = torch.tensor(std_feat_1.values,
                              dtype=torch.float32).to(device)
            X2 = torch.tensor(std_feat_2.values,
                              dtype=torch.float32).to(device)

            z1 = model.encoder1(X1).cpu().numpy()
            z2 = model.encoder2(X2).cpu().numpy()
            z_joint = (z1 + z2) / 2

            all_z1.append(pd.DataFrame(z1, index=common_ids))
            all_z2.append(pd.DataFrame(z2, index=common_ids))
            all_zjoint.append(pd.DataFrame(z_joint, index=common_ids))

            # Labels
            if celltypes_path.exists():
                labels = _extract_labels_from_df(ct_df, common_ids)
            else:
                obs = adata.obs.loc[common_ids]
                picked = None
                for col in LABEL_CANDIDATES:
                    if col in obs.columns:
                        picked = obs[col].astype(str).to_numpy()
                        break
                labels = picked if picked is not None else np.array(
                    ["unknown"] * len(common_ids), dtype=object)

            labels = np.asarray(labels).reshape(-1)
            all_celltypes.append(labels)
            all_samples.append(
                np.array([sample] * len(common_ids), dtype=object))

    # Concatenate
    z1_df = pd.concat(all_z1) if all_z1 else pd.DataFrame()
    z2_df = pd.concat(all_z2) if all_z2 else pd.DataFrame()
    z_joint_df = pd.concat(all_zjoint) if all_zjoint else pd.DataFrame()

    celltypes = np.concatenate(
        all_celltypes, axis=0) if all_celltypes else np.array([], dtype=object)
    samples = np.concatenate(
        all_samples, axis=0) if all_samples else np.array([], dtype=object)

    # Final safety: ensure indices are strings
    if not z_joint_df.empty:
        z1_df.index = _to_str_index(z1_df.index)
        z2_df.index = _to_str_index(z2_df.index)
        z_joint_df.index = _to_str_index(z_joint_df.index)

    return z1_df, z2_df, z_joint_df, celltypes, samples


def save_embeddings_separately(z1_df, z2_df, z_joint_df, celltypes, samples, out_dir, mode='train', compression="gzip"):
    """
    Save embeddings and metadata to disk as Parquet and HDF5 files.

    Args:
        z1_df (pd.DataFrame): Embeddings from encoder1.
        z2_df (pd.DataFrame): Embeddings from encoder2.
        z_joint_df (pd.DataFrame): Joint embeddings.
        celltypes (np.ndarray): Cell type labels.
        samples (np.ndarray): Sample names.
        out_dir (str or Path): Output directory.
        mode (str): Mode string for filenames (e.g., 'train').
        compression (str): Compression type for HDF5 datasets.
    """
    os.makedirs(out_dir, exist_ok=True)
    z1_df.to_parquet(f"{out_dir}/z1_{mode}.parquet")
    z2_df.to_parquet(f"{out_dir}/z2_{mode}.parquet")
    z_joint_df.to_parquet(f"{out_dir}/z_joint_{mode}.parquet")

    dt = h5py.string_dtype(encoding='utf-8')
    celltypes = np.asarray(celltypes, dtype=object)
    samples = np.asarray(samples, dtype=object)

    with h5py.File(f"{out_dir}/metadata_{mode}.h5", "w") as f:
        f.create_dataset("celltypes", data=celltypes,
                         dtype=dt, compression=compression)
        f.create_dataset("samples", data=samples,
                         dtype=dt, compression=compression)

    print(f"✓ Saved embeddings and metadata to: {out_dir}")
