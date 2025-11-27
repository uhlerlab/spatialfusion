"""
Utility functions for loading and preprocessing multi-modal AE data.

This module provides:
- load_file_with_fallback: Load DataFrame from CSV or Parquet with fallback.
- safe_standardize: Robust z-score standardization for DataFrames.
- load_and_preprocess_sample: Load, intersect, impute, and standardize paired sample embeddings.
"""

import pathlib as pl
import pandas as pd
import warnings
import random
import numpy as np


def load_file_with_fallback(base_path, filename_base):
    """
    Attempts to load a DataFrame from CSV or Parquet.
    Raises FileNotFoundError if neither is available.

    Args:
        base_path (Path): Directory containing the file.
        filename_base (str): Base filename (without extension).

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    csv_path = base_path / f"{filename_base}.csv"
    parquet_path = base_path / f"{filename_base}.parquet"

    if csv_path.exists():
        return pd.read_csv(csv_path, index_col=0)
    elif parquet_path.exists():
        return pd.read_parquet(parquet_path)
    else:
        raise FileNotFoundError(
            f"Neither {csv_path.name} nor {parquet_path.name} found in {base_path}")


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


def load_and_preprocess_sample(sample_name, base_path, max_cells=30000):
    """
    Loads and preprocesses paired sample embeddings for AE training.
    - Loads UNI and scGPT embeddings for a sample.
    - Intersects cell IDs, samples up to max_cells.
    - Imputes NaNs with mean values.
    - Standardizes features robustly.

    Args:
        sample_name (str): Sample identifier.
        base_path (str or Path): Directory containing sample data.
        max_cells (int): Maximum number of cells to sample.

    Returns:
        tuple: (std_feat_1, std_feat_2, selected_ids)
            std_feat_1 (pd.DataFrame): Standardized UNI features.
            std_feat_2 (pd.DataFrame): Standardized scGPT features.
            selected_ids (list): List of selected cell IDs.
    """
    datapath = pl.Path(base_path) / sample_name
    embedding_path = datapath / "embeddings"

    # Load either .csv or .parquet
    uni = load_file_with_fallback(embedding_path, "UNI")
    scgpt = load_file_with_fallback(embedding_path, "scGPT")

    # Intersect cell IDs
    cell_ids = set(uni.index).intersection(scgpt.index)
    if not cell_ids:
        raise ValueError(f"No common cells found in {sample_name}.")

    common_ids = list(cell_ids)
    n_cells = min(len(common_ids), max_cells)
    selected_ids = random.sample(common_ids, n_cells)

    patho_feat = uni.loc[selected_ids]
    transcr_feat = scgpt.loc[selected_ids]

    # Impute NaNs
    patho_nans = patho_feat.isna().any()
    transcr_nans = transcr_feat.isna().any()

    if patho_nans.any():
        bad_dims = list(patho_nans[patho_nans].index)
        warnings.warn(
            f"[{sample_name}] UNI has NaNs in dims: {bad_dims}. Applying mean imputation.")
        patho_feat = patho_feat.fillna(patho_feat.mean())

    if transcr_nans.any():
        bad_dims = list(transcr_nans[transcr_nans].index)
        warnings.warn(
            f"[{sample_name}] scGPT has NaNs in dims: {bad_dims}. Applying mean imputation.")
        transcr_feat = transcr_feat.fillna(transcr_feat.mean())

    # Safe Standardization
    std_feat_1 = safe_standardize(patho_feat)
    std_feat_2 = safe_standardize(transcr_feat)

    return std_feat_1, std_feat_2, selected_ids
