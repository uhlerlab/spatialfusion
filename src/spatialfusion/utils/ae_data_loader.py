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


# ---------------------------------------------------------
# Embedding filename registry
# ---------------------------------------------------------

HE_EMBEDDINGS = {
    "uni": "UNI",
    "virchow": "Virchow2",
}

RNA_EMBEDDINGS = {
    "scgpt": "scGPT",
    "nicheformer": "nicheformer",
}


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
            f"Neither {csv_path.name} nor "
            f"{parquet_path.name} found in {base_path}"
        )


def safe_standardize(
    df: pd.DataFrame,
    fill_value: float = 0.0,
    min_std: float = 1e-5
) -> pd.DataFrame:
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

    # prevent float16 overflow + float64 memory blowup
    df = df.astype(np.float32, copy=False)

    means = df.mean()
    stds = df.std()

    low_variance_mask = stds < min_std
    stds_clipped = stds.clip(lower=min_std)

    standardized = (df - means) / stds_clipped
    standardized.loc[:, low_variance_mask] = fill_value

    if low_variance_mask.any():
        print(
            f"⚠️ Columns with std < {min_std} "
            f"set to {fill_value}: "
            f"{list(df.columns[low_variance_mask])}"
        )

    # final safety check
    assert np.isfinite(standardized.values).all(), (
        "Non-finite values in standardized data"
    )

    return standardized.astype(np.float32)


def load_and_preprocess_sample(
    sample_name,
    base_path,
    max_cells=30000,
    he_encoder="uni",
    rna_encoder="scgpt",
):
    """
    Loads and preprocesses paired sample embeddings for AE training.

    Steps:
    - Load selected HE + RNA embeddings
    - Intersect cell IDs
    - Randomly sample up to max_cells
    - Impute NaNs
    - Standardize features

    Args:
        sample_name (str): Sample identifier.
        base_path (str or Path): Directory containing sample data.
        max_cells (int): Maximum number of cells to sample.
        he_encoder (str): HE encoder name.
            Options: "uni", "virchow"
        rna_encoder (str): RNA encoder name.
            Options: "scgpt", "nicheformer"

    Returns:
        tuple:
            std_feat_1 (pd.DataFrame):
                Standardized HE features

            std_feat_2 (pd.DataFrame):
                Standardized RNA features

            selected_ids (list):
                Selected cell IDs
    """

    he_encoder = he_encoder.lower()
    rna_encoder = rna_encoder.lower()

    if he_encoder not in HE_EMBEDDINGS:
        raise ValueError(
            f"Unknown HE encoder: {he_encoder}. "
            f"Valid options: {list(HE_EMBEDDINGS.keys())}"
        )

    if rna_encoder not in RNA_EMBEDDINGS:
        raise ValueError(
            f"Unknown RNA encoder: {rna_encoder}. "
            f"Valid options: {list(RNA_EMBEDDINGS.keys())}"
        )

    datapath = pl.Path(base_path) / sample_name
    embedding_path = datapath / "embeddings"

    he_file = HE_EMBEDDINGS[he_encoder]
    rna_file = RNA_EMBEDDINGS[rna_encoder]

    print(
        f"[{sample_name}] "
        f"Loading HE={he_encoder} ({he_file}) | "
        f"RNA={rna_encoder} ({rna_file})"
    )

    # ---------------------------------------------------------
    # Load embeddings
    # ---------------------------------------------------------

    he_df = load_file_with_fallback(
        embedding_path,
        he_file
    )

    rna_df = load_file_with_fallback(
        embedding_path,
        rna_file
    )

    # ---------------------------------------------------------
    # Intersect cell IDs
    # ---------------------------------------------------------

    cell_ids = set(he_df.index).intersection(rna_df.index)

    if not cell_ids:
        raise ValueError(
            f"No common cells found in {sample_name}."
        )

    common_ids = list(cell_ids)

    n_cells = min(len(common_ids), max_cells)

    selected_ids = random.sample(common_ids, n_cells)

    he_feat = he_df.loc[selected_ids]
    rna_feat = rna_df.loc[selected_ids]

    # ---------------------------------------------------------
    # NaN imputation
    # ---------------------------------------------------------

    he_nans = he_feat.isna().any()
    rna_nans = rna_feat.isna().any()

    if he_nans.any():

        bad_dims = list(he_nans[he_nans].index)

        warnings.warn(
            f"[{sample_name}] "
            f"{he_file} has NaNs in dims: {bad_dims}. "
            f"Applying mean imputation."
        )

        he_feat = he_feat.fillna(he_feat.mean())

    if rna_nans.any():

        bad_dims = list(rna_nans[rna_nans].index)

        warnings.warn(
            f"[{sample_name}] "
            f"{rna_file} has NaNs in dims: {bad_dims}. "
            f"Applying mean imputation."
        )

        rna_feat = rna_feat.fillna(rna_feat.mean())

    # ---------------------------------------------------------
    # Standardization
    # ---------------------------------------------------------

    std_feat_1 = safe_standardize(he_feat)
    std_feat_2 = safe_standardize(rna_feat)

    return std_feat_1, std_feat_2, selected_ids
