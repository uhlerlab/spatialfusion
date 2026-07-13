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

HE_EMBEDDINGS = {
    "uni": "UNI",
    "virchow": "Virchow2",
}

RNA_EMBEDDINGS = {
    "scgpt": "scGPT",
    "nicheformer": "nicheformer",
}


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


def extract_embeddings_for_all_samples(
    model,
    sample_list,
    base_path,
    device="cpu",
    batch_size=None,
    he_encoder="uni",
    rna_encoder="scgpt",
):
    """
    Extract latent embeddings for all samples using a trained paired AE.

    This function:

    1. Loads the requested HE embeddings
       (UNI or Virchow).

    2. Loads the requested RNA embeddings
       (scGPT or Nicheformer).

    3. Intersects cell identifiers across:
       - HE embeddings
       - RNA embeddings
       - cell type annotations

    4. Standardizes both modalities.

    5. Computes latent representations using:
       - encoder1 (HE branch)
       - encoder2 (RNA branch)

    6. Computes a joint embedding via arithmetic mean:
           z_joint = (z1 + z2) / 2

    Args:
        model:
            Trained PairedAE model.

        sample_list:
            List of sample names or sample dictionaries.

        base_path:
            Root directory containing sample folders.

        device:
            Torch device for inference.

        batch_size:
            Optional inference batch size.
            If None, a memory-aware batch size is chosen.

        he_encoder:
            HE embedding type.

            Supported:
                - "uni"
                - "virchow"

        rna_encoder:
            RNA embedding type.

            Supported:
                - "scgpt"
                - "nicheformer"

    Returns:
        tuple:

            z1_df:
                Latent embeddings from encoder1
                (HE modality).

            z2_df:
                Latent embeddings from encoder2
                (RNA modality).

            z_joint_df:
                Averaged latent embeddings.

            celltypes:
                Cell type labels.

            samples:
                Sample identifiers.
    """

    he_encoder = he_encoder.lower()
    rna_encoder = rna_encoder.lower()

    if he_encoder not in HE_EMBEDDINGS:
        raise ValueError(
            f"Unknown HE encoder '{he_encoder}'. "
            f"Valid options: {list(HE_EMBEDDINGS.keys())}"
        )

    if rna_encoder not in RNA_EMBEDDINGS:
        raise ValueError(
            f"Unknown RNA encoder '{rna_encoder}'. "
            f"Valid options: {list(RNA_EMBEDDINGS.keys())}"
        )

    he_file = HE_EMBEDDINGS[he_encoder]
    rna_file = RNA_EMBEDDINGS[rna_encoder]

    print(
        f"Using HE encoder: {he_encoder} ({he_file}) | "
        f"RNA encoder: {rna_encoder} ({rna_file})"
    )

    all_z1 = []
    all_z2 = []
    all_zjoint = []

    all_celltypes = []
    all_samples = []

    model.eval()

    with torch.no_grad():

        for sample_info in tqdm(
            sample_list,
            desc="Processing samples"
        ):

            if isinstance(sample_info, Mapping):
                sample = str(sample_info["name"])
                sample_path = str(
                    sample_info.get("path", base_path)
                )
            else:
                sample = str(sample_info)
                sample_path = str(base_path)

            datapath = pl.Path(sample_path) / sample

            embeddings_path = datapath / "embeddings"
            celltypes_path = datapath / "celltypes.csv"

            print(
                f"🔍 [{sample}] "
                f"Searching for "
                f"{he_file} + {rna_file}"
            )

            # --------------------------------------------------
            # Locate embedding files
            # --------------------------------------------------

            he_path = None

            for ext in (".csv", ".parquet"):
                candidate = embeddings_path / f"{he_file}{ext}"

                if candidate.exists():
                    he_path = candidate
                    break

            rna_path = None

            for ext in (".csv", ".parquet"):
                candidate = embeddings_path / f"{rna_file}{ext}"

                if candidate.exists():
                    rna_path = candidate
                    break

            if not (he_path and rna_path):
                print(
                    f"⚠️ Missing embeddings for {sample} "
                    f"({he_file}, {rna_file})"
                )
                continue

            # --------------------------------------------------
            # Load embeddings
            # --------------------------------------------------

            try:

                he_df = (
                    pd.read_csv(
                        he_path,
                        index_col=0,
                    )
                    if he_path.suffix == ".csv"
                    else pd.read_parquet(he_path)
                ).astype(np.float32)

                rna_df = (
                    pd.read_csv(
                        rna_path,
                        index_col=0,
                    )
                    if rna_path.suffix == ".csv"
                    else pd.read_parquet(rna_path)
                ).astype(np.float32)

                he_df.index = _to_str_index(
                    he_df.index
                )

                rna_df.index = _to_str_index(
                    rna_df.index
                )

            except Exception as e:

                print(
                    f"❌ Failed reading "
                    f"{sample}: {e}"
                )

                continue

            # --------------------------------------------------
            # Load labels
            # --------------------------------------------------

            adata = None

            try:

                if celltypes_path.exists():

                    ct_df = pd.read_csv(
                        celltypes_path,
                        index_col=0,
                    )

                    ct_df.index = _to_str_index(
                        ct_df.index
                    )

                    base_ids = set(ct_df.index)

                else:

                    adata = sc.read_h5ad(
                        datapath / "adata.h5ad"
                    )

                    adata.obs_names = _to_str_index(
                        adata.obs_names
                    )

                    base_ids = set(
                        adata.obs_names
                    )

            except Exception as e:

                print(
                    f"⚠️ Unable to load labels "
                    f"for {sample}: {e}"
                )

                continue

            # --------------------------------------------------
            # Intersect cell IDs
            # --------------------------------------------------

            cell_ids = (
                set(he_df.index)
                & set(rna_df.index)
                & base_ids
            )

            if not cell_ids:

                print(
                    f"⚠️ No overlapping cells "
                    f"for {sample}"
                )

                continue

            common_ids = sorted(cell_ids)

            he_feat = he_df.loc[common_ids]
            rna_feat = rna_df.loc[common_ids]

            std_feat_1 = safe_standardize(
                he_feat
            )

            std_feat_2 = safe_standardize(
                rna_feat
            )

            X1_np = std_feat_1.values.astype(
                np.float32
            )

            X2_np = std_feat_2.values.astype(
                np.float32
            )

            n_samples = X1_np.shape[0]

            # --------------------------------------------------
            # Auto batch size
            # --------------------------------------------------

            effective_batch_size = batch_size

            if effective_batch_size is None:

                est_mem_per_sample = (
                    (X1_np.shape[1] + X2_np.shape[1])
                    * 4
                    * 2
                )

                effective_batch_size = max(
                    1,
                    int(
                        300 * 1024 * 1024
                        / est_mem_per_sample
                    ),
                )

                effective_batch_size = min(
                    effective_batch_size,
                    5000,
                )

                effective_batch_size = max(
                    effective_batch_size,
                    100,
                )

            # --------------------------------------------------
            # Forward pass
            # --------------------------------------------------

            if n_samples <= effective_batch_size:

                X1 = torch.from_numpy(
                    X1_np
                ).to(device)

                X2 = torch.from_numpy(
                    X2_np
                ).to(device)

                z1 = model.encoder1(
                    X1
                ).cpu().numpy()

                z2 = model.encoder2(
                    X2
                ).cpu().numpy()

            else:

                z1_chunks = []
                z2_chunks = []

                for start in range(
                    0,
                    n_samples,
                    effective_batch_size,
                ):

                    end = min(
                        start + effective_batch_size,
                        n_samples,
                    )

                    X1_batch = torch.from_numpy(
                        X1_np[start:end]
                    ).to(device)

                    X2_batch = torch.from_numpy(
                        X2_np[start:end]
                    ).to(device)

                    z1_chunks.append(
                        model.encoder1(
                            X1_batch
                        ).cpu().numpy()
                    )

                    z2_chunks.append(
                        model.encoder2(
                            X2_batch
                        ).cpu().numpy()
                    )

                z1 = np.vstack(z1_chunks)
                z2 = np.vstack(z2_chunks)

            z_joint = (z1 + z2) / 2.0

            all_z1.append(
                pd.DataFrame(
                    z1,
                    index=common_ids,
                )
            )

            all_z2.append(
                pd.DataFrame(
                    z2,
                    index=common_ids,
                )
            )

            all_zjoint.append(
                pd.DataFrame(
                    z_joint,
                    index=common_ids,
                )
            )

            if celltypes_path.exists():

                labels = _extract_labels_from_df(
                    ct_df,
                    common_ids,
                )

            else:

                obs = adata.obs.loc[
                    common_ids
                ]

                picked = None

                for col in LABEL_CANDIDATES:

                    if col in obs.columns:
                        picked = (
                            obs[col]
                            .astype(str)
                            .to_numpy()
                        )
                        break

                labels = (
                    picked
                    if picked is not None
                    else np.array(
                        ["unknown"] * len(common_ids),
                        dtype=object,
                    )
                )

            all_celltypes.append(
                np.asarray(labels)
            )

            all_samples.append(
                np.array(
                    [sample] * len(common_ids),
                    dtype=object,
                )
            )

    z1_df = (
        pd.concat(all_z1)
        if all_z1
        else pd.DataFrame()
    )

    z2_df = (
        pd.concat(all_z2)
        if all_z2
        else pd.DataFrame()
    )

    z_joint_df = (
        pd.concat(all_zjoint)
        if all_zjoint
        else pd.DataFrame()
    )

    celltypes = (
        np.concatenate(all_celltypes)
        if all_celltypes
        else np.array([], dtype=object)
    )

    samples = (
        np.concatenate(all_samples)
        if all_samples
        else np.array([], dtype=object)
    )

    if not z_joint_df.empty:

        z1_df.index = _to_str_index(
            z1_df.index
        )

        z2_df.index = _to_str_index(
            z2_df.index
        )

        z_joint_df.index = _to_str_index(
            z_joint_df.index
        )

    return (
        z1_df,
        z2_df,
        z_joint_df,
        celltypes,
        samples,
    )


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
