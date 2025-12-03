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

# ---------------------------
# Small utilities
# ---------------------------


def _combine_embeddings(z1: pd.DataFrame, z2: pd.DataFrame, mode: Literal["average", "concat", "z1", "z2"]) -> pd.DataFrame:
    mode = mode.lower()
    if mode not in {"average", "concat", "z1", "z2"}:
        raise ValueError(
            "combine_mode must be one of: 'average', 'concat', 'z1', 'z2'")
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

    if mode == "concat":
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
    p = str(path)
    if p.endswith(".csv"):
        return pd.read_csv(p, index_col=0, nrows=1)
    return pd.read_parquet(p, engine="pyarrow")


def infer_input_dims_from_files(uni_path: Union[str, Path], scgpt_path: Union[str, Path]) -> Tuple[int, int]:
    uni = _read_tabular_one_row(uni_path)
    scgpt = _read_tabular_one_row(scgpt_path)
    return uni.shape[1], scgpt.shape[1]


def infer_input_dims(sample_list: Iterable[str], base_path: Union[str, Path],
                     uni_path: Optional[Union[str, Path]] = None,
                     scgpt_path: Optional[Union[str, Path]] = None) -> Tuple[int, int]:
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
    adata: sc.AnnData
    z_uni: pd.DataFrame
    z_scgpt: Optional[pd.DataFrame] = None   # ← allow None


def load_paired_ae(ae_ckpt: Union[str, Path], d1_dim: int, d2_dim: int,
                   latent_dim: int = 64, device: str = "cuda:0") -> PairedAE:
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
    combine_mode: Literal["average", "concat", "z1", "z2"] = "average",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run PairedAE on already-loaded adata + UNI/scGPT embeddings.
    Standardizes inputs to match training preprocessing.
    Returns: (z1_df, z2_df, z_joint_df) aligned to inputs.adata.obs_names.
    """
    # determine which inputs are required based on combine_mode
    needs_z1 = True
    needs_z2 = (combine_mode in ["average", "concat", "z2"])

    idx = inputs.adata.obs_names.astype(str)

    if needs_z1:
        common = idx.intersection(inputs.z_uni.index)
    if needs_z2:
        if inputs.z_scgpt is None:
            raise ValueError(
                "combine_mode requires scGPT input, but z_scgpt=None")
        common = common.intersection(inputs.z_scgpt.index)

    if len(common) == 0:
        raise ValueError(
            "No overlapping cells between adata and required embeddings.")

    # --- Standardize before feeding to AE
    x1_df = safe_standardize(inputs.z_uni.loc[common])
    x1_np = x1_df.astype(np.float32).values

    if needs_z2:
        x2_df = safe_standardize(inputs.z_scgpt.loc[common])
        x2_np = x2_df.astype(np.float32).values
    else:
        x2_np = None

    with torch.no_grad():
        x1 = torch.from_numpy(x1_np).to(device)
        if needs_z2:
            x2 = torch.from_numpy(x2_np).to(device)
        else:
            x2 = None

        out = model(x1, x2)

        z1_t = out.get("z1")
        z2_t = out.get("z2")
        if z1_t is None and z2_t is None:
            raise ValueError("Model output does not contain 'z1' or 'z2'.")

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
    save_dir: Optional[Union[str, Path]] = None,
):
    """
    The original disk-based AE extraction, but returns the DataFrames and (optionally) saves.
    """
    z1, z2, z_joint_unused, celltypes, samples = extract_embeddings_for_all_samples(
        model, sample_list, base_path, device
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
    """Inputs for GCN step when you already have the AE joint embedding."""
    z_joint: pd.DataFrame                        # indexed by cell, columns = features
    # each AnnData contains obs_names that intersect z_joint.index
    adata_by_sample: Dict[str, sc.AnnData]


def load_gcn(gcn_ckpt: Union[str, Path], in_dim: int, device: str = "cuda:0") -> GCNAutoencoder:
    model = GCNAutoencoder(
        in_dim=in_dim, hidden_dim=10, out_dim=in_dim,
        node_mask_ratio=0.9, num_layers=2, n_classes=0
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
) -> pd.DataFrame:
    """
    Build graphs from (adata, z_joint) and run the GCN to produce embeddings with metadata.
    """
    full_graphs, ordered_samples = graphs_from_embeddings_and_adata(
        z_joint, adata_by_sample, spatial_key=spatial_key, k=k
    )
    emb_df = extract_gcn_embeddings_with_metadata(
        gcn_model, full_graphs, ordered_samples, Path(base_path), z_joint,
        device=device, spatial_key=spatial_key, celltype_key=celltype_key,
        adata_by_sample=adata_by_sample,
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
    combine_mode: Literal["average", "concat", "z1", "z2"] = "average",

    # optional explicit input-dim inference from files for disk path mode
    uni_path: Optional[Union[str, Path]] = None,
    scgpt_path: Optional[Union[str, Path]] = None,

    # outputs
    save_ae_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Flexible end-to-end:
      - If `ae_inputs_by_sample` is provided, compute AE embeddings in-memory per sample,
        then run GCN on the combined joint embedding.
      - Else, use `sample_list` + `base_path` to load from disk like the original pipeline.

    Returns:
        emb_df (pd.DataFrame): final GCN embeddings + metadata
    """
    # --- AE stage ---
    if ae_inputs_by_sample is not None:
        # In-memory branch
        # infer dims from the first sample’s matrices
        first = next(iter(ae_inputs_by_sample.values()))
        d1_dim = first.z_uni.shape[1]
        if first.z_scgpt is not None:
            d2_dim = first.z_scgpt.shape[1]
        else:
            # load temporarily to inspect expected input dim
            tmp_state = torch.load(ae_model_path, map_location="cpu")
            # encoder2 layers always start with a weight of shape (latent_dim, d2_dim)
            for k, v in tmp_state.items():
                if k.startswith("encoder2.model.0.weight"):   # first Linear layer
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
                ae_model, inputs, device=device, combine_mode=combine_mode)
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
            ae_model, sample_list, base_path, device=device, combine_mode=combine_mode, save_dir=save_ae_dir
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
            gcn_model_path, in_dim=z_joint_df.shape[1], device=device)

    emb_df = gcn_embeddings_from_joint(
        gcn_model=gcn_model,
        z_joint=z_joint_df,
        adata_by_sample=adata_by_sample,
        base_path=base_path if base_path is not None else ".",
        device=device,
        spatial_key=spatial_key,
        celltype_key=celltype_key,
        k=k,
    )
    return emb_df
