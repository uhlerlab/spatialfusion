"""
Utility functions for extracting GCN embeddings and metadata for downstream analysis.

This module provides:
- extract_gcn_embeddings_with_metadata: Extracts GCN embeddings and merges with cell type,
  spatial, and ligand-receptor metadata. Supports both full-graph and batched subgraph inference.
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


# ------------------------------------------------
# Helper utilities
# ------------------------------------------------

def infer_output_dim(model, feat_dim: int) -> int:
    """
    Infer the output dimensionality of a GCN encoder.

    This function runs a dummy forward pass through `model.encode`
    using a single-node graph to determine the latent output size.

    Args:
        model (torch.nn.Module):
            GCN model with an `encode` method.
        feat_dim (int):
            Input feature dimensionality.

    Returns:
        int:
            Output embedding dimensionality.
    """
    device = next(model.parameters()).device
    dummy_g = dgl.graph(([], []), num_nodes=1).to(device)
    dummy_g = dgl.add_self_loop(dummy_g)
    dummy_feat = torch.zeros((1, feat_dim), device=device)
    z = model.encode(dummy_g, dummy_feat)
    return z.shape[1]


def expand_k_hop(g: dgl.DGLGraph, seeds: torch.Tensor, k: int) -> torch.Tensor:
    """
    Expand a set of seed nodes to include all nodes reachable within k hops.

    This function performs exact k-hop expansion for both incoming and outgoing
    edges and is used for memory-efficient subgraph inference.

    Args:
        g (dgl.DGLGraph):
            Input graph.
        seeds (torch.Tensor):
            Tensor of seed node indices.
        k (int):
            Number of hops.

    Returns:
        torch.Tensor:
            Sorted tensor of expanded node indices.
    """
    seeds = seeds.to(g.device)
    visited = set(seeds.tolist())
    frontier = seeds

    for _ in range(k):
        out_nbrs = g.out_edges(frontier)[1]
        in_nbrs = g.in_edges(frontier)[0]
        nbrs = torch.cat([out_nbrs, in_nbrs]).unique()

        new_nodes = [int(n) for n in nbrs.tolist() if n not in visited]
        if not new_nodes:
            break

        for n in new_nodes:
            visited.add(n)

        frontier = torch.tensor(new_nodes, dtype=torch.long, device=g.device)

    return torch.tensor(sorted(visited), dtype=torch.long)


# ------------------------------------------------
# Main extraction function
# ------------------------------------------------

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
    batch_size: Optional[int] = None,
    k_hop: int = 2,
) -> pd.DataFrame:
    """
    Extract GCN embeddings along with metadata such as spatial coordinates and cell types.

    This function supports two inference modes:

    - **Full-graph inference** (default, exact behavior):
      If `batch_size` is None, the entire graph is moved to the device and
      processed in a single forward pass.

    - **Batched subgraph inference** (memory-efficient):
      If `batch_size` is an integer, nodes are processed in batches.
      For each batch of seed nodes, a k-hop subgraph is constructed to
      preserve exact receptive fields for multi-layer GCNs.

    Supports both in-memory AnnData inputs and on-disk loading via `base_path`.

    Args:
        model (torch.nn.Module):
            Trained GCN model with an `encode(graph, features)` method.
        graphs (list[dgl.DGLGraph]):
            List of DGL graphs, one per sample, containing node features in `ndata["feat"]`.
        sample_list (list[str]):
            List of sample IDs corresponding to the graphs.
        base_path (str or Path):
            Root directory containing per-sample subdirectories.
            Used only if `adata_by_sample` is not provided.
        z_joint (pd.DataFrame):
            Joint embeddings from the autoencoder step.
            Used to align graph nodes with cell identifiers.
        device (str):
            Device string for inference (e.g., `"cuda:0"` or `"cpu"`).
        spatial_key (str):
            Key name for spatial coordinates stored in `adata.obsm`.
        celltype_key (str):
            Column name in `adata.obs` or `celltypes.csv` for cell type annotation.
        adata_by_sample (Optional[Dict[str, sc.AnnData]]):
            Optional mapping from sample IDs to preloaded AnnData objects.
            If provided, this takes precedence over disk loading.
        batch_size (Optional[int]):
            Number of seed nodes per subgraph batch.
            If None, full-graph inference is used.
        k_hop (int):
            Number of hops for subgraph expansion when batching.
            Should match the effective receptive field of the GCN.

    Returns:
        pd.DataFrame:
            Concatenated DataFrame of GCN embeddings across all samples,
            including metadata:

            - sample_id
            - cell_id
            - celltype (and optional subtype/niche labels)
            - spatial coordinates (X_coord, Y_coord)
            - optional ligand–receptor features if present

    Raises:
        FileNotFoundError:
            If AnnData cannot be loaded from memory or disk.
        ValueError:
            If graph node count does not match aligned cell identifiers.
    """
    model.eval()
    all_dfs = []
    base_path = None if base_path in (None, "", ".") else pl.Path(base_path)

    out_dim_cache = None

    for g_cpu, sample in tqdm(
        zip(graphs, sample_list),
        total=len(sample_list),
        desc="Running GCN inference"
    ):

        # -----------------------------
        # Load AnnData
        # -----------------------------
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

        cell_ids = adata.obs_names.astype(str).intersection(z_joint.index)

        if len(cell_ids) != g_cpu.num_nodes():
            raise ValueError(
                f"[{sample}] node/cell mismatch: "
                f"{g_cpu.num_nodes()} nodes vs {len(cell_ids)} matched cells."
            )

        # -----------------------------------------------------
        # Full-graph inference
        # -----------------------------------------------------
        if batch_size is None:

            g = dgl.add_self_loop(g_cpu.to(device))
            feats = g.ndata["feat"].to(device)

            z = model.encode(g, feats)
            z = F.dropout(z, p=getattr(model, "dropout", 0.0), training=False)
            z_np = z.cpu().numpy()

        # -----------------------------------------------------
        # Batched subgraph inference
        # -----------------------------------------------------
        else:

            g = g_cpu
            node_ids = torch.arange(g.num_nodes())

            if out_dim_cache is None:
                feat_dim = g.ndata["feat"].shape[1]
                out_dim_cache = infer_output_dim(model, feat_dim)

            z_out = torch.zeros((g.num_nodes(), out_dim_cache), device="cpu")

            for start in tqdm(
                range(0, g.num_nodes(), batch_size),
                desc=f"{sample}: subgraph inference"
            ):
                end = min(start + batch_size, g.num_nodes())
                seeds = node_ids[start:end]

                expanded = expand_k_hop(g, seeds, k=k_hop)

                subg = dgl.node_subgraph(g, expanded)
                subg = dgl.add_self_loop(subg).to(device)

                feats = subg.ndata["feat"].to(device)
                z_sub = model.encode(subg, feats).cpu()

                global_nodes = subg.ndata[dgl.NID].cpu()
                z_out[global_nodes] = z_sub

            z_np = z_out.numpy()

        # -----------------------------
        # Assemble DataFrame
        # -----------------------------
        df = pd.DataFrame(z_np, index=cell_ids)
        df["sample_id"] = sample
        df["cell_id"] = cell_ids

        # Cell type handling
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
        elif not added_celltype:
            df["celltype"] = "unknown"

        # Optional metadata
        for key in ["cellsubtypes", "CNiche", "TNiche"]:
            if key in adata.obs.columns:
                df[key] = adata.obs.loc[df.index, key].values

        # Spatial coordinates
        if spatial_key in adata.obsm:
            coords = pd.DataFrame(
                adata.obsm[spatial_key],
                index=adata.obs_names,
                columns=["X_coord", "Y_coord"],
            )
            df[["X_coord", "Y_coord"]] = coords.loc[df.index].values
        else:
            df[["X_coord", "Y_coord"]] = np.nan

        # Ligand-receptor matrix
        if datapath is not None:
            lr_path = datapath / "training_LR_matrix.csv"
            if lr_path.exists():
                lr_df = pd.read_csv(lr_path, index_col=0)
                lr_df.index = lr_df.index.astype(str)
                df = pd.concat([df, lr_df.reindex(df.index)], axis=1)

        all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True)