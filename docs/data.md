# Data Format

SpatialFusion supports paired H&E and spatial transcriptomics data, H&E-only data, and RNA-only data. All modes require cell coordinates; provide H&E embeddings for H&E-only mode, RNA embeddings for RNA-only mode, or both for paired mode.

To generate H&E embeddings, the method requires:

- Image (WSI): (n_px_height, n_px_width)
- Coordinates of cells in image space: (n_cells, 2)

For paired or RNA-only data, the method accepts an AnnData object.
At minimum, the adata object must contain:

- adata.obsm['spatial_px']: this should contain the X and Y coordinates of each cell/nucleus (in high-resolution pixel space when using H&E). This is the default key expected by SpatialFusion; if your AnnData uses a different key, pass `spatial_key=<your_key>` to `run_full_embedding` (check available keys with `list(adata.obsm.keys())`).
- adata.X: this should be the cell x gene matrix of raw counts (! this needs to be single-cell resolution data)
- (optional): adata.obs['celltypes']: the annotated cell types. This is the default key; if your AnnData uses a different column name, pass `celltype_key=<your_key>` to `run_full_embedding` (check available columns with `adata.obs.columns.tolist()`).

SpatialFusion expects preprocessed and aligned data.
