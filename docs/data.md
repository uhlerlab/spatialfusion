# Data Format

Note: This method requires at least the H&E high-resolution image and X and Y coordinates extracted through segmentation (see the tutorial on extracting embeddings from H&E only). Paired ST data is optional.

The method requires:

- Image (WSI): (n_px_height, n_px_width)
- Coordinates of cells in image space: (n_cells, 2)

If providing paired ST data, the method accepts an AnnData object. 
At minimum, the adata object must contain:

- adata.obsm['spatial_px']: this should contain the X and Y coordinates of the cell/nuclei centroid in the high-resolution pixel space of the associated WSI. This is the default key expected by SpatialFusion; if your AnnData uses a different key, pass `spatial_key=<your_key>` to `run_full_embedding` (check available keys with `list(adata.obsm.keys())`).
- adata.X: this should be the cell x gene matrix of raw counts (! this needs to be single-cell resolution data)
- (optional): adata.obs['celltypes']: the annotated cell types. This is the default key; if your AnnData uses a different column name, pass `celltype_key=<your_key>` to `run_full_embedding` (check available columns with `adata.obs.columns.tolist()`).

SpatialFusion expects preprocessed and aligned data.

