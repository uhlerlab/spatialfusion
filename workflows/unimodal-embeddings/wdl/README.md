# Unimodal Embeddings for SpatialFusion

This workflow generates the unimodal embedding inputs used by SpatialFusion from:

- spatial transcriptomics data with `scGPT`
- H&E / whole-slide imaging data with `UNI`

By default, the workflow runs both embedding modes in parallel and returns:

- `scGPT.parquet`
- `UNI.parquet`

## Main inputs

- `adata`: AnnData (`.h5ad`) file with spatial coordinates stored in `adata.obsm["spatial"]`
- `input_is_log_normalized`: whether the AnnData expression values (.X) are already log-normalized
- `wsi`: H&E / whole-slide image in TIFF / OME-TIFF format
- `uni_weights`: UNI model weights file (`pytorch_model.bin`) from Mahmood Lab. See Notes section on how to obtain access.

## Notes

- `UNI` weights are not bundled with the workflow. Users must request access to the UNI2-h weights from Mahmood Lab at <https://huggingface.co/MahmoodLab/UNI2-h>, upload the weights file to an accessible `gs://` bucket, and provide that path as the `uni_weights` input.
- The default `scgpt_weights` archive is derived from the demo weights released with the figshare dataset accompanying *Assessing the limits of zero-shot foundation models in single-cell biology* (DOI: <https://doi.org/10.6084/m9.figshare.24747228>).
- To run only one embedding mode, set `embedding_mode` to `scgpt` or `uni`.
