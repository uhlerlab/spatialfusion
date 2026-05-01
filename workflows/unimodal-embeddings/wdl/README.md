# Unimodal Embeddings for SpatialFusion

This workflow generates the unimodal embedding inputs used by SpatialFusion from:

- spatial transcriptomics data with `scGPT`
- H&E / whole-slide imaging data with `UNI`

By default, the workflow runs both embedding modes in parallel and returns:

- `scGPT.parquet`
- `UNI.parquet`

## Main inputs

- `adata`: AnnData (`.h5ad`) file with spatial coordinates stored in `adata.obsm["spatial"]`
- `input_is_log_normalized`: whether the AnnData expression values (.X) are already log-normalized. Required for `both` and `scgpt` runs.
- `embedding_mode`: which embeddings to generate: `both`, `scgpt`, or `uni`
- `wsi`: H&E / whole-slide image in TIFF / OME-TIFF format. Required for `both` and `uni` runs.
- `uni_weights`: UNI model weights file (`pytorch_model.bin`) from Mahmood Lab. Required for `both` and `uni` runs. See Notes section on how to obtain access.

The default/common run uses `embedding_mode = "both"`. For single-modality runs, start from:

- `unimodal_embeddings_for_spatialfusion_scgpt_inputs.json` for scGPT-only
- `unimodal_embeddings_for_spatialfusion_uni_inputs.json` for UNI-only

## Notes

- `UNI` weights are not bundled with the workflow. Users must request access to the UNI2-h weights from Mahmood Lab at <https://huggingface.co/MahmoodLab/UNI2-h>, upload the weights file to an accessible `gs://` bucket, and provide that path as the `uni_weights` input.
- `scgpt_weights` defaults to a VA lab `gs://` archive derived from the demo weights released with the figshare dataset accompanying *Assessing the limits of zero-shot foundation models in single-cell biology* (DOI: <https://doi.org/10.6084/m9.figshare.24747228>). If you do not have access to that bucket, download `best_model.pt`, `args.json`, and `vocab.json` from the figshare source, package them into a `.tar.gz`, upload that archive to an accessible `gs://` bucket, and override the `scgpt_weights` input with your own path.
- To run only one embedding mode, set `embedding_mode` to `scgpt` or `uni`. The workflow only localizes the files used by the selected mode, so UNI-only runs do not need `scgpt_weights` or `input_is_log_normalized`.
