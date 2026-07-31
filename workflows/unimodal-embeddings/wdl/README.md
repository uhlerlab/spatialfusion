# Unimodal Embeddings for SpatialFusion

These WDL workflows generate the unimodal embedding inputs used by SpatialFusion:

- `GenerateUnimodalEmbeddingsForSpatialFusion` generates both `scGPT.parquet` and `UNI.parquet` from one Terra submission.
- `GenerateScgptEmbeddingsForSpatialFusion` generates `scGPT.parquet` from spatial transcriptomics data.
- `GenerateUniEmbeddingsForSpatialFusion` generates `UNI.parquet` from H&E / whole-slide imaging data.
- `GenerateNicheformerEmbeddingsForSpatialFusion` generates `nicheformer.parquet` from spatial transcriptomics data.

Use the combined workflow if you want both outputs from one Terra submission. Use the modality-specific workflows for scGPT-only or UNI-only runs.

## Combined Inputs

- `adata`: AnnData (`.h5ad`) file used for scGPT embeddings and for the spatial coordinates consumed by UNI. Spatial coordinates are expected in `adata.obsm["spatial"]`.
- `input_is_log_normalized`: whether the AnnData expression values in the selected layer are already log-normalized.
- `wsi`: H&E / whole-slide image in TIFF / OME-TIFF format.
- `uni_weights`: UNI model weights file (`pytorch_model.bin`) from Mahmood Lab.

## scGPT Inputs

- `adata`: AnnData (`.h5ad`) file used for scGPT embeddings.
- `input_is_log_normalized`: whether the AnnData expression values in the selected layer are already log-normalized.

scGPT weights are bundled in the workflow Docker image at `/app/scgpt_weights`, so users do not need to provide a scGPT weights input.

## UNI Inputs

- `adata`: AnnData (`.h5ad`) file with spatial coordinates stored in `adata.obsm["spatial"]`.
- `wsi`: H&E / whole-slide image in TIFF / OME-TIFF format.
- `uni_weights`: UNI model weights file (`pytorch_model.bin`) from Mahmood Lab.

UNI weights are not bundled with the workflow. Users must request access to the UNI2-h weights from Mahmood Lab at <https://huggingface.co/MahmoodLab/UNI2-h>, upload the weights file to an accessible `gs://` bucket, and provide that path as the `uni_weights` input.

## Nicheformer Inputs

- `adata`: AnnData (`.h5ad`) file used for Nicheformer embeddings.
- `technology`: required spatial transcriptomics technology used to select bundled Nicheformer defaults. Choose one of: `xenium`, `cosmx`, or `merfish`.

The prototype Nicheformer Docker image bundles the Nicheformer checkpoint (`nicheformer.ckpt`), vocabulary AnnData file (`model.h5ad`), technology mean files, and GTF annotation file. Users provide only the AnnData input and technology label at runtime.
