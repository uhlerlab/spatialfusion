# Unimodal Embeddings for SpatialFusion

These WDL workflows generate the unimodal embedding inputs used by SpatialFusion:

- `GenerateScgptEmbeddingsForSpatialFusion` generates `scGPT.parquet` from spatial transcriptomics data.
- `GenerateNicheformerEmbeddingsForSpatialFusion` generates `nicheformer.parquet` from spatial transcriptomics data.
- `GenerateUniEmbeddingsForSpatialFusion` generates `UNI.parquet` from H&E / whole-slide imaging data.
- `GenerateVirchow2EmbeddingsForSpatialFusion` generates `Virchow2.parquet` from H&E / whole-slide imaging data.

Use the model-specific workflow for each embedding you want to generate. The H&E workflows share a Docker image, but keep separate WDLs so each workflow has a small, explicit input surface.

## scGPT Inputs

- `adata`: AnnData (`.h5ad`) file used for scGPT embeddings.
- `input_is_log_normalized`: whether the AnnData expression values in the selected layer are already log-normalized.

scGPT weights are bundled in the workflow Docker image at `/app/scgpt_weights`, so users do not need to provide a scGPT weights input.

## Nicheformer Inputs

- `adata`: AnnData (`.h5ad`) file used for Nicheformer embeddings.
- `technology`: required spatial transcriptomics technology used to select bundled Nicheformer defaults. Choose one of: `xenium`, `cosmx`, or `merfish`.

The Nicheformer Docker image bundles the Nicheformer checkpoint (`nicheformer.ckpt`), vocabulary AnnData file (`model.h5ad`), technology mean files, and GTF annotation file. Users provide only the AnnData input and technology label at runtime.

## UNI Inputs

- `adata`: AnnData (`.h5ad`) file with spatial coordinates stored in `adata.obsm["spatial"]`.
- `wsi`: H&E / whole-slide image in TIFF / OME-TIFF format.
- `uni_weights`: UNI2-h model weights file (`pytorch_model.bin`) from Mahmood Lab.

UNI weights are not bundled with the workflow. Users must request access to the UNI2-h weights from Mahmood Lab at <https://huggingface.co/MahmoodLab/UNI2-h>, upload the weights file to an accessible `gs://` bucket, and provide that path as the `uni_weights` input.

## Virchow2 Inputs

- `adata`: AnnData (`.h5ad`) file with spatial coordinates stored in `adata.obsm["spatial"]`.
- `wsi`: H&E / whole-slide image in TIFF / OME-TIFF format.
- `virchow2_weights`: Virchow2 model weights file, usually `model.safetensors` or `pytorch_model.bin`, from Paige AI.

Virchow2 weights are not bundled with the workflow. Users must request access to the Virchow2 weights from Paige AI at <https://huggingface.co/paige-ai/Virchow2>, upload the weights file to an accessible `gs://` bucket, and provide that path as the `virchow2_weights` input.

## Notes

- H&E workflows read coordinates from `adata.obsm["spatial"]` by default. Advanced users can override this with `spatial_key`.
- Runtime overrides are available for memory, CPU, disk, and preemptible tries.
