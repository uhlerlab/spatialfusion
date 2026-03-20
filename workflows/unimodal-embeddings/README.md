# Generate unimodal embeddings for Spatial Fusion

This workflow generates the unimodal embedding inputs used by SpatialFusion:

- `scGPT.parquet` from spatial transcriptomics data with `scGPT`
- `UNI.parquet` from H&E / whole-slide imaging data with `UNI`

## 1. What you need

Before running this step, you need:

- access to a GPU-enabled machine (note: we tested this using a NVIDIA Tesla T4)
- Docker
- your own input data:
  - an AnnData `.h5ad` file
  - an H&E / whole-slide image in TIFF / OME-TIFF format
- model weights for both embedding models:
  - `scGPT` weights
  - `UNI` weights

## 2. Gather the required files

Your inputs should look like this:

- `adata`: AnnData (`.h5ad`) used for scGPT embeddings and for the spatial coordinates consumed by UNI. Spatial coordinates are expected in `adata.obsm["spatial"]`.
- `wsi`: whole-slide image / H&E TIFF used to generate UNI image embeddings. TIFF / OME-TIFF format is expected.
- `scgpt_weights`: a directory containing `best_model.pt`, `args.json`, and `vocab.json`.
- `uni_weights`: the UNI model weights file `pytorch_model.bin`.
- `input_is_log_normalized`: decide whether your AnnData expression values are already log-normalized. You will pass `True` if they are already log-normalized and `False` if they are not.

To get the model weights:

- `scgpt_weights`: download `best_model.pt`, `args.json`, and `vocab.json` from the figshare dataset accompanying *Assessing the limits of zero-shot foundation models in single-cell biology* (DOI: <https://doi.org/10.6084/m9.figshare.24747228>), then place those three files in one directory.
- `uni_weights`: request access to the UNI2-h weights from Mahmood Lab at <https://huggingface.co/MahmoodLab/UNI2-h>, then download `pytorch_model.bin`.

## 3. Set local paths

Pull the public Docker image:

```bash
docker pull vanallenlab/unimodal-embeddings:v0.1
```

Set local path variables for each required input. These should be absolute paths.

```bash
ADATA=/absolute/path/to/object.h5ad
WSI=/absolute/path/to/image.ome.tif
SCGPT_WEIGHTS_DIR=/absolute/path/to/scgpt
UNI_WEIGHTS=/absolute/path/to/pytorch_model.bin
OUTPUT_DIR=/absolute/path/to/output
```

Notes:

- `SCGPT_WEIGHTS_DIR` should point to a directory containing `best_model.pt`, `args.json`, and `vocab.json`.

## 4. Run the Docker command

```bash
docker run --rm --gpus all \
  -v "$ADATA":/inputs/object.h5ad \
  -v "$WSI":/inputs/image.ome.tif \
  -v "$SCGPT_WEIGHTS_DIR":/weights/scgpt \
  -v "$UNI_WEIGHTS":/weights/pytorch_model.bin \
  -v "$OUTPUT_DIR":/out \
  vanallenlab/unimodal-embeddings:latest \
  python /app/unimodal-embeddings.py \
  --mode both \
  --adata /inputs/object.h5ad \
  --input-is-log-normalized False \
  --wsi /inputs/image.ome.tif \
  --output-dir /out \
  --scgpt-weights /weights/scgpt \
  --uni-weights /weights/pytorch_model.bin
```

This will write:

- `/out/scGPT.parquet`
- `/out/UNI.parquet`


## Notes

- This README shows the minimal inputs for the common case. The script exposes additional optional parameters for advanced use; see `scripts/unimodal-embeddings.py` for the full CLI.
