# Generate unimodal embeddings for Spatial Fusion

This workflow generates the unimodal embedding inputs used by SpatialFusion:

- `scGPT.parquet` from spatial transcriptomics data with `scGPT`
- `UNI.parquet` from H&E / whole-slide imaging data with `UNI`

## 1. What you need

Before running this step, you need:

- access to GPUs (we tested this on NVIDIA Tesla T4 GPU)
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
- `input_is_log_normalized`: before running the script, decide whether your AnnData expression values are already log-normalized. You will pass `True` if they are already log-normalized and `False` if they are not.

To get the model weights:

- `scgpt_weights`: download `best_model.pt`, `args.json`, and `vocab.json` from the figshare dataset accompanying *Assessing the limits of zero-shot foundation models in single-cell biology* (DOI: <https://doi.org/10.6084/m9.figshare.24747228>), then place those three files in one directory.
- `uni_weights`: request access to the UNI2-h weights from Mahmood Lab at <https://huggingface.co/MahmoodLab/UNI2-h>, then download `pytorch_model.bin`.

## 3. Set local paths

Pull the public Docker image:

```bash
docker pull vanallenlab/unimodal-embeddings:latest
```

Set local path variables for your data, weights, and output directory:

```bash
DATA_DIR=/path/to/data
WEIGHTS_DIR=/path/to/weights
OUTPUT_DIR=/path/to/output
```

Expected contents:

- `$DATA_DIR/object.h5ad`
- `$DATA_DIR/image.ome.tif`
- `$WEIGHTS_DIR/scgpt/` containing `best_model.pt`, `args.json`, and `vocab.json`
- `$WEIGHTS_DIR/uni2/pytorch_model.bin`

## 4. Run the Docker command

```bash
docker run --rm --gpus all \
  -v "$DATA_DIR":/data \
  -v "$WEIGHTS_DIR":/weights \
  -v "$OUTPUT_DIR":/out \
  vanallenlab/unimodal-embeddings:latest \
  python /app/unimodal-embeddings.py \
  --mode both \
  --adata /data/object.h5ad \
  --input-is-log-normalized False \
  --wsi /data/image.ome.tif \
  --output-dir /out \
  --scgpt-weights /weights/scgpt \
  --uni-weights /weights/uni2/pytorch_model.bin
```

This will write:

- `/out/scGPT.parquet`
- `/out/UNI.parquet`
