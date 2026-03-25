# Generate SpatialFusion inputs

## Overview

Before running SpatialFusion, you need to generate unimodal embeddings from:

- spatial transcriptomics data → using **scGPT**
- H&E / whole-slide images → using **UNI**

These embeddings are required inputs to run SpatialFusion.

This step requires a GPU to run efficiently.

We provide two ways to run it:
- **no GPU** → use the WDL workflow (runs on platforms like Terra)
- **have a GPU** → run it yourself

## Which workflow should I choose?

### WDL workflow
Best if you:
- do not have access to a GPU
- use a platform like Terra

Launch via Dockstore: 
https://dockstore.org/workflows/github.com/uhlerlab/spatialfusion/unimodal-embeddings-for-spatialfusion:main?tab=info

### Local / self-managed GPU workflow (this guide) 

Best if you:
- have access to a GPU machine

---


The remainder of this guide covers the **local/ self-managed GPU workflow**

## 1. Requirements

Before running this step, you will need:

- a GPU-enabled machine (tested with NVIDIA Tesla T4)
- Docker installed


## 2. Gather the required files

Your inputs should include:

- `adata`: AnnData (`.h5ad`) used for scGPT embeddings and for the spatial coordinates consumed by UNI. Spatial coordinates are expected in `adata.obsm["spatial"]`.
- `wsi`: whole-slide image / H&E TIFF used to generate UNI image embeddings. TIFF / OME-TIFF format is expected.
- `scgpt_weights`: a directory containing `best_model.pt`, `args.json`, and `vocab.json`.
    - Download from https://doi.org/10.6084/m9.figshare.24747228
- `uni_weights`: the UNI model weights file `pytorch_model.bin`.
    - Request access and download from Mahmood Lab at <https://huggingface.co/MahmoodLab/UNI2-h>
- `input_is_log_normalized`: decide whether your AnnData expression values are already log-normalized. You will pass `True` if they are already log-normalized and `False` if they are not.


## 3. Set local paths

Pull the public Docker image:

```bash
docker pull vanallenlab/unimodal-embeddings:v0.1
```

Set local path variables (absolute paths):

```bash
ADATA=/absolute/path/to/object.h5ad
WSI=/absolute/path/to/image.ome.tif
SCGPT_WEIGHTS_DIR=/absolute/path/to/scgpt
UNI_WEIGHTS=/absolute/path/to/pytorch_model.bin
OUTPUT_DIR=/absolute/path/to/output
# Depends on your data
LOG_NORM="False"
```

Notes:

- `SCGPT_WEIGHTS_DIR` should point to a directory containing `best_model.pt`, `args.json`, and `vocab.json`.


## 4. Run embedding generation

```bash
docker run --rm --gpus all \
  -v "$ADATA":/inputs/object.h5ad \
  -v "$WSI":/inputs/image.ome.tif \
  -v "$SCGPT_WEIGHTS_DIR":/weights/scgpt \
  -v "$UNI_WEIGHTS":/weights/pytorch_model.bin \
  -v "$OUTPUT_DIR":/out \
  vanallenlab/unimodal-embeddings:v0.1 \
  python /app/unimodal-embeddings.py \
    --mode both \
    --adata /inputs/object.h5ad \
    --input-is-log-normalized "$LOG_NORM" \
    --wsi /inputs/image.ome.tif \
    --output-dir /out \
    --scgpt-weights /weights/scgpt \
    --uni-weights /weights/pytorch_model.bin
```

## 5. Expected outputs
After successful execution, you should see:

```
$OUTPUT_DIR/
  ├── scGPT.parquet
  └── UNI.parquet
```


## Notes
- This guide covers the most common use case with minimal inputs
- Additional optional parameters are available, see
[`unimodal-embeddings.py`](https://github.com/uhlerlab/spatialfusion/blob/mkdocs-update/workflows/unimodal-embeddings/scripts/unimodal-embeddings.py#L208)
