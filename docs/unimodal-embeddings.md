# Generate SpatialFusion inputs

## Overview

Before running SpatialFusion, you need to generate unimodal embeddings from:

- spatial transcriptomics data → using **scGPT**
- H&E / whole-slide images → using **UNI**


This step requires a GPU to run efficiently and we provide two ways to run it.

## Which workflow should I choose?

### WDL workflow
Best if you:

- do not have access to a GPU
- use a platform like Terra

Launch via Dockstore:

- Both scGPT and UNI embeddings: <https://dockstore.org/workflows/github.com/uhlerlab/spatialfusion/unimodal-embeddings-for-spatialfusion:main?tab=info>
- scGPT embeddings: <https://dockstore.org/workflows/github.com/uhlerlab/spatialfusion/scgpt-embeddings-for-spatialfusion:main?tab=info>
- UNI embeddings: <https://dockstore.org/workflows/github.com/uhlerlab/spatialfusion/uni-embeddings-for-spatialfusion:main?tab=info>

### Local / self-managed GPU workflow (this guide) 

Best if you:

- have access to a GPU machine

---


The remainder of this guide covers the **local/ self-managed GPU workflow**.

## 1. Requirements

Before running this step, you will need:

- a GPU-enabled machine (tested with NVIDIA Tesla T4)
- Docker installed


## 2. Gather the required files

Your inputs should include:

- `adata`: AnnData (`.h5ad`) used for scGPT embeddings and for the spatial coordinates consumed by UNI. Spatial coordinates are expected in `adata.obsm["spatial"]`.
- `wsi`: whole-slide image / H&E TIFF used to generate UNI image embeddings. TIFF / OME-TIFF format is expected.
- `uni_weights`: the UNI model weights file `pytorch_model.bin`.
    - Request access and download from Mahmood Lab at <https://huggingface.co/MahmoodLab/UNI2-h>
- `input_is_log_normalized`: decide whether your AnnData expression values are already log-normalized. You will pass `True` if they are already log-normalized and `False` if they are not.

scGPT weights are bundled in the Docker image.


## 3. Set local paths

Pull the public Docker image:

```bash
docker pull vanallenlab/unimodal-embeddings:v0.2
```

Set local path variables (absolute paths):

```bash
ADATA=/absolute/path/to/object.h5ad
WSI=/absolute/path/to/image.ome.tif
UNI_WEIGHTS=/absolute/path/to/pytorch_model.bin
OUTPUT_DIR=/absolute/path/to/output
# Depends on your data
LOG_NORM="False"
```

## 4. Run embedding generation

### Run both embeddings

```bash
docker run --rm --gpus all \
  -v "$ADATA":/inputs/object.h5ad \
  -v "$WSI":/inputs/image.ome.tif \
  -v "$UNI_WEIGHTS":/weights/pytorch_model.bin \
  -v "$OUTPUT_DIR":/out \
  vanallenlab/unimodal-embeddings:v0.2 \
  python /app/unimodal-embeddings.py \
    --mode both \
    --adata /inputs/object.h5ad \
    --input-is-log-normalized "$LOG_NORM" \
    --wsi /inputs/image.ome.tif \
    --output-dir /out \
    --scgpt-weights /app/scgpt_weights \
    --uni-weights /weights/pytorch_model.bin
```

### Run only scGPT

```bash
docker run --rm --gpus all \
  -v "$ADATA":/inputs/object.h5ad \
  -v "$OUTPUT_DIR":/out \
  vanallenlab/unimodal-embeddings:v0.2 \
  python /app/unimodal-embeddings.py \
    --mode scgpt \
    --adata /inputs/object.h5ad \
    --input-is-log-normalized "$LOG_NORM" \
    --output-dir /out \
    --scgpt-weights /app/scgpt_weights
```

### Run only UNI

```bash
docker run --rm --gpus all \
  -v "$ADATA":/inputs/object.h5ad \
  -v "$WSI":/inputs/image.ome.tif \
  -v "$UNI_WEIGHTS":/weights/pytorch_model.bin \
  -v "$OUTPUT_DIR":/out \
  vanallenlab/unimodal-embeddings:v0.2 \
  python /app/unimodal-embeddings.py \
    --mode uni \
    --adata /inputs/object.h5ad \
    --wsi /inputs/image.ome.tif \
    --output-dir /out \
    --uni-weights /weights/pytorch_model.bin
```

## 5. Expected outputs
After successful execution, you should see the output file for the mode you ran:

```
$OUTPUT_DIR/
  ├── scGPT.parquet
  └── UNI.parquet
```


## Notes
- This guide covers the most common use case with minimal inputs
- Additional optional parameters are available, see
[`unimodal-embeddings.py`](https://github.com/uhlerlab/spatialfusion/blob/main/workflows/unimodal-embeddings/scripts/unimodal-embeddings.py)
