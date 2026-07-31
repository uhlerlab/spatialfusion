# Generate SpatialFusion inputs

## Overview

Before running SpatialFusion, you need to generate unimodal embeddings from:

- spatial transcriptomics data using **scGPT** or **Nicheformer**
- H&E / whole-slide images using **UNI** or **Virchow2**

This step requires a GPU to run efficiently. SpatialFusion provides model-specific scripts, Docker images, and WDLs for each supported embedding model.

## Which workflow should I choose?

### WDL workflow
Best if you:

- do not have access to a GPU
- use a platform like Terra

Launch via Dockstore:

| Data type | Model | Dockstore workflow |
| --- | --- | --- |
| Spatial transcriptomics | scGPT | [scgpt-embeddings-for-spatialfusion](https://dockstore.org/workflows/github.com/uhlerlab/spatialfusion/scgpt-embeddings-for-spatialfusion:main?tab=info) |
| Spatial transcriptomics | Nicheformer | [nicheformer-embeddings-for-spatialfusion](https://dockstore.org/workflows/github.com/uhlerlab/spatialfusion/nicheformer-embeddings-for-spatialfusion:main?tab=info) |
| H&E / WSI | UNI | [uni-embeddings-for-spatialfusion](https://dockstore.org/workflows/github.com/uhlerlab/spatialfusion/uni-embeddings-for-spatialfusion:main?tab=info) |
| H&E / WSI | Virchow2 | [virchow2-embeddings-for-spatialfusion](https://dockstore.org/workflows/github.com/uhlerlab/spatialfusion/virchow2-embeddings-for-spatialfusion:main?tab=info) |

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

### Spatial transcriptomics embeddings

For scGPT, you need:

- `adata`: AnnData (`.h5ad`) file used for scGPT embeddings.
- `input_is_log_normalized`: whether the selected AnnData expression values are already log-normalized.

scGPT weights are bundled in the scGPT Docker image at `/app/scgpt_weights`. For the SpatialFusion tutorial data, use `False` for `input_is_log_normalized`.

For Nicheformer, you need:

- `adata`: AnnData (`.h5ad`) file used for Nicheformer embeddings.
- `technology`: one of `xenium`, `cosmx`, or `merfish`.

Nicheformer weights and reference files are bundled in the Nicheformer Docker image.

### H&E / WSI embeddings

For both H&E models, you need:

- `adata`: AnnData (`.h5ad`) file with spatial coordinates in `adata.obsm["spatial"]`.
- `wsi`: H&E / whole-slide image in TIFF / OME-TIFF format.

For UNI, you also need `uni_weights`, the UNI2-h `pytorch_model.bin` from <https://huggingface.co/MahmoodLab/UNI2-h>.

For Virchow2, you also need `virchow2_weights`, either `model.safetensors` or `pytorch_model.bin` from <https://huggingface.co/paige-ai/Virchow2>.


## 3. Set local paths

Pull the public Docker image for the model you want to run:

```bash
docker pull vanallenlab/scgpt-embeddings:workflow-0.1
docker pull vanallenlab/he-embeddings:workflow-0.1
docker pull vanallenlab/nicheformer-embeddings:workflow-0.1
```

Set local path variables (absolute paths):

```bash
ADATA=/absolute/path/to/object.h5ad
OUTPUT_DIR=/absolute/path/to/output

# H&E inputs
WSI=/absolute/path/to/image.ome.tif
UNI_WEIGHTS=/absolute/path/to/pytorch_model.bin
VIRCHOW2_WEIGHTS=/absolute/path/to/model.safetensors

# ST model settings
LOG_NORM="False"
TECHNOLOGY=xenium
```

## 4. Run embedding generation

### Spatial transcriptomics

#### Run scGPT

```bash
docker run --rm --gpus all \
  -v "$ADATA":/inputs/object.h5ad \
  -v "$OUTPUT_DIR":/out \
  vanallenlab/scgpt-embeddings:workflow-0.1 \
  python /app/embed_scgpt.py \
    --adata /inputs/object.h5ad \
    --input-is-log-normalized "$LOG_NORM" \
    --output-dir /out \
    --scgpt-weights /app/scgpt_weights
```

#### Run Nicheformer

```bash
docker run --rm --gpus all \
  -v "$ADATA":/inputs/object.h5ad \
  -v "$OUTPUT_DIR":/out \
  vanallenlab/nicheformer-embeddings:workflow-0.1 \
  python /app/embed_nicheformer.py \
    --adata /inputs/object.h5ad \
    --technology "$TECHNOLOGY" \
    --output-dir /out
```

### H&E / WSI

#### Run UNI

```bash
docker run --rm --gpus all \
  -v "$ADATA":/inputs/object.h5ad \
  -v "$WSI":/inputs/image.ome.tif \
  -v "$UNI_WEIGHTS":/weights/pytorch_model.bin \
  -v "$OUTPUT_DIR":/out \
  vanallenlab/he-embeddings:workflow-0.1 \
  python /app/embed_uni.py \
    --adata /inputs/object.h5ad \
    --wsi /inputs/image.ome.tif \
    --output-dir /out \
    --uni-weights /weights/pytorch_model.bin
```

#### Run Virchow2

```bash
docker run --rm --gpus all \
  -v "$ADATA":/inputs/object.h5ad \
  -v "$WSI":/inputs/image.ome.tif \
  -v "$VIRCHOW2_WEIGHTS":/weights/model.safetensors \
  -v "$OUTPUT_DIR":/out \
  vanallenlab/he-embeddings:workflow-0.1 \
  python /app/embed_virchow2.py \
    --adata /inputs/object.h5ad \
    --wsi /inputs/image.ome.tif \
    --output-dir /out \
    --virchow2-weights /weights/model.safetensors
```

## 5. Expected outputs
After successful execution, you should see the output file for the mode you ran:

```
$OUTPUT_DIR/
  scGPT.parquet
  UNI.parquet
  Virchow2.parquet
  nicheformer.parquet
```


## Notes
- This guide covers the most common use case with minimal inputs
- Additional optional parameters are available, see
[`embed_scgpt.py`](https://github.com/uhlerlab/spatialfusion/blob/main/workflows/unimodal-embeddings/scripts/embed_scgpt.py),
[`embed_uni.py`](https://github.com/uhlerlab/spatialfusion/blob/main/workflows/unimodal-embeddings/scripts/embed_uni.py),
[`embed_virchow2.py`](https://github.com/uhlerlab/spatialfusion/blob/main/workflows/unimodal-embeddings/scripts/embed_virchow2.py), and
[`embed_nicheformer.py`](https://github.com/uhlerlab/spatialfusion/blob/main/workflows/unimodal-embeddings/scripts/embed_nicheformer.py).
