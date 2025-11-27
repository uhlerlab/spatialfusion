# SpatialFusion

**SpatialFusion** is a Python package for deep learning–based analysis of spatial omics data.
It provides a lightweight framework that integrates **spatial transcriptomics (ST)** with **H&E histopathology** to learn **joint multimodal embeddings** of cellular neighborhoods and group them into **spatial niches**.

The method operates at **single-cell resolution**, and can be applied to:

* paired ST + H&E datasets
* H&E whole-slide images alone

By combining molecular and morphological features, SpatialFusion captures coordinated patterns of tissue architecture and gene expression. A key design principle is a biologically informed definition of niches: not simply spatial neighborhoods, but **reproducible microenvironments** characterized by pathway-level activation signatures and functional coherence across tissues. To reflect this prior, the latent space of the model is trained to encode biologically meaningful pathway activations, enabling robust discovery of integrated niches.

The method is described in the paper: **XXX** (citation forthcoming).

---

## Installation

We provide pretrained weights for the **multimodal autoencoder (AE)** and **graph convolutional masked autoencoder (GCN)** under `data/`.

SpatialFusion depends on **PyTorch** and **DGL**, which have different builds for CPU and GPU systems. You can install it using **pip** or inside a **conda/mamba** environment.

---

### 1. Quick Setup

```bash
mamba create -n spatialfusion python=3.10 -y
mamba activate spatialfusion
# Then install GPU or CPU version below
```

---

### GPU (CUDA 12.4)

```bash
pip install "torch==2.4.1" "torchvision==0.19.1" \
  --index-url https://download.pytorch.org/whl/cu124
conda install -c dglteam/label/th24_cu124 dgl
cd spatialfusion/
pip install -e .
```

**Note:** TorchText issues exist for this version:
[https://github.com/pytorch/text/issues/2272](https://github.com/pytorch/text/issues/2272) — this may affect scGPT.

---

### GPU (CUDA 12.1) — *Recommended if using scGPT*

```bash
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
  --index-url https://download.pytorch.org/whl/cu121
conda install -c dglteam/label/th21_cu121 dgl

# Optional: embeddings used by scGPT
pip install --no-cache-dir torchtext==0.18.0 torchdata==0.9.0

# Optional: UNI (H&E embedding model)
pip install timm

cd spatialfusion/
pip install -e .
```

---

### CPU-only

```bash
pip install "torch==2.4.1" "torchvision==0.19.1" \
  --index-url https://download.pytorch.org/whl/cpu
conda install -c dglteam -c conda-forge dgl

# Optional, used for scGPT
pip install --no-cache-dir torchtext==0.18.0 torchdata==0.9.0

# Optional, used for UNI
pip install timm

cd spatialfusion/
pip install -e .
```

> 💡 Replace `cu124` with the CUDA version matching your system (e.g., `cu121`).

---

### 2. Development Install (Optional)

```bash
pip install -e ".[dev,docs]"
```

Includes: **pytest**, **black**, **ruff**, **sphinx**, **matplotlib**, **seaborn**.

---

### 3. Verify Installation

```bash
python - <<'PY'
import torch, dgl, spatialfusion
print("Torch:", torch.__version__, "CUDA available:", torch.cuda.is_available())
print("DGL:", dgl.__version__)
print("SpatialFusion OK")
PY
```

---

### 4. Notes

* Default output directory is:

  ```
  $HOME/spatialfusion_runs
  ```

  Override with:

  ```
  export SPATIALFUSION_ROOT=/your/path
  ```
* CPU installations work everywhere but are significantly slower.

---

## Usage Example

A minimal example showing how to embed a dataset using the pretrained AE and GCN:

```python
from spatialfusion.embed.embed import AEInputs, run_full_embedding
import pandas as pd
import pathlib as pl

# Load external embeddings (UNI + scGPT)
uni_df = pd.read_parquet('UNI.parquet')
scgpt_df = pd.read_parquet('scGPT.parquet')

# Paths to pretrained models
ae_model_dir = pl.Path('../data/checkpoint_dir_ae/')
gcn_model_dir = pl.Path('../data/checkpoint_dir_gcn/')

# Mapping sample_name -> AEInputs
ae_inputs_by_sample = {
    sample_name: AEInputs(
        adata=adata,
        z_uni=uni_df,
        z_scgpt=scgpt_df,
    ),
}

# Run the multimodal embedding pipeline
emb_df = run_full_embedding(
    ae_inputs_by_sample=ae_inputs_by_sample,
    ae_model_path=ae_model_dir / "spatialfusion-multimodal-ae.pt",
    gcn_model_path=gcn_model_dir / "spatialfusion-full-gcn.pt",
    device="cuda:0",
    combine_mode="average",
    spatial_key='spatial',
    celltype_key='major_celltype',
    save_ae_dir=None,  # optional
)
```

This produces a DataFrame containing the final integrated embedding for all cells/nuclei.

---

## Required Inputs

SpatialFusion operates on a **single-cell AnnData object** paired with an **H&E whole-slide image**.

### **AnnData fields**

| Key                                | Description                                                       |
| ---------------------------------- | ----------------------------------------------------------------- |
| `adata.obsm['spatial']`            | X/Y centroid coordinates of each cell/nucleus in WSI pixel space. |
| `adata.X`                          | Raw counts (cell × gene). Must be single-cell resolution.         |
| `adata.obs['celltype']` (optional) | Annotated cell types (`major_celltype` in examples).              |

### **Whole-Slide Image (WSI)**

A high-resolution H&E image corresponding to the same tissue section used for ST.
Used to compute morphology embeddings such as **UNI**.

---

## Typical Workflow

1. **Prepare ST AnnData and the matched H&E WSI**
2. **Run scGPT** to compute molecular embeddings
3. **Run UNI** to compute morphology embeddings
4. **Run SpatialFusion** to integrate all modalities into joint embeddings
5. **Cluster & visualize**

   * Leiden clustering
   * UMAP
   * Spatial niche maps

---

## Tutorials

A complete tutorial notebook is available at:

```
tutorials/embed-and-finetune-sample.ipynb
```

Additional required packages (scGPT, UNI dependencies) must be installed manually.
Follow the instructions at: [https://github.com/bowang-lab/scGPT](https://github.com/bowang-lab/scGPT)

We also provide a ready-to-use environment file:

```
spatialfusion_env.yml
```

Tutorial data is available on Zenodo:
[https://zenodo.org/records/17594071](https://zenodo.org/records/17594071)

---

## Repository Structure

```
.
├── data
│   ├── checkpoint_dir_ae
│   │   └── spatialfusion-multimodal-ae.pt
│   └── checkpoint_dir_gcn
│       ├── spatialfusion-full-gcn.pt
│       └── spatialfusion-he-gcn.pt
├── LICENSE
├── pyproject.toml
├── README.md
├── src
│   └── spatialfusion
│       ├── embed/
│       ├── finetune/
│       ├── models/
│       └── utils/
├── tests
│   ├── test_basic.py
│   ├── test_finetune.py
│   └── test_imports.py
└── tutorials
    ├── data
    └── embed-and-finetune-sample.ipynb
```

**Highlights:**

* **data/** — pretrained AE and GCN checkpoints
* **src/spatialfusion/** — main library modules

  * **embed/** — embedding utilities & pipeline
  * **finetune/** — niche-level finetuning
  * **models/** — neural network architectures
  * **utils/** — loaders, graph utilities, checkpoint code
* **tests/** — basic test suite
* **tutorials/** — practical examples and sample data

---

## Citing

If you use SpatialFusion, please cite:

> Broad Institute Spatial Foundation, *SpatialFusion* (2025).
> [https://github.com/broadinstitute/spatialfusion](https://github.com/broadinstitute/spatialfusion)

Full manuscript citation will be added when available.

---

## Version

### Version

This is the initial public release (**v0.1.0**).

---

## License

MIT License. See `LICENSE` for details.