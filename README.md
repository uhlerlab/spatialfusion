# SpatialFusion

[![Docs](https://img.shields.io/badge/docs-online-blue.svg)](
https://uhlerlab.github.io/spatialfusion/
)

**SpatialFusion** is a Python package for deep learning–based analysis of spatial omics data.
It provides a lightweight framework that integrates **spatial transcriptomics (ST)** with **H&E histopathology** to learn **joint multimodal embeddings** of cellular neighborhoods and group them into **spatial niches**.

The method operates at **single-cell resolution**, and can be applied to:

* paired ST + H&E datasets
* H&E whole-slide images alone
* spatial transcriptomics data alone

By combining molecular and morphological features, SpatialFusion captures coordinated patterns of tissue architecture and gene expression. A key design principle is a biologically informed definition of niches: not simply spatial neighborhoods, but **reproducible microenvironments** characterized by pathway-level activation signatures and functional coherence across tissues. To reflect this prior, the latent space of the model is trained to encode biologically meaningful pathway activations, enabling robust discovery of integrated niches.

The method is described in the paper: [SpatialFusion: A lightweight multimodal framework for pathway-informed spatial niche mapping](https://doi.org/10.64898/2026.03.16.712056).

You can find detailed documentation at https://uhlerlab.github.io/spatialfusion/

---
## Prepare SpatialFusion inputs

Before running SpatialFusion, you need to generate unimodal embeddings from two modalities. You can choose which foundation model to use for each:

| Modality | Supported models |
| -------- | --------------- |
| H&E / whole-slide image | [UNI](https://huggingface.co/MahmoodLab/UNI2-h) · [Virchow](https://huggingface.co/paige-ai/Virchow2) |
| Spatial transcriptomics | [scGPT](https://github.com/bowang-lab/scGPT) · [Nicheformer](https://github.com/theislab/nicheformer) |

We provide Dockstore workflows for generating these embeddings, detailed on our [documentation website](https://uhlerlab.github.io/spatialfusion/unimodal-embeddings/).
## Installation

### 1. Create virtual environment

```bash
mamba create -n spatialfusion python=3.10 -y
mamba activate spatialfusion
```

### 2. Install platform-specific libraries

SpatialFusion depends on PyTorch and DGL, which have different builds for CPU and GPU systems. 

#### CPU

```bash
pip install "torch==2.4.1" "torchvision==0.19.1" \
  --index-url https://download.pytorch.org/whl/cpu

pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/repo.html
```

#### GPU (CUDA 12.4)

```bash
pip install "torch==2.4.1" "torchvision==0.19.1" \
  --index-url https://download.pytorch.org/whl/cu124

pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
```
---

### 3. Install SpatialFusion package

#### Basic installation — *Recommended for users*
```bash
pip install spatialfusion
```
---
#### Install from source - *Recommended for contributors*
Includes: `pytest`, `black`, `ruff`, `sphinx`, `matplotlib`, `seaborn`.

```bash
git clone https://github.com/uhlerlab/spatialfusion.git

cd spatialfusion/

pip install -e ".[dev,docs]"
```



---

### 4. Verify Installation

```bash
python - <<'PY'
import torch, dgl, spatialfusion
print("Torch:", torch.__version__, "CUDA available:", torch.cuda.is_available())
print("DGL:", dgl.__version__)
print("SpatialFusion OK")
PY
```

---

### 5. Notes

* Default output directory is:

  ```
  $HOME/spatialfusion_runs
  ```

  Override with:

  ```
  export SPATIALFUSION_ROOT=/your/path
  ```
* CPU installations work everywhere but are significantly slower.
Estimated install time: 15-30 min on a 'normal' desktop.

**Tested system configuration**
OS: Rocky Linux 8.10 (Green Obsidian), x86_64
CPU: AMD EPYC 7763 64-Core Processor
GPU used: 1 × NVIDIA RTX A6000
GPU memory: 48 GiB VRAM
NVIDIA driver: 570.124.04

---

## Tutorials

A complete tutorial notebook to embed / finetune SpatialFusion on a sample is available at:

```
tutorials/embed-and-finetune-sample.ipynb
```

To get unimodal embeddings, we provide instructions or a tutorial at: 

```
tutorials/get-unimodal-embeddings-sample.ipynb
```

To run SpatialFusion on H&E only, we provide a tutorial at: 

```
tutorials/embed-he-only.ipynb
```

To run SpatialFusion on RNA only, we provide a tutorial at:

```
tutorials/embed-rna-only.ipynb
```

Additional packages for the foundation models you choose must be installed manually:

- **scGPT**: [https://github.com/bowang-lab/scGPT](https://github.com/bowang-lab/scGPT)
- **Nicheformer**: [https://github.com/theislab/nicheformer](https://github.com/theislab/nicheformer)
- **UNI**: [https://huggingface.co/MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h)
- **Virchow**: [https://huggingface.co/paige-ai/Virchow2](https://huggingface.co/paige-ai/Virchow2)

We also provide a ready-to-use environment file:

```
spatialfusion_env.yml
```

Tutorial data is available on Zenodo:
[https://zenodo.org/records/17594071](https://zenodo.org/records/17594071)

The estimated tutorial time running on a single GPU on the tutorial data, given FM embeddings are already computed, is 5-10 min. 

---

## Pretrained Weights

SpatialFusion ships pretrained checkpoints for every combination of supported foundation models. Select the pair that matches the embeddings you generated:

| H&E model | RNA model | AE checkpoint | GCN checkpoint |
| --------- | --------- | ------------- | -------------- |
| UNI | scGPT | `spatialfusion-ae-uni-scgpt.pt` | `spatialfusion-gcn-uni-scgpt.pt` |
| UNI | Nicheformer | `spatialfusion-ae-uni-nicheformer.pt` | `spatialfusion-gcn-uni-nicheformer.pt` |
| Virchow | scGPT | `spatialfusion-ae-virchow-scgpt.pt` | `spatialfusion-gcn-virchow-scgpt.pt` |
| Virchow | Nicheformer | `spatialfusion-ae-virchow-nicheformer.pt` | `spatialfusion-gcn-virchow-nicheformer.pt` |
| UNI | *(H&E only)* | — | `spatialfusion-he-gcn-uni-scgpt.pt` |
| *(RNA only)* | scGPT | — | `spatialfusion-rna-gcn-uni-scgpt.pt` |

All checkpoints are bundled with the package under `src/spatialfusion/data/` and are resolved automatically via `spatialfusion.utils.pkg_ckpt.resolve_pkg_ckpt`.

---

## Usage Example

A minimal example showing how to embed a dataset using the pretrained AE and GCN. Swap in the checkpoint filenames that match your chosen foundation models (see the table above):

```python
from spatialfusion.embed.embed import AEInputs, run_full_embedding
from spatialfusion.utils.pkg_ckpt import resolve_pkg_ckpt
import pandas as pd
import scanpy as sc

# Load H&E embeddings (UNI or Virchow — use whichever you generated)
he_df = pd.read_parquet('UNI.parquet')       # or 'Virchow2.parquet'

# Load RNA embeddings (scGPT or Nicheformer — use whichever you generated)
rna_df = pd.read_parquet('scGPT.parquet')    # or 'nicheformer.parquet'

# Load AnnData object
adata = sc.read_h5ad("object.h5ad")

# Mapping sample_name -> AEInputs
# z_he holds H&E embeddings (UNI or Virchow); z_rna holds RNA embeddings (scGPT or Nicheformer)
sample_name = 'sample1'
ae_inputs_by_sample = {
    sample_name: AEInputs(
        adata=adata,
        z_he=he_df,
        z_rna=rna_df,
    ),
}

# Select the checkpoints matching your FM combination (e.g. UNI + scGPT shown here)
ae_ckpt  = resolve_pkg_ckpt("checkpoint_dir_ae/spatialfusion-ae-uni-scgpt.pt")
gcn_ckpt = resolve_pkg_ckpt("checkpoint_dir_gcn/spatialfusion-gcn-uni-scgpt.pt")

# Run the multimodal embedding pipeline
# spatial_key: key in adata.obsm holding X/Y coordinates — set to match your AnnData
#   (e.g. check with list(adata.obsm.keys()))
# celltype_key: key in adata.obs holding cell type labels — set to match your AnnData
#   (e.g. check with adata.obs.columns.tolist())
emb_df = run_full_embedding(
    ae_inputs_by_sample=ae_inputs_by_sample,
    ae_model_path=ae_ckpt,
    gcn_model_path=gcn_ckpt,
    device="cuda:0",  # or "cpu"
    combine_mode="average",
    spatial_key='spatial_px',
    celltype_key='celltypes',
    save_ae_dir=None,  # optional
)
```

This produces a DataFrame containing the final integrated embedding for all cells/nuclei.

---

## Required Inputs

SpatialFusion operates on a **single-cell AnnData object** paired with an **H&E whole-slide image**. It also accepts only a WSI with cell coordinates in the H&E only mode, or only an AnnData object with RNA embeddings (no WSI) in the RNA only mode.

### **AnnData fields**

| Key                                | Description                                                       |
| ---------------------------------- | ----------------------------------------------------------------- |
| `adata.obsm['spatial_px']`         | X/Y centroid coordinates of each cell/nucleus in WSI pixel space. This is the default key expected by SpatialFusion; pass `spatial_key=` to `run_full_embedding` if your AnnData uses a different key (check with `list(adata.obsm.keys())`). |
| `adata.X`                          | Raw counts (cell × gene). Must be single-cell resolution.         |
| `adata.obs['celltypes']` (optional) | Annotated cell types. This is the default key; pass `celltype_key=` to `run_full_embedding` if your AnnData uses a different column name (check with `adata.obs.columns.tolist()`). |

### **Whole-Slide Image (WSI)**

A high-resolution H&E image corresponding to the same tissue section used for ST.
Used to compute morphology embeddings with your chosen H&E foundation model (**UNI** or **Virchow**).

---

## Typical Workflow

1. **Prepare ST AnnData and the matched H&E WSI**
2. **Run your RNA foundation model** (scGPT or Nicheformer) to compute molecular embeddings
3. **Run your H&E foundation model** (UNI or Virchow) to compute morphology embeddings
4. **Select the matching pretrained checkpoint pair** (see [Pretrained Weights](#pretrained-weights) above)
5. **Run SpatialFusion** to integrate all modalities into joint embeddings
6. **Cluster & visualize**

   * Leiden clustering
   * UMAP
   * Spatial niche maps

---

## Repository Structure

```
.
├── LICENSE
├── pyproject.toml
├── README.md
├── src
│   └── spatialfusion
│       ├── data
│       │   ├── checkpoint_dir_ae
│       │   │   ├── spatialfusion-ae-uni-scgpt.pt
│       │   │   ├── spatialfusion-ae-uni-nicheformer.pt
│       │   │   ├── spatialfusion-ae-virchow-scgpt.pt
│       │   │   └── spatialfusion-ae-virchow-nicheformer.pt
│       │   └── checkpoint_dir_gcn
│       │       ├── spatialfusion-gcn-uni-scgpt.pt
│       │       ├── spatialfusion-gcn-uni-nicheformer.pt
│       │       ├── spatialfusion-gcn-virchow-scgpt.pt
│       │       ├── spatialfusion-gcn-virchow-nicheformer.pt
│       │       ├── spatialfusion-he-gcn-uni-scgpt.pt
│       │       └── spatialfusion-rna-gcn-uni-scgpt.pt
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

* **src/spatialfusion/data/** — packaged pretrained AE and GCN checkpoints
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

> Yates J, Shavakhi M, Choueiri T, Camp SY, Van Allen EM, Uhler C. SpatialFusion: A lightweight multimodal framework for pathway-informed spatial niche mapping. _bioRxiv_. 2026. doi:10.64898/2026.03.16.712056

---

## Version

### Version

This release adds RNA-only SpatialFusion support (**v0.3.0**).

---

## License

MIT License. See `LICENSE` for details.
