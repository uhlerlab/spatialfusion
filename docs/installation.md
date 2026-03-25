# Installation


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