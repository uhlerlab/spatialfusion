FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN python -m pip install --upgrade pip

# PyTorch CUDA 12.1 stack
RUN pip install --no-cache-dir \
    torch==2.3.0 \
    torchvision==0.18.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Runtime dependencies used by workflows/unimodal-embeddings/scripts/unimodal-embeddings.py
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    pyarrow \
    scipy \
    anndata==0.9.2 \
    h5py \
    scanpy \
    tifffile \
    imagecodecs \
    timm \
    Pillow \
    tqdm \
    colorlog \
    wandb \
    scgpt \
    IPython

RUN git clone https://github.com/microsoft/zero-shot-scfoundation.git /app/zero-shot-scfoundation && \
    cd /app/zero-shot-scfoundation && \
    git checkout v0.1.1

COPY scripts/unimodal-embeddings.py /app/unimodal-embeddings.py