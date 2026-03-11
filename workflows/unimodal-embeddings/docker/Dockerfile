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
    numpy==2.2.6 \
    pandas==2.3.3 \
    pyarrow==23.0.1 \
    scipy==1.15.3 \
    anndata==0.9.2 \
    h5py==3.16.0 \
    scanpy==1.11.5 \
    tifffile==2025.5.10 \
    imagecodecs==2025.3.30 \
    timm==1.0.25 \
    Pillow==12.0.0 \
    tqdm==4.67.3 \
    colorlog==6.10.1 \
    wandb==0.25.0 \
    scgpt==0.2.4 \
    IPython==8.38.0

# Zero shot scfoundation repo - functions are used in scGPT embedding
RUN git clone https://github.com/microsoft/zero-shot-scfoundation.git /app/zero-shot-scfoundation && \
    cd /app/zero-shot-scfoundation && \
    git checkout v0.1.1

COPY scripts/unimodal-embeddings.py /app/unimodal-embeddings.py