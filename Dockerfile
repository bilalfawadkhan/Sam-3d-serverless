# CUDA base for Torch cu121 wheels
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# 1. System dependencies - Added cmake, libglm-dev, and libgl1 for gsplat/p3d
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates build-essential \
    cmake libglm-dev libgl1-mesa-dev \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge (conda + mamba)
ENV MAMBA_ROOT_PREFIX=/opt/conda
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p ${MAMBA_ROOT_PREFIX} \
    && rm -f /tmp/miniforge.sh
ENV PATH=${MAMBA_ROOT_PREFIX}/bin:$PATH

# --- RTX 4090 & BUILD OPTIMIZATIONS ---
# Fixes IndexError by targeting RTX 4090 architecture (8.9)
# ENV TORCH_CUDA_ARCH_LIST="8.9"
# ENV TCNN_CUDA_ARCHITECTURES=89
# Limits parallel CPU jobs to keep memory usage low on WSL (5GB limit)
ENV MAX_JOBS=1 
# --------------------------------------

# Clone SAM3D Objects repo
RUN git clone https://github.com/facebookresearch/sam-3d-objects.git /workspace/sam-3d-objects

# Create conda env from repo file
WORKDIR /workspace/sam-3d-objects
RUN mamba env create -f environments/default.yml

# Make env available
SHELL ["bash", "-lc"]
ENV CONDA_DEFAULT_ENV=sam3d-objects
ENV PATH=/opt/conda/envs/sam3d-objects/bin:$PATH

# PIP indices used by the project for torch/cu121 + kaolin links
ENV PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
ENV PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"

# Install SAM3D and inference deps
# Note: Installing gsplat via pre-compiled wheel to prevent "Failed to build gsplat"
RUN pip install --no-cache-dir -e '.[dev]' \
    && pip install --no-cache-dir -e '.[p3d]' \
    && pip install --no-cache-dir gsplat \
    && pip install --no-cache-dir -e '.[inference]' \
    && ./patching/hydra \
    && pip install --no-cache-dir "huggingface-hub[cli]<1.0" \
    && pip install --no-cache-dir runpod pillow numpy

# Copy your serverless files
WORKDIR /workspace
COPY handler.py /workspace/handler.py
COPY start.sh /workspace/start.sh
RUN chmod +x /workspace/start.sh

# Defaults (override in RunPod env vars)
ENV SAM3D_TAG=hf
ENV SAM3D_CONFIG=checkpoints/hf/pipeline.yaml
ENV HF_REPO=facebook/sam-3d-objects


CMD ["/workspace/start.sh"]

