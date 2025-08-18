# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev \
    git curl build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create venv inside container
RUN python3.11 -m venv $VIRTUAL_ENV && \
    $VIRTUAL_ENV/bin/pip install --upgrade pip setuptools wheel

WORKDIR /workspace
COPY pyproject.toml README.md /workspace/
COPY src /workspace/src

# Install project (no dev deps by default)
RUN pip install -e .

# Optional: install CUDA-enabled torch matching image (user can override)
# RUN pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

CMD ["bash"]
