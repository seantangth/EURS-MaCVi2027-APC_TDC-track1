# Same PyTorch build as our leaderboard run (torch 2.11.0+cu128, Python 3.12). The base image's PyTorch 2.4.0 cannot run
# our Mask2Former teacher: the transformers release it was trained and run with (5.20.2) needs torch >= 2.6.
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime
# The image's Python is Ubuntu's system Python 3.12 (PEP 668): allow pip to install into it.
ENV PIP_NO_CACHE_DIR=1 PIP_BREAK_SYSTEM_PACKAGES=1 HOME=/tmp XDG_CACHE_HOME=/tmp/.cache HF_HOME=/tmp/hf_home TORCH_HOME=/tmp/torch_home \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# git: repro_check.py calls it when /pkg has a .git folder and the weights were downloaded
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
# torchvision from the same CUDA 12.8 index as the base image's torch (no-op if the image already has it)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 torchvision==0.26.0
COPY docker-requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN python -c "import torch, torchvision, timm, transformers, skimage, PIL; print(torch.__version__, torchvision.__version__, timm.__version__, transformers.__version__, PIL.__version__)"
COPY . /pkg
WORKDIR /pkg
