FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# 基本
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git wget unzip nano vim \
    libgl1-mesa-glx libglib2.0-0 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN pip install --upgrade pip

# PyTorch (CUDA12)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Anomalib & CV
RUN pip install \
    anomalib \
    opencv-python \
    scikit-learn \
    scikit-image \
    matplotlib \
    seaborn \
    pillow \
    tqdm \
    pandas \
    numpy \
    einops \
    wandb \
    tensorboard

# あなた用の追加ツール（GAF / RP）
RUN pip install \
    pyts

# コードを内包する場所
WORKDIR /workspace

# test_code を丸ごとコピー
COPY ./test_code /workspace/test_code

CMD ["/bin/bash"]
