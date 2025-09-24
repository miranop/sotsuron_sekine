FROM ubuntu:22.04
WORKDIR /app

# 非対話的インストール設定
ENV DEBIAN_FRONTEND=noninteractive

# システム更新とPythonインストール
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    gcc g++ git \
    # OpenCV完全サポート用ライブラリ
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # 画像処理用
    libjpeg-dev libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Python3をpythonとして使用
RUN ln -s /usr/bin/python3 /usr/bin/python

# Pythonライブラリを段階的にインストール
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir pytorch-lightning lightning torchmetrics
RUN pip install --no-cache-dir numpy pillow matplotlib opencv-python pandas seaborn scikit-learn
RUN pip install --no-cache-dir kornia scikit-image pyshark
RUN pip install --no-cache-dir "anomalib[full]"

COPY . .
CMD ["/bin/bash"]