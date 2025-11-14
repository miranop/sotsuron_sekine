FROM ubuntu:22.04
WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

# 必要なシステムパッケージをインストール
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    gcc g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libjpeg-dev libpng-dev \
    libpcap-dev \
    tcpdump \
    && rm -rf /var/lib/apt/lists/*

# Pythonのシンボリックリンク作成
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Pythonパッケージをインストール
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    pandas numpy scikit-learn pyts pillow \
    matplotlib \
    scapy \
    anomalib

# コマンド（最後に独立して配置）
CMD ["/bin/bash"]