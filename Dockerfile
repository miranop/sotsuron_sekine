FROM python:3.11-slim

WORKDIR /app

# システムライブラリ
RUN apt-get update && apt-get install -y \
    gcc g++ git curl wget \
    libglib2.0-0 libgl1-mesa-glx \
    # 追加: 画像処理用ライブラリ
    libjpeg-dev libpng-dev libfreetype6-dev \
    && rm -rf /var/lib/apt/lists/*

# Pythonライブラリを段階的にインストール
RUN pip install --no-cache-dir --upgrade pip

# PyTorch (CPU版)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Lightning関連（両方必要）
RUN pip install --no-cache-dir pytorch-lightning lightning torchmetrics

# 基本ライブラリ
RUN pip install --no-cache-dir numpy pillow matplotlib opencv-python-headless pandas seaborn scikit-learn

# 追加の必要ライブラリ（今回の経験から）
RUN pip install --no-cache-dir kornia scikit-learn scikit-image

# anomalib（最後に）
RUN pip install --no-cache-dir "anomalib[full]"

CMD ["python"]