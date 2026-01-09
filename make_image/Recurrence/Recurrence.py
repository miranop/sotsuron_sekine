#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
============================================================
 Recurrence Plot 画像生成ミニフレームワーク（統合版）
============================================================
前処理 → 小窓化 → RP画像生成 → train/test 保存
このファイルだけでRPデータセットを作成可能。
============================================================
"""

import os
import random
import numpy as np
from pyts.image import RecurrencePlot
from PIL import Image


# ============================================================
# 1. 前処理（log1p → min-max正規化）
# ============================================================
def preprocess_series(series):
    arr = np.array(series, dtype=float)

    # log1p（値の圧縮）
    arr = np.log1p(arr)

    # min-max（0〜1）
    min_val = arr.min()
    max_val = arr.max()

    if max_val - min_val > 1e-8:
        arr = (arr - min_val) / (max_val - min_val)
    else:
        arr = np.zeros_like(arr)

    return arr.tolist()


# ============================================================
# 2. スライディングウィンドウ
# ============================================================
def sliding_windows(series, window_size, stride):
    return [
        series[i:i + window_size]
        for i in range(0, len(series) - window_size + 1, stride)
    ]


# ============================================================
# 3. Recurrence Plot 生成
# ============================================================
def generate_rp(window, image_dim=256):
    arr = np.array(window).reshape(1, -1)

    rp = RecurrencePlot(threshold=None, dimension=1)
    rp_img = rp.fit_transform(arr)[0]     # shape=(N,N)

    # 画像化
    rp_img = (rp_img * 255).astype(np.uint8)
    img = Image.fromarray(rp_img)

    # サイズ変更
    if img.size != (image_dim, image_dim):
        img = img.resize((image_dim, image_dim), Image.NEAREST)

    return img


# ============================================================
# 4. 出力フォルダ設定
# ============================================================
ROOT = "./datasets/rp"
TRAIN_GOOD = os.path.join(ROOT, "train", "good")
TEST_GOOD = os.path.join(ROOT, "test", "good")
TEST_ANOM = os.path.join(ROOT, "test", "anomaly")

os.makedirs(TRAIN_GOOD, exist_ok=True)
os.makedirs(TEST_GOOD, exist_ok=True)
os.makedirs(TEST_ANOM, exist_ok=True)


# ============================================================
# 5. RP画像を train/test/good/anomaly に保存
# ============================================================
def save_rp_image(img, label_value, is_train, pcap_name, idx):
    """
    img: PIL Image
    label_value: BENIGN または 攻撃名
    is_train: True → train, False → test
    pcap_name: ファイル名用
    idx: 連番
    """

    if is_train:  # ----- 学習用 -----
        out_dir = TRAIN_GOOD
        filename = f"{pcap_name}_train_{idx:06d}.png"

    else:  # ----- テスト用 -----
        if label_value == "BENIGN":
            out_dir = TEST_GOOD
            filename = f"{pcap_name}_test_{idx:06d}.png"
        else:
            sanitized = label_value.replace(" ", "_").replace("/", "_")
            out_dir = os.path.join(TEST_ANOM, sanitized)
            os.makedirs(out_dir, exist_ok=True)
            filename = f"{pcap_name}_{sanitized}_{idx:06d}.png"

    filepath = os.path.join(out_dir, filename)
    img.save(filepath)
    return filepath


# ============================================================
# 6. train/test 分割（シンプル版）
# ============================================================
def split_pcap_files(pcap_files, train_ratio=0.7, seed=42):
    random.seed(seed)

    # Monday → 必ず学習用
    monday = [p for p in pcap_files if "monday" in p.lower()]
    other = [p for p in pcap_files if "monday" not in p.lower()]

    random.shuffle(other)
    s = int(len(other) * train_ratio)

    train = monday + other[:s]
    test = other[s:]

    return train, test
