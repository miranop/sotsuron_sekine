#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCAP(バイト列) → Recurrence Plot 画像生成スクリプト
特徴: PAA平滑化により、バイト列の「繰り返しパターン」を可視化
"""

import os
import sys
import random
import shutil
from collections import defaultdict
from datetime import datetime
import warnings

import numpy as np
from PIL import Image
from tqdm import tqdm
from pyts.image import RecurrencePlot

# 警告抑制
warnings.filterwarnings('ignore')

# ===== パス設定 (ModuleNotFoundError回避) =====
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../make_image")))

try:
    import same
    import label
except ImportError:
    print("❌ エラー: 'same.py' または 'label.py' が見つかりません。")
    print("   export PYTHONPATH=$PYTHONPATH:../make_image を実行してください。")
    sys.exit(1)

# ===== 設定 =====
PCAP_DIR = "../Pcap"
OUT_ROOT = "../datasets/pcap_rp"  # 出力先
TRAIN_GOOD = os.path.join(OUT_ROOT, "train", "good")
TEST_GOOD = os.path.join(OUT_ROOT, "test", "good")
TEST_ANOM_ROOT = os.path.join(OUT_ROOT, "test", "anomaly")

# ★ RP生成のコア設定 ★
# 1画像を作るために読み込むバイト数
# 例: 512バイト読んで、PAAで64点に圧縮し、64x64のRP行列を作る
RAW_BYTES_LEN = 512
RP_MATRIX_SIZE = 64  # RPの計算サイズ (小さいほうが高速)
PAA_WINDOW = RAW_BYTES_LEN // RP_MATRIX_SIZE  # 自動計算 (8バイト平均)

# 保存する画像の解像度 (CNNに入力するサイズ)
IMAGE_DIM = 256

# データ分割設定
TRAIN_RATIO = 0.7
RANDOM_SEED = 42
MAX_IMAGES_TRAIN_GOOD = 2000
MAX_IMAGES_TEST_GOOD = 1000
MAX_IMAGES_PER_ATTACK_TEST = 400
MIN_IMAGES_PER_ATTACK_TEST = 20
MOVE_RATIO_FROM_TEST_TO_TRAIN = 0.6
TZ_OFFSET_HOURS = -3

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ===== 関数定義 =====

def apply_paa_1d(array, target_length, window_size):
    """修正版: 0パディングではなく、データを繰り返して埋める"""
    length = target_length * window_size

    if len(array) < length:
        # 足りない分を、自分自身の繰り返しで埋める
        # 例: [1,2] -> [1,2,1,2,1,2...]
        repeats = (length // len(array)) + 1
        array = np.tile(array, repeats)[:length]
    else:
        array = array[:length]

    return array.reshape(target_length, window_size).mean(axis=1)


def generate_rp_from_bytes(data_bytes, out_dim=256):
    """バイト列からRP画像を生成"""
    try:
        # バイト列を数値(0-255)の配列に変換
        arr = np.frombuffer(data_bytes, dtype=np.uint8)

        # 何もデータがない場合
        if len(arr) == 0:
            return Image.new('L', (out_dim, out_dim), 0)

        # 1. PAAで縮小 (例: 512 -> 64)
        # 数値の変動トレンドを見るため、floatにしておく
        smoothed = apply_paa_1d(arr.astype(float), RP_MATRIX_SIZE, PAA_WINDOW)

        # 2. RP変換
        # threshold=None: 距離そのものを輝度にする（グレースケールRP）
        rp = RecurrencePlot(threshold=None, dimension=1)
        rp_matrix = rp.fit_transform(smoothed.reshape(1, -1))[0]

        # 3. 画像化 (0-255)
        rp_img = (rp_matrix * 255).astype(np.uint8)
        img = Image.fromarray(rp_img)

        # 4. リサイズ (例: 64x64 -> 256x256)
        if img.size != (out_dim, out_dim):
            img = img.resize((out_dim, out_dim), Image.NEAREST)

        return img

    except Exception:
        return Image.new('L', (out_dim, out_dim), 0)


def packet_list_to_rp_images(packet_list, pcap_filename):
    """パケットリストからRP画像を生成"""
    images_with_labels = []
    packet_list.sort(key=lambda p: p["timestamp"])

    # パケットごとに画像を生成するか、複数をまとめるか
    # ここでは「1パケットのペイロード」から「1枚のRP」を作るロジックにします
    # (攻撃の特徴は1パケットの中身に出やすいため)

    for pkt in packet_list:
        payload = pkt["bytes"]
        timestamp = pkt["timestamp"]

        # ペイロードが短すぎるパケットはスキップ（ARPやACKのみなど）
        if len(payload) < 32:
            continue

        try:
            label_value = label.get_label(timestamp, pcap_filename)
        except Exception:
            label_value = "BENIGN"

        # 画像生成
        # 先頭から RAW_BYTES_LEN だけ切り出して使う
        target_bytes = payload[:RAW_BYTES_LEN]
        img = generate_rp_from_bytes(target_bytes, IMAGE_DIM)

        # 画像データをnumpy配列として保持しないと処理が重くなるため
        # ここではPIL Imageオブジェクトのままリストに入れる
        images_with_labels.append((img, label_value))

    return images_with_labels


def save_images(images_with_labels, pcap_name, is_train, label_counters):
    saved_counts = defaultdict(int)

    for idx, (img, label_value) in enumerate(images_with_labels):
        if label_value == "BENIGN":
            if is_train:
                key = "train_good"
                max_c = MAX_IMAGES_TRAIN_GOOD
                d = TRAIN_GOOD
            else:
                key = "test_good"
                max_c = MAX_IMAGES_TEST_GOOD
                d = TEST_GOOD
            fname = f"{pcap_name}_{key}_{idx:06d}.png"
        else:
            if is_train:
                continue
            key = f"test_anom_{label_value}"
            max_c = MAX_IMAGES_PER_ATTACK_TEST
            sanitized = label_value.replace(" ", "_").replace("/", "_")
            d = os.path.join(TEST_ANOM_ROOT, sanitized)
            fname = f"{pcap_name}_{sanitized}_{idx:06d}.png"

        if label_counters[key] >= max_c:
            continue

        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, fname)

        if not os.path.exists(path):
            img.save(path)
            saved_counts[key] += 1
            label_counters[key] += 1

    return saved_counts

# ===== メイン処理 =====


def main():
    print("="*80)
    print(f"PCAP → RP画像変換 (Byte-level Recurrence Plot)")
    print(
        f"  Input: {RAW_BYTES_LEN} bytes -> PAA -> {RP_MATRIX_SIZE}x{RP_MATRIX_SIZE} Matrix")
    print("="*80)

    pcap_files = [
        {"path": os.path.join(PCAP_DIR, f), "is_train": False}
        for f in os.listdir(PCAP_DIR) if f.endswith(".pcap")
    ]

    # Train/Test分割
    monday = [f for f in pcap_files if "monday" in f["path"].lower()]
    others = [f for f in pcap_files if "monday" not in f["path"].lower()]
    random.shuffle(others)
    split = int(len(others) * TRAIN_RATIO)
    train_files = monday + others[:split]
    test_files = others[split:]

    for f in train_files:
        f["is_train"] = True

    all_files = train_files + test_files
    counters = defaultdict(int)

    for info in all_files:
        path = info["path"]
        if not os.path.exists(path):
            print(f"Skip: {path}")
            continue

        print(
            f"\n処理中: {os.path.basename(path)} ({'Train' if info['is_train'] else 'Test'})")

        try:
            packets = same.read_packet_dpkt(path, tz_offset=TZ_OFFSET_HOURS)
        except:
            continue

        if not packets:
            continue

        # グルーピング
        traffic = same.group_packets_dpkt(packets)

        # 生成
        for _, pkts in tqdm(traffic.items(), leave=False):
            imgs = packet_list_to_rp_images(pkts, path)
            if imgs:
                save_images(imgs, os.path.splitext(os.path.basename(path))[
                            0], info["is_train"], counters)

    # 再配分
    if MOVE_RATIO_FROM_TEST_TO_TRAIN > 0:
        print("\n再配分処理...")
        test_imgs = [f for f in os.listdir(TEST_GOOD) if f.endswith('.png')]
        random.shuffle(test_imgs)
        for f in test_imgs[:int(len(test_imgs)*MOVE_RATIO_FROM_TEST_TO_TRAIN)]:
            shutil.move(os.path.join(TEST_GOOD, f),
                        os.path.join(TRAIN_GOOD, f))

    print("\n完了")
    print(f"出力: {OUT_ROOT}")


if __name__ == "__main__":
    main()
