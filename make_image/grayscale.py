#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
グレースケール画像生成スクリプト - ヘッダ限定版 (Header Only)
特徴: 暗号化されたペイロードを捨て、ヘッダ構造のみを画像化して精度向上を狙う
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
import dpkt  # pip install dpkt

# 警告抑制
warnings.filterwarnings('ignore')

# ===== パス設定 =====
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
IMAGE_DIM = 32
PCAP_DIR = "../Pcap"
OUT_ROOT = "../datasets/grayscale_header"  # 出力先を変更
TRAIN_GOOD = os.path.join(OUT_ROOT, "train", "good")
TEST_GOOD = os.path.join(OUT_ROOT, "test", "good")
TEST_ANOM_ROOT = os.path.join(OUT_ROOT, "test", "anomaly")

# データ分割設定
TRAIN_RATIO = 0.7
RANDOM_SEED = 42
MAX_IMAGES_TRAIN_GOOD = 5000
MAX_IMAGES_TEST_GOOD = 2000
MAX_IMAGES_PER_ATTACK_TRAIN = 800
MAX_IMAGES_PER_ATTACK_TEST = 400
MIN_IMAGES_PER_ATTACK_TEST = 20
MOVE_RATIO_FROM_TEST_TO_TRAIN = 0.6
TZ_OFFSET_HOURS = -3

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ===== ヘッダ抽出ロジック =====
def extract_headers(raw_bytes):
    """
    パケットの生データからヘッダ部分(Ethernet + IP + TCP/UDP)のみを抽出・結合する。
    ペイロード(データ中身)は捨てる。
    """
    try:
        # Ethernetフレームとして解析
        eth = dpkt.ethernet.Ethernet(raw_bytes)

        # IPパケットでない場合は、とりあえず先頭54バイト(一般的なヘッダ長)を返す
        if not isinstance(eth.data, dpkt.ip.IP):
            return raw_bytes[:54]

        ip = eth.data
        ip_header_len = ip.hl * 4

        trans = ip.data
        trans_header_len = 0

        if isinstance(trans, dpkt.tcp.TCP):
            trans_header_len = trans.off * 4
        elif isinstance(trans, dpkt.udp.UDP):
            trans_header_len = 8
        elif isinstance(trans, dpkt.icmp.ICMP):
            trans_header_len = 8

        # 抽出したい全長 = Eth(14) + IPヘッダ長 + トランスポートヘッダ長
        total_header_len = 14 + ip_header_len + trans_header_len

        return raw_bytes[:total_header_len]

    except Exception:
        # 解析エラー時は安全策として先頭54バイトを返す
        return raw_bytes[:54]


# ===== ラベル関連 =====
def _to_epoch_seconds(ts_obj):
    if isinstance(ts_obj, (int, float)):
        return float(ts_obj)
    if isinstance(ts_obj, datetime):
        return ts_obj.timestamp()
    raise TypeError(f"Unsupported timestamp type: {type(ts_obj)}")


def _label_from_timestamp(ts_obj, pcap_filename):
    ts = _to_epoch_seconds(ts_obj)
    return label.get_label(ts, pcap_filename)


def sanitize_label(label_str):
    return label_str.replace(" ", "_").replace("/", "_")


def count_images_in_directory(directory):
    if not os.path.exists(directory):
        return 0
    return len([f for f in os.listdir(directory) if f.lower().endswith('.png')])


def split_pcap_files(pcap_files, train_ratio=0.7):
    monday = [f for f in pcap_files if "monday" in f["path"].lower()]
    others = [f for f in pcap_files if "monday" not in f["path"].lower()]
    random.shuffle(others)
    split = int(len(others) * train_ratio)
    t = monday + others[:split]
    e = others[split:]
    for f in t:
        f["is_train"] = True
    for f in e:
        f["is_train"] = False
    return t, e


# ===== 画像生成ロジック =====
def packet_list_to_gray_images(packet_list, pcap_filename, image_dim=64):
    """ヘッダのみを用いてグレースケール画像を生成"""
    images_with_labels = []
    packet_list.sort(key=lambda p: p["timestamp"])

    if len(packet_list) == 0:
        return []

    for i in range(0, len(packet_list), image_dim):
        chunk = packet_list[i:i + image_dim]

        first_packet = chunk[0]
        timestamp = first_packet["timestamp"]

        try:
            label_value = _label_from_timestamp(timestamp, pcap_filename)
        except Exception:
            label_value = "BENIGN"

        # 画像データ生成 (初期値黒)
        image_data = np.zeros((image_dim, image_dim), dtype=np.uint8)

        for j, pkt in enumerate(chunk):
            raw_bytes = pkt["bytes"]

            # ★ ここでヘッダのみ抽出
            header_bytes = extract_headers(raw_bytes)

            # 長さが足りない部分は黒(0x00)で埋める
            padded = header_bytes[:image_dim].ljust(image_dim, b"\x00")

            image_data[j] = np.frombuffer(padded, dtype=np.uint8)

        images_with_labels.append((image_data, label_value))

    return images_with_labels


def save_images(images_with_labels, pcap_name, is_train, label_counters):
    saved_counts = defaultdict(int)

    for idx, (data, lbl) in enumerate(images_with_labels):
        if lbl == "BENIGN":
            if is_train:
                key, max_c, d = "train_good", MAX_IMAGES_TRAIN_GOOD, TRAIN_GOOD
            else:
                key, max_c, d = "test_good", MAX_IMAGES_TEST_GOOD, TEST_GOOD
            fname = f"{pcap_name}_{key}_{idx:06d}.png"
        else:
            if is_train:
                continue
            key = f"test_anom_{lbl}"
            max_c = MAX_IMAGES_PER_ATTACK_TEST
            s_lbl = sanitize_label(lbl)
            d = os.path.join(TEST_ANOM_ROOT, s_lbl)
            fname = f"{pcap_name}_{s_lbl}_{idx:06d}.png"

        if label_counters[key] >= max_c:
            continue

        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, fname)

        if not os.path.exists(path):
            Image.fromarray(data, "L").save(path)
            label_counters[key] += 1
            saved_counts[key] += 1

    return saved_counts


def main():
    print("=" * 80)
    print("🔹 pcap → グレースケール画像変換 (Header Only版)")
    print("=" * 80)

    pcap_files = [
        {"path": os.path.join(PCAP_DIR, f), "is_train": False}
        for f in os.listdir(PCAP_DIR) if f.endswith(".pcap")
    ]

    train_files, test_files = split_pcap_files(pcap_files, TRAIN_RATIO)
    all_files = train_files + test_files
    label_counters = defaultdict(int)

    print("Phase 1: 画像生成")
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
        traffic = same.group_packets_dpkt(packets)

        for _, pkts in tqdm(traffic.items(), leave=False):
            imgs = packet_list_to_gray_images(pkts, path, IMAGE_DIM)
            if imgs:
                save_images(imgs, os.path.splitext(os.path.basename(path))[0],
                            info["is_train"], label_counters)

    if MOVE_RATIO_FROM_TEST_TO_TRAIN > 0:
        redistribute_test_to_train(
            TEST_GOOD, TRAIN_GOOD, MOVE_RATIO_FROM_TEST_TO_TRAIN)

    print("\n完了")
    print(f"出力先: {OUT_ROOT}")


def redistribute_test_to_train(test_dir, train_dir, ratio):
    print("\n再配分処理...")
    if not os.path.exists(test_dir):
        return
    os.makedirs(train_dir, exist_ok=True)
    files = [f for f in os.listdir(test_dir) if f.endswith('.png')]
    random.shuffle(files)
    count = int(len(files) * ratio)
    for f in files[:count]:
        try:
            shutil.move(os.path.join(test_dir, f), os.path.join(train_dir, f))
        except:
            pass
    print(f"{count}枚を train へ移動")


if __name__ == "__main__":
    main()
