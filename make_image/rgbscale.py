#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RGB画像生成スクリプト - ヘッダ限定版 (Header Only)
特徴: 暗号化されたペイロードを捨て、ヘッダ構造のみを画像化
      RGBチャネルを活用して構造を強調
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
import dpkt

warnings.filterwarnings('ignore')

# ===== パス設定 =====
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../make_image")))

try:
    import same
    import label
except ImportError:
    print("❌ エラー: 'same.py' または 'label.py' が見つかりません。")
    sys.exit(1)

# ===== 設定 =====
IMAGE_DIM = 32
PCAP_DIR = "../Pcap"
OUT_ROOT = "../datasets/rgb_header"  # 出力先を変更
TRAIN_GOOD = os.path.join(OUT_ROOT, "train", "good")
TEST_GOOD = os.path.join(OUT_ROOT, "test", "good")
TEST_ANOM_ROOT = os.path.join(OUT_ROOT, "test", "anomaly")

# 1画像あたりのパケット数
PACKETS_PER_IMAGE = IMAGE_DIM * 3

# データ分割設定
TRAIN_RATIO = 0.7
RANDOM_SEED = 42
MAX_IMAGES_TRAIN_GOOD = 5000
MAX_IMAGES_TEST_GOOD = 2000
MAX_IMAGES_PER_ATTACK_TRAIN = 800
MAX_IMAGES_PER_ATTACK_TEST = 400
MIN_IMAGES_PER_ATTACK_TEST = 20

# ★ 修正: 移動比率を0.6に下げてテストデータを確保
MOVE_RATIO_FROM_TEST_TO_TRAIN = 0.6

TZ_OFFSET_HOURS = -3

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ===== ヘッダ抽出 =====
def extract_headers(raw_bytes):
    """ヘッダのみ抽出 (Eth + IP + TCP/UDP)"""
    try:
        eth = dpkt.ethernet.Ethernet(raw_bytes)
        if not isinstance(eth.data, dpkt.ip.IP):
            return raw_bytes[:54]

        ip = eth.data
        ip_len = ip.hl * 4
        trans = ip.data
        trans_len = 0
        if isinstance(trans, dpkt.tcp.TCP):
            trans_len = trans.off * 4
        elif isinstance(trans, dpkt.udp.UDP):
            trans_len = 8
        elif isinstance(trans, dpkt.icmp.ICMP):
            trans_len = 8

        total = 14 + ip_len + trans_len
        return raw_bytes[:total]
    except:
        return raw_bytes[:54]

# ===== ユーティリティ =====


def _to_epoch(ts):
    if isinstance(ts, (int, float)):
        return float(ts)
    return ts.timestamp()


def _get_lbl(ts, name):
    return label.get_label(_to_epoch(ts), name)


def sanitize(s): return s.replace(" ", "_").replace("/", "_")


def split_files(files, ratio):
    mon = [f for f in files if "monday" in f["path"].lower()]
    oth = [f for f in files if "monday" not in f["path"].lower()]
    random.shuffle(oth)
    idx = int(len(oth) * ratio)
    t = mon + oth[:idx]
    e = oth[idx:]
    for f in t:
        f["is_train"] = True
    for f in e:
        f["is_train"] = False
    return t, e

# ===== 画像生成 =====


def packet_list_to_rgb_images(packet_list, pcap_filename, image_dim=32):
    """ヘッダのみを用いてRGB画像を生成"""
    images = []
    packet_list.sort(key=lambda p: p["timestamp"])

    if len(packet_list) == 0:
        return []

    # パケットごとに埋める
    # 1画像 = 32x32ピクセル = 3チャンネル x 32行 = 96パケット分
    packets_per_img = image_dim * 3

    for i in range(0, len(packet_list), packets_per_img):
        chunk = packet_list[i:i + packets_per_img]

        ts = chunk[0]["timestamp"]
        try:
            lbl = _get_lbl(ts, pcap_filename)
        except:
            lbl = "BENIGN"

        # 黒で初期化
        img_data = np.zeros((image_dim, image_dim, 3), dtype=np.uint8)

        for k, pkt in enumerate(chunk):
            ch = k // image_dim
            row = k % image_dim
            if ch >= 3:
                break  # RGB3チャンネル埋まったら終了

            raw = pkt["bytes"]

            # ★ ヘッダ抽出
            header = extract_headers(raw)

            # 黒パディング (0x00)
            padded = header[:image_dim].ljust(image_dim, b"\x00")

            img_data[row, :, ch] = np.frombuffer(padded, dtype=np.uint8)

        images.append((img_data, lbl))

    return images


def save_imgs(imgs, name, is_train, counters):
    saved = defaultdict(int)
    for idx, (data, lbl) in enumerate(imgs):
        if lbl == "BENIGN":
            if is_train:
                k, lim, d = "train_good", MAX_IMAGES_TRAIN_GOOD, TRAIN_GOOD
            else:
                k, lim, d = "test_good", MAX_IMAGES_TEST_GOOD, TEST_GOOD
            fn = f"{name}_{k}_{idx:06d}.png"
        else:
            if is_train:
                continue
            k = f"test_anom_{lbl}"
            lim = MAX_IMAGES_PER_ATTACK_TEST
            sl = sanitize(lbl)
            d = os.path.join(TEST_ANOM_ROOT, sl)
            fn = f"{name}_{sl}_{idx:06d}.png"

        if counters[k] >= lim:
            continue

        os.makedirs(d, exist_ok=True)
        fp = os.path.join(d, fn)
        if not os.path.exists(fp):
            Image.fromarray(data, "RGB").save(fp)
            counters[k] += 1
            saved[k] += 1
    return saved


def redistribute(src, dst, ratio):
    print("\n再配分処理...")
    if not os.path.exists(src):
        return
    os.makedirs(dst, exist_ok=True)
    fs = [f for f in os.listdir(src) if f.endswith('.png')]
    random.shuffle(fs)
    cnt = int(len(fs) * ratio)
    for f in fs[:cnt]:
        try:
            shutil.move(os.path.join(src, f), os.path.join(dst, f))
        except:
            pass
    print(f"{cnt}枚を train へ移動")


def main():
    print("=" * 80)
    print("🔹 pcap → RGB画像変換 (Header Only版)")
    print("=" * 80)

    files = [{"path": os.path.join(PCAP_DIR, f), "is_train": False}
             for f in os.listdir(PCAP_DIR) if f.endswith(".pcap")]

    train_fs, test_fs = split_files(files, TRAIN_RATIO)
    all_fs = train_fs + test_fs
    cnts = defaultdict(int)

    for info in all_fs:
        path = info["path"]
        print(
            f"\n処理中: {os.path.basename(path)} ({'Train' if info['is_train'] else 'Test'})")
        try:
            pkts = same.read_packet_dpkt(path, tz_offset=TZ_OFFSET_HOURS)
        except:
            continue
        if not pkts:
            continue

        traf = same.group_packets_dpkt(pkts)
        for _, p in tqdm(traf.items(), leave=False):
            imgs = packet_list_to_rgb_images(p, path, IMAGE_DIM)
            if imgs:
                save_imgs(imgs, os.path.splitext(os.path.basename(path))[
                          0], info["is_train"], cnts)

    if MOVE_RATIO_FROM_TEST_TO_TRAIN > 0:
        redistribute(TEST_GOOD, TRAIN_GOOD, MOVE_RATIO_FROM_TEST_TO_TRAIN)

    print("\n完了")
    print(f"出力先: {OUT_ROOT}")


if __name__ == "__main__":
    main()
