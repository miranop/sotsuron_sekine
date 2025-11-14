#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CIC-IDS2017 の pcap から Recurrence Plot 画像を生成（改良版）
 - 小窓化 (256秒)
 - STRIDE=64秒
 - log1pスケーリングで安定化
 - 既存画像スキップ対応
"""

import os
import sys
import math
import numpy as np
from datetime import datetime
from collections import defaultdict
from pyts.image import RecurrencePlot
from tqdm import tqdm
from PIL import Image

import same         # same.read_packet_dpkt, same.group_packets_dpkt
import label        # label.get_label, ATTACK_SCHEDULES
from label import ATTACK_SCHEDULES

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ===== 設定 =====
PCAP_DIR = "../Pcap"
OUT_ROOT = "../datasets/rp"
TRAIN_GOOD = os.path.join(OUT_ROOT, "train", "good")
TEST_GOOD = os.path.join(OUT_ROOT, "test", "good")
TEST_ANOM_ROOT = os.path.join(OUT_ROOT, "test", "anomaly")

IMAGE_DIM = 64        # 出力画像サイズ
WINDOW_SIZE = 256     # 1枚の画像に使う「秒」の長さ（← 小窓化）
STRIDE = 64           # スライディングウィンドウのステップ（秒）
TZ_OFFSET_HOURS = -3  # CIC-IDS2017の現地時間補正（Atlantic）

# ===== ユーティリティ =====


def _to_epoch_seconds(ts_obj):
    if isinstance(ts_obj, (int, float)):
        return float(ts_obj)
    if isinstance(ts_obj, datetime):
        return ts_obj.timestamp()
    raise TypeError(f"Unsupported timestamp type: {type(ts_obj)}")


def _label_from_timestamp(ts_obj, pcap_filename):
    ts = _to_epoch_seconds(ts_obj)
    return label.get_label(ts, pcap_filename)


def _sanitize(label_str):
    return label_str.replace(" ", "_").replace("-", "_").replace("/", "_")

# ===== 主要処理 =====


def extract_series_per_second(packets, pcap_filename):
    """
    1秒ごとの合計バイト数の系列とその代表ラベルを返す
    """
    if not packets:
        return np.array([]), []

    traffic_per_sec = defaultdict(int)
    label_per_sec = {}

    for pkt in packets:
        try:
            ts = pkt["timestamp"]
            size = len(pkt["bytes"])
            sec = math.floor(_to_epoch_seconds(ts))
            traffic_per_sec[sec] += size
            label_per_sec[sec] = _label_from_timestamp(ts, pcap_filename)
        except Exception:
            continue

    if not traffic_per_sec:
        return np.array([]), []

    secs_sorted = sorted(traffic_per_sec.keys())
    series = np.array([traffic_per_sec[s]
                      for s in secs_sorted], dtype=np.float32)
    labels = [label_per_sec[s] for s in secs_sorted]

    # --- logスケーリングで値域を圧縮 ---
    series = np.log1p(series)
    # --- 正規化（0〜1） ---
    min_val, max_val = np.min(series), np.max(series)
    if max_val - min_val > 1e-8:
        series = (series - min_val) / (max_val - min_val)
    else:
        series = np.zeros_like(series)
    return series, labels


def save_rp_image(data_1d, label_value, pcap_name, idx, is_train):
    """Recurrence Plot画像を保存（既存はスキップ）"""
    if label_value == "BENIGN":
        out_dir = TRAIN_GOOD if is_train else TEST_GOOD
        filename = f"{pcap_name}_{'train' if is_train else 'test'}_{idx:06d}.png"
    else:
        sanitized = _sanitize(label_value)
        out_dir = os.path.join(TEST_ANOM_ROOT, sanitized)
        filename = f"{pcap_name}_{sanitized}_{idx:06d}.png"

    os.makedirs(out_dir, exist_ok=True)
    filepath = os.path.join(out_dir, filename)
    if os.path.exists(filepath):
        return False

    rp = RecurrencePlot(threshold=None, dimension=1)
    rp_img = rp.fit_transform(data_1d.reshape(1, -1))[0]
    rp_img = (rp_img * 255).astype(np.uint8)

    im = Image.fromarray(rp_img)
    if im.size != (IMAGE_DIM, IMAGE_DIM):
        im = im.resize((IMAGE_DIM, IMAGE_DIM), resample=Image.NEAREST)
    im.save(filepath)
    return True


def main():
    print("=" * 80)
    print("🔹 pcap → Recurrence Plot 画像生成（log正規化 + 小窓化）")
    print("=" * 80)

    pcap_files = [
        {"path": os.path.join(PCAP_DIR, "Monday-WorkingHours.pcap"),
         "is_train": True,  "time_filter": None},
        {"path": os.path.join(PCAP_DIR, "Tuesday-WorkingHours.pcap"),
         "is_train": False, "time_filter": None},
        {"path": os.path.join(PCAP_DIR, "Wednesday-workingHours.pcap"),
         "is_train": False, "time_filter": None},
        {"path": os.path.join(PCAP_DIR, "Thursday-WorkingHours.pcap"),
         "is_train": False, "time_filter": None},
        {"path": os.path.join(PCAP_DIR, "Friday-WorkingHours.pcap"),
         "is_train": False, "time_filter": None},
    ]

    total_counts = {"train/good": 0, "test/good": 0}
    attack_counts = defaultdict(int)

    for info in pcap_files:
        path = info["path"]
        is_train = info["is_train"]
        time_filter = info["time_filter"]
        if not os.path.exists(path):
            print(f"⚠ スキップ（見つからない）: {path}")
            continue

        pcap_name = os.path.splitext(os.path.basename(path))[0]
        print(f"\n▶ {pcap_name} 読み込み中…")

        packets = same.read_packet_dpkt(
            path, time_filter=time_filter, tz_offset=TZ_OFFSET_HOURS)
        if not packets:
            print("  パケットなし。スキップ。")
            continue

        traffic = same.group_packets_dpkt(packets)
        print(f"  IPペア数: {len(traffic)}")

        produced = 0
        for ip_pair, pkt_list in tqdm(traffic.items(), desc=f"  {pcap_name} 処理中", leave=False):
            series, labels = extract_series_per_second(pkt_list, path)
            if len(series) < WINDOW_SIZE:
                continue

            i = 0
            while i + WINDOW_SIZE <= len(series):
                window = series[i:i + WINDOW_SIZE]
                label_value = labels[min(i + WINDOW_SIZE - 1, len(labels) - 1)]
                idx = (i // STRIDE)

                if save_rp_image(window, label_value, pcap_name, idx, is_train):
                    if label_value == "BENIGN":
                        key = "train/good" if is_train else "test/good"
                        total_counts[key] += 1
                    else:
                        attack_counts[label_value] += 1
                    produced += 1

                i += STRIDE

        print(f"  生成枚数: {produced:,}")

    print("\n=== ✅ 生成完了 ===")
    print(f"train/good: {total_counts['train/good']:,}")
    print(f"test/good : {total_counts['test/good']:,}")
    for k in sorted(attack_counts.keys()):
        print(f"{k:30s}: {attack_counts[k]:,}")
    print(f"\n出力先: {OUT_ROOT}")
    print("=" * 80)


if __name__ == "__main__":
    main()
