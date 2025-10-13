#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Recurrence Plot（RP）生成スクリプト
---------------------------------
- pcapファイルからパケット長を抽出
- 秒単位の通信量（バイト数）に変換
- Recurrence Plot 画像を作成
- ファイル名から BENIGN/攻撃を自動判定
- anomalib 用フォルダ構成で保存

出力構成：
dataset_rp/
  train/
    good/
  test/
    anomaly/
"""

from collections import defaultdict
from matplotlib import pyplot as plt
from pyts.image import RecurrencePlot
import pyshark
import numpy as np
import math
import os

# ======== 設定 ========
PCAP_PATH = "Monday-WorkingHours.pcap"  # 対象pcapファイル
OUTPUT_ROOT = "dataset_rp"              # 出力ルートフォルダ
MAX_PACKETS = 10000                     # 処理する最大パケット数
WINDOW_SIZE = 512                       # 1画像あたりの時系列長
RP_TIME_DELAY = 2                       # 再帰プロットの time_delay
RP_DIMENSION = 2                        # 次元
RP_THRESHOLD = "point"                  # 閾値設定（"point"が無閾値）
# ======================


def infer_label_from_filename(filename: str):
    """
    ファイル名から自動で正常／異常を判定
    """
    name = filename.lower()
    if "monday" in name or "benign" in name:
        return "good"
    else:
        return "anomaly"


def ensure_output_dirs(label: str):
    """
    出力フォルダを作成
    """
    if label == "good":
        out_dir = os.path.join(OUTPUT_ROOT, "train", "good")
    else:
        out_dir = os.path.join(OUTPUT_ROOT, "test", "anomaly")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def extract_traffic_series(pcap_path: str, max_packets: int):
    """
    pcapから秒単位トラフィック時系列を抽出
    """
    print(f"📂 pcap読み込み中: {pcap_path}")
    cap = pyshark.FileCapture(pcap_path)
    time_stamps, sizes = [], []

    for i, pkt in enumerate(cap):
        if i >= max_packets:
            break
        try:
            time_stamps.append(float(pkt.sniff_time.timestamp()))
            sizes.append(int(pkt.length))
        except Exception:
            continue

    cap.close()

    if len(time_stamps) == 0:
        raise RuntimeError("⚠️ パケットが読み込めませんでした。")

    # 秒単位に丸めて集約
    ts_rounded = [math.floor(t) for t in time_stamps]
    traffic_per_sec = defaultdict(int)
    for t, s in zip(ts_rounded, sizes):
        traffic_per_sec[t] += s

    sorted_times = sorted(traffic_per_sec.keys())
    traffic_values = [traffic_per_sec[t] for t in sorted_times]
    print(f"📊 時系列長: {len(traffic_values)}")

    return np.array(traffic_values)


def generate_recurrence_plots(series: np.ndarray, out_dir: str,
                              window_size: int, time_delay: int,
                              dimension: int, threshold: str):
    """
    時系列から Recurrence Plot 画像をウィンドウ単位で生成
    """
    rp = RecurrencePlot(time_delay=time_delay,
                        threshold=threshold, dimension=dimension)
    num_windows = len(series) // window_size

    for i in range(num_windows):
        start = i * window_size
        end = start + window_size
        window = series[start:end].reshape(1, -1)

        X_rp = rp.fit_transform(window)
        plt.imshow(X_rp[0], cmap="binary")
        plt.axis("off")
        plt.title(f"Recurrence Plot {i:04d}")
        out_path = os.path.join(out_dir, f"rp_{i:04d}.png")
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
        plt.close()

    print(f"✅ {num_windows} 枚の画像を保存しました: {out_dir}/")


def main():
    label = infer_label_from_filename(PCAP_PATH)
    out_dir = ensure_output_dirs(label)

    series = extract_traffic_series(PCAP_PATH, MAX_PACKETS)
    generate_recurrence_plots(series, out_dir, WINDOW_SIZE,
                              RP_TIME_DELAY, RP_DIMENSION, RP_THRESHOLD)

    print("\n✨ 完了：以下のフォルダに出力しました ✨")
    print(f"- {out_dir}")


if __name__ == "__main__":
    main()
