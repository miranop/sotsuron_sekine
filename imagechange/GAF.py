#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CIC-IDS系CSVを Gramian Angular Field (GAF) 画像に変換（BENIGN とそれ以外で分割）
- 正常（BENIGN） → dataset/train/good/ と dataset/test/good/
- 異常（BENIGN以外） → dataset/test/anomaly/
- スケーリングは BENIGN で fit し、両クラスに適用
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pyts.image import GramianAngularField
from PIL import Image


# ====== 設定 ======
CSV_PATH = "Monday-WorkingHours.pcap_ISCX.csv"

FEATURE_COLUMNS = [" Flow IAT Mean", " Flow Duration", "Flow Bytes/s"]

LABEL_CANDIDATES = [" Label", "Label", "label", " class", "Class"]
BENIGN_TOKENS = {"BENIGN", "Benign", "benign", "NORMAL", "Normal"}

WINDOW_SIZE = 32
GAF_METHOD = "difference"  # "summation" でもOK

OUT_ROOT = "dataset_gaf"
TRAIN_GOOD = os.path.join(OUT_ROOT, "train", "good")
TEST_GOOD = os.path.join(OUT_ROOT, "test", "good")
TEST_ANOM = os.path.join(OUT_ROOT, "test", "anomaly")

MAX_IMAGES_TRAIN_GOOD = None
MAX_IMAGES_TEST_GOOD = None
MAX_IMAGES_TEST_ANOM = None
# ===================


def find_label_column(df: pd.DataFrame) -> str:
    for cand in LABEL_CANDIDATES:
        if cand in df.columns:
            return cand
    lower_map = {c.lower().strip(): c for c in df.columns}
    for cand in ["label", "class", "attack_cat", "attack_type"]:
        if cand in lower_map:
            return lower_map[cand]
    raise ValueError("ラベル列（Label/Class）が見つかりません。LABEL_CANDIDATES を調整してください。")


def ensure_features(df: pd.DataFrame, feature_cols):
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"特徴量カラムが見つかりません: {missing}\n"
                         f"CSVの列名を確認するか FEATURE_COLUMNS を合わせてください。")


def split_normal_anomaly(df: pd.DataFrame, label_col: str):
    df = df.replace([np.inf, -np.inf], np.nan)
    label_series = df[label_col].astype(str).str.strip()
    is_benign = label_series.isin(BENIGN_TOKENS)

    normal_df = df[is_benign].copy()
    anomaly_df = df[~is_benign].copy()

    if normal_df.empty:
        raise ValueError("BENIGN（正常）行が見つかりません。BENIGN_TOKENS を見直してください。")
    if anomaly_df.empty:
        print("警告: 異常行がゼロです。今回は正常画像だけを出力します。")

    return normal_df, anomaly_df


def fit_scalers_on_normal(normal_df: pd.DataFrame, feature_cols):
    scalers = {}
    for col in feature_cols:
        vals = normal_df[col].dropna().values.reshape(-1, 1)
        if vals.size == 0:
            raise ValueError(f"正常データ内で {col} に有効値がありません。")
        scaler = MinMaxScaler()
        scaler.fit(vals)
        scalers[col] = scaler
    return scalers


def extract_windows_3ch(df: pd.DataFrame, feature_cols, scalers, window_size: int):
    scaled_features = []
    for col in feature_cols:
        data = df[col].dropna().values.reshape(-1, 1)
        if data.size == 0:
            raise ValueError(f"{col} に有効値がありません。")
        scaled = scalers[col].transform(data).flatten()
        scaled_features.append(scaled)

    min_len = min(len(f) for f in scaled_features)
    scaled_features = [f[:min_len] for f in scaled_features]
    if min_len < window_size:
        return None

    num_windows = min_len // window_size
    windows_per_channel = []
    for feat in scaled_features:
        windows = [feat[i * window_size:(i + 1) * window_size]
                   for i in range(num_windows)]
        windows_per_channel.append(np.asarray(windows))
    return windows_per_channel


def gaf_rgb_from_3ch_windows(windows_per_channel, image_size, method="difference"):
    gaf = GramianAngularField(image_size=image_size, method=method)
    gaf_channels = []
    for ch in windows_per_channel:
        gaf_ch = gaf.fit_transform(ch)  # shape: [num_windows, H, W], 値域 [-1,1]
        gaf_channels.append(gaf_ch)

    num_windows = gaf_channels[0].shape[0]
    rgb_list = []
    for i in range(num_windows):
        r = gaf_channels[0][i]
        g = gaf_channels[1][i]
        b = gaf_channels[2][i]
        rgb = np.stack([r, g, b], axis=-1)
        rgb = ((rgb + 1) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
        rgb_list.append(rgb)
    return rgb_list


def save_images(rgb_list, out_dir, prefix="img", max_images=None, start_index=0):
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    for idx, rgb in enumerate(rgb_list):
        if max_images is not None and count >= max_images:
            break
        fname = f"{prefix}_{start_index + idx:06d}.png"
        Image.fromarray(rgb).save(os.path.join(out_dir, fname))
        count += 1
    return count


def main():
    df = pd.read_csv(CSV_PATH)
    print("CSVファイルを読み込みました。")

    label_col = find_label_column(df)
    print(f"ラベル列: {label_col!r}")

    ensure_features(df, FEATURE_COLUMNS)
    normal_df, anomaly_df = split_normal_anomaly(df, label_col)
    print(f"正常行: {len(normal_df)}  異常行: {len(anomaly_df)}")

    scalers = fit_scalers_on_normal(normal_df, FEATURE_COLUMNS)
    print("MinMaxScaler を正常データで fit 済み。")

    win_normal = extract_windows_3ch(
        normal_df, FEATURE_COLUMNS, scalers, WINDOW_SIZE)
    if win_normal is None:
        raise RuntimeError("正常データから窓が作れませんでした。WINDOW_SIZE を小さくしてください。")

    rgb_normal = gaf_rgb_from_3ch_windows(
        win_normal, image_size=WINDOW_SIZE, method=GAF_METHOD)
    n_train = save_images(rgb_normal, TRAIN_GOOD, prefix="rgb_gaf_train_good",
                          max_images=MAX_IMAGES_TRAIN_GOOD, start_index=0)
    print(f"train/good: {n_train} 枚保存")

    rest_normal = rgb_normal[n_train:] if MAX_IMAGES_TRAIN_GOOD is not None else rgb_normal
    n_test_good = save_images(rest_normal, TEST_GOOD, prefix="rgb_gaf_test_good",
                              max_images=MAX_IMAGES_TEST_GOOD, start_index=0)
    print(f"test/good: {n_test_good} 枚保存")

    n_test_anom = 0
    if not anomaly_df.empty:
        win_anom = extract_windows_3ch(
            anomaly_df, FEATURE_COLUMNS, scalers, WINDOW_SIZE)
        if win_anom is None:
            print("警告: 異常データから窓が作れなかったため、test/anomaly は 0 枚です。")
        else:
            rgb_anom = gaf_rgb_from_3ch_windows(
                win_anom, image_size=WINDOW_SIZE, method=GAF_METHOD)
            n_test_anom = save_images(rgb_anom, TEST_ANOM, prefix="rgb_gaf_test_anomaly",
                                      max_images=MAX_IMAGES_TEST_ANOM, start_index=0)
    print(f"test/anomaly: {n_test_anom} 枚保存")

    print("\n完了：anomalib 用フォルダが整いました。構成は以下の通りです：")
    print(f"- {TRAIN_GOOD}/  ... 正常（学習）")
    print(f"- {TEST_GOOD}/   ... 正常（テスト）")
    print(f"- {TEST_ANOM}/   ... 異常（テスト）")


if __name__ == "__main__":
    main()
