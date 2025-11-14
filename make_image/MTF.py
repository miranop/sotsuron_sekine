#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSVファイルからMTF(RGB)画像を生成（改良版）
- log1pスケーリング対応
- オーバーラップウィンドウ
- カラム名strip
- bins=8, quantile
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pyts.image import MarkovTransitionField
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ====== 設定 ======
CSV_DIR = "../CSV"
OUT_ROOT = "../datasets/mtf"
TRAIN_GOOD = os.path.join(OUT_ROOT, "train", "good")
TEST_GOOD = os.path.join(OUT_ROOT, "test", "good")
TEST_ANOM_ROOT = os.path.join(OUT_ROOT, "test", "anomaly")

FEATURE_COLUMNS = ["Flow IAT Mean", "Flow IAT Std", "Flow Packets/s"]
LABEL_CANDIDATES = ["Label", "Class", "label", "class"]
BENIGN_TOKENS = {"BENIGN", "Benign", "benign", "NORMAL", "Normal"}

WINDOW_SIZE = 32
STRIDE = WINDOW_SIZE // 2
MTF_BINS = 8
MTF_STRATEGY = "quantile"

MAX_IMAGES_TRAIN_GOOD = 2000
MAX_IMAGES_TEST_GOOD = 1000
MAX_IMAGES_TEST_ANOM = 1000

LOG_FEATURES = {"Flow Bytes/s", "Flow Packets/s",
                "Fwd Packets/s", "Bwd Packets/s"}

# ====== 関数群 ======


def sanitize_label(label_str):
    return label_str.replace(" ", "_").replace("-", "_").replace("/", "_")


def find_label_column(df: pd.DataFrame) -> str:
    for cand in LABEL_CANDIDATES:
        if cand in df.columns:
            return cand
    lower_map = {c.lower().strip(): c for c in df.columns}
    for cand in ["label", "class", "attack_cat", "attack_type"]:
        if cand in lower_map:
            return lower_map[cand]
    raise ValueError("ラベル列が見つかりません")


def ensure_features(df: pd.DataFrame, feature_cols):
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"特徴量カラムが見つかりません: {missing}")


def split_by_label(df: pd.DataFrame, label_col: str):
    df = df.replace([np.inf, -np.inf], np.nan)
    label_series = df[label_col].astype(str).str.strip()
    normal_df = df[label_series.isin(BENIGN_TOKENS)].copy()

    anomaly_dfs = {}
    for lbl in label_series.unique():
        if lbl not in BENIGN_TOKENS:
            anomaly_dfs[lbl] = df[label_series == lbl].copy()
    return normal_df, anomaly_dfs


def fit_scalers_on_normal(normal_df: pd.DataFrame, feature_cols):
    scalers = {}
    for col in feature_cols:
        vals = normal_df[col].astype(float)
        if col in LOG_FEATURES:
            vals = np.log1p(vals)
        vals = vals.replace([np.inf, -np.inf],
                            np.nan).dropna().to_numpy().reshape(-1, 1)
        scaler = MinMaxScaler().fit(vals)
        scalers[col] = ("log1p" if col in LOG_FEATURES else "identity", scaler)
    return scalers


def transform_with_scaler(arr, scaler_info):
    mode, scaler = scaler_info
    arr = np.log1p(arr) if mode == "log1p" else arr
    arr = np.nan_to_num(arr, nan=np.nanmedian(arr))
    return scaler.transform(arr.reshape(-1, 1)).ravel()


def extract_windows_3ch(df: pd.DataFrame, feature_cols, scalers, window_size: int, stride: int):
    scaled_features = []
    for col in feature_cols:
        data = df[col].astype(float).to_numpy()
        scaled = transform_with_scaler(data, scalers[col])
        scaled_features.append(scaled)

    min_len = min(len(f) for f in scaled_features)
    if min_len < window_size:
        return None

    scaled_features = [f[:min_len] for f in scaled_features]
    indices = range(0, min_len - window_size + 1, stride)

    windows_per_channel = []
    for feat in scaled_features:
        windows = [feat[i:i + window_size] for i in indices]
        windows_per_channel.append(np.asarray(windows))
    return windows_per_channel


def mtf_rgb_from_3ch_windows(windows_per_channel, image_size, n_bins=8, strategy="quantile"):
    mtf = MarkovTransitionField(
        image_size=image_size, n_bins=n_bins, strategy=strategy)

    mtf_channels = []
    for ch in windows_per_channel:
        mtf_ch = mtf.fit_transform(ch)
        mtf_channels.append(mtf_ch)

    num_windows = mtf_channels[0].shape[0]
    rgb_list = []
    for i in range(num_windows):
        r = mtf_channels[0][i]
        g = mtf_channels[1][i]
        b = mtf_channels[2][i]
        rgb = np.stack([r, g, b], axis=-1)
        rgb = (rgb * 255.0).clip(0, 255).astype(np.uint8)
        rgb_list.append(rgb)
    return rgb_list


def save_images(rgb_list, out_dir, prefix="img", max_images=None):
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    for idx, rgb in enumerate(rgb_list):
        if max_images is not None and count >= max_images:
            break
        fname = f"{prefix}_{idx:06d}.png"
        Image.fromarray(rgb).save(os.path.join(out_dir, fname))
        count += 1
    return count


# ====== メイン処理 ======

def main():
    print("=" * 80)
    print("CSV → MTF画像変換（攻撃タイプ別 + 時系列分割）")
    print("=" * 80)

    csv_files = []
    for fname in sorted(os.listdir(CSV_DIR)):
        if fname.lower().endswith(".csv"):
            path = os.path.join(CSV_DIR, fname)
            is_train = "monday" in fname.lower()
            csv_files.append({"path": path, "is_train": is_train})

    print("\n[検出されたCSVファイル]")
    for c in csv_files:
        mark = "★" if c["is_train"] else " "
        print(f" {mark} {os.path.basename(c['path'])}")

    total_counts = {"train/good": 0, "test/good": 0}
    all_attack_counts = {}

    print("\n[学習用データ読み込み]")
    train_dfs = []
    for csv_info in csv_files:
        if csv_info["is_train"] and os.path.exists(csv_info["path"]):
            df = pd.read_csv(csv_info["path"])
            df.columns = [c.strip() for c in df.columns]
            train_dfs.append(df)
            print(f"  ✓ {os.path.basename(csv_info['path'])}: {len(df):,} 行")

    if not train_dfs:
        raise RuntimeError("学習用CSVが読み込めませんでした")

    train_df = pd.concat(train_dfs, ignore_index=True)
    label_col = find_label_column(train_df)
    ensure_features(train_df, FEATURE_COLUMNS)

    train_normal, _ = split_by_label(train_df, label_col)
    print(f"  正常データ: {len(train_normal):,} 行")

    scalers = fit_scalers_on_normal(train_normal, FEATURE_COLUMNS)
    print("  ✓ スケーラー学習完了")

    print("\n[学習用画像生成]")
    win_train = extract_windows_3ch(
        train_normal, FEATURE_COLUMNS, scalers, WINDOW_SIZE, STRIDE)
    if win_train:
        rgb_train = mtf_rgb_from_3ch_windows(win_train, image_size=WINDOW_SIZE,
                                             n_bins=MTF_BINS, strategy=MTF_STRATEGY)
        n_train = save_images(rgb_train, TRAIN_GOOD, prefix="mtf_train",
                              max_images=MAX_IMAGES_TRAIN_GOOD)
        total_counts["train/good"] = n_train
        print(f"  train/good: {n_train:,} 枚")

    print("\n[テスト用データ処理]")
    for csv_info in csv_files:
        if csv_info["is_train"] or not os.path.exists(csv_info["path"]):
            continue

        csv_path = csv_info["path"]
        csv_name = os.path.splitext(os.path.basename(csv_path))[0]
        print(f"\n  処理中: {csv_name}")

        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]

        test_normal, test_anomaly_dfs = split_by_label(df, label_col)

        if not test_normal.empty:
            win_test = extract_windows_3ch(
                test_normal, FEATURE_COLUMNS, scalers, WINDOW_SIZE, STRIDE)
            if win_test:
                rgb_test = mtf_rgb_from_3ch_windows(win_test, image_size=WINDOW_SIZE,
                                                    n_bins=MTF_BINS, strategy=MTF_STRATEGY)
                n_test = save_images(rgb_test, TEST_GOOD, prefix=f"mtf_test_good_{csv_name}",
                                     max_images=MAX_IMAGES_TEST_GOOD)
                total_counts["test/good"] += n_test

        for attack_label, attack_df in test_anomaly_dfs.items():
            if attack_df.empty:
                continue

            sanitized = sanitize_label(attack_label)
            win_anom = extract_windows_3ch(
                attack_df, FEATURE_COLUMNS, scalers, WINDOW_SIZE, STRIDE)
            if win_anom:
                rgb_anom = mtf_rgb_from_3ch_windows(win_anom, image_size=WINDOW_SIZE,
                                                    n_bins=MTF_BINS, strategy=MTF_STRATEGY)
                out_dir = os.path.join(TEST_ANOM_ROOT, sanitized)
                n_anom = save_images(rgb_anom, out_dir, prefix=f"mtf_anom_{csv_name}_{sanitized}",
                                     max_images=MAX_IMAGES_TEST_ANOM)
                all_attack_counts[sanitized] = all_attack_counts.get(
                    sanitized, 0) + n_anom

    print("\n" + "=" * 80)
    print("MTF画像変換完了！")
    print("=" * 80)
    print(f"\n【学習用データ】")
    print(f"  train/good: {total_counts['train/good']:,} 枚")
    print(f"\n【テスト用データ】")
    print(f"  test/good: {total_counts['test/good']:,} 枚")

    if all_attack_counts:
        print(f"\n【攻撃タイプ別】")
        for attack_type in sorted(all_attack_counts.keys()):
            print(f"  {attack_type:30s}: {all_attack_counts[attack_type]:,} 枚")

    print(f"\n出力先: {OUT_ROOT}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
