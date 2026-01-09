#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSVファイルからMTF(RGB)画像を生成 - 最終調整版
CICIDS-2017データセット用

追加機能:
- test/goodからtrain/goodへの画像移動機能
- 最終的な画像枚数の詳細レポート
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pyts.image import MarkovTransitionField
from PIL import Image
from collections import defaultdict
import random
import shutil

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

WINDOW_SIZE = 64
STRIDE = WINDOW_SIZE // 2  # オーバーラップウィンドウ
MTF_BINS = 8
MTF_STRATEGY = "quantile"

# データ分割設定
TRAIN_RATIO = 0.7
RANDOM_SEED = 42

# 攻撃タイプ別の画像数制限
MIN_IMAGES_PER_ATTACK_TRAIN = 30
MIN_IMAGES_PER_ATTACK_TEST = 20
MAX_IMAGES_PER_ATTACK_TRAIN = 800
MAX_IMAGES_PER_ATTACK_TEST = 400

# 正常データの制限
MAX_IMAGES_TRAIN_GOOD = 2000
MAX_IMAGES_TEST_GOOD = 1000

# test/goodからtrain/goodへ移動する割合（0.0〜1.0）
MOVE_RATIO_FROM_TEST_TO_TRAIN = 0.6

# log1p変換する特徴量
LOG_FEATURES = {"Flow Bytes/s", "Flow Packets/s",
                "Fwd Packets/s", "Bwd Packets/s"}

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def sanitize_label(label_str):
    return label_str.replace(" ", "_").replace("-", "_").replace("/", "_")


def find_label_column(df):
    for cand in LABEL_CANDIDATES:
        if cand in df.columns:
            return cand
    lower_map = {c.lower().strip(): c for c in df.columns}
    for cand in ["label", "class", "attack_cat", "attack_type"]:
        if cand in lower_map:
            return lower_map[cand]
    raise ValueError("ラベル列が見つかりません")


def ensure_features(df, feature_cols):
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"特徴量カラムが見つかりません: {missing}")


def count_images_in_directory(directory):
    """ディレクトリ内の画像ファイル数をカウント"""
    if not os.path.exists(directory):
        return 0
    return len([f for f in os.listdir(directory)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))])


def redistribute_test_to_train(test_good_dir, train_good_dir, move_ratio=0.6):
    """
    test/goodの画像の一部をtrain/goodに移動

    Args:
        test_good_dir: テスト用正常画像ディレクトリ
        train_good_dir: 学習用正常画像ディレクトリ
        move_ratio: 移動する割合（0.0〜1.0）

    Returns:
        移動した画像数
    """
    print("\n" + "="*80)
    print("🔄 画像再配分: test/good → train/good")
    print("="*80)

    if not os.path.exists(test_good_dir):
        print(f"  ⚠️ test/goodディレクトリが存在しません")
        return 0

    os.makedirs(train_good_dir, exist_ok=True)

    # test/goodの画像ファイル一覧を取得
    test_images = [f for f in os.listdir(test_good_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not test_images:
        print(f"  ⚠️ test/goodに画像が見つかりません")
        return 0

    original_count = len(test_images)
    print(f"\n  移動前:")
    print(f"    test/good:  {original_count:,}枚")
    print(f"    train/good: {count_images_in_directory(train_good_dir):,}枚")

    # ランダムに移動対象を選択
    num_to_move = int(original_count * move_ratio)
    random.shuffle(test_images)
    images_to_move = test_images[:num_to_move]

    print(f"\n  移動設定: {move_ratio:.1%} ({num_to_move:,}枚)")

    # 画像を移動
    moved_count = 0
    for img_file in images_to_move:
        src = os.path.join(test_good_dir, img_file)
        dst = os.path.join(train_good_dir, img_file)

        try:
            shutil.move(src, dst)
            moved_count += 1
        except Exception as e:
            print(f"  ⚠️ 移動エラー ({img_file}): {e}")

    print(f"\n  移動後:")
    print(f"    test/good:  {count_images_in_directory(test_good_dir):,}枚")
    print(f"    train/good: {count_images_in_directory(train_good_dir):,}枚")
    print(f"  ✓ {moved_count:,}枚を移動完了")

    return moved_count


def print_final_dataset_statistics(out_root):
    """最終的なデータセット統計を表示"""
    print("\n" + "="*80)
    print("📊 最終データセット統計")
    print("="*80)

    train_good_dir = os.path.join(out_root, "train", "good")
    test_good_dir = os.path.join(out_root, "test", "good")
    test_anomaly_root = os.path.join(out_root, "test", "anomaly")

    # 学習用正常画像
    train_good_count = count_images_in_directory(train_good_dir)
    print(f"\n【学習用データ】")
    print(f"  train/good: {train_good_count:,}枚")

    # テスト用正常画像
    test_good_count = count_images_in_directory(test_good_dir)
    print(f"\n【テスト用データ】")
    print(f"  test/good:  {test_good_count:,}枚")

    # テスト用異常画像（攻撃タイプ別）
    attack_stats = {}
    total_anomaly = 0

    if os.path.exists(test_anomaly_root):
        attack_dirs = [d for d in os.listdir(test_anomaly_root)
                       if os.path.isdir(os.path.join(test_anomaly_root, d))]

        if attack_dirs:
            print(f"\n【攻撃タイプ別（テスト用）】")
            for attack_type in sorted(attack_dirs):
                attack_path = os.path.join(test_anomaly_root, attack_type)
                count = count_images_in_directory(attack_path)
                attack_stats[attack_type] = count
                total_anomaly += count

                status = "✅" if count >= MIN_IMAGES_PER_ATTACK_TEST else "⚠️"
                print(f"  {status} test/anomaly/{attack_type:30s}: {count:,}枚")

            print(f"\n  {'テスト用異常画像合計':33s}: {total_anomaly:,}枚")

    # 総計
    total_images = train_good_count + test_good_count + total_anomaly
    print(f"\n{'='*80}")
    print(f"  総画像数: {total_images:,}枚")
    print(f"{'='*80}")

    # データバランス確認
    if train_good_count > 0 and test_good_count > 0:
        ratio = test_good_count / train_good_count
        print(f"\n  正常画像の比率 (test/train): {ratio:.2f}")
        if ratio > 1.5:
            print(f"  ⚠️ test/goodの方が{ratio:.1f}倍多いです")
        elif ratio < 0.5:
            print(f"  ⚠️ train/goodの方が{1/ratio:.1f}倍多いです")
        else:
            print(f"  ✅ バランスが取れています")


def analyze_dataset_distribution(csv_files):
    print("\n" + "="*80)
    print("📊 データセット分布分析")
    print("="*80)

    distribution = defaultdict(lambda: {"count": 0, "files": []})

    for csv_info in csv_files:
        try:
            df = pd.read_csv(csv_info["path"])
            df.columns = [c.strip() for c in df.columns]
            label_col = find_label_column(df)
            labels = df[label_col].astype(str).str.strip()

            for label in labels.unique():
                count = (labels == label).sum()
                if label in BENIGN_TOKENS:
                    label = "BENIGN"
                distribution[label]["count"] += count
        except Exception as e:
            print(f"  ⚠️ {os.path.basename(csv_info['path'])}: 読み込みエラー - {e}")

    print("\n[ラベル別データ数]")
    sorted_labels = sorted(distribution.items(),
                           key=lambda x: x[1]["count"], reverse=True)

    total_rows = 0
    for label, info in sorted_labels:
        total_rows += info["count"]
        marker = "🟢" if label == "BENIGN" else "🔴"
        print(f"  {marker} {label:30s}: {info['count']:>10,}行")

    print(f"\n  {'合計':31s}: {total_rows:>10,}行")
    print("\n[推定画像数（ウィンドウサイズ=32, stride=16の場合）]")
    print("  ※オーバーラップウィンドウのため、実際の画像数は増加します")

    for label, info in sorted_labels:
        estimated_windows = (info["count"] - WINDOW_SIZE) // STRIDE + 1
        if estimated_windows < 0:
            estimated_windows = 0
        marker = "🟢" if label == "BENIGN" else "🔴"
        print(f"  {marker} {label:30s}: 約{estimated_windows:>8,}枚")

    return distribution


def split_csv_files_for_train_test(csv_files, train_ratio=0.7):
    monday_files = [c for c in csv_files if "monday" in c["path"].lower()]
    other_files = [c for c in csv_files if "monday" not in c["path"].lower()]

    random.shuffle(other_files)
    split_idx = int(len(other_files) * train_ratio)

    train_files = monday_files + other_files[:split_idx]
    test_files = other_files[split_idx:]

    for f in train_files:
        f["is_train"] = True
    for f in test_files:
        f["is_train"] = False

    return train_files, test_files


def split_by_label(df, label_col):
    df = df.replace([np.inf, -np.inf], np.nan)
    label_series = df[label_col].astype(str).str.strip()
    normal_df = df[label_series.isin(BENIGN_TOKENS)].copy()

    anomaly_dfs = {}
    for lbl in label_series.unique():
        if lbl not in BENIGN_TOKENS:
            anomaly_dfs[lbl] = df[label_series == lbl].copy()

    return normal_df, anomaly_dfs


def fit_scalers_on_normal(normal_df, feature_cols):
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
    arr = np.nan_to_num(arr, nan=np.nanmedian(arr) if len(arr) > 0 else 0.0)
    return scaler.transform(arr.reshape(-1, 1)).ravel()


def extract_windows_3ch(df, feature_cols, scalers, window_size, stride):
    """
    3特徴量からウィンドウを抽出（オーバーラップ対応）
    時系列順序を保持
    """
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
    """MTF変換してRGB画像生成"""
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


def save_images_with_limit(rgb_list, out_dir, prefix="img", max_images=None):
    """
    画像を保存（枚数制限付き）
    時系列順序を保持するため、先頭からmax_images枚を保存
    """
    os.makedirs(out_dir, exist_ok=True)

    if max_images is not None and len(rgb_list) > max_images:
        rgb_list = rgb_list[:max_images]

    count = 0
    for idx, rgb in enumerate(rgb_list):
        fname = f"{prefix}_{idx:06d}.png"
        Image.fromarray(rgb).save(os.path.join(out_dir, fname))
        count += 1

    return count


def verify_dataset_structure():
    """生成されたデータセットの構造を確認"""
    print("\n" + "="*80)
    print("📋 データセット読み込み確認")
    print("="*80)

    dirs_to_check = {
        "学習用正常画像": TRAIN_GOOD,
        "テスト用正常画像": TEST_GOOD,
    }

    counts = {}
    for name, path in dirs_to_check.items():
        print(f"\n[{name}]")
        print(f"  パス: {path}")

        if not os.path.exists(path):
            print(f"  ❌ ディレクトリが存在しません")
            counts[name] = 0
            continue

        image_files = [f for f in os.listdir(path)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        count = len(image_files)
        counts[name] = count

        if count == 0:
            print(f"  ⚠️ 画像ファイルが見つかりません")
        else:
            print(f"  ✅ {count:,}枚の画像を検出")

    print(f"\n[攻撃タイプ別ディレクトリ]")
    print(f"  パス: {TEST_ANOM_ROOT}")

    total_anomaly_count = 0
    attack_type_info = {}

    if os.path.exists(TEST_ANOM_ROOT):
        attack_dirs = [d for d in os.listdir(TEST_ANOM_ROOT)
                       if os.path.isdir(os.path.join(TEST_ANOM_ROOT, d))]

        if attack_dirs:
            print(f"  ✅ {len(attack_dirs)}種類の攻撃タイプを検出")
            for attack_type in sorted(attack_dirs):
                attack_path = os.path.join(TEST_ANOM_ROOT, attack_type)
                attack_images = [f for f in os.listdir(attack_path)
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                count = len(attack_images)
                total_anomaly_count += count
                attack_type_info[attack_type] = count

                status = "✅" if count >= MIN_IMAGES_PER_ATTACK_TEST else "⚠️"
                print(f"    {status} {attack_type}: {count:,}枚")

    print("\n" + "="*80)
    print("📊 データセットサマリー")
    print("="*80)
    print(f"\n  学習用正常画像:     {counts.get('学習用正常画像', 0):,}枚")
    print(f"  テスト用正常画像:   {counts.get('テスト用正常画像', 0):,}枚")
    print(
        f"  テスト用異常画像:   {total_anomaly_count:,}枚 ({len(attack_type_info)}種類)")

    print("\n" + "="*80)
    print("⚙️ 適用された設定")
    print("="*80)
    print(f"  学習/テスト分割比: {TRAIN_RATIO:.1%} / {1-TRAIN_RATIO:.1%}")
    print(f"  ランダムシード: {RANDOM_SEED}")
    print(f"  ウィンドウサイズ: {WINDOW_SIZE}")
    print(f"  ストライド: {STRIDE} (オーバーラップ率: {(1-STRIDE/WINDOW_SIZE)*100:.0f}%)")
    print(f"  MTF bins: {MTF_BINS}")
    print(f"  MTF strategy: {MTF_STRATEGY}")
    print(f"  画像再配分比率: test→trainに{MOVE_RATIO_FROM_TEST_TO_TRAIN:.1%}移動")
    print(f"\n  攻撃タイプ別の制限:")
    print(
        f"    学習用: {MIN_IMAGES_PER_ATTACK_TRAIN}枚〜{MAX_IMAGES_PER_ATTACK_TRAIN}枚")
    print(
        f"    テスト用: {MIN_IMAGES_PER_ATTACK_TEST}枚〜{MAX_IMAGES_PER_ATTACK_TEST}枚")


def main():
    print("=" * 80)
    print("CSV → MTF画像変換（最終調整版）")
    print("=" * 80)

    if not os.path.exists(CSV_DIR):
        print(f"\n❌ エラー: CSVディレクトリが見つかりません: {CSV_DIR}")
        sys.exit(1)

    csv_files = []
    for fname in sorted(os.listdir(CSV_DIR)):
        if fname.lower().endswith(".csv"):
            path = os.path.join(CSV_DIR, fname)
            csv_files.append({"path": path, "is_train": False})

    if not csv_files:
        print(f"\n❌ エラー: CSVファイルが見つかりません: {CSV_DIR}")
        sys.exit(1)

    print(f"\n[検出されたCSVファイル: {len(csv_files)}個]")
    for c in csv_files:
        print(f"  - {os.path.basename(c['path'])}")

    distribution = analyze_dataset_distribution(csv_files)

    print("\n" + "="*80)
    print("📂 Train/Test分割")
    print("="*80)

    train_files, test_files = split_csv_files_for_train_test(
        csv_files, TRAIN_RATIO)

    print(f"\n  学習用: {len(train_files)}ファイル")
    for f in train_files:
        print(f"    ★ {os.path.basename(f['path'])}")

    print(f"\n  テスト用: {len(test_files)}ファイル")
    for f in test_files:
        print(f"      {os.path.basename(f['path'])}")

    attack_stats = {
        "train": defaultdict(int),
        "test": defaultdict(int)
    }

    print("\n" + "="*80)
    print("Phase 1: 学習用データ読み込み & スケーラー学習")
    print("="*80)

    train_dfs = []
    for csv_info in train_files:
        try:
            df = pd.read_csv(csv_info["path"])
            df.columns = [c.strip() for c in df.columns]
            train_dfs.append(df)
            print(f"  ✓ {os.path.basename(csv_info['path'])}: {len(df):,}行")
        except Exception as e:
            print(f"  ❌ {os.path.basename(csv_info['path'])}: エラー - {e}")

    if not train_dfs:
        print("\n❌ エラー: 学習用CSVが読み込めませんでした")
        sys.exit(1)

    train_df = pd.concat(train_dfs, ignore_index=True)
    label_col = find_label_column(train_df)
    ensure_features(train_df, FEATURE_COLUMNS)

    train_normal, train_anomaly_dfs = split_by_label(train_df, label_col)
    print(f"\n  正常データ: {len(train_normal):,}行")
    print(
        f"  異常データ: {sum(len(df) for df in train_anomaly_dfs.values()):,}行 ({len(train_anomaly_dfs)}種類)")

    scalers = fit_scalers_on_normal(train_normal, FEATURE_COLUMNS)
    print("  ✓ スケーラー学習完了")

    print("\n" + "="*80)
    print("Phase 2: 学習用画像生成")
    print("="*80)

    print("\n[正常データ]")
    win_train = extract_windows_3ch(
        train_normal, FEATURE_COLUMNS, scalers, WINDOW_SIZE, STRIDE)
    if win_train:
        print(f"  ウィンドウ数: {len(win_train[0]):,}")
        rgb_train = mtf_rgb_from_3ch_windows(win_train, image_size=WINDOW_SIZE,
                                             n_bins=MTF_BINS, strategy=MTF_STRATEGY)
        n_train = save_images_with_limit(rgb_train, TRAIN_GOOD, prefix="mtf_train",
                                         max_images=MAX_IMAGES_TRAIN_GOOD)
        attack_stats["train"]["BENIGN"] = n_train
        print(f"  ✓ 保存完了: {n_train:,}枚 → {TRAIN_GOOD}")

    print("\n" + "="*80)
    print("Phase 3: テスト用画像生成")
    print("="*80)

    for csv_info in test_files:
        csv_path = csv_info["path"]
        csv_name = os.path.splitext(os.path.basename(csv_path))[0]
        print(f"\n  処理中: {csv_name}")

        try:
            df = pd.read_csv(csv_path)
            df.columns = [c.strip() for c in df.columns]
            test_normal, test_anomaly_dfs = split_by_label(df, label_col)

            if not test_normal.empty:
                win_test = extract_windows_3ch(
                    test_normal, FEATURE_COLUMNS, scalers, WINDOW_SIZE, STRIDE)
                if win_test:
                    rgb_test = mtf_rgb_from_3ch_windows(win_test, image_size=WINDOW_SIZE,
                                                        n_bins=MTF_BINS, strategy=MTF_STRATEGY)
                    n_test = save_images_with_limit(rgb_test, TEST_GOOD,
                                                    prefix=f"mtf_test_good_{csv_name}",
                                                    max_images=MAX_IMAGES_TEST_GOOD)
                    attack_stats["test"]["BENIGN"] += n_test
                    print(f"    - 正常: {n_test:,}枚")

            for attack_label, attack_df in test_anomaly_dfs.items():
                if attack_df.empty:
                    continue

                sanitized = sanitize_label(attack_label)
                win_anom = extract_windows_3ch(
                    attack_df, FEATURE_COLUMNS, scalers, WINDOW_SIZE, STRIDE)

                if win_anom:
                    rgb_anom = mtf_rgb_from_3ch_windows(win_anom, image_size=WINDOW_SIZE,
                                                        n_bins=MTF_BINS, strategy=MTF_STRATEGY)
                    available_count = len(rgb_anom)
                    actual_count = min(
                        available_count, MAX_IMAGES_PER_ATTACK_TEST)

                    out_dir = os.path.join(TEST_ANOM_ROOT, sanitized)
                    n_anom = save_images_with_limit(
                        rgb_anom, out_dir,
                        prefix=f"mtf_anom_{csv_name}_{sanitized}",
                        max_images=MAX_IMAGES_PER_ATTACK_TEST)

                    attack_stats["test"][sanitized] += n_anom

                    status = "✓" if n_anom >= MIN_IMAGES_PER_ATTACK_TEST else "⚠️"
                    print(f"    {status} {sanitized}: {n_anom:,}枚", end="")
                    if available_count > MAX_IMAGES_PER_ATTACK_TEST:
                        print(f" (制限適用: 元{available_count:,}枚)")
                    else:
                        print()

        except Exception as e:
            print(f"    ❌ エラー: {e}")
            continue

    # ★ 新機能: test/goodからtrain/goodへの画像再配分
    if MOVE_RATIO_FROM_TEST_TO_TRAIN > 0:
        redistribute_test_to_train(TEST_GOOD, TRAIN_GOOD,
                                   MOVE_RATIO_FROM_TEST_TO_TRAIN)

    print("\n" + "=" * 80)
    print("✅ MTF画像変換完了！")
    print("=" * 80)

    # ★ 新機能: 最終的な画像枚数統計を表示
    print_final_dataset_statistics(OUT_ROOT)

    print(f"\n📁 出力先: {OUT_ROOT}/")

    print("\n" + "=" * 80)
    print("🎉 処理完了")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n処理を中断しました")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
