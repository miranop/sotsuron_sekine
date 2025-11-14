#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSVファイルからGAF(RGB)画像を生成
統計特徴量を時系列データに変換してGAF変換

このバージョン:
- 攻撃タイプごとにサブディレクトリを作成
- データセット読み込み確認機能を追加
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pyts.image import GramianAngularField
from PIL import Image
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ====== 設定 ======
CSV_DIR = "../CSV"
OUT_ROOT = "../datasets/gaf"
TRAIN_GOOD = os.path.join(OUT_ROOT, "train", "good")
TEST_GOOD = os.path.join(OUT_ROOT, "test", "good")
TEST_ANOMALY_ROOT = os.path.join(OUT_ROOT, "test", "anomaly")  # 攻撃タイプ別のルート

FEATURE_COLUMNS = [" Flow IAT Mean", " Flow IAT Std", " Flow Packets/s"]
LABEL_CANDIDATES = [" Label", "Label", "label", " class", "Class"]
BENIGN_TOKENS = {"BENIGN", "Benign", "benign", "NORMAL", "Normal"}

WINDOW_SIZE = 32
GAF_METHOD = "difference"  # "summation" または "difference"

MAX_IMAGES_TRAIN_GOOD = 2000
MAX_IMAGES_TEST_GOOD = 1000
MAX_IMAGES_TEST_ANOM = 1000


def sanitize_label(label_str):
    """ラベル名をファイルシステムで使える形式に変換"""
    return label_str.replace(" ", "_").replace("-", "_").replace("/", "_")


def find_label_column(df: pd.DataFrame) -> str:
    """ラベル列を探す"""
    for cand in LABEL_CANDIDATES:
        if cand in df.columns:
            return cand
    lower_map = {c.lower().strip(): c for c in df.columns}
    for cand in ["label", "class", "attack_cat", "attack_type"]:
        if cand in lower_map:
            return lower_map[cand]
    raise ValueError("ラベル列が見つかりません")


def ensure_features(df: pd.DataFrame, feature_cols):
    """特徴量の存在確認"""
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"特徴量カラムが見つかりません: {missing}")


def split_by_label(df: pd.DataFrame, label_col: str):
    """ラベルごとにデータを分割"""
    df = df.replace([np.inf, -np.inf], np.nan)
    label_series = df[label_col].astype(str).str.strip()

    normal_df = df[label_series.isin(BENIGN_TOKENS)].copy()

    # 攻撃タイプ別に分割
    anomaly_dfs = {}
    for lbl in label_series.unique():
        if lbl not in BENIGN_TOKENS:
            anomaly_dfs[lbl] = df[label_series == lbl].copy()

    return normal_df, anomaly_dfs


def fit_scalers_on_normal(normal_df: pd.DataFrame, feature_cols):
    """正常データでMinMaxScalerを学習"""
    scalers = {}
    for col in feature_cols:
        vals = normal_df[col].dropna().values.reshape(-1, 1)
        if vals.size == 0:
            raise ValueError(f"正常データ内で {col} に有効値がありません")
        scaler = MinMaxScaler()
        scaler.fit(vals)
        scalers[col] = scaler
    return scalers


def extract_windows_3ch(df: pd.DataFrame, feature_cols, scalers, window_size: int):
    """3特徴量からウィンドウを抽出"""
    scaled_features = []
    for col in feature_cols:
        data = df[col].dropna().values.reshape(-1, 1)
        if data.size == 0:
            return None
        scaled = scalers[col].transform(data).flatten()
        scaled_features.append(scaled)

    min_len = min(len(f) for f in scaled_features)
    scaled_features = [f[:min_len] for f in scaled_features]

    if min_len < window_size:
        return None

    num_windows = min_len // window_size
    if num_windows == 0:
        return None

    windows_per_channel = []
    for feat in scaled_features:
        windows = [feat[i * window_size:(i + 1) * window_size]
                   for i in range(num_windows)]
        windows_per_channel.append(np.asarray(windows))

    return windows_per_channel


def gaf_rgb_from_3ch_windows(windows_per_channel, image_size, method="difference"):
    """GAF変換してRGB画像生成"""
    gaf = GramianAngularField(image_size=image_size, method=method)

    gaf_channels = []
    for ch in windows_per_channel:
        gaf_ch = gaf.fit_transform(ch)  # shape: [num_windows, H, W]
        gaf_channels.append(gaf_ch)

    assert len(gaf_channels) == 3
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


def save_images(rgb_list, out_dir, prefix="img", max_images=None):
    """画像を保存"""
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    for idx, rgb in enumerate(rgb_list):
        if max_images is not None and count >= max_images:
            break
        fname = f"{prefix}_{idx:06d}.png"
        Image.fromarray(rgb).save(os.path.join(out_dir, fname))
        count += 1
    return count


def verify_dataset_structure():
    """生成されたデータセットの構造を確認"""
    print("\n" + "="*80)
    print("📋 データセット読み込み確認")
    print("="*80)

    # 基本ディレクトリの存在確認
    dirs_to_check = {
        "学習用正常画像": TRAIN_GOOD,
        "テスト用正常画像": TEST_GOOD,
    }

    all_ok = True
    counts = {}

    for name, path in dirs_to_check.items():
        print(f"\n[{name}]")
        print(f"  パス: {path}")

        if not os.path.exists(path):
            print(f"  ❌ ディレクトリが存在しません")
            all_ok = False
            counts[name] = 0
            continue

        # 画像ファイルをカウント
        image_files = [f for f in os.listdir(path)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        count = len(image_files)
        counts[name] = count

        if count == 0:
            print(f"  ⚠️ 画像ファイルが見つかりません")
            all_ok = False
        else:
            print(f"  ✅ {count:,}枚の画像を検出")

            # サンプルファイル名を表示
            print(f"  サンプルファイル:")
            for i, fname in enumerate(sorted(image_files)[:3]):
                print(f"    - {fname}")
            if count > 3:
                print(f"    ... (他 {count-3:,}枚)")

    # 攻撃タイプ別ディレクトリの確認
    print(f"\n[攻撃タイプ別ディレクトリ]")
    print(f"  パス: {TEST_ANOMALY_ROOT}")

    total_anomaly_count = 0
    attack_type_info = {}

    if os.path.exists(TEST_ANOMALY_ROOT):
        attack_dirs = [d for d in os.listdir(TEST_ANOMALY_ROOT)
                       if os.path.isdir(os.path.join(TEST_ANOMALY_ROOT, d))]

        if attack_dirs:
            print(f"  ✅ {len(attack_dirs)}種類の攻撃タイプを検出")
            for attack_type in sorted(attack_dirs):
                attack_path = os.path.join(TEST_ANOMALY_ROOT, attack_type)
                attack_images = [f for f in os.listdir(attack_path)
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                count = len(attack_images)
                total_anomaly_count += count
                attack_type_info[attack_type] = count
                print(f"    - {attack_type}: {count:,}枚")
        else:
            print(f"  ⚠️ 攻撃タイプディレクトリが見つかりません")
            all_ok = False
    else:
        print(f"  ❌ ディレクトリが存在しません")
        all_ok = False

    # サマリー
    print("\n" + "="*80)
    print("📊 データセットサマリー")
    print("="*80)
    print(f"\n  学習用正常画像:     {counts.get('学習用正常画像', 0):,}枚")
    print(f"  テスト用正常画像:   {counts.get('テスト用正常画像', 0):,}枚")
    print(
        f"  テスト用異常画像:   {total_anomaly_count:,}枚 ({len(attack_type_info)}種類)")
    print(
        f"  合計:               {counts.get('学習用正常画像', 0) + counts.get('テスト用正常画像', 0) + total_anomaly_count:,}枚")

    # 警告チェック
    print("\n" + "="*80)
    print("⚠️ データセット検証")
    print("="*80)

    warnings = []
    errors = []

    if counts.get('学習用正常画像', 0) == 0:
        errors.append("❌ 学習用正常画像が0枚です（学習不可）")
    elif counts.get('学習用正常画像', 0) < 100:
        warnings.append(f"⚠️ 学習用正常画像が少なすぎます（{counts['学習用正常画像']}枚 < 100枚推奨）")

    if total_anomaly_count == 0:
        errors.append("❌ テスト用異常画像が0枚です（評価不可）")
    elif total_anomaly_count < 30:
        warnings.append(f"⚠️ テスト用異常画像が少ないです（{total_anomaly_count}枚 < 30枚推奨）")

    if counts.get('テスト用正常画像', 0) == 0:
        warnings.append("⚠️ テスト用正常画像が0枚です")
    elif counts.get('テスト用正常画像', 0) < 30:
        warnings.append(f"⚠️ テスト用正常画像が少ないです（{counts['テスト用正常画像']}枚 < 30枚推奨）")

    # 攻撃タイプごとの画像数チェック
    for attack_type, count in attack_type_info.items():
        if count < 10:
            warnings.append(f"⚠️ {attack_type}の画像が少ないです（{count}枚 < 10枚推奨）")

    if errors:
        print("\n致命的エラー:")
        for err in errors:
            print(f"  {err}")
        all_ok = False

    if warnings:
        print("\n注意事項:")
        for warn in warnings:
            print(f"  {warn}")

    if not errors and not warnings:
        print("\n✅ データセット検証: すべて正常")

    # 推奨事項
    print("\n" + "="*80)
    print("💡 推奨データ量")
    print("="*80)
    print("  学習用正常画像: 最低100枚（理想は500枚以上）")
    print("  テスト用正常画像: 30枚以上")
    print("  テスト用異常画像: 攻撃タイプごとに10枚以上")
    print("  正常:異常の比率: 1:1〜5:1程度")

    # ディレクトリ構造の説明
    print("\n" + "="*80)
    print("📁 生成されたディレクトリ構造")
    print("="*80)
    print(f"""
{OUT_ROOT}/
├── train/
│   └── good/                    ← 学習用正常画像
└── test/
    ├── good/                    ← テスト用正常画像
    └── anomaly/                 ← 攻撃タイプ別ディレクトリ
        ├── DDoS/
        │   └── gaf_anom_xxx.png
        ├── PortScan/
        │   └── gaf_anom_xxx.png
        ├── BruteForce/
        │   └── gaf_anom_xxx.png
        └── ...

💡 使い方:
  攻撃タイプごとに整理されているので:
  - 手動で各攻撃タイプを確認・分析できます
  - 特定の攻撃タイプだけを使った実験も可能です
    """)

    return all_ok


def main():
    """メイン処理"""
    print("=" * 80)
    print("CSV → GAF画像変換（攻撃タイプ別サブディレクトリ版）")
    print("=" * 80)

    # CSVディレクトリの確認
    if not os.path.exists(CSV_DIR):
        print(f"\n❌ エラー: CSVディレクトリが見つかりません: {CSV_DIR}")
        sys.exit(1)

    csv_files = []
    for fname in sorted(os.listdir(CSV_DIR)):
        if fname.lower().endswith(".csv"):
            path = os.path.join(CSV_DIR, fname)
            # 月曜日のデータのみ学習用、それ以外はテスト用にする
            is_train = "monday" in fname.lower()
            csv_files.append({"path": path, "is_train": is_train})

    if not csv_files:
        print(f"\n❌ エラー: CSVファイルが見つかりません: {CSV_DIR}")
        sys.exit(1)

    print("\n[検出されたCSVファイル]")
    train_count = sum(1 for c in csv_files if c["is_train"])
    test_count = len(csv_files) - train_count
    print(f"  学習用: {train_count}ファイル")
    print(f"  テスト用: {test_count}ファイル")
    print()
    for c in csv_files:
        mark = "★" if c["is_train"] else " "
        print(f" {mark} {os.path.basename(c['path'])}")

    total_counts = {"train/good": 0, "test/good": 0}
    attack_type_counts = {}

    # 学習用データでスケーラー学習
    print("\n" + "="*80)
    print("Phase 1: 学習用データ読み込み")
    print("="*80)
    train_dfs = []
    for csv_info in csv_files:
        if csv_info["is_train"] and os.path.exists(csv_info["path"]):
            try:
                df = pd.read_csv(csv_info["path"])
                train_dfs.append(df)
                print(
                    f"  ✓ {os.path.basename(csv_info['path'])}: {len(df):,}行")
            except Exception as e:
                print(f"  ❌ {os.path.basename(csv_info['path'])}: エラー - {e}")

    if not train_dfs:
        print("\n❌ エラー: 学習用CSVが読み込めませんでした")
        sys.exit(1)

    train_df = pd.concat(train_dfs, ignore_index=True)
    label_col = find_label_column(train_df)
    ensure_features(train_df, FEATURE_COLUMNS)

    train_normal, _ = split_by_label(train_df, label_col)
    print(f"\n  正常データ: {len(train_normal):,}行")

    scalers = fit_scalers_on_normal(train_normal, FEATURE_COLUMNS)
    print("  ✓ スケーラー学習完了")

    # 学習用画像生成
    print("\n" + "="*80)
    print("Phase 2: 学習用画像生成")
    print("="*80)
    win_train = extract_windows_3ch(
        train_normal, FEATURE_COLUMNS, scalers, WINDOW_SIZE)
    if win_train:
        print(f"  ウィンドウ数: {len(win_train[0]):,}")
        rgb_train = gaf_rgb_from_3ch_windows(
            win_train, image_size=WINDOW_SIZE, method=GAF_METHOD)
        n_train = save_images(rgb_train, TRAIN_GOOD, prefix="gaf_train",
                              max_images=MAX_IMAGES_TRAIN_GOOD)
        total_counts["train/good"] = n_train
        print(f"  ✓ 保存完了: {n_train:,}枚 → {TRAIN_GOOD}")
    else:
        print("  ⚠️ 警告: 学習用データからウィンドウを抽出できませんでした")

    # テスト用データ処理
    print("\n" + "="*80)
    print("Phase 3: テスト用データ処理")
    print("="*80)

    test_csvs = [c for c in csv_files if not c["is_train"]]

    for csv_info in test_csvs:
        if not os.path.exists(csv_info["path"]):
            continue

        csv_path = csv_info["path"]
        csv_name = os.path.splitext(os.path.basename(csv_path))[0]
        print(f"\n  処理中: {csv_name}")

        try:
            df = pd.read_csv(csv_path)
            test_normal, test_anomaly_dfs = split_by_label(df, label_col)

            # テスト正常
            if not test_normal.empty:
                win_test = extract_windows_3ch(
                    test_normal, FEATURE_COLUMNS, scalers, WINDOW_SIZE)
                if win_test:
                    rgb_test = gaf_rgb_from_3ch_windows(
                        win_test, image_size=WINDOW_SIZE, method=GAF_METHOD)
                    n_test = save_images(rgb_test, TEST_GOOD,
                                         prefix=f"gaf_test_good_{csv_name}",
                                         max_images=MAX_IMAGES_TEST_GOOD)
                    total_counts["test/good"] += n_test
                    print(f"    - 正常: {n_test:,}枚")

            # テスト異常（攻撃タイプ別サブディレクトリに保存）
            for attack_label, attack_df in test_anomaly_dfs.items():
                if attack_df.empty:
                    continue

                sanitized = sanitize_label(attack_label)
                win_anom = extract_windows_3ch(
                    attack_df, FEATURE_COLUMNS, scalers, WINDOW_SIZE)
                if win_anom:
                    rgb_anom = gaf_rgb_from_3ch_windows(
                        win_anom, image_size=WINDOW_SIZE, method=GAF_METHOD)

                    # 攻撃タイプ別サブディレクトリに保存
                    attack_dir = os.path.join(TEST_ANOMALY_ROOT, sanitized)
                    n_anom = save_images(
                        rgb_anom, attack_dir,
                        prefix=f"gaf_anom_{csv_name}_{sanitized}",
                        max_images=MAX_IMAGES_TEST_ANOM)

                    if sanitized not in attack_type_counts:
                        attack_type_counts[sanitized] = 0
                    attack_type_counts[sanitized] += n_anom
                    print(f"    - {sanitized}: {n_anom:,}枚")

        except Exception as e:
            print(f"    ❌ エラー: {e}")
            continue

    # 結果表示
    print("\n" + "=" * 80)
    print("GAF画像変換完了！")
    print("=" * 80)
    print(f"\n【学習用データ】")
    print(f"  train/good: {total_counts['train/good']:,}枚")
    print(f"\n【テスト用データ】")
    print(f"  test/good: {total_counts['test/good']:,}枚")

    if attack_type_counts:
        print(f"\n【攻撃タイプ別】")
        total_anomaly = sum(attack_type_counts.values())
        for attack_type in sorted(attack_type_counts.keys()):
            print(
                f"  test/anomaly/{attack_type:30s}: {attack_type_counts[attack_type]:,}枚")
        print(f"  {'合計':33s}: {total_anomaly:,}枚")

    print(f"\n📁 出力先: {OUT_ROOT}/")

    # データセット検証を実行
    verify_dataset_structure()

    print("\n" + "=" * 80)
    print("✅ 処理完了")
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
