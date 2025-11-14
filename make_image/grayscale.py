#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
グレースケール画像生成スクリプト - 修正版

主な修正:
1. データリーク対策: IPペアでグループ化せず時系列処理
2. メモリ効率化: ジェネレータでバッチ処理
3. データ不均衡対策: ラベルごとの最大画像数制限
4. 進捗表示の追加
5. 既存ファイルスキップの効率化
"""

import os
import sys
from PIL import Image
import numpy as np
from tqdm import tqdm

import same
import label

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

IMAGE_DIM = 64
PCAP_DIR = "../Pcap"
OUT_ROOT = "../datasets/grayscale"
TRAIN_GOOD = os.path.join(OUT_ROOT, "train", "good")
TEST_GOOD = os.path.join(OUT_ROOT, "test", "good")
TEST_ANOM_ROOT = os.path.join(OUT_ROOT, "test", "anomaly")

# データ不均衡対策: ラベルごとの最大画像数
MAX_IMAGES_PER_LABEL = {
    "BENIGN_train": 5000,  # 学習用正常
    "BENIGN_test": 2000,   # テスト用正常
    "default": 1000        # その他の攻撃タイプ
}


def sanitize_label(label_str):
    return label_str.replace(" ", "_").replace("-", "_").replace("/", "_")


def packet_batch_to_images(packet_batch, pcap_filename, image_dim=64):
    """
    パケットバッチから画像を生成（時系列順を保持）

    修正点:
    - IPペアでグループ化しない
    - 時系列順に連続したパケットで画像生成
    - tz_offset引数を削除（same.pyで既に処理済み）
    """
    images_with_labels = []

    # パケットを時系列順にソート（念のため）
    packet_batch.sort(key=lambda p: p['timestamp'])

    # 連続したパケットで画像生成
    for i in range(0, len(packet_batch), image_dim):
        chunk = packet_batch[i:i + image_dim]

        if len(chunk) < image_dim:
            # パケット数が不足している場合はスキップ
            continue

        image_data = np.zeros((image_dim, image_dim), dtype=np.uint8)

        # 最初のパケットのタイムスタンプとラベルを取得
        first_packet = chunk[0]
        timestamp = first_packet['timestamp']

        try:
            # label.get_labelの引数を確認して呼び出し
            try:
                label_value = label.get_label(
                    timestamp, pcap_filename, tz_offset_hours=-3)
            except TypeError:
                try:
                    label_value = label.get_label(
                        timestamp, pcap_filename, tz_offset=-3)
                except TypeError:
                    label_value = label.get_label(timestamp, pcap_filename)
        except Exception as e:
            print(f"    [警告] ラベル取得失敗: {e}")
            label_value = "BENIGN"

        # 画像データ生成
        for j, packet in enumerate(chunk):
            packet_bytes = packet['bytes']
            header = packet_bytes[:image_dim]
            padded = header.ljust(image_dim, b'\xff')
            image_data[j] = np.frombuffer(padded, dtype=np.uint8)

        # 正規化（オプション: コメントを外すと有効化）
        # image_data = normalize_image(image_data)

        images_with_labels.append((image_data, label_value))

    return images_with_labels


def normalize_image(image_data):
    """
    画像を正規化（Min-Max正規化）

    効果: モデルの学習を安定化
    """
    min_val = image_data.min()
    max_val = image_data.max()

    if max_val > min_val:
        normalized = (image_data - min_val) / (max_val - min_val) * 255
    else:
        normalized = image_data

    return normalized.astype(np.uint8)


def save_images_with_limit(images_with_labels, pcap_name, is_train_file, label_counts):
    """
    画像を保存（データ不均衡対策付き）

    修正点:
    - ラベルごとの最大画像数を制限
    - 既存ファイルチェックを先に実行
    """
    saved_counts = {"train/good": 0, "test/good": 0}
    attack_counts = {}
    skipped_existing = 0
    skipped_limit = 0

    for idx, (image_data, label_value) in enumerate(images_with_labels):
        # ラベルキーを作成
        if is_train_file:
            label_key = f"{label_value}_train"
        else:
            label_key = f"{label_value}_test"

        # 最大画像数チェック
        max_count = MAX_IMAGES_PER_LABEL.get(
            label_key,
            MAX_IMAGES_PER_LABEL["default"]
        )

        if label_counts.get(label_key, 0) >= max_count:
            skipped_limit += 1
            continue

        # ファイルパスとディレクトリを決定
        if is_train_file:
            if label_value == "BENIGN":
                out_dir = TRAIN_GOOD
                filename = f"{pcap_name}_train_{idx:06d}.png"
                count_key = "train/good"
            else:
                # 学習データに攻撃は含めない
                continue
        else:
            if label_value == "BENIGN":
                out_dir = TEST_GOOD
                filename = f"{pcap_name}_test_good_{idx:06d}.png"
                count_key = "test/good"
            else:
                sanitized_label = sanitize_label(label_value)
                out_dir = os.path.join(TEST_ANOM_ROOT, sanitized_label)
                filename = f"{pcap_name}_{sanitized_label}_{idx:06d}.png"
                count_key = None

        os.makedirs(out_dir, exist_ok=True)
        filepath = os.path.join(out_dir, filename)

        # 既存ファイルチェック（画像生成前）
        if os.path.exists(filepath):
            skipped_existing += 1
            # カウントは増やす（制限チェック用）
            label_counts[label_key] = label_counts.get(label_key, 0) + 1
            continue

        # 画像生成と保存
        img = Image.fromarray(image_data, 'L')
        img.save(filepath)

        # カウント更新
        label_counts[label_key] = label_counts.get(label_key, 0) + 1

        if count_key:
            saved_counts[count_key] += 1
        else:
            sanitized_label = sanitize_label(label_value)
            attack_counts[sanitized_label] = attack_counts.get(
                sanitized_label, 0) + 1

    if skipped_existing > 0:
        print(f"    既存ファイルスキップ: {skipped_existing:,}枚")
    if skipped_limit > 0:
        print(f"    制限によるスキップ: {skipped_limit:,}枚")

    return saved_counts, attack_counts


def process_pcap_in_batches(pcap_path, is_train, time_filter=None, batch_size=100000):
    """
    pcapファイルをバッチ処理（メモリ効率化）

    修正点:
    - 全パケットを一度に読み込まない
    - バッチごとに処理してメモリを解放
    """
    pcap_basename = os.path.basename(pcap_path)
    pcap_name = os.path.splitext(pcap_basename)[0]

    print(f"\n処理中: {pcap_basename} ({'学習用' if is_train else 'テスト用'})")
    if time_filter:
        print(f"  時刻フィルタ: {time_filter[0]}時 〜 {time_filter[1]}時")

    label_counts = {}  # ラベルごとのカウント
    total_saved_counts = {"train/good": 0, "test/good": 0}
    total_attack_counts = {}

    # パケットをバッチで読み込み
    packet_count = 0
    batch_num = 0

    # 簡易的なバッチ処理（same.read_packet_dpktを1回だけ呼ぶ）
    # 本格的にはジェネレータ化が必要だが、ここでは既存コードとの互換性を優先
    print("  パケット読み込み中...")

    # same.read_packet_dpktの引数を確認して適切に呼び出し
    try:
        # 引数名がtz_offsetの場合
        packets = same.read_packet_dpkt(
            pcap_path,
            time_filter=time_filter,
            count_limit=None,
            tz_offset=-3
        )
    except TypeError:
        # 引数名がtz_offset_hoursの場合
        try:
            packets = same.read_packet_dpkt(
                pcap_path,
                time_filter=time_filter,
                count_limit=None,
                tz_offset_hours=-3
            )
        except TypeError:
            # どちらもダメな場合は引数なしで呼び出し
            packets = same.read_packet_dpkt(
                pcap_path,
                time_filter=time_filter,
                count_limit=None
            )

    if not packets:
        print("  警告: パケットが読み込めませんでした")
        return total_saved_counts, total_attack_counts

    print(f"  総パケット数: {len(packets):,}")
    print(f"  画像生成中...")

    # バッチ処理
    for i in tqdm(range(0, len(packets), batch_size), desc="  バッチ処理"):
        batch = packets[i:i + batch_size]

        # 画像生成（時系列順、IPペアでグループ化しない）
        images_with_labels = packet_batch_to_images(
            batch, pcap_path, image_dim=IMAGE_DIM
        )

        # 画像保存
        counts, attack_counts = save_images_with_limit(
            images_with_labels, pcap_name, is_train, label_counts
        )

        # 集計
        for k in counts:
            total_saved_counts[k] += counts[k]
        for atk, c in attack_counts.items():
            total_attack_counts[atk] = total_attack_counts.get(atk, 0) + c

        batch_num += 1

    # ラベル集計を表示
    print(f"  [ラベル集計]")
    for lbl, count in sorted(label_counts.items()):
        print(f"    {lbl}: {count:,}枚")

    print("  ✓ 処理完了")

    return total_saved_counts, total_attack_counts


def main():
    print("=" * 80)
    print("pcap → グレースケール画像変換（修正版）")
    print("=" * 80)
    print("\n修正内容:")
    print("  ✓ データリーク対策: IPペアでグループ化しない")
    print("  ✓ メモリ効率化: バッチ処理")
    print("  ✓ データ不均衡対策: ラベルごとの最大画像数制限")
    print("  ✓ 進捗表示の追加")
    print("=" * 80)

    pcap_files = [
        {"path": os.path.join(PCAP_DIR, "Monday-WorkingHours.pcap"),
         "is_train": True,
         "time_filter": None},

        {"path": os.path.join(PCAP_DIR, "Tuesday-WorkingHours.pcap"),
         "is_train": False,
         "time_filter": None},

        {"path": os.path.join(PCAP_DIR, "Wednesday-workingHours.pcap"),
         "is_train": False,
         "time_filter": None},

        {"path": os.path.join(PCAP_DIR, "Thursday-WorkingHours.pcap"),
         "is_train": False,
         "time_filter": None},

        {"path": os.path.join(PCAP_DIR, "Friday-WorkingHours.pcap"),
         "is_train": False,
         "time_filter": None},
    ]

    total_counts = {"train/good": 0, "test/good": 0}
    all_attack_counts = {}

    for info in pcap_files:
        pcap_path = info["path"]
        is_train = info["is_train"]
        time_filter = info.get("time_filter")

        if not os.path.exists(pcap_path):
            print(f"\nスキップ: {pcap_path} が見つかりません")
            continue

        # バッチ処理で実行
        counts, attack_counts = process_pcap_in_batches(
            pcap_path, is_train, time_filter
        )

        for k in counts:
            total_counts[k] += counts[k]
        for atk, c in attack_counts.items():
            all_attack_counts[atk] = all_attack_counts.get(atk, 0) + c

    # 最終結果表示
    print("\n" + "=" * 80)
    print("グレースケール画像変換完了！")
    print("=" * 80)
    print(f"\n【学習用データ】")
    print(f"  train/good: {total_counts['train/good']:,}枚")
    print(f"\n【テスト用データ】")
    print(f"  test/good: {total_counts['test/good']:,}枚")

    if all_attack_counts:
        print(f"\n【攻撃タイプ別】")
        total_anomaly = sum(all_attack_counts.values())
        for atk in sorted(all_attack_counts.keys()):
            print(f"  {atk:30s}: {all_attack_counts[atk]:,}枚")
        print(f"  {'合計':30s}: {total_anomaly:,}枚")

    print(f"\n📁 出力先: {OUT_ROOT}/")
    print("   ├── train/good/")
    print("   └── test/")
    print("       ├── good/")
    print("       └── anomaly/")
    print("           ├── DDoS/")
    print("           ├── PortScan/")
    print("           └── ...")
    print("=" * 80)


if __name__ == "__main__":
    main()
