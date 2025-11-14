#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RGB画像生成スクリプト - 修正版（攻撃タイプ別サブディレクトリ対応）

主な修正:
1. データリーク対策: IPペアでグループ化せず時系列処理
2. タイムゾーン補正の統一
3. メモリ効率化: バッチ処理
4. データ不均衡対策: ラベルごとの最大画像数制限
5. 進捗表示の追加
6. 攻撃タイプ別サブディレクトリに保存
"""

import os
import sys
import numpy as np
from PIL import Image
from datetime import datetime, timezone, timedelta
import socket
import dpkt
from tqdm import tqdm
import label

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ===== 設定 =====
IMAGE_DIM = 64
PACKETS_PER_IMAGE = IMAGE_DIM * 3
PCAP_DIR = "../Pcap"
OUT_ROOT = "../datasets/rgb"
TRAIN_GOOD = os.path.join(OUT_ROOT, "train", "good")
TEST_GOOD = os.path.join(OUT_ROOT, "test", "good")
TEST_ANOM_ROOT = os.path.join(OUT_ROOT, "test", "anomaly")

TZ_OFFSET = -3

# データ不均衡対策
MAX_IMAGES_PER_LABEL = {
    "BENIGN_train": 5000,
    "BENIGN_test": 2000,
    "default": 1000
}


def read_packet_dpkt_batched(filepath, time_filter=None, batch_size=100000, tz_offset=TZ_OFFSET):
    """
    dpktでpcapをバッチ読み込み（ジェネレータ）

    修正点:
    - メモリ効率化のためジェネレータを使用
    """
    offset = timedelta(hours=tz_offset)

    print("  パケット読み込み開始...")

    with open(filepath, 'rb') as f:
        try:
            pcap = dpkt.pcap.Reader(f)
        except:
            f.seek(0)
            pcap = dpkt.pcapng.Reader(f)

        batch = []
        count = 0

        for timestamp, buf in pcap:
            # タイムゾーン補正
            dt_local = datetime.fromtimestamp(
                timestamp, tz=timezone.utc) + offset
            hour = dt_local.hour

            # time_filter指定がある場合
            if time_filter and not (time_filter[0] <= hour < time_filter[1]):
                continue

            try:
                eth = dpkt.ethernet.Ethernet(buf)
                if not isinstance(eth.data, dpkt.ip.IP):
                    continue
                ip = eth.data
                src_ip = socket.inet_ntoa(ip.src)
                dst_ip = socket.inet_ntoa(ip.dst)

                batch.append({
                    "timestamp": dt_local,  # datetimeのまま保持
                    "src_ip": src_ip,
                    "dst_ip": dst_ip,
                    "bytes": bytes(ip)
                })
                count += 1

                # バッチサイズに達したらyield
                if len(batch) >= batch_size:
                    yield batch
                    batch = []

                # 進捗表示
                if count % 500000 == 0:
                    print(f"    読み込み済み: {count:,}パケット")

            except Exception:
                continue

        # 残りのパケットをyield
        if batch:
            yield batch

    print(f"  ✓ 読み込み完了: {count:,}パケット")


def packets_to_rgb_with_labels(packet_list, pcap_filename, image_dim=IMAGE_DIM):
    """
    パケットリストからRGB画像を生成

    修正点:
    - タイムゾーン補正を統一（datetimeのまま渡す）
    """
    images_with_labels = []
    packets_per_image = image_dim * 3

    for i in range(0, len(packet_list), packets_per_image):
        chunk = packet_list[i:i + packets_per_image]

        if len(chunk) < packets_per_image:
            # パケット数が不足している場合はスキップ
            continue

        first_packet = chunk[0]
        timestamp = first_packet["timestamp"]

        # ★修正: datetimeのまま渡す（label.pyの実装に応じて調整）
        try:
            # 引数パターン1: datetime対応
            label_value = label.get_label(timestamp, pcap_filename)
        except TypeError:
            try:
                # 引数パターン2: floatのみ対応
                timestamp_float = timestamp.timestamp()
                label_value = label.get_label(timestamp_float, pcap_filename)
            except Exception as e:
                print(f"    [警告] ラベル取得失敗: {e}")
                label_value = "BENIGN"
        except Exception as e:
            print(f"    [警告] ラベル取得失敗: {e}")
            label_value = "BENIGN"

        # RGB画像生成
        image_data = np.zeros((image_dim, image_dim, 3), dtype=np.uint8)

        for ch in range(3):
            for j in range(image_dim):
                idx = j + image_dim * ch
                if idx < len(chunk):
                    pkt_bytes = chunk[idx]["bytes"][:image_dim]
                    padded = pkt_bytes.ljust(image_dim, b'\xff')
                    image_data[j, :, ch] = np.frombuffer(
                        padded, dtype=np.uint8)
                else:
                    image_data[j, :, ch] = 255

        images_with_labels.append((image_data, label_value))

    return images_with_labels


def sanitize_label(label_str):
    """ラベル名をファイルシステムで使える形式に変換"""
    return label_str.replace(" ", "_").replace("-", "_").replace("/", "_")


def save_images_with_limit(images_with_labels, pcap_name, is_train_file, label_counts):
    """
    画像を保存（データ不均衡対策＋効率化）

    修正点:
    - ラベルごとの最大画像数を制限
    - 既存ファイルチェックを先に実行
    - 攻撃タイプ別サブディレクトリに保存
    """
    saved_counts = {"train/good": 0, "test/good": 0}
    attack_counts = {}
    skipped_existing = 0
    skipped_limit = 0

    for idx, (image_data, label_value) in enumerate(images_with_labels):
        # ラベルキーを作成
        label_key = f"{label_value}_{'train' if is_train_file else 'test'}"

        # 最大画像数チェック
        max_count = MAX_IMAGES_PER_LABEL.get(
            label_key, MAX_IMAGES_PER_LABEL["default"])

        if label_counts.get(label_key, 0) >= max_count:
            skipped_limit += 1
            continue

        # ファイルパスとディレクトリを決定
        if label_value == "BENIGN":
            if is_train_file:
                out_dir = TRAIN_GOOD
                filename = f"{pcap_name}_train_{idx:06d}.png"
                count_key = "train/good"
            else:
                out_dir = TEST_GOOD
                filename = f"{pcap_name}_test_{idx:06d}.png"
                count_key = "test/good"
        else:
            if is_train_file:
                # 学習データに攻撃は含めない
                continue
            # 攻撃タイプ別サブディレクトリに保存
            sanitized = sanitize_label(label_value)
            out_dir = os.path.join(TEST_ANOM_ROOT, sanitized)
            filename = f"{pcap_name}_{sanitized}_{idx:06d}.png"
            count_key = None

        os.makedirs(out_dir, exist_ok=True)
        filepath = os.path.join(out_dir, filename)

        # 既存ファイルチェック（画像生成前）
        if os.path.exists(filepath):
            skipped_existing += 1
            label_counts[label_key] = label_counts.get(label_key, 0) + 1
            continue

        # 画像生成と保存
        img = Image.fromarray(image_data, "RGB")
        img.save(filepath)

        # カウント更新
        label_counts[label_key] = label_counts.get(label_key, 0) + 1

        if count_key:
            saved_counts[count_key] += 1
        else:
            sanitized = sanitize_label(label_value)
            attack_counts[sanitized] = attack_counts.get(sanitized, 0) + 1

    if skipped_existing > 0:
        print(f"    既存ファイルスキップ: {skipped_existing:,}枚")
    if skipped_limit > 0:
        print(f"    制限によるスキップ: {skipped_limit:,}枚")

    return saved_counts, attack_counts


def process_pcap_file(pcap_file, is_train, time_filter=None):
    """
    pcapファイルを処理（時系列順、IPペアでグループ化しない）

    修正点:
    - バッチごとに処理してメモリ効率化
    - IPペアでグループ化しない
    """
    pcap_name = os.path.basename(pcap_file)
    print(f"\n処理中: {pcap_name} ({'学習用' if is_train else 'テスト用'})")

    label_counts = {}
    total_counts = {"train/good": 0, "test/good": 0}
    total_attack_counts = {}

    batch_count = 0

    # パケットをバッチで処理
    for packet_batch in read_packet_dpkt_batched(pcap_file, time_filter=time_filter):
        batch_count += 1

        if batch_count == 1:
            print(f"  バッチサイズ: {len(packet_batch):,}パケット")

        # パケットを時系列順にソート
        packet_batch.sort(key=lambda p: p['timestamp'])

        print(f"  バッチ {batch_count} 画像生成中...")

        # RGB画像生成（時系列順）
        images = packets_to_rgb_with_labels(
            packet_batch, pcap_file, image_dim=IMAGE_DIM
        )

        if images:
            print(f"    生成画像数: {len(images):,}枚")

        # 画像保存
        counts, attack_counts = save_images_with_limit(
            images, os.path.splitext(pcap_name)[0], is_train, label_counts
        )

        # 集計
        for k in counts:
            total_counts[k] += counts[k]
        for atk, c in attack_counts.items():
            total_attack_counts[atk] = total_attack_counts.get(atk, 0) + c

    # ラベル集計表示
    print(f"  [ラベル集計]")
    for lbl, count in sorted(label_counts.items()):
        print(f"    {lbl}: {count:,}枚")

    print("  ✓ 処理完了")

    return total_counts, total_attack_counts


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

    counts = {}

    for name, path in dirs_to_check.items():
        print(f"\n[{name}]")
        print(f"  パス: {path}")

        if not os.path.exists(path):
            print(f"  ❌ ディレクトリが存在しません")
            counts[name] = 0
            continue

        # 画像ファイルをカウント
        image_files = [f for f in os.listdir(path)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        count = len(image_files)
        counts[name] = count

        if count == 0:
            print(f"  ⚠️ 画像ファイルが見つかりません")
        else:
            print(f"  ✅ {count:,}枚の画像を検出")
            print(f"  サンプルファイル:")
            for fname in sorted(image_files)[:3]:
                print(f"    - {fname}")
            if count > 3:
                print(f"    ... (他 {count-3:,}枚)")

    # 攻撃タイプ別ディレクトリの確認
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
                print(f"    - {attack_type}: {count:,}枚")
        else:
            print(f"  ⚠️ 攻撃タイプディレクトリが見つかりません")
    else:
        print(f"  ❌ ディレクトリが存在しません")

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

    print("\n" + "="*80)
    print("📁 生成されたディレクトリ構造")
    print("="*80)
    print(f"""
{OUT_ROOT}/
├── train/
│   └── good/                    ← 学習用正常画像
└── test/
    ├── good/                    ← テスト用正常画像
    └── anomaly/                 ← 攻撃タイプ別サブディレクトリ
        ├── DDoS/
        ├── PortScan/
        ├── BruteForce/
        └── ...
    """)


def main():
    print("=" * 80)
    print("pcap → RGB画像変換（修正版 - 攻撃タイプ別）")
    print("=" * 80)
    print("\n修正内容:")
    print("  ✓ データリーク対策: IPペアでグループ化しない")
    print("  ✓ タイムゾーン補正の統一")
    print("  ✓ メモリ効率化: バッチ処理")
    print("  ✓ データ不均衡対策: ラベルごとの最大画像数制限")
    print("  ✓ 攻撃タイプ別サブディレクトリに保存")
    print("=" * 80)

    pcap_files = [
        {"path": os.path.join(
            PCAP_DIR, "Monday-WorkingHours.pcap"), "is_train": True},
        {"path": os.path.join(
            PCAP_DIR, "Tuesday-WorkingHours.pcap"), "is_train": False},
        {"path": os.path.join(
            PCAP_DIR, "Wednesday-workingHours.pcap"), "is_train": False},
        {"path": os.path.join(
            PCAP_DIR, "Thursday-WorkingHours.pcap"), "is_train": False},
        {"path": os.path.join(
            PCAP_DIR, "Friday-WorkingHours.pcap"), "is_train": False},
    ]

    total_counts = {"train/good": 0, "test/good": 0}
    all_attack_counts = {}

    for pcap_info in pcap_files:
        pcap_file = pcap_info["path"]
        is_train = pcap_info["is_train"]

        if not os.path.exists(pcap_file):
            print(f"\nスキップ: {pcap_file} が見つかりません")
            continue

        # pcapファイル処理
        counts, attack_counts = process_pcap_file(pcap_file, is_train)

        for k in counts:
            total_counts[k] += counts[k]
        for atk, c in attack_counts.items():
            all_attack_counts[atk] = all_attack_counts.get(atk, 0) + c

    # 最終結果表示
    print("\n" + "=" * 80)
    print("RGB画像生成完了！")
    print("=" * 80)
    print(f"\n【学習用】")
    print(f"  train/good: {total_counts['train/good']:,}枚")
    print(f"\n【テスト用】")
    print(f"  test/good: {total_counts['test/good']:,}枚")

    if all_attack_counts:
        print(f"\n【攻撃タイプ別】")
        total_anomaly = sum(all_attack_counts.values())
        for atk, c in sorted(all_attack_counts.items()):
            print(f"  {atk:30s}: {c:,}枚")
        print(f"  {'合計':30s}: {total_anomaly:,}枚")

    # データセット検証
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
