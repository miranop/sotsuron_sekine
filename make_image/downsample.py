#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import shutil
from pathlib import Path

# 元データ
ROOT = Path("../datasets/rgb")
SRC_TEST_GOOD = ROOT / "test" / "good"
SRC_ANOM_ROOT = ROOT / "test" / "anomaly"

# 出力先
DST_ROOT = ROOT / "test_balanced"
DST_GOOD = DST_ROOT / "good"
DST_ANOM_ROOT = DST_ROOT / "anomaly"

random.seed(42)


def list_images(path: Path):
    return [p for p in path.iterdir()
            if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]


def main():
    print("=== Test Set Balancing ===")

    # 正常画像
    good_imgs = list_images(SRC_TEST_GOOD)
    N_good = len(good_imgs)

    print(f"正常画像数: {N_good}")

    # 異常は正常の4倍まで使う
    TARGET_RATIO = 4
    N_target_anomaly = N_good * TARGET_RATIO  # 例: 52 * 4 = 208

    # 出力ディレクトリ作成
    DST_GOOD.mkdir(parents=True, exist_ok=True)
    DST_ANOM_ROOT.mkdir(parents=True, exist_ok=True)

    # 正常をすべてコピー
    for img in good_imgs:
        shutil.copy2(img, DST_GOOD / img.name)

    # 各攻撃ごとに均等割り当て
    attack_dirs = [d for d in SRC_ANOM_ROOT.iterdir() if d.is_dir()]
    K = len(attack_dirs)
    per_attack = max(1, N_target_anomaly // K)

    print(f"異常の目標総数: {N_target_anomaly}（= 正常×{TARGET_RATIO}）")
    print(f"攻撃タイプ数: {K} → 1攻撃あたり {per_attack} 枚採用")

    total_selected = 0
    for attack_dir in attack_dirs:
        imgs = list_images(attack_dir)
        random.shuffle(imgs)

        selected = imgs[:per_attack]
        total_selected += len(selected)

        out_dir = DST_ANOM_ROOT / attack_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        for img in selected:
            shutil.copy2(img, out_dir / img.name)

        print(f"{attack_dir.name}: {len(imgs)} → {len(selected)} 枚使う")

    print("\n=== 完了！ ===")
    print(f"balanced test/good: {N_good}")
    print(f"balanced test/anomaly: {total_selected}")
    print(f"最終比率 anomaly/good = {total_selected / N_good:.2f}")


if __name__ == "__main__":
    main()
