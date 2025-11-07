#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV to GAF Converter - 使用例

リファクタリング版の様々な使用方法を示すサンプルコード
"""

import logging
from pathlib import Path

from csv_to_gaf_converter import (
    GAFConverter,
    GAFConverterConfig,
    DataValidator,
    DataProcessor,
    GAFImageGenerator
)


def example_1_basic_usage():
    """例1: 基本的な使い方"""
    print("=" * 80)
    print("例1: 基本的な使い方（デフォルト設定）")
    print("=" * 80)

    # デフォルト設定で実行
    converter = GAFConverter()
    converter.run()


def example_2_custom_config():
    """例2: カスタム設定での実行"""
    print("=" * 80)
    print("例2: カスタム設定での実行")
    print("=" * 80)

    # 設定をカスタマイズ
    config = GAFConverterConfig(
        csv_dir=Path("../CSV"),
        output_root=Path("../datasets/gaf_custom"),
        window_size=64,  # ウィンドウサイズを大きく
        gaf_method="summation",  # GAF方式を変更
        max_train_good=5000,  # 最大画像数を増やす
        max_test_good=2000,
        max_test_anomaly=2000,
    )

    converter = GAFConverter(config)
    converter.run()


def example_3_with_logging():
    """例3: ログレベルを変更して実行"""
    print("=" * 80)
    print("例3: DEBUGログを有効化")
    print("=" * 80)

    # DEBUGレベルのログを表示
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    converter = GAFConverter()
    converter.run()


def example_4_step_by_step():
    """例4: ステップバイステップでの実行"""
    print("=" * 80)
    print("例4: ステップバイステップでの実行")
    print("=" * 80)

    config = GAFConverterConfig()
    converter = GAFConverter(config)

    # ステップ1: CSVファイルを検出
    print("\nStep 1: Discovering CSV files...")
    csv_files = converter.discover_csv_files()
    print(f"Found {len(csv_files)} CSV files")

    # ステップ2: 学習データを読み込み
    print("\nStep 2: Loading training data...")
    train_df = converter.load_training_data(csv_files)
    print(f"Training data shape: {train_df.shape}")

    # ステップ3: 学習データを処理
    print("\nStep 3: Processing training data...")
    converter.process_training_data(train_df)

    # ステップ4: テストデータを処理
    print("\nStep 4: Processing test data...")
    label_col = converter.validator.find_label_column(
        train_df, config.label_candidates
    )
    converter.process_test_data(csv_files, label_col)

    # ステップ5: サマリー表示
    print("\nStep 5: Showing summary...")
    converter.print_summary()


def example_5_individual_components():
    """例5: 個別コンポーネントの使用"""
    print("=" * 80)
    print("例5: 個別コンポーネントを直接使用")
    print("=" * 80)

    import pandas as pd
    import numpy as np

    # サンプルデータを作成
    data = {
        " Flow IAT Mean": np.random.rand(1000),
        " Flow Duration": np.random.rand(1000) * 1000,
        "Flow Bytes/s": np.random.rand(1000) * 10000,
        " Label": ["BENIGN"] * 800 + ["Attack"] * 200
    }
    df = pd.DataFrame(data)

    config = GAFConverterConfig()

    # データバリデーター
    validator = DataValidator()
    label_col = validator.find_label_column(df, config.label_candidates)
    validator.validate_features(df, config.feature_columns)
    print(f"✓ Validation passed. Label column: {label_col}")

    # データプロセッサー
    processor = DataProcessor(config)
    normal_df, anomaly_dfs = processor.split_by_label(df, label_col)
    print(f"✓ Normal data: {len(normal_df)} rows")
    print(f"✓ Anomaly data: {sum(len(adf) for adf in anomaly_dfs.values())} rows")

    # スケーラーを学習
    processor.fit_scalers(normal_df, config.feature_columns)
    print("✓ Scalers fitted")

    # ウィンドウを抽出
    windows = processor.extract_windows(
        normal_df, config.feature_columns, config.window_size
    )
    if windows:
        print(f"✓ Extracted {windows[0].shape[0]} windows")

        # GAF画像を生成
        image_generator = GAFImageGenerator(
            image_size=config.window_size,
            method=config.gaf_method
        )
        rgb_images = image_generator.generate_rgb_images(windows)
        print(f"✓ Generated {len(rgb_images)} RGB images")

        # 画像を保存
        output_dir = Path("../datasets/gaf_example")
        saved_count = image_generator.save_images(
            rgb_images,
            output_dir,
            prefix="example",
            max_images=10
        )
        print(f"✓ Saved {saved_count} images to {output_dir}")


def example_6_error_handling():
    """例6: エラーハンドリングの例"""
    print("=" * 80)
    print("例6: エラーハンドリング")
    print("=" * 80)

    try:
        # 存在しないディレクトリを指定
        config = GAFConverterConfig(
            csv_dir=Path("/nonexistent/directory")
        )
        converter = GAFConverter(config)
        converter.run()

    except FileNotFoundError as e:
        print(f"✓ FileNotFoundError caught: {e}")

    try:
        # 不正な設定値
        config = GAFConverterConfig(
            window_size=0  # 不正な値
        )
        # この場合、実行時にエラーが発生する可能性がある

    except Exception as e:
        print(f"✓ Error caught: {e}")


def example_7_batch_processing():
    """例7: バッチ処理の例"""
    print("=" * 80)
    print("例7: 複数の設定でバッチ処理")
    print("=" * 80)

    # 複数の設定を試す
    configs = [
        {
            "name": "small_window",
            "window_size": 16,
            "gaf_method": "difference",
            "output_root": Path("../datasets/gaf_16_diff")
        },
        {
            "name": "large_window",
            "window_size": 64,
            "gaf_method": "difference",
            "output_root": Path("../datasets/gaf_64_diff")
        },
        {
            "name": "summation_method",
            "window_size": 32,
            "gaf_method": "summation",
            "output_root": Path("../datasets/gaf_32_sum")
        },
    ]

    for config_params in configs:
        name = config_params.pop("name")
        print(f"\n処理中: {name}")

        try:
            config = GAFConverterConfig(**config_params)
            converter = GAFConverter(config)
            converter.run()
            print(f"✓ {name} 完了")

        except Exception as e:
            print(f"✗ {name} 失敗: {e}")


def main():
    """メイン関数"""
    examples = {
        "1": ("基本的な使い方", example_1_basic_usage),
        "2": ("カスタム設定", example_2_custom_config),
        "3": ("ログレベル変更", example_3_with_logging),
        "4": ("ステップバイステップ", example_4_step_by_step),
        "5": ("個別コンポーネント", example_5_individual_components),
        "6": ("エラーハンドリング", example_6_error_handling),
        "7": ("バッチ処理", example_7_batch_processing),
    }

    print("\n使用例の選択:")
    for key, (desc, _) in examples.items():
        print(f"  {key}. {desc}")
    print("  0. 全て実行")
    print("  q. 終了")

    choice = input("\n選択してください: ").strip()

    if choice == "q":
        print("終了します")
        return

    if choice == "0":
        for key in sorted(examples.keys()):
            _, func = examples[key]
            try:
                func()
            except Exception as e:
                print(f"エラー: {e}")
            print("\n" + "=" * 80 + "\n")
    elif choice in examples:
        _, func = examples[choice]
        func()
    else:
        print("無効な選択です")


if __name__ == "__main__":
    # 特定の例を直接実行する場合は、以下のコメントを外してください
    # example_1_basic_usage()
    # example_2_custom_config()
    # example_5_individual_components()

    # インタラクティブモード
    main()
