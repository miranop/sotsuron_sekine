#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSVファイルからGAF(RGB)画像を生成（リファクタリング版）

統計特徴量を時系列データに変換してGAF（Gramian Angular Field）変換を行い、
RGB画像として保存します。

主な改善点:
- dataclassによる設定管理
- loggingモジュールの使用
- pathlibによるパス操作
- 完全な型ヒント
- クラスベースの設計
- エラーハンドリングの強化
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from pyts.image import GramianAngularField
from sklearn.preprocessing import MinMaxScaler

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class GAFConverterConfig:
    """
    GAF変換の設定を管理するデータクラス

    Attributes:
        csv_dir: CSVファイルが格納されているディレクトリ
        output_root: 出力ディレクトリのルート
        feature_columns: 使用する特徴量のカラム名リスト
        label_candidates: ラベル列の候補名リスト
        benign_tokens: 正常データを示すラベルのトークン集合
        window_size: 時系列ウィンドウのサイズ
        gaf_method: GAF変換の方法（'summation' or 'difference'）
        max_train_good: 学習用正常データの最大画像数
        max_test_good: テスト用正常データの最大画像数
        max_test_anomaly: テスト用異常データの最大画像数
        train_file_pattern: 学習用ファイルの識別パターン
    """
    csv_dir: Path = Path("../CSV")
    output_root: Path = Path("../datasets/gaf")

    feature_columns: List[str] = field(default_factory=lambda: [
        " Flow IAT Mean",
        " Flow Duration",
        "Flow Bytes/s"
    ])

    label_candidates: List[str] = field(default_factory=lambda: [
        " Label", "Label", "label", " class", "Class"
    ])

    benign_tokens: Set[str] = field(default_factory=lambda: {
        "BENIGN", "Benign", "benign", "NORMAL", "Normal"
    })

    window_size: int = 32
    gaf_method: str = "difference"

    max_train_good: int = 2000
    max_test_good: int = 1000
    max_test_anomaly: int = 1000

    train_file_pattern: str = "monday"

    @property
    def train_good_dir(self) -> Path:
        """学習用正常データの出力ディレクトリ"""
        return self.output_root / "train" / "good"

    @property
    def test_good_dir(self) -> Path:
        """テスト用正常データの出力ディレクトリ"""
        return self.output_root / "test" / "good"

    @property
    def test_anomaly_root(self) -> Path:
        """テスト用異常データの出力ルートディレクトリ"""
        return self.output_root / "test" / "anomaly"

    def __post_init__(self):
        """初期化後の処理：パスをPathオブジェクトに変換"""
        if not isinstance(self.csv_dir, Path):
            self.csv_dir = Path(self.csv_dir)
        if not isinstance(self.output_root, Path):
            self.output_root = Path(self.output_root)


class DataValidator:
    """データの検証を行うクラス"""

    @staticmethod
    def sanitize_label(label: str) -> str:
        """
        ラベル名をファイルシステムで使える形式に変換

        Args:
            label: 元のラベル文字列

        Returns:
            サニタイズされたラベル文字列
        """
        return label.replace(" ", "_").replace("-", "_").replace("/", "_")

    @staticmethod
    def find_label_column(df: pd.DataFrame, candidates: List[str]) -> str:
        """
        ラベル列を探す

        Args:
            df: 対象のDataFrame
            candidates: ラベル列の候補名リスト

        Returns:
            見つかったラベル列の名前

        Raises:
            ValueError: ラベル列が見つからない場合
        """
        # 直接マッチング
        for candidate in candidates:
            if candidate in df.columns:
                logger.debug(f"Label column found: {candidate}")
                return candidate

        # 小文字変換してマッチング
        lower_map = {c.lower().strip(): c for c in df.columns}
        for candidate in ["label", "class", "attack_cat", "attack_type"]:
            if candidate in lower_map:
                matched_col = lower_map[candidate]
                logger.debug(f"Label column found (case-insensitive): {matched_col}")
                return matched_col

        raise ValueError(
            f"ラベル列が見つかりません。利用可能な列: {list(df.columns)}"
        )

    @staticmethod
    def validate_features(df: pd.DataFrame, feature_cols: List[str]) -> None:
        """
        特徴量の存在確認

        Args:
            df: 対象のDataFrame
            feature_cols: 必要な特徴量のカラム名リスト

        Raises:
            ValueError: 必要な特徴量が見つからない場合
        """
        missing = [col for col in feature_cols if col not in df.columns]
        if missing:
            raise ValueError(
                f"特徴量カラムが見つかりません: {missing}\n"
                f"利用可能な列: {list(df.columns)}"
            )
        logger.debug(f"All required features found: {feature_cols}")


class DataProcessor:
    """データの前処理を行うクラス"""

    def __init__(self, config: GAFConverterConfig):
        """
        Args:
            config: GAF変換の設定
        """
        self.config = config
        self.scalers: Dict[str, MinMaxScaler] = {}

    def split_by_label(
        self,
        df: pd.DataFrame,
        label_col: str
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        ラベルごとにデータを分割

        Args:
            df: 対象のDataFrame
            label_col: ラベル列の名前

        Returns:
            (正常データのDataFrame, 攻撃タイプ別のDataFrame辞書)
        """
        # inf値をNaNに置換
        df = df.replace([np.inf, -np.inf], np.nan)
        label_series = df[label_col].astype(str).str.strip()

        # 正常データを抽出
        normal_df = df[label_series.isin(self.config.benign_tokens)].copy()
        logger.info(f"Normal data: {len(normal_df):,} rows")

        # 攻撃タイプ別に分割
        anomaly_dfs = {}
        unique_labels = label_series.unique()
        for label in unique_labels:
            if label not in self.config.benign_tokens:
                anomaly_df = df[label_series == label].copy()
                anomaly_dfs[label] = anomaly_df
                logger.debug(f"Attack type '{label}': {len(anomaly_df):,} rows")

        return normal_df, anomaly_dfs

    def fit_scalers(
        self,
        normal_df: pd.DataFrame,
        feature_cols: List[str]
    ) -> None:
        """
        正常データでMinMaxScalerを学習

        Args:
            normal_df: 正常データのDataFrame
            feature_cols: 特徴量のカラム名リスト

        Raises:
            ValueError: 有効なデータが見つからない場合
        """
        self.scalers.clear()

        for col in feature_cols:
            vals = normal_df[col].dropna().values.reshape(-1, 1)

            if vals.size == 0:
                raise ValueError(f"正常データ内で {col} に有効値がありません")

            scaler = MinMaxScaler()
            scaler.fit(vals)
            self.scalers[col] = scaler

            logger.debug(
                f"Scaler fitted for '{col}': "
                f"min={scaler.data_min_[0]:.4f}, max={scaler.data_max_[0]:.4f}"
            )

        logger.info(f"Scalers fitted for {len(self.scalers)} features")

    def extract_windows(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        window_size: int
    ) -> Optional[List[np.ndarray]]:
        """
        3特徴量からウィンドウを抽出

        Args:
            df: 対象のDataFrame
            feature_cols: 特徴量のカラム名リスト
            window_size: ウィンドウサイズ

        Returns:
            チャンネルごとのウィンドウ配列のリスト、または抽出できない場合はNone
        """
        if not self.scalers:
            raise RuntimeError("Scalers are not fitted. Call fit_scalers() first.")

        scaled_features = []

        # 各特徴量をスケーリング
        for col in feature_cols:
            data = df[col].dropna().values.reshape(-1, 1)

            if data.size == 0:
                logger.warning(f"No valid data for feature '{col}'")
                return None

            scaled = self.scalers[col].transform(data).flatten()
            scaled_features.append(scaled)

        # 最小長に合わせる
        min_len = min(len(feat) for feat in scaled_features)
        scaled_features = [feat[:min_len] for feat in scaled_features]

        # ウィンドウサイズのチェック
        if min_len < window_size:
            logger.warning(
                f"Data length ({min_len}) is less than window size ({window_size})"
            )
            return None

        num_windows = min_len // window_size

        if num_windows == 0:
            logger.warning("No complete windows available")
            return None

        # ウィンドウを抽出
        windows_per_channel = []
        for feat in scaled_features:
            windows = [
                feat[i * window_size:(i + 1) * window_size]
                for i in range(num_windows)
            ]
            windows_per_channel.append(np.asarray(windows))

        logger.debug(
            f"Extracted {num_windows} windows of size {window_size} "
            f"from {min_len} samples"
        )

        return windows_per_channel


class GAFImageGenerator:
    """GAF画像を生成するクラス"""

    # GAF値を画像ピクセル値に変換する際の定数
    GAF_MIN_VALUE = -1.0  # GAFの最小値
    GAF_MAX_VALUE = 1.0   # GAFの最大値
    PIXEL_MAX = 255       # ピクセルの最大値

    def __init__(self, image_size: int, method: str = "difference"):
        """
        Args:
            image_size: 出力画像のサイズ
            method: GAF変換の方法（'summation' or 'difference'）
        """
        self.image_size = image_size
        self.method = method
        self.gaf = GramianAngularField(image_size=image_size, method=method)
        logger.debug(f"GAF generator initialized: size={image_size}, method={method}")

    def generate_rgb_images(
        self,
        windows_per_channel: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        GAF変換してRGB画像を生成

        Args:
            windows_per_channel: チャンネルごとのウィンドウ配列リスト

        Returns:
            RGB画像の配列リスト

        Raises:
            ValueError: チャンネル数が3でない場合
        """
        if len(windows_per_channel) != 3:
            raise ValueError(
                f"Expected 3 channels, got {len(windows_per_channel)}"
            )

        # 各チャンネルをGAF変換
        gaf_channels = []
        for i, channel_windows in enumerate(windows_per_channel):
            gaf_transformed = self.gaf.fit_transform(channel_windows)
            gaf_channels.append(gaf_transformed)
            logger.debug(
                f"GAF transformed channel {i}: shape={gaf_transformed.shape}"
            )

        num_windows = gaf_channels[0].shape[0]

        # RGB画像を生成
        rgb_images = []
        for i in range(num_windows):
            r = gaf_channels[0][i]
            g = gaf_channels[1][i]
            b = gaf_channels[2][i]

            # RGB配列を作成
            rgb = np.stack([r, g, b], axis=-1)

            # GAF値（-1〜1）をピクセル値（0〜255）に変換
            rgb = self._normalize_to_pixel_values(rgb)

            rgb_images.append(rgb)

        logger.info(f"Generated {len(rgb_images)} RGB images")
        return rgb_images

    def _normalize_to_pixel_values(self, gaf_array: np.ndarray) -> np.ndarray:
        """
        GAF値をピクセル値に正規化

        Args:
            gaf_array: GAF値の配列（-1〜1の範囲）

        Returns:
            ピクセル値の配列（0〜255の範囲、uint8型）
        """
        # [-1, 1] -> [0, 1]
        normalized = (gaf_array - self.GAF_MIN_VALUE) / (
            self.GAF_MAX_VALUE - self.GAF_MIN_VALUE
        )

        # [0, 1] -> [0, 255]
        pixel_values = normalized * self.PIXEL_MAX

        # クリップして整数型に変換
        pixel_values = np.clip(pixel_values, 0, self.PIXEL_MAX).astype(np.uint8)

        return pixel_values

    def save_images(
        self,
        rgb_images: List[np.ndarray],
        output_dir: Path,
        prefix: str = "img",
        max_images: Optional[int] = None
    ) -> int:
        """
        画像をファイルに保存

        Args:
            rgb_images: RGB画像の配列リスト
            output_dir: 出力ディレクトリ
            prefix: ファイル名のプレフィックス
            max_images: 保存する最大画像数（Noneの場合は全て保存）

        Returns:
            保存した画像数
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_count = 0
        for idx, rgb in enumerate(rgb_images):
            if max_images is not None and saved_count >= max_images:
                break

            filename = f"{prefix}_{idx:06d}.png"
            filepath = output_dir / filename

            try:
                Image.fromarray(rgb).save(filepath)
                saved_count += 1
            except Exception as e:
                logger.error(f"Failed to save image {filepath}: {e}")
                continue

        logger.info(f"Saved {saved_count} images to {output_dir}")
        return saved_count


class GAFConverter:
    """CSV→GAF画像変換のメインクラス"""

    def __init__(self, config: Optional[GAFConverterConfig] = None):
        """
        Args:
            config: GAF変換の設定（Noneの場合はデフォルト設定を使用）
        """
        self.config = config or GAFConverterConfig()
        self.validator = DataValidator()
        self.processor = DataProcessor(self.config)
        self.image_generator = GAFImageGenerator(
            image_size=self.config.window_size,
            method=self.config.gaf_method
        )

        # 統計情報
        self.stats = {
            "train_good": 0,
            "test_good": 0,
            "test_anomaly": {}
        }

    def discover_csv_files(self) -> List[Dict[str, any]]:
        """
        CSVファイルを検出

        Returns:
            CSVファイル情報のリスト

        Raises:
            FileNotFoundError: CSVディレクトリが存在しない場合
        """
        if not self.config.csv_dir.exists():
            raise FileNotFoundError(
                f"CSV directory not found: {self.config.csv_dir}"
            )

        csv_files = []
        for filepath in sorted(self.config.csv_dir.glob("*.csv")):
            is_train = self.config.train_file_pattern.lower() in filepath.name.lower()
            csv_files.append({
                "path": filepath,
                "is_train": is_train,
                "name": filepath.stem
            })

        logger.info(f"Discovered {len(csv_files)} CSV files")
        for file_info in csv_files:
            mark = "★" if file_info["is_train"] else " "
            logger.info(f"{mark} {file_info['path'].name}")

        return csv_files

    def load_training_data(
        self,
        csv_files: List[Dict[str, any]]
    ) -> pd.DataFrame:
        """
        学習用データを読み込み

        Args:
            csv_files: CSVファイル情報のリスト

        Returns:
            結合された学習用DataFrame

        Raises:
            RuntimeError: 学習用CSVが見つからない場合
        """
        logger.info("Loading training data...")

        train_dfs = []
        for file_info in csv_files:
            if not file_info["is_train"]:
                continue

            try:
                df = pd.read_csv(file_info["path"])
                train_dfs.append(df)
                logger.info(f"  ✓ {file_info['path'].name}: {len(df):,} rows")
            except Exception as e:
                logger.error(f"Failed to load {file_info['path']}: {e}")
                continue

        if not train_dfs:
            raise RuntimeError("学習用CSVが読み込めませんでした")

        train_df = pd.concat(train_dfs, ignore_index=True)
        logger.info(f"Total training data: {len(train_df):,} rows")

        return train_df

    def process_training_data(self, train_df: pd.DataFrame) -> None:
        """
        学習用データを処理してモデルを学習

        Args:
            train_df: 学習用DataFrame
        """
        logger.info("Processing training data...")

        # ラベル列を検出
        label_col = self.validator.find_label_column(
            train_df, self.config.label_candidates
        )

        # 特徴量を検証
        self.validator.validate_features(train_df, self.config.feature_columns)

        # 正常データを抽出
        normal_df, _ = self.processor.split_by_label(train_df, label_col)

        if normal_df.empty:
            raise ValueError("正常データが見つかりませんでした")

        # スケーラーを学習
        self.processor.fit_scalers(normal_df, self.config.feature_columns)

        # 画像を生成
        self._generate_and_save_images(
            normal_df,
            output_dir=self.config.train_good_dir,
            prefix="gaf_train",
            max_images=self.config.max_train_good,
            category="train_good"
        )

    def process_test_data(
        self,
        csv_files: List[Dict[str, any]],
        label_col: str
    ) -> None:
        """
        テスト用データを処理

        Args:
            csv_files: CSVファイル情報のリスト
            label_col: ラベル列の名前
        """
        logger.info("Processing test data...")

        for file_info in csv_files:
            if file_info["is_train"]:
                continue

            logger.info(f"Processing: {file_info['name']}")

            try:
                df = pd.read_csv(file_info["path"])
            except Exception as e:
                logger.error(f"Failed to load {file_info['path']}: {e}")
                continue

            # 正常・異常データに分割
            normal_df, anomaly_dfs = self.processor.split_by_label(df, label_col)

            # テスト正常データ処理
            if not normal_df.empty:
                self._generate_and_save_images(
                    normal_df,
                    output_dir=self.config.test_good_dir,
                    prefix=f"gaf_test_good_{file_info['name']}",
                    max_images=self.config.max_test_good,
                    category="test_good"
                )

            # テスト異常データ処理（攻撃タイプ別）
            for attack_label, attack_df in anomaly_dfs.items():
                if attack_df.empty:
                    continue

                sanitized_label = self.validator.sanitize_label(attack_label)
                output_dir = self.config.test_anomaly_root / sanitized_label

                count = self._generate_and_save_images(
                    attack_df,
                    output_dir=output_dir,
                    prefix=f"gaf_anom_{file_info['name']}_{sanitized_label}",
                    max_images=self.config.max_test_anomaly,
                    category=None
                )

                # 統計を更新
                if sanitized_label not in self.stats["test_anomaly"]:
                    self.stats["test_anomaly"][sanitized_label] = 0
                self.stats["test_anomaly"][sanitized_label] += count

    def _generate_and_save_images(
        self,
        df: pd.DataFrame,
        output_dir: Path,
        prefix: str,
        max_images: Optional[int],
        category: Optional[str]
    ) -> int:
        """
        画像を生成して保存（内部ヘルパーメソッド）

        Args:
            df: 対象のDataFrame
            output_dir: 出力ディレクトリ
            prefix: ファイル名のプレフィックス
            max_images: 最大画像数
            category: 統計カテゴリ（'train_good'、'test_good'、またはNone）

        Returns:
            保存した画像数
        """
        # ウィンドウを抽出
        windows = self.processor.extract_windows(
            df,
            self.config.feature_columns,
            self.config.window_size
        )

        if windows is None:
            logger.warning(f"No windows extracted for {prefix}")
            return 0

        # RGB画像を生成
        rgb_images = self.image_generator.generate_rgb_images(windows)

        # 画像を保存
        count = self.image_generator.save_images(
            rgb_images,
            output_dir,
            prefix=prefix,
            max_images=max_images
        )

        # 統計を更新
        if category:
            self.stats[category] += count

        return count

    def print_summary(self) -> None:
        """変換結果のサマリーを表示"""
        print("\n" + "=" * 80)
        print("GAF画像変換完了！")
        print("=" * 80)

        print("\n【学習用データ】")
        print(f"  train/good: {self.stats['train_good']:,} 枚")

        print("\n【テスト用データ】")
        print(f"  test/good: {self.stats['test_good']:,} 枚")

        if self.stats["test_anomaly"]:
            print("\n【攻撃タイプ別】")
            for attack_type in sorted(self.stats["test_anomaly"].keys()):
                count = self.stats["test_anomaly"][attack_type]
                print(f"  {attack_type:30s}: {count:,} 枚")

        print(f"\n出力先: {self.config.output_root}/")
        print("=" * 80)

    def run(self) -> None:
        """変換処理を実行"""
        try:
            logger.info("=" * 80)
            logger.info("CSV → GAF画像変換（攻撃タイプ別 + 時系列分割）")
            logger.info("=" * 80)

            # CSVファイルを検出
            csv_files = self.discover_csv_files()

            if not csv_files:
                raise RuntimeError("CSVファイルが見つかりませんでした")

            # 学習用データを処理
            train_df = self.load_training_data(csv_files)

            # ラベル列を検出（後でテストデータ処理にも使用）
            label_col = self.validator.find_label_column(
                train_df, self.config.label_candidates
            )

            # 学習データ処理
            self.process_training_data(train_df)

            # テストデータ処理
            self.process_test_data(csv_files, label_col)

            # 結果を表示
            self.print_summary()

        except Exception as e:
            logger.error(f"変換処理中にエラーが発生しました: {e}", exc_info=True)
            raise


def main() -> None:
    """メイン処理"""
    # 設定を作成（カスタマイズ可能）
    config = GAFConverterConfig(
        # csv_dir=Path("../CSV"),
        # output_root=Path("../datasets/gaf"),
        # window_size=32,
        # gaf_method="difference",
        # max_train_good=2000,
        # max_test_good=1000,
        # max_test_anomaly=1000,
    )

    # 変換を実行
    converter = GAFConverter(config)
    converter.run()


if __name__ == "__main__":
    main()
