# CSV to GAF Image Converter - リファクタリング版

## 概要

このディレクトリには、CSVファイルからGAF（Gramian Angular Field）画像を生成するスクリプトが含まれています。

- `csv_to_gaf_original.py`: 元のスクリプト
- `csv_to_gaf_converter.py`: リファクタリング版スクリプト（推奨）

## 主な改善点

### 1. **設定管理の改善**
```python
# 改善前：グローバル定数
CSV_DIR = "../CSV"
WINDOW_SIZE = 32
MAX_IMAGES_TRAIN_GOOD = 2000

# 改善後：dataclassで管理
@dataclass
class GAFConverterConfig:
    csv_dir: Path = Path("../CSV")
    window_size: int = 32
    max_train_good: int = 2000
    # ...設定を一箇所にまとめて管理
```

**メリット:**
- 設定の変更が容易
- 型安全性の向上
- デフォルト値の管理が明確

### 2. **ログ管理の改善**
```python
# 改善前：print文
print("CSV → GAF画像変換")
print(f"  ✓ {filename}: {len(df)} 行")

# 改善後：loggingモジュール
logger.info("CSV → GAF画像変換")
logger.info(f"  ✓ {filename}: {len(df):,} 行")
logger.error(f"Failed to load {filepath}: {e}")
```

**メリット:**
- ログレベルの制御（DEBUG、INFO、WARNING、ERROR）
- タイムスタンプの自動付与
- ログファイルへの出力が可能

### 3. **パス操作の改善**
```python
# 改善前：os.path
CSV_DIR = "../CSV"
path = os.path.join(CSV_DIR, fname)
os.makedirs(out_dir, exist_ok=True)

# 改善後：pathlib
csv_dir = Path("../CSV")
path = csv_dir / fname
output_dir.mkdir(parents=True, exist_ok=True)
```

**メリット:**
- より直感的な操作
- OS間の互換性向上
- 型安全性の向上

### 4. **型ヒントの追加**
```python
# 改善前：型ヒント不完全
def find_label_column(df: pd.DataFrame) -> str:
    pass

def split_by_label(df: pd.DataFrame, label_col: str):
    pass

# 改善後：完全な型ヒント
def find_label_column(df: pd.DataFrame, candidates: List[str]) -> str:
    pass

def split_by_label(
    df: pd.DataFrame,
    label_col: str
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    pass
```

**メリット:**
- IDEの補完機能が向上
- バグの早期発見
- コードの可読性向上

### 5. **クラスベースの設計**
```python
# 改善前：長大なmain()関数（200行超）
def main():
    # すべての処理がここに...
    pass

# 改善後：責務ごとにクラス分割
class DataValidator:
    """データ検証"""
    pass

class DataProcessor:
    """データ前処理"""
    pass

class GAFImageGenerator:
    """画像生成"""
    pass

class GAFConverter:
    """メイン変換クラス"""
    pass
```

**メリット:**
- 各クラスの責務が明確
- テストが容易
- 再利用性の向上
- 保守性の向上

### 6. **エラーハンドリングの強化**
```python
# 改善前：エラーハンドリング不足
df = pd.read_csv(csv_path)
Image.fromarray(rgb).save(os.path.join(out_dir, fname))

# 改善後：適切なエラーハンドリング
try:
    df = pd.read_csv(file_info["path"])
except Exception as e:
    logger.error(f"Failed to load {file_info['path']}: {e}")
    continue

try:
    Image.fromarray(rgb).save(filepath)
    saved_count += 1
except Exception as e:
    logger.error(f"Failed to save image {filepath}: {e}")
    continue
```

**メリット:**
- 一部のファイルでエラーが発生しても処理を継続
- エラーの詳細がログに記録される
- デバッグが容易

### 7. **マジックナンバーの削除**
```python
# 改善前：マジックナンバー
rgb = ((rgb + 1) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)

# 改善後：定数で管理
class GAFImageGenerator:
    GAF_MIN_VALUE = -1.0  # GAFの最小値
    GAF_MAX_VALUE = 1.0   # GAFの最大値
    PIXEL_MAX = 255       # ピクセルの最大値

    def _normalize_to_pixel_values(self, gaf_array: np.ndarray) -> np.ndarray:
        normalized = (gaf_array - self.GAF_MIN_VALUE) / (
            self.GAF_MAX_VALUE - self.GAF_MIN_VALUE
        )
        pixel_values = normalized * self.PIXEL_MAX
        return np.clip(pixel_values, 0, self.PIXEL_MAX).astype(np.uint8)
```

**メリット:**
- コードの意図が明確
- 定数の変更が容易
- 可読性の向上

### 8. **詳細なdocstringの追加**
```python
# 改善前：簡素なdocstring
def sanitize_label(label_str):
    """ラベル名をファイルシステムで使える形式に変換"""
    pass

# 改善後：詳細なdocstring
def sanitize_label(label: str) -> str:
    """
    ラベル名をファイルシステムで使える形式に変換

    Args:
        label: 元のラベル文字列

    Returns:
        サニタイズされたラベル文字列
    """
    pass
```

**メリット:**
- APIドキュメントの自動生成が可能
- 使い方が明確
- メンテナンスが容易

## 使い方

### 基本的な使い方

```bash
cd scripts
python csv_to_gaf_converter.py
```

### カスタム設定で実行

```python
from pathlib import Path
from csv_to_gaf_converter import GAFConverter, GAFConverterConfig

# 設定をカスタマイズ
config = GAFConverterConfig(
    csv_dir=Path("path/to/csv"),
    output_root=Path("path/to/output"),
    window_size=64,  # ウィンドウサイズを変更
    gaf_method="summation",  # GAF方式を変更
    max_train_good=5000,  # 最大画像数を変更
)

# 変換を実行
converter = GAFConverter(config)
converter.run()
```

### ログレベルの変更

```python
import logging

# DEBUGレベルのログを表示
logging.basicConfig(level=logging.DEBUG)

# WARNINGレベル以上のみ表示
logging.basicConfig(level=logging.WARNING)
```

## ディレクトリ構造

```
project_root/
├── scripts/
│   ├── csv_to_gaf_original.py    # 元のスクリプト
│   ├── csv_to_gaf_converter.py   # リファクタリング版（推奨）
│   └── README.md                  # このファイル
├── CSV/                           # 入力CSVファイル
│   ├── monday.csv                 # 学習用データ
│   ├── tuesday.csv                # テスト用データ
│   └── ...
└── datasets/
    └── gaf/                       # 出力ディレクトリ
        ├── train/
        │   └── good/
        └── test/
            ├── good/
            └── anomaly/
                ├── attack_type_1/
                ├── attack_type_2/
                └── ...
```

## 必要なパッケージ

```bash
pip install numpy pandas scikit-learn pyts pillow
```

または

```bash
pip install -r requirements.txt
```

## パフォーマンス

両方のスクリプトは機能的に同等ですが、リファクタリング版には以下の利点があります：

- エラー時の継続処理により、より多くのデータを処理可能
- ログによる処理状況の詳細な追跡
- メモリ効率は同等（大規模データセットには別途最適化が必要）

## テスト

```python
# 設定をテスト
config = GAFConverterConfig()
assert config.window_size == 32
assert config.gaf_method == "difference"

# バリデーションをテスト
validator = DataValidator()
sanitized = validator.sanitize_label("Web Attack - XSS")
assert sanitized == "Web_Attack___XSS"
```

## トラブルシューティング

### CSVファイルが見つからない

```
FileNotFoundError: CSV directory not found: ../CSV
```

**解決策:** スクリプトのディレクトリから相対パスが正しいか確認してください。

### ラベル列が見つからない

```
ValueError: ラベル列が見つかりません
```

**解決策:** `GAFConverterConfig` の `label_candidates` に正しいラベル列名を追加してください。

### 特徴量カラムが見つからない

```
ValueError: 特徴量カラムが見つかりません: [' Flow IAT Mean']
```

**解決策:** CSVファイルのカラム名を確認し、`feature_columns` を正しく設定してください。

## 今後の改善案

1. **マルチプロセス対応**: 複数のCSVファイルを並列処理
2. **メモリ効率の改善**: チャンク読み込みによる大規模データ対応
3. **進捗表示**: tqdmを使用した進捗バーの追加
4. **設定ファイル対応**: YAMLやTOMLでの設定管理
5. **ユニットテスト**: pytest による自動テスト
6. **CLI引数**: argparse による柔軟な設定

## ライセンス

研究用途のコードです。

## 参考文献

- Gramian Angular Field: [pyts documentation](https://pyts.readthedocs.io/)
