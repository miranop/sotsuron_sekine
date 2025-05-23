#!/usr/bin/env python3
"""
堅牢なAnomalibテスト
"""

def test_package(package_name, import_name=None, version_attr="__version__"):
    """個別パッケージテスト"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = __import__(import_name)
        if hasattr(module, version_attr):
            version = getattr(module, version_attr)
            print(f"✓ {package_name} {version} インポート成功")
        else:
            print(f"✓ {package_name} インポート成功")
        return True
    except ImportError:
        print(f"✗ {package_name} が見つかりません")
        return False

def test_opencv():
    """OpenCV特別テスト"""
    try:
        import cv2
        print(f"✓ OpenCV (cv2) {cv2.__version__} インポート成功")
        return True
    except ImportError:
        print("✗ OpenCV (cv2) が見つかりません")
        print("  解決方法: pip install opencv-python-headless")
        return False

def test_pytorch_lightning():
    """PyTorch Lightning特別テスト"""
    try:
        import pytorch_lightning as pl
        print(f"✓ PyTorch Lightning {pl.__version__} インポート成功")
        return True
    except ImportError:
        print("✗ PyTorch Lightning が見つかりません")
        print("  解決方法: pip install pytorch-lightning")
        return False

def test_anomalib_model():
    """Anomalibモデルテスト"""
    try:
        from anomalib.models import Padim
        model = Padim()
        print("✓ PADIMモデル作成成功")
        return True
    except ImportError as e:
        print(f"✗ Anomalibモデル作成失敗: {e}")
        return False
    except Exception as e:
        print(f"✗ モデル作成エラー: {e}")
        return False

def main():
    """メインテスト関数"""
    print("=== Anomalib環境テスト ===\n")
    
    # 基本パッケージテスト
    results = []
    results.append(test_package("anomalib"))
    results.append(test_package("torch"))
    results.append(test_package("numpy"))
    results.append(test_package("PIL", "PIL", "__version__"))
    results.append(test_package("matplotlib", "matplotlib"))
    
    # 特別テスト
    results.append(test_opencv())
    results.append(test_pytorch_lightning())
    
    # Anomalibモデルテスト
    if all(results):
        results.append(test_anomalib_model())
    
    # CUDA情報
    try:
        import torch
        print(f"\n--- GPU情報 ---")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
    except:
        pass
    
    # 結果サマリー
    print(f"\n=== テスト結果 ===")
    success_count = sum(results)
    total_count = len(results)
    
    if success_count == total_count:
        print("✓ すべてのテストが成功！環境は正常です。")
    else:
        print(f"✗ {total_count - success_count}/{total_count} のテストが失敗")
        print("\n--- 不足パッケージの一括インストール ---")
        print("pip install opencv-python-headless pytorch-lightning pillow numpy matplotlib")

if __name__ == "__main__":
    main()