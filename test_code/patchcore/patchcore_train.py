#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score
from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore

# ============================================================
# ★ 設定 ★
# ============================================================
DATASET_CHOICE = ""   # "rgb", "grayscale", "mtf", "gaf", "rp" etc.
MAX_EPOCHS = 1
BATCH_SIZE = 4
CORESET_RATIO = 0.01
# ============================================================


def check_dataset(root):
    """データセットの枚数を確認する"""
    print("\n" + "=" * 60)
    print("📊 データセット診断")
    print("=" * 60)

    root_path = Path(root)
    train_good = root_path / "train/good"
    test_good = root_path / "test/good"
    test_anomaly = root_path / "test/anomaly"

    total_train = len(list(train_good.rglob("*.png"))) + len(list(train_good.rglob("*.jpg")))
    total_test_good = len(list(test_good.rglob("*.png"))) + len(list(test_good.rglob("*.jpg")))
    total_test_anomaly = len(list(test_anomaly.rglob("*.png"))) + len(list(test_anomaly.rglob("*.jpg")))

    print(f"  Train (Normal): {total_train} 画像")
    print(f"  Test (Normal):  {total_test_good} 画像")
    print(f"  Test (Anomaly): {total_test_anomaly} 画像")
    print("=" * 60)
    return total_train, total_test_good, total_test_anomaly


def safe_extract(batch, key, default=None):
    """
    シンプルなデータ抽出関数
    """
    if isinstance(batch, dict):
        return batch.get(key, default)
    elif hasattr(batch, key):
        return getattr(batch, key, default)
    return default


def process_tensor_data(data):
    """Tensor/Arrayをフラットなリストに変換"""
    if data is None:
        return []
    
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    
    if np.ndim(data) == 0:
        return [float(data)]
    else:
        return data.tolist()


def evaluate_and_plot(engine, model, datamodule, output_plot_dir):
    """
    全データに対して推論を行い、FAR/FRRの計算とグラフ描画を行う関数
    """
    print("\n📊 詳細分析とグラフ作成を開始します...")
    
    # 1. 全テストデータの予測を実行
    try:
        predictions = engine.predict(model=model, datamodule=datamodule)
    except Exception as e:
        print(f"❌ 予測実行中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- デバッグ: 最初のバッチの中身を詳細調査 ---
    if len(predictions) > 0:
        first_batch = predictions[0]
        print(f"🔍 バッチ型確認: {type(first_batch)}")
        if hasattr(first_batch, "keys"):
            print(f"🔍 キー確認: {list(first_batch.keys())}")
        elif isinstance(first_batch, dict):
            print(f"🔍 キー確認: {list(first_batch.keys())}")
        
        # デバッグ：最初のバッチの各フィールドの詳細を表示
        print(f"\n🔍 最初のバッチの詳細:")
        pred_score_sample = safe_extract(first_batch, "pred_score")
        pred_label_sample = safe_extract(first_batch, "pred_label")
        gt_label_sample = safe_extract(first_batch, "gt_label")
        
        if pred_score_sample is not None:
            if isinstance(pred_score_sample, torch.Tensor):
                print(f"   pred_score shape: {pred_score_sample.shape}")
                print(f"   pred_score 値: {pred_score_sample.cpu().numpy()}")
                print(f"   pred_score min/max: {pred_score_sample.min().item():.6f} / {pred_score_sample.max().item():.6f}")
        
        if pred_label_sample is not None:
            if isinstance(pred_label_sample, torch.Tensor):
                print(f"   pred_label shape: {pred_label_sample.shape}")
                print(f"   pred_label 値: {pred_label_sample.cpu().numpy()}")
        
        if gt_label_sample is not None:
            if isinstance(gt_label_sample, torch.Tensor):
                print(f"   gt_label shape: {gt_label_sample.shape}")
                print(f"   gt_label 値: {gt_label_sample.cpu().numpy()}")
                print(f"   gt_label dtype: {gt_label_sample.dtype}")
    else:
        print("❌ エラー: 予測結果が空です。")
        return

    pred_scores = []
    pred_labels = []
    gt_labels = []
    
    print(f"\n🔄 データを抽出中 (全 {len(predictions)} バッチ)...")

    for i, batch in enumerate(predictions):
        # --- スコア抽出 ---
        score_data = safe_extract(batch, "pred_score")
        if score_data is None:
            score_data = safe_extract(batch, "pred_scores")
        if score_data is None:
            score_data = safe_extract(batch, "anomaly_maps")
            if score_data is not None and isinstance(score_data, torch.Tensor):
                score_data = score_data.reshape(score_data.shape[0], -1).max(dim=1).values
        
        extracted_scores = process_tensor_data(score_data)
        pred_scores.extend(extracted_scores)

        # --- 予測ラベル抽出（使わないが一応取得） ---
        pred_data = safe_extract(batch, "pred_label")
        if pred_data is None:
            pred_data = safe_extract(batch, "pred_labels")
        pred_labels.extend(process_tensor_data(pred_data))
        
        # --- 正解ラベル抽出（ブール型対応） ---
        gt_data = safe_extract(batch, "gt_label")
        if gt_data is None:
            gt_data = safe_extract(batch, "label")
        if gt_data is None:
            gt_data = safe_extract(batch, "labels")
        
        # ブール型をint型に変換（True=1, False=0）
        if gt_data is not None:
            if isinstance(gt_data, torch.Tensor):
                gt_data = gt_data.int()
        
        gt_labels.extend(process_tensor_data(gt_data))

    # 抽出結果の確認
    print(f"✅ 抽出完了: スコア {len(pred_scores)}件, ラベル {len(gt_labels)}件")
    
    # numpy配列化
    pred_scores_array = np.array(pred_scores)
    gt_labels = np.array(gt_labels, dtype=int)
    
    # スコアの統計情報を表示
    print(f"\n📊 スコア統計:")
    print(f"   最小値: {pred_scores_array.min():.6f}")
    print(f"   最大値: {pred_scores_array.max():.6f}")
    print(f"   平均値: {pred_scores_array.mean():.6f}")
    print(f"   中央値: {np.median(pred_scores_array):.6f}")
    print(f"   標準偏差: {pred_scores_array.std():.6f}")
    print(f"   ユニーク値の数: {len(np.unique(pred_scores_array))}")
    
    # 正常データと異常データのスコア分布を確認
    normal_scores = pred_scores_array[gt_labels == 0]
    anomaly_scores = pred_scores_array[gt_labels == 1]
    
    print(f"\n📊 クラス別スコア統計:")
    print(f"   正常データ: min={normal_scores.min():.4f}, max={normal_scores.max():.4f}, mean={normal_scores.mean():.4f}, std={normal_scores.std():.4f}")
    print(f"   異常データ: min={anomaly_scores.min():.4f}, max={anomaly_scores.max():.4f}, mean={anomaly_scores.mean():.4f}, std={anomaly_scores.std():.4f}")
    print(f"   平均値の差: {anomaly_scores.mean() - normal_scores.mean():.4f}")

    if len(gt_labels) == 0 or len(pred_scores) == 0:
        print(f"❌ エラー: データ抽出に失敗しました。")
        return

    pred_scores = pred_scores_array
    
    # 正解ラベルの分布を表示
    print(f"\n📊 正解ラベル分布:")
    unique_gt, counts_gt = np.unique(gt_labels, return_counts=True)
    for label, count in zip(unique_gt, counts_gt):
        label_name = "Normal" if label == 0 else "Anomaly"
        print(f"   {label_name} (label={label}): {count}件")
    
    # ROC曲線から最適閾値を計算
    fpr, tpr, thresholds = roc_curve(gt_labels, pred_scores)
    
    # Youden's J統計量で最適閾値を求める
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"\n🎯 最適閾値の計算:")
    print(f"   ROC曲線から算出した最適閾値: {optimal_threshold:.6f}")
    print(f"   この閾値でのTPR: {tpr[optimal_idx]:.4f}, FPR: {fpr[optimal_idx]:.4f}")
    print(f"   この閾値でのYouden's J: {j_scores[optimal_idx]:.4f}")
    
    # 最適閾値で予測ラベルを再計算
    pred_labels_optimal = (pred_scores > optimal_threshold).astype(int)
    
    # 予測ラベルの分布を表示
    print(f"\n📊 予測ラベル分布（最適閾値適用後）:")
    unique_pred, counts_pred = np.unique(pred_labels_optimal, return_counts=True)
    for label, count in zip(unique_pred, counts_pred):
        label_name = "Normal" if label == 0 else "Anomaly"
        print(f"   {label_name} (label={label}): {count}件")

    # 2. 評価指標の計算 (FAR, FRR)
    try:
        cm = confusion_matrix(gt_labels, pred_labels_optimal, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0
        f1 = f1_score(gt_labels, pred_labels_optimal)
        
        # AUROCも計算
        roc_auc = auc(fpr, tpr)
        
        print("\n" + "="*50)
        print("📈 詳細評価結果 (論文用データ)")
        print("="*50)
        print(f"  AUROC: {roc_auc:.4f}")
        print(f"  F1 Score : {f1:.4f}")
        print(f"  FAR (誤検知率): {far:.4f} ({far*100:.2f}%)")
        print(f"  FRR (見逃し率): {frr:.4f} ({frr*100:.2f}%)")
        print(f"  TP: {tp}, FN: {fn}, FP: {fp}, TN: {tn}")
        print(f"  最適閾値: {optimal_threshold:.6f}")
        print(f"  Accuracy: {(tp + tn) / (tp + tn + fp + fn):.4f}")
        print("="*50)
    except Exception as e:
        print(f"❌ 評価指標計算エラー: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. グラフ作成：異常度ヒストグラム（閾値ラインも表示）
    try:
        plt.figure(figsize=(10, 6))
        
        plt.hist(normal_scores, bins=50, alpha=0.6, color='blue', label='Normal', density=True)
        plt.hist(anomaly_scores, bins=50, alpha=0.6, color='red', label='Anomaly', density=True)
        
        # 最適閾値のラインを追加
        plt.axvline(x=optimal_threshold, color='green', linestyle='--', linewidth=2, 
                    label=f'Optimal Threshold ({optimal_threshold:.3f})')
        
        plt.xlabel('Anomaly Score')
        plt.ylabel('Density')
        plt.title(f'Score Distribution ({DATASET_CHOICE})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        hist_path = output_plot_dir / "histogram.png"
        plt.savefig(hist_path, dpi=150)
        plt.close()
        print(f"\n✅ ヒストグラムを保存: {hist_path}")
    except Exception as e:
        print(f"⚠️ ヒストグラム作成中にエラー: {e}")
        import traceback
        traceback.print_exc()

    # 4. グラフ作成：ROC曲線（最適点もプロット）
    try:
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        
        # 最適点をプロット
        plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'go', markersize=12, 
                 label=f'Optimal (TPR={tpr[optimal_idx]:.3f}, FPR={fpr[optimal_idx]:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FAR)')
        plt.ylabel('True Positive Rate (1 - FRR)')
        plt.title(f'ROC Curve ({DATASET_CHOICE})')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        roc_path = output_plot_dir / "roc_curve.png"
        plt.savefig(roc_path, dpi=150)
        plt.close()
        print(f"✅ ROC曲線を保存: {roc_path}")
    except Exception as e:
        print(f"⚠️ ROC曲線作成中にエラー: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("🚀 PatchCore トレーニング（最終安定版 v3.2）")
    print("=" * 60)

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.set_float32_matmul_precision("high")

    dataset_map = {
        "mtf": "../../datasets/mtf",
        "gaf": "../../datasets/gaf",
        "grayscale": "../../datasets/grayscale_header",
        "rgb": "../../datasets/rgb_header",
        "rp": "../../datasets/rp",
    }
    
    root = dataset_map.get(DATASET_CHOICE, "../../datasets/grayscale")
    output_root = f"results/patchcore_{DATASET_CHOICE}_v3"

    check_dataset(root)

    # ===== Datamodule =====
    datamodule = Folder(
        name=f"PatchCore_{DATASET_CHOICE}_v3",
        root=root,
        normal_dir="train/good",
        abnormal_dir="test/anomaly",
        normal_test_dir="test/good",
        train_batch_size=BATCH_SIZE,
        eval_batch_size=BATCH_SIZE,
        num_workers=2,
        test_split_mode="from_dir",
        val_split_mode="from_test",
        val_split_ratio=0.0,
        seed=42,
    )

    # ===== モデル =====
    model = Patchcore(
        backbone="resnet18",
        layers=["layer2"],
        coreset_sampling_ratio=CORESET_RATIO,
        num_neighbors=3,
        pre_trained=True,
    )

    # ===== エンジン設定 =====
    engine = Engine(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu",
        devices=1,
        precision="32",
        default_root_dir=output_root,
        enable_progress_bar=True,
        log_every_n_steps=50,
        num_sanity_val_steps=0,
        limit_val_batches=0,
    )

    print("\n🎯 設定概要")
    print("=" * 60)
    print(f"  モデル: PatchCore (ResNet18)")
    print(f"  データセット: {DATASET_CHOICE}")
    print(f"  バッチサイズ: {BATCH_SIZE}")
    print(f"  Coreset比率: {CORESET_RATIO}")
    print(f"  検証: スキップ（limit_val_batches=0）")
    print(f"  画像サイズ: Anomalibデフォルト (256x256)")
    print("=" * 60)

    try:
        print("\n🧠 学習開始...")
        engine.fit(datamodule=datamodule, model=model)
        
        print("\n🔍 モデルの状態確認...")
        # メモリバンクのサイズを確認
        if hasattr(model, 'model') and hasattr(model.model, 'memory_bank'):
            memory_bank = model.model.memory_bank
            if hasattr(memory_bank, 'shape'):
                print(f"   Memory Bank サイズ: {memory_bank.shape}")
            elif isinstance(memory_bank, torch.Tensor):
                print(f"   Memory Bank サイズ: {memory_bank.size()}")
            else:
                print(f"   Memory Bank タイプ: {type(memory_bank)}")

        print("\n🔍 テスト実行中...")
        test_results = engine.test(datamodule=datamodule, model=model)

        print("\n📊 基本結果 (Anomalib標準)")
        print("=" * 60)
        if test_results:
            auroc = test_results[0].get("image_AUROC", 0)
            f1 = test_results[0].get("image_F1Score", 0)
            print(f"  AUROC: {auroc:.4f}")
            print(f"  F1 Score: {f1:.4f}")

        # 詳細分析
        output_plot_dir = Path(output_root) / "plots"
        output_plot_dir.mkdir(parents=True, exist_ok=True)
        
        evaluate_and_plot(engine, model, datamodule, output_plot_dir)

        print(f"\n✅ 全工程完了！結果は {output_root} に保存されました")

    except Exception as e:
        print(f"\n❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()